import os
import shutil
from pathlib import Path
from typing import IO

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter, ParquetWriter

from llm_datasets.datasets.base import BaseDataset
from llm_datasets.datasets.dataset_registry import (
    get_dataset_class_by_id,
    get_datasets_list_from_string,
)
from llm_datasets.datatrove_reader import LLMDatasetsDatatroveReader
from llm_datasets.utils import get_auto_workers, get_bytes_from_int_or_string
from llm_datasets.utils.config import Config


def extract_text(config: Config):
    logger = config.init_logger(__name__)

    datasets_list = get_datasets_list_from_string(config.datasets, config)

    # Iterate over datasets
    for i, dataset_id in enumerate(datasets_list, 1):
        logger.info(f"Dataset ID: {dataset_id} ({i} / {len(datasets_list)})")

        try:
            dataset_cls = get_dataset_class_by_id(dataset_id, config.extra_dataset_registries)
            dataset: BaseDataset = dataset_cls(
                raw_datasets_dir=config.raw_datasets_dir,
                text_datasets_dir=config.text_datasets_dir,
                workers=get_auto_workers(config.workers),
                limit=config.limit,
                override_output=config.override,
                output_format=config.output_format,
                output_compression=config.output_compression,
                output_batch_size=config.output_batch_size,
                json_ensure_ascii=config.json_ensure_ascii,
                skip_items=config.skip_items,
                max_output_chunk_uncompressed_bytes=get_bytes_from_int_or_string(
                    config.max_output_chunk_uncompressed_bytes
                ),
                min_length=config.min_text_length,
                config=config,
                **config.get_extra_dataset_kwargs(dataset_id),
            )

            if dataset.is_dummy():
                logger.warning(f"Skipping {dataset_id} (cannot extract from dummy datasets)")
                continue

            if config.only_selected_datasets and not dataset.is_selected():
                logger.info("Skip %s (not part of selected datasets or sources)", dataset_id)
                continue

            try:
                dataset.extract_plaintext()
            except FileExistsError as e:
                logger.error(f"Cannot extract text from {dataset_id}: {e}")
        except KeyboardInterrupt as e:
            logger.error(f"Stopping... {e}")
            break
        except BaseException as e:
            if config.ignore_errors:
                logger.error(f"Unexpected error occured with {dataset_id}: {e}")
            else:
                raise e

    logger.info("Done")


def extract_text_with_datatrove(config: Config):
    """Using DataTrove framework to extract text and write to dataset specific outputs."""
    logger = config.init_logger(__name__)
    log_file_path = Path(config.log_file)
    logging_dir = log_file_path.parent / log_file_path.stem

    if logging_dir.exists() and config.override:
        logger.warning("Removing existing logging dir (override is enabled): %s", logging_dir)
        shutil.rmtree(logging_dir)

    if config.output_format == "jsonl":
        output_stage_cls = JsonlWriter
    elif config.output_format == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        class DatatroveParquetWriterWithSchema(ParquetWriter):
            # TODO hard-coded schema
            def _write_batch(self, filename):
                if not self._batches[filename]:
                    return
                import pyarrow as pa

                # prepare batch
                batch = pa.RecordBatch.from_pylist(self._batches.pop(filename))
                # write batch
                try:
                    self._writers[filename].write_batch(batch)
                except ValueError as e:
                    print(batch)
                    raise e

            def _write(self, document: dict, file_handler: IO, filename: str):
                parquet_schema = pa.schema(
                    [
                        ("text", pa.string()),
                        ("id", pa.string()),
                        (
                            "metadata",
                            pa.struct(
                                [
                                    ("tlsh", pa.string()),
                                    ("url", pa.string()),
                                ]
                            ),
                        ),
                    ]
                )

                if filename not in self._writers:
                    self._writers[filename] = pq.ParquetWriter(file_handler, schema=parquet_schema)
                self._batches[filename].append(document)
                if len(self._batches[filename]) == self.batch_size:
                    self._write_batch(filename)

        output_stage_cls = DatatroveParquetWriterWithSchema  # ParquetWriter
    else:
        raise ValueError(f"Unsupported output format: {config.output_format }")

    datasets_list = get_datasets_list_from_string(config.datasets, config)

    # Iterate over datasets
    for i, dataset_id in enumerate(datasets_list, 1):
        logger.info(f"Dataset ID: {dataset_id} ({i} / {len(datasets_list)})")

        dataset_output_dir = os.path.join(config.text_datasets_dir, dataset_id)

        if not os.path.exists(dataset_output_dir):
            os.makedirs(dataset_output_dir)

        output_kwargs = dict(
            output_folder=dataset_output_dir,
            output_filename="${rank}." + config.output_format,
            # compression=config.output_compression,  # parquet compression not supported!
            max_file_size=get_bytes_from_int_or_string(config.max_output_chunk_uncompressed_bytes),
        )

        executor = LocalPipelineExecutor(
            pipeline=[
                LLMDatasetsDatatroveReader(dataset_id, config, limit=config.limit),
                output_stage_cls(**output_kwargs),  # JSONL or Parquet writer
            ],
            logging_dir=str(logging_dir),
            tasks=1,
            workers=get_auto_workers(config.workers),
        )
        executor.run()

    logger.info("Done")
