import argparse

import logging
import shutil

from llm_datasets.utils import get_auto_workers, get_bytes_from_int_or_string
from llm_datasets.utils.settings import DEFAULT_MIN_TEXT_LENGTH

from llm_datasets.datasets.dataset_registry import (
    get_dataset_class_by_id,
    get_datasets_list_from_string,
    get_registered_dataset_classes,
    get_registered_dataset_ids,
)
from llm_datasets.datasets.base import BaseDataset
from llm_datasets.utils.config import Config, get_common_argparser, parse_args_and_get_config

import json
from pathlib import Path
import pytest

from llm_datasets.datasets.base import BaseDataset
from llm_datasets.datasets.dataset_registry import get_dataset_class_by_id
from llm_datasets.datatrove_reader import LLMDatasetsDatatroveReader
from llm_datasets.utils.config import Config

from datatrove.executor import LocalPipelineExecutor

from datatrove.pipeline.readers import CSVReader
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import JsonlWriter, ParquetWriter


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
    """
    Using DataTrove framework to extract text and write to dataset specific outputs.
    """
    logger = config.init_logger(__name__)
    log_file_path = Path(config.log_file)
    logging_dir = log_file_path.parent / log_file_path.stem

    if logging_dir.exists() and config.override:
        logger.warning("Removing existing logging dir (override is enabled): %s", logging_dir)
        shutil.rmtree(logging_dir)

    if config.output_format == "jsonl":
        output_stage_cls = JsonlWriter
    elif config.output_format == "parquet":
        output_stage_cls = ParquetWriter
    else:
        raise ValueError(f"Unsupported output format: {config.output_format }")

    datasets_list = get_datasets_list_from_string(config.datasets, config)

    # Iterate over datasets
    for i, dataset_id in enumerate(datasets_list, 1):
        logger.info(f"Dataset ID: {dataset_id} ({i} / {len(datasets_list)})")

        output_kwargs = dict(
            output_folder=config.text_datasets_dir,
            output_filename=dataset_id + ".${rank}." + config.output_format,
            compression=config.output_compression,
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
