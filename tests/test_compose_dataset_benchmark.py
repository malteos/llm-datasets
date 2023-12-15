import asyncio
import logging
from pathlib import Path
import tempfile
import time
import pyarrow as pa
from lm_datasets.io.parquet import save_texts_to_parquet_chunks

from lm_datasets.utils.config import Config, get_config_from_paths
from lm_datasets.utils.dataset_generator import DatasetGenerator, DatasetSplit

logger = logging.getLogger(__name__)


def test_compose_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        config: Config = get_config_from_paths(
            ["../eulm/lm_datasets_configs/euro_dataset.yml", "../eulm/lm_datasets_configs/pegasus.yml"],
            dict(
                interleave_random_batch_size=1_000,
                input_batch_size=100,
                # output_batch_size=100,  # 10_000
                output_batch_size=10_000,
                use_sampling=True,
                save_dataset_ids=True,
                limit=1_000_000,
                shuffled_output_dir="/data/datasets/lm-datasets_data/euro_dataset_v1_shuffled",
                split=DatasetSplit.TRAIN,
                extra_dataset_registries="internal_lm_datasets.dataset_registry",
                selected_source_ids=[
                    # "colossal_oscar"
                    "norwegian_cc",
                ],
                selected_dataset_ids=[],
            ),
        )

        output_text_field = "text"
        output_dataset_id_field = "dataset_id"
        output_compression = "ZSTD"
        max_output_chunk_uncompressed_bytes = 10 * 1024 * 1024 * 1024  # 10 GB

        dataset_generator = DatasetGenerator(
            config,
            shuffled_output_dir=config.shuffled_output_dir,
            output_format="parquet",
            save_to_dir=Path(temp_dir),
            split=config.split,
        )
        dataset_generator.prepare_datasets(
            use_sampling=config.use_sampling,
            print_progress=True,
        )

        # By default, the only column is the text column
        parquet_columns = [
            (output_text_field, pa.string()),
        ]

        if config.save_dataset_ids:
            # Add `dataset_id` column
            parquet_columns.append((output_dataset_id_field, pa.string()))

        parquet_schema = pa.schema(parquet_columns)

        start_time = time.perf_counter()

        # Save texts from iterator to disk
        save_texts_to_parquet_chunks(
            dataset_generator.generate_texts_from_interleaved_datasets(
                generate_dataset_ids=config.save_dataset_ids,
                print_progress=True,
                random_batch_size=config.interleave_random_batch_size,
                reader_batch_size=config.input_batch_size,
            ),
            parquet_schema,
            output_path_func=dataset_generator.save_to_path,
            max_chunk_uncompressed_bytes=max_output_chunk_uncompressed_bytes,
            compression=output_compression,
            batch_size=config.output_batch_size,
            print_write_progress=0,
            limit=config.limit,
        )

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        logger.info("done")

        logger.info(f"{start_time=} (after preparation)")
        logger.info(f"{end_time=}")
        logger.info(f"{elapsed_time=:0.2f} seconds")
        logger.info(f"{config.limit/(elapsed_time)=}")

        pass


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # level=logging.DEBUG,
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    # asyncio.run(
    test_compose_dataset()
    # )
