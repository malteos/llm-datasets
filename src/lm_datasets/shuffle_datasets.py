import os

from .datasets.dataset_registry import (
    get_dataset_class_by_id,
    get_registered_dataset_ids,
)
from .datasets.base import BaseDataset
from .utils.config import Config

from datasets import load_dataset

from tqdm.auto import tqdm

import pyarrow.parquet as pq

import polars as pl


def shuffle_datasets(config: Config):
    """
    input: parquet files, configs

    output: shuffled files
    """

    logger = config.init_logger(__name__)

    if config.output_format != "parquet":
        raise ValueError(
            f"Output format is not supported (currently only parquet is supported; {config.output_format=})"
        )

    seed = config.seed
    logger.info(f"Seed: {seed}")

    shuffle_buffer_size = config.shuffle_buffer_size

    datasets_list = config.datasets.split(",")

    if len(datasets_list) == 1:
        if datasets_list[0] == "all":
            # Get list of all regsitered datasets
            datasets_list = get_registered_dataset_ids(config.extra_dataset_registries)

        elif datasets_list[0] == "all_from_source":
            # Get registered datasets based on source
            if config.source_id is None:
                raise ValueError("The argument --source_id must be set.")

            datasets_list = get_registered_dataset_ids(
                config.extra_dataset_registries, needed_source_id=config.source_id
            )

    min_file_size_for_buffered_shuffling = config.min_file_size_for_buffered_shuffling

    # Iterate over datasets
    for i, dataset_id in enumerate(datasets_list, 1):
        logger.info(f"Dataset ID: {dataset_id} ({i} / {len(datasets_list)})")

        dataset_cls = get_dataset_class_by_id(dataset_id, config.extra_dataset_registries)
        dataset: BaseDataset = dataset_cls(
            output_dir=config.output_dir,
            output_format=config.output_format,
            output_batch_size=shuffle_buffer_size,
            output_compression=config.output_compression,
            shuffled_output_dir=config.shuffled_output_dir,
            config=config,
        )

        if not dataset.has_output_files():
            logger.warning(f"Skipping {dataset_id}: cannot shuffle dataset without processed output files")
            continue

        unshuffled_output_file_paths = dataset.get_output_file_paths()

        for i, unshuffled_file_path in enumerate(sorted(unshuffled_output_file_paths), i):
            logger.info(
                f"Shuffling {unshuffled_file_path} ({i} / {len(unshuffled_output_file_paths)} files of {dataset_id})"
            )

            shuffled_output_file_path = dataset.get_shuffled_output_file_path(unshuffled_file_path)

            assert str(shuffled_output_file_path) != str(unshuffled_file_path)

            if os.path.exists(shuffled_output_file_path):
                if config.override:
                    logger.warning("Overriding existing shuffled output file")
                else:
                    logger.warning(
                        f"Skipping {dataset_id}: Shuffled output file exist already ({shuffled_output_file_path})"
                    )
                    continue

            # File size
            file_stats = os.stat(unshuffled_file_path)

            if min_file_size_for_buffered_shuffling > 0 and file_stats.st_size > min_file_size_for_buffered_shuffling:
                # File is too large to be shuffled all at once => shuffle in chunks
                if config.skip_large_datasets:
                    logger.info(f"Skip because too large dataset ({file_stats.st_size=})")
                    continue

                # Reading meta from parquet (for progress bar)
                logger.info("Reading metadata ...")
                metadata = pq.read_metadata(unshuffled_file_path)
                docs_count = metadata.num_rows

                logger.info("Initializing HF streaming dataset ...")
                hf_dataset = load_dataset(
                    "parquet", data_files={"train": unshuffled_file_path}, split="train", streaming=True
                )

                logger.info("Shuffling and writing to new file ...")

                def generate_text():
                    for item in tqdm(hf_dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed), total=docs_count):
                        yield str(item[dataset.get_output_text_field()])  # force to str

                # Writer
                shuffled_docs_count, saved_chunks = dataset.save_texts_to_parquet(
                    generate_text(), file_path=shuffled_output_file_path, apply_filter=False
                )

                if docs_count != shuffled_docs_count:
                    logger.error(
                        f"Original and shuffled docs count do not match: {docs_count=} vs {shuffled_docs_count=}"
                    )

            else:
                # Shuffle in memory
                selected_columns = [dataset.get_output_text_field()]
                logger.info("Initializing PL in-memory dataframe (%s) ...", selected_columns)
                df = pl.read_parquet(unshuffled_file_path, columns=selected_columns)

                logger.info("Shuffling and writing to new file ...")
                df = df.sample(fraction=1, shuffle=True, seed=seed).write_parquet(
                    shuffled_output_file_path, compression="zstd"
                )

    logger.info("Done")
