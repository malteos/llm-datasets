"""

input: parquet files, configs

shuffle + train/test split

"""
import argparse

import os
import logging


from pathlib import Path

from .datasets.dataset_registry import get_registered_dataset_classes
from .datasets.base import BaseDataset, GB

from datasets import load_dataset

from tqdm.auto import tqdm

import pyarrow.parquet as pq

import polars as pl


DEFAULT_MIN_FILE_SIZE_FOR_BUFFERED_SHUFFLING = 5 * GB


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", help="Name of datasets to shuffle (comma separated)")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Unshuffled datasets are loaded from this directory",
    )
    parser.add_argument(
        "--shuffled_output_dir",
        default=None,
        type=str,
        help="Shuffled dataset are saved in this directory",
    )
    parser.add_argument(
        "--output_format",
        default="parquet",
        type=str,
        help="Format of processed dataset",
    )
    parser.add_argument(
        "--output_compression",
        default=None,
        type=str,
        help="""Compression of output (jsonl: "gzip"; parquet: "NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD")""",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        default=1_000_000,
        type=int,
        help="Number of items in buffer to be shuffled at once (larger buffer = more memory but better shuffing)",
    )
    parser.add_argument(
        "--min_file_size_for_buffered_shuffling",
        default=DEFAULT_MIN_FILE_SIZE_FOR_BUFFERED_SHUFFLING,
        type=int,
        help="Min. file size bytes for buffered shuffling (default: 5GB; set to 0 to disable)",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        type=str,
        help="Log file is saved at this path",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (log level = debug)")
    parser.add_argument("--override", action="store_true", help="Override existing output files")
    parser.add_argument(
        "--skip_large_datasets",
        action="store_true",
        help="Skip datasets with bytes > --min_file_size_for_buffered_shuffling",
    )
    args = parser.parse_args()

    log_handlers = [logging.StreamHandler()]

    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
        handlers=log_handlers,
    )
    logger = logging.getLogger(__name__)

    if args.output_format != "parquet":
        raise ValueError(f"Output format is not supported (currently only parquet is supported; {args.output_format=})")

    seed = args.seed
    logger.info(f"Seed: {seed}")

    shuffle_buffer_size = args.shuffle_buffer_size

    id_to_dataset_class = {cls.DATASET_ID: cls for cls in get_registered_dataset_classes()}

    datasets_list = args.datasets.split(",")

    if len(datasets_list) == 1 and datasets_list[0] == "all":
        # Get list of all non-dummy datasets
        datasets_list = id_to_dataset_class.keys()

    min_file_size_for_buffered_shuffling = args.min_file_size_for_buffered_shuffling

    # Iterate over datasets
    for i, dataset_id in enumerate(datasets_list, 1):
        if dataset_id in id_to_dataset_class:
            dataset_cls = id_to_dataset_class[dataset_id]
        else:
            raise ValueError(f"Unknown dataset ID: {dataset_id} (available: {id_to_dataset_class.keys()})")

        logger.info(f"Dataset ID: {dataset_id} ({i} / {len(datasets_list)})")

        dataset: BaseDataset = dataset_cls(
            output_dir=args.output_dir,
            output_format=args.output_format,
            output_batch_size=shuffle_buffer_size,
            output_compression=args.output_compression,
        )

        if not dataset.has_output_files():
            logger.warning(f"Skipping {dataset_id}: cannot shuffle dataset without processed output files")
            continue

        output_fps = dataset.get_output_file_paths()

        for i, output_fp in enumerate(sorted(output_fps), i):
            logger.info(f"Shuffling {output_fp} ({i} / {len(output_fps)} files of {dataset_id})")

            output_file_name = Path(output_fp).name

            shuffled_output_file_path = os.path.join(
                args.shuffled_output_dir, output_file_name.replace(".parquet", ".shuffled.parquet")
            )

            assert shuffled_output_file_path != output_fp

            if os.path.exists(shuffled_output_file_path):
                if args.override:
                    logger.warning("Overriding existing shuffled output file")
                else:
                    logger.warning(
                        f"Skipping {dataset_id}: Shuffled output file exist already ({shuffled_output_file_path})"
                    )
                    continue

            # File size
            file_stats = os.stat(output_fp)

            if min_file_size_for_buffered_shuffling > 0 and file_stats.st_size > min_file_size_for_buffered_shuffling:
                # File is too large to be shuffled all at once => shuffle in chunks
                if args.skip_large_datasets:
                    logger.info(f"Skip because too large dataset ({file_stats.st_size=})")
                    continue

                # Reading meta from parquet (for progress bar)
                logger.info("Reading metadata ...")
                metadata = pq.read_metadata(output_fp)
                docs_count = metadata.num_rows

                logger.info("Initializing HF streaming dataset ...")
                hf_dataset = load_dataset("parquet", data_files={"train": output_fp}, split="train", streaming=True)

                logger.info("Shuffling and writing to new file ...")

                def generate_text():
                    for item in tqdm(hf_dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed), total=docs_count):
                        yield str(item[dataset.output_text_field])  # force to str

                # Writer
                shuffled_docs_count = dataset.save_texts_to_parquet(
                    generate_text(), file_path=shuffled_output_file_path, apply_filter=False
                )

                if docs_count != shuffled_docs_count:
                    logger.error(
                        f"Original and shuffled docs count do not match: {docs_count=} vs {shuffled_docs_count=}"
                    )

            else:
                # Shuffle in memory
                logger.info("Initializing PL in-memory dataframe ...")
                df = pl.read_parquet(output_fp)
                logger.info("Shuffling and writing to new file ...")
                df = df.sample(fraction=1, shuffle=True, seed=seed).write_parquet(
                    shuffled_output_file_path, compression="zstd"
                )

    logger.info("Done")
