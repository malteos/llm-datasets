import argparse

import logging

from lm_datasets.utils import get_auto_workers, get_bytes_from_int_or_string

from .datasets.dataset_registry import get_registered_dataset_classes
from .datasets.base import BaseDataset
from .utils.config import get_common_argparser, parse_args_and_get_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_common_argparser()], add_help=False)

    parser.add_argument("datasets", help="Name of datasets to process (comma separated)")
    parser.add_argument(
        "output_dir",
        help="Output is saved in this directory (<language code>/<source name>.<jsonl/parquet>)",
    )
    parser.add_argument("--override", action="store_true", help="Override existing output files")
    parser.add_argument(
        "--ignore_errors",
        action="store_true",
        help="Ignore dataset-level errors (use when processing multiple datasets)",
    )
    parser.add_argument("--json_ensure_ascii", action="store_true", help="Escape non-ASCII characters in JSON output")
    parser.add_argument(
        "--output_compression",
        default=None,
        type=str,
        help="""Compression of output (jsonl: "gzip"; parquet: "NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD")""",
    )
    parser.add_argument(
        "--output_batch_size",
        default=1000,
        type=int,
        help="""Write batch size; smaller batch size = more accurate splitts but slower""",
    )
    parser.add_argument(
        "--max_output_chunk_uncompressed_bytes",
        default="10GB",
        type=str,
        help="Chunks are splitted if they exceed this byte count (<n>, <n>KB, <n>MB, or <n>GB)",
    )
    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        type=str,
        help="Log file is saved at this path",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (log level = debug)")
    parser.add_argument("--limit", default=0, type=int, help="Limit dataset size (for debugging)")
    parser.add_argument(
        "--skip_items",
        default=0,
        type=int,
        help="Skip N items (depending on dataset: directories, subsets, files, documents) (for debugging)",
    )
    parser.add_argument("--hf_auth_token", default=None, type=str, help="HuggingFace auth token")

    config = parse_args_and_get_config(parser)

    log_handlers = [logging.StreamHandler()]

    if config.log_file:
        log_handlers.append(logging.FileHandler(config.log_file))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if config.verbose else logging.INFO,
        handlers=log_handlers,
    )
    logger = logging.getLogger(__name__)

    id_to_dataset_class = {
        cls.DATASET_ID: cls for cls in get_registered_dataset_classes(config.extra_dataset_registries)
    }

    datasets_list = config.datasets.split(",")

    if len(datasets_list) == 1 and datasets_list[0] == "all":
        # Get list of all non-dummy datasets
        datasets_list = id_to_dataset_class.keys()

    # Iterate over datasets
    for i, dataset_id in enumerate(datasets_list, 1):
        if dataset_id in id_to_dataset_class:
            dataset_cls = id_to_dataset_class[dataset_id]
        else:
            raise ValueError(f"Unknown dataset ID: {dataset_id} (available: {id_to_dataset_class.keys()})")

        logger.info(f"Dataset ID: {dataset_id} ({i} / {len(datasets_list)})")

        try:
            dataset: BaseDataset = dataset_cls(
                raw_datasets_dir=config.raw_datasets_dir,
                output_dir=config.output_dir,
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
                config=config,
            )

            if dataset.is_dummy():
                logger.warning(f"Skipping {dataset_id} (cannot extract from dummy datasets)")
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
