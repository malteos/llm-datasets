import argparse

import logging

from lm_datasets.utils import get_auto_workers, get_bytes_from_int_or_string
from lm_datasets.utils.settings import DEFAULT_MIN_TEXT_LENGTH

from .datasets.dataset_registry import (
    get_dataset_class_by_id,
    get_registered_dataset_classes,
    get_registered_dataset_ids,
)
from .datasets.base import BaseDataset
from .utils.config import Config, get_common_argparser, parse_args_and_get_config


def extract_text(config: Config):
    logger = config.init_logger(__name__)

    datasets_list = config.datasets.split(",")

    if len(datasets_list) == 1:
        if datasets_list[0] == "all":
            # Get list of all regsitered datasets
            datasets_list = get_registered_dataset_ids(config.extra_dataset_registries)

        elif datasets_list[0] == "all_from_source":
            # Get registered datasets based on source
            if config.source_id is None:
                raise ValueError("The argument or config `source_id` must be set.")

            datasets_list = get_registered_dataset_ids(
                config.extra_dataset_registries, needed_source_id=config.source_id
            )

    # Iterate over datasets
    for i, dataset_id in enumerate(datasets_list, 1):
        logger.info(f"Dataset ID: {dataset_id} ({i} / {len(datasets_list)})")

        try:
            dataset_cls = get_dataset_class_by_id(dataset_id, config.extra_dataset_registries)
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
                min_length=config.min_text_length,
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
