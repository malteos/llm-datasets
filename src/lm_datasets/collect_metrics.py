import argparse
import json

import os
import logging


from pathlib import Path

from .datasets.dataset_registry import (
    get_dataset_class_by_id,
    get_registered_dataset_classes,
    get_registered_dataset_ids,
)
from .datasets.base import BaseDataset, GB
from .utils.config import get_common_argparser, parse_args_and_get_config

from datasets import load_dataset

from tqdm.auto import tqdm

import pyarrow.parquet as pq

import polars as pl

from transformers import AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_common_argparser()], add_help=False)

    parser.add_argument("datasets", help="Name of datasets to shuffle (comma separated)")
    parser.add_argument(
        "--shuffled_output_dir",
        default=None,
        type=str,
        help="Shuffled dataset are saved in this directory",
    )
    parser.add_argument(
        "--save_to",
        default=None,
        type=str,
        help="""Save collected stats to this file path (JSON format)""",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        type=str,
        help="Log file is saved at this path",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (log level = debug)")
    parser.add_argument("--override", action="store_true", help="Override existing output files")
    parser.add_argument("--texts_limit", type=int, default=0, help="Limit number of texts generated for each dataset")
    parser.add_argument(
        "--datasets_limit", type=int, default=0, help="Limit number of texts generated for each dataset"
    )
    parser.add_argument("--skip_datasets", type=int, default=0, help="Skip n datasets before starting")
    parser.add_argument(
        "--hf_tokenizer_name_or_path",
        default=None,
        type=str,
        help="""Name or path to HF tokenizer (if is set, tokens are counted)""",
    )
    parser.add_argument(
        "--source_id",
        default=None,
        type=str,
        help="Filter datasets by source ID (used if `datasets`='all_from_source')",
    )
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

    save_to_path = Path(config.save_to)

    if save_to_path.exists() and not config.override:
        raise FileExistsError(f"Cannot save stats because path exists already (fix with --override): {save_to_path}")

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

    if config.hf_tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            config.hf_tokenizer_name_or_path,
            use_fast=True,
        )

        if not config.texts_limit:
            raise ValueError("Tokenizer must be used with --texts_limit argument.")
    else:
        tokenizer = None

    # Iterate over datasets
    dataset_id_to_stats = {}

    for i, dataset_id in enumerate(datasets_list, 1):
        logger.info(f"Dataset ID: {dataset_id} ({i} / {len(datasets_list)})")

        if i <= config.skip_datasets:
            logger.warning(f"Skip dataset")
            continue

        dataset_cls = get_dataset_class_by_id(dataset_id, config.extra_dataset_registries)
        dataset: BaseDataset = dataset_cls(
            output_dir=config.output_dir,
            output_format=config.output_format,
            shuffled_output_dir=config.shuffled_output_dir,
            config=config,
        )

        total_ws_count = 0
        total_byte_count = 0
        texts = []

        for i, text in enumerate(dataset.generate_texts_from_output(shuffled=True, limit=config.texts_limit)):
            # cast to string
            text = str(text)

            ws_count = text.count(" ")
            byte_count = len(text.encode("utf-8"))

            total_ws_count += ws_count
            total_byte_count += byte_count

            if tokenizer:
                texts.append(text)

        # Append to stats
        dataset_id_to_stats[dataset.DATASET_ID] = {
            "whitespace_count": total_ws_count,
            "byte_count": total_byte_count,
        }

        if tokenizer and texts:
            tokenizer_out = tokenizer(
                text=texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_length=True,
            )
            dataset_id_to_stats[dataset.DATASET_ID].update(
                {
                    "tokenizer": config.hf_tokenizer_name_or_path,
                    "tokens_count": sum(tokenizer_out["length"]),
                }
            )

        # Save stats to to JSON after each dataset
        with open(save_to_path, "w") as f:
            json.dump(dataset_id_to_stats, f)

        if config.datasets_limit > 0 and len(dataset_id_to_stats) >= config.datasets_limit:
            logger.warning("Datasets limit reached")
            break

    logger.info("Done")
