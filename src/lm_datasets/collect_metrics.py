import json

from pathlib import Path

from .datasets.dataset_registry import (
    get_dataset_class_by_id,
    get_registered_dataset_ids,
)
from .datasets.base import BaseDataset
from .utils.config import Config

from transformers import AutoTokenizer


def collect_metrics(config: Config):
    logger = config.init_logger(__name__)

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

        if config.only_selected_datasets and not dataset.is_selected():
            logger.info("Skip %s (not part of selected datasets or sources)", dataset_id)
            continue

        total_ws_count = 0
        total_byte_count = 0
        texts = []

        for i, text in enumerate(
            dataset.generate_texts_from_output(shuffled=True, limit=config.texts_limit, shuffle_output_file_paths=True)
        ):
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
            logger.info("Saving stats to %s", save_to_path)
            json.dump(dataset_id_to_stats, f)

        if config.datasets_limit > 0 and len(dataset_id_to_stats) >= config.datasets_limit:
            logger.warning("Datasets limit reached")
            break

    logger.info("Done")
