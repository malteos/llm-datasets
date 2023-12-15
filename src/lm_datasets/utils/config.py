import argparse
from typing import List, Iterable, Literal, Union
import yaml
import logging


logger = logging.getLogger(__name__)


def get_common_argparser(required_configs: bool = False):
    common_parser = argparse.ArgumentParser()

    common_parser.add_argument(
        "--raw_datasets_dir",
        default=None,
        type=str,
        help="Dataset files are read from this directory (needed for `is_downloaded` field)",
    )
    common_parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Processed dataset are saved in this directory (need for `has_output_file` field)",
    )
    common_parser.add_argument(
        "--output_format",
        default="jsonl",
        type=str,
        help="Format of processed dataset (jsonl,parquet; need for `has_output_file` field)",
    )
    common_parser.add_argument(
        "--extra_dataset_registries",
        default=None,
        type=str,
        help="List of Python packages to load dataset registries",
    )
    common_parser.add_argument(
        "-c",
        "--configs",
        nargs="+",
        help=(
            "Paths to one or more YAML-config files (duplicated settings are override based on config order; "
            "cmd args override configs)"
        ),
        default=None,
        dest="config_paths",
        required=required_configs,
    )
    common_parser.add_argument(
        "--log_file",
        default=None,
        type=str,
        help="Log file is saved at this path",
    )
    common_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (log level = debug)")
    return common_parser


class Config:
    composed_dataset_dir = None  # composed dataset (train/val split) is saved into this directory
    local_dirs_by_dataset_id = {}
    local_dirs_by_source_id = {}
    sampling_factor_by_dataset_id = {}
    sampling_factor_by_source_id = {}
    sampling_factor_by_language = {}

    only_selected_datasets: bool = False
    selected_dataset_ids: List[str] = []
    selected_source_ids: List[str] = []

    validation_ratio = 0.005  # number of documents in the split: len(dataset) * ratio
    validation_min_total_docs = 1_000  # to be used as validation set, the dataset must have at least n docs
    validation_max_split_docs = 1_000  # number of documents in validation split are capped at this numbers
    validation_min_split_docs = 10  # split must have at least this number of documents, otherwise it will be discarded
    tokenizer_train_ratio = 0.1  # % of train data used for tokenizer training

    # Vocab size should divisble by 8
    # - Jan's recommendation: 250680
    # - NVIDIA recommendation for multilingual models: 256000
    tokenizer_vocab_size: int = 256000
    tokenizer_model_type: Literal["bpe", "unigram", "word", "char"] = "bpe"  # SP model types

    seed: int = 0

    extra_dataset_registries: Union[None, str, List[str]] = None
    extra_dataset_classes: Union[None, List] = None
    use_default_dataset_registry: bool = True

    verbose = False
    log_file = None

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def init_logger(self, logger_name):
        log_handlers = [logging.StreamHandler()]

        if self.log_file:
            log_handlers.append(logging.FileHandler(self.log_file))

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.DEBUG if self.verbose else logging.INFO,
            handlers=log_handlers,
        )
        logger = logging.getLogger(logger_name)

        return logger


def get_config_from_paths(config_paths: Iterable, override: dict = None) -> Config:
    config = {}

    if config_paths:
        for config_path in config_paths:
            logger.info("Loading config: %s", config_path)
            with open(config_path) as f:
                _config = yaml.safe_load(f)

                config.update(_config)
    if override:
        # Override with args
        config.update(override)

    config = Config(**config)

    return config


def parse_args_and_get_config(parser):
    args = parser.parse_args()
    config = get_config_from_paths(args.config_paths, override=args.__dict__)

    return config
