import argparse
import yaml
import logging


logger = logging.getLogger(__name__)


def get_common_argparser():
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
        required=False,
    )

    return common_parser


class Config:
    local_dirs_by_dataset_id = {}
    local_dirs_by_source_id = {}

    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_args_and_get_config(parser):
    args = parser.parse_args()
    config = {}

    config = {}

    if args.config_paths:
        for config_path in args.config_paths:
            logger.info(f"Loading config: {config_path}")
            with open(config_path) as f:
                _config = yaml.safe_load(f)

                config.update(_config)

    # Override with args
    config.update(args.__dict__)

    return Config(**config)
