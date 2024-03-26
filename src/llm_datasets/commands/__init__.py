from abc import ABC, abstractmethod
from argparse import _SubParsersAction


class BaseCLICommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: _SubParsersAction):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    @staticmethod
    def add_common_args(
        parser: _SubParsersAction,
        raw_datasets_dir=False,
        output=False,
        extra_dataset_registries=False,
        configs=False,
        required_configs=False,
        log=False,
    ):
        if raw_datasets_dir:
            parser.add_argument(
                "--raw_datasets_dir",
                default=None,
                type=str,
                help="Dataset files are read from this directory (needed for `is_downloaded` field)",
            )

        if output:
            parser.add_argument(
                "--output_dir",
                default=None,
                type=str,
                help="Processed dataset are saved in this directory (need for `has_output_file` field)",
            )
            parser.add_argument(
                "--output_format",
                default="jsonl",
                type=str,
                help="Format of processed dataset (jsonl,parquet; need for `has_output_file` field)",
            )

        if extra_dataset_registries:
            parser.add_argument(
                "--extra_dataset_registries",
                default=None,
                type=str,
                help="List of Python packages to load dataset registries",
            )

        if configs:
            parser.add_argument(
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

        if log:
            parser.add_argument(
                "--log_file",
                default=None,
                type=str,
                help="Log file is saved at this path",
            )
            parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (log level = debug)")

        return parser
