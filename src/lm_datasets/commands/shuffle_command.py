from argparse import Namespace, _SubParsersAction

from lm_datasets.shuffle_datasets import shuffle_datasets
from lm_datasets.commands import BaseCLICommand
from lm_datasets.utils.config import Config, get_config_from_paths
from lm_datasets.utils.settings import DEFAULT_MIN_FILE_SIZE_FOR_BUFFERED_SHUFFLING


class ShuffleCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        subcommand_parser = parser.add_parser(
            "shuffle", help="Shuffle the individual datasets on the file-chunk level (no global shuffle!)"
        )

        subcommand_parser.add_argument("datasets", help="Name of datasets to shuffle (comma separated)")
        subcommand_parser.add_argument(
            "--shuffled_output_dir",
            default=None,
            type=str,
            help="Shuffled dataset are saved in this directory",
        )
        subcommand_parser.add_argument(
            "--output_compression",
            default=None,
            type=str,
            help="""Compression of output (jsonl: "gzip"; parquet: "NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD")""",
        )
        subcommand_parser.add_argument(
            "--seed",
            default=0,
            type=int,
            help="Random seed",
        )
        subcommand_parser.add_argument(
            "--shuffle_buffer_size",
            default=1_000_000,
            type=int,
            help="Number of items in buffer to be shuffled at once (larger buffer = more memory but better shuffing)",
        )
        subcommand_parser.add_argument(
            "--min_file_size_for_buffered_shuffling",
            default=DEFAULT_MIN_FILE_SIZE_FOR_BUFFERED_SHUFFLING,
            type=int,
            help="Min. file size bytes for buffered shuffling (default: 5GB; set to 0 to disable)",
        )
        subcommand_parser.add_argument("--override", action="store_true", help="Override existing output files")
        subcommand_parser.add_argument(
            "--skip_large_datasets",
            action="store_true",
            help="Skip datasets with bytes > --min_file_size_for_buffered_shuffling",
        )
        subcommand_parser.add_argument(
            "--source_id",
            default=None,
            type=str,
            help="Filter datasets by source ID (used if `datasets`='all_from_source')",
        )

        subcommand_parser = BaseCLICommand.add_common_args(
            subcommand_parser,
            raw_datasets_dir=True,
            output=True,
            extra_dataset_registries=True,
            configs=True,
            required_configs=False,
            log=True,
        )
        subcommand_parser.set_defaults(func=ShuffleCommand)

    def __init__(self, args: Namespace) -> None:
        self.config: Config = get_config_from_paths(args.config_paths, override=args.__dict__)

    def run(self) -> None:
        shuffle_datasets(config=self.config)
