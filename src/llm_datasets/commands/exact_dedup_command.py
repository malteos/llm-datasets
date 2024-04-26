from argparse import Namespace, _SubParsersAction
import logging

from llm_datasets.dedup import exact_dedup
from llm_datasets.extract_text import extract_text
from llm_datasets.commands import BaseCLICommand
from llm_datasets.utils.config import Config, get_config_from_paths
from llm_datasets.utils.settings import DEFAULT_MIN_TEXT_LENGTH


logger = logging.getLogger(__name__)


class ExactDedupCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        subcommand_parser = parser.add_parser(
            "exact_dedup", help="Exact deduplication using TLSH local-sensitive hashing"
        )

        subcommand_parser.add_argument("input_dir", help="Dataset files are loaded from this diretory (*.jsonl)")
        subcommand_parser.add_argument(
            "output_dir",
            help="Output is saved in this directory (same file structured as the input data)",
        )
        subcommand_parser.add_argument("--override", action="store_true", help="Override existing output files")
        subcommand_parser.add_argument("--output_gzip", action="store_true", help="Gzip output files")

        subcommand_parser.add_argument("--workers", default=None, type=int, help="Number of parallel processes")
        subcommand_parser.add_argument("--max_lines_per_file", default=0, type=int, help="Limit (debugging)")
        subcommand_parser.add_argument("--max_files", default=0, type=int, help="Limit (debugging)")

        subcommand_parser.add_argument("--print_file_progress", action="store_true", help="print_file_progress")

        subcommand_parser = BaseCLICommand.add_common_args(
            subcommand_parser,
            log=True,
        )
        subcommand_parser.set_defaults(func=ExactDedupCommand)

    def __init__(self, args: Namespace) -> None:
        self.config: Config = get_config_from_paths(args.config_paths, override=args.__dict__)

    def run(self) -> None:
        exact_dedup(
            **self.config.get_key_value_pairs(
                [
                    "input_dir",
                    "output_dir",
                    "override",
                    "output_gzip",
                    "workers",
                    "max_lines_per_file",
                    "max_files",
                    "print_file_progress",
                ]
            )
        )
