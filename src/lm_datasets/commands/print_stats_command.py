from argparse import Namespace, _SubParsersAction

from lm_datasets.print_stats import print_stats
from lm_datasets.commands import BaseCLICommand
from lm_datasets.utils.config import Config, get_config_from_paths


class PrintStatsCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        subcommand_parser = parser.add_parser("print_stats", help="Print dataset statistics as CSV, Markdown, ...")

        subcommand_parser.add_argument(
            "--shuffled_output_dir",
            default=None,
            type=str,
            help="Shuffled dataset are saved in this directory",
        )
        subcommand_parser.add_argument(
            "--metrics_dir",
            default=None,
            type=str,
            help="Dataset metrics (whitespace count, byte count, ...) are loaded from this directory (*.json files)",
        )
        subcommand_parser.add_argument(
            "--token_estimation_path",
            default=None,
            type=str,
            help="Path to dataset metrics with token count based on a sample (JSON-file; this is used to estimate the total token count)",
        )
        subcommand_parser.add_argument(
            "--rows_count", action="store_true", help="Extract number of rows from output files"
        )
        subcommand_parser.add_argument(
            "--shuffled_rows_count", action="store_true", help="Extract number of rows from shuffled output files"
        )
        subcommand_parser.add_argument(
            "--output_compression", action="store_true", help="Extract compression from output files"
        )
        subcommand_parser.add_argument("--print_format", default="csv", type=str, help="Print format (tsv,csv,md)")
        subcommand_parser.add_argument("--limit", default=0, type=int, help="Limit the datasets (for debugging)")
        subcommand_parser.add_argument(
            "--save_to",
            default=None,
            type=str,
            help="Save output to this path on the disk (default: only print to stdout)",
        )
        subcommand_parser.add_argument("--exclude_dummy_datasets", action="store_true", help="Exclude dummy datasets")
        subcommand_parser.add_argument(
            "--only_selected_datasets",
            action="store_true",
            help="Include only datasets there were explicitly selected (via config)",
        )
        subcommand_parser.add_argument(
            "--extra_columns",
            default=None,
            type=str,
            help="Comma separated list of columns (see AVAILABLE_DATAFRAME_COLUMNS)",
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
        subcommand_parser.set_defaults(func=PrintStatsCommand)

    def __init__(self, args: Namespace) -> None:
        self.config: Config = get_config_from_paths(args.config_paths, override=args.__dict__)

    def run(self) -> None:
        print_stats(config=self.config)
