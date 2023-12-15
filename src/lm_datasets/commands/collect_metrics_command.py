from argparse import Namespace, _SubParsersAction
import logging

from lm_datasets.collect_metrics import collect_metrics
from lm_datasets.commands import BaseCLICommand
from lm_datasets.utils.config import Config, get_config_from_paths


logger = logging.getLogger(__name__)


class CollectMetricsCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        subcommand_parser = parser.add_parser(
            "collect_metrics", help="Collect metrics (token count etc.) from extracted texts"
        )

        subcommand_parser.add_argument("datasets", help="Name of datasets to shuffle (comma separated)")
        subcommand_parser.add_argument(
            "--shuffled_output_dir",
            default=None,
            type=str,
            help="Shuffled dataset are saved in this directory",
        )
        subcommand_parser.add_argument(
            "--save_to",
            default=None,
            type=str,
            help="""Save collected stats to this file path (JSON format)""",
        )
        subcommand_parser.add_argument("--override", action="store_true", help="Override existing output files")
        subcommand_parser.add_argument(
            "--texts_limit", type=int, default=0, help="Limit number of texts generated for each dataset"
        )
        subcommand_parser.add_argument(
            "--datasets_limit", type=int, default=0, help="Limit number of texts generated for each dataset"
        )
        subcommand_parser.add_argument("--skip_datasets", type=int, default=0, help="Skip n datasets before starting")
        subcommand_parser.add_argument(
            "--hf_tokenizer_name_or_path",
            default=None,
            type=str,
            help="""Name or path to HF tokenizer (if is set, tokens are counted)""",
        )
        subcommand_parser.add_argument(
            "--source_id",
            default=None,
            type=str,
            help="Filter datasets by source ID (used if `datasets`='all_from_source')",
        )
        subcommand_parser.add_argument(
            "--only_selected_datasets",
            action="store_true",
            help="Include only datasets there were explicitly selected (via config)",
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
        subcommand_parser.set_defaults(func=CollectMetricsCommand)

    def __init__(self, args: Namespace) -> None:
        self.config: Config = get_config_from_paths(args.config_paths, override=args.__dict__)

    def run(self) -> None:
        collect_metrics(config=self.config)
