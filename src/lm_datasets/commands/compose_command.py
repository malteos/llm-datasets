from argparse import Namespace, _SubParsersAction
import logging

from lm_datasets.compose_dataset import compose_dataset
from lm_datasets.commands import BaseCLICommand
from lm_datasets.utils.config import Config, get_config_from_paths
from lm_datasets.utils.dataset_generator import DatasetSplit


logger = logging.getLogger(__name__)


class ComposeCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        subcommand_parser = parser.add_parser(
            "compose", help="Compose the final train/validation set based on the individual datasets"
        )

        subcommand_parser.add_argument(
            "--split", type=DatasetSplit, help="Dataset split (full, train, tokenizer_train, validation)"
        )
        subcommand_parser.add_argument(
            "--shuffled_output_dir",
            help="Shuffled output is saved in this directory (<language code>/<source name>.<jsonl/parquet>)",
        )
        subcommand_parser.add_argument(
            "--composed_dataset_dir",
            required=True,
            type=str,
            help="""Save composed dataset this directory""",
        )
        subcommand_parser.add_argument(
            "--save_dataset_ids", action="store_true", help="Save dataset ID in addition to text field"
        )
        subcommand_parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="""Limit number of output samples (for debugging)""",
        )
        subcommand_parser.add_argument(
            "--output_batch_size",
            type=int,
            default=1000,
            help="""Write batch size; smaller batch size = more accurate splitts but slower""",
        )
        subcommand_parser.add_argument(
            "--input_batch_size",
            type=int,
            default=1000,
            help="""Reader batch size; smaller batch size = less memory consuption but slower""",
        )
        subcommand_parser.add_argument(
            "--interleave_random_batch_size",
            type=int,
            default=100,
            help="""Datasets are randomly interleaves with this batch size""",
        )
        subcommand_parser.add_argument(
            "--max_output_chunk_uncompressed_bytes",
            type=str,
            default="10GB",
            help="Chunks are splitted if they exceed this byte count (<n>, <n>KB, <n>MB, or <n>GB)",
        )
        subcommand_parser.add_argument(
            "--disable_sampling",
            action="store_false",
            dest="use_sampling",
            help="Disable dataset up/down sampling based on sampling factors (see config file)",
        )
        subcommand_parser.add_argument(
            "--use_separated_validation_sets",
            action="store_true",
            help="If enabled, validation set is separated stored on disk (one for each dataset)",
        )

        subcommand_parser = BaseCLICommand.add_common_args(
            subcommand_parser,
            raw_datasets_dir=True,
            output=True,
            extra_dataset_registries=True,
            configs=True,
            required_configs=True,
            log=True,
        )
        subcommand_parser.set_defaults(func=ComposeCommand)

    def __init__(self, args: Namespace) -> None:
        self.config: Config = get_config_from_paths(args.config_paths, override=args.__dict__)

    def run(self) -> None:
        compose_dataset(config=self.config)
