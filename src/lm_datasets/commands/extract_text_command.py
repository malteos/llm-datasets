from argparse import Namespace, _SubParsersAction
import logging

from lm_datasets.extract_text import extract_text
from lm_datasets.commands import BaseCLICommand
from lm_datasets.utils.config import Config, get_config_from_paths
from lm_datasets.utils.settings import DEFAULT_MIN_TEXT_LENGTH


logger = logging.getLogger(__name__)


class ExtractTextCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        subcommand_parser = parser.add_parser("extract_text", help="Extract text from raw datasets")

        subcommand_parser.add_argument("datasets", help="Name of datasets to process (comma separated)")
        subcommand_parser.add_argument(
            "output_dir",
            help="Output is saved in this directory (<language code>/<source name>.<jsonl/parquet>)",
        )
        subcommand_parser.add_argument("--override", action="store_true", help="Override existing output files")
        subcommand_parser.add_argument(
            "--ignore_errors",
            action="store_true",
            help="Ignore dataset-level errors (use when processing multiple datasets)",
        )
        subcommand_parser.add_argument(
            "--json_ensure_ascii", action="store_true", help="Escape non-ASCII characters in JSON output"
        )
        subcommand_parser.add_argument(
            "--output_compression",
            default=None,
            type=str,
            help="""Compression of output (jsonl: "gzip"; parquet: "NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD")""",
        )
        subcommand_parser.add_argument(
            "--output_batch_size",
            default=1000,
            type=int,
            help="""Write batch size; smaller batch size = more accurate splitts but slower""",
        )
        subcommand_parser.add_argument(
            "--max_output_chunk_uncompressed_bytes",
            default="10GB",
            type=str,
            help="Chunks are splitted if they exceed this byte count (<n>, <n>KB, <n>MB, or <n>GB)",
        )
        subcommand_parser.add_argument(
            "--workers",
            default=1,
            type=int,
            help="Number of workers for parallel processing",
        )
        subcommand_parser.add_argument("--limit", default=0, type=int, help="Limit dataset size (for debugging)")
        subcommand_parser.add_argument(
            "--min_text_length", default=DEFAULT_MIN_TEXT_LENGTH, type=int, help="Text with less length is discarded"
        )
        subcommand_parser.add_argument(
            "--skip_items",
            default=0,
            type=int,
            help="Skip N items (depending on dataset: directories, subsets, files, documents) (for debugging)",
        )
        subcommand_parser.add_argument("--hf_auth_token", default=None, type=str, help="HuggingFace auth token")
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
        subcommand_parser.set_defaults(func=ExtractTextCommand)

    def __init__(self, args: Namespace) -> None:
        self.config: Config = get_config_from_paths(args.config_paths, override=args.__dict__)

    def run(self) -> None:
        extract_text(config=self.config)
