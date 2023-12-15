from argparse import Namespace, _SubParsersAction

from lm_datasets.chunkify_datasets import chunkify_datasets
from lm_datasets.commands import BaseCLICommand
from lm_datasets.utils.config import Config, get_config_from_paths


class ChunkifyCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        subcommand_parser = parser.add_parser(
            "chunkify", help="Split the individual datasets into equally-sized file chunks (based on bytes or rows)"
        )

        subcommand_parser.add_argument("datasets", help="Name of datasets to shuffle (comma separated)")
        subcommand_parser.add_argument(
            "--max_uncompressed_bytes_per_chunk",
            default="5GB",
            type=str,
            help="Chunks are splitted if they exceed this byte count (<n>, <n>KB, <n>MB, or <n>GB)",
        )
        subcommand_parser.add_argument(
            "--safety_factor",
            default=0.975,
            type=float,
            help="Max. chunk size is multiplied with this factor (accounts for inaccurate chunk sizes due to batching)",
        )
        subcommand_parser.add_argument(
            "--output_compression",
            default=None,
            type=str,
            help="""Compression of output (default: compression of existing output; jsonl: "gzip"; """
            + """parquet: "NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD")""",
        )
        subcommand_parser.add_argument(
            "--batch_size",
            default=16,
            type=int,
            help="""Read and write batch size; smaller batch size = more accurate splitts but slower""",
        )
        subcommand_parser.add_argument("--override", action="store_true", help="Override existing output files")
        subcommand_parser.add_argument("--rename_original", action="store_true", help="Renames orignal output file")

        subcommand_parser = BaseCLICommand.add_common_args(
            subcommand_parser,
            raw_datasets_dir=False,
            output=True,
            extra_dataset_registries=True,
            configs=True,
            required_configs=False,
            log=True,
        )
        subcommand_parser.set_defaults(func=ChunkifyCommand)

    def __init__(self, args: Namespace) -> None:
        self.config: Config = get_config_from_paths(args.config_paths, override=args.__dict__)

    def run(self) -> None:
        chunkify_datasets(config=self.config)
