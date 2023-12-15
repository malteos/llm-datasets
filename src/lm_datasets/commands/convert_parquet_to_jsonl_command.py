from argparse import Namespace, _SubParsersAction

from lm_datasets.convert_parquet_to_jsonl import convert_parquet_to_jsonl
from lm_datasets.commands import BaseCLICommand
from lm_datasets.utils.config import Config


class ConvertParquetToJSONLCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        subcommand_parser = parser.add_parser("convert_parquet_to_jsonl", help="Convert Parquet files to JSONL")

        subcommand_parser.add_argument(
            "input_dir_or_file", help="Directory containing *.parquet files or the file itself"
        )
        subcommand_parser.add_argument("output_dir", help="Directory containing *.parquet files")
        subcommand_parser.add_argument("--override", action="store_true", help="Override existing output files")
        subcommand_parser.add_argument("--input_glob", type=str, default="*.parquet", help="Glob pattern")

        subcommand_parser = BaseCLICommand.add_common_args(
            subcommand_parser,
            log=True,
        )
        subcommand_parser.set_defaults(func=ConvertParquetToJSONLCommand)

    def __init__(self, args: Namespace) -> None:
        self.config: Config = Config(**args.__dict__)

    def run(self) -> None:
        self.config.init_logger(__name__)

        convert_parquet_to_jsonl(
            input_dir_or_file=self.config.input_dir_or_file,
            output_dir=self.config.output_dir,
            override=self.config.override,
            input_glob=self.config.input_glob,
        )
