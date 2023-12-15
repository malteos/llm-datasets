from argparse import Namespace, _SubParsersAction

from lm_datasets.train_sp_tokenizer import train_sp_tokenizer
from lm_datasets.commands import BaseCLICommand
from lm_datasets.utils.config import Config, get_config_from_paths

from lm_datasets.utils.settings import DEFAULT_TOKENIZER_RATIO


class TrainTokenizerCommand(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        subcommand_parser = parser.add_parser(
            "train_tokenizer", help="Train a tokenizer (only: sentencepiece supproted)"
        )

        subcommand_parser.add_argument(
            "tokenizer_backend", help="Tokenizer backend (currently only `sentencepiece` supported)"
        )

        subcommand_parser.add_argument(
            "--composed_dataset_dir",
            required=True,
            type=str,
            help="""Save composed dataset this directory""",
        )
        subcommand_parser.add_argument(
            "--source_tokenizer_path",
            default=None,
            type=str,
            help="Source tokenizer is loaded from this path (.model file)",
        )
        subcommand_parser.add_argument(
            "--output_tokenizer_path",
            default=None,
            type=str,
            help="SP tokenizer model is saved to this path",
        )
        subcommand_parser.add_argument(
            "--text_field_name",
            default="text",
            type=str,
            help="Text is read from this field name from composed dataset files.",
        )
        subcommand_parser.add_argument(
            "--tokenizer_ratio",
            default=DEFAULT_TOKENIZER_RATIO,
            type=float,
            help="This ratio of the training set is used for the tokenizer training.",
        )
        subcommand_parser.add_argument(
            "--tokenizer_vocab_size",
            default=256000,
            type=int,
            help="Vocab size.",
        )
        subcommand_parser.add_argument(
            "--tokenizer_model_type",
            default="bpe",
            type=str,
            help="Tokenizer model type (options: bpe, unigram, word, char).",
        )
        subcommand_parser.add_argument(
            "--input_batch_size",
            default=10_000,
            type=int,
            help="Input data is read with this batch size",
        )
        subcommand_parser.add_argument("--override", action="store_true", help="Override existing output files")
        subcommand_parser.add_argument(
            "--sentence_splitting", action="store_true", help="Split into texts into sentences"
        )
        subcommand_parser.add_argument(
            "--workers",
            default=1,
            type=int,
            help="Number of workers for parallel processing",
        )

        subcommand_parser = BaseCLICommand.add_common_args(
            subcommand_parser,
            raw_datasets_dir=False,
            output=True,
            extra_dataset_registries=True,
            configs=True,
            required_configs=True,
            log=True,
        )
        subcommand_parser.set_defaults(func=TrainTokenizerCommand)

    def __init__(self, args: Namespace) -> None:
        self.tokenizer_backend = args.tokenizer_backend
        self.config: Config = get_config_from_paths(args.config_paths, override=args.__dict__)

    def run(self) -> None:
        if self.tokenizer_backend == "sentencepiece":
            train_sp_tokenizer(config=self.config)
        else:
            raise ValueError(f"Unsupported tokenizer backend: {self.tokenizer_backend }")
