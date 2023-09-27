import argparse

from lm_datasets.utils.config import get_common_argparser, parse_args_and_get_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_common_argparser(required_configs=True)], add_help=False)

    parser.add_argument(
        "--source_tokenizer_path",
        default=None,
        type=str,
        help="Source tokenizer is loaded from this path (.model file)",
    )
    parser.add_argument(
        "--tokenizer_output_path",
        default=None,
        type=str,
        help="SP tokenizer model is saved to this path",
    )
    parser.add_argument(
        "--output_format",
        default="parquet",
        type=str,
        help="Format of processed dataset",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        type=str,
        help="Log file is saved at this path",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (log level = debug)")
    parser.add_argument("--override", action="store_true", help="Override existing output files")

    config = parse_args_and_get_config(parser)

    logger = config.init_logger(__name__)

    # from_pretrained_model_path = ""  # existing SP model

    # # train with the same settings
    # spm.SentencePieceTrainer.train(
    #     sentence_iterator=ds.generate_texts_from_output(),
    #     model_prefix=model_prefix,
    #     **self.tokenizer_config,
    # )

    raise NotImplementedError()
