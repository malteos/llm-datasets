from argparse import ArgumentParser

from llm_datasets.commands.chunkify_command import ChunkifyCommand
from llm_datasets.commands.collect_metrics_command import CollectMetricsCommand
from llm_datasets.commands.compose_command import ComposeCommand
from llm_datasets.commands.convert_parquet_to_jsonl_command import ConvertParquetToJSONLCommand
from llm_datasets.commands.extract_text_command import ExtractTextCommand
from llm_datasets.commands.hf_upload_command import HFUploadCommand
from llm_datasets.commands.print_stats_command import PrintStatsCommand
from llm_datasets.commands.shuffle_command import ShuffleCommand
from llm_datasets.commands.train_tokenizer_command import TrainTokenizerCommand
from llm_datasets.commands.render_docs_command import RenderDocsCommand


def main():
    parser = ArgumentParser("lm-datasets", usage="lm-datasets <command> [<args>]")
    commands_parser = parser.add_subparsers(help="lm-datasets command helpers")

    # Register commands
    ChunkifyCommand.register_subcommand(commands_parser)
    CollectMetricsCommand.register_subcommand(commands_parser)
    ComposeCommand.register_subcommand(commands_parser)
    ConvertParquetToJSONLCommand.register_subcommand(commands_parser)
    ExtractTextCommand.register_subcommand(commands_parser)
    HFUploadCommand.register_subcommand(commands_parser)
    PrintStatsCommand.register_subcommand(commands_parser)
    ShuffleCommand.register_subcommand(commands_parser)
    TrainTokenizerCommand.register_subcommand(commands_parser)
    RenderDocsCommand.register_subcommand(commands_parser)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
