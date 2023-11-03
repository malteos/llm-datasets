import argparse

import logging

from .utils.config import get_common_argparser, parse_args_and_get_config
from .utils.dataframe import get_datasets_as_dataframe


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_common_argparser()], add_help=False)
    parser.add_argument(
        "--shuffled_output_dir",
        default=None,
        type=str,
        help="Shuffled dataset are saved in this directory",
    )
    parser.add_argument(
        "--metrics_dir",
        default=None,
        type=str,
        help="Dataset metrics (whitespace count, byte count, ...) are loaded from this directory (*.json files)",
    )
    parser.add_argument(
        "--token_estimation_path",
        default=None,
        type=str,
        help="Path to dataset metrics with token count based on a sample (JSON-file; this is used to estimate the total token count)",
    )
    parser.add_argument("--rows_count", action="store_true", help="Extract number of rows from output files")
    parser.add_argument(
        "--shuffled_rows_count", action="store_true", help="Extract number of rows from shuffled output files"
    )
    parser.add_argument("--output_compression", action="store_true", help="Extract compression from output files")
    parser.add_argument("--print_format", default="csv", type=str, help="Print format (tsv,csv,md)")
    parser.add_argument("--limit", default=0, type=int, help="Limit the datasets (for debugging)")
    parser.add_argument(
        "--save_to", default=None, type=str, help="Save output to this path on the disk (default: only print to stdout)"
    )
    parser.add_argument("--exclude_dummy_datasets", action="store_true", help="Exclude dummy datasets")
    parser.add_argument(
        "--only_selected_datasets",
        action="store_true",
        help="Include only datasets there were explicitly selected (via config)",
    )
    parser.add_argument(
        "--extra_columns",
        default=None,
        type=str,
        help="Comma separated list of columns (see AVAILABLE_DATAFRAME_COLUMNS)",
    )
    config = parse_args_and_get_config(parser)

    print_format = config.print_format

    df = get_datasets_as_dataframe(
        output_dir=config.output_dir if config.output_dir else "/dev/null",
        output_format=config.output_format,
        shuffled_output_dir=config.shuffled_output_dir,
        raw_datasets_dir=config.raw_datasets_dir,
        extra_dataset_registries=config.extra_dataset_registries,
        rows_count=config.rows_count,
        shuffled_rows_count=config.shuffled_rows_count,
        output_compression=config.output_compression,
        limit=config.limit,
        exclude_dummy_datasets=config.exclude_dummy_datasets,
        show_progress=True,
        metrics_dir=config.metrics_dir,
        token_estimation_path=config.token_estimation_path,
        config=config,
        extra_columns=config.extra_columns.split(",") if config.extra_columns else None,
    )

    to_kwargs = dict(index=False)

    if print_format == "tsv":
        out = df.to_csv(sep="\t", **to_kwargs)
    elif print_format == "csv":
        out = df.to_csv(**to_kwargs)
    elif print_format == "md":
        out = df.to_markdown(**to_kwargs)
    else:
        raise ValueError("Unsupported output format: %s" % print_format)

    print(out)

    if config.save_to:
        to_kwargs["path_or_buf"] = config.save_to

        if print_format == "tsv":
            df.to_csv(sep="\t", **to_kwargs)
        elif print_format == "csv":
            df.to_csv(**to_kwargs)
        elif print_format == "md":
            df.to_markdown(**to_kwargs)

        logger.info("Saved to %s", config.save_to)
