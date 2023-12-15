from .utils.config import Config
from .utils.dataframe import get_datasets_as_dataframe


def print_stats(config: Config) -> str:
    logger = config.init_logger(__name__)

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

    # Print to stdout
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

    return out
