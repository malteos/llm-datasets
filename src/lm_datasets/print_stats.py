import argparse

import logging

import pandas as pd

from .datasets.dataset_registry import get_registered_dataset_classes
from .datasets.base import BaseDataset
from .utils.config import get_common_argparser, parse_args_and_get_config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_datasets_as_dataframe(
    output_dir=None, output_format="jsonl", raw_datasets_dir=None, extra_dataset_registries=None
):
    # convert datasets to table rows
    rows = []

    for ds_cls in get_registered_dataset_classes(extra_dataset_registries=extra_dataset_registries):
        ds: BaseDataset = ds_cls(
            output_dir=output_dir,
            output_format=output_format,
            raw_datasets_dir=raw_datasets_dir,
        )
        row = dict(
            dataset_id=ds.DATASET_ID,
            source_id=ds.get_source_id(),
            title=ds.TITLE,
            homepage=ds.HOMEPAGE,
            language=ds.get_language_code(),
            tokens=ds.get_tokens(),
            bytes=ds.BYTES,
            is_downloaded=None,
            web_crawled=1 if ds.WEB_CRAWLED else 0,
            availibility=ds.AVAILIBILITY,
            dummy=1 if ds.DUMMY else 0,
            has_output_file=None,
        )

        if raw_datasets_dir:
            row["is_downloaded"] = 1 if ds.is_downloaded() else 0

        if output_dir:
            row["has_output_file"] = 1 if ds.has_output_files(min_file_size=1024) else 0

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_common_argparser()], add_help=False)

    parser.add_argument("--print_format", default="csv", type=str, help="Print format (tsv,csv,md)")
    config = parse_args_and_get_config(parser)

    print_format = config.print_format

    df = get_datasets_as_dataframe(
        output_dir=config.output_dir if config.output_dir else "/dev/null",
        output_format=config.output_format,
        raw_datasets_dir=config.raw_datasets_dir,
        extra_dataset_registries=config.extra_dataset_registries,
    )

    # to_markdown
    # to_csv
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
