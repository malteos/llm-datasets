import argparse

import logging

import pandas as pd

from .datasets.dataset_registry import get_registered_dataset_classes
from .datasets.base import BaseDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_datasets_as_dataframe(
        output_dir=None,
        output_format="jsonl",
        raw_datasets_dir=None,
        extra_dataset_registries=None
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

    df = pd.DataFrame(rows)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("dataset", help="Name of dataset to process")
    parser.add_argument("--print_format", default="csv", type=str, help="Print format (tsv,csv,md)")
    parser.add_argument(
        "--raw_datasets_dir",
        default=None,
        type=str,
        help="Dataset files are read from this directory (needed for `is_downloaded` field)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Processed dataset are saved in this directory (need for `has_output_file` field)",
    )
    parser.add_argument(
        "--output_format",
        default="jsonl",
        type=str,
        help="Format of processed dataset (need for `has_output_file` field)",
    )
    parser.add_argument(
        "--extra_dataset_registries",
        default=None,
        type=str,
        help="List of Python packages to load dataset registries",
    )
    args = parser.parse_args()
    print_format = args.print_format

    df = get_datasets_as_dataframe(
        output_dir=args.output_dir if args.output_dir else "/dev/null",
        output_format=args.output_format,
        raw_datasets_dir=args.raw_datasets_dir,
        extra_dataset_registries=args.extra_dataset_registries,
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
