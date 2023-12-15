import logging
import json
from pathlib import Path
from typing import Iterable, List, Optional
import pandas as pd
from tqdm.auto import tqdm
from lm_datasets.datasets.dataset_registry import get_registered_dataset_classes
from lm_datasets.datasets.base import TOKENS_PER_BYTE, BaseDataset, License
from lm_datasets.utils.config import Config

logger = logging.getLogger(__name__)


def stringify_list(list: Iterable, sep: str = ",") -> str:
    return sep.join([str(e) for e in list])


DEFAULT_TOKENS_PER_WHITESPACE = 2.33  # based on EURO dataset

AVAILABLE_DATAFRAME_COLUMNS = {
    "license": lambda ds: ds.LICENSE,
    "license_commercial_use": lambda ds: ds.LICENSE.commercial_use
    if ds.LICENSE is not None and isinstance(ds.LICENSE, License)
    else None,
    "license_sharealike": lambda ds: ds.LICENSE.sharealike
    if ds.LICENSE is not None and isinstance(ds.LICENSE, License)
    else None,
    "overlap": lambda ds: ",".join(ds.HAS_OVERLAP_WITH),
    "quality_warnings": lambda ds: stringify_list(ds.QUALITY_WARNINGS),
    "genres": lambda ds: stringify_list(ds.GENRES),
    "languages": lambda ds: stringify_list(ds.LANGUAGES),
}


def get_dataframe_row_from_dataset(
    ds: BaseDataset,
    rows_count=False,
    shuffled_rows_count=False,
    output_compression=False,
    metrics: Optional[dict] = None,
    estimated_tokens_per_whitespace: Optional[float] = None,
    extra_columns: Optional[Iterable[str]] = None,
) -> dict:
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
        has_output_files=None,
        sampling_factor=ds.get_sampling_factor(),
    )

    if extra_columns:
        for col_name in extra_columns:
            if col_name in AVAILABLE_DATAFRAME_COLUMNS:
                row[col_name] = AVAILABLE_DATAFRAME_COLUMNS[col_name](ds)
            else:
                raise ValueError(f"Unsupported column: {col_name}")

    if ds.raw_datasets_dir:
        row["is_downloaded"] = 1 if ds.is_downloaded() else 0

    if ds.output_dir:
        row["has_output_files"] = 1 if ds.has_output_files(min_file_size=1024) else 0

        if rows_count:
            row["rows_count"] = ds.get_output_rows_count()

        if output_compression:
            row["output_compression"] = ds.get_compression_from_output_files()

    if ds.shuffled_output_dir is not None:
        row["has_shuffled_output_files"] = 1 if ds.has_output_files(min_file_size=1024, shuffled=True) else 0

        if shuffled_rows_count:
            row["shuffled_rows_count"] = ds.get_output_rows_count(shuffled=True)

        if output_compression:
            row["shuffled_output_compression"] = ds.get_compression_from_output_files(shuffled=True)

        # Estimations are very bad -> do not use
        # row["estimated_bytes"] = ds.get_estimated_bytes_from_output(shuffled=True, read_first_n_rows=10_000)
        # row["estimated_tokens"] = int(row["estimated_bytes"] * TOKENS_PER_BYTE)

        # TODOs
        # has_shuffled_output_files
        # number of (shuffled) output_files
        # number of output_rows
        # compression (gzip/parquet) ---> make everything in the same compression to enable sampling based on byte size

    if metrics:
        row.update(metrics)

        if estimated_tokens_per_whitespace:
            row.update(
                {
                    "estimated_tokens_per_whitespace": estimated_tokens_per_whitespace,
                    "total_estimated_tokens": metrics["whitespace_count"] * estimated_tokens_per_whitespace,
                }
            )

    return row


def get_datasets_as_dataframe(
    output_dir=None,
    output_format="jsonl",
    raw_datasets_dir=None,
    extra_dataset_registries=None,
    shuffled_output_dir=None,
    rows_count=False,
    shuffled_rows_count=False,
    output_compression=False,
    limit: int = 0,
    exclude_dummy_datasets: bool = False,
    show_progress: bool = False,
    workers: int = 1,
    metrics_dir: Optional[str] = None,
    token_estimation_path=None,
    config: Optional[Config] = None,
    extra_columns: Optional[Iterable[str]] = None,
):
    # convert datasets to table rows
    rows = []
    ds_clss = get_registered_dataset_classes(extra_dataset_registries=extra_dataset_registries)

    if limit > 0:
        ds_clss = ds_clss[:limit]

    if workers > 1:
        raise NotImplementedError

    ds_clss_iterator = tqdm(ds_clss, desc="Loading dataset details") if show_progress else ds_clss

    # Metrics (whitespaces, byte count)
    dataset_id_to_metrics = {}

    if metrics_dir:
        for json_path in Path(metrics_dir).glob("*.json"):
            with open(json_path) as f:
                dataset_id_to_metrics.update(json.load(f))

    # Load token estimations (tokens per whitespace)
    dataset_id_to_tokens_per_whitespace = {}

    if token_estimation_path:
        dataset_id_to_tokens_per_whitespace = {}

        with open(token_estimation_path) as f:
            dataset_id_to_metrics_with_tokens = json.load(f)

            for dataset_id, metrics in dataset_id_to_metrics_with_tokens.items():
                # use whitespace_count or bytee?
                if (
                    "whitespace_count" in metrics
                    and metrics["whitespace_count"] > 0
                    and "tokens_count" in metrics
                    and metrics["tokens_count"] > 0
                ):
                    dataset_id_to_tokens_per_whitespace[dataset_id] = (
                        metrics["tokens_count"] / metrics["whitespace_count"]
                    )

    for ds_cls in ds_clss_iterator:
        dataset_id = ds_cls.DATASET_ID
        ds: BaseDataset = ds_cls(
            output_dir=output_dir,
            output_format=output_format,
            raw_datasets_dir=raw_datasets_dir,
            shuffled_output_dir=shuffled_output_dir,
            config=config,
        )

        if config.only_selected_datasets and not ds.is_selected():
            logger.info("Skip %s (not part of selected datasets or sources)", dataset_id)
            continue

        row = get_dataframe_row_from_dataset(
            ds,
            rows_count=rows_count,
            shuffled_rows_count=shuffled_rows_count,
            output_compression=output_compression,
            metrics=dataset_id_to_metrics[dataset_id] if dataset_id in dataset_id_to_metrics else None,
            estimated_tokens_per_whitespace=dataset_id_to_tokens_per_whitespace[dataset_id]
            if dataset_id in dataset_id_to_tokens_per_whitespace
            else None,
            extra_columns=extra_columns,
        )

        if exclude_dummy_datasets and ds.is_dummy():
            logger.info("Skip %s (dummy dataset)", dataset_id)
            continue

        rows.append(row)

    return pd.DataFrame(rows)
