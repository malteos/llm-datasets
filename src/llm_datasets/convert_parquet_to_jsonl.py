import argparse
from pathlib import Path
from typing import List
import polars as pl
import logging

logger = logging.getLogger(__name__)


def convert_parquet_to_jsonl(
    input_dir_or_file: str, output_dir: str, override: bool = False, input_glob="*.parquet"
) -> List[str]:
    output_dir_path = Path(output_dir)
    input_path = Path(input_dir_or_file)
    input_paths = [input_path] if input_path.is_file() else input_path.glob(input_glob)
    output_file_paths = []

    logger.debug("Input paths: %s", input_paths)

    for file_path in input_paths:
        logger.debug(f"Converting {file_path}")

        output_file_path = output_dir_path / file_path.with_suffix(".jsonl").name

        if output_file_path.exists() and not override:
            logger.warning(f"Skipping because output exists already: {output_file_path}")

        else:
            df = pl.scan_parquet(file_path, low_memory=True).collect(streaming=True)

            df.write_ndjson(output_file_path)

        output_file_paths.append(output_file_path)

    return output_file_paths
