import argparse
from pathlib import Path
import polars as pl
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir_or_file", help="Directory containing *.parquet files or the file itself")
    parser.add_argument("output_dir", help="Directory containing *.parquet files")
    parser.add_argument("--override", action="store_true", help="Override existing output files")
    args = parser.parse_args()
    output_dir_path = Path(args.output_dir)

    input_path = Path(args.input_dir_or_file)
    input_paths = [input_path] if input_path.is_file() else input_path.glob("*.parquet")

    for file_path in input_paths:
        logger.info(f"Converting {file_path}")

        output_file_path = output_dir_path / file_path.with_suffix(".jsonl").name

        if output_file_path.exists() and not args.override:
            logger.warning(f"Skipping because output exists already: {output_file_path}")

        else:
            df = pl.scan_parquet(file_path, low_memory=True).collect(streaming=True)

            df.write_ndjson(output_file_path)

    logger.info("done")
