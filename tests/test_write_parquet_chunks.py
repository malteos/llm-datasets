import random
import string
import logging

import os
import pyarrow as pa
from lm_datasets.io.parquet import save_texts_to_parquet_chunks

from lm_datasets.utils.settings import LOGGING_KWARGS

from pathlib import Path
from random import randint
import tempfile

logging.basicConfig(**LOGGING_KWARGS)

logger = logging.getLogger(__name__)

# TMP_DIR = Path(__file__).parent.parent / "data/tmp"


def get_texts(n: int, min_len: int, max_len: int):
    for i in range(n):
        random_len = randint(min_len, max_len)
        random_str = "".join(random.choices(string.ascii_lowercase, k=random_len))

        yield random_str


def test_write_parquet_chunks():
    random.seed(0)

    with tempfile.TemporaryDirectory() as TMP_DIR:
        TMP_DIR = Path(TMP_DIR)

        print(f"Temp dir = {TMP_DIR}")

        def get_output_path(part=None, total_parts=None) -> str:
            if total_parts:
                return TMP_DIR / f"output.part-{part:04d}-of-{total_parts:04d}.parquet"
            elif part:
                return TMP_DIR / f"output.part-{part:04d}.parquet"
            else:
                return TMP_DIR / "output.parquet"

        if not os.path.exists(TMP_DIR):
            os.makedirs(TMP_DIR)

        # remove all tmp files
        for fp in TMP_DIR.glob("*.parquet"):
            os.remove(fp)
            print(f"Removed {fp}")

        output_text_field = "text"
        schema = pa.schema(
            [
                (output_text_field, pa.string()),
            ]
        )

        saved_docs, saved_chunks = save_texts_to_parquet_chunks(
            texts=get_texts(n=10_000, min_len=250, max_len=10_000),  # approx 50 MB  (none compression)
            schema=schema,
            max_chunk_uncompressed_bytes=10 * 1024 * 1024,  # 10 MB
            output_path_func=get_output_path,
            compression="ZSTD",
            batch_size=24,
        )

        assert saved_docs == 10_000
        assert saved_chunks == 5

        limited_saved_docs, limited_saved_chunks = save_texts_to_parquet_chunks(
            texts=get_texts(n=10_000, min_len=250, max_len=10_000),  # approx 50 MB  (none compression)
            schema=schema,
            max_chunk_uncompressed_bytes=10 * 1024 * 1024,  # 10 MB
            output_path_func=get_output_path,
            compression="ZSTD",
            batch_size=24,
            limit=1000,
        )

        assert limited_saved_docs == 1008  # more than limit due to batches
        assert limited_saved_chunks == 1

        single_file_saved_docs, single_file_saved_chunks = save_texts_to_parquet_chunks(
            texts=get_texts(n=1_000, min_len=250, max_len=10_000),  # approx 5 MB (none compression)
            schema=schema,
            max_chunk_uncompressed_bytes=0,
            output_path_func=get_output_path,
            compression="ZSTD",
            batch_size=24,
        )

        assert single_file_saved_docs == 1_000
        assert single_file_saved_chunks == 1

        print("done")


if __name__ == "__main__":
    test_write_parquet_chunks()
