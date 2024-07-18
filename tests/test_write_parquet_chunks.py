import logging
import os
import random
import string
import tempfile
from pathlib import Path
from random import randint

import pyarrow as pa
from llm_datasets.io.parquet import save_texts_to_parquet_chunks
from llm_datasets.utils.settings import LOGGING_KWARGS

logging.basicConfig(**LOGGING_KWARGS)

logger = logging.getLogger(__name__)

# TMP_DIR = Path(__file__).parent.parent / "data/tmp"


def _get_texts(n: int, min_len: int, max_len: int, extra_column=False):
    for i in range(n):
        random_len = randint(min_len, max_len)
        random_str = "".join(random.choices(string.ascii_lowercase, k=random_len))

        if extra_column:
            yield {"text": random_str, "extra_column": "EXTRA_COLUMN"}
        else:
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
        text_only_schema = pa.schema(
            [
                (output_text_field, pa.string()),
            ]
        )
        # text_and_extra_col_schema = pa.schema([
        #     (output_text_field, pa.string()),
        #     ("extra_column", pa.string()),
        # ])

        # text + extra column

        # saved_docs, saved_chunks = save_texts_to_parquet_chunks(
        #     texts=_get_texts(
        #         n=10_000, min_len=250, max_len=10_000, extra_column=True
        #     ),  # approx 50 MB  (none compression)
        #     schema=text_and_extra_col_schema,
        #     max_chunk_uncompressed_bytes=10 * 1024 * 1024,  # 10 MB
        #     output_path_func=get_output_path,
        #     compression="ZSTD",
        #     batch_size=24,
        # )

        # assert saved_docs == 10_000
        # assert saved_chunks == 5

        # text only

        saved_docs, saved_chunks = save_texts_to_parquet_chunks(
            texts=_get_texts(n=10_000, min_len=250, max_len=10_000),  # approx 50 MB  (none compression)
            schema=text_only_schema,
            max_chunk_uncompressed_bytes=10 * 1024 * 1024,  # 10 MB
            output_path_func=get_output_path,
            compression="ZSTD",
            batch_size=24,
        )

        assert saved_docs == 10_000
        assert saved_chunks == 5

        limited_saved_docs, limited_saved_chunks = save_texts_to_parquet_chunks(
            texts=_get_texts(n=10_000, min_len=250, max_len=10_000),  # approx 50 MB  (none compression)
            schema=text_only_schema,
            max_chunk_uncompressed_bytes=10 * 1024 * 1024,  # 10 MB
            output_path_func=get_output_path,
            compression="ZSTD",
            batch_size=24,
            limit=1000,
        )

        assert limited_saved_docs == 1008  # more than limit due to batches
        assert limited_saved_chunks == 1

        single_file_saved_docs, single_file_saved_chunks = save_texts_to_parquet_chunks(
            texts=_get_texts(n=1_000, min_len=250, max_len=10_000),  # approx 5 MB (none compression)
            schema=text_only_schema,
            max_chunk_uncompressed_bytes=None,
            max_chunk_rows=None,
            output_path_func=get_output_path,
            compression="ZSTD",
            batch_size=24,
        )

        assert single_file_saved_docs == 1_000
        assert single_file_saved_chunks == 1

        print("done")


if __name__ == "__main__":
    test_write_parquet_chunks()
