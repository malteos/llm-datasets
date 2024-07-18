import json
import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter, ParquetWriter
from llm_datasets.datatrove_reader import LLMDatasetsDatatroveReader
from llm_datasets.utils.config import Config

from tests.dummy_datasets import get_dummy_dataset_cls, save_texts_for_temp_datasets


def test_datatrove_reader_to_jsonl():
    expected_docs_len = 1000
    ds_clss = [
        get_dummy_dataset_cls(size, prefix)
        for size, prefix in zip([expected_docs_len], ["a"])
    ]
    config = Config(
        selected_source_ids=["dummy"],
        extra_dataset_classes=ds_clss,
        use_default_dataset_registry=False,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "output")
        output_dir_path = Path(output_dir)

        list_of_saved_texts, list_of_dataset_ids = save_texts_for_temp_datasets(
            config,
            temp_dir,
            output_format="parquet",
        )

        assert len(list_of_dataset_ids) == 1

        executor = LocalPipelineExecutor(
            pipeline=[
                LLMDatasetsDatatroveReader(
                    "dummy_a1000", config, limit=expected_docs_len
                ),
                JsonlWriter(output_folder=output_dir, compression=None),
            ],
            tasks=1,
            workers=1,
        )
        executor.run()

        # load output docs from disk
        texts = []
        for fp in sorted(output_dir_path.glob("*.jsonl")):
            with open(fp) as f:
                for line in f:
                    texts.append(json.loads(line)["text"])

        assert len(texts) == expected_docs_len

        assert len(texts) == expected_docs_len
        assert np.array_equal(texts, list_of_saved_texts[0])

    print("done")


def test_datatrove_reader_to_parquet_chunks():
    expected_docs_len = 1000
    ds_clss = [
        get_dummy_dataset_cls(size, prefix)
        for size, prefix in zip([expected_docs_len], ["a"])
    ]
    config = Config(
        selected_source_ids=["dummy"],
        extra_dataset_classes=ds_clss,
        use_default_dataset_registry=False,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "output")
        output_dir_path = Path(output_dir)

        list_of_saved_texts, list_of_dataset_ids = save_texts_for_temp_datasets(
            config,
            temp_dir,
            output_format="parquet",
        )

        assert len(list_of_dataset_ids) == 1

        executor = LocalPipelineExecutor(
            pipeline=[
                LLMDatasetsDatatroveReader(
                    "dummy_a1000", config, limit=expected_docs_len
                ),
                ParquetWriter(
                    output_folder=output_dir,
                    # compression="gzip",
                    # zstd -> error
                    max_file_size=1024,
                ),
            ],
            tasks=1,
            workers=1,
        )
        executor.run()

        # read data from disk and check
        df = pl.read_parquet(sorted(output_dir_path.glob("*.parquet")))
        texts = df["text"]

        assert len(texts) == expected_docs_len
        assert np.array_equal(texts, list_of_saved_texts[0])

    print("done")


if __name__ == "__main__":
    test_datatrove_reader_to_jsonl()
    # test_datatrove_reader_to_parquet_chunks()
