import json
from pathlib import Path
import pytest

from llm_datasets.datasets.base import BaseDataset
from llm_datasets.datasets.dataset_registry import get_dataset_class_by_id
from llm_datasets.datatrove_reader import LLMDatasetsDatatroveReader
from llm_datasets.utils.config import Config

from datatrove.executor import LocalPipelineExecutor

from datatrove.pipeline.readers import CSVReader
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import JsonlWriter, ParquetWriter


def test_datatrove_reader():
    output_dir = "./data/tmp-datatrove-out"
    output_dir_path = Path(output_dir)
    expected_docs_len = 10
    config = Config()
    executor = LocalPipelineExecutor(
        pipeline=[
            LLMDatasetsDatatroveReader("legal_mc4_en", config, limit=expected_docs_len),
            JsonlWriter(output_folder=output_dir, compression=None),
        ],
        tasks=1,
        workers=1,
    )
    executor.run()

    # load output docs from disk
    docs = []
    for fp in sorted(output_dir_path.glob("*.jsonl")):
        with open(fp) as f:
            for line in f:
                docs.append(json.loads(line))

    assert len(docs) == expected_docs_len
    assert docs[0]["text"].startswith("(1) The scope of the individual services")
    assert docs[3]["text"].endswith("is not technically feasible.")
    assert docs[9]["text"].startswith("The Second Circuit’s recent decision")

    print("done")


def test_datatrove_reader_to_parquet_chunks():
    output_dir = "./data/tmp-datatrove-out"
    output_dir_path = Path(output_dir)
    expected_docs_len = 10000
    config = Config()
    executor = LocalPipelineExecutor(
        pipeline=[
            LLMDatasetsDatatroveReader("legal_mc4_en", config, limit=expected_docs_len),
            ParquetWriter(output_folder=output_dir, compression=None, max_file_size=1024),
        ],
        tasks=1,
        workers=1,
    )
    executor.run()

    pass

    print("done")


if __name__ == "__main__":
    # test_datatrove_reader()
    test_datatrove_reader_to_parquet_chunks()
