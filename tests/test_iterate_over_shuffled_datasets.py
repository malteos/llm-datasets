import os
import pytest
from dataclasses import dataclass
from lm_datasets.datasets.dataset_registry import get_registered_dataset_classes
from lm_datasets.datasets.base import BaseDataset

import pyarrow.parquet as pq

SHUFFLED_OUTPUT_DIR = "/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts_shuffled"


@pytest.mark.skipif(not os.path.exists(SHUFFLED_OUTPUT_DIR), reason="test file not exists")
def test_iterate_over_datasets():
    id_to_dataset_class = {cls.DATASET_ID: cls for cls in get_registered_dataset_classes()}
    dataset_cls = id_to_dataset_class["openlegaldata"]

    @dataclass
    class Arguments(object):
        output_dir = "/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts"
        shuffled_output_dir = SHUFFLED_OUTPUT_DIR
        output_format = "parquet"
        output_compression = "zst"

        skip_ratio = 0.01  # use these as vaidation set
        skip_n = 1000

        train_ratio = 0.05  # number of rows to use for tokenizer training

    args = Arguments()

    dataset: BaseDataset = dataset_cls(
        output_dir=args.output_dir,
        output_format=args.output_format,
        output_compression=args.output_compression,
        shuffled_output_dir=args.shuffled_output_dir,
    )

    batch_size = 1000
    # text_column_name = "text"

    def generate_texts():
        for fp in sorted(dataset.get_chunked_output_file_paths(shuffled=True)):
            pq_fp = fp

            with open(pq_fp, "rb") as file_handler:
                # pq_file = open_parquet_file_with_retries(dataset.get_output_file_path(), retries=2)
                pq_file = pq.ParquetFile(file_handler)

                print(f"rows {pq_file.metadata.num_rows}")

                rows = 0
                max_rows = int(pq_file.metadata.num_rows * args.train_ratio)
                skip_rows = min(int(pq_file.metadata.num_rows * args.skip_ratio), args.skip_n)

                print(f"reading {max_rows} rows")
                print(f"skipping {skip_rows} rows")

                for batch in pq_file.iter_batches(batch_size):
                    for text in batch[dataset.output_text_field]:
                        yield text

                        rows += 1

                print()
            break

    for text in generate_texts():
        print(text)

    print("x")
