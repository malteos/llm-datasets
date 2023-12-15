from collections import Counter
import tempfile

import pytest
from lm_datasets.utils.config import Config
from lm_datasets.utils.dataset_generator import DatasetGenerator, DatasetSplit
from tests.dummy_datasets import get_dummy_dataset_cls, save_texts_for_temp_datasets
import logging

logger = logging.getLogger(__name__)


def _test_interval_datasets(
    dataset_sizes,
    dataset_prefixes,
    sampling_factors,
    expected_output_rows,
    split=DatasetSplit.FULL,
    seed=0,
    shuffle_output_file_paths=False,
    max_output_chunk_rows=12,
    max_output_chunk_uncompressed_bytes=None,
    output_batch_size=5,
    output_format="parquet",
    validation_ratio=0.25,
    tokenizer_train_ratio=0.1,
    validation_min_total_docs=None,
    validation_max_split_docs=None,
    validation_min_split_docs=None,
):
    ds_clss = [get_dummy_dataset_cls(size, prefix) for size, prefix in zip(dataset_sizes, dataset_prefixes)]

    config = Config()
    config.seed = seed
    config.extra_dataset_classes = ds_clss
    config.use_default_dataset_registry = False
    config.selected_source_ids = ["dummy"]
    config.validation_ratio = validation_ratio  # number of documents in the split: len(dataset) * ratio
    config.validation_min_total_docs = (
        validation_min_total_docs  # to be used as validation set, the dataset must have at least n docs
    )
    config.validation_max_split_docs = (
        validation_max_split_docs  # number of documents in validation split are capped at this numbers
    )
    config.validation_min_split_docs = (
        validation_min_split_docs  # split must have at least this number of documents, otherwise it will be discarded
    )
    config.tokenizer_train_ratio = tokenizer_train_ratio  # % of train data used for tokenizer training

    # expected_rows_per_dataset_id = {"dummy_b_100": 200, "dummy_a_100": 150, "dummy_c_100": 10}

    with tempfile.TemporaryDirectory() as temp_dir:
        list_of_saved_texts, list_of_dataset_ids = save_texts_for_temp_datasets(
            config,
            temp_dir,
            max_output_chunk_rows=max_output_chunk_rows,
            max_output_chunk_uncompressed_bytes=max_output_chunk_uncompressed_bytes,
            output_format=output_format,
            output_batch_size=output_batch_size,
        )

        config.sampling_factor_by_dataset_id = {dsid: f for dsid, f in zip(list_of_dataset_ids, sampling_factors)}

        dataset_generator = DatasetGenerator(
            config,
            split=split,
            shuffled_output_dir=temp_dir,
            output_format=output_format,
            save_to_dir=temp_dir,
        )
        dataset_generator.prepare_datasets(
            use_sampling=True,
            print_progress=True,
        )
        output_rows = list(
            dataset_generator.generate_texts_from_interleaved_datasets(
                generate_dataset_ids=True,
                print_progress=True,
            )
        )
        rows_per_dataset_id = Counter([ds_id.as_py() for ds_id, sample in output_rows])

        logger.debug(rows_per_dataset_id)

        # compare expected and true row count
        expected_rows_per_dataset_id = {
            dsid: row_count for dsid, row_count in zip(list_of_dataset_ids, expected_output_rows)
        }
        for dataset_id, row_count in expected_rows_per_dataset_id.items():
            assert (
                rows_per_dataset_id[dataset_id] == row_count
            ), f"compare expected and true row count for {dataset_id}. given: {rows_per_dataset_id[dataset_id]}; expected: {row_count}"

    logger.debug("test completed")


def test_simple():
    _test_interval_datasets(
        dataset_sizes=[100, 100, 100],
        dataset_prefixes=["a", "b", "c"],
        sampling_factors=[1.0, 1.5, 0.1],
        expected_output_rows=[100, 150, 10],
        split=DatasetSplit.FULL,
    )


def test_with_sampling():
    _test_interval_datasets(
        dataset_sizes=[100, 51, 93],
        dataset_prefixes=["a", "b", "c"],
        sampling_factors=[3.5, 1.5, 0.1],
        expected_output_rows=[350, 76, 9],
        split=DatasetSplit.FULL,
    )


def test_train_split():
    _test_interval_datasets(
        dataset_sizes=[100, 100, 100],
        dataset_prefixes=["a", "b", "c"],
        sampling_factors=[1.0, 1.0, 1.0],
        expected_output_rows=[75, 75, 75],
        split=DatasetSplit.TRAIN,
        validation_ratio=0.25,
        tokenizer_train_ratio=0.1,
    )


def test_validation_split():
    _test_interval_datasets(
        dataset_sizes=[200, 200, 200],
        dataset_prefixes=["a", "b", "c"],
        sampling_factors=[0.5, 2.0, 3.0],
        expected_output_rows=[25, 100, 150],
        split=DatasetSplit.VALIDATION,
        validation_ratio=0.25,
        tokenizer_train_ratio=0.1,
    )


def test_tokenizer_train_split():
    _test_interval_datasets(
        dataset_sizes=[200, 200, 200],
        dataset_prefixes=["a", "b", "c"],
        sampling_factors=[0.5, 2.0, 3.0],
        expected_output_rows=[7, 30, 45],
        split=DatasetSplit.TOKENIZER_TRAIN,
        validation_ratio=0.25,
        tokenizer_train_ratio=0.1,
    )


def test_validation_with_max_split_docs():
    _test_interval_datasets(
        dataset_sizes=[1000, 1000, 100],
        dataset_prefixes=["a", "b", "c"],
        sampling_factors=[1.0, 1.0, 1.0],
        expected_output_rows=[100, 100, 30],
        split=DatasetSplit.VALIDATION,
        validation_ratio=0.3,
        tokenizer_train_ratio=0.1,
        validation_max_split_docs=100,
    )


def test_validation_min_split_docs():
    _test_interval_datasets(
        dataset_sizes=[1000, 1000, 100],
        dataset_prefixes=["a", "b", "c"],
        sampling_factors=[1.0, 1.0, 1.0],
        expected_output_rows=[300, 300, 0],
        split=DatasetSplit.VALIDATION,
        validation_ratio=0.3,
        tokenizer_train_ratio=0.1,
        validation_min_split_docs=50,
    )


def test_validation_min_total_docs():
    _test_interval_datasets(
        dataset_sizes=[1000, 1000, 100],
        dataset_prefixes=["a", "b", "c"],
        sampling_factors=[1.0, 1.0, 1.0],
        expected_output_rows=[300, 300, 0],
        split=DatasetSplit.VALIDATION,
        validation_ratio=0.3,
        tokenizer_train_ratio=0.1,
        validation_min_total_docs=100 + 1,
    )


@pytest.mark.skip()
def test_large():
    factor = 1000
    _test_interval_datasets(
        dataset_sizes=[100] * factor,
        dataset_prefixes=[f"{i}_" for i in range(factor)],
        sampling_factors=[1.0] * factor,
        expected_output_rows=[7_500] * factor,
        split=DatasetSplit.TRAIN,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        # level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    # test_simple()
    test_train_split()
    test_validation_min_total_docs()
    test_tokenizer_train_split()
    # test_large()
    print("done")
