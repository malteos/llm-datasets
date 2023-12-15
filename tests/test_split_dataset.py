import math
import random
import tempfile
from lm_datasets.datasets.base import BaseDataset
from lm_datasets.utils.config import Config

from lm_datasets.utils.dataset_generator import (
    generate_texts_from_dataset,
    DatasetSplit,
    get_splits_as_offsets_and_limits,
)

from .dummy_datasets import get_dummy_dataset_cls


def _test_train_validation_split(
    dataset_size,
    validation_ratio,
    tokenizer_train_ratio,
    max_output_chunk_rows=None,
    max_output_chunk_uncompressed_bytes=None,
):
    random.seed(0)

    config = Config()
    config.validation_ratio = validation_ratio
    config.validation_min_total_docs = 0
    config.validation_min_split_docs = 0
    config.tokenizer_train_ratio = tokenizer_train_ratio

    expected_validation_size = math.floor(dataset_size * validation_ratio)
    expected_train_size = dataset_size - expected_validation_size
    expected_tokenizer_train_size = math.floor(expected_train_size * tokenizer_train_ratio)

    ds_cls = get_dummy_dataset_cls(dataset_size)

    with tempfile.TemporaryDirectory() as temp_dir:
        ds: BaseDataset = ds_cls(
            output_dir=temp_dir,
            shuffled_output_dir=temp_dir,
            max_output_chunk_rows=max_output_chunk_rows,
            max_output_chunk_uncompressed_bytes=max_output_chunk_uncompressed_bytes,
            output_batch_size=5,
            output_format="parquet",
            config=config,
        )

        # Generate data
        texts = list(ds.get_texts())

        shuffled_docs_count, saved_chunks = ds.save_texts_to_parquet(texts, apply_filter=False)

        assert shuffled_docs_count == dataset_size

        use_shuffled_output = False
        split_offset_limit = get_splits_as_offsets_and_limits(ds, use_shuffled_output=use_shuffled_output)

        full_texts = list(
            generate_texts_from_dataset(
                ds,
                split=DatasetSplit.FULL,
                split_offset_limit=split_offset_limit,
                use_shuffled_output=use_shuffled_output,
            )
        )
        print("full_texts", len(full_texts))

        train_texts = list(
            generate_texts_from_dataset(
                ds,
                split=DatasetSplit.TRAIN,
                split_offset_limit=split_offset_limit,
                use_shuffled_output=use_shuffled_output,
            )
        )
        print("train_texts", len(train_texts), expected_train_size)

        tokenizer_train_texts = list(
            generate_texts_from_dataset(
                ds,
                split=DatasetSplit.TOKENIZER_TRAIN,
                split_offset_limit=split_offset_limit,
                use_shuffled_output=use_shuffled_output,
            )
        )
        print("tokenizer_train_texts", len(tokenizer_train_texts), expected_tokenizer_train_size)

        assert len(tokenizer_train_texts) == expected_tokenizer_train_size

        val_texts = list(
            generate_texts_from_dataset(
                ds,
                split=DatasetSplit.VALIDATION,
                split_offset_limit=split_offset_limit,
                use_shuffled_output=use_shuffled_output,
            )
        )
        print("val_texts", len(val_texts), expected_validation_size)

        assert full_texts == texts

        assert len(full_texts) == dataset_size
        assert len(train_texts) + len(val_texts) == dataset_size

        # no overlap train-val
        assert len(set(val_texts) & set(train_texts)) == 0

        assert len(train_texts) == expected_train_size
        assert len(val_texts) == expected_validation_size

        assert len(set(tokenizer_train_texts) & set(train_texts)) == expected_tokenizer_train_size


def test_train_validation_split_1000_02_01():
    _test_train_validation_split(
        dataset_size=1000, validation_ratio=0.2, tokenizer_train_ratio=0.1, max_output_chunk_rows=21
    )


def test_train_validation_split_512_033_033():
    _test_train_validation_split(
        dataset_size=512,
        validation_ratio=0.33,
        tokenizer_train_ratio=0.33,
        max_output_chunk_uncompressed_bytes=1024 * 2.5,
    )


def test_train_validation_split_100_025_033():
    _test_train_validation_split(
        dataset_size=100, validation_ratio=0.25, tokenizer_train_ratio=0.33, max_output_chunk_rows=10
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    test_train_validation_split_1000_02_01()
    test_train_validation_split_512_033_033()
    test_train_validation_split_100_025_033()
