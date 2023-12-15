import tempfile
from .dummy_datasets import get_dummy_dataset_cls
from lm_datasets.datasets.base import BaseDataset
from lm_datasets.utils.config import Config


def _test_generate_texts_from_output(
    dataset_size=100,
    offset=0,
    limit=0,
    seed=0,
    shuffle_output_file_paths=False,
    compare_text_indicies=None,
    max_output_chunk_rows=12,
    max_output_chunk_uncompressed_bytes=None,
):
    config = Config()
    config.seed = seed

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

        saved_docs_count, saved_chunks = ds.save_texts_to_parquet(texts, apply_filter=False)

        assert saved_docs_count == dataset_size

        # read again
        output_texts = list(
            ds.generate_texts_from_output(
                shuffled=False,
                offset=offset,
                limit=limit,
                shuffle_output_file_paths=shuffle_output_file_paths,
                reader_implementation="pyarrow",
            )
        )

        if limit > 0:
            expected_texts_count = limit - offset
        elif offset > 0:
            expected_texts_count = len(texts) - offset
        else:
            expected_texts_count = len(texts)

        assert expected_texts_count == len(output_texts)

        if compare_text_indicies:
            for original_text_idx, output_text_idx in compare_text_indicies:
                assert texts[original_text_idx] == output_texts[output_text_idx]


def test_1():
    _test_generate_texts_from_output(dataset_size=100, offset=0, limit=0, compare_text_indicies=[(0, 0), (-1, -1)])


def test_2():
    _test_generate_texts_from_output(dataset_size=100, offset=0, limit=19, compare_text_indicies=[(0, 0), (19 - 1, -1)])


def test_3():
    _test_generate_texts_from_output(
        dataset_size=100,
        offset=0,
        limit=0,
        shuffle_output_file_paths=True,
        seed=0,
        compare_text_indicies=[(60, 0), (-1, -1)],
    )


def test_4():
    _test_generate_texts_from_output(
        dataset_size=100,
        offset=10,
        limit=0,
        shuffle_output_file_paths=True,
        seed=1,
        compare_text_indicies=[(55, 0), (29, -1)],
    )


def test_5():
    _test_generate_texts_from_output(
        dataset_size=1000,
        offset=10,
        limit=200,
        shuffle_output_file_paths=True,
        compare_text_indicies=[(95, 0), (734, -1)],
        max_output_chunk_uncompressed_bytes=512,
        max_output_chunk_rows=None,
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

    test_1()
    test_2()
    test_3()
    test_4()
    test_5()  # TODO

    print("done")
