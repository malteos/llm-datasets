from lm_datasets.datasets.base import BaseDataset
from lm_datasets.datasets.dataset_registry import get_registered_dataset_classes
from lm_datasets.utils.config import Config


class DummyBaseDataset(BaseDataset):
    """
    A dummy dataset for debugging and unit tests.
    """

    DATASET_ID = None
    SOURCE_ID = "dummy"
    SIZE = None
    PREFIX = ""

    def get_output_rows_count(self, shuffled: bool = False) -> int:
        return self.SIZE

    def get_texts(self):
        """
        Each dataset sample is string reflected the sample index, optinally with a datatset prefix.
        """
        for i in range(self.SIZE):
            text = self.PREFIX + str(i)

            yield text

    def get_output_extension(self, with_dot: bool = True, shuffled: bool = False) -> str:
        # use the same file names for shuffled and unshuffled
        return super().get_output_extension(with_dot=with_dot, shuffled=False)


def get_dummy_dataset_cls(ds_size: int, text_prefix=""):
    class DummyDataset(DummyBaseDataset):
        DATASET_ID = "dummy_%s%s" % (text_prefix, ds_size)
        SIZE = ds_size
        PREFIX = text_prefix

    return DummyDataset


def save_texts_for_temp_datasets(
    config: Config,
    temp_dir,
    max_output_chunk_rows=None,
    max_output_chunk_uncompressed_bytes=None,
    output_batch_size=5,
    output_format="parquet",
):
    list_of_texts = []
    list_of_dataset_ids = []

    for ds_cls in get_registered_dataset_classes(
        extra_dataset_registries=config.extra_dataset_registries,
        extra_dataset_classes=config.extra_dataset_classes,
        use_default_registry=config.use_default_dataset_registry,
    ):
        ds: BaseDataset = ds_cls(
            output_dir=temp_dir,
            shuffled_output_dir=temp_dir,
            max_output_chunk_rows=max_output_chunk_rows,
            max_output_chunk_uncompressed_bytes=max_output_chunk_uncompressed_bytes,
            output_batch_size=output_batch_size,
            output_format=output_format,
            config=config,
        )

        # Generate data
        texts = list(ds.get_texts())

        saved_docs_count, saved_chunks = ds.save_texts_to_parquet(texts, apply_filter=False)

        list_of_texts.append(texts)
        list_of_dataset_ids.append(ds.DATASET_ID)

    return list_of_texts, list_of_dataset_ids
