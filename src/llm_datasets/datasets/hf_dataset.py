import logging
from datasets import load_dataset, DatasetDict

from typing import Dict, Iterable, List, Optional

from llm_datasets.datasets.base import BaseDocumentDataset
from datatrove.data import Document


logger = logging.getLogger(__name__)


class HFDataset(BaseDocumentDataset):
    HF_DATASET_ID: str = None
    HF_DATASET_SPLIT: Optional[str] = None
    HF_DATASET_CONFIGS: Optional[List[str]] = None
    HF_DATA_DIR = None
    HF_KWARGS = None
    HF_REVISION: Optional[str] = None

    config_to_dataset: Optional[Dict] = None
    id_column_name = None
    text_column_name = "text"
    title_column_name = None
    metadata_column_names = None
    remove_columns = None
    streaming = False
    keep_columns = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        if self.HF_DATASET_ID is None:
            raise ValueError("HF_DATASET_ID is not set")

    def get_hf_configs(self):
        if self.HF_DATASET_CONFIGS:
            return self.HF_DATASET_CONFIGS
        else:
            # if no config is used
            return [None]

    def download(self):
        self.config_to_dataset = {}

        for hf_config in self.get_hf_configs():
            logger.info(f"Downloading for {hf_config=}")

            if self.HF_KWARGS:
                # use additional kwargs as defined by dataset class
                ds = load_dataset(
                    self.HF_DATASET_ID,
                    hf_config,
                    split=self.HF_DATASET_SPLIT,
                    data_dir=self.HF_DATA_DIR,
                    streaming=self.streaming,
                    use_auth_token=self.get_hf_auth_token(),
                    keep_in_memory=False,
                    revision=self.HF_REVISION,
                    **self.HF_KWARGS,
                )
            else:
                ds = load_dataset(
                    self.HF_DATASET_ID,
                    hf_config,
                    split=self.HF_DATASET_SPLIT,
                    data_dir=self.HF_DATA_DIR,
                    streaming=self.streaming,
                    use_auth_token=self.get_hf_auth_token(),
                    keep_in_memory=False,
                    revision=self.HF_REVISION,
                )

            # check dataset split
            if isinstance(ds, DatasetDict) and not self.HF_DATASET_SPLIT:
                logger.warning(f"HF returned DatasetDict but split not set: {DatasetDict}")

            if self.limit > 0:
                if self.streaming:
                    logger.warning(f"Limit requested ({self.limit=}) but streaming is enabled!")
                else:
                    logger.warning(f"Limiting dataset to: {self.limit}")
                    ds = ds.select(range(self.limit))

            if self.remove_columns is not None:
                logger.info(f"Removing columns (at download): {self.remove_columns}")

                ds = ds.remove_columns(self.remove_columns)

            filter_func = self.get_filter_func()
            if filter_func:
                logger.info(f"Dataset size before filter: {len(ds):,}")

                ds = ds.filter(filter_func, num_proc=self.workers)

                logger.info(f"Dataset size after filter: {len(ds):,}")

            self.config_to_dataset[hf_config] = ds

    def get_filter_func(self):
        return None

    def get_document_from_item(self, item, index: Optional[int] = None) -> Document:
        return Document(
            text=item[self.text_column_name],
            id=item[self.id_column_name] if self.id_column_name else index,
            metadata={col: item[col] for col in self.metadata_column_names} if self.metadata_column_names else {},
        )

    def prepend_title(self, example):
        example[self.text_column_name] = (
            example[self.title_column_name] + self.title_delimiter + example[self.text_column_name]
        )

        return example

    def get_documents(self) -> Iterable[Document]:
        self.download()
        doc_idx = 0
        # drop all non-text columns
        for ds_idx, config in enumerate(self.config_to_dataset):
            # remove non-text and non-title columns
            if not self.keep_columns:
                columns_to_remove = set(self.config_to_dataset[config].column_names) - {self.text_column_name}

                if self.title_column_name:
                    columns_to_remove = columns_to_remove - {self.title_column_name}

                logger.info("Removing columns (get texts): %s", columns_to_remove)

                self.config_to_dataset[config] = self.config_to_dataset[config].remove_columns(columns_to_remove)

            if self.title_column_name:
                logger.info(f"Prepending title to text column ({self.title_column_name=})")

                self.config_to_dataset[config] = self.config_to_dataset[config].map(self.prepend_title)

                # remove title column
                self.config_to_dataset[config] = self.config_to_dataset[config].remove_columns([self.title_column_name])

            ds_iterator = iter(self.config_to_dataset[config])

            for item in ds_iterator:
                if hasattr(self, "get_documents_from_item"):
                    # multiple documents from a single item
                    yield from self.get_documents_from_item(item)
                else:
                    yield self.get_document_from_item(item, doc_idx)
                    doc_idx += 1
