import logging

from typing import Dict, List, Optional

from lm_datasets.datasets.base import Availability, License
from lm_datasets.datasets.hf_dataset import HFDataset


from datasets import load_dataset, DatasetDict


logger = logging.getLogger(__name__)


class PleiasBooksBase(HFDataset):
    HF_DATASET_ID = "PleIAs/French-PD-Books"
    HF_DATASET_SPLIT = "train"
    HF_DATASET_CONFIGS: Optional[List[str]] = None

    text_column_name = "complete_text"
    title_column_name = "title"
    remove_columns = ["file_id", "ocr", "author", "page_count",
                      "word_count", "character_count"]

    def process(self, example):
        date = list(example["date"])[0:3]
        date.append("0")
        example['date'] = int("".join(date))
        return example

    def download(self):
        self.config_to_dataset = {}
        ds = load_dataset(
            self.HF_DATASET_ID,
            split=self.HF_DATASET_SPLIT,
            data_dir=self.HF_DATA_DIR,
            streaming=self.streaming,
            use_auth_token=self.get_hf_auth_token(),
            keep_in_memory=False,
            revision=self.HF_REVISION,
        )
        # ds = ds.take(10000)
        ds = ds.map(self.process, batched=False)
        self.config_to_dataset[None] = ds


def get_decade_pleias_books(decade):
    class PleiasBooks(PleiasBooksBase):
        DATASET_ID = f"PleIAs_French_PD_Books{str(decade)}"
        DECADE = decade

        def get_texts(self):
            self.download()

            for item in iter(self.config_to_dataset[None]):
                if item['date'] == self.DECADE:
                    yield item[self.text_column_name]

    return PleiasBooks


def get_pleias_books_auto_classes():
    """
    Auto generate classes for each decade.
    """
    decades = [1920, 1540, 1670, 1800, 1930, 1550, 1680, 1810, 1940, 1560, 1690, 1820, 1950, 1570, 1700, 1830, 1960, 1580, 1710, 1840, 1970, 1590, 1720, 1850, 1980,
               1600, 1730, 1860, 1990, 1610, 1740, 1870, 2000, 1620, 1750, 1880, 2010, 1630, 1760, 1890, 1380, 1510, 1640, 1770, 1900, 1650, 1780, 1910, 1530, 1660, 1790]

    return [get_decade_pleias_books(decade) for decade in decades]
