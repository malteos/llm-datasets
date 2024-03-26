import logging

from typing import Dict, List, Optional

from llm_datasets.datasets.base import Availability, License
from llm_datasets.datasets.hf_dataset import HFDataset


from datasets import load_dataset, DatasetDict


logger = logging.getLogger(__name__)


class PleiasNewsBase(HFDataset):
    HF_DATASET_ID = "PleIAs/French-PD-Newspapers"
    HF_DATASET_SPLIT = "train"
    HF_DATASET_CONFIGS: Optional[List[str]] = None
    streaming = True
    text_column_name = "complete_text"
    title_column_name = "title"
    remove_columns = ["file_id", "ocr", "author", "page_count", "word_count", "character_count"]

    def process(self, example):
        """
        Function for batch processing of the date column on the PLeias News Dataset.
        Groups date by decade. If no date is provided return the row as is.
        """

        try:
            date = list(example["date"])[0:3]
            date.append("0")
            example["date"] = int("".join(date))
            return example
        except ValueError:
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

        ds = ds.map(self.process, batched=False)
        self.config_to_dataset[None] = ds

    def get_texts(self):
        self.download()

        for item in iter(self.config_to_dataset[None]):
            if item["date"] == self.DECADE:
                yield item[self.text_column_name]


def get_decade_pleias_news(decade):
    class PleiasNews(PleiasNewsBase):
        DATASET_ID = f"PleIAs_French_PD_Newspapers_{str(decade)}"
        DECADE = decade

        def get_texts(self):
            self.download()

            for item in iter(self.config_to_dataset[None]):
                if item["date"] == self.DECADE:
                    yield item[self.text_column_name]

    return PleiasNews


def get_pleias_news_auto_classes():
    """
    Auto generate classes for each decade.
    """
    decades = [
        1920,
        1670,
        1800,
        1930,
        1680,
        1810,
        1940,
        1690,
        1820,
        1950,
        1700,
        1830,
        1960,
        1710,
        1840,
        1970,
        1720,
        1850,
        1980,
        None,
        1860,
        1990,
        1740,
        1870,
        2000,
        1750,
        1880,
        2010,
        1630,
        1760,
        1890,
        1640,
        1770,
        1900,
        1650,
        1780,
        1910,
        1790,
    ]

    return [get_decade_pleias_news(decade) for decade in decades]
