from lm_datasets.datasets.base import Availability, GB, License
from lm_datasets.datasets.hf_dataset import HFDataset


class EdgarCorpus(HFDataset):

    DATASET_ID = "edgarcorpus"

    TITLE = "EdgarCorpus"
    DESCRIPTION = (
        """This dataset card is based on the paper EDGAR-CORPUS: Billions of
         Tokens Make The World Go Round authored by Lefteris Loukas et.al, as
         published in the ECONLP 2021 workshop."""
    )

    HOMEPAGE = "https://huggingface.co/datasets/eloukas/edgar-corpus"
    LICENSE = License(
        name="Apache License Version 2.0",
        url="http://www.apache.org/licenses/LICENSE-2.0"
    )
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["en"]

    DESCRIPTION = """The dataset contains annual filings (10K) of all publicly
    traded firms from 1993-2020. The table data is stripped but all text is retained.
    This dataset allows easy access to the EDGAR-CORPUS dataset based on the paper
    EDGAR-CORPUS: Billions of Tokens Make The World Go Round."""

    HF_DATASET_ID = "eloukas/edgar-corpus"
    HF_DATASET_CONFIGS = ["full"]
    HF_DATASET_SPLIT = "train"

    BYTES = 23 * GB

    remove_columns = ["filename","cik","year"]
    keep_columns = True

    def get_text_from_item(self,item) -> str:
        """
        Subscribing the original method since this dataset
        has multiple columns.

        Iterates over the row columns and concatenates the columns content
        item: <dict:{column_name: content}>
        """
        txt = ""
        for column in item.keys():
            txt+=item[column]
        return txt