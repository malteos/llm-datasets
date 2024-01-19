from lm_datasets.datasets.base import Availability, GB, License
from lm_datasets.datasets.hf_dataset import HFDataset

from lm_datasets.utils.config import Config

conf = Config()

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

    DESCRIPTION = """EDGAR-CORPUS, a novel corpus comprising annual reports from
     all the publicly traded companies in the US spanning a period of more than 25 years."""

    HF_DATASET_ID = "eloukas/edgar-corpus"
    HF_DATASET_CONFIGS = ["year_2000"]
    # HF_DATASET_SPLIT = ["train","validation","test"]
    HF_DATASET_SPLIT = "train[:20]"

    BYTES = 23 * GB

    # remove_columns = ["filename","cik","year"]
    # text_column_name = ["section_1","section_1A", "section_1B","section_2",
    # "section_3","section_4","section_5","section_6","section_7","section_7A"]
    # text_column_name = "section_1"

    # def get_texts():