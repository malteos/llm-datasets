from llm_datasets.datasets.base import Availability, GB, License
from llm_datasets.datasets.hf_dataset import HFDataset


class EdgarCorpus(HFDataset):
    DATASET_ID = "edgarcorpus"

    TITLE = "EdgarCorpus"
    HOMEPAGE = "https://huggingface.co/datasets/eloukas/edgar-corpus"
    LICENSE = License(name="Apache License Version 2.0", url="http://www.apache.org/licenses/LICENSE-2.0")
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["en"]

    DESCRIPTION = """The dataset contains annual filings (10K) of all publicly
    traded firms from 1993-2020. The table data is stripped but all text is retained.
    This dataset allows easy access to the EDGAR-CORPUS dataset based on the paper
    EDGAR-CORPUS: Billions of Tokens Make The World Go Round."""
    CITATION = r"""@inproceedings{loukas-etal-2021-edgar,
        title = "{EDGAR}-{CORPUS}: Billions of Tokens Make The World Go Round",
        author = "Loukas, Lefteris  and
        Fergadiotis, Manos  and
        Androutsopoulos, Ion  and
        Malakasiotis, Prodromos",
        booktitle = "Proceedings of the Third Workshop on Economics and Natural Language Processing",
        month = nov,
        year = "2021",
        address = "Punta Cana, Dominican Republic",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.econlp-1.2",
        pages = "13--18",
    }
    """
    HF_DATASET_ID = "eloukas/edgar-corpus"
    HF_DATASET_CONFIGS = ["full"]
    HF_DATASET_SPLIT = "train"

    BYTES = 23 * GB

    remove_columns = ["filename", "cik", "year"]
    keep_columns = True

    def get_text_from_item(self, item) -> str:
        """
        Subscribing the original method since this dataset
        has multiple columns.

        Iterates over the row columns and concatenates the columns content
        item: <dict:{column_name: content}>
        """
        txt = ""
        txt_colums = [
            "section_1",
            "section_1A",
            "section_1B",
            "section_2",
            "section_3",
            "section_4",
            "section_5",
            "section_6",
            "section_7",
            "section_7A",
            "section_8",
            "section_9",
            "section_9A",
            "section_9B",
            "section_10",
            "section_11",
            "section_12",
            "section_13",
            "section_14",
            "section_15",
        ]

        for column in txt_colums:
            txt += item[column]
        return txt
