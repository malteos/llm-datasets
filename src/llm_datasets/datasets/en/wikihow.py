from llm_datasets.datasets.base import Availability, MB, License
from llm_datasets.datasets.hf_dataset import HFDataset


class WikihowDataset(HFDataset):
    DATASET_ID = "wikihow"

    TITLE = "WikiHow"
    DESCRIPTION = (
        "WikiHow is a new large-scale dataset using the online WikiHow (http://www.wikihow.com/) knowledge base."
    )
    CITATION = r"""@misc{koupaee2018wikihow,
        title={WikiHow: A Large Scale Text Summarization Dataset},
        author={Mahnaz Koupaee and William Yang Wang},
        year={2018},
        eprint={1810.09305},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }"""
    HOMEPAGE = "https://github.com/mahnazkoupaee/WikiHow-Dataset"
    LICENSE = License(
        name="CC BY-NC-SA 3.0",
        url="https://creativecommons.org/licenses/by-nc-sa/3.0/",
        commercial_use=False,
        sharealike=True,
        research_use=True,
    )
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["en"]

    DESCRIPTION = """WikiHow is a new large-scale dataset using the online WikiHow"""

    HF_DATASET_ID = "wikihow"
    HF_DATASET_CONFIGS = ["all"]
    HF_DATASET_SPLIT = "train"

    BYTES = 5.21 * MB

    @property
    def HF_DATA_DIR(self):
        return self.get_local_dataset_dir()

    text_column_name = "text"
    # remove_columns = ["doc_id", "LICENSE", "uri", "date_built"]
