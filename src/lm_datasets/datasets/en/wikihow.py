from lm_datasets.datasets.base import Availability, MB
from lm_datasets.datasets.hf_dataset import HFDataset


class WikihowDataset(HFDataset):
    DATASET_ID = "wikihow"

    TITLE = "WikiHow"
    DESCRIPTION = (
        "WikiHow is a new large-scale dataset using the online WikiHow (http://www.wikihow.com/) knowledge base."
    )

    HOMEPAGE = "https://github.com/mahnazkoupaee/WikiHow-Dataset"

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
