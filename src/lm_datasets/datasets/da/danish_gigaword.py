from lm_datasets.datasets.base import BILLION, Availability
from lm_datasets.datasets.hf_dataset import HFDataset


class DanishGigawordDataset(HFDataset):
    DATASET_ID = "danish_gigaword"

    TITLE = "Danish GiagaWord"
    HOMEPAGE = "https://sprogteknologi.dk/dataset/danish-gigaword"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["da"]

    DESCRIPTION = (
        "A billion-word corpus of Danish text. Split into many sections, and covering many dimensions ",
        "of variation (spoken/written, formal/informal, modern/old, rigsdansk/dialect, and so on).",
        "",
        "The license is CC-BY 4.0, Creative Commons with Attribution. Owners: ITU; Leon Derczynski, Manuel R. Ciosici",
    )
    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = "public domain"

    # Size: 1 billion words
    TOKENS = 1 * BILLION

    HF_DATASET_ID = "DDSC/partial-danish-gigaword-no-twitter"
    HF_DATASET_SPLIT = "train"

    excluded_sources = {
        # exclude all Wikimedia sources to avoid duplicated content with the original Wikimedia dataset
        "wiki",
        "wikibooks",
        "wikisource",
    }

    text_column_name = "text"
    remove_columns = ["doc_id", "LICENSE", "uri", "date_built"]

    def get_filter_func(self):
        def filter_func(example):
            return example["source"] not in self.excluded_sources

        return filter_func
