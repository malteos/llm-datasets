from lm_datasets.datasets.base import Availability
from lm_datasets.datasets.hf_dataset import HFDataset


class NorwegianCCBaseDataset(HFDataset):
    DATASET_ID = "norwegian_cc"
    SOURCE_ID = "norwegian_cc"

    TITLE = "Norwegian Colossal Corpus"
    HOMEPAGE = "https://huggingface.co/datasets/NbAiLab/NCC"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    DESCRIPTION = (
        "The Norwegian Colossal Corpus is a collection of multiple smaller Norwegian corpuses "
        "suitable for training large language models. We have done extensive cleaning on the datasets, "
        "and have made them available in a common format. The total size of the NCC is currently 45GB. "
        "Documents: 20,830,348; Words/document: 331"
    )

    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = "mixed"

    HF_DATASET_ID = "NbAiLab/NCC"
    HF_DATASET_SPLIT = "train"

    streaming = False

    excluded_doc_types = {
        # exclude all Wikimedia sources to avoid duplicated content with the original Wikimedia dataset
        "wikipedia",
    }

    text_column_name = "text"
    remove_columns = ["id", "publish_year", "lang_fasttext_conf"]

    def get_filter_func(self):
        languages_set = set(self.LANGUAGES)

        def filter_func(example):
            return example["doc_type"] not in self.excluded_doc_types and example["lang_fasttext"] in languages_set

        return filter_func


class NorwegianCCNODataset(NorwegianCCBaseDataset):
    DATASET_ID = "norwegian_cc_no"
    LANGUAGES = ["no"]

    # total: BYTES = 45 * GB
    TOKENS = 5_050_752_505  # total: 6_905_570_165


class NorwegianCCNNDataset(NorwegianCCBaseDataset):
    DATASET_ID = "norwegian_cc_no"
    LANGUAGES = ["nn"]

    # total: BYTES = 45 * GB
    TOKENS = 299_753_996  # total: 6_905_570_165
