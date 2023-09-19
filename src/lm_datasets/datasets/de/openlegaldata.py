from lm_datasets.datasets.base import Availability, Genre
from lm_datasets.datasets.jsonl_dataset import JSONLDataset


class OpenLegalDataDataset(JSONLDataset):
    DATASET_ID = "openlegaldata"
    TITLE = "Open Legal Data - German court decisions and laws"
    HOMEPAGE = "https://openlegaldata.io/"
    VERSION = "oldp_cases_v1_v2_fix"  # "oldp-cases_merged_v1_v2"  # dump date: 2022-10-18

    LANGUAGES = ["de"]
    GENRES = [Genre.LEGAL]
    AVAILIBILITY = Availability.ON_REQUEST

    TOKENS = 9_700_000_000

    raw_jsonl_paths = [
        "v1.jsonl.gz",
        "v2.jsonl.gz",
        # "merged_v1_v2.jsonl.gz",
        # "merged_v1_v2.jsonl",
    ]

    doc_ids = set()

    def get_text_from_item(self, item):
        # filter by doc ID
        if item["doc_id"] in self.doc_ids:
            return None
        else:
            self.doc_ids.add(item["doc_id"])
            return item[self.raw_jsonl_text_field]
