import logging
from typing import Iterable
from lm_datasets.datasets.base import BILLION, Availability, BaseDataset
from smart_open import open

logger = logging.getLogger(__name__)


class DEWacDataset(BaseDataset):
    """
    See also: ITWacDataset
    """

    DATASET_ID = "dewac"
    TITLE = "DeWaC"
    DESCRIPTION = (
        "DeWaC is a 1.7 billion word corpus constructed from the Web limiting the crawl to the .de domain and using"
        " medium-frequency words from the SudDeutsche Zeitung corpus and basic German vocabulary lists as seeds."
    )
    LANGUAGES = ["de"]
    AVAILIBILITY = Availability.ON_REQUEST
    HOMEPAGE = "https://docs.sslmit.unibo.it/doku.php?id=corpora:dewac"

    TOKENS = 1.7 * BILLION

    def decompress(self):
        #  7z x dewac_preproc.7z
        pass

    def get_texts(self) -> Iterable[str]:
        file_path = self.get_dataset_file_paths(single_file=True, needed_suffix=".txt")

        # chardet.detect(line)

        # New documents are separated by "CURRENT URL <url>"
        doc_text = ""
        with open(file_path, "r", encoding="ISO-8859-1") as f:
            for i, line in enumerate(f):
                if line.startswith("CURRENT URL "):
                    if doc_text:
                        # print(doc_text)
                        yield doc_text

                    doc_text = ""
                else:
                    doc_text += line.strip()

                pass

            if doc_text:
                # last document
                yield doc_text
