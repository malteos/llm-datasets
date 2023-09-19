import logging
from typing import Iterable
from lm_datasets.datasets.base import BILLION, Availability, BaseDataset
from smart_open import open

logger = logging.getLogger(__name__)


class ITWacDataset(BaseDataset):
    """
    See also: DeWacDataset
    """

    DATASET_ID = "itwac"
    TITLE = "ITWaC"
    DESCRIPTION = (
        "itWaC: a 2 billion word corpus constructed from the Web limiting the crawl to the .it domain and using"
        " medium-frequency words from the Repubblica corpus and basic Italian vocabulary lists as seeds. "
    )
    LANGUAGES = ["it"]
    AVAILIBILITY = Availability.ON_REQUEST
    HOMEPAGE = "https://docs.sslmit.unibo.it/doku.php?id=corpora:itwac"

    TOKENS = 2 * BILLION

    def decompress(self):
        #  7z x itwac_preproc.7z
        pass

    def get_texts(self) -> Iterable[str]:
        file_path = self.get_dataset_file_paths(single_file=True, needed_suffix=".filtered.pre.pos.corpus")

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
