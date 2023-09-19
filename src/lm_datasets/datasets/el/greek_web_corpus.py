from typing import Iterable
from lm_datasets.datasets.base import GB, BaseDataset, Availability, QualityWarning
from smart_open import open
import logging

logger = logging.getLogger(__name__)


class GreekWebCorpus(BaseDataset):
    DATASET_ID = "greek_web_corpus"
    TITLE = "Greek Web Corpus"
    AVAILIBILITY = Availability.ON_REQUEST
    HOMEPAGE = "http://nlp.polytechnique.fr/resources-greek"
    CITATION = """@article{Outsios2018,
        title = {Word Embeddings from Large-Scale Greek Web content},
        author = {Outsios, Stamatis and Skianis, Konstantinos and Meladianos, Polykarpos and Xypolopoulos, Christos and Vazirgiannis, Michalis},
         year={2018},
         journal = {arXiv preprint arXiv:1810.06694},
    }"""  # noqa
    LANGUAGES = ["el"]
    USED_BY = [
        "https://arxiv.org/abs/2304.00869",  # GreekBART
    ]

    BYTES = 10 * GB

    QUALITY_WARNINGS = [QualityWarning.SHORT_TEXT, QualityWarning.BAD_PUNCTUATION]

    def get_texts(self) -> Iterable[str]:
        # txt.gz
        fp = self.get_dataset_file_paths(needed_suffix=".txt.gz", single_file=True)
        logger.info(f"Reading from {fp}")

        with open(fp) as f:
            for i, line in enumerate(f):
                text = line.strip()  # TODO each line a new doc?

                if text:
                    yield text
