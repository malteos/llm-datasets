import json
import logging
from typing import Iterable
from smart_open import open

from lm_datasets.datasets.base import GB, BaseDataset, Availability, License

logger = logging.getLogger(__name__)


class DANewsroomDataset(BaseDataset):
    DATASET_ID = "danewsroom"
    TITLE = "DaNewsroom: A Large-scale Danish Summarisation Dataset"
    HOMEPAGE = "https://github.com/danielvarab/da-newsroom"  # "https://aclanthology.org/2020.lrec-1.831/"
    CITATION = """@inproceedings{varab-schluter-2020-danewsroom,
        title = "{D}a{N}ewsroom: A Large-scale {D}anish Summarisation Dataset",
        author = "Varab, Daniel  and
        Schluter, Natalie",
        booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference",
        month = may,
        year = "2020",
        address = "Marseille, France",
        publisher = "European Language Resources Association",
        url = "https://www.aclweb.org/anthology/2020.lrec-1.831",
        pages = "6731--6739",
        abstract = "Dataset development for automatic summarisation systems is notoriously English-oriented. In this paper we present the first large-scale non-English language dataset specifically curated for automatic summarisation. The document-summary pairs are news articles and manually written summaries in the Danish language. There has previously been no work done to establish a Danish summarisation dataset, nor any published work on the automatic summarisation of Danish. We provide therefore the first automatic summarisation dataset for the Danish language (large-scale or otherwise). To support the comparison of future automatic summarisation systems for Danish, we include system performance on this dataset of strong well-established unsupervised baseline systems, together with an oracle extractive summariser, which is the first account of automatic summarisation system performance for Danish. Finally, we make all code for automatically acquiring the data freely available and make explicit how this technology can easily be adapted in order to acquire automatic summarisation datasets for further languages.",
        language = "English",
        ISBN = "979-10-95546-34-4",
        }
        """  # noqa
    LANGUAGES = ["da"]
    LICENSE = License("research-only (unknown license)", commercial_use=False, research_use=True)
    AVAILIBILITY = Availability.ON_REQUEST
    BYTES = 1.5 * GB

    def download(self):
        """
        gdown "https://drive.google.com/u/0/uc?id=1u22Hcs__CUu_GAzY4HQ275gfHvLGOckv&export=download"
        """
        pass

    def get_texts(self) -> Iterable[str]:
        """
        Extracts the text from each JSONL file.
        """

        with open(self.get_dataset_file_paths(single_file=True, needed_suffix=".jsonl.gz")) as f:
            for line in f:
                doc = json.loads(line)
                text = doc["title"] + self.title_delimiter + doc["text"]

                yield text
