import json
import logging
from typing import Iterable
from smart_open import open

from lm_datasets.datasets.base import GB, BaseDataset, Availability

logger = logging.getLogger(__name__)


class DANewsroomDataset(BaseDataset):
    DATASET_ID = "danewsroom"
    TITLE = "DaNewsroom: A Large-scale Danish Summarisation Dataset"
    HOMEPAGE = "https://github.com/danielvarab/da-newsroom"  # "https://aclanthology.org/2020.lrec-1.831/"

    LANGUAGES = ["da"]
    # LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/danewsroom"]

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
