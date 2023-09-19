"""
https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1448/kas.json.tar.gz

academic_slovene_kas

"""

import json
import logging

from lm_datasets.datasets.base import BaseDataset, Availability

from tqdm.auto import tqdm
from smart_open import open


logger = logging.getLogger(__name__)


class AcademicSloveneKASDataset(BaseDataset):
    """
    The KAS corpus of Slovene academic writing consists of almost 65,000 BSc/BA, 16,000 MSc/MA and 1,600 PhD theses
    (82 thousand texts, 5 million pages or 1,5 billion tokens) written 2000 - 2018 and gathered from the digital
    libraries of Slovene higher education institutions via the Slovene Open Science portal (http://openscience.si/).
    """

    DATASET_ID = "academic_slovene_kas"
    TITLE = "Corpus of academic Slovene KAS 2.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1448"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["sl"]

    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/sl/academic_slovene_kas"]

    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1448/kas.json.tar.gz"]

    TOKENS = 1_496_079_001

    def decompress(self):
        # tar gz
        pass

    def get_texts(self):
        # TODO read directly from tar gz

        # Read from many JSON files
        for fp in tqdm(self.get_dataset_file_paths(needed_suffix=".json", subdirectories=True), desc="Reading files"):
            # logger.info(f"Reading from {fp}")

            with open(fp) as f:
                doc = json.load(f)
                text = ""
                for line, paragraphs in doc.items():
                    for p in paragraphs:
                        text += p.replace("\n", " ").strip() + "\n\n"

                    # text += "\n"
                # print(text.strip())
                # print("#######")

                yield text.strip()
