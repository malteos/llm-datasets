import logging

from lm_datasets.datasets.base import BaseDataset, Availability, MB
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class BGNCNewsCorpusDataset(BaseDataset):
    """
    Part of Bulgarian National Corpus
    """

    DATASET_ID = "bgnc_news_corpus"
    TITLE = "News Corpus (bg)"
    HOMEPAGE = "https://dcl.bas.bg/BulNC-registration/?lang=EN#feeds/page/2"

    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["bg"]

    DOWNLOAD_URLS = ["https://dcl.bas.bg/BulNC-registration/dl.php?dl=feeds/JOURNALISM.BG.zip"]

    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/bg/BgNC/news_corpus"]

    BYTES = 57 * MB

    def download(self):
        """
        DOWNLOAD
        -----------

        Instruction

        - Downloaded locally by clicking on the download link in the browser:

        https://dcl.bas.bg/BulNC-registration/dl.php?dl=feeds/JOURNALISM.BG.zip

        - Copy local file to server:

        scp /Local/Path/to/JOURNALISM.BG.zip username@clustername:/data/datasets/ele/bg/BgNC/news_corpus

        - Extract files:

        unzip JOURNALISM.BG.zip

        """
        pass

    def decompress(self):
        # unzip JOURNALISM.BG.zip
        pass

    def get_texts(self):
        files_path = self.get_dataset_file_paths(subdirectories=True, needed_suffix=".txt")

        logger.info(f"Found {len(files_path):,} files")

        for input_file in tqdm(files_path, desc="Reading files"):
            # skip if is metadata
            if "METADATA" in input_file:
                logger.warning(f"Skip {input_file}")
                continue

            # each file is one documentt
            with open(input_file, "r", encoding="utf-8") as inp:
                text = inp.read()
                yield text.strip()
