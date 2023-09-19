import logging

from lm_datasets.datasets.base import BaseDataset, Availability, MB
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class BGNCAdminEURDataset(BaseDataset):
    """
    Part of Bulgarian National Corpus

    TODO overlap with eurlex_bg?
    """

    DATASET_ID = "bgnc_admin_eur"
    TITLE = "ADMIN_EUR Corpus of EU legislation (bg)"
    HOMEPAGE = "https://eur-lex.europa.eu/homepage.html"

    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["bg"]

    DOWNLOAD_URLS = ["https://dcl.bas.bg/BulNC-registration/dl.php?dl=feeds/ADMIN_EUR.BG.zip"]

    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/bg/BgNC/admin_eur/ADMIN_EUR.BG"]

    BYTES = 257 * MB

    def download(self):
        """
        DOWNLOAD
        -----------

        Instruction

        - Downloaded locally by clicking on the download link in the browser:

        https://dcl.bas.bg/BulNC-registration/dl.php?dl=feeds/ADMIN_EUR.BG.zip

        - Copy local file to server:

        scp /Local/Path/to/ADMIN_EUR.BG.zip username@clustername:/data/datasets/ele/bg/BgNC/admin_eur

        - Extract files:

        unzip ADMIN_EUR.BG.zip

        """
        pass

    def decompress(self):
        # unzip ADMIN_EUR.BG.zip
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
            with open(input_file, "r") as inp:
                text = inp.read()
                yield text.strip()
