from pathlib import Path
from lm_datasets.datasets.base import BaseDataset, GB, Availability, License
import logging

logger = logging.getLogger(__name__)


class BulNCDataset(BaseDataset):
    DATASET_ID = "bulnc"
    TITLE = "Bulgarian National Corpus"
    AVAILIBILITY = Availability.ON_REQUEST
    DOWNLOAD_URLS = ["http://old.dcl.bas.bg/dataset/BulNC.7z"]  # password-protected file!
    DESCRIPTION = (
        "The Bulgarian National Corpus contains a wide range of texts in various sizes, media types (written and "
        "spoken), styles, periods (synchronic and diachronic), and licenses. Each text in the collection is supplied "
        "with metadata. The Bulgarian National Corpus  was first compiled using the Bulgarian Lexicographic Archive "
        "and the Text Archive of Written Bulgarian, which account for 55.95% of the corpus. Later, the EMEA corpus "
        "(medical administrative texts) and the OpenSubtitles corpus (film subtitles) were added, accounting for "
        "1.27% and 8.61% of the BulNC, respectively. The remaining texts were crawled automatically and include a "
        "large number of administrative texts, news from monolingual and multilingual sources, scientific texts, and "
        "popular science. The BulNC is not fully downloadable due to the inclusion of copyrighted material. We've "
        "provided a link to a password-protected archive for evaluation."
    )
    AVAILIBILITY = Availability.ON_REQUEST
    LICENSE = License("research only", sharealike=False)
    LANGUAGES = ["bg"]
    BYTES = 1.8 * GB

    def decompress(self):
        """
        7z x BulNC.7z

        Folders: 125
        Files: 256906
        Size:       13279357395
        Compressed: 1981942477
        """
        pass

    def get_texts(self):
        # read from extracted TXT files
        for file_path in Path(self.get_local_dataset_dir()).rglob(
            "*.txt"
        ):  # self.get_dataset_file_paths(subdirectories=True, needed_suffix=".txt"):
            with open(file_path) as f:
                text = f.read()

                yield text
