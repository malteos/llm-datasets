from lm_datasets.datasets.base import BaseDataset, GB, Availability
import logging

logger = logging.getLogger(__name__)


class BulNCDataset(BaseDataset):
    DATASET_ID = "bulnc"
    TITLE = "Bulgarian National Corpus"
    AVAILIBILITY = Availability.ON_REQUEST
    DOWNLOAD_URLS = ["http://old.dcl.bas.bg/dataset/BulNC.7z"]
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

    LANGUAGES = ["bg"]
    BYTES = 1.8 * GB

    def decompress(self):
        # 7z x BulNC.7z
        pass

    def get_texts(self):
        # # read from extracted JSON files
        # fps = self.get_dataset_file_paths(subdirectories=True, needed_suffix=".json")
        raise NotImplementedError()
