import logging
import zipfile

from lm_datasets.datasets.base import MB, Availability, BaseDataset

logger = logging.getLogger(__name__)


class StateRelatedLatvianWebDataset(BaseDataset):
    DATASET_ID = "state_related_latvian_web"
    TITLE = "Corpus of State-related content from the Latvian Web (Processed)"
    HOMEPAGE = "http://catalog.elra.info/en-us/repository/browse/ELRA-W0169/"
    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD

    BYTES = 3.4 * MB
    LANGUAGES = ["lv"]

    def get_texts(self):
        from translate.storage.tmx import tmxfile

        zip_fp = self.get_dataset_file_paths(needed_suffix=".zip", single_file=True)

        with zipfile.ZipFile(zip_fp) as zf:
            for fn in zf.namelist():
                if fn.endswith(".tmx"):
                    with zf.open(fn) as member_f:
                        tmx_file = tmxfile(member_f, "lv", "en")

                    for i, node in enumerate(tmx_file.unit_iter()):
                        text = node.target  # lv
                        # en => node.source

                        yield text
