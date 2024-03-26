import logging
import zipfile

from llm_datasets.datasets.base import MB, Availability, BaseDataset, License

logger = logging.getLogger(__name__)


class StateRelatedLatvianWebDataset(BaseDataset):
    DATASET_ID = "state_related_latvian_web"
    TITLE = "Corpus of State-related content from the Latvian Web (Processed)"
    HOMEPAGE = "http://catalog.elra.info/en-us/repository/browse/ELRA-W0169/"
    DESCRIPTION = "Latvian Web, home pages of ministries and state public services, army, etc. were crawled, and parallel Latvian-English content was collected."
    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD
    LICENSE = License("CC-BY-SA-4.0", attribution=True, sharealike=True, commercial_use=True, research_use=True)
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
