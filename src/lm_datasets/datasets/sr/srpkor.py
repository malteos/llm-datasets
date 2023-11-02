import json
from typing import Iterable
from lm_datasets.datasets.base import BaseDataset, Availability, QualityWarning, License
import zipfile
import logging

logger = logging.getLogger(__name__)


class SrpKorDataset(BaseDataset):
    DATASET_ID = "srpkor"
    TITLE = "SrpKorSubset (news, legal, academic, conversation, literary)"

    AVAILIBILITY = Availability.ON_REQUEST

    LICENSE = License("Do not redistribute, DFKI has permission to use it for pre-training LLMs")

    LANGUAGES = ["sr"]
    QUALITY_WARNINGS = [QualityWarning.SHORT_TEXT]

    def get_texts(self) -> Iterable[str]:
        # read from OpenGptx-JeRTeh.zip
        zip_fp = self.get_dataset_file_paths(needed_suffix=".zip", single_file=True)
        logger.info("Extracting from %s", zip_fp)

        with zipfile.ZipFile(zip_fp) as zf:
            member_fns = zf.namelist()
            for fn in member_fns:
                if fn.endswith(".json"):
                    logger.info("Reading from %s", fn)

                    with zf.open(fn) as member_f:
                        obj = json.load(member_f)

                        for sent in obj["sents"]:
                            # yield str(len(obj["sents"]))
                            yield sent

                # break
