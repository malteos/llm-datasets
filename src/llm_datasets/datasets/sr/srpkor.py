import json
from typing import Iterable
from llm_datasets.datasets.base import BaseDataset, Availability, QualityWarning, License
import zipfile
import logging

logger = logging.getLogger(__name__)


class SrpKorDataset(BaseDataset):
    DATASET_ID = "srpkor"
    TITLE = "SrpKorSubset (news, legal, academic, conversation, literary)"
    DESCRIPTION = "The Corpus of contemporary Serbian, SrpKor, consists of 4,925 texts."
    AVAILIBILITY = Availability.ON_REQUEST
    HOMEPAGE = "http://www.korpus.matf.bg.ac.rs/"  # http://metashare.elda.org/repository/browse/corpus-of-contemporary-serbian/00cc41168bdf11e29c9e0015171445924cdac8693bf840f780418187133495b8/
    LICENSE = License("Do not redistribute, DFKI has permission to use it for pre-training LLMs", distribution=False)

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
