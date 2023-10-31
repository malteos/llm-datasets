import logging
import os
from pathlib import Path

import numpy as np

from lm_datasets.datasets.base import Availability, License
from lm_datasets.datasets.jsonl_dataset import JSONLDataset


logger = logging.getLogger(__name__)


# Magic numbers
DEFAULT_OSCAR_MIN_HARMFUL_PP = 25.0
DEFAULT_OSCAR_MAX_HARMFUL_PP = 100_000

"""
Extract list:
",".join([f"colossal_oscar_{dump}_ca" for dump in  OSCAR_DUMPS])
'colossal_oscar_05-06-23_bg,colossal_oscar_05-06-23_cs,colossal_oscar_05-06-23_da,colossal_oscar_05-06-23_el,colossal_oscar_05-06-23_et,colossal_oscar_05-06-23_fi,colossal_oscar_05-06-23_fr,colossal_oscar_05-06-23_ga,colossal_oscar_05-06-23_hr,colossal_oscar_05-06-23_hu,colossal_oscar_05-06-23_lt,colossal_oscar_05-06-23_lv,colossal_oscar_05-06-23_mt,colossal_oscar_05-06-23_nl,colossal_oscar_05-06-23_pl,colossal_oscar_05-06-23_pt,colossal_oscar_05-06-23_ro,colossal_oscar_05-06-23_sk,colossal_oscar_05-06-23_sl,colossal_oscar_05-06-23_sv,colossal_oscar_05-06-23_uk,colossal_oscar_05-06-23_sr,colossal_oscar_05-06-23_sh,colossal_oscar_05-06-23_nn,colossal_oscar_05-06-23_no,colossal_oscar_05-06-23_eu,colossal_oscar_05-06-23_ca,colossal_oscar_05-06-23_gl,colossal_oscar_03-04-23_bg,colossal_oscar_03-04-23_cs,colossal_oscar_03-04-23_da,colossal_oscar_03-04-23_el,colossal_oscar_03-04-23_et,colossal_oscar_03-04-23_fi,colossal_oscar_03-04-23_fr,colossal_oscar_03-04-23_ga,colossal_oscar_03-04-23_hr,colossal_oscar_03-04-23_hu,colossal_oscar_03-04-23_lt,colossal_oscar_03-04-23_lv,colossal_oscar_03-04-23_mt,colossal_oscar_03-04-23_nl,colossal_oscar_03-04-23_pl,colossal_oscar_03-04-23_pt,colossal_oscar_03-04-23_ro,colossal_oscar_03-04-23_sk,colossal_oscar_03-04-23_sl,colossal_oscar_03-04-23_sv,colossal_oscar_03-04-23_uk,colossal_oscar_03-04-23_sr,colossal_oscar_03-04-23_sh,colossal_oscar_03-04-23_nn,colossal_oscar_03-04-23_no,colossal_oscar_03-04-23_eu,colossal_oscar_03-04-23_ca,colossal_oscar_03-04-23_gl'
"""  # noqa
# all dumps: 2015-14  2016-40  2017-43  2018-47  2019-22  2020-24  2020-45  2021-49  2022-27  2022-49  2023-14  2023-23
# first sample:
# OSCAR_DUMPS = ["05-06-23", "03-04-23"]
OSCAR_DUMPS = [
    "2015-14",
    "2016-40",
    "2017-43",
    "2018-47",
    "2019-22",
    "2020-24",
    "2020-45",
    "2021-49",
    "2022-27",
    "2022-49",
    "2023-14",
    "2023-23",
]
LANGUAGES = "bg cs da el et fi ga hr hu lt lv mt nl pl pt ro sk sl sv uk sr sh nn no eu ca gl".split(" ")  # removed fr
LANGUAGES += ["en", "de", "fr", "es", "it"]  # EU top-5 languages

EXCLUDE_CATEGORIES = {
    # See http://dsi.ut-capitole.fr/blacklists/index_en.php
    "agressif",
    "adult",
    "cryptojacking",
    "dangerous_material",
    "phishing",
    "warez",
    "ddos",
    "hacking",
    "malware",
    "mixed_adult",
    "sect",
}


class ColossalOscarBaseDataset(JSONLDataset):
    """
    Read OSCAR output from jsonl.zst files (as provided on HF)
    """

    DATASET_ID = None
    SOURCE_ID = "colossal_oscar"

    TITLE = "Colossal OSCAR 1"
    DESCRIPTION = "Colossal OSCAR 1"
    HOMEPAGE = "https://huggingface.co/datasets/oscar-corpus/colossal-oscar-1.0"
    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD

    LANGUAGES = ["da"]
    DUMP_VERSION = "05-06-23"
    WEB_CRAWLED = True

    LICENSE = License(
        "CommonCrawl terms of use; Only the annotations are distributed under a cc0-1.0 license",
        url="https://commoncrawl.org/terms-of-use",
        commercial_use=True,
        research_use=True,
    )

    min_harmful_pp = None
    max_harmful_pp = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.min_harmful_pp = DEFAULT_OSCAR_MIN_HARMFUL_PP
        self.max_harmful_pp = DEFAULT_OSCAR_MAX_HARMFUL_PP

        # Load language specific settings from config
        if hasattr(self.config, "colossal_oscar_min_harmful_pp_by_language"):
            colossal_oscar_min_harmful_pp_by_language = self.config.colossal_oscar_min_harmful_pp_by_language

            if self.get_language_code() in colossal_oscar_min_harmful_pp_by_language:
                self.min_harmful_pp = colossal_oscar_min_harmful_pp_by_language[self.get_language_code()]

        if hasattr(self.config, "colossal_oscar_max_harmful_pp_by_language"):
            colossal_oscar_max_harmful_pp_by_language = self.config.colossal_oscar_max_harmful_pp_by_language

            if self.get_language_code() in colossal_oscar_max_harmful_pp_by_language:
                self.max_harmful_pp = colossal_oscar_max_harmful_pp_by_language[self.get_language_code()]

        # logger.info(f"{self.min_harmful_pp=}")

    def get_text_from_item(self, doc):
        if doc["metadata"]["quality_warnings"]:
            self.counter.update({"filtered_quality_warnings": 1})
            return None
        elif (
            "harmful_pp" in doc["metadata"]
            and doc["metadata"]["harmful_pp"]
            and doc["metadata"]["harmful_pp"] < self.min_harmful_pp
        ):
            self.counter.update({"min_filtered_harmful_pp": 1})
            return None
        elif (
            "harmful_pp" in doc["metadata"]
            and doc["metadata"]["harmful_pp"]
            and doc["metadata"]["harmful_pp"] > self.max_harmful_pp
        ):
            self.counter.update({"max_filtered_harmful_pp": 1})
            return None
        elif doc["metadata"]["categories"] and len(set(doc["metadata"]["categories"]) & EXCLUDE_CATEGORIES) > 0:
            self.counter.update({"filtered_categories": 1})
            return None
        else:
            return doc["content"]

    def get_raw_jsonl_paths(self):
        lang = self.get_language_code()
        dataset_path = Path(os.path.join(self.get_local_dataset_dir(), self.DUMP_VERSION, f"{lang}_meta"))

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        return sorted([str(p) for p in dataset_path.glob("*.jsonl.zst")])

    def get_bytes(self):
        try:
            return sum(os.stat(fp).st_size for fp in self.get_raw_jsonl_paths())
        except FileNotFoundError:
            return -1


def get_colossal_oscar_class(lang, dump_version):
    class ColossalOscarDataset(ColossalOscarBaseDataset):
        DATASET_ID = f"colossal_oscar_{dump_version}_{lang}"
        TITLE = f"Colossal OSCAR 1 ({lang}; {dump_version})"
        LANGUAGES = [lang]
        DUMP_VERSION = dump_version

    return ColossalOscarDataset


def get_colossal_oscar_auto_classes():
    """
    Auto generate dataset classes
    """

    return [get_colossal_oscar_class(lang, dump_version) for dump_version in OSCAR_DUMPS for lang in LANGUAGES]
