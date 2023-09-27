import logging
import random
from typing import List

from lm_datasets.datasets.base import BaseDataset, QualityWarning
from lm_datasets.datasets.hf_dataset import HFDataset
from lm_datasets.utils.settings import EURO_LANGUAGES
from lm_datasets.utils.languages import LANGUAGE_CODE_TO_NAME
from lm_datasets.datasets.multilingual.translation.templates import get_templates

from datasets import load_dataset

logger = logging.getLogger(__name__)

OPUS_SUPERVISED_LANGUAGE_PAIRS = [
    "af-en",
    "am-en",
    "an-en",
    "ar-en",
    "as-en",
    "az-en",
    "be-en",
    "bg-en",
    "bn-en",
    "br-en",
    "bs-en",
    "ca-en",
    "cs-en",
    "cy-en",
    "da-en",
    "de-en",
    "dz-en",
    "el-en",
    "en-eo",
    "en-es",
    "en-et",
    "en-eu",
    "en-fa",
    "en-fi",
    "en-fr",
    "en-fy",
    "en-ga",
    "en-gd",
    "en-gl",
    "en-gu",
    "en-ha",
    "en-he",
    "en-hi",
    "en-hr",
    "en-hu",
    "en-hy",
    "en-id",
    "en-ig",
    "en-is",
    "en-it",
    "en-ja",
    "en-ka",
    "en-kk",
    "en-km",
    "en-ko",
    "en-kn",
    "en-ku",
    "en-ky",
    "en-li",
    "en-lt",
    "en-lv",
    "en-mg",
    "en-mk",
    "en-ml",
    "en-mn",
    "en-mr",
    "en-ms",
    "en-mt",
    "en-my",
    "en-nb",
    "en-ne",
    "en-nl",
    "en-nn",
    "en-no",
    "en-oc",
    "en-or",
    "en-pa",
    "en-pl",
    "en-ps",
    "en-pt",
    "en-ro",
    "en-ru",
    "en-rw",
    "en-se",
    "en-sh",
    "en-si",
    "en-sk",
    "en-sl",
    "en-sq",
    "en-sr",
    "en-sv",
    "en-ta",
    "en-te",
    "en-tg",
    "en-th",
    "en-tk",
    "en-tr",
    "en-tt",
    "en-ug",
    "en-uk",
    "en-ur",
    "en-uz",
    "en-vi",
    "en-wa",
    "en-xh",
    "en-yi",
    "en-yo",
    "en-zh",
    "en-zu",
]


def get_euro_opus_pairs():
    euro_langs = set(EURO_LANGUAGES)
    pairs = []
    for pair in OPUS_SUPERVISED_LANGUAGE_PAIRS:
        source_lang, target_lang = pair.split("-")

        if source_lang in euro_langs and target_lang in euro_langs:
            pairs.append((source_lang, target_lang))

    return pairs


class Opus100TranslationBaseDataset(HFDataset):
    """
    OPUS-100 is English-centric, meaning that all training pairs include English on either the source or target side. The corpus covers 100 languages (including English). Selected the languages based on the volume of parallel data available in OPUS.
    """

    DATASET_ID = None
    LANGUAGES: List = None  # [source_language, target_language]

    SOURCE_ID = "opus100_translation"
    TITLE = "OPUS-100"
    DESCRIPTION = "OPUS-100 is English-centric, meaning that all training pairs include English on either the source or target side."
    CITATION = """@misc{zhang2020improving,
      title={Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation},
      author={Biao Zhang and Philip Williams and Ivan Titov and Rico Sennrich},
      year={2020},
      eprint={2004.11867},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}"""
    HOMEPAGE = "https://huggingface.co/datasets/opus100"
    TRANSLATIONS = True

    QUALITY_WARNINGS = [QualityWarning.SHORT_TEXT]

    HF_DATASET_ID = "opus100"
    HF_DATASET_SPLIT = "train"
    HF_DATASET_CONFIGS = None  # replace with "source-target"

    TOKENS = 0  # unknown

    # streaming = True
    keep_columns = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        from jinja2 import Template

        self.templates = [Template(tpl) for tpl in get_templates()]

        assert len(self.LANGUAGES) == 2

        # override min_length filter!
        # self.min_length = 32

    def get_texts_from_item(self, item):
        """
        Fill a random template with sample data.

        Template variables:

        - SOURCE_LANG
        - TARGET_LANG
        - SOURCE_TEXT
        - TARGET_TEXT
        """

        # both source-target combintations
        for source_lang, target_lang in [
            (self.LANGUAGES[0], self.LANGUAGES[1]),
            (self.LANGUAGES[1], self.LANGUAGES[0]),
        ]:
            tpl = random.choice(self.templates)

            # for tpl in self.templates:
            text = tpl.render(
                SOURCE_LANG=LANGUAGE_CODE_TO_NAME[source_lang],
                TARGET_LANG=LANGUAGE_CODE_TO_NAME[target_lang],
                SOURCE_TEXT=item["translation"][source_lang],
                TARGET_TEXT=item["translation"][target_lang],
            )
            yield text


# source_lang = "de"
# target_lang = "en"

# hf_dataset = load_dataset(
#     "opus100",
#     f"{source_lang}-{target_lang}",
#     split="train",
#     streaming=True,
# )

# ds_iterator = iter(hf_dataset)

# for item in ds_iterator:
#     print(item)
#     break


# print("done")


def get_opus_dataset(source_lang, target_lang):
    class Opus100TranslationDataset(Opus100TranslationBaseDataset):
        DATASET_ID = f"opus100_translation_{source_lang}_{target_lang}"
        LANGUAGES = [source_lang, target_lang]
        HF_DATASET_CONFIGS = [f"{source_lang}-{target_lang}"]

    return Opus100TranslationDataset


def get_opus100_auto_classes():
    clss = [get_opus_dataset(*pair) for pair in get_euro_opus_pairs()]

    return clss
