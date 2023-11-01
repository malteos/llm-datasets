import logging
import random
from typing import List

from lm_datasets.datasets.base import QualityWarning, License
from lm_datasets.datasets.hf_dataset import HFDataset
from lm_datasets.utils.settings import EURO_LANGUAGES
from lm_datasets.utils.languages import LANGUAGE_CODE_TO_NAME
from lm_datasets.datasets.multilingual.translation.templates import get_templates


logger = logging.getLogger(__name__)

WMT_LANGUAGE_PAIRS = [(lang, "en") for lang in ["cs", "de", "fi", "gu", "kk", "lt", "ru", "zh"]] + [("fr", "de")]


def get_euro_wmt19_pairs():
    euro_langs = set(EURO_LANGUAGES)
    pairs = []
    for pair in WMT_LANGUAGE_PAIRS:
        source_lang, target_lang = pair

        if source_lang in euro_langs and target_lang in euro_langs:
            pairs.append((source_lang, target_lang))

    return pairs


class WMT19TranslationBaseDataset(HFDataset):
    """
    Translation dataset based on the data from statmt.org.
    """

    DATASET_ID = None
    LANGUAGES: List = None  # [source_language, target_language]

    SOURCE_ID = "wmt19_translation"
    TITLE = "WMT19 (Workshop on Statistical Machine Translation)"
    DESCRIPTION = "Shared Task: Machine Translation of News."
    CITATION = """@inproceedings{barrault-etal-2019-findings,
    title = "Findings of the 2019 Conference on Machine Translation ({WMT}19)",
    author = {Barrault, Lo{\"\i}c  and
      Bojar, Ond{\v{r}}ej  and
      Costa-juss{\`a}, Marta R.  and
      Federmann, Christian  and
      Fishel, Mark  and
      Graham, Yvette  and
      Haddow, Barry  and
      Huck, Matthias  and
      Koehn, Philipp  and
      Malmasi, Shervin  and
      Monz, Christof  and
      M{\"u}ller, Mathias  and
      Pal, Santanu  and
      Post, Matt  and
      Zampieri, Marcos},
    booktitle = "Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1)",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-5301",
    doi = "10.18653/v1/W19-5301",
    pages = "1--61",
    abstract = "This paper presents the results of the premier shared task organized alongside the Conference on Machine Translation (WMT) 2019. Participants were asked to build machine translation systems for any of 18 language pairs, to be evaluated on a test set of news stories. The main metric for this task is human judgment of translation quality. The task was also opened up to additional test suites to probe specific aspects of translation.",
}
"""  # noqa
    HOMEPAGE = "https://www.statmt.org/wmt19/translation-task.html"
    TRANSLATIONS = True
    LICENSE = License("free for research purposes", research_use=True)
    QUALITY_WARNINGS = [QualityWarning.SHORT_TEXT]

    HF_DATASET_ID = "wmt19"
    HF_DATASET_SPLIT = "train"
    HF_DATASET_CONFIGS = None  # replace with "source-target"

    TOKENS = 0  # unknown

    streaming = True
    keep_columns = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        from jinja2 import Template

        self.templates = [Template(tpl) for tpl in get_templates()]

        assert len(self.LANGUAGES) == 2

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


def get_wmt_dataset(source_lang, target_lang):
    class WMT19TranslationDataset(WMT19TranslationBaseDataset):
        DATASET_ID = f"wmt19_translation_{source_lang}_{target_lang}"
        LANGUAGES = [source_lang, target_lang]
        HF_DATASET_CONFIGS = [f"{source_lang}-{target_lang}"]

    return WMT19TranslationDataset


def get_wmt19_auto_classes():
    clss = [get_wmt_dataset(*pair) for pair in get_euro_wmt19_pairs()]

    return clss
