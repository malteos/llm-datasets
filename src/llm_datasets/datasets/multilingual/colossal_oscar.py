import logging
import os
from pathlib import Path

import numpy as np

from llm_datasets.datasets.base import Availability, License
from llm_datasets.datasets.jsonl_dataset import JSONLDataset


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
EURO_LANGUAGES = "bg cs da el et fi ga hr hu lt lv mt nl pl pt ro sk sl sv uk sr sh nn no eu ca gl".split(
    " "
)  # removed fr
EURO_LANGUAGES += ["en", "de", "fr", "es", "it"]  # EU top-5 languages

# all languages available in OSCAR
OSCAR_LANGUAGES = {
    "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Aragonese": "an",
    "Armenian": "hy",
    "Assamese": "as",
    "Asturian": "ast",
    "Avaric": "av",
    "Azerbaijani": "az",
    "Bangla": "bn",
    "Bashkir": "ba",
    "Basque": "eu",
    "Belarusian": "be",
    "Bihari languages": "bh",
    "Bishnupriya": "bpy",
    "Bosnian": "bs",
    "Breton": "br",
    "Bulgarian": "bg",
    "Burmese": "my",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Central Kurdish": "ckb",
    "Chechen": "ce",
    "Chinese": "zh",
    "Chuvash": "cv",
    "Cornish": "kw",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Divehi": "dv",
    "Dutch": "nl",
    "Eastern Mari": "mhr",
    "Egyptian Arabic": "arz",
    "English": "en",
    "Esperanto": "eo",
    "Estonian": "et",
    "Filipino": "tl",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Goan Konkani": "gom",
    "Greek": "el",
    "Guarani": "gn",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Ido": "io",
    "Iloko": "ilo",
    "Indonesian": "id",
    "Interlingua": "ia",
    "Interlingue": "ie",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Kalmyk": "xal",
    "Kannada": "kn",
    "Karachay-Balkar": "krc",
    "Kazakh": "kk",
    "Khmer": "km",
    "Komi": "kv",
    "Korean": "ko",
    "Kurdish": "ku",
    "Kyrgyz": "ky",
    "Lao": "lo",
    "Latin": "la",
    "Latvian": "lv",
    "Lezghian": "lez",
    "Limburgish": "li",
    "Lithuanian": "lt",
    "Lojban": "jbo",
    "Lombard": "lmo",
    "Low German": "nds",
    "Lower Sorbian": "dsb",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Maithili": "mai",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Marathi": "mr",
    "Mazanderani": "mzn",
    "Minangkabau": "min",
    "Mingrelian": "xmf",
    "Mirandese": "mwl",
    "Mongolian": "mn",
    "Multilingual": "multi",
    "Nahuatl languages": "nah",
    "Nepali": "ne",
    "Newari": "new",
    "Norwegian": "no",
    "Norwegian Nynorsk": "nn",
    "Occitan": "oc",
    "Odia": "or",
    "Ossetic": "os",
    "Pashto": "ps",
    "Persian": "fa",
    "Piedmontese": "pms",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Quechua": "qu",
    "Romanian": "ro",
    "Russia Buriat": "bxr",
    "Russian": "ru",
    "Sakha": "sah",
    "Sanskrit": "sa",
    "Scottish Gaelic": "gd",
    "Serbian": "sr",
    "Serbian (Latin)": "sh",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "South Azerbaijani": "azb",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swedish": "sv",
    "Swiss German": "gsw",
    "Tajik": "tg",
    "Tamil": "ta",
    "Tatar": "tt",
    "Telugu": "te",
    "Thai": "th",
    "Tibetan": "bo",
    "Turkish": "tr",
    "Turkmen": "tk",
    "Ukrainian": "uk",
    "Emiliano-Romagnolo": "x-eml",
    "Upper Sorbian": "hsb",
    "Urdu": "ur",
    "Uyghur": "ug",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Volapük": "vo",
    "Walloon": "wa",
    "Waray": "war",
    "Welsh": "cy",
    "Western Frisian": "fy",
    "Western Mari": "mrj",
    "Western Panjabi": "pnb",
    "Wu Chinese": "wuu",
    "Yiddish": "yi",
    "Yoruba": "yo",
}

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
    LANGUAGES = None

    SOURCE_ID = "colossal_oscar"
    TITLE = "Colossal OSCAR 1.0"
    DESCRIPTION = (
        "The OSCAR project (Open Super-large Crawled Aggregated coRpus) is an Open Source project aiming "
        "to provide web-based multilingual resources and datasets for Machine Learning (ML) and Artificial "
        "Intelligence (AI) applications. The project focuses specifically in providing large quantities of "
        "unannotated raw data that is commonly used in the pre-training of large deep learning models."
    )
    HOMEPAGE = "https://huggingface.co/datasets/oscar-corpus/colossal-oscar-1.0"
    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD
    CITATION = r"""@misc{jansen2022perplexed,
      title={Perplexed by Quality: A Perplexity-based Method for Adult and Harmful Content Detection in Multilingual Heterogeneous Web Data},
      author={Tim Jansen and Yangling Tong and Victoria Zevallos and Pedro Ortiz Suarez},
      year={2022},
      eprint={2212.10440},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }"""  # noqa

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

    def download(self):
        raise ValueError(
            "Follow the instruction: https://huggingface.co/datasets/oscar-corpus/colossal-oscar-1.0#downloading-the-data"
        )

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
        except (FileNotFoundError, ValueError):
            return -1


def get_colossal_oscar_class(lang, dump_version):
    class ColossalOscarDataset(ColossalOscarBaseDataset):
        DATASET_ID = f"colossal_oscar_{dump_version}_{lang}"
        TITLE = f"Colossal OSCAR 1 [{lang}; {dump_version}]"
        LANGUAGES = [lang]
        DUMP_VERSION = dump_version

    return ColossalOscarDataset


def get_colossal_oscar_auto_classes():
    """
    Auto generate dataset classes
    """

    return [
        get_colossal_oscar_class(lang, dump_version)
        for dump_version in OSCAR_DUMPS
        for lang in OSCAR_LANGUAGES.values()
    ]
