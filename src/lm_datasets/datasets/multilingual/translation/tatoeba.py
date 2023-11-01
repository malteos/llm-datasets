import logging
import random
from typing import List

from lm_datasets.datasets.base import QualityWarning, License
from lm_datasets.datasets.hf_dataset import HFDataset

from lm_datasets.datasets.multilingual.translation.templates import get_templates
from lm_datasets.utils.languages import LANGUAGE_CODE_TO_NAME


logger = logging.getLogger(__name__)


# ISO 639-3 codes
TATOEBA_LANGUAGES = "acm,af,ain,ang,ar,arq,arz,ast,avk,az,be,ber,bg,bn,bod,br,bs,ca,ch,ckt,cmn,cs,cy,cycl,da,de,dsb,ee,el,en,eo,es,et,eu,fi,fo,fr,fy,ga,gd,gl,gn,he,hi,hil,hr,hsb,hu,hy,ia,id,ie,io,is,it,ja,jbo,ka,kk,km,ko,ksh,ku,kw,la,lad,lld,lo,lt,lv,lzh,mg,mi,ml,mn,mr,mt,nan,nb,nds,nl,non,nov,npi,oc,orv,os,pcd,pes,pl,pms,pnb,prg,pt,qu,qya,rm,ro,ru,sa,scn,sjn,sl,sq,sr,sv,swh,te,tg,th,tl,tlh,toki,tpi,tpw,tr,tt,ug,uk,ur,uz,vi,vo,wuu,xal,xh,yi,yue,zsm".split(
    ","
)  # noqa

VALID_EURO_TATOEBA_LANGUAGE_PAIRS = [
    ("sv", "uk"),
    ("mt", "pt"),
    ("mt", "nl"),
    ("mt", "sl"),
    ("mt", "pl"),
    ("pt", "sv"),
    ("pt", "uk"),
    ("pt", "sr"),
    ("pt", "sl"),
    ("pt", "ro"),
    ("ga", "sv"),
    ("ga", "nl"),
    ("ga", "sl"),
    ("ga", "ro"),
    ("ga", "pl"),
    ("ga", "hu"),
    ("ga", "lv"),
    ("sr", "uk"),
    ("cs", "sv"),
    ("cs", "mt"),
    ("cs", "uk"),
    ("cs", "pt"),
    ("cs", "sr"),
    ("cs", "da"),
    ("cs", "en"),
    ("cs", "fr"),
    ("cs", "nl"),
    ("cs", "sl"),
    ("cs", "et"),
    ("cs", "es"),
    ("cs", "fi"),
    ("cs", "ro"),
    ("cs", "lt"),
    ("cs", "pl"),
    ("cs", "hu"),
    ("cs", "de"),
    ("cs", "it"),
    ("cs", "el"),
    ("cs", "hr"),
    ("da", "sv"),
    ("da", "uk"),
    ("da", "pt"),
    ("da", "ga"),
    ("da", "en"),
    ("da", "fr"),
    ("da", "nl"),
    ("da", "sl"),
    ("da", "es"),
    ("da", "fi"),
    ("da", "ro"),
    ("da", "lt"),
    ("da", "pl"),
    ("da", "hu"),
    ("da", "de"),
    ("da", "lv"),
    ("da", "it"),
    ("da", "el"),
    ("bg", "sv"),
    ("bg", "uk"),
    ("bg", "pt"),
    ("bg", "sr"),
    ("bg", "cs"),
    ("bg", "da"),
    ("bg", "en"),
    ("bg", "fr"),
    ("bg", "nl"),
    ("bg", "sl"),
    ("bg", "es"),
    ("bg", "fi"),
    ("bg", "ro"),
    ("bg", "pl"),
    ("bg", "hu"),
    ("bg", "de"),
    ("bg", "lv"),
    ("bg", "it"),
    ("bg", "el"),
    ("en", "sv"),
    ("en", "mt"),
    ("en", "uk"),
    ("en", "pt"),
    ("en", "ga"),
    ("en", "sr"),
    ("en", "fr"),
    ("en", "nl"),
    ("en", "sl"),
    ("en", "et"),
    ("en", "es"),
    ("en", "fi"),
    ("en", "ro"),
    ("en", "lt"),
    ("en", "pl"),
    ("en", "hu"),
    ("en", "gl"),
    ("en", "lv"),
    ("en", "eu"),
    ("en", "it"),
    ("en", "hr"),
    ("fr", "sv"),
    ("fr", "mt"),
    ("fr", "uk"),
    ("fr", "pt"),
    ("fr", "ga"),
    ("fr", "sr"),
    ("fr", "nl"),
    ("fr", "sl"),
    ("fr", "ro"),
    ("fr", "lt"),
    ("fr", "pl"),
    ("fr", "hu"),
    ("fr", "gl"),
    ("fr", "lv"),
    ("fr", "it"),
    ("fr", "hr"),
    ("nl", "sv"),
    ("nl", "uk"),
    ("nl", "pt"),
    ("nl", "sr"),
    ("nl", "sl"),
    ("nl", "ro"),
    ("nl", "pl"),
    ("sl", "sv"),
    ("sl", "uk"),
    ("sl", "sr"),
    ("et", "sv"),
    ("et", "uk"),
    ("et", "fr"),
    ("et", "nl"),
    ("et", "sl"),
    ("et", "fi"),
    ("et", "ro"),
    ("et", "lt"),
    ("et", "pl"),
    ("et", "lv"),
    ("et", "it"),
    ("es", "sv"),
    ("es", "mt"),
    ("es", "uk"),
    ("es", "pt"),
    ("es", "ga"),
    ("es", "sr"),
    ("es", "fr"),
    ("es", "nl"),
    ("es", "sl"),
    ("es", "et"),
    ("es", "fi"),
    ("es", "ro"),
    ("es", "lt"),
    ("es", "pl"),
    ("es", "hu"),
    ("es", "gl"),
    ("es", "lv"),
    ("es", "eu"),
    ("es", "it"),
    ("es", "hr"),
    ("fi", "sv"),
    ("fi", "uk"),
    ("fi", "pt"),
    ("fi", "sr"),
    ("fi", "fr"),
    ("fi", "nl"),
    ("fi", "sl"),
    ("fi", "ro"),
    ("fi", "lt"),
    ("fi", "pl"),
    ("fi", "hu"),
    ("fi", "lv"),
    ("fi", "it"),
    ("fi", "hr"),
    ("ca", "sv"),
    ("ca", "uk"),
    ("ca", "pt"),
    ("ca", "en"),
    ("ca", "fr"),
    ("ca", "nl"),
    ("ca", "et"),
    ("ca", "es"),
    ("ca", "fi"),
    ("ca", "ro"),
    ("ca", "lt"),
    ("ca", "pl"),
    ("ca", "hu"),
    ("ca", "de"),
    ("ca", "gl"),
    ("ca", "lv"),
    ("ca", "it"),
    ("ca", "el"),
    ("ro", "sv"),
    ("ro", "uk"),
    ("ro", "sl"),
    ("lt", "sv"),
    ("lt", "uk"),
    ("lt", "pt"),
    ("lt", "nl"),
    ("lt", "sl"),
    ("lt", "pl"),
    ("lt", "lv"),
    ("pl", "sv"),
    ("pl", "uk"),
    ("pl", "pt"),
    ("pl", "sr"),
    ("pl", "sl"),
    ("pl", "ro"),
    ("hu", "sv"),
    ("hu", "uk"),
    ("hu", "pt"),
    ("hu", "sr"),
    ("hu", "nl"),
    ("hu", "sl"),
    ("hu", "ro"),
    ("hu", "lt"),
    ("hu", "pl"),
    ("hu", "lv"),
    ("hu", "it"),
    ("de", "sv"),
    ("de", "mt"),
    ("de", "uk"),
    ("de", "pt"),
    ("de", "ga"),
    ("de", "sr"),
    ("de", "en"),
    ("de", "fr"),
    ("de", "nl"),
    ("de", "sl"),
    ("de", "et"),
    ("de", "es"),
    ("de", "fi"),
    ("de", "ro"),
    ("de", "lt"),
    ("de", "pl"),
    ("de", "hu"),
    ("de", "gl"),
    ("de", "lv"),
    ("de", "eu"),
    ("de", "it"),
    ("de", "el"),
    ("de", "hr"),
    ("gl", "pt"),
    ("gl", "nl"),
    ("gl", "lt"),
    ("gl", "pl"),
    ("gl", "it"),
    ("lv", "sv"),
    ("lv", "uk"),
    ("lv", "pt"),
    ("lv", "nl"),
    ("lv", "sl"),
    ("lv", "ro"),
    ("lv", "pl"),
    ("eu", "pt"),
    ("eu", "fr"),
    ("eu", "nl"),
    ("eu", "pl"),
    ("eu", "it"),
    ("it", "sv"),
    ("it", "mt"),
    ("it", "uk"),
    ("it", "pt"),
    ("it", "sr"),
    ("it", "nl"),
    ("it", "sl"),
    ("it", "ro"),
    ("it", "lt"),
    ("it", "pl"),
    ("it", "lv"),
    ("el", "sv"),
    ("el", "mt"),
    ("el", "uk"),
    ("el", "pt"),
    ("el", "ga"),
    ("el", "en"),
    ("el", "fr"),
    ("el", "nl"),
    ("el", "es"),
    ("el", "ro"),
    ("el", "lt"),
    ("el", "pl"),
    ("el", "hu"),
    ("el", "gl"),
    ("el", "lv"),
    ("el", "it"),
    ("hr", "sv"),
    ("hr", "uk"),
    ("hr", "pt"),
    ("hr", "sr"),
    ("hr", "sl"),
    ("hr", "ro"),
    ("hr", "pl"),
    ("hr", "hu"),
    ("hr", "lv"),
    ("hr", "it"),
]

DATASET_IDS = "tatoeba_translation_sv_uk,tatoeba_translation_mt_pt,tatoeba_translation_mt_nl,tatoeba_translation_mt_sl,tatoeba_translation_mt_pl,tatoeba_translation_pt_sv,tatoeba_translation_pt_uk,tatoeba_translation_pt_sr,tatoeba_translation_pt_sl,tatoeba_translation_pt_ro,tatoeba_translation_ga_sv,tatoeba_translation_ga_nl,tatoeba_translation_ga_sl,tatoeba_translation_ga_ro,tatoeba_translation_ga_pl,tatoeba_translation_ga_hu,tatoeba_translation_ga_lv,tatoeba_translation_sr_uk,tatoeba_translation_cs_sv,tatoeba_translation_cs_mt,tatoeba_translation_cs_uk,tatoeba_translation_cs_pt,tatoeba_translation_cs_sr,tatoeba_translation_cs_da,tatoeba_translation_cs_en,tatoeba_translation_cs_fr,tatoeba_translation_cs_nl,tatoeba_translation_cs_sl,tatoeba_translation_cs_et,tatoeba_translation_cs_es,tatoeba_translation_cs_fi,tatoeba_translation_cs_ro,tatoeba_translation_cs_lt,tatoeba_translation_cs_pl,tatoeba_translation_cs_hu,tatoeba_translation_cs_de,tatoeba_translation_cs_it,tatoeba_translation_cs_el,tatoeba_translation_cs_hr,tatoeba_translation_da_sv,tatoeba_translation_da_uk,tatoeba_translation_da_pt,tatoeba_translation_da_ga,tatoeba_translation_da_en,tatoeba_translation_da_fr,tatoeba_translation_da_nl,tatoeba_translation_da_sl,tatoeba_translation_da_es,tatoeba_translation_da_fi,tatoeba_translation_da_ro,tatoeba_translation_da_lt,tatoeba_translation_da_pl,tatoeba_translation_da_hu,tatoeba_translation_da_de,tatoeba_translation_da_lv,tatoeba_translation_da_it,tatoeba_translation_da_el,tatoeba_translation_bg_sv,tatoeba_translation_bg_uk,tatoeba_translation_bg_pt,tatoeba_translation_bg_sr,tatoeba_translation_bg_cs,tatoeba_translation_bg_da,tatoeba_translation_bg_en,tatoeba_translation_bg_fr,tatoeba_translation_bg_nl,tatoeba_translation_bg_sl,tatoeba_translation_bg_es,tatoeba_translation_bg_fi,tatoeba_translation_bg_ro,tatoeba_translation_bg_pl,tatoeba_translation_bg_hu,tatoeba_translation_bg_de,tatoeba_translation_bg_lv,tatoeba_translation_bg_it,tatoeba_translation_bg_el,tatoeba_translation_en_sv,tatoeba_translation_en_mt,tatoeba_translation_en_uk,tatoeba_translation_en_pt,tatoeba_translation_en_ga,tatoeba_translation_en_sr,tatoeba_translation_en_fr,tatoeba_translation_en_nl,tatoeba_translation_en_sl,tatoeba_translation_en_et,tatoeba_translation_en_es,tatoeba_translation_en_fi,tatoeba_translation_en_ro,tatoeba_translation_en_lt,tatoeba_translation_en_pl,tatoeba_translation_en_hu,tatoeba_translation_en_gl,tatoeba_translation_en_lv,tatoeba_translation_en_eu,tatoeba_translation_en_it,tatoeba_translation_en_hr,tatoeba_translation_fr_sv,tatoeba_translation_fr_mt,tatoeba_translation_fr_uk,tatoeba_translation_fr_pt,tatoeba_translation_fr_ga,tatoeba_translation_fr_sr,tatoeba_translation_fr_nl,tatoeba_translation_fr_sl,tatoeba_translation_fr_ro,tatoeba_translation_fr_lt,tatoeba_translation_fr_pl,tatoeba_translation_fr_hu,tatoeba_translation_fr_gl,tatoeba_translation_fr_lv,tatoeba_translation_fr_it,tatoeba_translation_fr_hr,tatoeba_translation_nl_sv,tatoeba_translation_nl_uk,tatoeba_translation_nl_pt,tatoeba_translation_nl_sr,tatoeba_translation_nl_sl,tatoeba_translation_nl_ro,tatoeba_translation_nl_pl,tatoeba_translation_sl_sv,tatoeba_translation_sl_uk,tatoeba_translation_sl_sr,tatoeba_translation_et_sv,tatoeba_translation_et_uk,tatoeba_translation_et_fr,tatoeba_translation_et_nl,tatoeba_translation_et_sl,tatoeba_translation_et_fi,tatoeba_translation_et_ro,tatoeba_translation_et_lt,tatoeba_translation_et_pl,tatoeba_translation_et_lv,tatoeba_translation_et_it,tatoeba_translation_es_sv,tatoeba_translation_es_mt,tatoeba_translation_es_uk,tatoeba_translation_es_pt,tatoeba_translation_es_ga,tatoeba_translation_es_sr,tatoeba_translation_es_fr,tatoeba_translation_es_nl,tatoeba_translation_es_sl,tatoeba_translation_es_et,tatoeba_translation_es_fi,tatoeba_translation_es_ro,tatoeba_translation_es_lt,tatoeba_translation_es_pl,tatoeba_translation_es_hu,tatoeba_translation_es_gl,tatoeba_translation_es_lv,tatoeba_translation_es_eu,tatoeba_translation_es_it,tatoeba_translation_es_hr,tatoeba_translation_fi_sv,tatoeba_translation_fi_uk,tatoeba_translation_fi_pt,tatoeba_translation_fi_sr,tatoeba_translation_fi_fr,tatoeba_translation_fi_nl,tatoeba_translation_fi_sl,tatoeba_translation_fi_ro,tatoeba_translation_fi_lt,tatoeba_translation_fi_pl,tatoeba_translation_fi_hu,tatoeba_translation_fi_lv,tatoeba_translation_fi_it,tatoeba_translation_fi_hr,tatoeba_translation_ca_sv,tatoeba_translation_ca_uk,tatoeba_translation_ca_pt,tatoeba_translation_ca_en,tatoeba_translation_ca_fr,tatoeba_translation_ca_nl,tatoeba_translation_ca_et,tatoeba_translation_ca_es,tatoeba_translation_ca_fi,tatoeba_translation_ca_ro,tatoeba_translation_ca_lt,tatoeba_translation_ca_pl,tatoeba_translation_ca_hu,tatoeba_translation_ca_de,tatoeba_translation_ca_gl,tatoeba_translation_ca_lv,tatoeba_translation_ca_it,tatoeba_translation_ca_el,tatoeba_translation_ro_sv,tatoeba_translation_ro_uk,tatoeba_translation_ro_sl,tatoeba_translation_lt_sv,tatoeba_translation_lt_uk,tatoeba_translation_lt_pt,tatoeba_translation_lt_nl,tatoeba_translation_lt_sl,tatoeba_translation_lt_pl,tatoeba_translation_lt_lv,tatoeba_translation_pl_sv,tatoeba_translation_pl_uk,tatoeba_translation_pl_pt,tatoeba_translation_pl_sr,tatoeba_translation_pl_sl,tatoeba_translation_pl_ro,tatoeba_translation_hu_sv,tatoeba_translation_hu_uk,tatoeba_translation_hu_pt,tatoeba_translation_hu_sr,tatoeba_translation_hu_nl,tatoeba_translation_hu_sl,tatoeba_translation_hu_ro,tatoeba_translation_hu_lt,tatoeba_translation_hu_pl,tatoeba_translation_hu_lv,tatoeba_translation_hu_it,tatoeba_translation_de_sv,tatoeba_translation_de_mt,tatoeba_translation_de_uk,tatoeba_translation_de_pt,tatoeba_translation_de_ga,tatoeba_translation_de_sr,tatoeba_translation_de_en,tatoeba_translation_de_fr,tatoeba_translation_de_nl,tatoeba_translation_de_sl,tatoeba_translation_de_et,tatoeba_translation_de_es,tatoeba_translation_de_fi,tatoeba_translation_de_ro,tatoeba_translation_de_lt,tatoeba_translation_de_pl,tatoeba_translation_de_hu,tatoeba_translation_de_gl,tatoeba_translation_de_lv,tatoeba_translation_de_eu,tatoeba_translation_de_it,tatoeba_translation_de_el,tatoeba_translation_de_hr,tatoeba_translation_gl_pt,tatoeba_translation_gl_nl,tatoeba_translation_gl_lt,tatoeba_translation_gl_pl,tatoeba_translation_gl_it,tatoeba_translation_lv_sv,tatoeba_translation_lv_uk,tatoeba_translation_lv_pt,tatoeba_translation_lv_nl,tatoeba_translation_lv_sl,tatoeba_translation_lv_ro,tatoeba_translation_lv_pl,tatoeba_translation_eu_pt,tatoeba_translation_eu_fr,tatoeba_translation_eu_nl,tatoeba_translation_eu_pl,tatoeba_translation_eu_it,tatoeba_translation_it_sv,tatoeba_translation_it_mt,tatoeba_translation_it_uk,tatoeba_translation_it_pt,tatoeba_translation_it_sr,tatoeba_translation_it_nl,tatoeba_translation_it_sl,tatoeba_translation_it_ro,tatoeba_translation_it_lt,tatoeba_translation_it_pl,tatoeba_translation_it_lv,tatoeba_translation_el_sv,tatoeba_translation_el_mt,tatoeba_translation_el_uk,tatoeba_translation_el_pt,tatoeba_translation_el_ga,tatoeba_translation_el_en,tatoeba_translation_el_fr,tatoeba_translation_el_nl,tatoeba_translation_el_es,tatoeba_translation_el_ro,tatoeba_translation_el_lt,tatoeba_translation_el_pl,tatoeba_translation_el_hu,tatoeba_translation_el_gl,tatoeba_translation_el_lv,tatoeba_translation_el_it,tatoeba_translation_hr_sv,tatoeba_translation_hr_uk,tatoeba_translation_hr_pt,tatoeba_translation_hr_sr,tatoeba_translation_hr_sl,tatoeba_translation_hr_ro,tatoeba_translation_hr_pl,tatoeba_translation_hr_hu,tatoeba_translation_hr_lv,tatoeba_translation_hr_it"  # noqa


class TatoebaTranslationBaseDataset(HFDataset):
    """
    Tatoeba is a collection of sentences and translations.

    To load a language pair which isn't part of the config, all you need to do is specify the language code as pairs. You can find the valid pairs in Homepage section of Dataset Description: http://opus.nlpl.eu/Tatoeba.php E.g.
    """

    DATASET_ID = None
    LANGUAGES: List = None  # [source_language, target_language]

    SOURCE_ID = "tatoeba_translation"
    TITLE = "Tatoeba"
    DESCRIPTION = "Tatoeba is a collection of sentences and translations."
    HOMEPAGE = "http://opus.nlpl.eu/Tatoeba.php"
    LICENSE = License(
        "CC-BY 2.0 FR",
        url="https://creativecommons.org/licenses/by/2.0/fr/",
        attribution=True,
        research_use=True,
        commercial_use=True,
    )
    TRANSLATIONS = True

    QUALITY_WARNINGS = [QualityWarning.SHORT_TEXT]

    HF_DATASET_ID = "tatoeba"
    HF_DATASET_SPLIT = "train"
    HF_DATASET_CONFIGS = None

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


def get_tatoeba_dataset(language_pair):
    source_lang, target_lang = language_pair

    class TatoebaTranslationDataset(TatoebaTranslationBaseDataset):
        DATASET_ID = f"tatoeba_translation_{source_lang}_{target_lang}"
        LANGUAGES = [source_lang, target_lang]

        HF_KWARGS = dict(
            lang1=source_lang,
            lang2=target_lang,
            date="v2023-04-12",  # See http://opus.nlpl.eu/Tatoeba.php
        )

    return TatoebaTranslationDataset


def get_tatoeba_auto_classes():
    return [get_tatoeba_dataset(pair) for pair in VALID_EURO_TATOEBA_LANGUAGE_PAIRS]
