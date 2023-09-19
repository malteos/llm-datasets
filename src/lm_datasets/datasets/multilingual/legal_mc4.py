from lm_datasets.datasets.base import Availability
from lm_datasets.datasets.hf_dataset import HFDataset


# From https://docs.google.com/spreadsheets/d/1_rfLKa_Kq09YI0BPnfmoSL6-U3SHrW8tRwmk3-Qchzo/edit#gid=1206328236
RAW_LANG_TO_TOKENS = """bg	2390349
cs	1840827375
da	10466716
de	6184578784
el	1155977
en	966539309
es	9058939804
et	110198368
fi	62799074
fr	2117306229
ga	32772
hu	244911748
it	3053920779
lt	9142223
lv	58702
mt	3479869
nl	21962633
pl	2235839721
pt	1338147828
ro	551372510
sk	349265172
sl	107493024
sv	328471555"""


class LegalMC4BaseDataset(HFDataset):
    SOURCE_ID = "legal_mc4"
    DESCRIPTION = "MC4_Legal: A Corpus Covering the Legal Part of MC4 for European Languages"
    HOMEPAGE = "https://huggingface.co/datasets/joelito/legal-mc4"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    WEB_CRAWLED = True
    # DUMMY = True

    HF_DATASET_ID = "joelito/legal-mc4"
    HF_DATASET_SPLIT = "train"
    HF_DATASET_CONFIGS = None  # is set by language version

    streaming = True


def get_legal_mc4_auto_cls_by_language(lang, tokens):
    class LegalMC4LanguageDataset(LegalMC4BaseDataset):
        TOKENS = tokens
        DATASET_ID = "legal_mc4_" + lang
        TITLE = "legal_mc4_" + lang
        LANGUAGES = [lang]

        @property
        def HF_DATASET_CONFIGS(self):
            return [lang]

    return LegalMC4LanguageDataset


def get_legal_mc4_auto_classes():
    """
    Auto generate dataset classes with token count
    """
    lang_to_tokens = {row.split("\t")[0]: int(row.split("\t")[1]) for row in RAW_LANG_TO_TOKENS.splitlines()}

    return [get_legal_mc4_auto_cls_by_language(lang, tokens) for lang, tokens in lang_to_tokens.items()]
