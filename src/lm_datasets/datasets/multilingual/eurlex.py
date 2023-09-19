import os
from lm_datasets.datasets.base import Availability
from lm_datasets.datasets.jsonl_dataset import JSONLDataset

# From https://docs.google.com/spreadsheets/d/1_rfLKa_Kq09YI0BPnfmoSL6-U3SHrW8tRwmk3-Qchzo/edit#gid=1698256287
RAW_LANG_TO_TOKENS = """bg	398067053
cs	471961631
da	671484862
de	695512401
el	696216541
en	769465561
es	725125274
et	328068754
fi	404265224
fr	828959218
ga	65030095
hr	258816068
hu	375253894
it	768605772
lt	364361783
lv	363239195
mt	367834815
nl	770312808
pl	406648795
pt	675152149
ro	415038571
sk	392235510
sl	394814289
sv	500085970"""


class EURLexBaseDataset(JSONLDataset):
    """
    Read preprocessed JSONL files + token counts (see EULM repo: extract_text_eurlex.py)
    """

    SOURCE_ID = "eurlex"
    DESCRIPTION = "EurlexResources: A Corpus Covering the Largest EURLEX Resources."
    HOMEPAGE = "https://huggingface.co/datasets/joelito/eurlex_resources"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LOCAL_DIRS = ["pegasus:/netscratch/mostendorff/experiments/eulm/data/docs_by_language"]

    def get_raw_jsonl_paths(self):
        dataset_dir = self.get_local_dataset_dir()

        return [os.path.join(dataset_dir, lang, "eurlex.jsonl") for lang in self.LANGUAGES]


def get_eurlex_auto_cls_by_language(lang, tokens):
    class EURLexLanguageDataset(EURLexBaseDataset):
        TOKENS = tokens
        DATASET_ID = "eurlex_" + lang
        TITLE = f"EURLex ({lang} subset)"
        LANGUAGES = [lang]

    return EURLexLanguageDataset


def get_eurlex_auto_classes():
    """
    Generate automatically dataset classes with token count
    """
    pass

    lang_to_tokens = {row.split("\t")[0]: int(row.split("\t")[1]) for row in RAW_LANG_TO_TOKENS.splitlines()}

    return [
        get_eurlex_auto_cls_by_language(lang, tokens * 10)  # TODO token count like OSCAR based on ten CC dumps
        for lang, tokens in lang_to_tokens.items()
    ]
