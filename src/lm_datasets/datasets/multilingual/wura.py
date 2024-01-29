from lm_datasets.datasets.base import Availability, GB, License
from lm_datasets.datasets.hf_dataset import HFDataset


class WuraBaseDataset(HFDataset):
    DATASET_ID = "wura"
    TITLE = "WURA"
    DESCRIPTION = (
        """Wura is large-scale pretraining data for 20 languages popularly
        spoken in Africa."""
    )
    HOMEPAGE = "https://huggingface.co/datasets/castorini/wura"
    LICENSE = License(
        name="Apache License Version 2.0",
        url="http://www.apache.org/licenses/LICENSE-2.0"
    )
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["en", "fr", "pt", "af", "am", "ha","ar", "ig", "rw", "mg", "ny",
                 "om", "sn", "so", "st", "sw", "ti", "xh", "yo", "zu"]

    HF_DATASET_ID = "castorini/wura"
    HF_DATASET_CONFIGS = ["eng", "fra", "por", "afr", "amh", "hau", "arz", "ibo",
                          "kin", "mlg", "nya", "orm", "sna", "som", "sot", "swa",
                          "tir", "xho", "yor", "zul"]
    HF_DATASET_SPLIT = ["train", "validation"]
    HAS_OVERLAP_WITH = ["mC4"]

    keep_columns = True
    text_column_name = "content"


class WuraENDataset(WuraBaseDataset):
    DATASET_ID = "wura_en"
    TITLE = "WURA English"
    SOURCE_ID = "wura"

    LANGUAGES = ["en"]
    HF_DATASET_CONFIGS = ["eng"]


class WuraFRDataset(WuraBaseDataset):
    DATASET_ID = "wura_fr"
    TITLE = "WURA French"
    SOURCE_ID = "wura"

    LANGUAGES = ["fr"]
    HF_DATASET_CONFIGS = ["fra"]


class WuraPTDataset(WuraBaseDataset):
    DATASET_ID = "wura_pt"
    TITLE = "WURA Portuguese"
    SOURCE_ID = "wura"

    LANGUAGES = ["pt"]
    HF_DATASET_CONFIGS = ["por"]


class WuraAFDataset(WuraBaseDataset):
    DATASET_ID = "wura_af"
    TITLE = "WURA Afrikaans"
    SOURCE_ID = "wura"

    LANGUAGES = ["af"]
    HF_DATASET_CONFIGS = ["afr"]


class WuraAMDataset(WuraBaseDataset):
    DATASET_ID = "wura_am"
    TITLE = "WURA Amharic"
    SOURCE_ID = "wura"

    LANGUAGES = ["am"]
    HF_DATASET_CONFIGS = ["amh"]


class WuraHADataset(WuraBaseDataset):
    DATASET_ID = "wura_ha"
    TITLE = "WURA Hausa"
    SOURCE_ID = "wura"

    LANGUAGES = ["ha"]
    HF_DATASET_CONFIGS = ["hau"]

class WuraARDataset(WuraBaseDataset):
    DATASET_ID = "wura_arz"
    TITLE = "WURA Egyptian Arabic"
    SOURCE_ID = "wura"

    LANGUAGES = ["ar"]
    HF_DATASET_CONFIGS = ["arz"]

class WuraIGDataset(WuraBaseDataset):
    DATASET_ID = "wura_ig"
    TITLE = "WURA Igbo"
    SOURCE_ID = "wura"

    LANGUAGES = ["ig"]
    HF_DATASET_CONFIGS = ["ibo"]


class WuraRWDataset(WuraBaseDataset):
    DATASET_ID = "wura_rw"
    TITLE = "WURA Kinyarwanda"
    SOURCE_ID = "wura"

    LANGUAGES = ["rw"]
    HF_DATASET_CONFIGS = ["kin"]


class WuraMGDataset(WuraBaseDataset):
    DATASET_ID = "wura_mg"
    TITLE = "WURA Kirghiz"
    SOURCE_ID = "wura"

    LANGUAGES = ["ky"]
    HF_DATASET_CONFIGS = ["kir"]


class WuraNYDataset(WuraBaseDataset):
    DATASET_ID = "wura_ny"
    TITLE = "WURA Chichewa"
    SOURCE_ID = "wura"

    LANGUAGES = ["ny"]
    HF_DATASET_CONFIGS = ["nya"]


class WuraOMDataset(WuraBaseDataset):
    DATASET_ID = "wura_om"
    TITLE = "WURA Oromo"
    SOURCE_ID = "wura"

    LANGUAGES = ["om"]
    HF_DATASET_CONFIGS = ["orm"]


class WuraSNDataset(WuraBaseDataset):
    DATASET_ID = "wura_sn"
    TITLE = "WURA Shona"
    SOURCE_ID = "wura"

    LANGUAGES = ["sn"]
    HF_DATASET_CONFIGS = ["sna"]


class WuraSODataset(WuraBaseDataset):
    DATASET_ID = "wura_so"
    TITLE = "WURA Somali"
    SOURCE_ID = "wura"

    LANGUAGES = ["so"]
    HF_DATASET_CONFIGS = ["som"]


class WuraSTDataset(WuraBaseDataset):
    DATASET_ID = "wura_st"
    TITLE = "WURA Southern Sotho"
    SOURCE_ID = "wura"

    LANGUAGES = ["st"]
    HF_DATASET_CONFIGS = ["sot"]


class WuraSWDataset(WuraBaseDataset):
    DATASET_ID = "wura_sw"
    TITLE = "WURA Swahili"
    SOURCE_ID = "wura"

    LANGUAGES = ["sw"]
    HF_DATASET_CONFIGS = ["swa"]


class WuraTIDataset(WuraBaseDataset):
    DATASET_ID = "wura_ti"
    TITLE = "WURA Tigrinya"
    SOURCE_ID = "wura"

    LANGUAGES = ["ti"]
    HF_DATASET_CONFIGS = ["tir"]


class WuraXHDataset(WuraBaseDataset):
    DATASET_ID = "wura_xh"
    TITLE = "WURA Xhosa"
    SOURCE_ID = "wura"

    LANGUAGES = ["xh"]
    HF_DATASET_CONFIGS = ["xho"]


class WuraYODataset(WuraBaseDataset):
    DATASET_ID = "wura_yo"
    TITLE = "WURA Yoruba"
    SOURCE_ID = "wura"

    LANGUAGES = ["yo"]
    HF_DATASET_CONFIGS = ["yor"]


class WuraZUDataset(WuraBaseDataset):
    DATASET_ID = "wura_zu"
    TITLE = "WURA Zulu"
    SOURCE_ID = "wura"

    LANGUAGES = ["zu"]
    HF_DATASET_CONFIGS = ["zul"]
