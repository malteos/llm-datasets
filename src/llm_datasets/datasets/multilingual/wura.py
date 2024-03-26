from llm_datasets.datasets.base import Availability, GB, License
from llm_datasets.datasets.hf_dataset import HFDataset


class WuraBaseDataset(HFDataset):
    SOURCE_ID = "wura"
    TITLE = "WURA"
    DESCRIPTION = """Wura is large-scale pretraining data for 20 languages popularly
        spoken in Africa."""
    HOMEPAGE = "https://huggingface.co/datasets/castorini/wura"
    LICENSE = License(
        name="Apache License Version 2.0",
        url="http://www.apache.org/licenses/LICENSE-2.0",
        commercial_use=True,
        research_use=True,
    )
    CITATION = r"""@inproceedings{oladipo-etal-2023-better,
        title = "Better Quality Pre-training Data and T5 Models for {A}frican Languages",
        author = "Oladipo, Akintunde  and
        Adeyemi, Mofetoluwa  and
        Ahia, Orevaoghene  and
        Owodunni, Abraham  and
        Ogundepo, Odunayo  and
        Adelani, David  and
        Lin, Jimmy",
        editor = "Bouamor, Houda  and
        Pino, Juan  and
        Bali, Kalika",
        booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
        month = dec,
        year = "2023",
        address = "Singapore",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.emnlp-main.11",
        pages = "158--168",
        abstract = "In this study, we highlight the importance of enhancing the quality of pretraining data in multilingual language models. Existing web crawls have demonstrated quality issues, particularly in the context of low-resource languages. Consequently, we introduce a new multilingual pretraining corpus for 16 African languages, designed by carefully auditing existing pretraining corpora to understand and rectify prevalent quality issues. To compile this dataset, we undertake a rigorous examination of current data sources for thirteen languages within one of the most extensive multilingual web crawls, mC4, and extract cleaner data through meticulous auditing and improved web crawling strategies. Subsequently, we pretrain a new T5-based model on this dataset and evaluate its performance on multiple downstream tasks. Our model demonstrates better downstream effectiveness over existing pretrained models across four NLP tasks, underscoring the critical role data quality plays in pretraining language models in low-resource scenarios. Specifically, on cross-lingual QA evaluation, our new model is more than twice as effective as multilingual T5. All code, data and models are publicly available at https://github.com/castorini/AfriTeVa-keji.",
    }
    """
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = [
        "en",
        "fr",
        "pt",
        "af",
        "am",
        "ha",
        "ar",
        "ig",
        "rw",
        "mg",
        "ny",
        "om",
        "sn",
        "so",
        "st",
        "sw",
        "ti",
        "xh",
        "yo",
        "zu",
    ]

    HF_DATASET_ID = "castorini/wura"
    HF_DATASET_CONFIGS = [
        "eng",
        "fra",
        "por",
        "afr",
        "amh",
        "hau",
        "arz",
        "ibo",
        "kin",
        "mlg",
        "nya",
        "orm",
        "sna",
        "som",
        "sot",
        "swa",
        "tir",
        "xho",
        "yor",
        "zul",
    ]
    HF_DATASET_SPLIT = ["train", "validation"]
    HAS_OVERLAP_WITH = ["mC4"]

    keep_columns = True
    text_column_name = "content"


class WuraENDataset(WuraBaseDataset):
    DATASET_ID = "wura_en"
    TITLE = "WURA [English]"

    LANGUAGES = ["en"]
    HF_DATASET_CONFIGS = ["eng"]


class WuraFRDataset(WuraBaseDataset):
    DATASET_ID = "wura_fr"
    TITLE = "WURA [French]"

    LANGUAGES = ["fr"]
    HF_DATASET_CONFIGS = ["fra"]


class WuraPTDataset(WuraBaseDataset):
    DATASET_ID = "wura_pt"
    TITLE = "WURA [Portuguese]"

    LANGUAGES = ["pt"]
    HF_DATASET_CONFIGS = ["por"]


class WuraAFDataset(WuraBaseDataset):
    DATASET_ID = "wura_af"
    TITLE = "WURA [Afrikaans]"

    LANGUAGES = ["af"]
    HF_DATASET_CONFIGS = ["afr"]


class WuraAMDataset(WuraBaseDataset):
    DATASET_ID = "wura_am"
    TITLE = "WURA [Amharic]"

    LANGUAGES = ["am"]
    HF_DATASET_CONFIGS = ["amh"]


class WuraHADataset(WuraBaseDataset):
    DATASET_ID = "wura_ha"
    TITLE = "WURA [Hausa]"

    LANGUAGES = ["ha"]
    HF_DATASET_CONFIGS = ["hau"]


class WuraARDataset(WuraBaseDataset):
    DATASET_ID = "wura_arz"
    TITLE = "WURA [Egyptian Arabic]"

    LANGUAGES = ["ar"]
    HF_DATASET_CONFIGS = ["arz"]


class WuraIGDataset(WuraBaseDataset):
    DATASET_ID = "wura_ig"
    TITLE = "WURA [Igbo]"

    LANGUAGES = ["ig"]
    HF_DATASET_CONFIGS = ["ibo"]


class WuraRWDataset(WuraBaseDataset):
    DATASET_ID = "wura_rw"
    TITLE = "WURA [Kinyarwanda]"

    LANGUAGES = ["rw"]
    HF_DATASET_CONFIGS = ["kin"]


class WuraMGDataset(WuraBaseDataset):
    DATASET_ID = "wura_mg"
    TITLE = "WURA [Kirghiz]"

    LANGUAGES = ["ky"]
    HF_DATASET_CONFIGS = ["kir"]


class WuraNYDataset(WuraBaseDataset):
    DATASET_ID = "wura_ny"
    TITLE = "WURA [Chichewa]"

    LANGUAGES = ["ny"]
    HF_DATASET_CONFIGS = ["nya"]


class WuraOMDataset(WuraBaseDataset):
    DATASET_ID = "wura_om"
    TITLE = "WURA [Oromo]"

    LANGUAGES = ["om"]
    HF_DATASET_CONFIGS = ["orm"]


class WuraSNDataset(WuraBaseDataset):
    DATASET_ID = "wura_sn"
    TITLE = "WURA [Shona]"

    LANGUAGES = ["sn"]
    HF_DATASET_CONFIGS = ["sna"]


class WuraSODataset(WuraBaseDataset):
    DATASET_ID = "wura_so"
    TITLE = "WURA [Somali]"

    LANGUAGES = ["so"]
    HF_DATASET_CONFIGS = ["som"]


class WuraSTDataset(WuraBaseDataset):
    DATASET_ID = "wura_st"
    TITLE = "WURA [Southern Sotho]"

    LANGUAGES = ["st"]
    HF_DATASET_CONFIGS = ["sot"]


class WuraSWDataset(WuraBaseDataset):
    DATASET_ID = "wura_sw"
    TITLE = "WURA [Swahili]"

    LANGUAGES = ["sw"]
    HF_DATASET_CONFIGS = ["swa"]


class WuraTIDataset(WuraBaseDataset):
    DATASET_ID = "wura_ti"
    TITLE = "WURA [Tigrinya]"

    LANGUAGES = ["ti"]
    HF_DATASET_CONFIGS = ["tir"]


class WuraXHDataset(WuraBaseDataset):
    DATASET_ID = "wura_xh"
    TITLE = "WURA [Xhosa]"

    LANGUAGES = ["xh"]
    HF_DATASET_CONFIGS = ["xho"]


class WuraYODataset(WuraBaseDataset):
    DATASET_ID = "wura_yo"
    TITLE = "WURA [Yoruba]"

    LANGUAGES = ["yo"]
    HF_DATASET_CONFIGS = ["yor"]


class WuraZUDataset(WuraBaseDataset):
    DATASET_ID = "wura_zu"
    TITLE = "WURA [Zulu]"

    LANGUAGES = ["zu"]
    HF_DATASET_CONFIGS = ["zul"]
