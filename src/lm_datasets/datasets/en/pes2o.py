from lm_datasets.datasets.base import BILLION, Genre, License
from lm_datasets.datasets.hf_dataset import HFDataset


class PeS2oDataset(HFDataset):
    """
    peS2o V2

    Knowledge cutoff: 2023-01-03
    Number of documents: 38.97M
    Number of whitespace-separated tokens*: 42.01B

    """

    DATASET_ID = "pes2o"
    TITLE = "peS2o"
    DESCRIPTION = (
        "The peS2o dataset is a collection of ~40M creative open-access academic papers, cleaned, filtered, and"
        " formatted for pre-training of language models. It is derived from the Semantic Scholar Open Research"
        " Corpus(Lo et al, 2020), or S2ORC."
    )
    HOMEPAGE = "https://huggingface.co/datasets/allenai/peS2o"
    VERSION = "V2"
    LICENSE = License(
        name="Open Data Commons Attribution License (ODC-By) v1.0",
        url="https://opendatacommons.org/licenses/by/1-0/",
        attribution=True,
        commercial_use=True,
        research_use=True,
        sharealike=False,
    )
    LANGUAGES = ["en"]
    GENRES = [Genre.SCIENCE]

    TOKENS = 42.01 * BILLION

    HF_DATASET_ID = "allenai/peS2o"
    HF_DATASET_SPLIT = "train"
    HF_DATASET_CONFIGS = ["v2"]

    text_column = "text"
    streaming = True
