from lm_datasets.datasets.base import BILLION, Genre, License
from lm_datasets.datasets.hf_dataset import HFDataset


class ProofPileDataset(HFDataset):
    """
    Duplicated content / overlap with arxiv: PeS2oDataset
    """

    DATASET_ID = "proof_pile"
    TITLE = "proof-pile"
    DESCRIPTION = (
        "The proof-pile is a 13GB pre-training dataset of mathematical text that comprises 8.3 billion tokens (using"
        " the gpt-neox tokenizer). Models trained on this dataset are coming soon :) The dataset is composed of diverse"
        " sources of both informal and formal mathematics, namely"
    )
    HOMEPAGE = "https://huggingface.co/datasets/hoskinson-center/proof-pile"
    LICENSE = License("Apache 2.0 (probably code license instead of data license)")
    LANGUAGES = ["en"]
    GENRES = [Genre.SCIENCE, Genre.MATH]

    TOKENS = 8.3 * BILLION
    HAS_OVERLAP_WITH = ["proof_pile_2", "math_amps", "pes2o"]

    HF_DATASET_ID = "hoskinson-center/proof-pile"
    HF_DATASET_SPLIT = "train"

    text_column_name = "text"
