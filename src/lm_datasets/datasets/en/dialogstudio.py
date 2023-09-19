from typing import Iterable
from lm_datasets.datasets.base import Genre
from lm_datasets.datasets.hf_dataset import HFDataset


DIALOGSTUDIO_DATASETS = {
    "natural_language_understanding": [
        "ATIS",
        "ATIS-NER",
        "BANKING77",
        "BANKING77-OOS",
        "CLINC-Single-Domain-OOS-banking",
        "CLINC-Single-Domain-OOS-credit_cards",
        "CLINC150",
        "DSTC8-SGD",
        "HWU64",
        "MIT-Movie",
        "MIT-Restaurant",
        "RESTAURANTS8K",
        "SNIPS",
        "SNIPS-NER",
        "TOP",
        "TOP-NER",
    ],
    "task_oriented": [
        "ABCD",
        "AirDialogue",
        "BiTOD",
        "CaSiNo",
        "CraigslistBargains",
        "Disambiguation",
        "DSTC2-Clean",
        "FRAMES",
        "GECOR",
        "HDSA-Dialog",
        "KETOD",
        "KVRET",
        "MetaLWOZ",
        "MS-DC",
        "MuDoCo",
        "MulDoGO",
        "MultiWOZ_2.1",
        "MULTIWOZ2_2",
        "SGD",
        "SimJointGEN",
        "SimJointMovie",
        "SimJointRestaurant",
        "STAR",
        "Taskmaster1",
        "Taskmaster2",
        "Taskmaster3",
        "WOZ2_0",
    ],
    "dialogue_summarization": [
        "AMI",
        "CRD3",
        "DialogSum",
        "ECTSum",
        "ICSI",
        "MediaSum",
        "QMSum",
        "SAMSum",
        "TweetSumm",
        "ConvoSumm",
        "SummScreen_ForeverDreaming",
        "SummScreen_TVMegaSite",
    ],
    "conversational_recommendation": [
        "Redial",
        "DuRecDial-2.0",
        "OpenDialKG",
        "SalesBot",
    ],
    "open_domain": [
        "chitchat-dataset",
        "ConvAI2",
        "AntiScam",
        "Empathetic",
        "HH-RLHF",
        "PLACES3.5",
        "Prosocial",
        "SODA",
    ],
    "knowledge_grounded": [
        "CompWebQ",
        "CoQA",
        "CoSQL",
        "DART",
        "FeTaQA",
        "GrailQA",
        "HybridQA",
        "MTOP",
        "MultiModalQA",
        "SParC",
        "Spider",
        "SQA",
        "ToTTo",
        "WebQSP",
        "WikiSQL",
        "WikiTQ",
        "wizard_of_internet",
        "wizard_of_wikipedia",
    ],
}


class DialogstudioDataset(HFDataset):
    """
    Rather a fine-tuning dataset
    """

    DATASET_ID = "dialogstudio"
    TITLE = "DialogStudio: Unified Dialog Datasets and Instruction-Aware Models for Conversational AI"
    DESCRIPTION = (
        "The proof-pile is a 13GB pre-training dataset of mathematical text that comprises 8.3 billion tokens (using"
        " the gpt-neox tokenizer). Models trained on this dataset are coming soon :) The dataset is composed of diverse"
        " sources of both informal and formal mathematics, namely"
    )
    HOMEPAGE = "https://huggingface.co/datasets/Salesforce/dialogstudio"
    DOI = "10.48550/arXiv.2307.10172"

    LANGUAGES = ["en"]
    GENRES = [Genre.DIALOGUE]
    LICENSE = "mixed"

    HF_DATASET_ID = "Salesforce/dialogstudio"
    HF_DATASET_SPLIT = "train"

    text_column = "text"

    @property
    def HF_DATASET_CONFIGS(self):
        return [ds for datasets in DIALOGSTUDIO_DATASETS.values() for ds in datasets]

    def get_texts(self) -> Iterable[str]:
        # See https://github.com/salesforce/DialogStudio/tree/main
        raise NotImplementedError("Structured data needs to be converted into plaintext")
