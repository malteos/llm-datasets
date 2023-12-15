import json
from pathlib import Path
from lm_datasets.datasets.base import BaseDataset, Availability, License

from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)


class KorpusMaltiDataset(BaseDataset):
    DATASET_ID = "korpus_malti"

    TITLE = "Korpus Malti"
    HOMEPAGE = "https://huggingface.co/datasets/MLRS/korpus_malti"
    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD

    LANGUAGES = ["mt"]

    DESCRIPTION = (
        "General Corpora for the Maltese Language. This dataset is composed of texts ",
        "from various genres/domains written in Maltese.",
    )

    PII = "I have not checked the data source for personally identifiable or sensitive information."

    LICENSE = License(
        "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (DFKI has a permission for LLM training with commercial license)",  # noqa
        url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        commercial_use=False,
        attribution=True,
        sharealike=True,
        research_use=True,
    )

    # From paper https://aclanthology.org/2022.deeplo-1.10.pdf (Table 1)
    TOKENS = 466_601_373 - 98_582_031 - 1_885_661  # full dataset - law_eu - wiki

    HF_DATASET_ID = "MLRS/korpus_malti"
    HF_DATASET_SPLIT = "train"
    HF_DATASET_CONFIGS = [
        "belles_lettres",
        "blogs",
        "comics",
        "court",
        "eu_docs",
        "gov_docs",
        "government_gazzette",
        # "law_eu" # avoid duplication with EUR-LEx
        "law_mt",
        "legal",
        "nonfiction",
        "parliament",
        "press_eu",
        "press_mt",
        "speeches",
        "theses",
        "umlib_oar",
        "web_general",
        # "wiki",  # avoid duplication with Wiki
    ]

    def download(self):
        """
        git clone --depth 1 https://$HF_LOGIN:$HF_PASSWORD@huggingface.co/datasets/MLRS/korpus_malti
        """
        pass

    def get_texts(self):
        dataset_dir = self.get_local_dataset_dir()

        for subset in self.HF_DATASET_CONFIGS:
            subset_path = Path(dataset_dir) / "data" / subset
            # logger.info(f"")
            fps = list(subset_path.glob("*.jsonl"))
            for fp in tqdm(fps, desc=f"Reading subset: {subset}"):
                with open(fp) as f:
                    # logger.info(f"Reading from {fp}")
                    for line in f:
                        item = json.loads(line)
                        text = self.sentence_delimiter.join(item["text"])

                        yield text
