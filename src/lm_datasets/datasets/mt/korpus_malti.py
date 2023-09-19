import json
from pathlib import Path
from lm_datasets.datasets.base import BaseDataset, Availability
from lm_datasets.datasets.hf_dataset import HFDataset


from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)


class KorpusMaltiHFDataset(HFDataset):
    DATASET_ID = "korpus_malti"

    TITLE = "Korpus Malti"
    HOMEPAGE = "https://huggingface.co/datasets/MLRS/korpus_malti"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["mt"]

    DESCRIPTION = (
        "General Corpora for the Maltese Language. This dataset is composed of texts ",
        "from various genres/domains written in Maltese.",
    )

    PII = "I have not checked the data source for personally identifiable or sensitive information."

    LICENSE = "cc-by-nc-sa-4.0"  # DFKI has a permission for LLM training with commercial license

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
    # text_column_name = "text"
    # remove_columns = ["doc_id", "LICENSE", "uri", "date_built"]

    def get_text_from_item(self, item) -> str:
        sentences = item["text"]
        text = self.sentence_delimiter.join(sentences)

        return text

    # def get_texts(self):
    #     # load dataset for each config
    #     for config in self.dataset_configs:
    #         logger.info(f"Processing: {config}")
    #         self.datasets[config] = load_dataset(
    #             self.HF_DATASET_ID,
    #             config,
    #             streaming=False,
    #             use_auth_token=self.get_hf_auth_token(),
    #         )

    #         for sentences in self.datasets[config]["train"]["text"]:
    #             text = self.sentence_delimiter.join(sentences)

    #             yield text


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

    LICENSE = "cc-by-nc-sa-4.0"  # DFKI has a permission for LLM training with commercial license

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
