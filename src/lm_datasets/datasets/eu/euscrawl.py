from lm_datasets.datasets.base import MILLION, License
from lm_datasets.datasets.hf_dataset import HFDataset


class EUSCrawlDataset(HFDataset):
    DATASET_ID = "euscrawl"
    SOURCE_ID = "euscrawl"
    TITLE = "EusCrawl"
    LANGUAGES = ["eu"]
    DESCRIPTION = (
        "EusCrawl (http://www.ixa.eus/euscrawl/) is a high-quality corpus for Basque comprising 12.5 million ",
        "documents and 423 million tokens, totalling 2.1 GiB of uncompressed text. EusCrawl was built using ad-hoc ",
        "scrapers to extract text from 33 Basque websites with high-quality content, resulting in cleaner text ",
        "compared to general purpose approaches.",
    )
    LICENSE = License(
        "mixed (see Tab. 2 in paper, e.g., CC-BY-NC-ND, CC-BY-NC-SA)", url="https://arxiv.org/pdf/2203.08111.pdf"
    )
    TOKENS = 423 * MILLION
    WEB_CRAWLED = True

    HF_DATASET_ID = "HiTZ/euscrawl"
    HF_DATASET_SPLIT = "train"

    text_column_name = "plain_text"

    HAS_OVERLAP_WITH = ["wiki_eu"]


class EUSCrawlFilteredDataset(EUSCrawlDataset):
    DATASET_ID = "euscrawl_filtered"
    TITLE = "EusCrawl (filtered: no Wikipedia, no NC-licenses)"
    LICENSE = License(
        "CC-BY-SA",
        url="https://huggingface.co/datasets/HiTZ/euscrawl",
        commercial_use=True,
        attribution=True,
        sharealike=False,
        research_use=True,
    )

    def get_filter_func(self):
        def filter_func(example):
            return example["source"] != "wikipedia" and "nc" not in example["license"]

        return filter_func
