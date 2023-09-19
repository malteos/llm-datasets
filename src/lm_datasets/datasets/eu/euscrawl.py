from lm_datasets.datasets.base import MILLION
from lm_datasets.datasets.hf_dataset import HFDataset


class EUSCrawlDataset(HFDataset):
    DATASET_ID = "euscrawl"
    TITLE = "EusCrawl"
    LANGUAGES = ["eu"]
    DESCRIPTION = (
        "EusCrawl (http://www.ixa.eus/euscrawl/) is a high-quality corpus for Basque comprising 12.5 million ",
        "documents and 423 million tokens, totalling 2.1 GiB of uncompressed text. EusCrawl was built using ad-hoc ",
        "scrapers to extract text from 33 Basque websites with high-quality content, resulting in cleaner text ",
        "compared to general purpose approaches.",
    )

    TOKENS = 423 * MILLION
    WEB_CRAWLED = True

    HF_DATASET_ID = "HiTZ/euscrawl"
    HF_DATASET_SPLIT = "train"

    text_column_name = "plain_text"
