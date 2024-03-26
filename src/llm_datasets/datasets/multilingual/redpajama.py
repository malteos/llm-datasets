import os
from llm_datasets.datasets.base import Availability, BILLION, License
from llm_datasets.datasets.hf_dataset import HFDataset


class RedPajamaBaseDataset(HFDataset):
    """
    # full dataset
    wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'
    while read line; do
        dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
        mkdir -p $(dirname $dload_loc)
        wget "$line" -O "$dload_loc"
    done < urls.txt

    # filter urls (books, arxiv, stackexnchange)
    cat urls.txt | grep -E "book|arxiv|stackexchange" > filtered_urls.txt
    while read line; do
        dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
        mkdir -p $(dirname $dload_loc)
        wget "$line" -O "$dload_loc"
    done < filtered_urls.txt

    # checksums
    wget https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/arxiv_SHA256SUMS.txt
    wget https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/book_SHA256SUMS.txt
    wget https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/c4_SHA256SUMS.txt
    wget https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/common_crawl_SHA256SUMS.txt
    wget https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/github_SHA256SUMS.txt
    wget https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/stackexchange_SHA256SUMS.txt
    wget https://data.together.xyz/redpajama-data-1T/v1.0.0/sha256/wikipedia_SHA256SUMS.txt

    sha256sum --check stackexchange_SHA256SUMS.txt
    sha256sum --check book_SHA256SUMS.txt
    sha256sum --check arxiv_SHA256SUMS.txt

    """

    SOURCE_ID = "redpajama"

    TITLE = "RedPajama-Data T1 (selected subsets)"
    HOMEPAGE = "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    DESCRIPTION = """An Open Source Recipe to Reproduce LLaMA training dataset"""
    CITATION = r"""@software{together2023redpajama,
        author = {Together Computer},
        title = {RedPajama: An Open Source Recipe to Reproduce LLaMA training dataset},
        month = April,
        year = 2023,
        url = {https://github.com/togethercomputer/RedPajama-Data}
        }"""

    HF_DATASET_ID = "togethercomputer/RedPajama-Data-1T"
    HF_DATASET_SPLIT = "train"

    text_column_name = "text"

    def download(self):
        # set download diretory
        os.environ["RED_PAJAMA_DATA_DIR"] = self.get_local_dataset_dir()

        super().download()


class RedPajamaBookDataset(RedPajamaBaseDataset):
    DATASET_ID = "redpajama_book"
    HF_DATASET_CONFIGS = ["book"]
    TOKENS = 26 * BILLION
    LICENSE = License("partially copyrighted/pirated", commercial_use=False, research_use=False, distribution=False)
    LANGUAGES = ["en"]


class RedPajamaArxivDataset(RedPajamaBaseDataset):
    DATASET_ID = "redpajama_arxiv"
    HF_DATASET_CONFIGS = ["arxiv"]
    TOKENS = 28 * BILLION

    LICENSE = License("mixed arxiv licenses", research_use=True)
    LANGUAGES = ["en"]

    HAS_OVERLAP_WITH = ["pes2o"]


class RedPajamaStackexchangeDataset(RedPajamaBaseDataset):
    DATASET_ID = "redpajama_stackexchange"
    HF_DATASET_CONFIGS = ["stackexchange"]
    TOKENS = 20 * BILLION
    LICENSE = License(
        "cc-by-sa 4.0",
        url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution=True,
        commercial_use=True,
        research_use=True,
        sharealike=False,
    )
    LANGUAGES = ["en"]


# not implemented: commoncrawl, c4, wikipedia, github
# due to overlap with other datasets
