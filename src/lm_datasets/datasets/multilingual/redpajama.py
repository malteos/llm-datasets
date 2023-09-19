import os
from lm_datasets.datasets.base import Availability, BILLION
from lm_datasets.datasets.hf_dataset import HFDataset


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

    TITLE = "RedPajama T1"
    HOMEPAGE = "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/multilingual/redpajama"]

    HF_DATASET_ID = "togethercomputer/RedPajama-Data-1T"
    HF_DATASET_SPLIT = "train"

    text_column_name = "text"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # set download diretory
        os.environ["RED_PAJAMA_DATA_DIR"] = self.get_local_dataset_dir()


class RedPajamaBookDataset(RedPajamaBaseDataset):
    DATASET_ID = "redpajama_book"
    HF_DATASET_CONFIGS = ["book"]
    TOKENS = 26 * BILLION

    LANGUAGES = ["en"]


class RedPajamaArxivDataset(RedPajamaBaseDataset):
    DATASET_ID = "redpajama_arxiv"
    HF_DATASET_CONFIGS = ["arxiv"]
    TOKENS = 28 * BILLION

    LANGUAGES = ["en"]


class RedPajamaStackexchangeDataset(RedPajamaBaseDataset):
    DATASET_ID = "redpajama_stackexchange"
    HF_DATASET_CONFIGS = ["stackexchange"]
    TOKENS = 20 * BILLION

    LANGUAGES = ["en"]


# not implemented: commoncrawl, c4, wikipedia, github
