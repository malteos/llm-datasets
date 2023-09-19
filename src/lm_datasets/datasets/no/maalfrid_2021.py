from lm_datasets.datasets.base import BaseDataset, BILLION


class Maalfrid2021Dataset(BaseDataset):
    DATASET_ID = "maalfrid_2021"
    TITLE = "Målfrid 2021 - Freely Available Documents from Norwegian State Institutions"
    HOMEPAGE = "https://hdl.handle.net/21.11146/69"
    AVAILIBILITY = "Yes - it has a direct download link or links"

    LANGUAGES = ["nb", "nn", "en", "se", "other"]

    DESCRIPTION = (
        "This corpus consists of documents from 339 internet domains of Norwegian state institutions and comprises"
        " approximately 4.1 billion tokens in total, which makes it one of the largest freely available resources for"
        " Norwegian Bokmål and Nynorsk. In addition to Norwegian, the corpus contains texts in Northern Sami, Lule"
        " Sami, Southern Sami and English."
    )
    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = "open license"

    # Size: 4.1 billion tokens
    TOKENS = 4.1 * BILLION

    DOWNLOAD_URLS = ["https://www.nb.no/sbfil/tekst/maalfrid_2021/maalfrid_2021.tar.gz"]
    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/no/maalfrid_2021"]
