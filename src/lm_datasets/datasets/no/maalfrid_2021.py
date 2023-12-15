from lm_datasets.datasets.base import BaseDataset, BILLION, Availability, License


class Maalfrid2021Dataset(BaseDataset):
    DATASET_ID = "maalfrid_2021"
    TITLE = "Målfrid 2021 - Freely Available Documents from Norwegian State Institutions"
    HOMEPAGE = "https://hdl.handle.net/21.11146/69"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    LICENSE = License(
        "Norwegian Licence for Open Government Data (NLOD)", url="https://data.norge.no/nlod/en/2.0", attribution=True
    )
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

    DUMMY = True
    HAS_OVERLAP_WITH = ["norwegian_cc"]
