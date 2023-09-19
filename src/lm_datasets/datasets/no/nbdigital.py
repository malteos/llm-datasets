from lm_datasets.datasets.base import BaseDataset, GB


class NBDigitalDataset(BaseDataset):
    DATASET_ID = "nbdigital"
    TITLE = "Public Domain Texts from NBdigital"
    HOMEPAGE = "https://hdl.handle.net/21.11146/34"
    AVAILIBILITY = "Yes - it has a direct download link or links"

    LANGUAGES = ["nb", "nn"]

    DESCRIPTION = (
        "This text collection consists of texts from the digital National Library which are in the public domain. The"
        " collection contains 26.344 books (and other written material) by 10756 different authors (including, e.g.,"
        " public institutions for publically available material)."
    )
    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = "public domain"

    # Size: 26344 books
    BYTES = 2.7 * GB

    DOWNLOAD_URLS = ["https://www.nb.no/sbfil/tekst/20150526_nbdig_txt01.tar.gz"]
    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/multilingual"]
