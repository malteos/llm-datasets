from lm_datasets.datasets.base import BaseDataset, MILLION, Availability, License


class SakspapirNNODataset(BaseDataset):
    DATASET_ID = "sakspapir_nno"
    TITLE = "Legal Documents from Norwegian Nynorsk Municipialities"
    HOMEPAGE = "https://hdl.handle.net/21.11146/60"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["nn", "nb"]

    DESCRIPTION = (
        "The texts in this corpus have been collected with the web crawler Veidemann in collaboration with the National"
        " Library’s Web Archive, based on a revised list of municipalities from the National Association of Nynorsk"
        " Municipalities (see lnk.no)."
    )
    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = License("Creative_Commons-ZERO (CC-ZERO)", url="https://creativecommons.org/publicdomain/zero/1.0/")

    # Size: 127 million tokens
    TOKENS = 127 * MILLION

    DOWNLOAD_URLS = ["https://www.nb.no/sbfil/tekst/sakspapir_nno/sakspapir_nno_01.tar.gz"]

    DUMMY = True
