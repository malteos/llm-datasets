from lm_datasets.datasets.base import BaseDataset, MILLION


class SakspapirNNODataset(BaseDataset):
    DATASET_ID = "sakspapir_nno"
    TITLE = "Legal Documents from Norwegian Nynorsk Municipialities"
    HOMEPAGE = "https://hdl.handle.net/21.11146/60"
    AVAILIBILITY = "Yes - it has a direct download link or links"

    LANGUAGES = ["nn", "nb"]

    DESCRIPTION = (
        "The texts in this corpus have been collected with the web crawler Veidemann in collaboration with the National"
        " Library’s Web Archive, based on a revised list of municipalities from the National Association of Nynorsk"
        " Municipalities (see lnk.no)."
    )
    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = "open license"

    # Size: 127 million tokens
    TOKENS = 127 * MILLION

    DOWNLOAD_URLS = ["https://www.nb.no/sbfil/tekst/sakspapir_nno/sakspapir_nno_01.tar.gz"]
    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/no/sakspapir_nno"]

    DUMMY = True
