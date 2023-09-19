from lm_datasets.datasets.base import BaseDataset, BILLION


class ParliamentaryProceedingsDataset(BaseDataset):
    DATASET_ID = "parliamentary_proceedings"
    TITLE = "Norwegian Parliamentary Proceedings 1814-2000"
    HOMEPAGE = "https://hdl.handle.net/21.11146/74"
    AVAILIBILITY = "Yes - it has a direct download link or links"

    LANGUAGES = ["no", "nb", "nn"]

    DESCRIPTION = (
        "This corpus contains published historical proceedings from the Norwegian parliament 1814-2000. A total of 2136"
        " volumes were digitized, OCR-read and processed at the National Library of Norway, and made available online"
        " in 2014."
    )
    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = "open license"

    # Size: 1.5 billion tokens
    TOKENS = 1.5 * BILLION

    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/no/parliamentary_proceedings"]

    DUMMY = True
