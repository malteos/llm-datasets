from lm_datasets.datasets.base import BaseDataset, MILLION


class ParlaMintDataset(BaseDataset):
    DATASET_ID = "parlamint"
    TITLE = "Norwegian part of the Parlamint corpus"
    HOMEPAGE = "https://hdl.handle.net/21.11146/77"
    AVAILIBILITY = "Yes - it has a direct download link or links"

    LANGUAGES = ["no", "nn", "nb"]

    DESCRIPTION = (
        "ParlaMint-NO contains the Norwegian part of the ParlaMint project, an EU-funded project supported by CLARIN"
        " ERIC. The project’s aim is to create comparable and similarly annotated corpora of parliamentary meeting"
        " minutes."
    )
    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = "public domain"

    # Size: 975 million tokens
    TOKENS = 975 * MILLION

    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/no/parlamint"]

    DUMMY = True
