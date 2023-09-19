import json
import os
from lm_datasets.datasets.base import BILLION, Availability, BaseDataset
from lm_datasets.datasets.hf_dataset import HFDataset


class DanishGigawordDataset(HFDataset):
    DATASET_ID = "danish_gigaword"

    TITLE = "Danish GiagaWord"
    HOMEPAGE = "https://sprogteknologi.dk/dataset/danish-gigaword"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["da"]

    DESCRIPTION = (
        "A billion-word corpus of Danish text. Split into many sections, and covering many dimensions ",
        "of variation (spoken/written, formal/informal, modern/old, rigsdansk/dialect, and so on).",
        "",
        "The license is CC-BY 4.0, Creative Commons with Attribution. Owners: ITU; Leon Derczynski, Manuel R. Ciosici",
    )
    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = "public domain"

    # Size: 1 billion words
    TOKENS = 1 * BILLION

    HF_DATASET_ID = "DDSC/partial-danish-gigaword-no-twitter"
    HF_DATASET_SPLIT = "train"

    excluded_sources = {
        # exclude all Wikimedia sources to avoid duplicated content with the original Wikimedia dataset
        "wiki",
        "wikibooks",
        "wikisource",
    }

    text_column_name = "text"
    remove_columns = ["doc_id", "LICENSE", "uri", "date_built"]

    def get_filter_func(self):
        def filter_func(example):
            return example["source"] not in self.excluded_sources

        return filter_func


# deprecated
class DanishGigawordOriginalDataset(BaseDataset):
    """
    Dataset based on orignal source (not HF version)
    """

    DATASET_ID = "danish_gigaword"
    TITLE = "Danish GiagaWord"
    HOMEPAGE = "https://sprogteknologi.dk/dataset/danish-gigaword"
    AVAILIBILITY = "Yes - it has a direct download link or links"

    LANGUAGES = ["da"]

    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = "public domain"

    # Size: 1 billion words
    TOKENS = 1 * BILLION

    DOWNLOAD_URLS = ["https://itu.dk/research/dagw/dagw_v1.0-release.zip"]
    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/danish_gigaword"]

    def get_dataset_file_path(self):
        # TODO handle datasets with multiple filles
        dataset_dir = self.get_local_dataset_dir()
        files = os.listdir(dataset_dir)

        if len(files) == 1:
            return os.path.join(dataset_dir, files[0])
        else:
            raise ValueError(f"Cannot determine file path. Either none or multiple files in dataset dir: {files=}")

    def get_texts(self):
        import zipfile

        dataset_file_path = self.get_dataset_file_path()
        print(dataset_file_path)

        with zipfile.ZipFile(dataset_file_path, "r") as zf:
            # zf.printdir()
            zfns = zf.namelist()

            for fn in zfns:
                if "sektioner" in fn:
                    print(fn)

                    with zf.open(fn) as content_file:
                        for json_str in content_file:
                            doc = json.loads(json_str)
                            print(doc)

                            yield doc["text"]

                        # content = content_file.read()

                        # for l in myfile:
                        #     print(l)
                    # break
