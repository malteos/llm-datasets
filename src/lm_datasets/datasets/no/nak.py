import os
import tarfile
from lm_datasets.datasets.base import BaseDataset, BILLION, MILLION
from lm_datasets.systems import get_path_by_system


class NAKDataset(BaseDataset):
    DATASET_ID = "nak"
    TITLE = "Norwegian Newspaper Corpus"
    HOMEPAGE = "https://hdl.handle.net/21.11146/4"
    AVAILIBILITY = "Yes - it has a direct download link or links"

    LANGUAGES = ["nb", "nn"]

    DESCRIPTION = (
        "This version of Norwegian Newspaper Corpus contains text from 1998 to 2019. The corpus contains approximately"
        " 1,68 billion words for Norwegian Bokmål, and about 68 million words for Norwegian Nynorsk."
    )
    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = "non-commercial use"

    # Size: 1.68 billion words for Bokmål, 68 million words for Nynorsk
    TOKENS = 1.68 * BILLION + 68 * MILLION

    DOWNLOAD_URLS = [
        "https://www.nb.no/sbfil/tekst/nak_2019.tar",
        "https://www.nb.no/sbfil/tekst/nak_2018.tar",
        "https://www.nb.no/sbfil/tekst/nak_2017.tar",
        "https://www.nb.no/sbfil/tekst/nak_2016.tar",
        "https://www.nb.no/sbfil/tekst/nak_2015.tar",
        "https://www.nb.no/sbfil/tekst/nak_2014.tar",
        "https://www.nb.no/sbfil/tekst/nak_2013.tar",
        "https://www.nb.no/sbfil/tekst/nak_2012.tar",
    ]
    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/no/nak"]

    def get_dataset_file_paths(self):
        # TODO handle datasets with multiple filles
        dataset_dir = get_path_by_system(self.LOCAL_DIRS)
        fps = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
        fps = [fp for fp in fps if os.path.isfile(fp)]

        if len(fps) == 8:
            return fps
        else:
            raise ValueError(f"Cannot determine file path. Either none or wrong number of files in dataset dir: {fps=}")

    def decompress(self):
        dataset_dir = get_path_by_system(self.LOCAL_DIRS)
        dataset_file_paths = self.get_dataset_file_paths()

        decompressed_dir = os.path.join(dataset_dir, "decompressed")

        if not os.path.exists(decompressed_dir):
            os.makedirs(decompressed_dir)

        for fp in dataset_file_paths:
            with tarfile.open(fp) as file_handler:
                fdir, fn = os.path.split(fp)

                file_handler.extractall(path=os.path.join(decompressed_dir, fn))

    def extract_plaintext(self):
        self.decompress()

        return super().extract_plaintext()
