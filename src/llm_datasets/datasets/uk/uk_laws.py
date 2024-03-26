import logging
from smart_open import open

from llm_datasets.datasets.base import Availability, BaseDataset, License


logger = logging.getLogger(__name__)


class UKLawsDataset(BaseDataset):
    DATASET_ID = "uk_laws"
    TITLE = "Corpus of laws and legal acts of Ukraine"
    HOMEPAGE = "https://lang.org.ua/en/corpora/#anchor7"
    DESCRIPTION = "A large (more than 9 Gb) corpus of laws and legal acts of Ukraine."
    LANGUAGES = ["uk"]
    TOKENS = 578_988_264

    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LICENSE = License("Unknown, likely public domain")

    DOWNLOAD_URLS = ["https://lang.org.ua/static/downloads/corpora/laws.txt.tokenized.bz2"]

    def is_downloaded(self):
        return len(self.get_dataset_file_paths(needed_suffix=".tokenized.bz2")) == len(self.DOWNLOAD_URLS)

    def get_texts(self):
        if not self.is_downloaded():
            self.download()

        file_path = self.get_dataset_file_paths(needed_suffix=".tokenized.bz2", single_file=True)

        logger.info("Reading from %s", file_path)

        with open(file_path) as f:
            text = ""
            for i, line in enumerate(f):
                # print(line)
                # print("'", line, "'")

                if line.strip().isnumeric():
                    # new doc
                    # print(text)
                    yield text.strip()

                    text = ""
                else:
                    text += line
