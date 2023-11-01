import logging
from typing import Iterable
from lm_datasets.datasets.base import BaseDataset, License

from smart_open import open


logger = logging.getLogger(__name__)


class ENC2021Dataset(BaseDataset):
    DATASET_ID = "enc2021"
    TITLE = "Estonian National Corpus 2021"
    DESCRIPTION = (
        "Corpus is based on Estonian National Corpus 2013, which was renewed by Lexical Computing Ltd. "
        "in 2017 and 2019 at the request of Estonian Language Institute."
        "Subcorpora are: Estonian Reference Corpus 1990-2008, Estonian Web 2013, Estonian Web 2017, "
        "Estonian Web 2019, Estonian Wikipedia 2017, Estonian Wikipedia 2019, Estonian Open Access Journals (DOAJ),"
        " blogs, discussion, education, fiction, food, health, journals, news, religion, science, sex, society, sports.",
    )
    HOMEPAGE = (
        "https://entu.keeleressursid.ee/shared/9939/EVKultjxSeFA2QhFkbE7fGGDGNT1zmJLOUGFK9hw53tq9Rx2YBTejI1IoKhy65zq"
    )
    LANGUAGES = ["et"]
    DOI = "10.5128/ERYa18.12"
    LICENSE = License(
        "Creative Commons Attribution-NonCommercial 4.0 International License",
        url="http://arhiiv.rakenduslingvistika.ee/ajakirjad/index.php/aastaraamat/article/view/ERYa18.12",
        attribution=True,
        commercial_use=False,
        research_use=True,
        sharealike=False,
        derivates=True,
    )
    DOWNLOAD_URLS = [
        # (
        #     "https://entu.keeleressursid.ee/api2/file-24747?key=EVKultjxSeFA2QhFkbE7fGGDGNT1zmJLOUGFK9hw53tq9Rx2YBTejI1IoKhy65zq",
        #     "nc19_Web_2019.prevert.gz",
        # ),
        (
            "https://entu.keeleressursid.ee/api2/file-24753?key=EVKultjxSeFA2QhFkbE7fGGDGNT1zmJLOUGFK9hw53tq9Rx2YBTejI1IoKhy65zq",  # noqa
            "nc21_Web_2021.prevert.gz",
        ),
        (
            "https://entu.keeleressursid.ee/api2/file-24738?key=EVKultjxSeFA2QhFkbE7fGGDGNT1zmJLOUGFK9hw53tq9Rx2YBTejI1IoKhy65zq",  # noqa
            "nc21_Feeds.prevert.gz",
        ),
        (
            "https://entu.keeleressursid.ee/api2/file-24720?key=EVKultjxSeFA2QhFkbE7fGGDGNT1zmJLOUGFK9hw53tq9Rx2YBTejI1IoKhy65zq",  # noqa
            "nc21_DOAJ.prevert.gz",
        ),
        (
            "https://entu.keeleressursid.ee/api2/file-24729?key=EVKultjxSeFA2QhFkbE7fGGDGNT1zmJLOUGFK9hw53tq9Rx2YBTejI1IoKhy65zq",  # noqa
            "nc21_Fiction.prevert.gz",
        ),
    ]

    def is_downloaded(self):
        return len(self.get_dataset_file_paths(needed_suffix=".prevert.gz")) == len(self.DOWNLOAD_URLS)

    def get_texts(self) -> Iterable[str]:
        from lm_datasets.io.prevert_file import PrevertFile

        if not self.is_downloaded():
            self.download()

        file_paths = self.get_dataset_file_paths(needed_suffix=".prevert.gz")

        for fp in file_paths:
            logger.info(f"Reading from {fp}")
            with open(fp) as f:
                dset = PrevertFile(f)

                for doc in dset:  # iterating through documents of a dataset
                    text = str(doc).replace("\n", " ")  # TODO replace line breaks with white spaces

                    yield text
