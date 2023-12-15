import logging
import os
from lm_datasets.datasets.base import BaseDataset, MB, Availability, License
from lm_datasets.utils.systems import get_path_by_system

import wget

logger = logging.getLogger(__name__)


class GAUniversalDependenciesDataset(BaseDataset):
    """

    TODO only sentences no documents


    """

    DATASET_ID = "ga_universal_dependencies"
    TITLE = "Irish Universal Dependencies"
    HOMEPAGE = "https://universaldependencies.org/"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    LICENSE = License(
        "mixed (CC BY-SA 3.0 or CC BY-SA 4.0)",
        attribution=True,
        sharealike=True,
        commercial_use=True,
        research_use=True,
    )
    LANGUAGES = ["ga"]

    DOWNLOAD_URLS = [
        "https://github.com/UniversalDependencies/UD_Irish-IDT/raw/r2.12/ga_idt-ud-train.conllu",
        "https://github.com/UniversalDependencies/UD_Irish-IDT/raw/r2.12/ga_idt-ud-dev.conllu",
        "https://github.com/UniversalDependencies/UD_Irish-IDT/raw/r2.12/ga_idt-ud-test.conllu",
        "https://github.com/UniversalDependencies/UD_Irish-Cadhan/raw/r2.12/ga_cadhan-ud-test.conllu",
        "https://github.com/UniversalDependencies/UD_Irish-TwittIrish/raw/r2.12/ga_twittirish-ud-train.conllu",
        "https://github.com/UniversalDependencies/UD_Irish-TwittIrish/raw/r2.12/ga_twittirish-ud-dev.conllu",
        "https://github.com/UniversalDependencies/UD_Irish-TwittIrish/raw/r2.12/ga_twittirish-ud-test.conllu",
    ]

    USED_BY = ["gaBERT"]

    BYTES = 9.7 * MB

    def is_downloaded(self):
        return len(self.get_dataset_file_paths(needed_suffix=".conllu")) == len(self.DOWNLOAD_URLS)

    def get_texts(self):
        import conllu

        if not self.is_downloaded():
            self.download()

        # Parse CONLL files and extract sentences
        for fp in self.get_dataset_file_paths(needed_suffix=".conllu"):
            logger.info(f"Reading {fp}")

            with open(fp) as f:
                data = f.read()

            sentences = conllu.parse(data)

            for sent in sentences:
                text = sent.metadata["text"]

                yield text
