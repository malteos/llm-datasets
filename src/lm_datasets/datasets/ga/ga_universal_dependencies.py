import logging
import os
from lm_datasets.datasets.base import BaseDataset, MB, Availability
from lm_datasets.systems import get_path_by_system

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

    def download(self):
        output_dir = get_path_by_system(self.LOCAL_DIRS)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_paths = []

        for url in self.DOWNLOAD_URLS:
            fn = url.split("/")[-1]
            fp = os.path.join(output_dir, fn)
            # try:
            out_filename = wget.download(url, out=fp)
            logger.info(f"Completed {out_filename}")

            file_paths.append(fp)
            # except HTTPError as e:
            #     logger.error(f"Error {e}")

        return file_paths

    def get_dataset_file_paths(self):
        # TODO handle datasets with multiple filles
        dataset_dir = get_path_by_system(self.LOCAL_DIRS)

        if not os.path.exists(dataset_dir):
            return []

        fps = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
        fps = [fp for fp in fps if os.path.isfile(fp) and fp.endswith(".conllu")]

        return fps

    def get_texts(self):
        import conllu

        file_paths = self.get_dataset_file_paths()

        if len(file_paths) != len(self.DOWNLOAD_URLS):
            file_paths = self.download()

        # Parse CONLL files and extract sentences
        for fp in file_paths:
            logger.info(f"Reading {fp}")

            with open(fp) as f:
                data = f.read()

            sentences = conllu.parse(data)

            for sent in sentences:
                text = sent.metadata["text"]

                yield text
