from io import BytesIO
import json
import logging
import argparse
import os
import tarfile
from typing import Iterable
from smart_open import open
import re
import gzip

from lm_datasets.datasets.base import BaseDataset, Availability, License


logger = logging.getLogger(__name__)


class CzechEnglishParallelDataset(BaseDataset):
    """
    This dataset is only using the Czech part of the original parallel corpus.
    """

    DATASET_ID = "cs_en_parallel"
    TITLE = "Czech-English Parallel Corpus 1.0 (CzEng 1.0)"
    HOMEPAGE = "http://hdl.handle.net/11234/1-1458"
    DESCRIPTION = (
        "CzEng 1.0 is the fourth release of a sentence-parallel Czech-English corpus compiled at the "
        "Institute of Formal and Applied Linguistics (ÚFAL) freely available for non-commercial research purposes. "
        "CzEng 1.0 contains 15 million parallel sentences (233 million English and 206 million Czech tokens) from "
        "seven different types of sources automatically annotated at surface and deep (a- and t-) layers of "
        "syntactic representation."
    )
    LICENSE = License(
        "Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0)",
        url="http://creativecommons.org/licenses/by-nc-sa/3.0/",
        research_use=True,
        commercial_use=False,
        sharealike=True,
    )
    LANGUAGES = ["cs"]

    DOWNLOAD_URLS = [
        "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1458/data-plaintext-format.tar"
    ]
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    def is_downloaded(self):
        return len(self.get_dataset_file_paths(needed_suffix=".tar")) == len(self.DOWNLOAD_URLS)

    def get_texts(self):
        """
        Extracts all texts from each file. Each file contains several texts, and one line per sentence.
        This function concatenates all sentences belonging to one text and then outputs the text.
        """
        if not self.is_downloaded():
            self.download()

        # read from tar files
        tar_file_paths = self.get_dataset_file_paths(needed_suffix=".tar")
        for tar_fp in tar_file_paths:
            logger.info(f"Extracting from {tar_fp}")

            with tarfile.open(tar_fp) as tar_f:
                members = [m for m in tar_f.getmembers() if m.name.endswith(".gz")]

                # Read from gz files
                for member in members:
                    logger.info(f"Decompress: {member.name}")

                    member_content = tar_f.extractfile(member).read()

                    with gzip.open(BytesIO(member_content), "rt") as inp:
                        lines = inp.readlines()
                        cs_texts = []
                        current_filename = ""
                        for line in lines:
                            rows = line.split("\t")
                            name = rows[0]
                            filename = re.sub("-s[0-9]+", "", name)
                            if current_filename == "":
                                current_filename = filename
                            elif current_filename != filename:
                                yield " ".join(cs_texts)
                                cs_texts = []
                                current_filename = filename
                            # score = rows[1]
                            cs_text = rows[2]
                            # en_text = rows[3]
                            cs_texts.append(cs_text.strip("\n"))

                        text = " ".join(cs_texts)
                        # print(text)
                        yield text
