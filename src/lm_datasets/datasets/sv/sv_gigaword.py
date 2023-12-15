from io import BytesIO
import logging

import bz2

import tarfile

from lm_datasets.datasets.base import BaseDataset, Availability, BILLION, QualityWarning, License


logger = logging.getLogger(__name__)


class SVGigawordDataset(BaseDataset):
    DATASET_ID = "sv_gigaword"
    TITLE = "The Swedish Culturomics Gigaword Corpus"
    HOMEPAGE = "https://spraakbanken.gu.se/en/resources/gigaword"
    DESCRIPTION = (
        "One billion Swedish words from 1950 and onwards. Code to extract data from the corpus, as well as usage"
        " instructions, can be downloaded from https://svn.spraakdata.gu.se/sb-arkiv/tools/gigaword/"
    )

    LANGUAGES = ["sv"]
    TOKENS = 1 * BILLION

    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LICENSE = License(
        "BY-SA 4.0",
        attribution=True,
        sharealike=False,
        commercial_use=True,
        research_use=True,
        url="http://creativecommons.org/licenses/by-sa/4.0/",
    )

    QUALITY_WARNINGS = [QualityWarning.BAD_WHITESPACES]

    DOWNLOAD_URLS = [
        "https://spraakbanken.gu.se/lb/resurser/meningsmangder/gigaword-1950-59.tar",
        "https://spraakbanken.gu.se/lb/resurser/meningsmangder/gigaword-1960-69.tar",
        "https://spraakbanken.gu.se/lb/resurser/meningsmangder/gigaword-1970-79.tar",
        "https://spraakbanken.gu.se/lb/resurser/meningsmangder/gigaword-1980-89.tar",
        "https://spraakbanken.gu.se/lb/resurser/meningsmangder/gigaword-1990-99.tar",
        "https://spraakbanken.gu.se/lb/resurser/meningsmangder/gigaword-2000-09.tar",
        "https://spraakbanken.gu.se/lb/resurser/meningsmangder/gigaword-2010-15.tar",
    ]

    def is_downloaded(self):
        return len(self.get_dataset_file_paths(needed_suffix=".tar")) == len(self.DOWNLOAD_URLS)

    def get_texts(self):
        from lxml import etree

        if not self.is_downloaded():
            self.download()

        # read from tar files
        tar_file_paths = self.get_dataset_file_paths(needed_suffix=".tar")
        for tar_fp in tar_file_paths:
            logger.info("Extracting from %s", tar_fp)

            with tarfile.open(tar_fp) as tar_f:
                members = [m for m in tar_f.getmembers() if m.name.endswith(".xml.bz2")]

                # Read from XML bz2 files
                for member in members:
                    logger.info(f"Decompress: {member.name}")

                    decompressed_member = bz2.decompress(tar_f.extractfile(member).read())

                    logger.info(f"Parse: {member.name}")

                    # Iterate over "text" tags
                    for event, element in etree.iterparse(BytesIO(decompressed_member), tag="text"):
                        text = ""
                        # TODO: whitespaces before punctutation
                        for el in element.itertext():
                            text += el.replace("\n", " ")
                        text = text.replace("  ", " ").strip()

                        yield text

                        # Free up memory as describe in:
                        # https://stackoverflow.com/questions/12160418/why-is-lxml-etree-iterparse-eating-up-all-my-memory  # noqa
                        element.clear()

                        # Also eliminate now-empty references from the root node to elem
                        for ancestor in element.xpath("ancestor-or-self::*"):
                            while ancestor.getprevious() is not None:
                                del ancestor.getparent()[0]

                        # print(text)
