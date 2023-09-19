import logging
from typing import Iterable

import zipfile


from lm_datasets.datasets.base import BaseDataset, Availability, GB

logger = logging.getLogger(__name__)


class DKClarinDataset(BaseDataset):
    DATASET_ID = "dk_clarin"
    TITLE = "DK-CLARIN Reference Corpus of General Danish"
    HOMEPAGE = "https://korpus.dsl.dk/clarin/"

    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD
    LICENSE = "Academic Use; CLARIN-ACA-NC"

    LANGUAGES = ["da"]

    BYTES = 1.4 * GB

    def get_texts(self) -> Iterable[str]:
        """
        Extracts the text from the zip files
        """
        from bs4 import BeautifulSoup

        zip_fps = self.get_dataset_file_paths(needed_suffix=".zip")

        for zip_fp in zip_fps:
            logger.info(f"Extracting from {zip_fp}")

            with zipfile.ZipFile(zip_fp) as zf:
                needed_suffix = ".zip"
                members_fns = [fn for fn in zf.namelist() if fn.endswith(needed_suffix)]

                for member_fn in members_fns:
                    logger.info(f"---> Extracting from {member_fn}")

                    if "wikipedia" in member_fn:
                        logger.info(f"---> Skipping {member_fn} (exclude Wikipedia)")
                        continue

                    with zf.open(member_fn) as member_f:
                        with zipfile.ZipFile(member_f) as zf_2:
                            needed_suffix_2 = ".xml"
                            members_fns_2 = [fn for fn in zf_2.namelist() if fn.endswith(needed_suffix_2)]

                            for member_fn_2 in members_fns_2:
                                if "__MACOSX" in member_fn_2:
                                    logger.info(f"Skipping {member_fn_2}")
                                    continue

                                logger.info(f"Parsing from {member_fn_2}")

                                # TEIP5DKCLARIN-format
                                # p => paragraph
                                # w => word
                                # p => punctutation
                                # s => space
                                # type="s" => s

                                with zf_2.open(member_fn_2) as member_f_2:
                                    soup = BeautifulSoup(member_f_2, features="lxml")

                                paragraphs_texts = []
                                paragraphs = soup.select("p")

                                for i, p in enumerate(paragraphs):
                                    paragraph_text = ""
                                    for ele in p.find_all():
                                        paragraph_text += ele.get_text()
                                        attrs = ele.attrs

                                        if "type" in attrs and attrs["type"] == "s":
                                            paragraph_text += " "

                                    paragraphs_texts.append(paragraph_text)

                                title = soup.title.get_text().replace("CTB version: ", "")  # TODO strip "CTB version: "

                                if title == "nil":
                                    text = ""
                                else:
                                    text = title + self.title_delimiter

                                text += self.paragraph_delimiter.join(paragraphs_texts)

                                yield text

        logger.info("All ZIP files extracted")
