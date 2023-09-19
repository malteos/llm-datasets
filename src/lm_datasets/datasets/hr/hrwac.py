import logging
from lm_datasets.datasets.base import BaseDataset, Availability

import html

from smart_open import open


logger = logging.getLogger(__name__)


class HRWACDataset(BaseDataset):
    """
    TODO only paragraphs no documents

    curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1064{/hrWaC2.1.01.xml.gz,/hrWaC2.1.02.xml.gz,/hrWaC2.1.03.xml.gz,/hrWaC2.1.04.xml.gz,/hrWaC2.1.05.xml.gz,/hrWaC2.1.05.xml.gz,/hrWaC2.1.06.xml.gz,/hrWaC2.1.07.xml.gz,/hrWaC2.1.08.xml.gz,/hrWaC2.1.09.xml.gz,/hrWaC2.1.10.xml.gz,/hrWaC2.1.11.xml.gz,/hrWaC2.1.12.xml.gz,/hrWaC2.1.13.xml.gz,/hrWaC2.1.14.xml.gz}  # noqa
    """

    DATASET_ID = "hrwac"
    TITLE = "Croatian web corpus hrWaC 2.1"
    HOMEPAGE = "http://nlp.ffzg.hr/resources/corpora/hrwac/"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["hr"]

    DOWNLOAD_URLS = [
        f"https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1064/hrWaC2.1.{i:02d}.xml.gz"
        for i in range(1, 14)
    ]
    # LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/hr/hrWaC2.1"]

    TOKENS = 1_397_757_548

    def get_paragraphs(self, file_path: str, needed_language: str, min_length: int = 250):
        lang = None
        paragraph_i = 0

        with open(file_path, encoding="utf-8") as f:
            paragraph = ""
            sentence = ""
            last_line = ""

            for line in f:
                line = line.strip()
                # print(line)

                if line.startswith("<p"):
                    # print(line)
                    if 'lang="sr"' in line:
                        lang = "sr"
                    elif 'lang="hr"' in line:  # TODO
                        lang = "hr"
                    else:
                        raise ValueError("Cannot determine language")

                    paragraph = ""
                elif line == "</p>":
                    # end of paragraph
                    lang = None
                    # print(f"======={paragraph}")
                    paragraph = paragraph.strip()

                    if len(paragraph) > min_length:
                        paragraph_i += 1
                        yield paragraph

                        if paragraph_i > self.limit and self.limit > 0:
                            logger.warning("Limit reached")
                            break

                elif needed_language == lang:
                    if line == "<s>":
                        # empty = True
                        sentence = ""
                    elif line == "</s>":
                        # end of sentence
                        # print(f"======={sentence}")

                        paragraph += sentence

                        # if not empty:
                        #     print("")
                    # elif line == "<g/>":
                    #     whitespace = False
                    elif line.startswith("<"):
                        pass
                    else:
                        # empty = False
                        decoded = html.unescape(line)  # .decode("utf-8")
                        original, diacritic, lemma, pos = decoded.split("\t")
                        # word = (
                        #     diacritic[0].upper() + diacritic[1:] + "\t" + pos.rstrip() + "\t" + lemma
                        #     if lemma[0].isupper()
                        #     else diacritic + "\t" + pos.rstrip() + "\t" + lemma
                        # )

                        if last_line != "<g/>":
                            sentence += " "

                        sentence += original

                last_line = line

    def is_downloaded(self):
        return len(self.get_dataset_file_paths(needed_suffix=".xml.gz")) == len(self.DOWNLOAD_URLS)

    def get_texts(self):
        needed_language = "hr"  # TODO make sr dataset

        if not self.is_downloaded():
            self.download()

        file_paths = self.get_dataset_file_paths(needed_suffix=".xml.gz")

        # Parse XML files and extract paragraphs
        for i, fp in enumerate(file_paths):
            logger.info(f"Reading {fp}")

            yield from self.get_paragraphs(file_path=fp, needed_language=needed_language)
