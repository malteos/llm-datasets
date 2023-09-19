from typing import List
from lm_datasets.datasets.base import BILLION, BaseDataset, Availability
from lm_datasets.utils import generate_texts_from_tab_columns_in_xml
from smart_open import open


class SLWaCWebDataset(BaseDataset):
    DATASET_ID = "slwac_web"
    HOMEPAGE = "http://nlp.ffzg.hr/resources/corpora/slwac/"
    DOWNLOAD_URLS = ["http://nlp.ffzg.hr/data/corpora/slwac2.0.gz"]
    LANGUAGES = ["sl"]

    TITLE = "slWaC web corpus"
    DESCRIPTION = (
        "slWaC is a web corpus collected from the .si top-level domain in 2011 and 2014. The corpus is"
        " tokenized and annotated with the lemma and the morphosyntax layer."
    )

    TOKENS = 1.2 * BILLION
    LICENSE = "open license"
    AVAILIBITIY = Availability.DIRECT_DOWNLOAD
    WEB_CRAWLED = True

    def parse_doc_lines(self, doc_lines: List[str]):
        # Parse a full p item

        doc_text = ""
        sent = ""
        for doc_line in doc_lines:
            if doc_line.startswith("</block") or doc_line.startswith("</p"):
                # Each block is a paragraph -> paragraph completed
                # yield doc_text.rstrip()  # remove last white space
                # doc_text = ""
                # return doc_text
                doc_text += self.paragraph_delimiter
            elif doc_line.startswith("<s "):
                # beginning of sentence
                sent = ""
            elif doc_line.startswith("</s>"):
                # parse sentence
                s = ""
                last_line = "<g/>"
                for sent_line in sent.splitlines():
                    sent_line = sent_line.strip()

                    if sent_line.startswith("<"):
                        pass
                    else:
                        if last_line != "<g/>":
                            s += " "

                        cols = sent_line.split("\t", 2)

                        if len(cols) > 1:  # is valid entry (CONLLU-like format)
                            original = cols[0]  # first column contains original string
                            s += original

                    last_line = sent_line

                # sentence parsed
                doc_text += s + self.sentence_delimiter
            else:
                sent += doc_line

        return doc_text

    def get_texts(self):
        # read from slwac2.0.gz
        fp = self.get_dataset_file_paths(single_file=True, needed_suffix=".gz")

        with open(fp) as f:
            yield from generate_texts_from_tab_columns_in_xml(
                f,
                doc_start="<text ",
                doc_end="</text>",
                sentence_delimiter=self.sentence_delimiter,
                paragraph_delimiter=self.paragraph_delimiter,
            )

        raise NotImplementedError
