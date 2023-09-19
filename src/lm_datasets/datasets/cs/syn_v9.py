import logging

import lzma as xz

from lm_datasets.datasets.base import BaseDataset, Availability


logger = logging.getLogger(__name__)


class SynV9Dataset(BaseDataset):
    """ """

    DATASET_ID = "syn_v9"
    TITLE = "SYN v9: large corpus of written Czech"
    HOMEPAGE = "https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4635"
    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD

    LANGUAGES = ["cs"]

    LICENSE = "Academic Use"

    USED_BY = ["https://arxiv.org/abs/2103.13031"]

    TOKENS = 4_700_000_000

    def get_texts(self):
        filepath = self.get_dataset_file_paths(single_file=True, needed_suffix=".xz")

        logger.info(f"Reading from {filepath}")
        doc_counter = 0
        with xz.open(filepath, "rt", encoding="utf-8") as f_in:
            doc_lines = []

            # Read file line by line
            for i, line in enumerate(f_in):
                doc_lines.append(line)

                # Extract <doc> ... </doc>
                if line.startswith("<doc "):
                    doc_lines = [line]
                elif line.startswith("</doc>"):
                    # End of doc -> process
                    doc_counter += 1
                    doc_text = ""

                    # Parse a full <doc> item
                    sent = ""
                    for doc_line in doc_lines:
                        if doc_line.startswith("</block"):
                            # Each block is a paragraph -> paragraph completed
                            yield doc_text.rstrip()  # remove last white space
                            doc_text = ""
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
                            # WARNING: do not concatenate sentences to a document since
                            # they have been shuffled - only paragraph level.
                            doc_text += s + " "

                        else:
                            sent += doc_line

        logger.info(f"Done. Processed {doc_counter:,} shuffled documents (and more paragraphs)")
