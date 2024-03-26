import logging

import lzma as xz

from llm_datasets.datasets.base import BaseDataset, Availability, License


logger = logging.getLogger(__name__)


class SynV9Dataset(BaseDataset):
    DATASET_ID = "syn_v9"
    TITLE = "SYN v9: large corpus of written Czech"
    HOMEPAGE = "https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4635"
    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD
    LANGUAGES = ["cs"]
    DESCRIPTION = "Corpus of contemporary written (printed) Czech sized 4.7 GW (i.e. 5.7 billion tokens). It covers mostly the 1990-2019 period and features rich metadata including detailed bibliographical information, text-type classification etc. SYN v9 contains a wide variety of text types (fiction, non-fiction, newspapers), but the newspapers prevail noticeably. "
    LICENSE = License(
        "Academic Use - Czech National Corpus (Shuffled Corpus Data)",
        url="https://lindat.mff.cuni.cz/repository/xmlui/page/license-cnc",
        research_use=True,
        commercial_use=False,
        attribution=True,
        distribution=False,
    )
    CITATION = r"""@misc{11234/1-4635,
    title = {{SYN} v9: large corpus of written Czech},
    author = {K{\v r}en, Michal and Cvr{\v c}ek, V{\'a}clav and Heny{\v s}, Jan and Hn{\'a}tkov{\'a}, Milena and Jel{\'{\i}}nek, Tom{\'a}{\v s} and Kocek, Jan and Kov{\'a}{\v r}{\'{\i}}kov{\'a}, Dominika and K{\v r}ivan, Jan and Mili{\v c}ka, Ji{\v r}{\'{\i}} and Petkevi{\v c}, Vladim{\'{\i}}r and Proch{\'a}zka, Pavel and Skoumalov{\'a}, Hana and {\v S}indlerov{\'a}, Jana and {\v S}krabal, Michal},
    url = {http://hdl.handle.net/11234/1-4635},
    note = {{LINDAT}/{CLARIAH}-{CZ} digital library at the Institute of Formal and Applied Linguistics ({{\'U}FAL}), Faculty of Mathematics and Physics, Charles University},
    copyright = {Czech National Corpus (Shuffled Corpus Data)},
    year = {2021} }"""

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
