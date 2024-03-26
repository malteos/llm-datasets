from typing import List
from llm_datasets.datasets.base import BaseDataset, Genre, License, Availability


class SKCourtDecisionsDataset(BaseDataset):
    DATASET_ID = "sk_court_decisions"
    TITLE = "od-justice 2.0"
    HOMEPAGE = "https://www.juls.savba.sk/justicecorp.html"
    DOWNLOAD_URLS = ["https://www.juls.savba.sk/data/od-justice/od-justice-2.0.ver.xz"]
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    LICENSE = License("open data", url="https://obcan.justice.sk/opendata")
    LANGUAGES = ["sk"]
    GENRES = [Genre.LEGAL]
    DESCRIPTION = (
        "Slovak court decisions. The corpus is based on data made available by the "
        "Ministry of Justice of the Slovak Republic."
    )
    PII = "No"

    TOKENS = 10_618_105_036

    def parse_doc_lines(self, doc_lines: List[str]):
        # Parse a full <doc> item
        # extract URL from doc_lines[0]:
        # <doc url="https://obcan.justice.sk/content/public/item/48712d97-58f9-4033-ab41-313d0e8e6631" court="Okresný súd Stará Ľubovňa" zn="4Ps/4/2013" date="2014-01-13" tokcountdd="1422" tokcount="1422" >\  # noqa
        # print("".join(doc_lines))

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
        import lzma as xz

        filepath = self.get_dataset_file_paths(needed_suffix=".xz", single_file=True)

        with xz.open(open(filepath, "rb"), "rt", encoding="utf-8") as f_in:
            doc_lines = []
            doc_counter = 0

            # Read file line by line
            for i, line in enumerate(f_in):
                # line = line.strip()

                doc_lines.append(line)

                # Extract <doc> ... </doc>
                if line.startswith("<doc "):
                    doc_lines = [line]
                elif line.startswith("</doc>"):
                    # End of doc -> process
                    doc_counter += 1

                    text = self.parse_doc_lines(doc_lines)

                    if text:
                        yield text
