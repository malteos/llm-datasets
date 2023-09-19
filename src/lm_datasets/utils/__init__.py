import re
import logging
from typing import Iterable, List, Union


logger = logging.getLogger(__name__)


def remove_whitespaces_before_punctuation(input_text: str) -> str:
    # Taken from [1]
    # [1]: https://stackoverflow.com/questions/18878936/how-to-strip-whitespace-from-before-but-not-after-punctuation-in-python  # noqa
    return re.sub(r"\s([?.!,:](?:\s|$))", r"\1", input_text).replace("( ", "(").replace(" )", ")")


def get_text_from_tab_columns_in_xml(doc_lines: List[str], sentence_delimiter, paragraph_delimiter) -> str:
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
            doc_text += paragraph_delimiter
        elif doc_line.startswith("<s ") or doc_line.startswith("<s>"):
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
            doc_text += s + sentence_delimiter
        else:
            sent += doc_line

    # print(doc_text)
    # print(doc_lines[0])
    return doc_text


def generate_texts_from_tab_columns_in_xml(
    line_iterator: Iterable[str],
    doc_start: str = "<doc ",
    doc_end: str = "</doc>",
    sentence_delimiter: str = " ",
    paragraph_delimiter: str = "\n\n",
):
    doc_lines = []
    doc_counter = 0

    # Read file line by line
    for i, line in enumerate(line_iterator):
        # line = line.strip()

        doc_lines.append(line)

        # Extract <doc> ... </doc>
        if line.startswith(doc_start):
            doc_lines = [line]

        elif line.startswith(doc_end):
            # End of doc -> process
            doc_counter += 1

            text = get_text_from_tab_columns_in_xml(doc_lines, sentence_delimiter, paragraph_delimiter)

            if text:
                yield text


def get_auto_workers(workers: int) -> int:
    import multiprocessing

    if workers < 1:
        cpu_cores = multiprocessing.cpu_count()

        if workers == 0:
            workers = cpu_cores
            logger.info(f"Using all available CPU cores: {workers}")
        else:
            workers = cpu_cores - abs(workers)
            logger.info(f"Using workers: {workers} / (available cores {cpu_cores})")

    return workers


def get_parquet_compression(request_compression: str) -> str:
    supported_parquet_compressions = {"NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD"}

    if request_compression is None:
        return "NONE"
    elif request_compression in supported_parquet_compressions:
        return request_compression
    else:
        if request_compression.upper() in supported_parquet_compressions:
            return request_compression.upper()
        else:
            raise ValueError(
                f"Unsupported parquet compression: {request_compression} ({supported_parquet_compressions=})"
            )


def get_bytes_from_int_or_string(bytes_int_or_str: Union[int, str]) -> int:
    from lm_datasets.datasets.base import GB, KB, MB

    if isinstance(bytes_int_or_str, int):
        return bytes_int_or_str
    elif isinstance(bytes_int_or_str, str):
        if bytes_int_or_str.endswith("KB"):
            return int(bytes_int_or_str.rstrip("KB")) * KB
        elif bytes_int_or_str.endswith("MB"):
            return int(bytes_int_or_str.rstrip("MB")) * MB
        elif bytes_int_or_str.endswith("GB"):
            return int(bytes_int_or_str.rstrip("GB")) * GB
        else:
            return int(bytes_int_or_str)
    else:
        raise ValueError(f"Invalid type: {type(bytes_int_or_str)=}")
