import json
import logging
import argparse
import os
from typing import Iterable
from smart_open import open
import re
import gzip

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# dataset metadata
DATASET_ID = "cs_en_parallel"
TITLE = "Czech-English Parallel Corpus 1.0 (CzEng 1.0)"
HOMEPAGE = "http://hdl.handle.net/11234/1-1458"

# language codes of the dataset (see ./resources/languages.json)
LANGUAGES = ["cs"]


"""
DOWNLOAD
-----------

Instruction

- Run the command on the server:

curl --remote-name-all \
    https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1458/data-plaintext-format.tar

- Extract the archive:

tar -xvf data-plaintext-format.tar

"""


def file_iterator(data_path: str) -> Iterable[str]:
    """
    Iterates through all files in a given directory and outputs path names
    """

    for file in os.listdir(data_path):
        filename = os.path.join(data_path, file)
        yield filename


def text_extractor(files_path: Iterable[str]) -> Iterable[str]:
    """
    Extracts all texts from each file. Each file contains several texts, and one line per sentence.
    This function concatenates all sentences belonging to one text and then outputs the text.
    """
    for input_file in file_iterator(files_path):
        with gzip.open(input_file, "rt") as inp:
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
            yield " ".join(cs_texts)


def save_texts_to_jsonl(
    texts: Iterable[str],
    output_file_path: str,
    append: bool = False,
    limit: int = 0,
    min_length: int = 0,
    print_write_progress: int = 1_000,
):
    """
    Saves a list or generator of texts into a JSON-line file (each line is a valid JSON object)
    """
    mode = "a" if append else "w"
    output_text_field = "text"

    # Save as JSONL
    logger.info(f"Writing output to {output_file_path} ({mode=})")

    docs_count = 0

    with open(output_file_path, mode, encoding="utf-8") as f:
        for i, text in enumerate(texts):
            if min_length > 0 and len(text) < min_length:
                # skip because of short text length
                continue

            f.write(json.dumps({output_text_field: text}, ensure_ascii=False) + "\n")
            docs_count += 1

            if i > 0 and (i % print_write_progress) == 0:
                logger.info(f"Writen {i:,} lines ...")

            if limit > 0 and docs_count >= limit:
                logger.warning(f"Limit reached ({docs_count:,} docs)")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to input file or directory")
    parser.add_argument(
        "output_path",
        help="Output is saved in file path (JSONL file format)",
    )
    parser.add_argument("--override", action="store_true", help="Override existing output files")
    parser.add_argument("--download", action="store_true", help="Download dataset")
    parser.add_argument("--limit", default=0, type=int, help="Limit dataset size (for debugging)")
    parser.add_argument("--min_text_length", default=0, type=int, help="Min. text length (shorter texts are discarded)")
    args = parser.parse_args()

    if os.path.exists(args.output_path) and not args.override:
        raise FileExistsError(f"Output exists already ({args.output_path}). Fix this error with --override")

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input does not exist: {args.input_path}")

    # if args.download:
    #     download()

    texts = text_extractor(args.input_path)

    save_texts_to_jsonl(texts, args.output_path, limit=args.limit, min_length=args.min_text_length)
