from collections import Counter
import json
import logging
import os
from enum import Enum
from urllib.error import HTTPError
from typing import Iterable, List, Optional, TextIO, Tuple, Union

import wget

import pyarrow as pa

from smart_open import open

from pathlib import Path
from lm_datasets.io.parquet import save_texts_to_parquet_chunks

from lm_datasets.systems import get_path_by_system
from lm_datasets.utils import get_parquet_compression

MILLION = 1_000_000
BILLION = 1_000_000_000

KB = 1024
MB = 1024 * 1024
GB = 1024 * 1024 * 1024
TB = 1024 * 1024 * 1024


TOKENS_PER_BYTE = 0.29335  # The Pile GPT2 tokenizeer (from the paper)


class Availability(Enum):
    DIRECT_DOWNLOAD = "direct_download"  # Yes - it has a direct download link or links
    SIGNIN_DOWNLOAD = "signin_download"  # Yes - after signing a user agreement"
    ON_REQUEST = "on_request"  # No - but the current owners/custodians have contact information for data queries
    PRIVATE = "private"  # No - we would need to spontaneously reach out to the current owners/custodians

    def __str__(self):
        return str(self.value)


class QualityWarning(Enum):
    SHORT_TEXT = "short_text"  # Dataset contains mostly short text (below paragraphs)
    BAD_ENCODING = "bad_encoding"  # Text might bit wrong encoded (utf-8 etc.)
    BAD_WHITESPACES = (  # Text might contain bad whitespaces, e.g., before punctuation (converting tokens to text)
        "bad_whitespaces"
    )
    BAD_LINEBREAKS = "bad_linebreaks"  # Text has missing or too many line breaks
    BAD_PUNCTUATION = "bad_punctuation"

    def __str__(self):
        return str(self.value)


class Genre(Enum):
    LEGAL = "legal"
    SCIENCE = "science"
    MATH = "math"
    NEWS = "news"
    DIALOGUE = "dialogue"
    GOVERNMENT = "government"
    LITERATURE = "literature"
    BIOMEDICAL = "biomedical"

    def __str__(self):
        return str(self.value)


# deprecated
AVAILIBILITY_OPTIONS = [
    "Yes - it has a direct download link or links",
    "Yes - after signing a user agreement",
    "No - but the current owners/custodians have contact information for data queries",
    "No - we would need to spontaneously reach out to the current owners/custodians",
]

logger = logging.getLogger(__name__)


class BaseDataset(object):
    """
    Base class for all datasets. It implements all generic loading, processing, and writing methods.
    """

    DATASET_ID = None
    SOURCE_ID = None

    TITLE = None
    DESCRIPTION = None
    HOMEPAGE = None
    AVAILIBILITY = None
    DOWNLOAD_URLS = []
    LOCAL_DIRS = []
    VERSION = None
    DOI = None

    LICENSE = None
    PII = None

    LANGUAGES = []

    WEB_CRAWLED = False
    QUALITY_WARNINGS = []
    GENRES = []
    USED_BY = None
    DUMMY = False
    SINGLE_OUTPUT_FILE = True

    # Statistics
    TOKENS = None
    BYTES = None

    counter = Counter()

    def __init__(
        self,
        output_dir,
        raw_datasets_dir: Optional[str] = None,
        workers: int = 1,
        output_text_field: str = "text",
        override_output: bool = False,
        limit: int = 0,
        skip_items: int = 0,
        hf_auth_token: str = None,
        print_write_progress: int = 10_000,
        min_length: int = 250,
        json_ensure_ascii: bool = False,
        title_delimiter: str = ":\n\n",
        paragraph_delimiter: str = "\n\n",
        sentence_delimiter: str = " ",
        output_format: str = "jsonl",  # options: jsonl, parquet
        output_compression: Optional[
            str
        ] = None,  # jsonl: gzip, parquet: ‘NONE’, ‘SNAPPY’, ‘GZIP’, ‘BROTLI’, ‘LZ4’, ‘ZSTD’
        output_batch_size: int = 1000,
        shuffled_output_dir: Optional[str] = None,
        max_output_chunk_uncompressed_bytes: int = 0,
    ) -> None:
        self.output_dir = output_dir
        self.raw_datasets_dir = raw_datasets_dir
        self.workers = workers
        self.output_text_field = output_text_field
        self.override_output = override_output
        self.limit = limit
        self.skip_items = skip_items
        self.hf_auth_token = hf_auth_token
        self.print_write_progress = print_write_progress
        self.min_length = min_length
        self.json_ensure_ascii = json_ensure_ascii
        self.title_delimiter = title_delimiter
        self.paragraph_delimiter = paragraph_delimiter
        self.sentence_delimiter = sentence_delimiter
        self.output_format = output_format
        self.output_compression = output_compression
        self.output_batch_size = output_batch_size
        self.shuffled_output_dir = shuffled_output_dir
        self.max_output_chunk_uncompressed_bytes = max_output_chunk_uncompressed_bytes

    def get_source_id(self):
        if self.SOURCE_ID:
            return self.SOURCE_ID
        else:
            return self.DATASET_ID

    def get_language_code(self, unknown: str = "unknown", mixed: str = "mixed"):
        if len(self.LANGUAGES) == 1:
            lang = self.LANGUAGES[0]
        elif len(self.LANGUAGES) == 0:
            lang = unknown
        else:
            lang = mixed

        return lang

    def has_output_files(self, min_file_size: int = 1, shuffled=False) -> bool:
        return self.has_single_output_file(
            min_file_size=min_file_size, shuffled=shuffled
        ) or self.has_chunked_output_files(min_file_size=min_file_size, shuffled=shuffled)

    def has_single_output_file(self, min_file_size: int = 1, shuffled=False) -> bool:
        fp = self.get_single_output_file_path(shuffled=shuffled)

        return os.path.exists(fp) and os.stat(fp).st_size >= min_file_size

    def has_chunked_output_files(self, min_file_size: int = 1, shuffled=False) -> bool:
        for fp in self.get_chunked_output_file_paths(shuffled=shuffled):
            if os.path.exists(fp) and os.stat(fp).st_size >= min_file_size:
                return True
            break

        return False

    def get_output_file_paths(self, single=False, chunked=False, shuffled=False) -> List[str]:
        if single:
            return [self.get_single_output_file_path(shuffled=shuffled)]
        elif chunked:
            return self.get_chunked_output_file_paths(shuffled=shuffled)
        else:
            # auto determine based on existing files
            if self.has_chunked_output_files(shuffled=shuffled):
                return self.get_chunked_output_file_paths(shuffled=shuffled)
            else:
                return [self.get_single_output_file_path(shuffled=shuffled)]

    def get_output_file_path(self):
        raise NotImplementedError("Use `get_output_file_paths` instead!")

    def get_output_extension(self, with_dot: bool = True, shuffled: bool = False) -> str:
        extension = "." if with_dot else ""

        if shuffled:
            extension += "shuffled."

        extension += self.output_format

        if self.output_format == "jsonl" and self.output_compression == "gzip":
            # Simply add ".gz" as extension as smart_open will take about the compression
            extension += ".gz"

        return extension

    def get_output_dir(self, shuffled=False):
        if shuffled:
            if self.shuffled_output_dir:
                return self.shuffled_output_dir
            raise ValueError("shuffled_output_dir is not set")
        else:
            return self.output_dir

    def get_single_output_file_path(self, shuffled=False) -> str:
        return os.path.join(
            self.get_output_dir(shuffled=shuffled), self.DATASET_ID + self.get_output_extension(shuffled=shuffled)
        )

    def get_chunked_output_file_paths(self, shuffled=False) -> List[str]:
        output_dir_path = Path(self.get_output_dir(shuffled=shuffled))

        return list(
            output_dir_path.glob(f"{self.DATASET_ID}.part-*-of-*{self.get_output_extension(shuffled=shuffled)}")
        )

    def get_chunked_output_file_path(self, part: int, total_parts: Optional[int] = None, shuffled=False) -> str:
        if total_parts is None:
            fn = f"{self.DATASET_ID}.part-{part:04d}{self.get_output_extension(shuffled=shuffled)}"
        else:
            fn = f"{self.DATASET_ID}.part-{part:04d}-of-{total_parts:04d}{self.get_output_extension(shuffled=shuffled)}"

        return os.path.join(self.get_output_dir(shuffled=shuffled), fn)

    def get_single_or_chunked_output_file_path(
        self, part: Optional[int] = None, total_parts: Optional[int] = None, shuffled=False
    ) -> str:
        if part is None:
            return self.get_single_output_file_path(shuffled=shuffled)
        else:
            return self.get_chunked_output_file_path(part, total_parts, shuffled=shuffled)

    # def has_output_file(self, min_file_size: int = 1):
    #     if self.SINGLE_OUTPUT_FILE:
    #         fps = [self.get_output_file_path()]
    #     else:
    #         fps = self.get_output_file_paths()

    #     for fp in fps:
    #         if os.path.exists(fp) and os.stat(fp).st_size >= min_file_size:
    #             pass
    #         else:
    #             return False

    #     return True

    def filter_texts(self, texts: Iterable[str]):
        """
        Applies basic filtering on the texts before saving
        """
        for text in texts:
            if self.min_length > 0 and len(text) < self.min_length:
                # skip because of short text length
                self.counter.update({"filtered_short_text": 1})
                continue

            yield text

    def remove_texts(self):
        for fp in self.get_output_file_paths():
            logger.warning(f"Removing {fp}")
            os.remove(fp)

    def save_texts(self, texts: Iterable[str], append: bool = False):
        """
        Save texts in different formats
        """
        if self.has_output_files() and not self.override_output:
            raise FileExistsError(f"Output exists already (override not enabled): {self.get_output_file_paths()}")

        if self.output_format == "jsonl":
            docs_count = self.save_texts_to_jsonl(texts, append=append)

        elif self.output_format == "parquet":
            if append:
                raise NotImplementedError("Appending is not supported by parquet output format")

            docs_count = self.save_texts_to_parquet(texts)

        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

        logger.info(f"Documents saved: {docs_count:,}")

        if docs_count == 0:
            logger.warning("No documents have been saved!")

            # delete empty output file
            if self.has_output_files():
                self.remove_texts()

        return docs_count

    def save_texts_to_parquet(self, texts: Iterable[str], file_path: Optional[str] = None, apply_filter: bool = True):
        """
        Save text in parquet (single column schema, in batches)
        """
        if file_path is None:
            file_path = self.get_output_file_paths(single=True)[0]

        if apply_filter:
            texts = self.filter_texts(texts)

        schema = pa.schema(
            [
                (self.output_text_field, pa.string()),
            ]
        )
        # Max. chunk size is multiplied with this factor
        # (to account for inaccurate chunk sizes due to batching)
        safety_factor = 0.975

        # Save as Parquet file
        logger.info(f"Writing parquet output ({self.output_batch_size=}; {self.limit=}; {self.output_compression=})")

        saved_docs, saved_chunks = save_texts_to_parquet_chunks(
            texts=texts,
            schema=schema,
            max_chunk_uncompressed_bytes=self.max_output_chunk_uncompressed_bytes * safety_factor,
            output_path_func=self.get_single_or_chunked_output_file_path,
            compression=get_parquet_compression(self.output_compression),
            batch_size=self.output_batch_size,
            print_write_progress=self.print_write_progress,
            limit=self.limit,
        )

        if hasattr(texts, "terminate"):
            logger.info("Killing all remaining workers, if any (iterator end)")
            texts.terminate()

        return saved_docs

    def save_texts_to_jsonl(self, texts: Iterable[str], append: bool = False):
        """
        Write JSONL files to <output_dir>/<DATASET_ID>.jsonl
        (each line is a JSON object with "doc" field and text as plain text)
        """
        mode = "a" if append else "w"
        fp = self.get_output_file_paths(single=True)[0]

        # Save as JSONL
        logger.info(f"Writing JSONL output to {fp} ({mode=})")

        docs_count = 0

        with open(fp, mode) as f:
            for docs_count, text in enumerate(self.filter_texts(texts), 1):
                f.write(json.dumps({self.output_text_field: text}, ensure_ascii=self.json_ensure_ascii) + "\n")

                if docs_count > 0 and (docs_count % self.print_write_progress) == 0:
                    logger.info(f"Writen {docs_count:,} docs ...")

                if self.limit > 0 and docs_count >= self.limit:
                    logger.warning(f"Limit reached ({docs_count:,} docs)")

                    if hasattr(texts, "terminate"):
                        logger.info("Killing all remaining workers, if any")
                        texts.terminate()
                    break

        if hasattr(texts, "terminate"):
            logger.info("Killing all remaining workers, if any (iterator end)")
            texts.terminate()

        return docs_count

    def get_hf_auth_token(self):
        if self.hf_auth_token:
            return self.hf_auth_token
        else:
            env_token = os.environ.get("HF_PASSWORD")

            if env_token:
                logger.info("Using HF auth token from env var")
                return env_token

        return None

    def get_local_dataset_dir(self):
        if self.LOCAL_DIRS:
            # manually defined dataset directory
            return get_path_by_system(self.LOCAL_DIRS)
        elif self.raw_datasets_dir:
            # automatically based on language + dataset_id
            return os.path.join(self.raw_datasets_dir, self.get_language_code(), self.DATASET_ID)
        else:
            raise ValueError("Either `LOCAL_DIRS` or `raw_datasets_dir` must be defined.")

    def get_dataset_file_paths(
        self,
        dataset_dir: Optional[str] = None,
        single_file: bool = False,
        subdirectories: bool = False,
        needed_suffix: Optional[Union[str, Tuple[str]]] = None,
        return_none_if_not_dir_exists: bool = False,
    ):
        if dataset_dir is None:
            dataset_dir = self.get_local_dataset_dir()

        if not os.path.exists(dataset_dir):
            logger.warning(f"Download directory does not exist: {dataset_dir}")

            if return_none_if_not_dir_exists:
                return None
            else:
                return []

        if subdirectories:
            # find files in all subdirectories
            logger.info(f"Finding dataset files in all subdirectories: {dataset_dir}")
            fps = [os.path.join(path, name) for path, subdirs, files in os.walk(dataset_dir) for name in files]

        else:
            # root-level files
            fps = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]

        # filter by suffix
        fps = [f for f in fps if needed_suffix is None or f.endswith(needed_suffix)]

        # filter by file type
        fps = [fp for fp in fps if os.path.isfile(fp)]

        if single_file:
            if len(fps) != 1:
                raise ValueError(f"Multiple files in download directory but only a single one was expected: {fps}")

            return fps[0]

        return fps

    def decompress(self):
        raise NotImplementedError

    def is_dummy(self):
        return self.DUMMY

    def is_downloaded(self):
        return False

    def download(self):
        # Download all DOWNLOAD_URLS into local dataset dir
        output_dir = self.get_local_dataset_dir()

        logger.info(f"Downloading {len(self.DOWNLOAD_URLS)} files to {output_dir}")

        if not os.path.exists(output_dir):
            logger.info(f"Creating download dir: {output_dir}")
            os.makedirs(output_dir)

        for source_url in self.DOWNLOAD_URLS:
            if isinstance(source_url, tuple):
                source_url, target_filename = source_url
                output_filepath = os.path.join(output_dir, target_filename)

                if os.path.exists(output_filepath):
                    logger.warning(f"Output exists already: {output_filepath}")
                    continue
            else:
                output_filepath = output_dir  # auto file name

            try:
                logger.info(f"Download URL: {source_url}")
                logger.info(f"Output path: {output_filepath}")

                out_filename = wget.download(source_url, out=output_filepath)
                logger.info(f"Completed {out_filename}")
            except HTTPError as e:
                logger.error(f"Error {e}")

    def get_tokens(self):
        if self.TOKENS:
            return self.TOKENS
        elif self.get_bytes():
            return int(self.get_bytes() * TOKENS_PER_BYTE)
        else:
            return None

    def get_bytes(self):
        return self.BYTES

    def get_texts_from_conllu_file(self, file_handler: TextIO):
        import conllu

        text = None

        # try:
        for sentence in conllu.parse_incr(file_handler):
            if "newdoc id" in sentence.metadata:
                if text is not None:
                    # doc completed
                    yield text
                text = ""  # init empty document

            # append text to doc
            if "text" in sentence.metadata:
                if not text:
                    text = ""  # some conllu are not using doc ids -> force init
                else:
                    text += " "  # whitespace betweeen sentences

                text += sentence.metadata["text"]

            if "title" in sentence.metadata:
                text += self.title_delimiter

        # yield last document
        if text:
            yield text

        # except ParseException as e:
        #     # TODO
        #     logger.error(e)

    def get_texts(self) -> Iterable[str]:
        raise NotImplementedError

    def extract_plaintext(self):
        self.save_texts(self.get_texts())

        if self.counter:
            logger.info(f"Statistics {self.counter}")
