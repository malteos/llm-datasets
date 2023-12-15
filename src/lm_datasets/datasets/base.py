from collections import Counter
import json
import logging
import os
from enum import Enum
import random
from urllib.error import HTTPError
from typing import Iterable, List, Literal, Optional, TextIO, Tuple, Type, Union

import wget

import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

from smart_open import open as smart_open

from pathlib import Path
from lm_datasets.io.parquet import get_selected_row_groups, save_texts_to_parquet_chunks
from lm_datasets.utils.settings import DEFAULT_MIN_TEXT_LENGTH

from lm_datasets.utils.systems import get_path_by_system
from lm_datasets.utils import get_parquet_compression
from lm_datasets.utils.config import Config

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
    BAD_WHITESPACES = (  # Text might contain bad whitespaces, e.g., before punctuation (converting tokens to text).
        "bad_whitespaces"
    )
    BAD_LINEBREAKS = "bad_linebreaks"  # Text has missing or too many line breaks.
    BAD_PUNCTUATION = "bad_punctuation"  # Punctuation may be missing or incorrect.
    BAD_DOCUMENT_SPLITS = "bad_document_splits"  # Extract texts may span accross multi documents or documents are split into multiple texts.

    def __str__(self):
        return str(self.value)


class Genre(Enum):
    LEGAL = "legal"
    SCIENCE = "science"
    PATENT = "patent"
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


class License(object):
    """
    Basic licensing information. Set attributions must be verified. If an attribution is unset, it is unknown.

    See https://choosealicense.com/ and https://creativecommons.org/share-your-work/cclicenses/
    """

    def __init__(
        self,
        name,
        url: Optional[str] = None,
        commercial_use: Optional[bool] = None,
        research_use: Optional[bool] = None,
        modification: Optional[bool] = None,
        distribution: Optional[bool] = None,
        sharealike: Optional[bool] = None,
        attribution: Optional[bool] = None,
        derivates: Optional[bool] = None,
        informing: Optional[bool] = None,
    ):
        self.name = name  # name and/or description of license
        self.url = url  # link to full license text
        self.commercial_use = commercial_use  # like CC-NC
        self.research_use = research_use
        self.modification = modification
        self.distribution = distribution
        self.sharealike = sharealike  # like CC-SA
        self.attribution = attribution  # like CC-BY
        self.derivates = derivates  # like CC-ND
        self.informing = informing  # like ACA ID-BY-NC-INF-NORED

    def __str__(self):
        return f"{self.name} ({self.commercial_use=}; {self.sharealike=})"


logger = logging.getLogger(__name__)


class BaseDataset(object):
    """
    Base class for all datasets. It implements all generic loading, processing, and writing methods.
    """

    DATASET_ID = None
    SOURCE_ID = None

    TITLE = None
    DESCRIPTION = None
    HOMEPAGE: Optional[str] = None
    AVAILIBILITY: Availability = None
    DOWNLOAD_URLS: List[Union[str, Tuple[str]]] = []
    LOCAL_DIRS = []
    VERSION = None
    DOI = None
    CITATION = None

    LICENSE: Optional[Union[str, License]] = None
    PII = None

    LANGUAGES = []

    TRANSLATIONS = False
    WEB_CRAWLED = False
    QUALITY_WARNINGS: List[QualityWarning] = []
    GENRES: List[Genre] = []
    HAS_OVERLAP_WITH: List[Union[Type, str]] = []
    USED_BY = None
    DUMMY = False
    SINGLE_OUTPUT_FILE = True
    HAS_PREDEFINED_VALIDATION_SET = False

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
        min_length: Optional[int] = None,
        json_ensure_ascii: bool = False,
        title_delimiter: str = ":\n\n",
        paragraph_delimiter: str = "\n\n",
        sentence_delimiter: str = " ",
        output_format: Literal["jsonl", "parquet"] = "jsonl",
        output_compression: Optional[
            str
        ] = None,  # jsonl: gzip, parquet: ‘NONE’, ‘SNAPPY’, ‘GZIP’, ‘BROTLI’, ‘LZ4’, ‘ZSTD’
        output_batch_size: int = 1000,
        shuffled_output_dir: Optional[str] = None,
        max_output_chunk_uncompressed_bytes: Optional[int] = None,
        max_output_chunk_rows: Optional[int] = None,
        config: Config = None,
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
        self.min_length = min_length if min_length is not None else DEFAULT_MIN_TEXT_LENGTH
        self.json_ensure_ascii = json_ensure_ascii
        self.title_delimiter = title_delimiter
        self.paragraph_delimiter = paragraph_delimiter
        self.sentence_delimiter = sentence_delimiter
        self.output_format = output_format
        self.output_compression = output_compression
        self.output_batch_size = output_batch_size
        self.shuffled_output_dir = shuffled_output_dir
        self.max_output_chunk_uncompressed_bytes = max_output_chunk_uncompressed_bytes
        self.max_output_chunk_rows = max_output_chunk_rows
        self.config = config

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

    def get_output_text_field(self):
        return self.output_text_field

    def has_output_files(self, min_file_size: int = 1, shuffled=False) -> bool:
        return self.has_single_output_file(
            min_file_size=min_file_size, shuffled=shuffled
        ) or self.has_chunked_output_files(min_file_size=min_file_size, shuffled=shuffled)

    def has_single_output_file(self, min_file_size: int = 1, shuffled=False) -> bool:
        fp = self.get_single_output_file_path(shuffled=shuffled)

        return fp is not None and os.path.exists(fp) and os.stat(fp).st_size >= min_file_size

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

            docs_count, saved_chunks = self.save_texts_to_parquet(texts)

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
        assert self.output_format == "parquet"

        if file_path is None:
            file_path = self.get_output_file_paths(single=True)[0]

        if apply_filter:
            texts = self.filter_texts(texts)

        schema = pa.schema(
            [
                (self.get_output_text_field(), pa.string()),
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
            max_chunk_uncompressed_bytes=self.max_output_chunk_uncompressed_bytes * safety_factor
            if self.max_output_chunk_uncompressed_bytes is not None
            else None,
            max_chunk_rows=self.max_output_chunk_rows,
            output_path_func=self.get_single_or_chunked_output_file_path,
            compression=get_parquet_compression(self.output_compression),
            batch_size=self.output_batch_size,
            print_write_progress=self.print_write_progress,
            limit=self.limit,
        )

        if hasattr(texts, "terminate"):
            logger.info("Killing all remaining workers, if any (iterator end)")
            texts.terminate()

        return saved_docs, saved_chunks

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

        with smart_open(fp, mode) as f:
            for docs_count, text in enumerate(self.filter_texts(texts), 1):
                f.write(json.dumps({self.get_output_text_field(): text}, ensure_ascii=self.json_ensure_ascii) + "\n")

                if docs_count > 0 and (docs_count % self.print_write_progress) == 0:
                    logger.info(f"Written {docs_count:,} docs ...")

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
        if self.config:
            if self.DATASET_ID in self.config.local_dirs_by_dataset_id:
                return self.config.local_dirs_by_dataset_id[self.DATASET_ID]

            if self.get_source_id() in self.config.local_dirs_by_source_id:
                return self.config.local_dirs_by_source_id[self.get_source_id()]

        if self.LOCAL_DIRS:  # TODO deprecated -> use config instead!
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
            if len(fps) > 1:
                raise FileExistsError(f"Multiple files in download directory but only a single one was expected: {fps}")
            elif len(fps) == 0:
                raise FileNotFoundError(f"No file found but a single one was expected: {fps}")

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

    def extract_plaintext(self) -> int:
        saved_texts_count = self.save_texts(self.get_texts())

        if self.counter:
            logger.info(f"Statistics {self.counter}")

        return saved_texts_count

    def get_output_rows_count(self, shuffled: bool = False) -> int:
        """
        Read metadata from parquet files and extract number of rows
        """
        if self.output_format == "parquet":
            output_paths = list(self.get_output_file_paths(shuffled=shuffled))

            # Filter for existing
            output_paths = [output_path for output_path in output_paths if os.path.exists(output_path)]

            if output_paths:
                rows_count = 0

                for output_path in output_paths:
                    with open(output_path, "rb") as f:
                        parquet_file = pq.ParquetFile(
                            f,
                            # increased to avoid OSErrors
                            thrift_string_size_limit=1000000000,  # default: 100000000
                            thrift_container_size_limit=10000000,  # default: 1000000
                        )
                        rows_count += parquet_file.metadata.num_rows

                        logger.debug("Rows = %s in %s", rows_count, output_path)

                return rows_count

            logger.debug("No output files exists for %s", self.DATASET_ID)
            return -1
        else:
            raise ValueError(f"Cannot determine the output rows count with {self.output_format=}")

    def get_compression_from_output_files(self, shuffled=False):
        """
        NOTE: Currently only implemented for `parquet` format.
        """
        if self.output_format == "parquet":
            for output_path in self.get_output_file_paths(shuffled=shuffled):
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        parquet_file = pq.ParquetFile(
                            f,
                            # increased to avoid OSErrors
                            thrift_string_size_limit=1000000000,  # default: 100000000
                            thrift_container_size_limit=10000000,  # default: 1000000
                        )
                        parquet_metadata = parquet_file.metadata
                        for i in range(parquet_metadata.num_row_groups):
                            for j in range(parquet_metadata.num_columns):
                                return parquet_file.metadata.row_group(i).column(j).compression

        return "unknown"

    def generate_texts_from_output(
        self,
        shuffled: bool = False,
        batch_size: Optional[int] = None,
        limit: int = 0,
        offset: int = 0,
        shuffle_output_file_paths: bool = False,
        reader_implementation: Literal["polars_read_parquet", "pyarrow"] = "pyarrow",
    ):
        """
        A iterator over texts from processed output files.
        """
        if batch_size is None:
            batch_size = self.output_batch_size

        if self.output_format != "parquet":
            raise ValueError(f"Cannot texts with {self.output_format=}")

        # Check if output files exists and sort them
        output_paths = [
            file_path
            for file_path in sorted(self.get_output_file_paths(shuffled=shuffled))
            if os.path.exists(file_path)
        ]

        # Count generated rows
        rows = 0
        rows_limit = limit - offset

        # if limit > 0:
        #     batch_size = min(batch_size, limit)

        # Shuffle output chunks:
        # This changes the order in that the chunks are read ensure also shuffling on the full dataset level.
        if shuffle_output_file_paths:
            random.seed(self.config.seed)  # reset seed to avoid inference by other scripts
            random.shuffle(output_paths)

        chunk_start = 0
        chunk_end = None

        if output_paths:
            for file_path in output_paths:
                logger.info("Generating text from %s", file_path)

                # PyArrow implementation
                with open(file_path, "rb") as file_handler:
                    pq_file = pq.ParquetFile(
                        file_handler,
                        # memory_map=False,
                    )
                    file_rows_count = pq_file.metadata.num_rows

                    chunk_end = chunk_start + file_rows_count - 1

                    # Should we read from the current chunk?
                    # Yes, if
                    # - offset is smaller or equal chunk_start
                    # (- limit is greater or equal chunk_end) --- limit does not matter

                    # variants
                    # A) requested rows start in chunk and ends in chunk
                    # B) requested rows start in chunk but ends in following chunk
                    # C) requested rows start before chunk and ends in chunk
                    # D) requested rows start before chunk and ends in following chunk

                    if (
                        chunk_start <= offset < chunk_end
                        or offset < chunk_start
                        and (limit == 0 or chunk_start < limit)
                    ):
                        file_offset = max(
                            0, offset - chunk_start
                        )  # global offset minus start of current file (current chunk)
                        file_limit = (
                            max(0, limit - chunk_start) if limit > 0 else 0  # limit - chunk_start
                        )  # Length of the slice: global limit minus start of current chunk
                        # TODO before: limit - chunk_start - file_offset

                        logger.debug(
                            f"Reading file chunk from %s: file [%s - %s]; global [%s - %s]; chunk [%s - %s]",
                            file_path,
                            file_offset,
                            file_limit,
                            offset,
                            limit,
                            chunk_start,
                            chunk_end,
                        )
                        if reader_implementation == "pyarrow":
                            # PyArrow implementation with iter_batches
                            # with open(file_path, "rb") as file_handler:
                            #     parquet_file = pq.ParquetFile(file_handler)

                            for batch_idx, pq_batch in enumerate(
                                pq_file.iter_batches(
                                    columns=[self.get_output_text_field()], batch_size=batch_size, use_threads=False
                                )
                            ):
                                for row_idx, text_column in enumerate(pq_batch.columns[0], batch_idx * batch_size):
                                    if row_idx >= file_offset:
                                        if rows_limit > 0 and rows >= rows_limit:
                                            # break row loop
                                            logger.debug("break row loop")
                                            break

                                        # cast to string
                                        # text = text_column.as_py()

                                        text = text_column

                                        yield text
                                        rows += 1

                                if rows_limit > 0 and rows >= rows_limit:
                                    # break batch loop
                                    logger.debug("break batch loop")
                                    break

                            # PyArrow implementation with read_row_group
                            # with open(file_path, "rb") as file_handler:
                            #     parquet_file = pq.ParquetFile(file_handler)

                            #     # 1. What row groups need to be read?
                            #     row_groups, group_idx_to_offset_last_row = get_selected_row_groups(
                            #         parquet_file, file_offset, file_limit
                            #     )
                            #     logger.debug("Selected row groups: %s; %s", row_groups, group_idx_to_offset_last_row)

                            #     # 2. Read selected row groups
                            #     for selected_row_group in row_groups:
                            #         logger.debug("Read row group: %s", selected_row_group)
                            #         group_table = parquet_file.read_row_group(
                            #             selected_row_group, columns=[self.get_output_text_field()]
                            #         )

                            #         # What offsets and limit? (only if needed)
                            #         if group_idx_to_offset_last_row is not None:
                            #             group_offset, _ = group_idx_to_offset_last_row[selected_row_group]

                            #             row_offset = max(0, file_offset - group_offset)
                            #             logger.debug("Row group: %s; row offset: %s", selected_row_group, row_offset)

                            #         # Iterate over rows
                            #         for row_idx, text_column in enumerate(group_table.columns[0]):
                            #             # Skip rows before offset
                            #             if group_idx_to_offset_last_row is None or row_idx >= row_offset:
                            #                 if rows_limit > 0 and rows >= rows_limit:
                            #                     # break row loop
                            #                     logger.debug("break row loop")
                            #                     break

                            #                 text = text_column.as_py()  # cast to str
                            #                 yield text
                            #                 rows += 1

                            #         if rows_limit > 0 and rows >= rows_limit:
                            #             # break row group loop
                            #             logger.debug("break row group loop")
                            #             break

                        elif reader_implementation == "polars_read_parquet":
                            # Polars "scan_parquet" implementation: Error "Segmentation fault (core dumped)"
                            # df = (
                            #     pl.scan_parquet(file_path, low_memory=True).collect(
                            #     streaming=True
                            # ).slice(offset=file_offset, length=file_limit if file_limit != 0 else None)
                            #     .collect(streaming=True)
                            # )
                            # text_column_index = df.columns.index(self.get_output_text_field())

                            df = pl.read_parquet(
                                file_path, low_memory=True, columns=[self.get_output_text_field()]
                            ).slice(offset=file_offset, length=file_limit if file_limit != 0 else None)
                            text_column_index = 0

                            # Iterate over rows
                            for row in df.iter_rows():
                                text = str(row[text_column_index])
                                yield text
                                rows += 1

                                if rows_limit > 0 and rows >= rows_limit:
                                    # break row loop
                                    break
                            else:
                                raise ValueError("Invalid `reader_implementation`")
                    else:
                        logger.debug("Skip this file because output does not contain the requested rows: %s", file_path)

                    # current_offset += file_rows_count  # TODO +1?
                    chunk_start = chunk_end + 1  # set start for the next chunk

                if rows_limit > 0 and rows >= rows_limit:
                    # break file loop
                    logger.debug("break file loop")
                    break
        else:
            logger.warning("Cannot generate texts because output files do not exist.")

        logger.info(
            "Texts generated: %s (expected size: %s; offset: %s; limit: %s;)", rows, limit - offset, offset, limit
        )

    def get_estimated_bytes_from_output(self, shuffled: bool = False, read_first_n_rows: int = 1_000) -> int:
        """
        Estimate byte size of output text:
        - read first N rows of shuffled output files and count their byte size
        - multiply counted bytes by total number of rows
        """
        if not shuffled:
            raise NotImplementedError

        if self.output_format != "parquet":
            raise NotImplementedError

        bytes_sum = 0
        total_rows = 0

        # iterate over output files (use shuffled files for a better estimate)
        for output_path in self.get_output_file_paths(shuffled=shuffled):
            if os.path.exists(output_path):
                # read the first n rows
                df = pl.scan_parquet(
                    output_path,
                    low_memory=True,
                    n_rows=read_first_n_rows,
                ).collect(streaming=True)
                for row in df.iter_rows():
                    text = str(row[0])
                    bytes_sum += len(text.encode("utf-8"))  # count the byte size of the text

                # read total row count from metadata
                with open(output_path, "rb") as f:
                    parquet_file = pq.ParquetFile(
                        f,
                        # increased to avoid OSErrors
                        thrift_string_size_limit=1000000000,  # default: 100000000
                        thrift_container_size_limit=10000000,  # default: 1000000
                    )
                    total_rows += parquet_file.metadata.num_rows

        # estimated bytes
        bytes_per_row = bytes_sum / read_first_n_rows
        total_bytes = int(total_rows * bytes_per_row)

        return total_bytes

    def get_sampling_factor(self) -> float:
        """
        Sampling is defined based on dataset ID, source ID, or language.
        """
        if self.config:
            if self.DATASET_ID in self.config.sampling_factor_by_dataset_id:
                return self.config.sampling_factor_by_dataset_id[self.DATASET_ID]

            if self.get_source_id() in self.config.sampling_factor_by_source_id:
                return self.config.sampling_factor_by_source_id[self.get_source_id()]

            if self.get_language_code() in self.config.sampling_factor_by_language:
                return self.config.sampling_factor_by_language[self.get_language_code()]

        return 1.0  # default factor

    def is_selected(self) -> bool:
        """
        Is this dataset part of selected datasets or sources?
        """
        return (
            self.DATASET_ID in self.config.selected_dataset_ids
            or self.get_source_id() in self.config.selected_source_ids
        )

    def get_shuffled_output_file_path(self, unshuffled_output_file_path: str) -> str:
        output_file_name = Path(unshuffled_output_file_path).name

        return os.path.join(self.config.shuffled_output_dir, output_file_name.replace(".parquet", ".shuffled.parquet"))
