import io
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

import zstandard as zstd
from datatrove.data import Document
from smart_open import open

from llm_datasets.datasets.base import BaseDocumentDataset, BaseTextDataset

logger = logging.getLogger(__name__)


class JSONLMixin(object):
    raw_jsonl_paths = None
    raw_jsonl_text_field = "text"

    def is_downloaded(self):
        try:
            for fp in self.get_raw_jsonl_paths():
                if not os.path.exists(fp):
                    return False

            return True

        except FileNotFoundError:
            return False

    def get_raw_jsonl_paths(self):
        if self.raw_jsonl_paths is None:
            raise ValueError("Cannot load JSONL dataset since `raw_jsonl_paths` is not set")

        dataset_dir = self.get_local_dataset_dir()

        if not os.path.exists(dataset_dir):
            raise FileExistsError("Dataset directory does not exist")

        for fp in self.raw_jsonl_paths:
            if isinstance(fp, Path):
                yield fp
            else:
                if fp.startswith("*"):
                    # find based on suffix
                    yield from self.get_dataset_file_paths(
                        dataset_dir=dataset_dir, subdirectories=True, single_file=False, needed_suffix=fp.lstrip("*")
                    )
                else:
                    yield os.path.join(dataset_dir, fp)


class JSONLDataset(JSONLMixin, BaseTextDataset):  # TODO rename to JSONLTextDataset
    def get_text_from_item(self, item) -> str:
        """This simply returns the text field from item (but dataset classes can override this to implement filtering etc.)"""
        return item[self.raw_jsonl_text_field]

    def get_document_from_item(self, item) -> Document:
        """This simply returns the document with a text field from item (but dataset classes can override this to implement filtering etc.)"""
        return Document(text=item[self.raw_jsonl_text_field])

    def get_texts_from_file_handler(self, file_handler):
        if hasattr(self.config, "use_documents") and self.config.use_documents:
            getter_func = self.get_document_from_item
        else:
            getter_func = self.get_text_from_item

        for line in file_handler:
            item = json.loads(line)
            text = getter_func(item)

            if text:
                yield text

    def get_texts_from_file_path(self, file_path: str | Path):
        logger.info(f"Reading from {file_path}")

        if (
            isinstance(file_path, str) and file_path.endswith(".zst")
        ) or file_path.suffix == ".zst":  # zstd compression
            with open(file_path, "rb") as zf:
                dctx = zstd.ZstdDecompressor()  # uncompress zstd
                with dctx.stream_reader(zf) as reader:
                    f = io.BufferedReader(reader)
                    yield from self.get_texts_from_file_handler(f)
        else:
            with open(file_path) as f:  # jsonl or jsonl.fz (via smart_open)
                yield from self.get_texts_from_file_handler(f)

    def get_texts(self):
        """Iterate over all input files and read JSON from each line."""
        # if self.workers == 1:
        yield from self.get_texts_with_single_proc()
        # else:
        #     yield from self.get_texts_with_multi_proc()

    def get_texts_with_multi_proc(self):
        """Iterate over all input files in parallel and read JSON from each line."""
        raise NotImplementedError()
        # # with multiprocessing.Pool(self.workers) as pool:
        # with multiprocess.Pool(self.workers) as pool:
        #     for text in flatmap(pool, self.get_texts_from_file_path, self.get_raw_jsonl_paths()):
        #         yield text

        # print("all files done")

    def get_texts_with_single_proc(self):
        """Iterate over all input files and read JSON from each line."""
        processed_files = 0
        for file_path in self.get_raw_jsonl_paths():
            yield from self.get_texts_from_file_path(file_path)

            processed_files += 1

        if processed_files == 0:
            logger.warning("No file has been processed.")


class JSONLDocumentDataset(JSONLMixin, BaseDocumentDataset):  # TODO rename to JSONLTextDataset
    INDEX_OFFSET_PER_FILE = 10_000_000

    def get_document_from_item(self, item, index: Optional[int] = None) -> Document:
        """This simply returns the document with a text field from item (but dataset classes can override this to implement filtering etc.)"""
        return Document(text=item[self.raw_jsonl_text_field], id=index)

    def get_documents_from_file_handler(self, file_handler, file_index: Optional[int] = None) -> Iterable[Document]:
        file_index = file_index * self.INDEX_OFFSET_PER_FILE

        for line_index, line in enumerate(file_handler):
            item = json.loads(line)
            text = self.get_document_from_item(item, index=file_index + line_index)

            if text:
                yield text

    def get_documents_from_file_path(self, file_path: str, file_index: Optional[int] = None) -> Iterable[Document]:
        logger.info(f"Reading from {file_path}")

        if file_path.endswith(".zst"):  # zstd compression
            with open(file_path, "rb") as zf:
                dctx = zstd.ZstdDecompressor()  # uncompress zstd
                with dctx.stream_reader(zf) as reader:
                    f = io.BufferedReader(reader)
                    yield from self.get_documents_from_file_handler(f, file_index)
        else:
            with open(file_path) as f:  # jsonl or jsonl.fz (via smart_open)
                yield from self.get_documents_from_file_handler(f, file_index)

    def get_documents(self) -> Iterable[Document]:
        """Iterate over all input files and read JSON from each line."""
        yield from self.get_documents_with_single_proc()

    def get_documents_with_single_proc(self) -> Iterable[Document]:
        """Iterate over all input files and read JSON from each line."""
        file_index = -1
        for file_index, file_path in enumerate(self.get_raw_jsonl_paths()):
            yield from self.get_documents_from_file_path(file_path, file_index)

        if file_index == -1:
            logger.warning("No file has been processed.")
