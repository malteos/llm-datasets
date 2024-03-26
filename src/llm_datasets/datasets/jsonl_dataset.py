import logging
import json
import multiprocessing
import multiprocess

import os

from smart_open import open

import zstandard as zstd
import io


from llm_datasets.datasets.base import BaseDataset
from llm_datasets.utils.flatmap import flatmap


logger = logging.getLogger(__name__)


class JSONLDataset(BaseDataset):
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
            if fp.startswith("*"):
                # find based on suffix
                yield from self.get_dataset_file_paths(
                    dataset_dir=dataset_dir, subdirectories=True, single_file=False, needed_suffix=fp.lstrip("*")
                )
            else:
                yield os.path.join(dataset_dir, fp)

    def get_text_from_item(self, item):
        """
        This simply returns the text field from item (but dataset classes can override this to implement filtering etc.)
        """
        return item[self.raw_jsonl_text_field]

    def get_texts_from_file_handler(self, file_handler):
        for line in file_handler:
            item = json.loads(line)
            text = self.get_text_from_item(item)

            if text:
                yield text

    def get_texts_from_file_path(self, file_path: str):
        logger.info(f"Reading from {file_path}")

        if file_path.endswith(".zst"):  # zstd compression
            with open(file_path, "rb") as zf:
                dctx = zstd.ZstdDecompressor()  # uncompress zstd
                with dctx.stream_reader(zf) as reader:
                    f = io.BufferedReader(reader)
                    yield from self.get_texts_from_file_handler(f)
        else:
            with open(file_path) as f:  # jsonl or jsonl.fz (via smart_open)
                yield from self.get_texts_from_file_handler(f)

    def get_texts(self):
        """
        Iterate over all input files and read JSON from each line.
        """
        # if self.workers == 1:
        yield from self.get_texts_with_single_proc()
        # else:
        #     yield from self.get_texts_with_multi_proc()

    def get_texts_with_multi_proc(self):
        """
        Iterate over all input files in parallel and read JSON from each line.
        """
        raise NotImplementedError()
        # # with multiprocessing.Pool(self.workers) as pool:
        # with multiprocess.Pool(self.workers) as pool:
        #     for text in flatmap(pool, self.get_texts_from_file_path, self.get_raw_jsonl_paths()):
        #         yield text

        # print("all files done")

    def get_texts_with_single_proc(self):
        """
        Iterate over all input files and read JSON from each line.
        """
        processed_files = 0
        for file_path in self.get_raw_jsonl_paths():
            yield from self.get_texts_from_file_path(file_path)

            processed_files += 1

        if processed_files == 0:
            logger.warning("No file has been processed.")
