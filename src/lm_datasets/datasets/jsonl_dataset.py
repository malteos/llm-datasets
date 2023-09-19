import logging
import json
import os

from smart_open import open

import zstandard as zstd
import io


from lm_datasets.datasets.base import BaseDataset


logger = logging.getLogger(__name__)


class JSONLDataset(BaseDataset):
    raw_jsonl_paths = None
    raw_jsonl_text_field = "text"

    def is_downloaded(self):
        for fp in self.get_raw_jsonl_paths():
            if not os.path.exists(fp):
                return False

        return True

    def get_raw_jsonl_paths(self):
        if self.raw_jsonl_paths is None:
            raise ValueError("Cannot load JSONL dataset since `raw_jsonl_paths` is not set")

        dataset_dir = self.get_local_dataset_dir()

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

    def get_texts(self):
        """
        Iterate over all input files and read JSON from each line.
        """
        for fp in self.get_raw_jsonl_paths():
            logger.info(f"Reading from {fp}")

            if fp.endswith(".zst"):  # zstd compression
                with open(fp, "rb") as zf:
                    dctx = zstd.ZstdDecompressor()  # uncompress zstd
                    with dctx.stream_reader(zf) as reader:
                        f = io.BufferedReader(reader)
                        yield from self.get_texts_from_file_handler(f)
            else:
                with open(fp) as f:  # jsonl or jsonl.fz (via smart_open)
                    yield from self.get_texts_from_file_handler(f)
