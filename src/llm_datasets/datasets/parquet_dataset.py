from pathlib import Path
from typing import Iterable
from llm_datasets.datasets.base import BaseDataset


class ParquetDataset(BaseDataset):
    SINGLE_OUTPUT_FILE = False

    def get_texts(self):
        if self.output_format == "parquet":
            yield from self.generate_texts_from_output(shuffled=False)
        else:
            raise ValueError(
                "Dataset is already processed and in parquet format; no need for text extraction! Call `generate_texts_from_output()` instead."
            )

    def extract_plaintext(self):
        if self.output_format == "parquet":
            raise ValueError(
                "Dataset is already in parquet format; no text extraction needed! Call `generate_texts_from_output()` instead."
            )
        else:
            super().extract_plaintext()


class ShuffledParquetDataset(ParquetDataset):
    """
    The raw dataset files are already shuffled.
    """

    def get_file_name_glob_pattern(self):
        raise NotImplementedError()

    def get_single_output_file_path(self, shuffled=False) -> str:
        return None

    def has_chunked_output_files(self, **kwargs):
        return True

    def get_shuffled_output_file_path(self, unshuffled_output_file_path: str) -> str:
        raise ValueError("Dataset is a pre-shuffled dataset.")

    def get_chunked_output_file_paths(self, shuffled=True) -> Iterable[str]:
        return list(Path(self.get_output_dir(shuffled=shuffled)).glob(self.get_file_name_glob_pattern()))
