from lm_datasets.datasets.base import BaseDataset


class ParquetDataset(BaseDataset):
    SINGLE_OUTPUT_FILE = False

    def get_texts(self):
        if self.output_format == "parquet":
            yield from self.generate_texts_from_output(shuffled=False)
        else:
            raise ValueError("Dataset is already in parquet format; cannot generate texts for a different format!")

    def extract_plaintext(self):
        if self.output_format == "parquet":
            raise ValueError("Dataset is already in parquet format; no text extraction needed!")
        else:
            super().extract_plaintext()
