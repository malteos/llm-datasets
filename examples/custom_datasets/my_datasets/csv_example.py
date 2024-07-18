import logging
from pathlib import Path

import pandas as pd
from llm_datasets.datasets.base import Availability, BaseDataset, License

logger = logging.getLogger(__name__)


class CSVExampleDataset(BaseDataset):
    DATASET_ID = "csv_example"
    TITLE = "An example for a dataset from CSV files"
    AVAILIBITY = Availability.ON_REQUEST
    LANGUAGES = ["en"]
    LICENSE = License("mixed")

    def get_texts(self):
        """Extract texts from CSV files (format: "documen_id,text,score,url")"""
        # Iterate over CSV files in raw dataset directory
        for file_path in self.get_dataset_file_paths(needed_suffix=".csv"):
            file_name = Path(file_path).name

            if (
                file_name.startswith("mc4_")
                or file_name.startswith("colossal-oscar-")
                or file_name.startswith("wikimedia")
            ):
                # skip subsets that overlap with other datasets (baes on file name)
                continue

            logger.info("Reading CSV: %s", file_path)
            try:
                # Use chunks to reduce memory consumption
                for df in pd.read_csv(file_path, sep=",", chunksize=10_000):
                    for text in df.text.values:
                        # Pass extracted text
                        yield text
            except ValueError as e:
                logger.error("Error in file %s; error = %s", file_path, e)
