# Integrate a custom dataset

## Write a dataset class

The first step for adding a new dataset is write a new dataset class.
If your data comes from a common source such as Huggingface, you can build upon existing abstractions.

### Huggingface dataset

For example, Huggingface datasets only needed to specify some metadata like dataset ID, title etc. and the column where the textual data can be extracted from (by default `text` column):

```python
# my_datasets/pg19.py

from llm_datasets.datasets.hf_dataset import HFDataset
from llm_datasets.datasets.base import License, Availability

class PG19Dataset(HFDataset):
    DATASET_ID = "pg19"
    TITLE = "Project Gutenberg books published before 1919"
    HOMEPAGE = "https://huggingface.co/datasets/pg19"
    LICENSE = License("Apache License Version 2.0 (or public domain?)", url="https://www.apache.org/licenses/LICENSE-2.0.html")
    CITATION = r"""@article{raecompressive2019,
        author = {Rae, Jack W and Potapenko, Anna and Jayakumar, Siddhant M and
                    Hillier, Chloe and Lillicrap, Timothy P},
        title = {Compressive Transformers for Long-Range Sequence Modelling},
        journal = {arXiv preprint},
        url = {https://arxiv.org/abs/1911.05507},
        year = {2019},
        }
        """  # noqa
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    HF_DATASET_ID = "pg19"
    HF_DATASET_SPLIT = "train"
    streaming = True
    text_column_name = "text"
    title_column_name = "short_book_title"
```

### CSV dataset

Other datasets may require implementing the full text extraction logic. The example below reads text data from CSV files while excluding specific subsets:

```python
# my_datasets/csv_example.py

import logging
import pandas as pd
from pathlib import Path
from llm_datasets.datasets.base import BaseDataset, Availability, License

logger = logging.getLogger(__name__)


class CSVExampleDataset(BaseDataset):
    DATASET_ID = "csv_example"
    TITLE = "An example for a dataset from CSV files"
    AVAILIBITY = Availability.ON_REQUEST
    LANGUAGES = ["en"]
    LICENSE = License("mixed")

    def get_texts(self):
        """
        Extract texts from CSV files (format: "documen_id,text,score,url")
        """
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
```

## Register new dataset classes

Each dataset class needs to be registered with `llm-datasets` such that the commands know what classes are available.
This can be done by making a new Python module with a `get_registered_dataset_classes` method that returns a list of dataset classes:

```python
# my_datasets/dataset_registry.py
from my_datasets.pg19 import PG19Dataset

def get_registered_dataset_classes():
    return [
        PG19Dataset,
    ]
```

## Load registry in commands

To load the registerd datasets in the pipeline commands, you need to specify the `--extra_dataset_registries` argument:

```bash
llm-datasets compose ... -extra_dataset_registries=my_datasets.dataset_registry
```