# Integrate a custom dataset

## Write a dataset class

The first step for adding a new dataset is write a new dataset class.
If your data comes from a common source such as Huggingface, you can build upon existing abstractions.
For example, Huggingface datasets only needed to specify some metadata like dataset ID, title etc. and the column where the textual data can be extracted from (by default `text` column):

```python
# my_datasets/pg19.py

from lm_datasets.datasets.hf_dataset import HFDataset
from lm_datasets.datasets.base import License, Availability

class PG19Dataset(HFDataset):
    DATASET_ID = "pg19"
    TITLE = "Project Gutenberg books published before 1919"
    HOMEPAGE = "https://huggingface.co/datasets/pg19"
    LICENSE = License("Apache License Version 2.0 (or public domain?)", url="https://www.apache.org/licenses/LICENSE-2.0.html")
    CITATION = """@article{raecompressive2019,
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

## Register new dataset classes

Each dataset class needs to be registered with `lm-dataset` such that the commands know what classes are available.
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
lm_datasets compose ... -extra_dataset_registries=my_datasets.dataset_registry
```