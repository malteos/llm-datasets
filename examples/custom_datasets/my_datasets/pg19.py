from lm_datasets.datasets.hf_dataset import HFDataset
from lm_datasets.datasets.base import License, Availability


class PG19Dataset(HFDataset):
    DATASET_ID = "pg19"
    TITLE = "Project Gutenberg books published before 1919"
    HOMEPAGE = "https://huggingface.co/datasets/pg19"
    LICENSE = License(
        "Apache License Version 2.0 (or public domain?)", url="https://www.apache.org/licenses/LICENSE-2.0.html"
    )
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
