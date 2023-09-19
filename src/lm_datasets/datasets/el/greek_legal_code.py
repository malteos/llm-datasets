from lm_datasets.datasets.base import Availability, MB
from lm_datasets.datasets.hf_dataset import HFDataset


_CITATION = """
@inproceedings{papaloukas-etal-2021-glc,
    title = "Multi-granular Legal Topic Classification on Greek Legislation",
    author = "Papaloukas, Christos and Chalkidis, Ilias and Athinaios, Konstantinos and Pantazi, Despina-Athanasia and Koubarakis, Manolis",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2021",
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2109.15298",
    doi = "10.48550/arXiv.2109.15298",
    pages = "63--75"
}
"""  # noqa


class GreekLegalCodeDataset(HFDataset):
    DATASET_ID = "greek_legal_code"

    TITLE = "Greek Legal Code"
    DESCRIPTION = (
        "Greek_Legal_Code (GLC) is a dataset consisting of approx. 47k legal resources from Greek legislation. The"
        " origin of GLC is “Permanent Greek Legislation Code - Raptarchis”, a collection of Greek legislative "
        " documents classified into multi-level (from broader to more specialized) categories."
    )

    HOMEPAGE = "https://huggingface.co/datasets/greek_legal_code"

    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["el"]

    HF_DATASET_ID = "greek_legal_code"
    HF_DATASET_SPLIT = "train"
    HF_DATASET_CONFIGS = [  # all the same text but different labels
        "chapter",
        # "subject",
        # "volume",
    ]

    BYTES = (438 / 3) * MB

    text_column_name = "text"
    remove_columns = ["label"]
