import pytest

from llm_datasets.datasets.base import BaseDataset
from llm_datasets.datasets.dataset_registry import get_dataset_class_by_id

import os

from llm_datasets.utils.config import Config

RAW_DATASETS_DIR = os.environ.get("RAW_DATASETS_DIR", "/netscratch/ortiz/corpora/ELE/")
COLOSSAL_OSCAR_DIR = os.environ.get(
    "COLOSSAL_OSCAR_DIR", "/netscratch/mostendorff/experiments/eulm/data/colossal-oscar-1.0"
)


@pytest.mark.skipif(not os.path.exists(RAW_DATASETS_DIR), reason="RAW_DATASET_DIR does not exist")
def test_openlegaldata():
    dataset: BaseDataset = get_dataset_class_by_id("openlegaldata")(
        raw_datasets_dir=RAW_DATASETS_DIR,
    )
    for i, doc in enumerate(dataset.get_documents()):
        if i == 0:
            assert doc.text.startswith("\nTenor")
        elif i == 9:
            assert doc.id == "lsgsh-2022-10-11-l-6-as-8722-b-er"
        elif i > 9:
            break


@pytest.mark.skipif(not os.path.exists(COLOSSAL_OSCAR_DIR), reason="COLOSSAL_OSCAR_DIR does not exist")
def test_colossal_oscar():
    dataset: BaseDataset = get_dataset_class_by_id("colossal_oscar_2023-23_fr")(
        config=Config(local_dirs_by_source_id=dict(colossal_oscar=COLOSSAL_OSCAR_DIR)),
    )
    for i, doc in enumerate(dataset.get_documents()):
        if i == 0:
            assert doc.text.startswith("Les cookies nous permettent de personnaliser ")
        elif i == 9:
            assert doc.id == 75
        elif i > 9:
            break


def test_legal_mc4_en():
    dataset: BaseDataset = get_dataset_class_by_id("legal_mc4_en")()
    for i, doc in enumerate(dataset.get_documents()):
        if i == 0:
            assert doc.text.startswith("(1) The scope of the individual services is based on ")
        elif i == 1:
            assert doc.text.startswith("The courts do")
        else:
            break


if __name__ == "__main__":
    # test_openlegaldata()
    # test_legal_mc4_en()
    test_colossal_oscar()
