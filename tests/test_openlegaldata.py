from smart_open import open
import json
import pytest
import os

DATASET_FILE_PATH = "/netscratch/mostendorff/experiments/eulm/data/ele/de/openlegaldata/merged_v1_v2.jsonl.gz"


@pytest.mark.skipif(not os.path.exists(DATASET_FILE_PATH), reason="test file not exists")
def test_openlegaldata():
    docs_count = 1_658_611
    docs = []

    # after 250/1659 row groups
    # rows per group = 1000

    with open(DATASET_FILE_PATH) as f:
        for idx, line in enumerate(f):
            if idx > docs_count - 1_000:
                doc = json.loads(line)
                docs.append(doc)

    print("done")
