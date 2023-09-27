from lm_datasets.datasets.base import BaseDataset
from lm_datasets.datasets.dataset_registry import get_dataset_class_by_id


def test_tatoeba(tmp_path, raw_datasets_dir):
    dataset_id = "tatoeba_translation_sv_uk"
    dataset_cls = get_dataset_class_by_id(dataset_id)

    dataset: BaseDataset = dataset_cls(
        raw_datasets_dir=raw_datasets_dir,
        output_dir=tmp_path,
        min_length=0,
        # limit=10,
    )

    saved_texts_count = dataset.extract_plaintext()

    assert saved_texts_count == 28
    # assert dataset.counter["filtered_short_text"] == 198
