from lm_datasets.datasets.base import BaseDataset
from lm_datasets.datasets.dataset_registry import get_dataset_class_by_id


def test_opus100(tmp_path, raw_datasets_dir):
    dataset_id = "opus100_translation_de_en"
    dataset_cls = get_dataset_class_by_id(dataset_id)

    dataset: BaseDataset = dataset_cls(
        raw_datasets_dir=raw_datasets_dir,
        output_dir=tmp_path,
        limit=100,
    )

    saved_texts_count = dataset.extract_plaintext()

    assert saved_texts_count == 2
    assert dataset.counter["filtered_short_text"] == 198


# def test_tatoeba(tmp_path, raw_datasets_dir):
#     dataset_id = "tatoeba_translation_sv_uk"
#     dataset_cls = get_dataset_class_by_id(dataset_id)

#     dataset: BaseDataset = dataset_cls(
#         raw_datasets_dir=raw_datasets_dir,
#         output_dir=tmp_path,
#         limit=10,
#     )

#     saved_texts_count = dataset.extract_plaintext()

#     assert saved_texts_count == 2
#     assert dataset.counter["filtered_short_text"] == 198

# def test_opus_dataset(tmp_path, raw_datasets_dir):
#     # dataset_id = "opus100_translation_de_en"
#     # extra_dataset_registries = None

#     # id_to_dataset_class = {
#     #     cls.DATASET_ID: cls for cls in get_registered_dataset_classes(extra_dataset_registries)
#     # }
#     # dataset_cls = id_to_dataset_class[dataset_id]

#     raise ValueError("foo")

#     raw_datasets_dir = ""

#     dataset: BaseDataset = get_opus_dataset("de", "en")(
#         raw_datasets_dir=raw_datasets_dir,
#         output_dir=tmp_path,
#         workers=1,
#         limit=100,
#         override_output=False,
#         output_format="jsonl",
#         skip_items=0,
#         max_output_chunk_uncompressed_bytes=get_bytes_from_int_or_string(
#             0
#         ),
#         config=None,
#     )

#     saved_texts_count = dataset.extract_plaintext()

#     assert saved_texts_count == 1
