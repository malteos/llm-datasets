import math
from lm_datasets.datasets.base import BaseDataset
from lm_datasets.datasets.dataset_registry import get_registered_dataset_classes
from lm_datasets.utils.config import Config

from datasets import IterableDataset, interleave_datasets

from enum import Enum
import logging


logger = logging.getLogger(__name__)


class DatasetSplit(Enum):
    FULL = "full"
    TRAIN = "train"
    TOKENIZER_TRAIN = "tokenizer_train"
    VALIDATION = "validation"

    def __str__(self):
        return str(self.value)


def generate_texts_from_datasets(
    config: Config, shuffled_output_dir, output_format, split: DatasetSplit = DatasetSplit.TRAIN
):
    """
    Generate texts based on split or full datasets and the given config.
    """
    ds_clss = get_registered_dataset_classes(extra_dataset_registries=config.extra_dataset_registries)
    selected_dataset_ids = config.selected_dataset_ids
    selected_source_ids = config.selected_source_ids

    # hf_iterable_datasets = []

    # for ds_cls in ds_clss:
    #     ds: BaseDataset = ds_cls(
    #         output_dir=None,
    #         output_format=output_format,
    #         raw_datasets_dir=None,
    #         shuffled_output_dir=shuffled_output_dir,
    #         config=config,
    #     )
    #     dataset_id = ds.DATASET_ID
    #     source_id = ds.get_source_id()

    #     if dataset_id in selected_dataset_ids or source_id in selected_source_ids:
    #         hf_ds = IterableDataset.from_generator(
    #             generate_texts_from_dataset(ds, split=split)
    #         )

    #         hf_iterable_datasets.append(hf_ds)

    # interleave_datasets(datasets=hf_iterable_datasets, stopping_strategy)

    raise NotImplementedError


def generate_texts_from_dataset(dataset: BaseDataset, split: DatasetSplit, use_shuffled_output: bool = True):
    """

    Full dataset:

    [-------------]

    first n rows as validation:

    [xxxx][-------]

    last 1-n rows as train set:

    [----][xxxxxxx]

    first n rows of train set are the tokenizer train set:

    [----][xx][---]

    """
    if split == DatasetSplit.FULL:
        yield from dataset.generate_texts_from_output(
            shuffled=use_shuffled_output,
            offset=0,
            limit=0,
        )
        return

    n_docs = dataset.get_output_rows_count(shuffled=use_shuffled_output)
    sampling_factor = dataset.get_sampling_factor()
    validation_offset = 0  # validation is always the first n rows

    # Compute offsets + limits for split selection
    if dataset.HAS_PREDEFINED_VALIDATION_SET:
        logger.warning("No validation split - because a predefined validation set exists: %s", dataset.DATASET_ID)
        validation_n_docs = 0
    elif n_docs < dataset.config.validation_min_total_docs:
        logger.warning("No validation split - because dataset has too few docs: %s", dataset.DATASET_ID)
        validation_n_docs = 0

    validation_n_docs = min(
        int(dataset.config.validation_ratio * n_docs),
        dataset.config.validation_max_split_docs,
    )

    if validation_n_docs > 0:
        train_offset = validation_n_docs  # + 1
    else:
        train_offset = 0

    train_n_docs = int(sampling_factor * (n_docs - validation_n_docs))
    tokenizer_train_n_docs = int(train_n_docs * dataset.config.tokenizer_train_ratio)

    logger.debug("split: %s", split)
    logger.debug("train_n_docs: %s", train_n_docs)
    logger.debug("train_offset: %s", train_offset)
    logger.debug("validation_n_docs: %s", validation_n_docs)
    logger.debug("validation_offset: %s", validation_offset)
    logger.debug("tokenizer_train_n_docs: %s", tokenizer_train_n_docs)

    if validation_n_docs < dataset.config.validation_min_split_docs:
        logger.warning(
            f"No validation split - because split would be too small; {dataset.DATASET_ID=} {validation_n_docs=}"
        )
        validation_n_docs = 0

    if split == DatasetSplit.VALIDATION:
        # Validation set (only used if dataset is large enough and does not have a predefined validation set)

        if validation_n_docs > 0:
            yield from dataset.generate_texts_from_output(
                shuffled=use_shuffled_output, offset=validation_offset, limit=validation_n_docs
            )

    elif split == DatasetSplit.TRAIN:
        # The actual training data (corresponds to: full set without validation set)
        if sampling_factor == 1:
            yield from dataset.generate_texts_from_output(shuffled=use_shuffled_output, offset=train_offset, limit=0)

        elif sampling_factor < 1:
            # downsampling
            yield from dataset.generate_texts_from_output(
                shuffled=use_shuffled_output, offset=train_offset, limit=train_n_docs
            )
        else:
            # upsampling: requested dataset size is large than the actual dataset
            full_reads = math.floor(train_n_docs / n_docs)
            last_partial_read = (train_n_docs / n_docs) - full_reads
            last_partial_read_n = int(last_partial_read * n_docs)

            # full read (no limit): repeated reading from the dataset
            for _ in range(full_reads):
                yield from dataset.generate_texts_from_output(
                    shuffled=use_shuffled_output, offset=train_offset, limit=0
                )

            # last partial read (with limit)
            if last_partial_read_n > 0:
                yield from dataset.generate_texts_from_output(
                    shuffled=use_shuffled_output, offset=train_offset, limit=last_partial_read_n
                )

    elif split == DatasetSplit.TOKENIZER_TRAIN:
        # Training dataset for tokenizer (by default: 10% of training set)
        tokenizer_train_offset = train_offset
        tokenizer_train_limit = train_offset + tokenizer_train_n_docs  # + 1

        yield from dataset.generate_texts_from_output(
            shuffled=use_shuffled_output, offset=tokenizer_train_offset, limit=tokenizer_train_limit
        )
    else:
        raise ValueError("Invalid split: %s", split)
