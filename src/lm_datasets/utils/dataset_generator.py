import asyncio
import math
from pathlib import Path
import random
from typing import Iterable, Iterator, List, Optional, Tuple, Union
from lm_datasets.datasets.base import BaseDataset
from lm_datasets.datasets.dataset_registry import get_registered_dataset_classes
from lm_datasets.utils.config import Config

from tqdm.auto import tqdm
import numpy as np
import pyarrow as pa
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class DatasetSplit(str, Enum):
    """
    Full dataset:

    [-------------]

    first n rows as validation:

    [xxxx][-------]

    last 1-n rows as train set:

    [----][xxxxxxx]

    first n rows of train set are the tokenizer train set:

    [----][xx][---]

    Sampling: Up/down sampling is only applied to train and tokenizer_train set.
    """

    FULL = "full"
    TRAIN = "train"
    TOKENIZER_TRAIN = "tokenizer_train"
    VALIDATION = "validation"

    # def __str__(self):
    #     return str(self.value)


DatasetSplitOffsetLimit = dict[DatasetSplit, Tuple[int, int]]


def iter_random_indices(
    rng: np.random.Generator,
    num_sources: int,
    probabilities: List[float],
    random_batch_size: int = 100,
) -> Iterator[int]:
    """
    Get an infinite iterator that randomly samples the index of the source to pick examples from.
    """
    assert len(probabilities) == num_sources

    while True:
        yield from (int(i) for i in rng.choice(num_sources, size=random_batch_size, p=probabilities))


def interleave_dataset_iterators(
    datasets_iterators: List[Iterable],
    probabilities: List[float],
    seed: int = 0,
    random_indicies_batch_size: int = 100,
    datasets_prefixes: Optional[List] = None,
) -> Iterator:
    """
    Interleave dataset iterators.

    Implementation based on Huggingface's datasets library

    https://github.com/huggingface/datasets/blob/main/src/datasets/iterable_dataset.py#L31
    """
    datasets_count = len(datasets_iterators)
    random_generator = np.random.default_rng(seed)

    indices_iterator = iter_random_indices(
        random_generator,
        num_sources=datasets_count,
        probabilities=probabilities,
        random_batch_size=random_indicies_batch_size,
    )

    is_exhausted = np.full(datasets_count, False)

    # Iterators with zero probability may occurc (e.g., dataset size = 0)
    # Mark these iterators as already exhausted to avoid infinite loops.
    zero_probabilities = np.array(probabilities) == 0
    is_exhausted[zero_probabilities] = np.full_like(is_exhausted[zero_probabilities], True)

    for dataset_idx in indices_iterator:
        try:  # let's pick one sample from the iterator at index i
            dataset_iterator = datasets_iterators[dataset_idx]
            dataset_sample = next(dataset_iterator)

            # dataset_sample = await anext(dataset_iterator)
            # dataset_sample = anext(dataset_iterator)
            # dataset_sample = asyncio.run(dataset_iterator.__anext__())

            #         while True:
            # try:
            #     yield asyncio.run(interleaved_iterator.__anext__())
            # except StopAsyncIteration:
            #     break

            if datasets_prefixes:
                # prepend returned sample with dataset specific value, e.g., dataset IDs.
                yield datasets_prefixes[dataset_idx], dataset_sample
                # yield {"text": dataset_sample, "dataset_id": datasets_prefixes[dataset_idx]}
            else:
                yield dataset_sample

        except (StopIteration, StopAsyncIteration):
            # here it means that the i-th iterabledataset is empty, i.e we never have the occasion to yield an element of the i-th dataset.
            # we still check if the stopping criteria is met and if we break out of the loop in case of an oversampling strategy
            is_exhausted[dataset_idx] = True

            if np.all(is_exhausted):
                # if the stopping criteria is met, break the main for loop
                break


def sampled_dataset_iterator(
    iterator_func, true_dataset_rows, target_dataset_rows, offset, limit, **iterator_func_kwargs
) -> Iterator:
    if target_dataset_rows > true_dataset_rows:
        # upsampling
        full_reads = math.floor(target_dataset_rows / true_dataset_rows)
        last_partial_read = (target_dataset_rows / true_dataset_rows) - full_reads
        last_partial_read_n = math.floor(last_partial_read * true_dataset_rows)

        logger.debug("Upsample dataset. Full reads: %s times; Last read: %s rows", full_reads, last_partial_read_n)

        # full read (no limit): repeated reading from the dataset
        for _ in range(full_reads):
            for sample in iterator_func(offset=offset, limit=limit, **iterator_func_kwargs):
                yield sample

        # last partial read (with limit)
        if last_partial_read_n > 0:
            adjusted_limit = offset + last_partial_read_n
            for sample in iterator_func(offset=offset, limit=adjusted_limit, **iterator_func_kwargs):
                yield sample

    else:
        # no upsampling, simply read with given offset + limit
        adjusted_limit = limit - (true_dataset_rows - target_dataset_rows)

        for sample in iterator_func(offset=offset, limit=adjusted_limit, **iterator_func_kwargs):
            yield sample


class DatasetGenerator(object):
    """
    Dataset generator class

    Generate function needs to be implemented as a class to make the
    collected statistics accessible for an iterator.
    """

    dataset_id_to_stats: dict[str, dict] = {}
    dataset_id_to_args: dict[str, dict] = {}

    list_of_dataset_iterators = []
    list_of_dataset_rows = []
    list_of_dataset_ids = []

    def __init__(
        self,
        config: Config,
        shuffled_output_dir,
        output_format,
        save_to_dir: Union[str, Path],
        split: DatasetSplit = DatasetSplit.TRAIN,
    ) -> None:
        self.config = config
        self.shuffled_output_dir = shuffled_output_dir
        self.output_format = output_format
        self.split = split

        if isinstance(save_to_dir, str):
            save_to_dir = Path(save_to_dir)

        self.save_to_dir = save_to_dir

    def prepare_datasets(self, use_sampling: bool = False, print_progress: bool = False):
        available_dataset_classes = get_registered_dataset_classes(
            extra_dataset_registries=self.config.extra_dataset_registries,
            extra_dataset_classes=self.config.extra_dataset_classes,
            use_default_registry=self.config.use_default_dataset_registry,
        )
        selected_dataset_ids = self.config.selected_dataset_ids
        selected_source_ids = self.config.selected_source_ids

        self.list_of_dataset_rows = []
        self.list_of_dataset_ids = []
        self.dataset_id_to_stats = {}
        self.dataset_id_to_args = {}

        # Construct iterators and get dataset sizes
        if print_progress:
            available_dataset_classes = tqdm(available_dataset_classes, desc="Prepare datasets")

        for ds_cls in available_dataset_classes:
            ds: BaseDataset = ds_cls(
                output_dir=None,
                output_format=self.output_format,
                raw_datasets_dir=None,
                shuffled_output_dir=self.shuffled_output_dir,
                config=self.config,
            )
            dataset_id = ds.DATASET_ID
            source_id = ds.get_source_id()

            if dataset_id in selected_dataset_ids or source_id in selected_source_ids:
                split_offset_limit = get_splits_as_offsets_and_limits(ds, use_shuffled_output=True)
                offset, limit = split_offset_limit[self.split]
                dataset_rows = limit - offset

                if use_sampling:
                    # Expected dataset size depends on sampling factor -> influences probabilty
                    expected_dataset_rows = math.floor(dataset_rows * ds.get_sampling_factor())
                else:
                    expected_dataset_rows = dataset_rows

                if expected_dataset_rows > 0:
                    if dataset_id in self.dataset_id_to_args:
                        raise ValueError(f"Duplicated dataset ID: {dataset_id}")

                    self.dataset_id_to_args[dataset_id] = dict(
                        iterator_func=ds.generate_texts_from_output,
                        true_dataset_rows=dataset_rows,
                        target_dataset_rows=expected_dataset_rows,
                        offset=offset,
                        limit=limit,
                    )
                    self.list_of_dataset_rows.append(expected_dataset_rows)
                    self.list_of_dataset_ids.append(dataset_id)

                    # save stats
                    self.dataset_id_to_stats[dataset_id] = {
                        "dataset_rows": dataset_rows,
                        "expected_dataset_rows": expected_dataset_rows,
                        "split_to_offset_and_limit": split_offset_limit,
                    }
                else:
                    logger.warning("Dataset is zero expected rows: %s", dataset_id)

    def generate_texts_from_single_dataset(
        self,
        dataset_id,
        reader_batch_size: int = 100,
        print_progress: bool = False,
    ) -> Iterator:
        if not self.dataset_id_to_args:
            raise ValueError("Dataset args are not set. Did you run `prepare_datasets`?")

        args = self.dataset_id_to_args[dataset_id]
        dataset_iterator = sampled_dataset_iterator(
            shuffled=True,
            shuffle_output_file_paths=True,
            reader_implementation="pyarrow",
            batch_size=reader_batch_size,
            **args,
        )

        if print_progress:
            dataset_iterator = tqdm(
                dataset_iterator, desc=f"Generating from {dataset_id}", total=args["limit"] - args["offset"]
            )

        yield from dataset_iterator

    def generate_texts_from_interleaved_datasets(
        self,
        random_batch_size: int = 100,
        reader_batch_size: int = 100,
        generate_dataset_ids: bool = False,
        print_progress: bool = False,
    ) -> Iterator:
        """
        Generate texts based on split or full datasets and the given config.
        """
        if not self.dataset_id_to_args:
            raise ValueError("Dataset args are not set. Did you run `prepare_datasets`?")

        self.list_of_dataset_iterators = []

        # Iterate over prepared datasets
        for args in self.dataset_id_to_args.values():
            dataset_iterator = sampled_dataset_iterator(
                shuffled=True,
                shuffle_output_file_paths=True,
                reader_implementation="pyarrow",
                # reader_implementation="polars_read_parquet",
                batch_size=reader_batch_size,
                **args,
            )

            self.list_of_dataset_iterators.append(dataset_iterator)

        assert len(self.list_of_dataset_iterators) == len(self.dataset_id_to_args)
        assert len(self.list_of_dataset_iterators) == len(self.list_of_dataset_ids)
        assert len(self.list_of_dataset_iterators) == len(self.list_of_dataset_rows)

        if len(self.list_of_dataset_iterators) == 0:
            raise ValueError(
                "No dataset has been selected. Did you set the config for `selected_dataset_ids` and `selected_source_ids`?"
            )

        # Random order of datasets
        dataset_indices = list(range(len(self.list_of_dataset_iterators)))
        random.seed(self.config.seed)
        random.shuffle(dataset_indices)

        def select_by_indicies(list, indices):
            return [list[idx] for idx in indices]

        # Compute probabilites
        total_rows = sum(self.list_of_dataset_rows)
        probabilities = [dataset_rows / total_rows for dataset_rows in self.list_of_dataset_rows]

        interleaved_iterator = interleave_dataset_iterators(
            select_by_indicies(self.list_of_dataset_iterators, dataset_indices),
            select_by_indicies(probabilities, dataset_indices),
            datasets_prefixes=pa.array(select_by_indicies(self.list_of_dataset_ids, dataset_indices), type=pa.string())
            if generate_dataset_ids
            else None,
            seed=self.config.seed,
            random_indicies_batch_size=random_batch_size,
        )

        if print_progress:
            interleaved_iterator = tqdm(interleaved_iterator, total=total_rows, desc="Interleave datasets")

        for sample in interleaved_iterator:
            yield sample

    def save_to_path(self, part: Optional[int] = None, total_parts: Optional[int] = None, suffix=""):
        split_str = self.split + suffix
        output_extension = "." + self.output_format

        if total_parts is None:
            fn = f"{split_str}.part-{part:04d}{output_extension}"
        else:
            fn = f"{split_str}.part-{part:04d}-of-{total_parts:04d}{output_extension}"

        return self.save_to_dir / fn


def generate_texts_from_dataset(
    dataset: BaseDataset,
    split: DatasetSplit = None,
    split_offset_limit: Optional[DatasetSplitOffsetLimit] = None,
    use_shuffled_output: bool = True,
):
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

    if split_offset_limit is None:
        split_offset_limit = get_splits_as_offsets_and_limits(dataset, use_shuffled_output=use_shuffled_output)

    dataset_offset, dataset_limit = split_offset_limit[split]

    logger.info("Generating texts for split=%s with offset=%s and limit=%s", split, dataset_offset, dataset_limit)

    yield from dataset.generate_texts_from_output(
        shuffled=use_shuffled_output,
        shuffle_output_file_paths=use_shuffled_output,
        offset=dataset_offset,
        limit=dataset_limit,
    )


def get_splits_as_offsets_and_limits(dataset: BaseDataset, use_shuffled_output: bool = True) -> DatasetSplitOffsetLimit:
    """
    Generate splits as offsets and limit depending on total rows and defined split settings (ratio, min. doc count etc.)
    """
    if dataset.config is None:
        raise ValueError("dataset config is not set")

    n_docs = dataset.get_output_rows_count(shuffled=use_shuffled_output)  # total dataset size
    validation_offset = 0  # validation rows are always the first n rows
    train_limit = n_docs  # train rows are always the last n rows

    # Compute offsets + limits for split selection
    if dataset.HAS_PREDEFINED_VALIDATION_SET:
        logger.warning("No validation split - because a predefined validation set exists: %s", dataset.DATASET_ID)
        validation_n_docs = 0
    elif dataset.config.validation_min_total_docs is not None and n_docs < dataset.config.validation_min_total_docs:
        logger.warning(
            "No validation split for %s - because dataset has too few docs (given: %s; expected: %s).",
            dataset.DATASET_ID,
            n_docs,
            dataset.config.validation_min_total_docs,
        )
        validation_n_docs = 0
    else:
        validation_n_docs = min(
            math.floor(dataset.config.validation_ratio * n_docs),
            dataset.config.validation_max_split_docs if dataset.config.validation_max_split_docs else math.inf,
        )

        if dataset.config.validation_min_split_docs and validation_n_docs < dataset.config.validation_min_split_docs:
            logger.warning(
                "No validation split for %s - because split would be too small (given: %s; expected: %s).",
                dataset.DATASET_ID,
                validation_n_docs,
                dataset.config.validation_min_split_docs,
            )
            validation_n_docs = 0

    if validation_n_docs > 0:
        train_offset = validation_n_docs  # + 1
    else:
        train_offset = 0

    train_n_docs = n_docs - validation_n_docs
    tokenizer_train_n_docs = math.floor(train_n_docs * dataset.config.tokenizer_train_ratio)

    # Training dataset for tokenizer (by default: 10% of training set)
    tokenizer_train_offset = train_offset
    tokenizer_train_limit = train_offset + tokenizer_train_n_docs  # + 1

    logger.debug("validation_n_docs: %s", validation_n_docs)
    logger.debug("train_n_docs: %s", train_n_docs)
    logger.debug("tokenizer_train_n_docs: %s", tokenizer_train_n_docs)

    splits: DatasetSplitOffsetLimit = {
        DatasetSplit.FULL: (0, n_docs),
        DatasetSplit.VALIDATION: (validation_offset, validation_n_docs),
        DatasetSplit.TRAIN: (train_offset, train_limit),
        DatasetSplit.TOKENIZER_TRAIN: (tokenizer_train_offset, tokenizer_train_limit),
    }

    logger.debug("splits: %s", splits)

    return splits
