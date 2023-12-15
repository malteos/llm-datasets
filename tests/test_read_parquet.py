import asyncio
import itertools
import logging
from pathlib import Path
import tempfile
import time
import pyarrow as pa
import pyarrow.parquet as pq
from lm_datasets.io.parquet import save_texts_to_parquet_chunks

from lm_datasets.utils.config import Config, get_config_from_paths
from lm_datasets.utils.dataset_generator import DatasetGenerator, DatasetSplit

logger = logging.getLogger(__name__)


def sync_dataset_iterator(file_path, batch_size, output_text_field):
    with open(file_path, "rb") as file_handler:
        pq_file = pq.ParquetFile(file_handler)
        logger.info("Generate from %s with bs=%i", file_path, batch_size)

        for pq_batch in pq_file.iter_batches(columns=[output_text_field], batch_size=batch_size, use_threads=False):
            # logger.info(f"{file_path} , {len(pq_batch[0])=}")
            for v in pq_batch[0]:
                yield v


async def async_dataset_iterator(file_path, batch_size, output_text_field):
    with open(file_path, "rb") as file_handler:
        pq_file = pq.ParquetFile(file_handler)
        logger.info("Generate from %s with bs=%i", file_path, batch_size)

        for pq_batch in pq_file.iter_batches(columns=[output_text_field], batch_size=batch_size, use_threads=False):
            # logger.info(f"{file_path} , {len(pq_batch[0])=}")
            for v in pq_batch[0]:
                yield v


def sync_convert_to_batches(iterable, batch_size):
    py_batch = []

    for sample in iterable:
        py_batch.append(sample)

        if len(py_batch) >= batch_size:
            yield pa.array(py_batch)
            py_batch = []  # reset

    if py_batch:
        yield pa.array(py_batch)


async def async_convert_to_batches(iterable, batch_size):
    py_batch = []

    async for sample in iterable:
        py_batch.append(sample)

        if len(py_batch) >= batch_size:
            yield pa.array(py_batch)
            py_batch = []  # reset

    if py_batch:
        yield pa.array(py_batch)


# async def test_read_parquet():
#     output = []

#     async for batch in convert_to_batches(generate_from_parquet(), batch_size=5):
#         output.append(batch)

#     logger.info(f"{len(output)=}")


async def async_generate_from_parquet(file_limit):
    fps = list(sorted(Path("/data/datasets/lm-datasets_data/euro_dataset_v1_shuffled").glob("*.parquet")))[
        :file_limit
    ]  # colossal_oscar_2023-
    print(f"{len(fps)=}")
    output_text_field = "text"
    batch_size = 10

    dataset_iterators = [async_dataset_iterator(file_path, batch_size, output_text_field) for file_path in fps]

    for dataset_idx in itertools.cycle(range(len(dataset_iterators))):
        try:
            iterator = dataset_iterators[dataset_idx]
            # sample = next(iterator)
            # await sample = iterator.__anext__()
            sample = await anext(iterator)

            yield sample
        except (StopIteration, StopAsyncIteration):
            break


def sync_generate_from_parquet(file_limit):
    fps = list(sorted(Path("/data/datasets/lm-datasets_data/euro_dataset_v1_shuffled").glob("*.parquet")))[
        :file_limit
    ]  # colossal_oscar_2023-
    print(f"{len(fps)=}")
    output_text_field = "text"
    batch_size = 10

    dataset_iterators = [sync_dataset_iterator(file_path, batch_size, output_text_field) for file_path in fps]

    for dataset_idx in itertools.cycle(range(len(dataset_iterators))):
        try:
            iterator = dataset_iterators[dataset_idx]
            sample = next(iterator)
            # await sample = iterator.__anext__()
            # sample = await anext(iterator)

            yield sample
        except (StopIteration, StopAsyncIteration):
            break


async def async_save_texts(text_iterator, limit):
    output = []

    async for batch in async_convert_to_batches(text_iterator, batch_size=5):
        output.append(batch)

        if len(output) > limit:
            print("limit reached")
            break

    logger.info(f"{len(output)=}")


def sync_save_texts(text_iterator, limit):
    output = []

    for batch in sync_convert_to_batches(text_iterator, batch_size=5):
        output.append(batch)

        if len(output) > limit:
            print("limit reached")
            break

    logger.info(f"{len(output)=}")


async def async_generate_and_save(limit, file_limit):
    await async_save_texts(
        async_generate_from_parquet(file_limit),
        limit,
    )


def sync_generate_and_save(limit, file_limit):
    sync_save_texts(
        sync_generate_from_parquet(file_limit),
        limit,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # level=logging.DEBUG,
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    start_time = time.perf_counter()
    file_limit = 100
    limit = 100_000
    # asyncio.run(async_generate_and_save(limit, file_limit))
    sync_generate_and_save(limit, file_limit)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    logger.info("done")

    logger.info(f"{start_time=}")
    logger.info(f"{end_time=}")
    logger.info(f"{elapsed_time=:0.2f} seconds")  # approx. 20 seconds

    # limit=10000
    # file_limit=20
    # elapsed_time=8.72 seconds --- sync
    # elapsed_time=9.03 seconds -- async

    # limit=100_000
    # file_limit=100
    # async:  elapsed_time=18.46 seconds
    # sync: elapsed_time=18.17 seconds
