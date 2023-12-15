import asyncio
import os
import types
import pyarrow.parquet as pq
import logging

import itertools
from typing import Any, Generator, Iterable, Iterator, List, Optional, Tuple, Union

import pyarrow as pa
import polars as pl

from itertools import islice


logger = logging.getLogger(__name__)


def open_parquet_file_with_retries(file_path, retries: int = 2):
    """
    A little hack to avoid the "[Errno 14] Error reading bytes from file. Detail: [errno 14] Bad address"
    """
    for retry in range(1, retries):
        try:
            f = pq.ParquetFile(file_path)
            return f
        except OSError as e:
            logger.error(f"Could not open parquet file due to `Bad address` error (retry {retry}/{retries}): {e}")
            pass

    # last try that does not catch the error
    f = pq.ParquetFile(file_path)
    return f


def chunked(generator, size):
    """Read parts of the generator, pause each time after a chunk"""
    # islice returns results until 'size',
    # make_chunk gets repeatedly called by iter(callable).
    if isinstance(generator, types.GeneratorType):
        gen = iter(generator)
    else:
        # async genator
        gen = aiter(generator)

    make_chunk = lambda: list(islice(gen, size))
    return iter(make_chunk, [])


# def get_parquet_batches(rows_iterator: Iterator[Union[str, dict]], schema, batch_size):
#     async_iterator = get_async_parquet_batches(rows_iterator, schema, batch_size)
#     while True:
#         try:
#             yield asyncio.run(async_iterator.__anext__())
#         except StopAsyncIteration:
#             break

# def sync_iter(asyn_iter):
#     while True:
#         try:
#             yield asyncio.run(asyn_iter.__anext__())
#         except StopAsyncIteration:
#             break


def get_parquet_batches(rows_iterator: Iterator[Union[str, dict]], schema, batch_size):
    batched_columns = [[] for _ in range(len(schema.names))]

    # rows_iterator = sync_iter(rows_iterator)

    for py_row in rows_iterator:
        # while True:
        #     py_row = await anext(rows_iterator)

        if isinstance(py_row, tuple):
            for col_idx in range(len(schema.names)):
                batched_columns[col_idx].append(py_row[col_idx])
        else:
            # the only column is the text column
            batched_columns[0].append(py_row)

        if len(batched_columns[0]) == batch_size:
            yield pa.RecordBatch.from_arrays(
                [pa.array(batched_column) for batched_column in batched_columns], schema=schema
            )

            # reset
            batched_columns = [[] for _ in range(len(schema.names))]

    # last batch
    if len(batched_columns[0]) > 0:
        yield pa.RecordBatch.from_arrays(
            [pa.array(batched_column) for batched_column in batched_columns], schema=schema
        )

    # else:
    #     # text and non-text columns
    #     arrays = [pa.array([row[col] for row in py_batch]) for col in range(len(schema.names))]

    # pa_batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
    # pa_batch = pa.RecordBatch.from_pylist(py_batch, schema=schema)

    # yield pa_batch

    # while True:
    #     arr = pa.array(itertools.islice(texts, batch_size))
    #     batch = pa.RecordBatch.from_arrays([arr], schema=schema)

    #     if not batch:
    #         break
    #     yield batch


def save_texts_to_parquet_chunks(
    texts: Generator[str, Any, None],
    schema,
    output_path_func: callable,
    max_chunk_uncompressed_bytes: Optional[int] = None,
    max_chunk_rows: Optional[int] = None,
    compression: str = "ZSTD",
    batch_size: int = 1024,
    print_write_progress: int = 10_000,
    limit: int = 0,
) -> Tuple[int, int]:
    return write_to_parquet_chunks(
        data=get_parquet_batches(texts, schema=schema, batch_size=batch_size),
        schema=schema,
        output_path_func=output_path_func,
        max_chunk_uncompressed_bytes=max_chunk_uncompressed_bytes,
        max_chunk_rows=max_chunk_rows,
        compression=compression,
        print_write_progress=print_write_progress,
        limit=limit,
    )


def write_to_parquet_chunks(
    data: Iterable[pa.RecordBatch],
    schema,
    output_path_func: callable,
    max_chunk_uncompressed_bytes: Optional[int] = None,
    max_chunk_rows: Optional[int] = None,
    compression: str = "ZSTD",
    print_write_progress: int = 10_000,
    limit: int = 0,
) -> Tuple[int, int]:
    """
    This could be replaced by `pa.write_dataset` -- however their implementation
    does not support crtl+c (?) thus we do our own implementation.
    """
    max_chunks = 9999
    chunk_rows = None
    chunk_fp = None
    total_rows = 0
    total_bytes = 0
    # batch_iter = get_parquet_batches(texts, schema=schema, batch_size=batch_size)
    limit_reached = False

    if max_chunk_uncompressed_bytes is not None and max_chunk_rows is not None:
        raise ValueError("Cannot set both `max_chunk_uncompressed_bytes` and `max_chunk_rows`")
    elif max_chunk_uncompressed_bytes or max_chunk_rows:
        do_chunks = True
    else:
        do_chunks = False

    for chunk_i in range(1, max_chunks + 1):
        chunk_rows = 0
        chunk_nbytes = 0
        # chunk_buffer_size = 0
        chunk_fp = output_path_func(chunk_i) if do_chunks else output_path_func()

        logger.info(f"Writing to {chunk_fp}")

        with pq.ParquetWriter(chunk_fp, schema=schema, compression=compression) as writer:
            try:
                while True:
                    batch = next(data)

                    writer.write_batch(batch)
                    total_rows += len(batch)
                    total_bytes = batch.nbytes
                    chunk_rows += len(batch)
                    chunk_nbytes += batch.nbytes
                    # chunk_buffer_size += batch.get_total_buffer_size()

                    if total_rows > 0 and print_write_progress > 0 and (total_rows % print_write_progress) == 0:
                        logger.info(f"Written %s rows ...", total_rows)

                    if limit > 0 and total_rows >= limit:
                        logger.warning(f"Limit reached (%s rows)", total_rows)
                        limit_reached = True
                        break

                    if (max_chunk_uncompressed_bytes is not None and chunk_nbytes >= max_chunk_uncompressed_bytes) or (
                        max_chunk_rows is not None and chunk_rows >= max_chunk_rows
                    ):
                        # if chunk_buffer_size >= max_chunk_bytes_with_safety:
                        logger.info(f"Chunk {chunk_i} completed (rows: {chunk_rows:,}; nbytes: {chunk_nbytes:,})")
                        # buffer size: {chunk_buffer_size:,})"
                        # logger.info(f"Chunk size on disk: {os.stat(chunk_fp).st_size:,} bytes")
                        break

            except (StopIteration, StopAsyncIteration):
                logger.info(f"All rows written ({total_rows=}; {total_bytes=})")
                break

        if limit_reached:
            # break outer loop
            break

    total_chunks = chunk_i

    if chunk_rows == 0:
        logger.warning("Last chunk is empty. Removing file: %s", chunk_fp)
        os.remove(chunk_fp)
        total_chunks -= 1

    if do_chunks:
        # Rename files with total number of chunks
        for chunk_i in range(1, total_chunks + 1):
            new_chunk_file_path = output_path_func(chunk_i, total_chunks)
            logger.info("Renaming to %s", new_chunk_file_path)
            os.rename(output_path_func(chunk_i), new_chunk_file_path)

    return total_rows, total_chunks


def get_selected_row_groups(
    parquet_file: pq.ParquetFile, file_offset: int, file_limit: int
) -> Tuple[List[int], Union[dict, None]]:
    """
    Find the row groups that should be read for `file_offset` and `file_limit`
    """
    group_idx_to_offset_last_row: dict[int, Tuple[int, int]] = {}
    row_groups = []
    group_offset = 0

    logger.debug("Selecting the row groups within offset=%s and limit=%s", file_offset, file_limit)

    if file_limit == 0 and file_offset == 0:
        return list(range(parquet_file.num_row_groups)), None

    # Iterate over all row groups
    for group_idx in range(parquet_file.num_row_groups):
        # Fetch row group metadata
        group_metadata = parquet_file.metadata.row_group(group_idx)
        group_num_rows = group_metadata.num_rows
        group_last_row = group_offset + group_num_rows
        group_idx_to_offset_last_row[group_idx] = (group_offset, group_last_row)

        logger.debug("Row group #%s: %s - %s", group_idx, group_offset, group_last_row)

        # Row indicies of group based be part of selected rows
        if (file_offset >= group_offset and file_offset < group_last_row) or (
            file_offset < group_offset and file_limit > group_offset
        ):
            row_groups.append(group_idx)

        # increase offset for next group
        group_offset = group_last_row

    logger.debug("Selected row groups: %s", row_groups)

    return row_groups, group_idx_to_offset_last_row
