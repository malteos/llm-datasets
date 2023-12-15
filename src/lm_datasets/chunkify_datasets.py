import os
from pathlib import Path

from lm_datasets.utils.config import Config


from .datasets.dataset_registry import get_registered_dataset_classes
from .datasets.base import BaseDataset

from .utils import get_bytes_from_int_or_string, get_parquet_compression

import pyarrow.parquet as pq


def validate_original_and_parts():
    # TODO make sure original and parts are the same
    pass


def iter_parquet(pq_file, batch_size):
    # tbl = pq.ParquetFile(file_path)
    for batch in pq_file.iter_batches(batch_size):
        yield batch


def chunkify_datasets(config: Config):
    logger = config.init_logger(__name__)

    if config.output_format != "parquet":
        raise ValueError("Output format is not supported (currently only parquet is supported)")

    id_to_dataset_class = {cls.DATASET_ID: cls for cls in get_registered_dataset_classes()}

    datasets_list = config.datasets.split(",")

    if len(datasets_list) == 1 and datasets_list[0] == "all":
        # Get list of all non-dummy datasets
        datasets_list = id_to_dataset_class.keys()

    max_uncompressed_bytes_per_chunk = get_bytes_from_int_or_string(config.max_uncompressed_bytes_per_chunk)
    max_chunk_bytes_with_safety = int(max_uncompressed_bytes_per_chunk * config.safety_factor)
    logger.info(f"Max. chunk size with {config.safety_factor=}: {max_chunk_bytes_with_safety:,} bytes")

    # Iterate over datasets
    for i, dataset_id in enumerate(datasets_list, 1):
        if dataset_id in id_to_dataset_class:
            dataset_cls = id_to_dataset_class[dataset_id]
        else:
            raise ValueError(f"Unknown dataset ID: {dataset_id} (available: {id_to_dataset_class.keys()})")

        logger.info(f"Dataset ID: {dataset_id} ({i} / {len(datasets_list)})")

        dataset: BaseDataset = dataset_cls(
            output_dir=config.output_dir,
            output_format=config.output_format,
        )

        if dataset.has_chunked_output_files():
            logger.warning(f"Skipping {dataset_id}: dataset has already chunked output")

        if not dataset.has_single_output_file():
            logger.warning(f"Skipping {dataset_id}: cannot split dataset without processed output")
            continue

        logger.info(f"Splitting {dataset.get_single_output_file_path()}")

        # File size
        file_stats = os.stat(dataset.get_single_output_file_path())

        if file_stats.st_size > max_uncompressed_bytes_per_chunk:
            # File is too large => split into chunks
            output_file_path = Path(dataset.get_single_output_file_path())
            output_dir = output_file_path.parent

            # Check for existing chunks
            existing_chunk_fp_pattern = str(output_file_path.name).replace(".parquet", ".part-*.parquet")
            existing_chunk_fps = list(output_dir.glob(existing_chunk_fp_pattern))

            if existing_chunk_fps:
                if config.override:
                    logger.info("Removing existing part files ...")
                    for fp in existing_chunk_fps:
                        logger.info(f"Removing {fp} ...")
                        os.unlink(fp)
                else:
                    logger.warning("Skipping because part files already exist.")

            # Open parquet file
            # pq_file = pq.ParquetFile(dataset.get_output_file_path())
            with open(dataset.get_single_output_file_path(), "rb") as file_handler:
                # pq_file = open_parquet_file_with_retries(dataset.get_output_file_path(), retries=2)
                pq_file = pq.ParquetFile(file_handler)
                batch_iter = iter_parquet(pq_file, config.batch_size)  # iterate over row groups and batches

                # Auto-detect compression from source file
                if config.output_compression is None:
                    compression = pq_file.metadata.row_group(0).column(0).compression
                else:
                    compression = get_parquet_compression(config.output_compression)

                logger.info(f"Unsplitted file size: {file_stats.st_size:,} bytes ({compression=})")

                max_chunks = 9999  # max number of chunks
                total_docs_count = 0
                chunk_fps = []

                for chunk_i in range(1, max_chunks + 1):
                    # if total_docs_count >= pq_file.metadata.num_rows:
                    #     logger.warning("All docs written")
                    #     break
                    chunk_docs_count = 0
                    chunk_nbytes = 0
                    chunk_buffer_size = 0
                    chunk_fp = dataset.get_single_output_file_path().replace(".parquet", f".part-{chunk_i:04d}.parquet")
                    chunk_fps.append(chunk_fp)

                    logger.info(f"Writing to {chunk_fp}")

                    with pq.ParquetWriter(
                        chunk_fp, schema=pq_file.schema.to_arrow_schema(), compression=compression
                    ) as writer:
                        try:
                            while True:
                                batch = next(batch_iter)
                                writer.write_batch(batch)
                                total_docs_count += len(batch)
                                chunk_docs_count += len(batch)
                                chunk_nbytes += batch.nbytes
                                chunk_buffer_size += batch.get_total_buffer_size()

                                if chunk_nbytes >= max_chunk_bytes_with_safety:
                                    # if chunk_buffer_size >= max_chunk_bytes_with_safety:
                                    logger.info(
                                        f"Chunk {chunk_i} completed (docs: {chunk_docs_count:,}; nbytes:"
                                        f" {chunk_nbytes:,}; buffer size: {chunk_buffer_size:,})"
                                    )
                                    # logger.info(f"Chunk size on disk: {os.stat(chunk_fp).st_size:,} bytes")
                                    break

                        except StopIteration:
                            logger.info(f"All rows written ({total_docs_count=})")
                            break

                if total_docs_count != pq_file.metadata.num_rows:
                    logger.error(
                        f"Written rows do not match input rows: {total_docs_count:,} (expected:"
                        f" {pq_file.metadata.num_rows:,})"
                    )

                # Rename part files with total number of chunks
                total_chunks = len(chunk_fps)

                for chunk_fp in chunk_fps:
                    new_chunk_fp = chunk_fp.replace(".parquet", f"-of-{total_chunks:04d}.parquet")

                    logger.info(f"Renaming {new_chunk_fp}")
                    os.rename(chunk_fp, new_chunk_fp)

                # Keep original: by renaming to ".orignal"
                if config.rename_original:
                    new_path = str(output_file_path).replace(".parquet", ".parquet.original")
                    logger.info(f"Moving original file to {new_path}")
                    os.rename(str(output_file_path), new_path)

        else:
            # Small fie
            logger.info("File is small no split neded.")

    logger.info("Done")
