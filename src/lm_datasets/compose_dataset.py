import json
from pathlib import Path
from lm_datasets.io.parquet import save_texts_to_parquet_chunks
from datetime import datetime
import pyarrow as pa
from lm_datasets.utils import get_bytes_from_int_or_string

from lm_datasets.utils.config import Config
from lm_datasets.utils.dataset_generator import DatasetGenerator, DatasetSplit


def compose_dataset(config: Config):
    logger = config.init_logger(__name__)

    logger.info("Compose dataset")

    max_output_chunk_uncompressed_bytes = get_bytes_from_int_or_string(config.max_output_chunk_uncompressed_bytes)
    # split_str = config.split
    save_to_dir = Path(config.composed_dataset_dir)
    output_compression = "ZSTD"
    output_text_field = "text"
    output_dataset_id_field = "dataset_id"
    output_format = "parquet"

    if not save_to_dir.exists():
        logger.info("Save directory did not exist, creating it at %s", save_to_dir)
        save_to_dir.mkdir()

    # Initialize dataset generator (interleaves pre-shuffled datasets)
    dataset_generator = DatasetGenerator(
        config,
        shuffled_output_dir=config.shuffled_output_dir,
        output_format=output_format,
        save_to_dir=save_to_dir,
        split=config.split,
    )
    dataset_generator.prepare_datasets(
        use_sampling=config.use_sampling,
        print_progress=True,
    )

    # By default, the only column is the text column
    parquet_columns = [
        (output_text_field, pa.string()),
    ]
    if config.save_dataset_ids and config.split != DatasetSplit.VALIDATION:
        # Add `dataset_id` column at in the first place (ID are prepended)
        parquet_columns.insert(0, (output_dataset_id_field, pa.string()))

    parquet_schema = pa.schema(parquet_columns)

    if config.split == DatasetSplit.VALIDATION and config.use_separated_validation_sets:
        # Validation splits are saved into separate files (based on dataeset ID)
        for dataset_id in dataset_generator.list_of_dataset_ids:

            def validation_output_path(part, total_parts=None):
                return dataset_generator.save_to_path(part, total_parts, "_" + dataset_id)

            save_texts_to_parquet_chunks(
                dataset_generator.generate_texts_from_single_dataset(
                    dataset_id,
                    print_progress=True,
                    reader_batch_size=config.input_batch_size,
                ),
                parquet_schema,
                output_path_func=validation_output_path,
                max_chunk_uncompressed_bytes=max_output_chunk_uncompressed_bytes,
                compression=output_compression,
                batch_size=config.output_batch_size,
                print_write_progress=0,
                limit=config.limit,
            )
    else:
        # Save texts from iterator into a single (chunked) file to disk
        save_texts_to_parquet_chunks(
            dataset_generator.generate_texts_from_interleaved_datasets(
                generate_dataset_ids=config.save_dataset_ids,
                print_progress=True,
                random_batch_size=config.interleave_random_batch_size,
                reader_batch_size=config.input_batch_size,
            ),
            parquet_schema,
            output_path_func=dataset_generator.save_to_path,
            max_chunk_uncompressed_bytes=max_output_chunk_uncompressed_bytes,
            compression=output_compression,
            batch_size=config.output_batch_size,
            print_write_progress=0,
            limit=config.limit,
        )

    # Save stats
    stats = {
        "config": config.__dict__,
        "dataset_id_to_stats": dataset_generator.dataset_id_to_stats,
        "datetime": str(datetime.now()),
    }
    stats_path = save_to_dir / f"{config.split}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    logger.info("done")
