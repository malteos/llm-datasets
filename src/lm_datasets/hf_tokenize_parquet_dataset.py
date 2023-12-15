from dataclasses import dataclass, field
from itertools import chain
import logging
import os
from pathlib import Path
import sys
from typing import Optional
from transformers import AutoTokenizer
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from tqdm.auto import tqdm
import signal

import multiprocessing
from pathlib import Path
from typing import Iterable, List

from itertools import chain

import pyarrow as pa
from pyarrow.dataset import dataset as pa_dataset

from pyarrow.dataset import write_dataset, ParquetFileFormat
from pyarrow import RecordBatch

from transformers import AutoTokenizer
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments for this script
    """

    tokenizer_name_or_path: str = field(
        metadata={"help": ("Tokenizer name or path")},
    )
    dataset_dir: str = field(
        metadata={"help": ("Path to input dataset (where *.parquet files are located, see dataset_glob)")},
    )
    output_path_prefix: str = field(
        metadata={
            "help": (
                "Tokenized dataset is saved to this file path prefix on the disk (to each file name `.part-*-of-*.parquet` is appended)"
            )
        },
    )
    dataset_glob: str = field(
        default="*.parquet",
        metadata={"help": ("Glob for selecting files from datset dir")},
    )
    # output_format: Literal["jsonl", "hf"] = field(
    #     default="jsonl",
    #     metadata={"help": ("Output format")},
    # )
    # composed_dataset_split: Literal["train", "validation"] = field(
    #     default="train",
    #     metadata={"help": ("Split of composed dataset")},
    # )
    text_column_name: str = field(
        default="text",
        metadata={"help": ("Name of text column in input dataaset")},
    )
    max_seq_length: int = field(
        default=None,
        metadata={"help": ("Max. input sequence length (default = from tokenizer/model settings)")},
    )
    limit: int = field(
        default=0,
        metadata={"help": ("Limits number of input examples (only for debugg; 0 = no limit)")},
    )
    batch_size: int = field(
        default=10_000,
        metadata={"help": ("Batch size of reading/processing/writting data")},
    )
    output_max_rows_per_file: int = field(
        default=1024 * 1024,
        metadata={"help": ("Output max_rows_per_file")},
    )
    output_max_rows_per_group: int = field(
        default=10 * 1024,
        metadata={"help": ("Output max_rows_per_group:")},
    )
    num_proc: int = field(
        default=None,
        metadata={"help": ("Number of parallel processes (default all cpus)")},
    )
    # print_progress: int = field(
    #     default=10_000,
    #     metadata={"help": ("Print progress after every N examples")},
    # )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": ("Enable fast tokenizer (produces sometimes different results)")},
    )
    do_group: bool = field(
        default=False,
        metadata={"help": ("Group tokenized samples into same-length samples based on `max_seq_length`")},
    )
    # skip_tokenization: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "Skip tokenization step (apply only grouping)"
    #         )
    #     },
    # )
    override: bool = field(
        default=False,
        metadata={"help": ("Override existing output")},
    )
    return_special_tokens_mask: bool = field(
        default=False,
        metadata={"help": ("return_special_tokens_mask (for MLM data)")},
    )
    return_token_type_ids: bool = field(
        default=False,
        metadata={"help": ("return_token_type_ids (for MLM data)")},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": ("Verbose logging (debug)")},
    )


def generate_texts(
    pyarrow_dataset,
    reader_batch_size: int,
    print_progress: bool = True,
    row_limit: Optional[int] = None,
    text_column_name: str = "text",
) -> Iterable[List[str]]:
    """
    Reads PyArrow dataset in batches and generates texts
    """

    batch_iter = pyarrow_dataset.to_batches(columns=[text_column_name], batch_size=reader_batch_size)
    max_batches = round(row_limit / reader_batch_size) if row_limit is not None and row_limit > 0 else None

    if print_progress:
        # get total number of batches
        total_rows = pyarrow_dataset.count_rows()
        logger.info("Total input dataset rows: %i", total_rows)

        total_batches = round(total_rows / reader_batch_size)

        if max_batches is not None and max_batches > 0:
            total_batches = min(total_batches, max_batches)

        batch_iter = tqdm(batch_iter, total=total_batches, desc="Iterating over input")

    for batch_i, batch in enumerate(batch_iter):
        texts = batch[0].tolist()

        logger.debug("texts loaded %i", len(texts))

        yield texts

        if max_batches is not None and batch_i >= max_batches:
            break

    logger.info("Dataset completed.")


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    if args.do_group and args.max_seq_length is None:
        raise ValueError("Cannot `do_group` while `max_seq_length` is none.")

    # if os.path.exists(args.output_path) and not args.override:
    #     raise FileExistsError("Output path exists already: %s", args.output_path
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Settings
    output_path_prefix = Path(args.output_path_prefix)
    do_group = args.do_group
    text_column_name = args.text_column_name
    max_seq_length = args.max_seq_length
    return_special_tokens_mask = args.return_special_tokens_mask  # for MLM
    return_token_type_ids = args.return_token_type_ids
    print_progress = True
    batch_size = args.batch_size
    num_proc = args.num_proc
    override = args.override
    limit = args.limit

    # Output data schema
    pa_columns = [
        ("input_ids", pa.list_(pa.uint32())),
        ("attention_mask", pa.list_(pa.uint8())),
    ]
    if return_token_type_ids:
        pa_columns.append(("token_type_ids", pa.list_(pa.uint8())))

    if return_special_tokens_mask:
        pa_columns.append(("special_tokens_mask", pa.list_(pa.uint8())))

    tokenized_data_schema = pa.schema(pa_columns)

    logger.info(
        "Initializing dataset from %s (glob: %s)",
        args.dataset_dir,
        args.dataset_glob,
    )
    dataset_file_paths = list(sorted(Path(args.dataset_dir).glob(args.dataset_glob)))

    if not dataset_file_paths:
        raise FileNotFoundError(f"No files found in dataset dir: {args.dataset_dir} (glob: {args.dataset_glob})")

    logger.info("Files found in dataset dir: %i", len(dataset_file_paths))
    ds = pa_dataset(source=dataset_file_paths, format="parquet")

    logger.info("Initialize tokenizer from %s", args.tokenizer_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        fast=args.use_fast_tokenizer,
    )

    def group_texts(examples):
        """
        Main data processing function that will concatenate all texts from our dataset and generate chunks of max_seq_length.
        """
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        logger.debug("grouped: %i", len(result["input_ids"]))

        # This is done on the model level
        # if causal_lm:
        #     result["labels"] = result["input_ids"].copy()

        return result

    def tokenize_texts(list_of_text: List[str]) -> dict:
        """
        Perform the actual tokenization (is called by worker threads)

        Tokenized data is grouped if `do_group` is enabled.
        """
        tokenizer_out = tokenizer(list_of_text, return_special_tokens_mask=return_special_tokens_mask)
        logger.debug("tokenized: %i", len(list_of_text))

        if do_group:
            tokenizer_out = group_texts(tokenizer_out)

        return tokenizer_out

    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def tokenize_in_parallel(pyarrow_text_dataset, pyarrow_schema=None, processes=None) -> Iterable[RecordBatch]:
        with multiprocessing.Pool(processes=processes, initializer=init_worker) as pool:
            try:
                for grouped_tokenized_batch in pool.imap(
                    tokenize_texts,
                    generate_texts(
                        pyarrow_text_dataset,
                        reader_batch_size=batch_size,
                        print_progress=print_progress,
                        row_limit=limit,
                        text_column_name=text_column_name,
                    ),
                    chunksize=1,
                ):
                    record_batch = RecordBatch.from_pydict(grouped_tokenized_batch, schema=pyarrow_schema)

                    logger.debug("record_batch: %i", len(record_batch))

                    yield record_batch

                logger.info("All texts tokenized")

                # close pool
                # pool.terminate()
                # pool.join()
                pool.close()
                pool.join()

            except KeyboardInterrupt:
                logger.error("Stoppping ...")

                pool.terminate()
                pool.join()

    # Output settings
    output_base_dir = output_path_prefix.parent
    output_file_name_prefix = output_path_prefix.name

    file_options = ParquetFileFormat().make_write_options(compression="zstd")
    output_file_names = []  # written file names are saved here via `file_visitor`

    def file_visitor(written_file):
        """
        Keep track of written files (for later renaming)
        """
        file_name = Path(written_file.path).name

        logger.debug("Writing to %s", file_name)

        output_file_names.append(file_name)

    # Write from iterator
    write_dataset(
        data=tokenize_in_parallel(
            pyarrow_text_dataset=ds,
            pyarrow_schema=tokenized_data_schema,
            processes=num_proc,
        ),
        base_dir=output_base_dir,
        basename_template=output_file_name_prefix + ".part-{i}.parquet",
        format="parquet",
        file_options=file_options,
        # partitioning="",
        max_rows_per_group=args.output_max_rows_per_group,
        max_rows_per_file=args.output_max_rows_per_file,
        create_dir=True,
        schema=tokenized_data_schema,
        file_visitor=file_visitor,
        # create_dir=True,
        existing_data_behavior="overwrite_or_ignore"
        if override
        else "error",  # error, overwrite_or_ignore, delete_matching
    )

    # rename files with total number of parts
    for part, output_file_name in enumerate(output_file_names):
        # replace: .part-{i}. with part-{i:04d}-of-{len(output_file_names):04d}.
        new_file_name = output_file_name.replace(
            f".part-{part}.", f".part-{1+part:04d}-of-{len(output_file_names):04d}."
        )
        logger.info(f"Renaming {output_file_name} to {new_file_name}")

        os.rename(
            os.path.join(output_base_dir, output_file_name),
            os.path.join(output_base_dir, new_file_name),
        )

    logger.info("done")
