# Getting Started

## Installation

Install the `llm-datasets` package with [pip](https://pypi.org/project/llm-datasets/):

```bash
pip install llm-datasets
```

In order to keep the package minimal by default, `llm-datasets` comes with optional dependencies useful for some use cases.
For example, if you want to have the text extraction for all available datasets, run:

```bash
pip install llm-datasets[datasets]
```

## Quick start

### Download and text extraction

To download and extract the plain-text of one or more datasets, run the following command:

```bash
llm-datasets extract_text $DATASET_ID $OUTPUT_DIR
```

By default, output is saved as JSONL files. To change the output format, you can use the `--output_format` argument as below:

```bash
llm-datasets extract_text $DATASET_ID $OUTPUT_DIR --output_format parquet  --output_compression zstd
```

### Available datasets

A list or table with all available datasets can be print with the follow command:

```bash
llm-datasets print_stats --print_output md
```

### Pipeline commands

```
usage: llm-datasets <command> [<args>]

positional arguments:
  {chunkify,collect_metrics,compose,convert_parquet_to_jsonl,extract_text,hf_upload,print_stats,shuffle,train_tokenizer}
                        llm-datasets command helpers
    chunkify            Split the individual datasets into equally-sized file chunks (based on bytes or rows)
    collect_metrics     Collect metrics (token count etc.) from extracted texts
    compose             Compose the final train/validation set based on the individual datasets
    convert_parquet_to_jsonl
                        Convert Parquet files to JSONL
    extract_text        Extract text from raw datasets
    hf_upload           Upload files or directories to Huggingface Hub.
    print_stats         Print dataset statistics as CSV, Markdown, ...
    shuffle             Shuffle the individual datasets on the file-chunk level (no global shuffle!)
    train_tokenizer     Train a tokenizer (only: sentencepiece supproted)

options:
  -h, --help            show this help message and exit
```