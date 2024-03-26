
### Download and text extraction

To download and extract the plain-text of one or more datasets, run the following command:

```bash
llm-datasets extract_text $DATASET_ID $OUTPUT_DIR
```

By default, output is saved as JSONL files. To change the output format, you can use the `--output_format` argument as below:

```bash
llm-datasets extract_text $DATASET_ID $OUTPUT_DIR --output_format parquet  --output_compression zstd
```
