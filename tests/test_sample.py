import polars as pl


shuffled_pq_file_path = (
    "/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts_shuffled/pes2o.part-0026-of-0026.shuffled.parquet"
)
shuffled_pq_file_path = "/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts_shuffled/redpajama_book.part-0011-of-0011.shuffled.parquet"

train_file_path = "./train.parquet"
val_file_path = "./val.parquet"

split_at_row_n = 10

# first n-rows
train_df = pl.scan_parquet(
    shuffled_pq_file_path,
    low_memory=True,
    n_rows=split_at_row_n,
).collect(streaming=True)

# write
train_df.write_parquet(train_file_path, compression="zstd")

# compare original and written file
for i, row in enumerate(train_df.iter_rows()):
    break


[row[0] for row in df.iter_rows()]


df.write_parquet(train_file_path, compression="zstd")
# offset by n-rows: row_count_offset

print("done")
