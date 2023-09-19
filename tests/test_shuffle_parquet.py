import os
import pytest

DATASET_FILE_PATH = (
    "/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts_shuffled/opengptx_bt_drucksachen.shuffled.parquet"
)


@pytest.mark.skipif(not os.path.exists(DATASET_FILE_PATH), reason="test file not exists")
def test_shuffle_parquet():
    import polars as pl

    fp = DATASET_FILE_PATH

    df = pl.scan_parquet(fp)

    df = df.with_columns(pl.all().shuffle(seed=1)).with_row_count()
    train_fraction: float = 0.75

    df_train = df.filter(pl.col("row_nr") < pl.col("row_nr").max() * train_fraction)
    df_test = df.filter(pl.col("row_nr") >= pl.col("row_nr").max() * train_fraction)

    df_train.collect().write_parquet("./shuffled_train.parquet")

    df_train.sink_parquet("./shuffled_train.parquet")
    df_test.write_parquet("./shuffled_train.parquet")

    df = df.sample(fraction=1, shuffle=True)

    print("done")


# df.write_parquet(shuffled_output_file_path)
