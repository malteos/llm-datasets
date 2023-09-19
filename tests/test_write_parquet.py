import itertools
import pyarrow as pa
import pyarrow.parquet as pq


def test_write_and_shuffle_parquet():
    def generate_texts(n):
        for i in range(n):
            yield f"Some text {i}"

    def get_batches(rows_iterable, chunk_size, schema):
        rows_it = iter(rows_iterable)
        while True:
            arr = pa.array(itertools.islice(rows_it, chunk_size))
            batch = pa.RecordBatch.from_arrays([arr], schema=schema)

            if not batch:
                break
            yield batch

    schema = pa.schema(
        [
            ("text", pa.string()),
        ]
    )

    print("start")

    batches = get_batches(generate_texts(1_00), chunk_size=10, schema=schema)

    print("writing ...")
    # Write the batches
    out_fp = "example.parquet"

    with pq.ParquetWriter(out_fp, schema=schema) as writer:
        for batch in batches:
            print("write batch")
            writer.write_batch(batch)

    print("done")

    # read again
    table = pq.read_table("example.parquet", columns=["text"], memory_map=True)

    assert str(table["text"][99]) == "Some text 99"

    from datasets.arrow_dataset import Dataset

    ds = Dataset(arrow_table=table)
    ds.shuffle(seed=1, keep_in_memory=False).to_parquet("example.shuffled.parquet", batch_size=10)

    shuffled_table = pq.read_table("example.shuffled.parquet", columns=["text"], memory_map=True)
    print(str(shuffled_table["text"][99]))

    assert str(shuffled_table["text"][99]) != "Some text 99"

    print("reading done.")


if __name__ == "__main__":
    test_write_and_shuffle_parquet()
