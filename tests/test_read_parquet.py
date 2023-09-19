from pathlib import Path
import pyarrow.parquet as pq
import os
import pytest

from tqdm.auto import tqdm


@pytest.mark.skip(reason="requires dataset files")
def test_read_pq():
    fp = (  # macocu_el.parquet
        "/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts/opengptx_costep_de.parquet"  # noqa
    )
    fp = "/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts/opengptx_dissertations_de.parquet"  # noqa
    fp = "/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts/opengptx_dissertations_de.parquet"  # noqa
    fp = "/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts_shuffled/opengptx_bt_drucksachen.shuffled.parquet"  # noqa

    assert os.path.exists(fp)

    metadata = pq.read_metadata(fp)
    print(metadata)

    parquet_file = pq.ParquetFile(fp)

    rg = parquet_file.read_row_group(0)
    texts = rg["text"]

    print(str(texts[0])[:100])

    print(metadata)


@pytest.mark.skip(reason="requires dataset files")
def test_polars_read():
    fps = [
        "/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts_shuffled/opengptx_bt_drucksachen.shuffled.parquet"  # noqa
    ]  # 6445
    fps = Path("/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts/").glob("*.parquet")
    fps = list(sorted(fps))

    for i, fp in enumerate(tqdm(fps)):
        fp = str(fp)

        print(i, fp)

        if "__bak" in fp or "part-" in fp or i < 137:
            print("-- skip")
            continue

        fp = "/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts/pes2o.parquet"

    print("done")


@pytest.mark.skip(reason="requires dataset files")
def test_parquet_os_error():
    fps = ["/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts/danewsroom.parquet"]

    fps = Path("/netscratch/mostendorff/experiments/eulm/data/lm_datasets_texts/").glob("*.parquet")
    fps = list(sorted(fps))

    for i, fp in enumerate(tqdm(fps)):
        fp = str(fp)

        print(i, fp)

        if "__bak" in fp or "part-" in fp:  # or i < 137:
            print("-- skip")
            continue

        # if ""
        print(fp)

        # import pyarrow as pa
        # t = pq.read_table(fp, memory_map=True)

        ok = 0
        errors = 0

        for i in range(4):
            try:
                # with open(fp, "rb") as f:
                #     pq.read_schema(f)

                with open(fp, "rb") as f:
                    parquet_file = pq.ParquetFile(f)
                    for _, record_batch in enumerate(parquet_file.iter_batches(batch_size=100)):
                        # pa_table = pa.Table.from_batches([record_batch])
                        for text in record_batch["text"]:
                            text = str(text)
                            break
                        break

                # tab = pq.read_table(fp, memory_map=True)
                # for batch in tab.to_batches():
                #     for text in batch["text"]:
                #         text = str(text)
                #         break
                #     break

                # from datasets import load_dataset
                # ds = load_dataset("parquet", data_files={"train": fp+"1"}, split="train", streaming=True)
                # next(iter(ds))

                # with  pq.ParquetFile(fp) as parquet_file:
                #     rg = parquet_file.read_row_group(0)  # columns=["text"]
                #     texts = rg["text"]
                #     text = str(texts[0])

                # parquet_file = pq.ParquetFile(fp)

                # texts = rg["text"]
                # text = str(texts[0])

                # df = pl.scan_parquet(fp)
                # text = df.row(0)

                # print(parquet_file.closed)

                # for batch in parquet_file.iter_batches(row_groups=[0]):
                #     text = str(batch["text"][0])
                #     break

                # if "x" in text:
                #     pass

                ok += 1
                # print(i, "OK")
            except OSError as e:
                errors += 1
                print(i, "ERROR", fp)
                raise e

        print("stats = ", ok, errors, fp)

        if errors:
            print("#### error")

        print("")

    print("done")


def test_read_groups():
    # TODO loop over files and read random groups
    pass


if __name__ == "__main__":
    # test_read_pq()
    test_parquet_os_error()
    # test_polars_read()
