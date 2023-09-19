from pathlib import Path
from lm_datasets.datasets.base import BaseDataset, Availability
from lm_datasets.datasets.hf_dataset import HFDataset


# files already downloaded => use parquet DS instead!
class StarcoderHFDataset(HFDataset):
    DATASET_ID = "starcoder"

    TITLE = "StarCoder"
    HOMEPAGE = "https://huggingface.co/datasets/bigcode/starcoderdata"
    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD

    LANGUAGES = ["code"]

    TOKENS = 250_000_000_000

    HF_DATASET_ID = "bigcode/starcoderdata"
    HF_DATASET_SPLIT = "train"

    text_column_name = "text"
    # remove_columns = ["doc_id", "LICENSE", "uri", "date_built"]


class ParquetDataset(BaseDataset):
    SINGLE_OUTPUT_FILE = False

    def extract_plaintext(self):
        if self.output_format == "parquet":
            raise ValueError("Dataset is already in parquet format; no text extraction needed!")
        else:
            super().extract_plaintext()


class StarcoderDataset(ParquetDataset):
    """
    Output files:
    /netscratch/ortiz/corpora/starcoder/starcoderdata/<code_lang>/train-<i>-of-<n>.parquet
    """

    DATASET_ID = "starcoder"

    TITLE = "StarCoder"
    HOMEPAGE = "https://huggingface.co/datasets/bigcode/starcoderdata"
    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD

    LANGUAGES = ["code"]

    TOKENS = 250_000_000_000

    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/starcoder/starcoderdata"]

    HF_DATASET_ID = "bigcode/starcoderdata"
    HF_DATASET_SPLIT = "train"

    text_column_name = "text"
    # remove_columns = ["doc_id", "LICENSE", "uri", "date_built"]

    def get_output_file_paths(self):
        return Path(self.get_local_dataset_dir()).glob("*.parquet")
