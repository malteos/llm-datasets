from lm_datasets.datasets.base import Availability, BILLION
from lm_datasets.datasets.hf_dataset import HFDataset


class BrWacDataset(HFDataset):
    """
    Download instructions:

    You need to
    1. Manually download `brwac.vert.gz` from https://www.inf.ufrgs.br/pln/wiki/index.php?title=BrWaC
    2. Extract the brwac.vert.gz in; this will result in the file brwac.vert in a folder <path/to/folder>
    The <path/to/folder> can e.g. be `~/Downloads`.
    BrWaC can then be loaded using the following command `datasets.load_dataset("brwac", data_dir="<path/to/folder>")`.

    """

    DATASET_ID = "brwac"

    TITLE = "Brazilian Portuguese Web as Corpus"
    HOMEPAGE = "https://huggingface.co/datasets/brwac"
    AVAILIBILITY = Availability.ON_REQUEST

    LANGUAGES = ["pt"]

    DESCRIPTION = (
        "The BrWaC (Brazilian Portuguese Web as Corpus) is a large corpus constructed following"
        "the Wacky framework, which was made public for research purposes."
        "The current corpus version, released in January 2017, is composed by 3.53 million documents,"
        "2.68 billion tokens and 5.79 million types. Please note that this resource is available"
        "solely for academic research purposes, and you agreed not to use it for any commercial applications."
    )

    TOKENS = 2.6 * BILLION

    HF_DATASET_ID = "brwac"
    HF_DATASET_SPLIT = "train"
    HF_DATA_DIR = "/netscratch/ortiz/corpora/ELE/pt/"

    text_column_name = "text"
    remove_columns = ["doc_id", "uri"]

    title_column_name = "title"

    def get_text_from_item(self, item) -> str:
        return item["title"] + self.title_delimiter + (self.paragraph_delimiter.join(item["text"]["paragraphs"]))
