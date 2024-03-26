from llm_datasets.datasets.base import Availability, BILLION, License
from llm_datasets.datasets.hf_dataset import HFDataset


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
    CITATION = r"""@inproceedings{wagner2018brwac,
  title={The brwac corpus: A new open resource for brazilian portuguese},
  author={Wagner Filho, Jorge A and Wilkens, Rodrigo and Idiart, Marco and Villavicencio, Aline},
  booktitle={Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
"""  # noqa
    LICENSE = License("research-only", commercial_use=False, research_use=True)

    TOKENS = 2.6 * BILLION

    HF_DATASET_ID = "brwac"
    HF_DATASET_SPLIT = "train"
    HF_DATA_DIR = "/netscratch/ortiz/corpora/ELE/pt/brwac/"

    text_column_name = "text"
    keep_columns = True
    streaming = True
    #  title_column_name = "title" ## do not use auto-concatenate

    def get_text_from_item(self, item) -> str:
        paragraphs = [self.sentence_delimiter.join(sentences) for sentences in item["text"]["paragraphs"]]
        text = item["title"] + self.title_delimiter + (self.paragraph_delimiter.join(paragraphs))

        return text
