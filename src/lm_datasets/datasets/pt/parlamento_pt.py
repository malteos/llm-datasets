from lm_datasets.datasets.base import Availability, GB
from lm_datasets.datasets.hf_dataset import HFDataset

import logging

logger = logging.getLogger(__name__)


class ParlamentoPtDataset(HFDataset):
    DATASET_ID = "parlamento_pt"
    TITLE = "ParlamentoPT"
    HOMEPAGE = "https://huggingface.co/datasets/PORTULAN/parlamento-pt"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["pt"]

    DESCRIPTION = """
    The ParlamentoPT is a Portuguese language data set obtained by collecting publicly
    available documents containing transcriptions of debates in the Portuguese Parliament. The data was collected from
    the Portuguese Parliament portal in accordance with its open data policy.
    """

    BYTES = 2.6 * GB

    HF_DATASET_ID = "PORTULAN/parlamento-pt"
    HF_DATASET_SPLIT = "train"

    HF_DATASET_CONFIGS = ["PORTULAN--parlamento-pt"]

    text_column_name = "text"
    remove_columns = []


#    title_column_name = "title"
