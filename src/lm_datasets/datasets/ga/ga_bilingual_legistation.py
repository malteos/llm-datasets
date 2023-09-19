import logging
from lm_datasets.datasets.base import BaseDataset, MB, Availability


logger = logging.getLogger(__name__)


class GABilingualLegislationDataset(BaseDataset):
    """

    TODO only sentences no documents

    """

    DATASET_ID = "ga_bilingual_legistation"
    TITLE = "Irish legislation"
    HOMEPAGE = "https://portulanclarin.net/repository/browse/the-gaois-bilingual-corpus-of-english-irish-legislation-processed/daeac17c9e3511ea9b7f02420a000407b83de243dc0b469aab41084386c5b80f/"  # noqa

    LANGUAGES = ["ga"]

    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    DOWNLOAD_URLS = []

    USED_BY = ["gaBERT"]

    BYTES = 0.5 * 25 * MB

    def get_texts(self):
        from translate.storage.tmx import tmxfile

        with open(self.get_dataset_file_paths(single_file=True, needed_suffix=".tmx"), "rb") as fin:
            tmx_file = tmxfile(fin, "ga", "en")

        for i, node in enumerate(tmx_file.unit_iter()):
            text = node.source  # ga
            # en => node.target
            yield text
