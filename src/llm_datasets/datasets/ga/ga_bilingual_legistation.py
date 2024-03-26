import logging
from llm_datasets.datasets.base import BaseDataset, MB, Availability, License, QualityWarning


logger = logging.getLogger(__name__)


class GABilingualLegislationDataset(BaseDataset):
    """
    Quality warning: only sentences no documents
    """

    DATASET_ID = "ga_bilingual_legistation"
    TITLE = "The Gaois bilingual corpus of English-Irish legislation (Irish legislation)"
    HOMEPAGE = "https://portulanclarin.net/repository/browse/the-gaois-bilingual-corpus-of-english-irish-legislation-processed/daeac17c9e3511ea9b7f02420a000407b83de243dc0b469aab41084386c5b80f/"  # noqa
    DESCRIPTION = "Bilingual corpus of English-Irish legislation provided by the Department of Justice."
    LICENSE = License("Open Under - PSI", url="https://elrc-share.eu/terms/openUnderPSI.html")
    LANGUAGES = ["ga"]

    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD
    QUALITY_WARNINGS = [QualityWarning.SHORT_TEXT]
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
