import logging

import pandas as pd

from lm_datasets.datasets.base import BaseDataset, GB, Availability, Genre


logger = logging.getLogger(__name__)


class StyriaNewsDataset(BaseDataset):
    """
    https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1410/Styria-articles.csv
    https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1399/Styria-user-comments.zip?sequence=7&isAllowed=y

    """

    DATASET_ID = "styria_news"
    TITLE = "24sata news article archive 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1410"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["hr"]
    GENRES = [Genre.NEWS]

    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1410/Styria-articles.csv"]

    BYTES = 1.3 * GB

    def is_downloaded(self):
        return bool(self.get_dataset_file_paths(single_file=True))

    def get_texts(self):
        if not self.is_downloaded():
            self.download()

        # read CSV in chunks
        for df in pd.read_csv(self.get_dataset_file_paths(single_file=True), chunksize=10_000):
            for title, content in zip(df.title, df.content):
                text = f"{title}: {content}"

                yield text
