import logging

import pandas as pd

from llm_datasets.datasets.base import BaseDataset, GB, Availability, Genre, License


logger = logging.getLogger(__name__)


class StyriaNewsDataset(BaseDataset):
    """
    https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1410/Styria-articles.csv
    https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1399/Styria-user-comments.zip?sequence=7&isAllowed=y

    """

    DATASET_ID = "styria_news"
    TITLE = "24sata news article archive 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1410"
    DESCRIPTION = "The 24sata news portal consists of a portal with daily news and several smaller portals covering news from specific topics, such as automotive news, health, culinary content, and lifestyle advice. The dataset contains over  650,000 articles in Croatian from 2007 to 2019, as well as assigned tags."
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    LICENSE = License(
        "Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)",
        url="https://creativecommons.org/licenses/by-nc-nd/4.0/",
        attribution=True,
        commercial_use=False,
        derivates=False,
        sharealike=False,
    )
    CITATION = r"""@misc{11356/1410,
        title = {24sata news article archive 1.0},
        author = {Purver, Matthew and Shekhar, Ravi and Pranji{\'c}, Marko and Pollak, Senja and Martinc, Matej},
        url = {http://hdl.handle.net/11356/1410},
        note = {Slovenian language resource repository {CLARIN}.{SI}},
        copyright = {Creative Commons - Attribution-{NonCommercial}-{NoDerivatives} 4.0 International ({CC} {BY}-{NC}-{ND} 4.0)},
        issn = {2820-4042},
        year = {2021} }"""
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
