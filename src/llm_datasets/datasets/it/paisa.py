import gzip

from llm_datasets.datasets.base import GB, Availability, BaseDataset, License


class PaisaCorpus(BaseDataset):
    DATASET_ID = "paisa"
    TITLE = "PaisaCorpus"
    HOMEPAGE = "http://www.corpusitaliano.it/en/help/getting_started.html"
    LICENSE = License(
        name="Creative Commons",
        url="https://creativecommons.org/licenses/by-nc-sa/3.0/",
    )
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    DOWNLOAD_URLS = [
        "https://clarin.eurac.edu/repository/xmlui/bitstream/handle/20.500.12124/3/paisa.raw.utf8.gz?sequence=1&isAllowed=y"
    ]
    LANGUAGES = ["it"]
    DESCRIPTION = """
    The Paisà corpus is a large collection of Italian web texts, licensed under
    Creative Commons (Attribution-ShareAlike and Attribution-Noncommercial-ShareAlike).
    It has been created in the context of the project PAISÀ.
    """
    BYTES = 2.7 * GB

    def get_texts(self):
        """Reads directly from .gz file.
        Check for "wiki" in the dataset URL to avoid overlapping with other datasets
        """
        from bs4 import BeautifulSoup

        with gzip.open(self.get_dataset_file_paths(single_file=True), "rt", encoding="utf-8") as fin:
            print("Parsing file")
            soup = BeautifulSoup(fin, "lxml")
            print("File ready")
            for text in soup.find_all("text"):
                if "wiki" in text.get("url"):
                    continue
                else:
                    yield text.get_text()
