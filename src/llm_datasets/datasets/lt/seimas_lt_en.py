import zipfile
from llm_datasets.datasets.base import BaseDataset, Genre, License


class SeismasLTENDataset(BaseDataset):
    DATASET_ID = "seimas_lt_en"
    TITLE = "Bilingual English-Lithuanian parallel corpus from Seimas of the Republic of Lithuania website"
    HOMEPAGE = "https://live.european-language-grid.eu/catalogue/corpus/3009/download/"
    DESCRIPTION = "Contents of http://www.lrs.lt were crawled, aligned on document and sentence level and converted into a parallel corpus."
    DOWNLOAD_URLS = [
        "https://elrc-share.eu/repository/download/4486f8e4e72711e7b7d400155d0267060b3d0987d08b43fd9c065ce3f05f99f8"
    ]
    LANGUAGES = ["lt"]
    GENRES = [Genre.GOVERNMENT]
    LICENSE = License("Open under PSI", url="https://elrc-share.eu/terms/openUnderPSI.html")
    BYTES = 160 * 1024

    def get_texts(self):
        from translate.storage.tmx import tmxfile

        zip_fp = self.get_dataset_file_paths(needed_suffix=".zip", single_file=True)

        with zipfile.ZipFile(zip_fp) as zf:
            for fn in zf.namelist():
                if fn.endswith(".tmx"):
                    with zf.open(fn) as member_f:
                        tmx_file = tmxfile(member_f, "lt", "en")

                for i, node in enumerate(tmx_file.unit_iter()):
                    text = node.target  # lt
                    # en => node.source

                    yield text
