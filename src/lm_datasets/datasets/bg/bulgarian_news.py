from lm_datasets.datasets.base import MB, BaseDataset

import logging

logger = logging.getLogger(__name__)


class BulgarianNewsDataset(BaseDataset):
    DATASET_ID = "bulgarian_news"
    DOWNLOAD_URLS = ["http://old.dcl.bas.bg/dataset/Bulgarian_news.7z"]
    DESCRIPTION = (
        "The collection was collected by crawling Bulgarian websites in Bulgarian. Text samples are in json format. We"
        " can provide raw tests."
    )

    LANGUAGES = ["bg"]
    BYTES = 919 * MB

    def decompress(self):
        # 7z x Bulgarian_news.7z
        pass

    def get_texts(self):
        # from py7zr import SevenZipFile

        # read from extracted JSON files
        # fps = self.get_dataset_file_paths(subdirectories=True, needed_suffix=".json")

        # read from 7z
        # fp = self.get_dataset_file_paths(single_file=True, needed_suffix=".7z")
        # logger.info(f"Reading from {fp}")

        # with SevenZipFile(fp, "r") as zip:
        #     fn = "Bulgarian_news/n/nbox.bg/bg_00000634-24dd-4c46-b846-af4e5c3db559.json"
        #     print(fn)

        #     content = zip.read(fn)[fn].read()

        #     # for fname, bio in zip.read(targets=[fn]).items():
        #     #     print(f"{fname}: {bio.read(10)}...")
        #     #     break

        #     # allfiles = zip.getnames()

        # with SevenZipFile(fp, "r") as zip:
        #     for fn in allfiles:
        #         if fn.endswith(".json"):
        #             for fname, bio in zip.read(fn).items():
        #                 print(f"{fname}: {bio.read(10)}...")
        #         break

        # with lzma.open(fp) as f:
        #     file_content = f.read()

        print("x")
