import json
from lm_datasets.datasets.base import BaseDataset, Availability
import zipfile
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class YLENewsDataset(BaseDataset):
    DATASET_ID = "ylenews"
    TITLE = "Yle Finnish News Archive"
    HOMEPAGE = "http://urn.fi/urn:nbn:fi:lb-2021050401"

    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD

    LANGUAGES = ["fi"]

    def get_texts(self):
        """ """
        zip_fps = self.get_dataset_file_paths(needed_suffix=".zip")

        for zip_fp in zip_fps:
            logger.info(f"Extracting from {zip_fp}")
            with zipfile.ZipFile(zip_fp) as zf:
                member_fns = zf.namelist()
                for fn in tqdm(member_fns):
                    if fn.endswith(".json"):
                        with zf.open(fn) as member_f:
                            obj = json.load(member_f)
                            # data = obj["data"]

                            for doc in obj["data"]:
                                # doc_texts = []
                                doc_text = ""
                                for content in doc["content"]:
                                    if content["type"] == "text":
                                        doc_text += content["text"].rstrip()
                                        doc_text += " "

                                    elif content["type"] == "heading":
                                        if doc_text:
                                            doc_text += self.paragraph_delimiter

                                        doc_text += content["text"] + self.title_delimiter

                                # doc_text = "\n\n".join(doc_texts)
                                yield doc_text

                                # stop after doc
                                # break
                        # stop after file
                        # break
            # stop after archive
            # break
