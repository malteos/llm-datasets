import json
from llm_datasets.datasets.base import BaseDataset, Availability, License
import zipfile
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class YLENewsDataset(BaseDataset):
    DATASET_ID = "ylenews"
    TITLE = "Yle Finnish News Archive"
    HOMEPAGE = "http://urn.fi/urn:nbn:fi:lb-2021050401"
    DESCRIPTION = "The corpus, containing the articles from YLE https://yle.fi from 2019 and 2020, is available at www.kielipankki.fi/download"
    LICENSE = License(
        "CLARIN ACA - NC (Academic - Non Commercial Use, Attribution, No Redistribution, Other)",
        url="https://www.clarin.eu/content/licenses-and-clarin-categories",
        commercial_use=False,
        research_use=True,
        distribution=False,
    )
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
