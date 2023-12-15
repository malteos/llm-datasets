import json
import logging
import zipfile

from lm_datasets.datasets.base import BaseDataset, License


logger = logging.getLogger(__name__)


class EkspressDataset(BaseDataset):
    DATASET_ID = "ekspress"
    TITLE = "Ekspress news article archive (only Estonian) 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1408"
    DESCRIPTION = (
        "The dataset is an archive of articles from the Ekspress Meedia news site from 2009-2019, "
        "containing over 1.4M articles, mostly in Estonian language (1,115,120 articles) with some "
        " in Russian (325,952 articles)."
    )
    LICENSE = License(
        "Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)",
        commercial_use=False,
        attribution=True,
        derivates=False,
    )

    LANGUAGES = ["et"]

    CITATION = """@inproceedings{pollak-etal-2021-embeddia,
    title = "{EMBEDDIA} Tools, Datasets and Challenges: Resources and Hackathon Contributions",
    author = {Pollak, Senja  and
      Robnik-{\v{S}}ikonja, Marko  and
      Purver, Matthew  and
      Boggia, Michele  and
      Shekhar, Ravi  and
      Pranji{\'c}, Marko  and
      Salmela, Salla  and
      Krustok, Ivar  and
      Paju, Tarmo  and
      Linden, Carl-Gustav  and
      Lepp{\"a}nen, Leo  and
      Zosa, Elaine  and
      Ul{\v{c}}ar, Matej  and
      Freienthal, Linda  and
      Traat, Silver  and
      Cabrera-Diego, Luis Adri{\'a}n  and
      Martinc, Matej  and
      Lavra{\v{c}}, Nada  and
      {\v{S}}krlj, Bla{\v{z}}  and
      {\v{Z}}nidar{\v{s}}i{\v{c}}, Martin  and
      Pelicon, Andra{\v{z}}  and
      Koloski, Boshko  and
      Podpe{\v{c}}an, Vid  and
      Kranjc, Janez  and
      Sheehan, Shane  and
      Boros, Emanuela  and
      Moreno, Jose G.  and
      Doucet, Antoine  and
      Toivonen, Hannu},
    booktitle = "Proceedings of the EACL Hackashop on News Media Content Analysis and Automated Report Generation",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.hackashop-1.14",
    pages = "99--109",
    abstract = "This paper presents tools and data sources collected and released by the EMBEDDIA project, supported by the European Union{'}s Horizon 2020 research and innovation program. The collected resources were offered to participants of a hackathon organized as part of the EACL Hackashop on News Media Content Analysis and Automated Report Generation in February 2021. The hackathon had six participating teams who addressed different challenges, either from the list of proposed challenges or their own news-industry-related tasks. This paper goes beyond the scope of the hackathon, as it brings together in a coherent and compact form most of the resources developed, collected and released by the EMBEDDIA project. Moreover, it constitutes a handy source for news media industry and researchers in the fields of Natural Language Processing and Social Science.",
}"""  # noqa

    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1408/ee_articles_2009-2019.zip"]

    def is_downloaded(self):
        return len(self.get_dataset_file_paths(needed_suffix=".zip")) == len(self.DOWNLOAD_URLS)

    def get_texts(self):
        if not self.is_downloaded():
            self.download()

        # iterate over archives
        for archive_fp in self.get_dataset_file_paths(needed_suffix=".zip"):
            with zipfile.ZipFile(archive_fp, "r") as zf:
                for zfn in zf.namelist():
                    if zfn.endswith(".json"):
                        logger.info(f"Reading from {zfn}")

                        with zf.open(zfn) as file:
                            data = json.load(file)
                            for article in data:
                                if article["channelLanguage"] == "nat":  # only include estonian
                                    text = article["bodyText"]

                                    yield text
