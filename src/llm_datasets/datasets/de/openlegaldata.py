from llm_datasets.datasets.base import Availability, Genre, License
from llm_datasets.datasets.jsonl_dataset import JSONLDataset


class OpenLegalDataDataset(JSONLDataset):
    DATASET_ID = "openlegaldata"
    TITLE = "Open Legal Data - German court decisions and laws"
    HOMEPAGE = "https://openlegaldata.io/"
    DESCRIPTION = "OPENLEGALDATA.IO is a free and open platform that makes legal documents and information accessible to the public. "
    CITATION = r"""@inproceedings{10.1145/3383583.3398616,
        author = {Ostendorff, Malte and Blume, Till and Ostendorff, Saskia},
        title = {Towards an Open Platform for Legal Information},
        year = {2020},
        isbn = {9781450375856},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3383583.3398616},
        doi = {10.1145/3383583.3398616},
        abstract = {Recent advances in the area of legal information systems have led to a variety of applications that promise support in processing and accessing legal documents. Unfortunately, these applications have various limitations, e.g., regarding scope or extensibility. Furthermore, we do not observe a trend towards open access in digital libraries in the legal domain as we observe in other domains, e.g., economics of computer science. To improve open access in the legal domain, we present our approach for an open source platform to transparently process and access Legal Open Data. This enables the sustainable development of legal applications by offering a single technology stack. Moreover, the approach facilitates the development and deployment of new technologies. As proof of concept, we implemented six technologies and generated metadata for more than 250,000 German laws and court decisions. Thus, we can provide users of our platform not only access to legal documents, but also the contained information.},
        booktitle = {Proceedings of the ACM/IEEE Joint Conference on Digital Libraries in 2020},
        pages = {385-388},
        numpages = {4},
        keywords = {open source, open data, legal information system, legal data},
        location = {Virtual Event, China},
        series = {JCDL '20}
        }
    """
    VERSION = "oldp_cases_v1_v2_fix"  # "oldp-cases_merged_v1_v2"  # dump date: 2022-10-18

    LANGUAGES = ["de"]
    GENRES = [Genre.LEGAL]
    AVAILIBILITY = Availability.ON_REQUEST
    LICENSE = License(
        "public domain",
        url="https://www.gesetze-im-internet.de/urhg/__5.html",
        commercial_use=True,
        sharealike=False,
        research_use=True,
    )

    TOKENS = 9_700_000_000

    raw_jsonl_paths = [
        "v1.jsonl.gz",
        "v2.jsonl.gz",
        # "merged_v1_v2.jsonl.gz",
        # "merged_v1_v2.jsonl",
    ]

    doc_ids = set()

    def get_text_from_item(self, item):
        # filter by doc ID
        if item["doc_id"] in self.doc_ids:
            return None
        else:
            self.doc_ids.add(item["doc_id"])
            return item[self.raw_jsonl_text_field]
