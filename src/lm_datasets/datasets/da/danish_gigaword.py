from lm_datasets.datasets.base import BILLION, Availability, License
from lm_datasets.datasets.hf_dataset import HFDataset


class DanishGigawordDataset(HFDataset):
    DATASET_ID = "danish_gigaword"
    TITLE = "Danish GigaWord"
    HOMEPAGE = "https://sprogteknologi.dk/dataset/danish-gigaword"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    LICENSE = License(
        "CC-BY 4.0, Creative Commons with Attribution",
        research_use=True,
        commercial_use=True,
        attribution=True,
        sharealike=False,
    )
    LANGUAGES = ["da"]

    DESCRIPTION = (
        "A billion-word corpus of Danish text. Split into many sections, and covering many dimensions ",
        "of variation (spoken/written, formal/informal, modern/old, rigsdansk/dialect, and so on).",
        "",
        "The license is CC-BY 4.0, Creative Commons with Attribution. Owners: ITU; Leon Derczynski, Manuel R. Ciosici",
    )
    CITATION = """@inproceedings{stromberg-derczynski-etal-2021-danish,
    title = "The {D}anish {G}igaword Corpus",
    author = "Str{\o}mberg-Derczynski, Leon  and
      Ciosici, Manuel  and
      Baglini, Rebekah  and
      Christiansen, Morten H.  and
      Dalsgaard, Jacob Aarup  and
      Fusaroli, Riccardo  and
      Henrichsen, Peter Juel  and
      Hvingelby, Rasmus  and
      Kirkedal, Andreas  and
      Kjeldsen, Alex Speed  and
      Ladefoged, Claus  and
      Nielsen, Finn {\AA}rup  and
      Madsen, Jens  and
      Petersen, Malte Lau  and
      Rystr{\o}m, Jonathan Hvithamar  and
      Varab, Daniel",
    booktitle = "Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may # " 31--2 " # jun,
    year = "2021",
    address = "Reykjavik, Iceland (Online)",
    publisher = {Link{\"o}ping University Electronic Press, Sweden},
    url = "https://aclanthology.org/2021.nodalida-main.46",
    pages = "413--421",
    abstract = "Danish language technology has been hindered by a lack of broad-coverage corpora at the scale modern NLP prefers. This paper describes the Danish Gigaword Corpus, the result of a focused effort to provide a diverse and freely-available one billion word corpus of Danish text. The Danish Gigaword corpus covers a wide array of time periods, domains, speakers{'} socio-economic status, and Danish dialects.",
    }
    """  # noqa

    # Size: 1 billion words
    TOKENS = 1 * BILLION

    HF_DATASET_ID = "DDSC/partial-danish-gigaword-no-twitter"
    HF_DATASET_SPLIT = "train"

    excluded_sources = {
        # exclude all Wikimedia sources to avoid duplicated content with the original Wikimedia dataset
        "wiki",
        "wikibooks",
        "wikisource",
    }

    text_column_name = "text"
    remove_columns = ["doc_id", "LICENSE", "uri", "date_built"]

    def get_filter_func(self):
        def filter_func(example):
            return example["source"] not in self.excluded_sources

        return filter_func
