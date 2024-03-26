from io import TextIOWrapper
import logging
from typing import Set
import zipfile
from llm_datasets.datasets.base import BaseDataset, Availability, QualityWarning, License


logger = logging.getLogger(__name__)


class MacocuBaseDataset(BaseDataset):
    SOURCE_ID = "macocu"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    WEB_CRAWLED = True
    LICENSE = License(
        "CC0-No Rights Reserved",
        commercial_use=True,
        research_use=True,
        sharealike=False,
        attribution=False,
        distribution=True,
    )
    DESCRIPTION = "MaCoCu focuses on collecting monolingual and parallel data from the Internet, specially for under-resourced languages and DSI-specific data. See https://macocu.eu/"
    CITATION = r"""@inproceedings{non-etal-2022-macocu,
        title = "{M}a{C}o{C}u: Massive collection and curation of monolingual and bilingual data: focus on under-resourced languages",
        author = "Ba{\~n}{\'o}n, Marta  and
        Espl{\`a}-Gomis, Miquel  and
        Forcada, Mikel L.  and
        Garc{\'\i}a-Romero, Cristian  and
        Kuzman, Taja  and
        Ljube{\v{s}}i{\'c}, Nikola  and
        van Noord, Rik  and
        Sempere, Leopoldo Pla  and
        Ram{\'\i}rez-S{\'a}nchez, Gema  and
        Rupnik, Peter  and
        Suchomel, V{\'\i}t  and
        Toral, Antonio  and
        van der Werff, Tobias  and
        Zaragoza, Jaume",
        booktitle = "Proceedings of the 23rd Annual Conference of the European Association for Machine Translation",
        month = jun,
        year = "2022",
        address = "Ghent, Belgium",
        publisher = "European Association for Machine Translation",
        url = "https://aclanthology.org/2022.eamt-1.41",
        pages = "303--304",
        abstract = "We introduce the project {``}MaCoCu: Massive collection and curation of monolingual and bilingual data: focus on under-resourced languages{''}, funded by the Connecting Europe Facility, which is aimed at building monolingual and parallel corpora for under-resourced European languages. The approach followed consists of crawling large amounts of textual data from carefully selected top-level domains of the Internet, and then applying a curation and enrichment pipeline. In addition to corpora, the project will release successive versions of the free/open-source web crawling and curation software used.",
    }"""  # noqa

    languages_needed: Set[str] = None  # hbs_lat,en,hbs_cyr

    def is_downloaded(self):
        return len(self.get_dataset_file_paths(needed_suffix=".zip")) == len(self.DOWNLOAD_URLS)

    def get_texts(self):
        from llm_datasets.io.prevert_file import PrevertFile

        if not self.is_downloaded():
            self.download()

        if self.languages_needed is None:
            raise ValueError("languages_needed is not set")

        archive_path = self.get_dataset_file_paths(single_file=True, needed_suffix=".zip")

        logger.info(f"Extracting from {archive_path} ...")

        with zipfile.ZipFile(archive_path) as zf:
            member_fns = zf.namelist()
            for fn in member_fns:
                if fn.endswith(".xml"):
                    with zf.open(fn) as member_f:
                        logger.info(f"Parsing {fn} ...")

                        dset = PrevertFile(TextIOWrapper(member_f))

                        for doc in dset:  # iterating through documents of a dataset
                            lang = eval(doc.meta["lang_distr"])[0][0]  # hbs_lat,en,hbs_cyr
                            self.counter.update({f"lang_{lang}": 1})
                            text = str(doc)

                            if self.languages_needed is not None and lang in self.languages_needed:
                                yield text


class MacocuBGDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_bg"
    TITLE = "MaCoCu web corpus [Bulgarian 2.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1800"

    LANGUAGES = ["bg"]

    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1800/MaCoCu-bg-2.0.xml.zip"]

    TOKENS = 3_506_223_084

    languages_needed = {"bg"}


class MacocuHRDataset(MacocuBaseDataset):
    """
    Partially bad quality

    TODO maybe encoding error
    """

    DATASET_ID = "macocu_hr"
    TITLE = "MaCoCu web corpus [Croatian]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1806"

    LANGUAGES = ["hr"]

    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1806/MaCoCu-hr-2.0.xml.zip"]

    USED_BY = []

    TOKENS = 2_363_710_130

    QUALITY_WARNINGS = [QualityWarning.BAD_ENCODING]

    languages_needed = {"hr", "hbs_lat", "hbs_cyr"}


class MacocuELDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_el"
    TITLE = "MaCoCu web corpus [Greek 1.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1839"
    LANGUAGES = ["el"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1839/MaCoCu-el-1.0.xml.zip"]
    TOKENS = 4_384_614_674

    languages_needed = {"el"}


class MacocuSQDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_sq"
    TITLE = "MaCoCu web corpus [Albanian 1.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1804"
    LANGUAGES = ["sq"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1804/MaCoCu-sq-1.0.xml.zip"]
    TOKENS = 625_726_547

    languages_needed = {"sq"}


class MacocuBSDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_bs"
    TITLE = "MaCoCu web corpus [Bosnian 1.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1808"
    LANGUAGES = ["bs"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1808/MaCoCu-bs-1.0.xml.zip"]
    TOKENS = 730342880

    languages_needed = {"bs"}


class MacocuCADataset(MacocuBaseDataset):
    DATASET_ID = "macocu_ca"
    TITLE = "MaCoCu web corpus [Catalan 1.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1837"
    LANGUAGES = ["ca"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1837/MaCoCu-ca-1.0.xml.zip"]
    TOKENS = 1736578493

    languages_needed = {"ca"}


class MacocuISDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_is"
    TITLE = "MaCoCu web corpus [Icelandic 2.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1805"
    LANGUAGES = ["is"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1805/MaCoCu-is-2.0.xml.zip"]
    TOKENS = 887198687

    languages_needed = {"is"}


class MacocuMKDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_mk"
    TITLE = "MaCoCu web corpus [Macedonian 2.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1801"
    LANGUAGES = ["mk"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1801/MaCoCu-mk-2.0.xml.zip"]
    TOKENS = 524069254

    languages_needed = {"mk"}


class MacocuMTDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_mt"
    TITLE = "MaCoCu web corpus [Maltese 2.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1803"
    LANGUAGES = ["mt"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1803/MaCoCu-mt-2.0.xml.zip"]
    TOKENS = 347_855_619

    languages_needed = {"mt"}


class MacocuCNRDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_cnr"
    TITLE = "MaCoCu web corpus [Montenegrin 1.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1809"
    LANGUAGES = ["cnr"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1809/MaCoCu-cnr-1.0.xml.zip"]
    TOKENS = 161361667

    languages_needed = {"cnr"}


class MacocuSRDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_sr"
    TITLE = "MaCoCu web corpus [Serbian 1.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1807"
    LANGUAGES = ["sr"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1807/MaCoCu-sr-1.0.xml.zip"]
    TOKENS = 2491019884

    languages_needed = {"hbs_lat", "hbs_cyr", "sr"}


class MacocuSLDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_sl"
    TITLE = "MaCoCu web corpus [Slovene 2.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1795"
    LANGUAGES = ["sl"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1795/MaCoCu-sl-2.0.xml.zip"]
    TOKENS = 1_920_089_135

    languages_needed = {"sl"}


class MacocuTRDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_tr"
    TITLE = "MaCoCu web corpus [Turkish 2.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1802"
    LANGUAGES = ["tr"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1802/MaCoCu-tr-2.0.xml.zip"]
    TOKENS = 4344850253

    languages_needed = {"tr"}


class MacocuUKDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_uk"
    TITLE = "MaCoCu web corpus [Ukrainian 1.0]"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1838"
    LANGUAGES = ["uk"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1838/MaCoCu-uk-1.0.xml.zip"]
    TOKENS = 6181945683

    languages_needed = {"uk"}
