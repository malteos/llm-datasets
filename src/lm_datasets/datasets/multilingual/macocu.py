from io import TextIOWrapper
import logging
from typing import Set
import zipfile
from lm_datasets.datasets.base import BaseDataset, Availability, QualityWarning


logger = logging.getLogger(__name__)


class MacocuBaseDataset(BaseDataset):
    SOURCE_ID = "macocu"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    WEB_CRAWLED = True

    languages_needed: Set[str] = None  # hbs_lat,en,hbs_cyr

    def is_downloaded(self):
        return len(self.get_dataset_file_paths(needed_suffix=".zip")) == len(self.DOWNLOAD_URLS)

    def get_texts(self):
        from lm_datasets.io.prevert_file import PrevertFile

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
    TITLE = "Bulgarian web corpus MaCoCu-bg 2.0"
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
    TITLE = "Croatian MaCoCu corporus"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1806"

    LANGUAGES = ["hr"]

    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1806/MaCoCu-hr-2.0.xml.zip"]

    USED_BY = []

    TOKENS = 2_363_710_130

    QUALITY_WARNINGS = [QualityWarning.BAD_ENCODING]

    languages_needed = {"hr", "hbs_lat", "hbs_cyr"}


class MacocuELDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_el"
    TITLE = "Greek web corpus MaCoCu-el 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1839"
    LANGUAGES = ["el"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1839/MaCoCu-el-1.0.xml.zip"]
    TOKENS = 4_384_614_674

    languages_needed = {"el"}


class MacocuSQDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_sq"
    TITLE = "Albanian web corpus MaCoCu-sq 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1804"
    LANGUAGES = ["sq"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1804/MaCoCu-sq-1.0.xml.zip"]
    TOKENS = 625_726_547

    languages_needed = {"sq"}


class MacocuBSDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_bs"
    TITLE = "Bosnian web corpus MaCoCu-bs 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1808"
    LANGUAGES = ["bs"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1808/MaCoCu-bs-1.0.xml.zip"]
    TOKENS = 730342880

    languages_needed = {"bs"}


class MacocuCADataset(MacocuBaseDataset):
    DATASET_ID = "macocu_ca"
    TITLE = "Catalan web corpus MaCoCu-ca 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1837"
    LANGUAGES = ["ca"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1837/MaCoCu-ca-1.0.xml.zip"]
    TOKENS = 1736578493

    languages_needed = {"ca"}


class MacocuISDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_is"
    TITLE = "Icelandic web corpus MaCoCu-is 2.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1805"
    LANGUAGES = ["is"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1805/MaCoCu-is-2.0.xml.zip"]
    TOKENS = 887198687

    languages_needed = {"is"}


class MacocuMKDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_mk"
    TITLE = "Macedonian web corpus MaCoCu-mk 2.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1801"
    LANGUAGES = ["mk"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1801/MaCoCu-mk-2.0.xml.zip"]
    TOKENS = 524069254

    languages_needed = {"mk"}


class MacocuMTDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_mt"
    TITLE = "Maltese web corpus MaCoCu-mt 2.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1803"
    LANGUAGES = ["mt"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1803/MaCoCu-mt-2.0.xml.zip"]
    TOKENS = 347_855_619

    languages_needed = {"mt"}


class MacocuCNRDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_cnr"
    TITLE = "Montenegrin web corpus MaCoCu-cnr 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1809"
    LANGUAGES = ["cnr"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1809/MaCoCu-cnr-1.0.xml.zip"]
    TOKENS = 161361667

    languages_needed = {"cnr"}


class MacocuSRDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_sr"
    TITLE = "Serbian web corpus MaCoCu-sr 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1807"
    LANGUAGES = ["sr"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1807/MaCoCu-sr-1.0.xml.zip"]
    TOKENS = 2491019884

    languages_needed = {"hbs_lat", "hbs_cyr", "sr"}


class MacocuSLDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_sl"
    TITLE = "Slovene web corpus MaCoCu-sl 2.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1795"
    LANGUAGES = ["sl"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1795/MaCoCu-sl-2.0.xml.zip"]
    TOKENS = 1_920_089_135

    languages_needed = {"sl"}


class MacocuTRDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_tr"
    TITLE = "Turkish web corpus MaCoCu-tr 2.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1802"
    LANGUAGES = ["tr"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1802/MaCoCu-tr-2.0.xml.zip"]
    TOKENS = 4344850253

    languages_needed = {"tr"}


class MacocuUKDataset(MacocuBaseDataset):
    DATASET_ID = "macocu_uk"
    TITLE = "Ukrainian web corpus MaCoCu-uk 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1838"
    LANGUAGES = ["uk"]
    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1838/MaCoCu-uk-1.0.xml.zip"]
    TOKENS = 6181945683

    languages_needed = {"uk"}
