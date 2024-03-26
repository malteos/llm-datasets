import logging
from typing import Iterable
from smart_open import open

from llm_datasets.datasets.base import BaseDataset, Availability, License
from tqdm.auto import tqdm

from llm_datasets.io.conllu_file import get_texts_from_conllu_file

logger = logging.getLogger(__name__)


class CurlicatBaseDataset(BaseDataset):
    """
    Licenses are mostly mixed. See https://aclanthology.org/2022.lrec-1.11.pdf
    """

    SOURCE_ID = "curlicat"
    TITLE = "CURLICAT Corpus"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    CITATION = r"""@inproceedings{varadi-etal-2022-introducing,
        title = "Introducing the {CURLICAT} Corpora: Seven-language Domain Specific Annotated Corpora from Curated Sources",
        author = "V{\'a}radi, Tam{\'a}s  and
        Ny{\'e}ki, Bence  and
        Koeva, Svetla  and
        Tadi{\'c}, Marko  and
        {\v{S}}tefanec, Vanja  and
        Ogrodniczuk, Maciej  and
        Nito{\'n}, Bartlomiej  and
        Pezik, Piotr  and
        Barbu Mititelu, Verginica  and
        Irimia, Elena  and
        Mitrofan, Maria  and
        Tufi{\textcommabelow{s}}, Dan  and
        Garab{\'\i}k, Radovan  and
        Krek, Simon  and
        Repar, Andra{\v{z}}",
        editor = "Calzolari, Nicoletta  and
        B{\'e}chet, Fr{\'e}d{\'e}ric  and
        Blache, Philippe  and
        Choukri, Khalid  and
        Cieri, Christopher  and
        Declerck, Thierry  and
        Goggi, Sara  and
        Isahara, Hitoshi  and
        Maegaard, Bente  and
        Mariani, Joseph  and
        Mazo, H{\'e}l{\`e}ne  and
        Odijk, Jan  and
        Piperidis, Stelios",
        booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
        month = jun,
        year = "2022",
        address = "Marseille, France",
        publisher = "European Language Resources Association",
        url = "https://aclanthology.org/2022.lrec-1.11",
        pages = "100--108",
        abstract = "This article presents the current outcomes of the CURLICAT CEF Telecom project, which aims to collect and deeply annotate a set of large corpora from selected domains. The CURLICAT corpus includes 7 monolingual corpora (Bulgarian, Croatian, Hungarian, Polish, Romanian, Slovak and Slovenian) containing selected samples from respective national corpora. These corpora are automatically tokenized, lemmatized and morphologically analysed and the named entities annotated. The annotations are uniformly provided for each language specific corpus while the common metadata schema is harmonised across the languages. Additionally, the corpora are annotated for IATE terms in all languages. The file format is CoNLL-U Plus format, containing the ten columns specific to the CoNLL-U format and three extra columns specific to our corpora as defined by Var{\'a}di et al. (2020). The CURLICAT corpora represent a rich and valuable source not just for training NMT models, but also for further studies and developments in machine learning, cross-lingual terminological data extraction and classification.",
    }
    """  # noqa
    DESCRIPTION = "The CURLICAT corpus includes 7 monolingual corpora (Bulgarian, Croatian, Hungarian, Polish, Romanian, Slovak and Slovenian) containing selected samples from respective national corpora."

    conllup_encoding = "utf-8"

    def download(self):
        """
        DOWNLOAD
        -----------

        Instruction

        - Run the command on the server:

        wget -O file.zip \
            "https://elrc-share.eu/repository/download/fed6af2a590311ed9c1a00155d0267062ed273d01d2343f1b78d08d4d481679d/"

        - Extract the archive:

        unzip file.zip

        """  # noqa
        pass

    def decompress(self):
        # unzip
        pass

    def get_curclicat_file_paths(self):
        return self.get_dataset_file_paths(needed_suffix=".conllup")

    def get_texts(self) -> Iterable[str]:
        """
        Extracts the text from each conllu file.
        """

        # TODO read directly from compressed files
        fps = self.get_curclicat_file_paths()
        for i, fp in tqdm(enumerate(fps), total=len(fps), desc="Reading files"):
            # doc = []
            if self.skip_items > 0 and i < self.skip_items:
                logger.warning(f"Skipping {fp}")
                continue

            with open(fp, "r", encoding=self.conllup_encoding) as f:
                yield from get_texts_from_conllu_file(
                    f,
                    title_delimiter=self.title_delimiter,
                    sentence_delimiter=self.sentence_delimiter,
                    skip_sentence_prefixes=["tf.Tensor("],
                )


class CurlicatBGDataset(CurlicatBaseDataset):
    DATASET_ID = "curlicat_bg"
    TITLE = "CURLICAT Corpus [Bulgarian]"
    # DESCRIPTION = """The Bulgarian CURLICAT corpus consists of texts from different sources, provided with  appropriate licences for distribution. We used three general types of sources with regard to the metadata extraction: Bulgarian National Corpus (provided that they have redistributable licensing terms); some public repositories with open and copyright free data; blogs with redistributable licenses, open content websites, etc. The Bulgarian CURLICAT collection contains 113 087 documents, distributed in seven thematic domains: Culture, Education, European Union, Finance, Politics, Economics, and Science. For more information see the CURLICAT website (http:curlicat-project.eu/deliverables) """  # noqa
    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-bulgarian-corpus/fed6af2a590311ed9c1a00155d0267062ed273d01d2343f1b78d08d4d481679d/"  # noqa
    LANGUAGES = ["bg"]
    TOKENS = 35_319_695
    LICENSE = License(
        name="CC-BY-SA-4.0",
        url="https://elrc-share.eu/static/metashare/licences/CC-BY-SA-4.0.pdf",
        attribution=True,
        sharealike=True,
    )


class CurlicatHRDataset(CurlicatBaseDataset):
    """
    unzip and move to raw dir
    find hr_curlicat_2023-01-13 -name '*.*' -exec mv {} /home/mostendorff/experiments/eulm/data/ele/hr/curlicat_hr;
    """

    DATASET_ID = "curlicat_hr"
    TITLE = "CURLICAT Corpus [Croatian]"
    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-croatian-corpus/00815518592811ed9c1a00155d026706bc4c59740fce4f7986213e7eef133023/"  # noqa
    LANGUAGES = ["hr"]
    DOWNLOAD_URLS = ["https://zzl.ffzg.unizg.hr/files/5cd9d6151612489fafb2c1f116140519/hr_curlicat_2023-01-13.zip"]
    TOKENS = 49_007_508
    LICENSE = License("unknown")


class CurlicatHUDataset(CurlicatBaseDataset):
    DATASET_ID = "curlicat_hu"
    TITLE = "CURLICAT Corpus [Hungarian]"
    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-hungarian-corpus/8b6c8dcb58ea11ed9c1a00155d02670679a453431c8147079e5a7d9b879a9729/"  # noqa
    LANGUAGES = ["hu"]
    TOKENS = 61_196_946
    LICENSE = License(
        "CC-BY-SA-4.0",
        url="https://elrc-share.eu/static/metashare/licences/CC-BY-SA-4.0.pdf",
        attribution=True,
        sharealike=True,
    )

    def get_curclicat_file_paths(self):
        # Hugarian files are stored in *.txt files subdirectories
        return self.get_dataset_file_paths(subdirectories=True, needed_suffix=".txt")


class CurlicatPLDataset(CurlicatBaseDataset):
    """
    unzip and move to raw dir
    find pl-20221026-annotated -name '*.*' -exec mv {} /home/mostendorff/experiments/eulm/data/ele/pl/curlicat_pl;
    """

    DATASET_ID = "curlicat_pl"
    TITLE = "CURLICAT Corpus [Polish]"
    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-polish-corpus/f63ae912553911ed9c1a00155d02670648c0a234e0314895b52169af2af57dd7/"  # noqa
    LANGUAGES = ["pl"]
    TOKENS = 59_301_782

    LICENSE = License(
        name="CC-BY-SA-4.0",
        url="https://elrc-share.eu/static/metashare/licences/CC-BY-SA-4.0.pdf",
        attribution=True,
        sharealike=True,
    )


class CurlicatRODataset(CurlicatBaseDataset):
    DATASET_ID = "curlicat_ro"
    TITLE = "CURLICAT Corpus [Romanian]"
    # DESCRIPTION = """The corpus contains 26,477 files, which represent our contribution to the CURLICAT project. It contains texts from 7 domains: science, politics, culture, economy, health, education, nature. Each file has multiple levels of annotation: tokenized, lemmatized, morphologically annotated, dependency parsed, named entities, nominal phrases, IATE terms and automatic domain-specific terms were identified as well. All processing tools are available within the RELATE platform. The corpus was automatically anonymized. Alternate download location: https://relate.racai.ro/resources/curlicat/"""  # noqa
    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-romanian-corpus/8b6c8dca58ea11ed9c1a00155d026706fb03ef8b4c1847cfbe9cea869a82731e/"  # noqa
    LANGUAGES = ["ro"]  # TODO Romanian; Moldavian; Moldovan (ro)
    TOKENS = 94_925_454
    DOWNLOAD_URLS = ["https://relate.racai.ro/resources/curlicat/ro_curlicat_20221031.zip"]

    # See https://stackoverflow.com/questions/17912307/u-ufeff-in-python-string
    conllup_encoding = "utf-8-sig"

    LICENSE = License(
        name="CC-BY-SA-4.0",
        url="https://elrc-share.eu/static/metashare/licences/CC-BY-SA-4.0.pdf",
        attribution=True,
        sharealike=True,
    )


class CurlicatSKDataset(CurlicatBaseDataset):
    # WARNING: First document is very large! => http://bur.sk/share/2015/RozumnostAZ.pdf (536246 tokens)
    # https://www.juls.savba.sk/curlicat_en.html

    DATASET_ID = "curlicat_sk"
    TITLE = "CURLICAT Corpus [Slovak 3rd version]"
    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-slovak-corpus-v10/b419d7086ef411ed9c1a00155d0267066a930aa487824c08ba48f1183e993aca/"  # noqa
    LANGUAGES = ["sk"]
    DOWNLOAD_URLS = ["https://www.juls.savba.sk/data/curlicat/curlicat-sk-20221025-v1.0.conllup.xz"]
    TOKENS = 67_000_000
    LICENSE = License("unknown")

    def decompress(self):
        # xz --decompress  curlicat-sk-20221025-v1.0.conllup.xz
        pass


class CurlicatSLDataset(CurlicatBaseDataset):
    DATASET_ID = "curlicat_sl"
    TITLE = "CURLICAT Corpus [Slovenian]"
    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-slovenian-corpus/e549f298590711ed9c1a00155d026706db0d61d46f294d9a821307cf9c5df245/"  # noqa
    LANGUAGES = ["sl"]
    TOKENS = 43_481_563
    LICENSE = License(
        name="CC-BY-SA-4.0",
        url="https://elrc-share.eu/static/metashare/licences/CC-BY-SA-4.0.pdf",
        attribution=True,
        sharealike=True,
    )
