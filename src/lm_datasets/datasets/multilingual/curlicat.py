import logging
from typing import Iterable
from smart_open import open

from lm_datasets.datasets.base import BaseDataset, Availability
from tqdm.auto import tqdm

from lm_datasets.io.conllu_file import get_texts_from_conllu_file

logger = logging.getLogger(__name__)


class CurlicatBaseDataset(BaseDataset):
    SOURCE_ID = "curlicat"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

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
    TITLE = "CURLICAT Bulgarian Corpus"
    DESCRIPTION = """The Bulgarian CURLICAT corpus consists of texts from different sources, provided with  appropriate licences for distribution. We used three general types of sources with regard to the metadata extraction: Bulgarian National Corpus (provided that they have redistributable licensing terms); some public repositories with open and copyright free data; blogs with redistributable licenses, open content websites, etc. The Bulgarian CURLICAT collection contains 113 087 documents, distributed in seven thematic domains: Culture, Education, European Union, Finance, Politics, Economics, and Science. For more information see the CURLICAT website (http:curlicat-project.eu/deliverables) """  # noqa

    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-bulgarian-corpus/fed6af2a590311ed9c1a00155d0267062ed273d01d2343f1b78d08d4d481679d/"  # noqa

    LANGUAGES = ["bg"]

    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/bg/curlicat_bg/Bulgarian_Curlicat_corpus"]

    TOKENS = 35_319_695

    # done


class CurlicatHRDataset(CurlicatBaseDataset):
    """
    unzip and move to raw dir
    find hr_curlicat_2023-01-13 -name '*.*' -exec mv {} /home/mostendorff/experiments/eulm/data/ele/hr/curlicat_hr;
    """

    DATASET_ID = "curlicat_hr"
    TITLE = "CURLICAT Croatian Corpus"

    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-croatian-corpus/00815518592811ed9c1a00155d026706bc4c59740fce4f7986213e7eef133023/"  # noqa

    LANGUAGES = ["hr"]

    DOWNLOAD_URLS = ["https://zzl.ffzg.unizg.hr/files/5cd9d6151612489fafb2c1f116140519/hr_curlicat_2023-01-13.zip"]

    TOKENS = 49_007_508

    # done


class CurlicatHUDataset(CurlicatBaseDataset):
    DATASET_ID = "curlicat_hu"
    TITLE = "CURLICAT Hungarian Corpus"
    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-hungarian-corpus/8b6c8dcb58ea11ed9c1a00155d02670679a453431c8147079e5a7d9b879a9729/"  # noqa
    LANGUAGES = ["hu"]
    TOKENS = 61_196_946

    def get_curclicat_file_paths(self):
        # Hugarian files are stored in *.txt files subdirectories
        return self.get_dataset_file_paths(subdirectories=True, needed_suffix=".txt")

    # extracted


class CurlicatPLDataset(CurlicatBaseDataset):
    """
    unzip and move to raw dir
    find pl-20221026-annotated -name '*.*' -exec mv {} /home/mostendorff/experiments/eulm/data/ele/pl/curlicat_pl;
    """

    DATASET_ID = "curlicat_pl"
    TITLE = "CURLICAT Polish Corpus"

    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-polish-corpus/f63ae912553911ed9c1a00155d02670648c0a234e0314895b52169af2af57dd7/"  # noqa

    LANGUAGES = ["pl"]
    TOKENS = 59_301_782

    # done


class CurlicatRODataset(CurlicatBaseDataset):
    DATASET_ID = "curlicat_ro"
    TITLE = "CURLICAT Romanian Corpus"
    DESCRIPTION = """The corpus contains 26,477 files, which represent our contribution to the CURLICAT project. It contains texts from 7 domains: science, politics, culture, economy, health, education, nature. Each file has multiple levels of annotation: tokenized, lemmatized, morphologically annotated, dependency parsed, named entities, nominal phrases, IATE terms and automatic domain-specific terms were identified as well. All processing tools are available within the RELATE platform. The corpus was automatically anonymized. Alternate download location: https://relate.racai.ro/resources/curlicat/"""  # noqa

    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-romanian-corpus/8b6c8dca58ea11ed9c1a00155d026706fb03ef8b4c1847cfbe9cea869a82731e/"  # noqa

    LANGUAGES = ["ro"]  # TODO Romanian; Moldavian; Moldovan (ro)

    TOKENS = 94_925_454

    DOWNLOAD_URLS = ["https://relate.racai.ro/resources/curlicat/ro_curlicat_20221031.zip"]

    # See https://stackoverflow.com/questions/17912307/u-ufeff-in-python-string
    conllup_encoding = "utf-8-sig"
    # done


class CurlicatSKDataset(CurlicatBaseDataset):
    # WARNING: First document is very large! => http://bur.sk/share/2015/RozumnostAZ.pdf (536246 tokens)
    # https://www.juls.savba.sk/curlicat_en.html

    DATASET_ID = "curlicat_sk"
    TITLE = "CURLICAT Slovak Corpus (3rd version)"

    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-slovak-corpus-v10/b419d7086ef411ed9c1a00155d0267066a930aa487824c08ba48f1183e993aca/"  # noqa

    LANGUAGES = ["sk"]

    DOWNLOAD_URLS = ["https://www.juls.savba.sk/data/curlicat/curlicat-sk-20221025-v1.0.conllup.xz"]

    TOKENS = 67_000_000

    def decompress(self):
        # xz --decompress  curlicat-sk-20221025-v1.0.conllup.xz
        pass

    # done


class CurlicatSLDataset(CurlicatBaseDataset):
    DATASET_ID = "curlicat_sl"
    TITLE = "CURLICAT Slovenian Corpus"
    HOMEPAGE = "https://elrc-share.eu/repository/browse/curlicat-slovenian-corpus/e549f298590711ed9c1a00155d026706db0d61d46f294d9a821307cf9c5df245/"  # noqa
    LANGUAGES = ["sl"]
    TOKENS = 43_481_563

    # done
