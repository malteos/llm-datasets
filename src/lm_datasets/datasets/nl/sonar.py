import logging
import os

from multiprocessing.pool import Pool

from lm_datasets.datasets.base import BaseDataset, Availability, QualityWarning
from lm_datasets.utils import remove_whitespaces_before_punctuation


logger = logging.getLogger(__name__)

try:
    import folia.main as folia
except ImportError:
    logger.error("Cannot load imports")


class SonarBaseDataset(BaseDataset):
    SOURCE_ID = "sonar"

    TITLE = "SoNaR Corpus NC 1.2"
    HOMEPAGE = "https://taalmaterialen.ivdnt.org/download/tstc-sonar-corpus/"
    DESCRIPTION = (
        "The SoNaR Corpus contains more than 500 million words from texts in standard Dutch ",
        "later than 1954. All texts were tokenized, tagged for part of speech and lemmatized. ",
        "The named entities were also labelled. All annotations were produced automatically, ",
        "no manual verification took place.",
    )

    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD

    LANGUAGES = ["nl"]

    USED_BY = ["GroNLP/bert-base-dutch-cased"]

    TOKENS = 0
    # at least 1,030,000 docs

    QUALITY_WARNINGS = [QualityWarning.BAD_WHITESPACES]

    # LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/nl/sonar/SONAR500/FoLiA/WR-P-E-A_discussion_lists"]
    # subsets = ["WR-P-E-J_wikipedia"]

    # # Excluded
    # subsets = [
    #     # "WR-P-E-A_discussion_lists",  # very short text, too many line breaks and whitespaces
    #     # "WR-P-E-J_wikipedia",  # exclude wikipedia (already part of other dataset)
    # ]

    def get_local_dataset_dir(self):
        return os.path.join(self.raw_datasets_dir, self.get_language_code(), self.SOURCE_ID)

    def decompress(self):
        # tgz
        pass

    def get_text_from_item(self, item) -> str:
        """
        item: file path
        """

        doc = folia.Document(file=item)
        text = doc.text()
        text = remove_whitespaces_before_punctuation(text)

        # logger.info(f"Text extracted {item}")

        return text

    def get_file_paths(self):
        dataset_dir = self.get_local_dataset_dir()

        # Iterate over subsets
        for subset in self.subsets:
            logger.info(subset)

            subset_dir = os.path.join(dataset_dir, "SONAR500", "FoLiA", subset)

            fns = os.listdir(subset_dir)

            # Some subsets have files in subdirectories
            folia_fns = [fn for fn in fns if fn.endswith(".folia.xml")]

            if folia_fns:
                for fn in folia_fns:
                    yield os.path.join(subset_dir, fn)
            else:
                # subdirectories
                for fn in fns:
                    dir_path = os.path.join(subset_dir, fn)
                    if os.path.isdir(dir_path):
                        for fn in os.listdir(dir_path):
                            if fn.endswith(".folia.xml"):
                                yield os.path.join(dir_path, fn)

    def get_texts(self):
        pool = Pool(processes=self.workers)
        iterable = pool.imap_unordered(self.get_text_from_item, self.get_file_paths())

        def terminate():
            pool.terminate()
            pool.close()
            pool.join()

        iterable.terminate = terminate

        return iterable

        # with Pool(processes=self.workers) as pool:
        #     yield from pool.imap_unordered(self.get_text_from_item, self.get_file_paths())

        # for fp in self.get_file_paths():
        #     yield self.get_text_from_item(fp)

    # def get_texts(self):
    #     # SLOW => because gzipped files need to be decompressed for members!
    #     # with tarfile.open("/netscratch/mostendorff/experiments/eulm/data/ele/nl/sonar.tgz") as tf:
    #     #     tfps = tf.getmembers()

    #     dataset_dir = self.get_local_dataset_dir()

    #     # Iterate over subsets
    #     for subset in self.subsets:
    #         logger.info(subset)

    #         subset_dir = os.path.join(dataset_dir, "SONAR500", "FoLiA", subset)

    #         fns = os.listdir(subset_dir)

    #         # Some subsets have files in subdirectories
    #         folia_fns = [fn for fn in fns if fn.endswith(".folia.xml")]

    #         if folia_fns:
    #             yield from self.get_texts_from_folia(folia_fns, subset_dir)
    #         else:
    #             # subdirectories
    #             for fn in fns:
    #                 dir_path = os.path.join(subset_dir, fn)
    #                 if os.path.isdir(dir_path):
    #                     folia_fns = [fn for fn in os.listdir(dir_path) if fn.endswith(".folia.xml")]

    #                     yield from self.get_texts_from_folia(folia_fns, dir_path)


class SonarWebDataset(SonarBaseDataset):
    DATASET_ID = "sonar_web"
    TOKENS = 500_000_000  # TODO all other subsets have zero tokens

    # Web
    subsets = [
        "WR-P-E-K_blogs",
        "WR-P-E-I_web_sites",
        "WR-P-E-H_teletext_pages",
    ]


class SonarSubtitlesDataset(SonarBaseDataset):
    DATASET_ID = "sonar_subtitles"

    # Subtitles
    subsets = [
        "WR-P-E-G_subtitles",
    ]


class SonarGovDataset(SonarBaseDataset):
    DATASET_ID = "sonar_gov"

    # Gov
    subsets = [
        "WR-P-P-I_policy_documents",
        "WR-P-P-F_legal_texts",
        "WR-P-P-J_proceedings",
        "WR-P-P-K_reports",
    ]


class SonarBooksDataset(SonarBaseDataset):
    DATASET_ID = "sonar_books"

    # Books
    subsets = [
        "WR-P-P-B_books",
        "WR-P-P-H_periodicals_magazines",
        "WR-P-E-C_e-magazines",
        "WR-P-P-C_brochures",
        "WS-U-E-A_auto_cues",
    ]


class SonarNewsDataset(SonarBaseDataset):
    DATASET_ID = "sonar_news"

    # News
    subsets = [
        "WR-P-E-E_newsletters",
        "WR-P-P-G_newspapers",
        "WR-P-E-F_press_releases",
        "WR-P-P-D_newsletters",
    ]


class SonarEduDataset(SonarBaseDataset):
    DATASET_ID = "sonar_edu"

    # Education
    subsets = [
        "WS-U-T-B_texts_for_the_visually_impaired",
        "WR-U-E-E_written_assignments",  # extra dataset
        "WR-P-P-E_guides_manuals",
    ]


def get_sonar_classes():
    # sonar_web,sonar_subtitles,sonar_gov,sonar_books,sonar_news,sonar_edu
    return [
        SonarSubtitlesDataset,
        SonarWebDataset,
        SonarGovDataset,
        SonarBooksDataset,
        SonarNewsDataset,
        SonarEduDataset,
    ]
