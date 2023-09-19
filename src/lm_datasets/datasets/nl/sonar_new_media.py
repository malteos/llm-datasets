import logging


from lm_datasets.datasets.base import BaseDataset, Availability, QualityWarning

import zipfile


logger = logging.getLogger(__name__)


class SonarNewMediaDataset(BaseDataset):
    """ """

    DATASET_ID = "sonar_new_media"
    TITLE = "SoNaR Nieuwe Media Corpus (Version 1.0)"
    HOMEPAGE = "https://taalmaterialen.ivdnt.org/download/tstc-sonar-nieuwe-media-corpus-1/"
    DESCRIPTION = (
        "The SoNaR New Media Corpus 1.0 contains texts from new media (sms, tweets and ",
        "chat messages) that were collected within the STEVIN-project SoNaR.",
    )

    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD

    QUALITY_WARNINGS = [QualityWarning.SHORT_TEXT]

    LANGUAGES = ["nl"]

    TOKENS = 36_000_000

    def get_texts(self):
        from bs4 import BeautifulSoup

        # import folia.main as folia

        # read directly from zip file
        zip_fp = self.get_dataset_file_paths(needed_suffix=".zip", single_file=True)

        with zipfile.ZipFile(zip_fp) as zf:
            needed_suffix = ".folia.xml"
            members_fns = [fn for fn in zf.namelist() if fn.endswith(needed_suffix)]

            for member_fn in members_fns:
                # tweets,chat,sms
                # if "tweets" in member_fn:
                #     continue

                if "chat" in member_fn:
                    # only very short messages
                    continue

                if "sms" in member_fn:
                    # bad quality
                    continue

                logger.info(f"Extracting from {member_fn}")

                with zf.open(member_fn) as member_f:
                    soup = BeautifulSoup(member_f, features="lxml")

                for tweet in soup.find_all("event"):
                    t = tweet.find("t")

                    if t:
                        text = t.get_text()

                        # replace _ with white space
                        text = text.replace("_", " ")

                        yield text

        logger.info("ZIP file extracted")
