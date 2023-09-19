"""

https://repository.pfri.uniri.hr/islandora/object/pfri%3A2156/datastream/DATASET0/download

"""

import logging


from lm_datasets.datasets.base import BaseDataset, Availability


from smart_open import open


logger = logging.getLogger(__name__)


class CroatianNewsENGRIDataset(BaseDataset):
    """


    https://www.clarin.si/repository/xmlui/handle/11356/1416

    """

    DATASET_ID = "croatian_news_engri"
    TITLE = "Corpus of Croatian news portals ENGRI (2014-2018)"
    HOMEPAGE = "https://repository.pfri.uniri.hr/islandora/object/pfri%3A2156"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    DESCRIPTION = (
        "The corpus consists of texts collected from the most popular (based on the Reuters Institute Digital News ",
        "Report for 2018, retrieved from http://www.digitalnewsreport.org in April, 2019) news portals in Croatia ",
        "in the period from 2014 to 2018: Direktno, Dnevno, Net Hr, Hrt, Index_Hr, Jutarnji, Novilist, Rtl, ",
        "SlobodnaDalmacija, Večernji, Tportal, Dnevnik.",
    )

    LANGUAGES = ["hr"]

    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/hr/croatian_news_engri"]

    TOKENS = 694799268

    @property
    def DOWNLOAD_URLS(self):
        fns = "engri.24sata.hr.conllu.gz,/engri.direktno.hr.conllu.gz,/engri.dnevno.hr.conllu.gz,/engri.hrt.hr.conllu.gz,/engri.index.hr.conllu.gz,/engri.jutarnji.hr.conllu.gz,/engri.net.hr.conllu.gz,/engri.novilist.hr.conllu.gz,/engri.rtl.hr.conllu.gz,/engri.slobodnadalmacija.hr.conllu.gz,/engri.telegram.hr.conllu.gz,/engri.vecernji.hr.conllu.gz".split(  # noqa
            ",/"
        )

        return ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1416" + fn for fn in fns]

    def get_texts(self):
        import conllu

        # Parse CONLLU files and extract document-level texts

        for fp in self.get_dataset_file_paths(needed_suffix=".conllu.gz"):
            logger.info(f"Reading from {fp}")

            with open(fp) as f:
                text = None

                for sentence in conllu.parse_incr(f):
                    if "newdoc id" in sentence.metadata:
                        if text is not None:
                            # doc completed
                            yield text
                        text = ""  # init

                    # append text to doc
                    text += sentence.metadata["text"]

                    if "title" in sentence.metadata:
                        text += ":\n"

                # yield last document
                if text is not None:
                    yield text
