import os
import logging
from lm_datasets.datasets.base import Availability, BaseDataset, TOKENS_PER_BYTE
from lm_datasets.utils.wikimedia import parse_and_clean_wikicode


import xml.etree.cElementTree as etree

from smart_open import open

logger = logging.getLogger(__name__)


WIKI_TYPES = [
    "wiki",  # Wikipedia
    "wikibooks",  # Wikibooks
    "wikiquote",  # Wikiquote
    "wikinews",  # Wikinews
    "wikisource",  # Wikisource
    "wikivoyage",  # Wikivoyage
]


# stats_fp = "data/docs_by_language/stats.json"

# stats = json.load(open(stats_fp))

# bytes_per_language_and_source = {
#     lang: {s: b for s, b in source_bytes.items() if s in WIKI_TYPES}
#     for lang, source_bytes in stats["bytes_per_language_and_source"].items()
# }

bytes_per_language_and_source = {
    # EU24 languages
    "bg": {
        "wiki": 1214121628,
        "wikibooks": 10017340,
        "wikiquote": 37156441,
        "wikinews": 4227797,
        "wikisource": 86661269,
    },
    "cs": {
        "wiki": 930310918,
        "wikibooks": 13039351,
        "wikiquote": 8167752,
        "wikinews": 5203801,
        "wikisource": 259915843,
    },
    "da": {"wiki": 224403313, "wikibooks": 21476337, "wikiquote": 1034165, "wikisource": 20123952},
    "de": {
        "wiki": 5197735740,
        "wikibooks": 169962198,
        "wikiquote": 14929927,
        "wikinews": 29553128,
        "wikisource": 531482040,
        "wikivoyage": 100404450,
    },
    "el": {
        "wiki": 1992383542,
        "wikibooks": 65173299,
        "wikiquote": 15485796,
        "wikinews": 14364210,
        "wikisource": 558477104,
        "wikivoyage": 11892155,
    },
    "en": {
        "wiki": 12573466166,
        "wikibooks": 438671226,
        "wikiquote": 425528832,
        "wikinews": 50512180,
        "wikisource": 2491597933,
        "wikivoyage": 163322890,
    },
    "es": {
        "wiki": 3635029311,
        "wikibooks": 83041289,
        "wikiquote": 18004839,
        "wikinews": 23282513,
        "wikisource": 382827125,
        "wikivoyage": 46463569,
    },
    "et": {"wiki": 208343404, "wikibooks": 3589699, "wikiquote": 42857536, "wikisource": 4411631},
    "fi": {
        "wiki": 466350200,
        "wikibooks": 15630626,
        "wikiquote": 8245391,
        "wikinews": 2549156,
        "wikisource": 62883499,
        "wikivoyage": 4194479,
    },
    "fr": {
        "wiki": 4968876644,
        "wikibooks": 82544091,
        "wikiquote": 1613702,
        "wikinews": 25773373,
        "wikisource": 130101343,
        "wikivoyage": 24004716,
    },
    "ga": {"wiki": 21473185, "wikibooks": 0, "wikiquote": 797},
    "hr": {"wiki": 222907670, "wikibooks": 1835214, "wikiquote": 3393450, "wikisource": 68521110},
    "hu": {
        "wiki": 1050228505,
        "wikibooks": 64029424,
        "wikiquote": 8793400,
        "wikinews": 1453968,
        "wikisource": 121765396,
    },
    "it": {
        "wiki": 2798247221,
        "wikibooks": 99975368,
        "wikiquote": 188532632,
        "wikinews": 17511625,
        "wikisource": 225622077,
        "wikivoyage": 44689019,
    },
    "lt": {"wiki": 114077820, "wikibooks": 2025822, "wikiquote": 4709667, "wikisource": 9228644},
    "lv": {"wiki": 98460508, "wikibooks": 112635},
    "mt": {"wiki": 20129053},
    "nl": {
        "wiki": 895801299,
        "wikibooks": 28281089,
        "wikiquote": 255527,
        "wikinews": 8871342,
        "wikisource": 54423027,
        "wikivoyage": 10058074,
    },
    "pl": {
        "wiki": 1229781898,
        "wikibooks": 33877793,
        "wikiquote": 99998973,
        "wikinews": 29401114,
        "wikisource": 63420701,
        "wikivoyage": 30125414,
    },
    "pt": {
        "wiki": 1588856067,
        "wikibooks": 45431838,
        "wikiquote": 24565736,
        "wikinews": 53043049,
        "wikisource": 119019274,
        "wikivoyage": 10344423,
    },
    "ro": {
        "wiki": 518232680,
        "wikibooks": 3900601,
        "wikiquote": 1486942,
        "wikinews": 2536562,
        "wikisource": 166964076,
        "wikivoyage": 1727955,
    },
    "sk": {"wiki": 216967151, "wikibooks": 6584629, "wikiquote": 4781513, "wikisource": 4674463},
    "sl": {"wiki": 264134366, "wikibooks": 5769223, "wikiquote": 2279993, "wikisource": 401940397},
    "sv": {
        "wiki": 443696163,
        "wikibooks": 9014999,
        "wikiquote": 1298260,
        "wikinews": 1840363,
        "wikisource": 32096731,
        "wikivoyage": 4359374,
    },
    # Additional languages
    "uk": {
        "wiki": 1,
        "wikibooks": 1,
        "wikiquote": 1,
        "wikinews": 1,
        "wikisource": 1,
        "wikivoyage": 1,
    },
    "sr": {
        "wiki": 1,
        "wikibooks": 1,
        "wikiquote": 1,
        "wikinews": 1,
        "wikisource": 1,
    },
    "sh": {
        "wiki": 1,
    },
    "nn": {
        "wiki": 1,
        "wikiquote": 1,
    },
    "no": {
        "wiki": 1,
        "wikibooks": 1,
        "wikiquote": 1,
        "wikinews": 1,
        "wikisource": 1,
        "wikivoyage": 1,
    },
    "eu": {
        "wiki": 1,
        "wikibooks": 1,
        "wikiquote": 1,
        "wikinews": 1,
        "wikisource": 1,
    },
    "ca": {
        "wiki": 1,
        "wikibooks": 1,
        "wikiquote": 1,
        "wikinews": 1,
        "wikisource": 1,
        "wikivoyage": 1,
    },
    "gl": {
        "wiki": 1,
        "wikibooks": 1,
        "wikiquote": 1,
        "wikinews": 1,
        "wikisource": 1,
    },
}


class WikimediaBaseDataset(BaseDataset):
    DESCRIPTION = "Wikimedia dumps."
    HOMEPAGE = "https://wikimedia.org"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    # DUMMY = True
    SOURCE_ID = None  # wiki,wikibooks,wikiquote,wikinews,wikisource,wikivoyage

    download_base_url = "https://dumps.wikimedia.org/"
    dump_date = "20230801"

    @property
    def DOWNLOAD_URLS(self):
        dump_url = (
            f"{self.download_base_url}{self.LANGUAGES[0]}{self.SOURCE_ID}/{self.dump_date}/{self.get_raw_file_name()}"
        )

        return [dump_url]

    def get_raw_file_name(self) -> str:
        return f"{self.LANGUAGES[0]}{self.SOURCE_ID}-{self.dump_date}-pages-articles.xml.bz2"

    def get_raw_file_path(self):
        return os.path.join(self.get_local_dataset_dir(), self.get_raw_file_name())

    def is_downloaded(self):
        return os.path.exists(self.get_raw_file_path())

    def get_texts(self):
        import mwparserfromhell

        if not self.is_downloaded():
            self.download()

        # Parse wiki dump
        archive_path = self.get_raw_file_path()

        logger.info(f"Reading from {archive_path}")

        with open(archive_path, "rb") as xml_f:
            # Iterate over XML file
            context = etree.iterparse(xml_f, events=("end",))

            for unused_event, elem in context:
                if not elem.tag.endswith("page"):
                    continue

                namespace = elem.tag[:-4]
                title = elem.find(f"./{namespace}title").text
                ns = elem.find(f"./{namespace}ns").text
                # id_ = elem.find(f"./{namespace}id").text
                red_ = elem.find(f"./{namespace}redirect")

                # Filter pages that are not in the "main" namespace.
                if ns != "0":
                    elem.clear()
                    continue

                raw_content = elem.find(f"./{namespace}revision/{namespace}text").text
                elem.clear()

                if raw_content is None or red_ is not None:
                    self.counter.update({"wiki_redirect": 1})
                    continue

                if "#REDIRECT" in raw_content:
                    self.counter.update({"wiki_redirect": 1})
                    continue

                try:
                    text = parse_and_clean_wikicode(
                        raw_content, parser=mwparserfromhell, language=self.get_language_code()
                    )
                except mwparserfromhell.parser.ParserError as e:
                    logger.error(f"mwparserfromhell ParseError: {e}")

                    self.counter.update({"error_mwparserfromhell": 1})
                    continue

                except Exception as e:
                    logger.error(f"Error: {e}")

                    self.counter.update({"error_other": 1})
                    continue

                plain_text = title + self.title_delimiter + text

                yield plain_text


def get_wikimedia_auto_cls_by_language(lang, source, bytes):
    class WikimediaLanguageDataset(WikimediaBaseDataset):
        SOURCE_ID = source
        TOKENS = int(TOKENS_PER_BYTE * bytes)
        BYTES = bytes
        DATASET_ID = source + "_" + lang
        TITLE = source + "_" + lang
        LANGUAGES = [lang]

    return WikimediaLanguageDataset


def get_wikimedia_auto_classes():
    """
    Auto generate dataset classes with token count
    """

    return [
        get_wikimedia_auto_cls_by_language(lang, s, b)
        for lang, source_bytes in bytes_per_language_and_source.items()
        for s, b in source_bytes.items()
    ]
