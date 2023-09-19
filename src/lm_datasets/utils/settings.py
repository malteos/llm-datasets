# This file contains the settings for the project.
import logging


LANGUAGES = {
    # 24 languages
    # bg,cs,da,de,el,en,es,et,fi,fr,ga,hr,hu,it,lt,lv,mt,nl,pl,pt,ro,sk,sl,sv
    # Bulgarian, Czech, Danish, German, Greek, English, Spanish, Estonian, Finnish, French, Irish, Croatian, Hungarian, Italian, Lithuanian, Latvian, Maltese, Dutch, Polish, Portuguese, Romanian, Slovak, Slovenian, Swedish  # noqa
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mt": "Maltese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
}


HIGH_LANGUAGES = set("en,de,fr,es".split(","))
# MID_LANGUAGES = set(''.split(','))
# LOW_LANGUAGES = set(''.split(','))

# Max number of dumps to download at once (hard limit for Wikimedia dumps)
MAX_DOWNLOADS = 2

# wiki,wikibooks,wikiquote,wikinews,wikisource,wikivoyage
WIKI_TYPES = [
    "wiki",  # Wikipedia
    "wikibooks",  # Wikibooks
    "wikiquote",  # Wikiquote
    "wikinews",  # Wikinews
    "wikisource",  # Wikisource
    "wikivoyage",  # Wikivoyage
]

MIN_TEXT_LENGTH_BY_WIKI_TYPE = {
    "wiki": 5_000,
    "wikibooks": 1_000,
    "wikiquote": 500,
    "wikinews": 500,
    "wikisource": 500,
    "wikivoyage": 500,
}
WIKI_MIN_TEXT_LENGTH = 5_000

OSCAR_MIN_HARMFUL_PP_BY_LANGUAGE = {
    "en": 10.0,
    "fr": 10.0,
    "es": 10.0,
    "de": 10.0,
}
DEFAULT_OSCAR_MIN_HARMFUL_PP = 20.0

OSCAR_MIN_LANGUAGE_PROB = 0.85

LOGGING_KWARGS = dict(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)

# GPT-2 1.5B is trained with 40GB of Internet text, which is roughly 10 Billion tokens
# BILLION_TOKENS_PER_GIGA_BYTE = 4
BILLION_TOKENS_PER_GIGA_BYTE = 0.2  # (based on German OSCAR dataset)

DEFAULT_VOCAB_SIZE = 159900
DEFAULT_TOKENIZER_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.01
