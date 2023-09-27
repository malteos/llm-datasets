# flake8: noqa
"""

Available template for converting translation datasets (translation pairs) into plain text.

Template variables:

- SOURCE_LANG
- TARGET_LANG
- SOURCE_TEXT
- TARGET_TEXT

The template use the Jinja2 syntax.

"""


TRANSLATE_THIS_FROM_TO = """Translate this text from {{SOURCE_LANG}} to {{TARGET_LANG}}.

{{SOURCE_TEXT}}

Translation: {{TARGET_TEXT}}"""

TRANSLATE_THIS = """Translate this into {{TARGET_LANG}}: {{SOURCE_TEXT}}

{{TARGET_TEXT}}"""

PREVIOUS_TEXT = """{{SOURCE_TEXT}}

The previous text is in {{SOURCE_LANG}}. Here is a translation to {{TARGET_LANG}}: {{TARGET_TEXT}}"""

SAME_TEXT_IN = """{{SOURCE_TEXT}}

Give me the same text in {{TARGET_LANG}}.

{{TARGET_TEXT}}"""

TEXT_IN_WHATS_IN = """A text in {{SOURCE_LANG}}: {{SOURCE_TEXT}}
What's the text in {{TARGET_LANG}}? {{TARGET_TEXT}}"""

TRANSLATE_THE_FOLLOWING = """Translate the following text from {{SOURCE_LANG}} to {{TARGET_LANG}}.

Text: {{SOURCE_TEXT}}

Translation: {{TARGET_TEXT}}"""

IF_THE_VERSION_SAYS = """If the {{SOURCE_LANG}} version says: {{SOURCE_TEXT}}; then the {{TARGET_LANG}} version should say: {{TARGET_TEXT}}"""

IF_THE_VERSION_SAYS_THUS = """If the {{SOURCE_LANG}} version says: {{SOURCE_TEXT}}; Thus the {{TARGET_LANG}} version should say: {{TARGET_TEXT}}"""

IF_THE_VERSION_SAYS_HENCE = """If the {{SOURCE_LANG}} version says: {{SOURCE_TEXT}}; hence the {{TARGET_LANG}} version should say: {{TARGET_TEXT}}"""


GIVEN_THE_FOLLOWING = """Given the following source text in {{SOURCE_LANG}}: {{SOURCE_TEXT}} , a good {{TARGET_LANG}} translation is: {{TARGET_TEXT}}"""

GIVEN_THE_FOLLOWING_PASSAGE = """Given the following passage: {{SOURCE_TEXT}}

A good {{TARGET_LANG}} translation is: {{TARGET_TEXT}}"""


DOCUMENT_IN = """Document in {{SOURCE_LANG}}: {{SOURCE_TEXT}}

Translate the previous document to proper {{TARGET_LANG}}: {{TARGET_TEXT}}"""

DOCUMENT_IN_ENTIRE = """Document in {{SOURCE_LANG}}: {{SOURCE_TEXT}}

Translate the previous entire document to proper {{TARGET_LANG}} sentence for sentence (min 100 words): {{TARGET_TEXT}}"""

EQUAL_TO = """{{SOURCE_TEXT}} = {{TARGET_LANG}}: {{TARGET_TEXT}}"""

EQUAL_FROM_TO = """{{SOURCE_LANG}}: {{SOURCE_TEXT}} = {{TARGET_LANG}}: {{TARGET_TEXT}}"""

WHAT_IS = """What is the {{TARGET_LANG}} translation of: {{SOURCE_TEXT}}

{{TARGET_TEXT}}"""

WHAT_IS_SENTNCE = """What is the {{TARGET_LANG}} translation of the {{SOURCE_LANG}} sentence: {{SOURCE_TEXT}}

{{TARGET_TEXT}}"""


HOW_DO_YOU_SAY = """How do you say {{SOURCE_TEXT}} in {{TARGET_LANG}}? {{TARGET_TEXT}}"""

TRANSLATES_INTO = """{{SOURCE_TEXT}} translates into {{TARGET_LANG}} as: {{TARGET_TEXT}}"""

TRANSLATES_FROM_INTO = """{{SOURCE_LANG}}: {{SOURCE_TEXT}} translates into {{TARGET_LANG}} as:

{{TARGET_TEXT}}"""


# List of templates in English
def get_templates():
    tpls = [
        TRANSLATE_THIS_FROM_TO,
        TRANSLATE_THIS,
        PREVIOUS_TEXT,
        SAME_TEXT_IN,
        TEXT_IN_WHATS_IN,
        TRANSLATE_THE_FOLLOWING,
        IF_THE_VERSION_SAYS,
        IF_THE_VERSION_SAYS_THUS,
        IF_THE_VERSION_SAYS_HENCE,
        GIVEN_THE_FOLLOWING,
        GIVEN_THE_FOLLOWING_PASSAGE,
        DOCUMENT_IN,
        DOCUMENT_IN_ENTIRE,
        EQUAL_TO,
        EQUAL_FROM_TO,
        WHAT_IS,
        WHAT_IS_SENTNCE,
        HOW_DO_YOU_SAY,
        TRANSLATES_INTO,
        TRANSLATES_FROM_INTO,
    ]

    return tpls
