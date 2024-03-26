import typing as T
import logging

logger = logging.getLogger(__name__)


def get_texts_from_conllu_file(
    file_handler: T.TextIO,
    title_delimiter: str = ":\n\n",
    sentence_delimiter: str = " ",
    skip_sentence_prefixes: T.Optional[T.Iterable[str]] = None,
):
    """
    Reads CONLLU and CONLLU-Plus format: https://universaldependencies.org/ext-format.html

    https://github.com/EmilStenstrom/conllu/

    """
    import conllu
    from conllu.parser import (
        _FieldParserType,
        _MetadataParserType,
        parse_conllu_plus_fields,
        parse_sentences,
        parse_token_and_metadata,
    )
    from conllu.exceptions import ParseException

    # Custom parse function that handles errors
    def conllu_parse_incr(
        in_file: T.TextIO,
        fields: T.Optional[T.Sequence[str]] = None,
        field_parsers: T.Optional[T.Dict[str, _FieldParserType]] = None,
        metadata_parsers: T.Optional[T.Dict[str, _MetadataParserType]] = None,
        # skip_sentence_prefixes: T.Optional[T.Iterable[str]] = None,
    ) -> conllu.SentenceGenerator:
        if not hasattr(in_file, "read"):
            raise FileNotFoundError("Invalid file, 'parse_incr' needs an opened file as input")

        if not fields:
            fields = parse_conllu_plus_fields(in_file, metadata_parsers=metadata_parsers)

        def generator():
            for sentence in parse_sentences(in_file):
                if skip_sentence_prefixes:
                    skip = False
                    for prefix in skip_sentence_prefixes:
                        if sentence.startswith(prefix):
                            logger.debug(f"Skip sentence: {sentence}")
                            skip = True
                            break

                    if skip:
                        continue

                try:
                    yield parse_token_and_metadata(
                        sentence, fields=fields, field_parsers=field_parsers, metadata_parsers=metadata_parsers
                    )
                except ParseException as e:
                    logger.error(f"Cannot parse sentence: {sentence}; {e}")

        return conllu.SentenceGenerator(generator())

    # from conllu.exceptions import ParseException

    text = None

    # try:
    for sentence in conllu_parse_incr(file_handler):
        if "newdoc id" in sentence.metadata:
            if text is not None:
                # doc completed
                yield text
            text = ""  # init empty document

        # append text to doc
        if "text" in sentence.metadata:
            if not text:
                text = ""  # some conllu are not using doc ids -> force init
            else:
                text += sentence_delimiter  # whitespace betweeen sentences

            text += sentence.metadata["text"]

        if "title" in sentence.metadata:
            text += title_delimiter

    # yield last document
    if text:
        yield text
