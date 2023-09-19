from lm_datasets.datasets.base import BaseDataset, Availability


# @DeprecationWarning("Folketinget is already part of Danish GigaWord")
class DanishParliamentCorpusDataset(BaseDataset):
    DATASET_ID = "danish_parliament_corpus"
    TITLE = "The Danish Parliament Corpus 2009 - 2017, v1"
    HOMEPAGE = " http://hdl.handle.net/20.500.12115/8"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LANGUAGES = ["da"]

    DESCRIPTION = (
        "The Danish Parliament Corpus 2009 - 2017 contains Hansards (transcripts of parliamentary speeches) from the "
        "sittings in the Chamber of the Danish Parliament, Folketinget. The corpus consists of xml files, one for "
        "each parliamentary year, running from October to the following June. The files are marked for meetings, "
        "item title and number, speeches, name and party of speakers, date, time etc. The Danish Parliament Corpus "
        "2009-2017 follows the license for Open Data stating the following: `The Danish Parliament grants a "
        "world-wide, free, non-exclusive and otherwise unrestricted right of use of the data in the Danish "
        "Parliament's open data catalogue. The data can be freely: § copied, distributed and published, § "
        "adapted and combined with other material, § exploited commercially and non-commercially. ` Following "
        "the copyright act, the speeches can be distributed without the consent of the speaker but only in a way "
        "where the author/speaker of each text/speech is clearly stated. Furthermore, the Danish Parliament must "
        "be acknowledged as the source. Version 1 of the corpus includes meetings until May 4th, 2017, and the "
        "reports for the latest parliamentary year have not been published as the final edition. The reports of "
        "all other meetings are the final editions. "
        "Project Leader: Costanza Navarretta, UCPH"
    )
    PII = "I have not checked the data source for personally identifiable or sensitive information."
    LICENSE = "public domain"

    # Size:  40,841,226 words
    TOKENS = 40_841_226

    DOWNLOAD_URLS = []
    LOCAL_DIRS = ["pegasus:/netscratch/ortiz/corpora/ELE/danish_parliament_corpus"]
