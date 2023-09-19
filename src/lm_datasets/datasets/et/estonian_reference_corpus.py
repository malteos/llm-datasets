from typing import Iterable
import zipfile
from lm_datasets.datasets.base import MILLION, BaseDataset, Availability, QualityWarning
from lm_datasets.utils import remove_whitespaces_before_punctuation

import logging

logger = logging.getLogger(__name__)


class EstonianReferenceCorpusDataset(BaseDataset):
    DATASET_ID = "estonian_reference_corpus"
    TITLE = "Estonian Reference Corpus"
    HOMEPAGE = "https://www.cl.ut.ee/korpused/segakorpus/"
    DESCRIPTION = (
        "This corpus includes Estonian texts (fiction, PhD theses, newspapers, magazines, parliamentary ",
        "transcriptions, computer-mediated communication) published between 1990 and 2007. ",
        "The corpus is encoded in TEI. ",
        "The corpus is available for online browsing through a dedicated concordancer and is available ",
        "for download from CELR.",
    )

    LANGUAGES = ["et"]

    DOWNLOAD_URLS = [
        "https://www.cl.ut.ee/korpused/segakorpus/eesti_ilukirjandus_1990/failid/Ilukirjandus.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/postimees/failid/xml/Postimees.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/ekspress/failid/Ekspress.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/epl/failid/xml/Paevaleht.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/maaleht/failid/Maaleht.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/slohtuleht/failid/xml/SLOleht.tar.gz",
        "https://www.cl.ut.ee/korpused/segakorpus/valgamaalane/failid/Valgamaalane.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/laane_elu/failid/LaaneElu.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/horisont/failid/Horisont.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/luup/failid/Luup.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/kroonika/failid/Kroonika.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/eestiarst/failid/EestiArst.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/arvutitehnika/failid/Arvutitehnika.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/agraarteadus/failid/Agraarteadus.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/teadusartiklid/failid/Teadusartiklid.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/seadused/failid/xml/Seadused.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/riigikogu/failid/Riigikogu.zip",
        "https://www.cl.ut.ee/korpused/segakorpus/doktoritood/failid/Doktoritood.zip",
    ]
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

    LICENSE = "free for non-commercial use"

    QUALITY_WARNINGS = [QualityWarning.BAD_WHITESPACES]

    TOKENS = 175 * MILLION

    def is_downloaded(self):
        return len(self.get_dataset_file_paths(needed_suffix=(".tar.gz", ".zip"))) == len(self.DOWNLOAD_URLS)

    def get_texts(self) -> Iterable[str]:
        """
        Adapted from https://github.com/estnltk/estnltk/blob/123fadf204fb99661da5320cf95172aa9b61c697/tutorials/corpus_processing/importing_text_objects_from_corpora.ipynb  # noqa
        """
        from estnltk.corpus_processing.parse_koondkorpus import get_div_target
        from estnltk.corpus_processing.parse_koondkorpus import parse_tei_corpus_file_content

        if not self.is_downloaded():
            self.download()

        # iterate over archives
        for archive_fp in self.get_dataset_file_paths(needed_suffix=(".tar.gz", ".zip")):
            if archive_fp.endswith(".zip"):
                with zipfile.ZipFile(archive_fp, "r") as zf:
                    # Folders 'bin' contain headers and corpus descriptions.
                    # The '.xml' files outside the 'bin' folders are the files with the actual textual content.
                    for zfn in zf.namelist():
                        if "/bin/" in zfn:
                            continue

                        if ".xml" not in zfn:
                            continue

                        with zf.open(zfn) as xml_file:
                            xml_str = xml_file.read()

                        # find out which subsection of the XML file forms a single document
                        fake_path = archive_fp + "/" + zfn
                        target = get_div_target(fake_path)

                        # import documents as Text objects
                        for text_obj in parse_tei_corpus_file_content(
                            xml_str.decode(),
                            file_path=fake_path,
                            target=[target],
                            sentence_separator=self.sentence_delimiter,
                            paragraph_separator=self.paragraph_delimiter,
                        ):
                            yield remove_whitespaces_before_punctuation(text_obj.text)

            elif archive_fp.endswith(".tar.gz"):
                # raise NotImplementedError
                # TODO
                logger.warning(f"TAR.GZ not implemented: {archive_fp}")
