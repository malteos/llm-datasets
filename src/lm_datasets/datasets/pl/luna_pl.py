import logging
import tarfile
from typing import Iterable
from lm_datasets.datasets.base import BaseDataset


logger = logging.getLogger(__name__)


class LunaPL(BaseDataset):
    """
    Original text cannot be reconstructed from TEI XML.
    """

    DATASET_ID = "luna_pl"
    TITLE = "LUNA.PL corpus"
    DESCRIPTION = "Human-human dialogues in TEI P5 format annotated for the 'Spoken Language UNderstanding "
    +"in multilingual communication systems' project"
    DOWNLOAD_URLS = [
        ("http://zil.ipipan.waw.pl/LUNA?action=AttachFile&do=get&target=LUNA_TEI.tar.gz", "LUNA_TEI.tar.gz")
    ]
    LANGUAGES = ["pl"]

    DUMMY = True

    # BYTES = 932 * MB

    def decompress(self):
        # gzip -d LUNA_TEI.tar.gz
        pass

    def get_texts(self) -> Iterable[str]:
        raise ValueError("bad dataset")

        # tar + TEI
        # read from tar file
        archive_fp = self.get_dataset_file_paths(needed_suffix=".tar", single_file=True)

        logger.info(f"Extracting from {archive_fp}")

        with tarfile.open(archive_fp) as tar_f:
            member_fns = tar_f.getmembers()
            member_fns = [m for m in tar_f.getmembers() if m.name.endswith(".xml")]

            for member_fn in member_fns:
                member_content = tar_f.extractfile(member_fn).read()
                tei_xml = member_content.decode()
                # TODO extract plain text

                if tei_xml:
                    pass
                break
