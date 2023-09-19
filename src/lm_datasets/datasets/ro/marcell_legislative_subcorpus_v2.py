import logging
import zipfile
from tqdm.auto import tqdm
from lm_datasets.datasets.base import BaseDataset, MILLION, Genre

logger = logging.getLogger(__name__)


class MarcellLegislativeSubcorpusV2Dataset(BaseDataset):
    DATASET_ID = "marcell_legislative_subcorpus_v2"
    TITLE = "MARCELL Romanian legislative subcorpus v2"
    HOMEPAGE = "https://elrc-share.eu/repository/browse/marcell-romanian-legislative-subcorpus-v2/2da548428b9d11eb9c1a00155d026706ce94a6b59ffc4b0e9fb5cd9cebe6889e/"  # noqa
    AVAILIBILITY = "Yes - it has a direct download link or links"

    LANGUAGES = ["ro"]
    GENRES = [Genre.LEGAL]

    DESCRIPTION = (
        "The Romanian corpus contains 163,274 files, which represent the body of national legislation ranging from 1881"
        " to 2021. This corpus includes mainly: governmental decisions, ministerial orders, decisions, decrees and"
        " laws. All the texts were obtained via crawling from the public Romanian legislative portal. This corpus"
        " resulted from the MARCELL project. Alternate download location: https://relate.racai.ro/marcell/new/"
    )
    PII = "No"
    LICENSE = "public domain"

    TOKENS = 31 * MILLION

    def get_texts(self):
        archive_fp = self.get_dataset_file_paths(single_file=True, needed_suffix=".zip")
        logger.info(f"Reading from archive: {archive_fp}")

        with zipfile.ZipFile(archive_fp, "r") as zf:
            # zf.printdir()
            member_fns = [fn for fn in zf.namelist() if fn in {"ro-raw.zip"}]

            for member_fn in member_fns:
                logger.info(f"Reading from archive member: {member_fn}")

                with zf.open(member_fn) as member_f:  #
                    with zipfile.ZipFile(member_f) as zf_2:
                        needed_suffix_2 = ".txt"
                        members_fns_2 = zf_2.namelist()
                        members_fns_2 = [fn for fn in members_fns_2 if fn.endswith(needed_suffix_2)]

                        for member_fn_2 in tqdm(members_fns_2, desc="Extracting txt files"):
                            # logger.info(f"Reading archive member from {member_fn_2}")
                            with zf_2.open(member_fn_2) as member_f_2:
                                text = member_f_2.read().decode()
                                yield text
