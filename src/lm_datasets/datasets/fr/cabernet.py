import logging
import tarfile
from lm_datasets.datasets.base import Availability, BaseDataset

logger = logging.getLogger(__name__)


class CaBeRnetDataset(BaseDataset):
    DATASET_ID = "cabernet"
    TITLE = "CaBeRnet: a New French Balanced Reference Corpus"
    LANGUAGES = ["fr"]
    HOMEPAGE = "https://aclanthology.org/2020.cmlc-1.3/"

    AVAILIBITY = Availability.ON_REQUEST
    TOKENS = 711_792_861

    def decompress(self):
        # gzip -d CaBeRnet.tar.gz
        pass

    def get_texts(self):
        # read from tar file
        archive_fp = self.get_dataset_file_paths(needed_suffix=".tar", single_file=True)

        logger.info(f"Extracting from {archive_fp}")

        with tarfile.open(archive_fp) as tar_f:
            member_fns = tar_f.getmembers()
            member_fns = [m for m in tar_f.getmembers() if m.name.endswith(".txt")]

            # Read from txt files
            for member_fn in member_fns:
                logger.info(f"Decompress: {member_fn}")
                decompressed_member = tar_f.extractfile(member_fn).read()
                txt = decompressed_member.decode()

                if txt:
                    pass

                # for doc_text in txt.split("\n\n"):
                #     yield doc_text

                break
                print("x")
