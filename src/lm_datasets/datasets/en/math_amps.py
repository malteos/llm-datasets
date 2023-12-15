import json
import logging
import tarfile

from lm_datasets.datasets.base import BaseDataset, License

logger = logging.getLogger(__name__)


class MathAMPSDataset(BaseDataset):
    DATASET_ID = "math_amps"
    TITLE = "Auxiliary Mathematics Problems and Solutions (AMPS) dataset"
    HOMEPAGE = "https://github.com/hendrycks/math"
    DESCRIPTION = (
        "Our pretraining dataset, the Auxiliary Mathematics Problems and "
        "Solutions (AMPS) dataset, has problems and step-by-step solutions typeset "
        " in LATEX. AMPS contains over 100,000 problems pulled from Khan Academy and "
        " approximately 5 million problems generated from manually designed Mathematica scripts. "
        "Problems include various aspects of algebra, calculus, counting and statistics, geometry, "
        "linear algebra, and number theory."
    )
    LANGUAGES = ["en"]
    CITATION = """@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}"""  # noqa
    LICENSE = License("repository license is MIT; no specific data license")
    DOWNLOAD_URLS = ["https://drive.google.com/file/d/1hQsua3TkpEmcJD_UWQx8dmNdEZPyxw23/view?usp=sharing"]

    def download(self):
        """
        Manually download the archive from Google drive.
        """
        pass

    def decompress(self):
        """
        gzip -d ...
        """
        pass

    def get_texts(self):
        # read from tar files
        tar_file_paths = self.get_dataset_file_paths(needed_suffix=".tar")
        for tar_fp in tar_file_paths:
            logger.info(f"Extracting from {tar_fp}")

            with tarfile.open(tar_fp) as tar_f:
                while True:
                    member = tar_f.next()

                    if member is None:
                        break

                    if "README" in member.name:
                        continue

                    if member.name.endswith("json") or member.name.endswith(".txt"):
                        with tar_f.extractfile(member) as member_f:
                            if member.name.endswith(".json"):
                                doc = json.load(member_f)

                                # "problem"
                                # "hints" -> list
                                if "problem" in doc and "hints" in doc and len(doc) >= 2:
                                    text = doc["title"] + self.title_delimiter if "title" in doc else ""

                                    text += (
                                        doc["problem"]
                                        + self.paragraph_delimiter
                                        + (self.paragraph_delimiter.join(doc["hints"]))
                                    )
                                else:
                                    raise ValueError(f"Invalid JSON schema: {doc}")
                            else:
                                text = member_f.read().decode("utf-8")  # read directly text from *.txt file

                            yield text
