import logging
import tarfile

import xml.etree.cElementTree as ET

from tqdm.auto import tqdm
from lm_datasets.datasets.base import MILLION, BaseDataset, QualityWarning

logger = logging.getLogger(__name__)


class NKJPPodkorpusMilionowyDataset(BaseDataset):
    """
    DOWNLOAD
    -----------

    Instruction

    - Downloaded locally by clicking on the download link in the browser:

    http://clip.ipipan.waw.pl/NationalCorpusOfPolish?action=AttachFile&do=get&target=NKJP-PodkorpusMilionowy-1.2.tar.gz

    - Copy local file to server:

    $ scp /Local/Path/to/NKJP-PodkorpusMilionowy-1.2.tar.gz \
        username@clustername:/data/datasets/ele/pl/pl_nkjp_1-2/NKJP-PodkorpusMilionowy-1.2.tar.gz

    - Extract files:

    tar -xvf NKJP-PodkorpusMilionowy-1.2.tar.gz

    - Run python script locally with small sample:

    $ python pl_nkjp_1-2.py \
        /Users/melinaplakidis/Downloads/NKJP-PodkorpusMilionowy-1.2 pl_nkjp_processed.jsonl \
        --limit 100

    """

    DATASET_ID = "pl_nkjp"
    TITLE = "NKJP-PodkorpusMilionowy-1.2 (National Corpus of Polish)"
    HOMEPAGE = "http://clip.ipipan.waw.pl/NationalCorpusOfPolish"

    LANGUAGES = ["pl"]
    DOWNLOAD_URLS = [
        (
            "http://clip.ipipan.waw.pl/NationalCorpusOfPolish?action=AttachFile&do=get&target=NKJP-PodkorpusMilionowy-1.2.tar.gz",  # noqa
            "NKJP-PodkorpusMilionowy-1.2.tar.gz",
        )
    ]
    TOKENS = 1 * MILLION
    LICENSE = "CC-BY"
    QUALITY_WARNINGS = [QualityWarning.BAD_LINEBREAKS]

    def decompress(self):
        # gzip -d NKJP-PodkorpusMilionowy-1.2.tar.gz
        pass

    def get_texts(self):
        # read from tar file
        archive_fp = self.get_dataset_file_paths(needed_suffix=".tar", single_file=True)

        logger.info(f"Extracting from {archive_fp}")

        with tarfile.open(archive_fp) as tar_f:
            member_fns = tar_f.getmembers()
            member_fns = [m for m in member_fns if m.name.endswith("text.xml")]

            for member_fn in tqdm(member_fns):
                member_content = tar_f.extractfile(member_fn).read()
                xml_str = member_content.decode()

                # extract plain text
                tree = ET.ElementTree(ET.fromstring(xml_str))

                root = tree.getroot()
                # Print out some information ("name" and "desc" tags)
                utterances = [elem.text for elem in root.iter() if elem.tag.endswith("{http://www.tei-c.org/ns/1.0}ab")]
                text = " ".join(utterances)

                yield text
