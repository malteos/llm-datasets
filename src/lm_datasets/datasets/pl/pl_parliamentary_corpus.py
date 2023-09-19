import logging
import tarfile
from tqdm.auto import tqdm

import xml.etree.cElementTree as ET

from lm_datasets.datasets.base import BaseDataset, Genre


logger = logging.getLogger(__name__)


class PLParliamentaryCorpusDataset(BaseDataset):
    """
    DOWNLOAD
    -----------

    Instruction

    - Run the command on the server:

    wget http://manage.legis.nlp.ipipan.waw.pl/download/ppc-nanno.tar.gz

    - Extract files:

    tar -xvf ppc-nanno.tar.gz

    - Run python script (locally with small sample)

    $ python pl_parliamentary_corpus.py \
        /Users/melinaplakidis/Documents/DFKI/eulm/lm_datasets/pl/ppc-sample \
        pl_parl_processed.jsonl

    """

    DATASET_ID = "pl_parliamentary_corpus"
    TITLE = "Polish Parliamentary Corpus / Korpus Dyskursu Parlamentarnego"
    DESCRIPTION = (
        "The Polish Parliamentary Corpus (PPC) is a Polish corpus made up of documents from the proceedings of the"
        " Polish Parliament, Sejm, and Senate. The corpus includes data of the Polish Sejm corpus and consists of"
        " stenographic records of plenary sittings and committee sittings, segments of interpellations and questions."
        " Texts in the PPC corpus cover the period of a hundred years from 1919 to 2019."
    )
    HOMEPAGE = "http://clip.ipipan.waw.pl/PPC"

    LANGUAGES = ["pl"]
    LICENSE = "CC-BY"

    GENRES = [Genre.GOVERNMENT]

    DOWNLOAD_URLS = ["http://manage.legis.nlp.ipipan.waw.pl/download/ppc-nanno.tar.gz"]

    TOKENS = 671_292_351

    def decompress(self):
        # gzip -d ppc-nanno.tar.gz
        pass

    def get_texts(self):
        # read from tar file
        archive_fp = self.get_dataset_file_paths(needed_suffix=".tar", single_file=True)

        logger.info(f"Extracting from {archive_fp}")

        with tarfile.open(archive_fp) as tar_f:
            member_fns = tar_f.getmembers()
            member_fns = [m for m in member_fns if m.name.endswith("text_structure.xml")]

            for member_fn in tqdm(member_fns):
                member_content = tar_f.extractfile(member_fn).read()
                xml_str = member_content.decode()

                # extract plain text
                tree = ET.ElementTree(ET.fromstring(xml_str))
                root = tree.getroot()
                utterances = [elem.text for elem in root.iter() if elem.tag.endswith("{http://www.tei-c.org/ns/1.0}u")]
                text = " ".join(utterances)

                # print(text)
                yield text
