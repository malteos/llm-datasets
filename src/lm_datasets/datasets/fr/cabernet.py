import logging
import tarfile
from lm_datasets.datasets.base import Availability, BaseDataset, QualityWarning, License

logger = logging.getLogger(__name__)


class CaBeRnetDataset(BaseDataset):
    """
    Contents:
    - Fiction
    - Acad (Wikiepdia)
    - Oral
    - Pop (Ads)
    - News
    """

    DATASET_ID = "cabernet"
    TITLE = "CaBeRnet: a New French Balanced Reference Corpus"
    LANGUAGES = ["fr"]
    HOMEPAGE = "https://aclanthology.org/2020.cmlc-1.3/"
    CITATION = """@inproceedings{popa-fabre-etal-2020-french,
    title = "{F}rench Contextualized Word-Embeddings with a sip of {C}a{B}e{R}net: a New {F}rench Balanced Reference Corpus",
    author = "Popa-Fabre, Murielle  and
      Ortiz Su{\'a}rez, Pedro Javier  and
      Sagot, Beno{\^\i}t  and
      de la Clergerie, {\'E}ric",
    booktitle = "Proceedings of the 8th Workshop on Challenges in the Management of Large Corpora",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Ressources Association",
    url = "https://aclanthology.org/2020.cmlc-1.3",
    pages = "15--23",
    abstract = "This paper investigates the impact of different types and size of training corpora on language models. By asking the fundamental question of quality versus quantity, we compare four French corpora by pre-training four different ELMos and evaluating them on dependency parsing, POS-tagging and Named Entities Recognition downstream tasks. We present and asses the relevance of a new balanced French corpus, CaBeRnet, that features a representative range of language usage, including a balanced variety of genres (oral transcriptions, newspapers, popular magazines, technical reports, fiction, academic texts), in oral and written styles. We hypothesize that a linguistically representative corpus will allow the language models to be more efficient, and therefore yield better evaluation scores on different evaluation sets and tasks. This paper offers three main contributions: (1) two newly built corpora: (a) CaBeRnet, a French Balanced Reference Corpus and (b) CBT-fr a domain-specific corpus having both oral and written style in youth literature, (2) five versions of ELMo pre-trained on differently built corpora, and (3) a whole array of computational results on downstream tasks that deepen our understanding of the effects of corpus balance and register in NLP evaluation.",
    language = "English",
    ISBN = "979-10-95546-61-0",
}
"""  # noqa
    LICENSE = License(
        "Creative Commons License", url="https://aclanthology.org/2020.cmlc-1.3/ (Section 1)", research_use=True
    )
    AVAILIBITY = Availability.ON_REQUEST
    TOKENS = 711_792_861
    QUALITY_WARNINGS = [QualityWarning.BAD_DOCUMENT_SPLITS]

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
                if "_Acad.txt" in member_fn.name:
                    # Skip "Acad" because it is Wikipedia
                    continue

                if "_Pop.txt" in member_fn.name:
                    # Skip "Pop" because it is ads
                    continue

                logger.info(f"Decompress: {member_fn}")
                decompressed_member = tar_f.extractfile(member_fn).read()
                txt_file_content = decompressed_member.decode()

                if txt_file_content:
                    # File content does not contain proper document separators.
                    # We use simply use triple line breaks (not perfect but should do the job).
                    for doc_text in txt_file_content.split("\n\n\n"):
                        yield doc_text
