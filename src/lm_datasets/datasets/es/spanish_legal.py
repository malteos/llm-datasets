import logging
import zipfile
from lm_datasets.datasets.base import Availability, BaseDataset, Genre, GB, QualityWarning

logger = logging.getLogger(__name__)


class SpanishLegalDataset(BaseDataset):
    """
    Warning: Document splitting only done by "\n\n" => short texts and bad line breaks.
    """

    DATASET_ID = "spanish_legal"
    TITLE = "Spanish Legal Domain Corpora"
    HOMEPAGE = "https://github.com/PlanTL-GOB-ES/lm-legal-es"
    CITATION = """@misc{gutierrezfandino2021legal,
            title={Spanish Legalese Language Model and Corpora},
            author={Asier Gutiérrez-Fandiño and Jordi Armengol-Estapé and Aitor Gonzalez-Agirre and Marta Villegas},
            year={2021},
            eprint={2110.12201},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }
        """
    LANGUAGES = ["es"]
    GENRES = [Genre.LEGAL]
    AVAILIBITY = Availability.DIRECT_DOWNLOAD
    QUALITY_WARNINGS = [QualityWarning.BAD_LINEBREAKS, QualityWarning.SHORT_TEXT]
    BYTES = 8.9 * GB

    USED_BY = ["https://huggingface.co/PlanTL-GOB-ES/RoBERTalex"]

    DOWNLOAD_URLS = [("https://zenodo.org/record/5495529/files/corpus.zip?download=1", "corpus.zip")]

    def is_downloaded(self):
        return bool(self.get_dataset_file_paths(needed_suffix=".zip", single_file=True))

    def get_texts(self):
        if not self.is_downloaded():
            self.download()

        zip_fp = self.get_dataset_file_paths(needed_suffix=".zip", single_file=True)

        with zipfile.ZipFile(zip_fp) as zf:
            member_fns = zf.namelist()
            for fn in member_fns:
                # Exclude: EURLEX (part of other dataset)
                # TODO "opus", "europarl"
                if fn.endswith(".txt") and "eurlex" not in fn:
                    with zf.open(fn) as member_f:
                        txt = member_f.read().decode()

                        for doc_text in txt.split("\n\n"):  # TODO
                            doc_text = doc_text.strip()

                            yield doc_text
