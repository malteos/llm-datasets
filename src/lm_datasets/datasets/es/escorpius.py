from lm_datasets.datasets.hf_dataset import HFDataset


class ESCorpiusDataset(HFDataset):
    """
    OVERLAP with CC/OSCAR
    """

    DATASET_ID = "escorpius"
    TITLE = "esCorpius: A Massive Spanish Crawling Corpus "

    HOMEPAGE = "https://huggingface.co/datasets/LHF/escorpius"
    CITATION = """@inproceedings{gutierrezfandino22_iberspeech,
        author={Asier Gutiérrez-Fandiño and David Pérez-Fernández and Jordi Armengol-Estapé and David Griol and Zoraida Callejas},
        title={{esCorpius: A Massive Spanish Crawling Corpus}},
        year=2022,
        booktitle={Proc. IberSPEECH 2022},
        pages={126--130},
        doi={10.21437/IberSPEECH.2022-26}
        }"""  # noqa

    LANGUAGES = ["es"]
    WEB_CRAWLED = True

    HF_DATASET_ID = "LHF/escorpius"
    HF_DATASET_SPLIT = "train"

    text_column = "text"

    def get_texts(self):
        raise ValueError("Do not use -> overlap with OSCAR")
