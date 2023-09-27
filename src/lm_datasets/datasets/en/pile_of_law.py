from lm_datasets.datasets.hf_dataset import HFDataset


class PileOfLawDataset(HFDataset):
    """
    NOTE: This dataset uses only selected subsets of the original "Pile of Law". See `HF_DATASET_CONFIGS`
    """

    DATASET_ID = "pile_of_law"
    SOURCE_ID = "pile_of_law"
    TITLE = "Pile of Law"
    HOMEPAGE = "https://huggingface.co/datasets/pile-of-law/pile-of-law"
    DESCRIPTION = (
        "We curate a large corpus of legal and administrative data. The utility of this data is twofold: "
        "(1) to aggregate legal and administrative data sources that demonstrate different norms and legal "
        "standards for data filtering; (2) to collect a dataset that can be used in the future for pretraining "
        "legal-domain language models, a key direction in access-to-justice initiatives."
    )
    LANGUAGES = ["en"]  # Mainly English, but some other languages may appear in some portions of the data.
    CITATION = """@misc{hendersonkrass2022pileoflaw,
  url = {https://arxiv.org/abs/2207.00220},
  author = {Henderson*, Peter and Krass*, Mark S. and Zheng, Lucia and Guha, Neel and Manning, Christopher D. and Jurafsky, Dan and Ho, Daniel E.},
  title = {Pile of Law: Learning Responsible Data Filtering from the Law and a 256GB Open-Source Legal Dataset},
  publisher = {arXiv},
  year = {2022}
}"""  # noqa
    LICENSE = "CreativeCommons Attribution-NonCommercial-ShareAlike 4.0 International. But individual sources may have other licenses. See paper for details."

    HF_DATASET_ID = "pile-of-law/pile-of-law"
    HF_DATASET_SPLIT = "train"
    HF_DATASET_CONFIGS = [
        "r_legaladvice",
        "courtlistener_docket_entry_documents",
        "atticus_contracts",
        "courtlistener_opinions",
        # "federal_register",
        # "bva_opinions",
        # "us_bills",
        "cc_casebooks",
        "tos",
        # "euro_parl",
        # "nlrb_decisions",
        "scotus_oral_arguments",
        # "cfr",
        # "state_codes",
        # "scotus_filings",
        "exam_outlines",
        # "edgar",
        "cfpb_creditcard_contracts",
        "constitutions",
        "congressional_hearings",
        # "oig",
        # "olc_memos",
        # "uscode",
        # "founding_docs",
        # "ftc_advisory_opinions",
        "echr",
        # "eurlex",
        # "tax_rulings",
        "un_debates",
        # "fre",
        # "frcp",
        # "canadian_decisions",
        # "eoir",
        # "dol_ecab",
        "icj-pcij",
        "uspto_office_actions",
        "ed_policy_guidance",
        "acus_reports",
        # "hhs_alj_opinions",
        # "sec_administrative_proceedings",
        # "fmshrc_bluebooks",
        "resource_contracts",
        "medicaid_policy_guidance",
        # "irs_legal_advice_memos",
        "doj_guidance_documents",
    ]

    streaming = True

    def get_text_from_item(self, item) -> str:
        return item[self.text_column_name]
