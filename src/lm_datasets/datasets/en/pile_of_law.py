from lm_datasets.datasets.base import License, Availability, Genre
from lm_datasets.datasets.hf_dataset import HFDataset

# OK licenses
# r_legaladvice,courtlistener_docket_entry_documents,atticus_contracts,courtlistener_opinions,tos,scotus_oral_arguments,exam_outlines,cfpb_creditcard_contracts,constitutions,congressional_hearings,un_debates
# ['pile_of_law_r_legaladvice', 'pile_of_law_courtlistener_docket_entry_documents', 'pile_of_law_atticus_contracts', 'pile_of_law_courtlistener_opinions', 'pile_of_law_tos', 'pile_of_law_scotus_oral_arguments', 'pile_of_law_exam_outlines', 'pile_of_law_cfpb_creditcard_contracts', 'pile_of_law_constitutions', 'pile_of_law_congressional_hearings', 'pile_of_law_un_debates']
# pile_of_law_r_legaladvice,pile_of_law_courtlistener_docket_entry_documents,pile_of_law_atticus_contracts,pile_of_law_courtlistener_opinions,pile_of_law_tos,pile_of_law_scotus_oral_arguments,pile_of_law_exam_outlines,pile_of_law_cfpb_creditcard_contracts,pile_of_law_constitutions,pile_of_law_congressional_hearings,pile_of_law_un_debates

PILE_OF_LAW_SUBSETS_WITH_LICENSE = {
    "r_legaladvice": License("Creative Commons Attribution 4.0 International"),
    "courtlistener_docket_entry_documents": License("Underlying content is Public Domain."),
    "atticus_contracts": License("CC BY 4.0"),
    "courtlistener_opinions": License("Public domain"),
    # "federal_register",
    # "bva_opinions",
    # "us_bills",
    "cc_casebooks": License(
        "Mixed; Most restrictive: CC BY-NC-SA 4.0", commercial_use=True
    ),  # All CC, varying on exact restrictiveness. Most restrictive: CC BY-NC-SA 4.0. All licensing information preserved in individual documents.
    "tos": License(
        "Publicly available, unknown license. Assumed to be governed by fair use standards."
    ),  # Publicly available, unknown license. Assumed to be governed by fair use standards.
    # "euro_parl",
    # "nlrb_decisions",
    "scotus_oral_arguments": License("Public domain"),  # Public domain
    # "cfr",
    # "state_codes",
    # "scotus_filings",
    "exam_outlines": License(
        "Publicly available, unknown license. Assumed to be governed by fair use standards."
    ),  # Publicly available, unknown license. Assumed to be governed by fair use standards.
    # "edgar",
    "cfpb_creditcard_contracts": License(
        "Publicly available, unknown license. Assumed to be governed by fair use standards."
    ),  # # Publicly available, unknown license. Assumed to be governed by fair use standards.
    "constitutions": License("CC BY-NC 3.0", commercial_use=False, attribution=True),  # CC BY-NC 3.09
    "congressional_hearings": License("Public domain"),  # Public domain
    # "oig",
    # "olc_memos",
    # "uscode",
    # "founding_docs",
    # "ftc_advisory_opinions",
    "echr": License(
        "Non-commercial, commercial use requires written permission",
        commercial_use=False,
        url="https://www.echr.coe.int/en/copyright-and-disclaimer",
    ),  # Non-commercial, commercial use requires written permission
    # "eurlex",
    # "tax_rulings",
    "un_debates": License("Public domain"),  # Public domain
    # "fre",
    # "frcp",
    # "canadian_decisions",
    # "eoir",
    # "dol_ecab",
    "icj-pcij": None,  # unkown
    "uspto_office_actions": None,
    "ed_policy_guidance": None,
    "acus_reports": None,
    # "hhs_alj_opinions",
    # "sec_administrative_proceedings",
    # "fmshrc_bluebooks",
    "resource_contracts": None,
    "medicaid_policy_guidance": None,
    # "irs_legal_advice_memos",
    "doj_guidance_documents": None,
}


class PileOfLawDataset(HFDataset):
    """
    NOTE: This dataset uses only selected subsets of the original "Pile of Law". See `HF_DATASET_CONFIGS`
    """

    DATASET_ID = "pile_of_law"
    SOURCE_ID = "pile_of_law"
    TITLE = "Pile of Law (selected subsets)"
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
    GENRES = [Genre.LEGAL]
    LICENSE = License(
        "CreativeCommons Attribution-NonCommercial-ShareAlike 4.0 International. But individual sources may have other licenses. See paper for details.",
        commercial_use=False,
        sharealike=True,
        attribution=True,
    )
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD

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


def get_class(subset, license_info):
    class Dataset(PileOfLawDataset):
        DATASET_ID = f"pile_of_law_{subset}"
        TITLE = f"Pile of Law (subset: {subset})"
        LICENSE = license_info
        HF_DATASET_CONFIGS = [subset]
        HAS_OVERLAP_WITH = ["pile_of_law"]

    return Dataset


def get_pile_of_law_auto_classes():
    return [PileOfLawDataset] + [get_class(s, l) for s, l in PILE_OF_LAW_SUBSETS_WITH_LICENSE.items()]
