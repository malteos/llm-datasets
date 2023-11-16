import os
from pathlib import Path
from typing import List, Optional
from lm_datasets.datasets.base import Availability, License
from lm_datasets.datasets.hf_dataset import HFDataset

import logging

from lm_datasets.datasets.parquet_dataset import ParquetDataset


logger = logging.getLogger(__name__)

# Dataset IDs:
# starcoder_emacs-lisp,starcoder_literate-haskell,starcoder_shell,starcoder_ada,starcoder_erlang,starcoder_lua,starcoder_smalltalk,starcoder_agda,starcoder_f-sharp,starcoder_makefile,starcoder_solidity,starcoder_alloy,starcoder_fortran,starcoder_maple,starcoder_sparql,starcoder_antlr,starcoder_git-commits-cleaned,starcoder_markdown,starcoder_sql,starcoder_applescript,starcoder_github-issues-filtered-structured,starcoder_mathematica,starcoder_stan,starcoder_assembly,starcoder_glsl,starcoder_matlab,starcoder_standard-ml,starcoder_augeas,starcoder_go,starcoder_ocaml,starcoder_stata,starcoder_awk,starcoder_groovy,starcoder_pascal,starcoder_systemverilog,starcoder_batchfile,starcoder_haskell,starcoder_perl,starcoder_tcl,starcoder_bluespec,starcoder_html,starcoder_php,starcoder_tcsh,starcoder_c,starcoder_idris,starcoder_powershell,starcoder_tex,starcoder_c-sharp,starcoder_isabelle,starcoder_prolog,starcoder_thrift,starcoder_clojure,starcoder_java,starcoder_protocol-buffer,starcoder_typescript,starcoder_cmake,starcoder_java-server-pages,starcoder_python,starcoder_verilog,starcoder_coffeescript,starcoder_javascript,starcoder_r,starcoder_vhdl,starcoder_common-lisp,starcoder_json,starcoder_racket,starcoder_visual-basic,starcoder_cpp,starcoder_julia,starcoder_restructuredtext,starcoder_xslt,starcoder_css,starcoder_jupyter-scripts-dedup-filtered,starcoder_rmarkdown,starcoder_yacc,starcoder_cuda,starcoder_jupyter-structured-clean-dedup,starcoder_ruby,starcoder_yaml,starcoder_dart,starcoder_kotlin,starcoder_rust,starcoder_zig,starcoder_dockerfile,starcoder_lean,starcoder_sas,starcoder_elixir,starcoder_literate-agda,starcoder_scala,starcoder_elm,starcoder_literate-coffeescript,starcoder_scheme
STARCODER_PROGRAMMING_LANGUAGES = [
    "emacs-lisp",
    "literate-haskell",
    "shell",
    "ada",
    "erlang",
    "lua",
    "smalltalk",
    "agda",
    "f-sharp",
    "makefile",
    "solidity",
    "alloy",
    "fortran",
    "maple",
    "sparql",
    "antlr",
    "git-commits-cleaned",
    "markdown",
    "sql",
    "applescript",
    "github-issues-filtered-structured",
    "mathematica",
    "stan",
    "assembly",
    "glsl",
    "matlab",
    "standard-ml",
    "augeas",
    "go",
    "ocaml",
    "stata",
    "awk",
    "groovy",
    "pascal",
    "systemverilog",
    "batchfile",
    "haskell",
    "perl",
    "tcl",
    "bluespec",
    "html",
    "php",
    "tcsh",
    "c",
    "idris",
    "powershell",
    "tex",
    "c-sharp",
    "isabelle",
    "prolog",
    "thrift",
    "clojure",
    "java",
    "protocol-buffer",
    "typescript",
    "cmake",
    "java-server-pages",
    "python",
    "verilog",
    "coffeescript",
    "javascript",
    "r",
    "vhdl",
    "common-lisp",
    "json",
    "racket",
    "visual-basic",
    "cpp",
    "julia",
    "restructuredtext",
    "xslt",
    "css",
    "jupyter-scripts-dedup-filtered",
    "rmarkdown",
    "yacc",
    "cuda",
    "jupyter-structured-clean-dedup",
    "ruby",
    "yaml",
    "dart",
    "kotlin",
    "rust",
    "zig",
    "dockerfile",
    "lean",
    "sas",
    "elixir",
    "literate-agda",
    "scala",
    "elm",
    "literate-coffeescript",
    "scheme",
]


# files already downloaded => use parquet DS instead!
class StarcoderHFDataset(HFDataset):
    DATASET_ID = "starcoder"

    TITLE = "Starcoder"
    HOMEPAGE = "https://huggingface.co/datasets/bigcode/starcoderdata"
    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD
    LICENSE = "mixed permissive liceses (see https://huggingface.co/datasets/bigcode/the-stack-dedup)"
    LANGUAGES = ["code"]

    TOKENS = 250_000_000_000

    HF_DATASET_ID = "bigcode/starcoderdata"
    HF_DATASET_SPLIT = "train"

    text_column_name = "text"
    # remove_columns = ["doc_id", "LICENSE", "uri", "date_built"]


class StarcoderBaseDataset(ParquetDataset):
    """
    Output files: $LOCAL_DIR/<code_lang>/train-<i>-of-<n>.parquet
    """

    DATASET_ID = None
    PROGRAMMING_LANGUAGE = None

    SOURCE_ID = "starcoder"
    TITLE = "Starcoder"
    HOMEPAGE = "https://huggingface.co/datasets/bigcode/starcoderdata"
    AVAILIBILITY = Availability.SIGNIN_DOWNLOAD
    LICENSE = License(
        "mixed permissive liceses",
        url="https://huggingface.co/datasets/bigcode/the-stack-dedup",
        attribution=True,
        commercial_use=True,
        research_use=True,
        sharealike=False,
    )
    LANGUAGES = ["code"]

    TOKENS = 250_000_000_000 / len(STARCODER_PROGRAMMING_LANGUAGES)  # wrongly assuming uniform distribution

    HF_DATASET_ID = "bigcode/starcoderdata"
    HF_DATASET_SPLIT = "train"

    text_column_name = "text"
    # remove_columns = ["doc_id", "LICENSE", "uri", "date_built"]

    def get_output_text_field(self):
        # hard-coded text field in parquet schema
        return "content"

    def get_output_dir(self, shuffled=False):
        if shuffled:
            if self.shuffled_output_dir:
                return self.shuffled_output_dir
            raise ValueError("shuffled_output_dir is not set")
        else:
            return str(Path(self.get_local_dataset_dir()) / self.PROGRAMMING_LANGUAGE)

    def get_single_output_file_path(self, shuffled=False) -> str:
        return None

    def has_chunked_output_files(self, **kwargs):
        return True

    # def get_output_file_paths(self, single=False, chunked=False, shuffled=False):
    #     if shuffled is not None or single or not chunked:
    #         logger.warning(f"starcoder data is only provided as it is (no shuffled version, only chunked output files)")

    #     return (Path(self.get_local_dataset_dir()) / self.PROGRAMMING_LANGUAGE).rglob("*.parquet")

    def get_shuffled_output_file_path(self, unshuffled_output_file_path: str) -> str:
        output_file_name = Path(unshuffled_output_file_path).name
        shuffled_file_name = output_file_name.replace(".parquet", ".shuffled.parquet")

        return os.path.join(self.config.shuffled_output_dir, f"{self.DATASET_ID}_{shuffled_file_name}")

    def get_chunked_output_file_paths(self, shuffled=False) -> List[str]:
        if shuffled:
            # from shuffled output dir
            return list(Path(self.shuffled_output_dir).glob(f"{self.DATASET_ID}_*.shuffled.parquet"))
        else:
            # from starcoder data directory
            return list((Path(self.get_local_dataset_dir()) / self.PROGRAMMING_LANGUAGE).rglob("*.parquet"))


def get_starcoder_class(lang):
    class StarcoderDataset(StarcoderBaseDataset):
        DATASET_ID = "starcoder_" + lang
        PROGRAMMING_LANGUAGE = lang
        TITEL = f"Starcoder ({lang})"

    return StarcoderDataset


def get_auto_starcoder_classes():
    return [get_starcoder_class(lang) for lang in STARCODER_PROGRAMMING_LANGUAGES]
