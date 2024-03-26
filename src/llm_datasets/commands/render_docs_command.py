from argparse import Namespace, _SubParsersAction
import os
from pathlib import Path
from typing import Literal

import seaborn as sns
from matplotlib import pyplot as plt

from llm_datasets.shuffle_datasets import shuffle_datasets
from llm_datasets.commands import BaseCLICommand
from llm_datasets.utils.config import Config, get_config_from_paths
from llm_datasets.utils.dataframe import get_datasets_as_dataframe
from llm_datasets.utils.docs.plots import plot_tokens_by_language, plot_tokens_by_source
from llm_datasets.utils.docs.tables import (
    add_citation_to_title_row,
    get_tokens_by_language_dataframe,
    get_tokens_by_source_datafame,
    get_tokens_dataframe,
    tokens_by_source_dataframe_to_markdown,
)
from llm_datasets.utils.languages import LANGUAGE_CODE_TO_NAME
from llm_datasets.utils.settings import DEFAULT_MIN_FILE_SIZE_FOR_BUFFERED_SHUFFLING
from llm_datasets.viewer.viewer_utils import millify

AUTO_GEN_DISCLAIMER_MARKDOWN = "\n\n*This page is automatically generated.*\n\n"


class RenderDocsCommand(BaseCLICommand):
    docs_path = None
    docs_datasets_dir_path = None
    tokens_col = None

    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        subcommand_parser = parser.add_parser(
            "render_docs", help="Render files for documents (overview of datasets, statistics, plots)"
        )
        subcommand_parser.add_argument(
            "--docs_output_path",
            default=None,
            type=str,
            help="Path to `docs` directory (by default: ./docs in the base directory of the repository)",
        )
        subcommand_parser.add_argument(
            "--format",
            default="markdown",
            type=str,
            help="Output format (markdown, latex)",
        )
        subcommand_parser.add_argument(
            "--token_estimation_path",
            default=None,
            type=str,
            help="token_estimation_path",
        )
        subcommand_parser.add_argument(
            "--metrics_dir",
            default=None,
            type=str,
            help="metrics_dir",
        )
        subcommand_parser.add_argument(
            "--render_plots",
            action="store_true",
            help="Render plots",
        )
        subcommand_parser.add_argument(
            "--only_selected_datasets",
            action="store_true",
            help="Include only datasets, languages or datasources that are tagged as `selected` in the config",
        )

        subcommand_parser = BaseCLICommand.add_common_args(
            subcommand_parser,
            raw_datasets_dir=False,
            output=False,
            extra_dataset_registries=True,
            configs=True,
            required_configs=False,
            log=True,
        )
        subcommand_parser.set_defaults(func=RenderDocsCommand)

    def __init__(self, args: Namespace) -> None:
        self.config: Config = get_config_from_paths(args.config_paths, override=args.__dict__)

    def run(self) -> None:
        config = self.config
        logger = config.init_logger(__name__)

        sns.set_theme()

        # Index of datasets
        self.docs_path = Path(config.docs_output_path)
        self.docs_datasets_dir_path = self.docs_path / "datasets"

        if not self.docs_datasets_dir_path.exists():
            logger.warning(f"Directory does not exist yet, creating: {str(self.docs_datasets_dir_path)}")
            os.makedirs(self.docs_datasets_dir_path, exist_ok=True)

        if config.token_estimation_path and config.metrics_dir:
            logger.info("Using estimated tokens because `token_estimation_path` and `metrics_dir` are provided")
            self.tokens_col = "estimated_tokens"
        else:
            logger.info("Using reported tokens")
            self.tokens_col = "reported_tokens"

        tokens_df = get_tokens_dataframe(
            config,
            extra_columns=["citation", "license", "homepage", "description"],
        )

        tokens_by_lang_df = get_tokens_by_language_dataframe(tokens_df, tokens_col=self.tokens_col)
        tokens_by_source_df = get_tokens_by_source_datafame(tokens_df, tokens_col=self.tokens_col)

        if config.format == "markdown":
            self.dataset_languages_markdown(tokens_df, tokens_by_lang_df)
            self.datasets_index_markdown(tokens_df, tokens_by_lang_df, tokens_by_source_df)
        elif config.format == "latex":
            self.sources_latex(tokens_by_source_df)
            self.languages_latex(tokens_by_lang_df)

        else:
            logger.error("Unknown format")

        logger.info("done")

    def languages_latex(self, tokens_by_lang_df):
        """
        Generate `languages.tex` and `tokens_by_language.pdf`
        """
        df = tokens_by_lang_df.copy()

        langs_tex_path = self.docs_datasets_dir_path / "languages.tex"

        # plot
        tokens_by_language_plot_file_name = "tokens_per_language.pdf"
        plot_tokens_by_language(
            df.sort_values(self.tokens_col, ascending=False),
            self.tokens_col,
            self.docs_datasets_dir_path / tokens_by_language_plot_file_name,
            save_format="pdf",
        )
        factor = 1_000_000
        # Langage (lang-code) | Estimated tokens (M)
        df = df.reset_index().sort_values(self.tokens_col, ascending=False)
        df["language"] = [
            (LANGUAGE_CODE_TO_NAME[lang_code] if lang_code in LANGUAGE_CODE_TO_NAME else lang_code.title())
            + f" ({lang_code})"
            for lang_code in df["language"].values
        ]
        total_tokens = df[self.tokens_col].sum()
        df["percentage"] = [f"{(t * 100):.2f}" for t in (df[self.tokens_col] / total_tokens).values]

        df[self.tokens_col] = [f"{round(t / factor):,}" for t in df[self.tokens_col].values]

        df.to_latex(langs_tex_path, escape=True, index=False)
        print(f"Total tokens: {round(total_tokens / factor):,} M")
        print(df.head())

        pass

    def sources_latex(self, tokens_by_source_df):
        """
        Generate `sources.tex` and `sources.bib`
        """
        df = tokens_by_source_df.copy()

        sources_tex_path = self.docs_datasets_dir_path / "sources.tex"
        sources_bib_path = self.docs_datasets_dir_path / "sources.bib"

        bibtex = "\n\n".join(df["citation"].dropna().unique())
        with open(sources_bib_path, "w") as f:
            f.write(bibtex)

        df[self.tokens_col] = df[self.tokens_col].apply(millify)
        df["title"] = df.apply(add_citation_to_title_row, axis=1)
        df["description"] = df["description"].apply(lambda s: "" if s is None else s.replace("\n", " "))

        # remove columns
        df.drop(["homepage"], axis=1, inplace=True)
        df.drop(["citation"], axis=1, inplace=True)

        tex = df.to_latex(escape=True, index=False)
        tex = tex.replace("CITESTART", "\citep{").replace("CITEEND", "}")

        with open(sources_tex_path, "w") as f:
            f.write(tex)

        print("saves tex")

    def millify_tokens(self, df):
        return df[self.tokens_col].apply(millify)

    def dataset_languages_markdown(self, tokens_df, tokens_by_lang_df):
        # remove existing lang files
        for fp in self.docs_datasets_dir_path.glob("language_*.md"):
            os.remove(fp)

        # one list of datasets for each language
        for lang_code, tokens_cell in tokens_by_lang_df.iterrows():
            total_tokens_per_lang = tokens_cell[0]
            lang_path = self.docs_datasets_dir_path / f"language_{lang_code}.md"
            lang_name = LANGUAGE_CODE_TO_NAME[lang_code] if lang_code in LANGUAGE_CODE_TO_NAME else lang_code.title()
            lang_md = f"# {lang_name} Datasets\n\n"
            selector = tokens_df.language == lang_code
            datasets_df = tokens_df[selector]
            lang_md += f"There are in total {len(datasets_df):,} datasets with {millify(total_tokens_per_lang)} tokens in {lang_name} language.\n\n"

            for idx, ds in datasets_df.sort_values("title").iterrows():
                # | **Citation:**         | ```bibtex<br>{ds.citation}         |
                lang_md += f"## {ds.title}\n\n"
                lang_md += f"""| **Dataset ID:**       | `{ds.dataset_id}`       |
|-----------------------|-----------------------|
| **Title:**            | {ds.title}            |
| **Description:**      | {ds.description}      |
| **Availibility:**     | `{ds.availibility}`     |
| **Homepage:**         | [{ds.homepage}]         |
| **License:**          | {ds.license}          |
| **Tokens:** | {millify(ds[self.tokens_col])} |\n\n"""

                # break
            lang_md += AUTO_GEN_DISCLAIMER_MARKDOWN

            with open(lang_path, "w") as f:
                f.write(lang_md)

            # break

        pass

    def datasets_index_markdown(self, tokens_df, tokens_by_lang_df, tokens_by_source_df):
        index_path = self.docs_datasets_dir_path / "index.md"

        # Full language names
        lang_list = [
            LANGUAGE_CODE_TO_NAME[lang_code] if lang_code in LANGUAGE_CODE_TO_NAME else lang_code.title()
            for lang_code in tokens_by_lang_df.index
        ]

        index_md = """  # Datasets\n\n"""

        index_md += (
            f"The framework provides {len(tokens_df)} datasets from {len(tokens_by_source_df)} sources in {len(lang_list)} languages. The languages are as follows: "
            + ", ".join(lang_list)
        )
        index_md += "\n\n"

        # By language
        index_md += "\n\n## Languages\n"
        tokens_by_language_plot_file_name = "tokens_by_language.png"
        plot_tokens_by_language(
            tokens_by_lang_df.sort_values(self.tokens_col, ascending=False),
            self.tokens_col,
            self.docs_datasets_dir_path / tokens_by_language_plot_file_name,
            save_format="png",
            top_k=15,
        )
        index_md += f"![Tokens by language]({tokens_by_language_plot_file_name})\n\n"

        index_md += self.millify_tokens(tokens_by_lang_df).to_markdown()

        index_md += "\n\n"

        # By source
        index_md += "\n\n## Data sources\n"

        tokens_by_source_file_name = "tokens_by_source.png"
        plot_tokens_by_source(
            tokens_by_source_df.sort_values(self.tokens_col, ascending=False),
            tokens_col=self.tokens_col,
            save_to_path=self.docs_datasets_dir_path / tokens_by_source_file_name,
            save_format="png",
            top_k=15,
        )
        index_md += f"![Tokens by source]({tokens_by_source_file_name})\n\n"

        index_md += self.millify_tokens(tokens_by_source_dataframe_to_markdown(tokens_by_source_df)).to_markdown()

        index_md += AUTO_GEN_DISCLAIMER_MARKDOWN

        # print(index_md)

        with open(index_path, "w") as f:
            f.write(index_md)
