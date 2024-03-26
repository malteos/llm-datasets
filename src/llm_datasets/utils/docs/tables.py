import re
from typing import Literal

from llm_datasets.utils.config import Config
from llm_datasets.utils.dataframe import get_datasets_as_dataframe


from llm_datasets.utils.docs import TokensColumn
from llm_datasets.viewer.viewer_utils import millify


def get_tokens_dataframe(config, **dataframe_kwargs):
    config.rows_count = False
    config.shuffled_rows_count = False
    config.exclude_dummy_datasets = True

    df = get_datasets_as_dataframe(
        output_dir=None,
        output_format=config.output_format,
        shuffled_output_dir=config.shuffled_output_dir,
        raw_datasets_dir=config.raw_datasets_dir,
        extra_dataset_registries=config.extra_dataset_registries,
        exclude_dummy_datasets=config.exclude_dummy_datasets,
        show_progress=True,
        metrics_dir=config.metrics_dir,
        token_estimation_path=config.token_estimation_path,
        config=config,
        **dataframe_kwargs,
    )
    # rename tokens
    df.rename(columns={"tokens": "reported_tokens", "total_estimated_tokens": "estimated_tokens"}, inplace=True)

    return df


def get_tokens_by_language_dataframe(df, tokens_col: TokensColumn = "estimated_tokens", remove_zero_rows: bool = False):
    group_df = df.groupby("language")[[tokens_col]].sum()

    if remove_zero_rows:
        group_df = group_df[(group_df > 0).values]

    # group_df = group_df[tokens_col].apply(millify)

    return group_df


def get_tokens_by_source_datafame(df, tokens_col: TokensColumn = "estimated_tokens"):
    group_df = df.groupby("source_id")[[tokens_col]].sum()
    # group_df = group_df[(group_df > 0).values]

    sources = (
        df.drop_duplicates(subset=["source_id"])
        .set_index("source_id")[["title", "homepage", "description", "citation"]]  # citation
        .join(group_df)
    )

    return sources


def get_title_without_subset(title):
    """
    Subset identifier are marked with " [subset name]" in the end of dataset titles.
    """
    return re.sub(r" \[(.*?)\]$", "", title)


def add_homepage_link_to_title_row(row, title_column="title", homepage_column="homepage") -> str:
    title = get_title_without_subset(row[title_column])
    homepage = row[homepage_column]
    return f'<a href="{homepage}">{title}</a>'


def tokens_by_source_dataframe_to_markdown(sources):
    # sources[tokens_col] = sources[tokens_col].apply(millify)
    sources["title"] = sources.apply(add_homepage_link_to_title_row, axis=1)
    sources["description"] = sources["description"].apply(lambda s: "" if s is None else s.replace("\n", " "))

    sources.drop(["homepage"], axis=1, inplace=True)

    return sources


def add_citation_to_title_row(row, title_column="title", citation_column="citation") -> str:
    t = get_title_without_subset(row[title_column])

    if row["citation"] is not None:
        match = re.search(r"@([a-zA-Z]+)\{(.+?),", row[citation_column])
        if match:
            citation_key = match.group(2)

            if citation_key is not None:
                t += " CITESTART" + citation_key + "CITEEND"

    return t
