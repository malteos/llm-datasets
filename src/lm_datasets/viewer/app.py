"""
Streamlit app
"""
import math
import os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import textwrap
import sys
import pyarrow.parquet as pq
import random
import logging
from textwrap import TextWrapper
import argparse

from viewer_utils import millify, sizeof_fmt

from lm_datasets.datasets.dataset_registry import get_registered_dataset_classes
from lm_datasets.datasets.base import BaseDataset
from lm_datasets.utils.dataframe import get_datasets_as_dataframe
from lm_datasets.utils.config import get_common_argparser, parse_args_and_get_config
from lm_datasets.utils.dataframe import get_datasets_as_dataframe

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

logger.info("Starting streamlit app")

# parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(parents=[get_common_argparser()], add_help=False)

parser.add_argument(
    "--shuffled_output_dir",
    default=None,
    type=str,
    help="Shuffled dataset are saved in this directory",
)
parser.add_argument(
    "--dataframe_cache_path",
    default=None,
    type=str,
    help="Save and load dataframe cache file to this path (no cache if not set)",
)
parser.add_argument(
    "--only_selected_datasets",
    action="store_true",
    help="Include only datasets there were explicitly selected (via config)",
)
parser.add_argument("--rows_count", action="store_true", help="Extract number of rows from output files")
# args = parser.parse_args()
config = parse_args_and_get_config(parser)

raw_datasets_dir = config.raw_datasets_dir
output_dir = config.output_dir

styl = """
<style>
    textarea:disabled {
    background-color: #fff;
    color: #000;
    opacity: 1;
    -webkit-text-fill-color: #000;
    }
    div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
    }
</style>
"""

app_state = st.experimental_get_query_params()
# print(app_state)
start = True
loaded = True


st.set_page_config(layout="wide")

id_to_dataset_class = {
    cls.DATASET_ID: cls
    for cls in get_registered_dataset_classes(extra_dataset_registries=config.extra_dataset_registries)
}

logger.info("Loading df ...")


# @st.cache_data
def load_data():
    if config.dataframe_cache_path is None or (
        config.dataframe_cache_path is not None and not os.path.exists(config.dataframe_cache_path)
    ):
        _df = get_datasets_as_dataframe(
            output_dir=output_dir,
            shuffled_output_dir=config.shuffled_output_dir,
            raw_datasets_dir=raw_datasets_dir,
            output_format="parquet",
            extra_dataset_registries=config.extra_dataset_registries,
            rows_count=config.rows_count,
            config=config,
        )
    else:
        # load from disk
        logger.info("Reading df from disk: %s", config.dataframe_cache_path)
        _df = pd.read_csv(config.dataframe_cache_path, index_col=None)

    # save to disk?
    if config.dataframe_cache_path and not os.path.exists(config.dataframe_cache_path):
        logger.info("Writing df to disk: %s", config.dataframe_cache_path)
        _df.to_csv(config.dataframe_cache_path, index=False)

    return _df


df = load_data()

logger.info("Filtering df ...")
# selector_has_output_file = df.has_output_file == 1
# df = df[selector_has_output_file]

# Exclude dummies
selector_is_non_dummy = df.dummy != 1
df = df[selector_is_non_dummy]

# Exclude oscar
selector_is_non_oscar = ~df.source_id.str.contains("oscar")
df = df[selector_is_non_oscar]

# df = df.head()

# Exclude opengptx
# selector_is_non_opengptx = ~df.source_id.str.contains("opengptx")
# df = df[selector_is_non_opengptx]

# Exclude code
# selector_is_non_code = df.language != "code"
# df = df[selector_is_non_code]

logger.info(f"All columns loaded: {df.columns}")

selected_columns = [
    "dataset_id",
    # "title",
    # "source_id",
    "language",
    "tokens",
    # "web_crawled",
]

if config.output_dir:
    selected_columns.append("has_output_files")

if config.shuffled_output_dir:
    selected_columns.append("has_shuffled_output_files")

if config.rows_count:
    selected_columns.append("rows_count")

    if config.shuffled_output_dir:
        selected_columns.append("shuffled_rows_count")


# Sidebar
# st.sidebar.image(
#     str(
#         Path(__file__).parent.parent.parent.parent
#         / "images/A_colorful_parrot_sitting_on_a_pile_of_books__whit-removebg-preview.png"
#     ),
#     width=75,
# )
st.sidebar.markdown(
    """<center>
<h2><a href="https://github.com/malteos/lm-datasets">lm-datasets</a><br />Dataset Viewer</h2>
</center>""",
    unsafe_allow_html=True,
)
st.sidebar.subheader("")


mode_options = ["overview", "stats", "dataset_details"]
mode_index = 0

if "mode" in app_state:
    # print("app_state mode = ", app_state["mode"])
    mode_index = mode_options.index(app_state["mode"][0])
    # print("index ", mode_index)

mode = st.sidebar.selectbox(
    "Mode",
    mode_options,
    index=mode_index,
    # format_func=lambda a: a
)

app_state["mode"] = mode
st.experimental_set_query_params(**app_state)

if mode == "overview":
    st.header("Overview")

    if "dataset_id" in app_state:
        del app_state["dataset_id"]

    if "selected_row_group" in app_state:
        del app_state["selected_row_group"]

    if "offset" in app_state:
        del app_state["offset"]

    st.experimental_set_query_params(**app_state)

    # st.table(
    #     data=pd.DataFrame([dict(a=1), dict(a=4)]),
    #     # use_container_width=True,
    #     # height=(len(df) + 1) * 35 + 3
    # )

    # View selected columbs of dataframe
    st.dataframe(data=df[selected_columns], use_container_width=True, height=(len(df) + 1) * 35 + 3)


elif mode == "stats":
    st.header("Statistics")

    # Summary
    st.markdown(f"*Total datasets*: {len(df)}")
    # st.markdown(f"*Total bytes*: {sizeof_fmt(df.bytes.sum())}")
    st.markdown(f"*Total tokens*: {millify(df.tokens.sum())}")
    st.markdown(f"*Unique languages*: {len(df.language.unique())}")
    st.markdown(f"*Unique sources*: {len(df.source_id.unique())}")

    # Plots
    import plotly.express as px

    tokens_by_language = df.groupby("language")["tokens"].sum().sort_values(ascending=False).reset_index()
    tokens_by_web_crawled = df.groupby("web_crawled")["tokens"].sum().sort_values(ascending=False).reset_index()

    fig = px.bar(tokens_by_language, x="language", y="tokens", title="Tokens by language")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.pie(
        tokens_by_web_crawled, names="web_crawled", values="tokens", title="Tokens by Web-crawled (1) or not (0)"
    )
    st.plotly_chart(fig, use_container_width=True)

elif mode == "dataset_details":
    dataset_id_options = list(df.dataset_id.unique())
    dataset_id_index = 0

    if "dataset_id" in app_state:
        logger.info("app_state dataset_id = %s", app_state["dataset_id"])
        dataset_id_index = dataset_id_options.index(app_state["dataset_id"][0])
        logger.info("dataset_id_index = %s", dataset_id_index)

    # selection = "opengptx_costep_de"  #  dataset_ids[0]

    selected_dataset_id = st.sidebar.selectbox(
        "Dataset", dataset_id_options, index=dataset_id_index, format_func=lambda a: a
    )

    app_state["dataset_id"] = selected_dataset_id
    st.experimental_set_query_params(**app_state)

    # Main
    ds: BaseDataset = id_to_dataset_class[selected_dataset_id](
        output_dir=output_dir,
        shuffled_output_dir=config.shuffled_output_dir,
        raw_datasets_dir=raw_datasets_dir,
        output_format="parquet",
        config=config,
    )

    st.title("Dataset: " + selected_dataset_id)

    st.markdown(f"*Description*: {ds.DESCRIPTION}")
    st.markdown(f"*Homepage*: {ds.HOMEPAGE}")
    st.markdown(f"*Estimated tokens*: {ds.get_tokens()}")
    st.markdown(f"*Language*: {ds.get_language_code()}")
    st.markdown(f"*License*: {ds.LICENSE}")
    st.markdown(f"*Known quality issues*: {ds.QUALITY_WARNINGS}")
    st.markdown(f"*Genres*: {ds.GENRES}")

    # st.markdown(f"*Is downloaded?*: {ds.is_downloaded()}")

    st.markdown(styl, unsafe_allow_html=True)

    if config.shuffled_output_dir:
        use_shuffled_output_files = st.sidebar.checkbox("Shuffled output files")
    else:
        use_shuffled_output_files = False

    if ds.has_output_files(min_file_size=271, shuffled=use_shuffled_output_files):  # empty parquet files have 270 bytes
        st.header("Text preview:")

        TEXT_COLUMN_NAME = "text"
        MAX_TEXT_LENGTH = 50_000

        output_file_paths = list(sorted(ds.get_output_file_paths(shuffled=use_shuffled_output_files)))

        output_file_index = st.sidebar.number_input(
            f"Output file index (max: {len(output_file_paths) - 1})",
            value=0,
            min_value=0,
            max_value=len(output_file_paths) - 1,
            step=1,
        )

        pq_fp = str(output_file_paths[output_file_index])  # using only part 1 if multiple chunks

        st.markdown(f"*Output file*: `{pq_fp}`")

        with open(pq_fp, "rb") as file_handler:
            logger.info(f"Opening {pq_fp}")

            parquet_file = pq.ParquetFile(file_handler)

            logger.info("Reading metadata from parquet file ...")

            metadata = parquet_file.metadata
            num_row_groups = metadata.num_row_groups
            num_rows = metadata.num_rows  # pq_file.metadata.num_rows  # TODO

            rows_per_group = math.ceil(num_rows / num_row_groups)

            # print(metadata)

            # parquet_file = pq.ParquetFile(pq_fp)

            max_row_group = num_row_groups - 1
            # row_group_initial = 0

            if "row_group" not in st.session_state:
                if "selected_row_group" in app_state:
                    st.session_state.row_group = int(app_state["selected_row_group"][0])
                else:
                    st.session_state.row_group = 0

            if st.sidebar.button("Random offset"):
                # random_row_group =
                # row_group_initial = random_row_group
                st.session_state.row_group = random.randint(0, max_row_group)
                # st.sidebar.write(f"rand = {random_row_group}")
                use_random_offset = True
            else:
                use_random_offset = False

            selected_row_group = st.sidebar.number_input(
                "Row group (Max: %d)" % max_row_group,
                min_value=0,
                max_value=max_row_group,
                # value=row_group_initial,
                step=1,
                key="row_group",
            )
            app_state["selected_row_group"] = selected_row_group
            st.experimental_set_query_params(**app_state)

            st.markdown(f"*Row group*: `{selected_row_group}`")

            logger.info("Reading group")
            # try:
            row_group = parquet_file.read_row_group(selected_row_group)
            # except OSError:
            #     # Dirty bugfix: Some times reading at the first try does not work => simply retry
            #     parquet_file = pq.ParquetFile(pq_fp)
            #     row_group = parquet_file.read_row_group(selected_row_group)

            texts = row_group[TEXT_COLUMN_NAME]

            step = 50
            max_offset = max(1, len(texts) - step)

            if "offset" not in st.session_state:
                if "selected_offset" in app_state:
                    st.session_state.offset = int(app_state["selected_offset"][0])
                else:
                    st.session_state.offset = 0

            if use_random_offset:
                st.session_state.offset = random.randint(0, max_offset)

            # else:
            #     # if "offset" in app_state:
            #     #     st.session_state.offset = int(app_state["offset"][0])
            #     # else:
            #     st.session_state.offset = 0

            offset = st.sidebar.number_input(
                "Offset (Max: %d)" % max_offset,
                min_value=0,
                max_value=max_offset,
                # value=0,
                step=step,
                key="offset",
            )
            app_state["selected_offset"] = offset
            st.experimental_set_query_params(**app_state)

            st.markdown(f"*Offset*: `{offset}`")

            # Plain text viewer
            for idx in range(offset, min(offset + step, len(texts))):
                # st.text("        ")
                st.subheader(f"#{selected_row_group}.{idx}")
                # st.text("        ")

                text = str(texts[idx])

                # wrapper = TextWrapper(width=200, max_lines=50, replace_whitespace=False, drop_whitespace=False)

                # # text = "".join(["Foobar..."] * 1000)

                # wrapped_lines = wrapper.wrap(text)
                # wrapper_text = """\n<mark>&lt;wrap&gt;</mark> """.join(wrapped_lines)

                if len(text) > MAX_TEXT_LENGTH:
                    text = text[:MAX_TEXT_LENGTH] + " ... [truncated]"

                # # st.text(textwrap.fill(, width=120))
                # # st.text(f"<pre>{wrapper_text}</pre>")
                # # wrapper_text = text

                # # st.write(f"<pre>{wrapper_text}</pre>", unsafe_allow_html=True)
                # st.text(wrapper_text)

                # st.text(text)
                st.text_area(
                    label=f"{selected_row_group}.{idx}",
                    value=text,
                    height=450,
                    disabled=True,
                    label_visibility="hidden",
                )

                st.divider()

    else:
        st.error("Preview not possible!")
        st.markdown(
            f"The dataset's output file does not exist (or is empty): `{ds.get_output_file_paths(shuffled=use_shuffled_output_files)}`"
        )

else:
    st.error("Invalid mode selected.")
