# llm-datasets

<img align="left" src="https://github.com/malteos/llm-datasets/raw/main/docs/images/A_colorful_parrot_sitting_on_a_pile_of_books__whit-removebg-preview.png" height="200" />

![](https://img.shields.io/pypi/l/llm-datasets?style=flat-square)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

**llm-datasets is a collection of datasets for language model training including scripts for downloading, preprocesssing, and sampling.**

The documentation is available [here](https://malteos.github.io/llm-datasets/).

## Quick start

### Installation

Install the `llm-datasets` package with [pip](https://pypi.org/project/llm-datasets/):

```bash
pip install llm-datasets
```

In order to keep the package minimal by default, `llm-datasets` comes with optional dependencies useful for some use cases.
For example, if you want to have the text extraction for all available datasets, run:

```bash
pip install llm-datasets[datasets]
```

### Download and text extraction

To download and extract the plain-text of one or more datasets, run the following command:

```bash
llm-datasets extract_text $DATASET_ID $OUTPUT_DIR
```

By default, output is saved as JSONL files. To change the output format, you can use the `--output_format` argument as below:

```bash
llm-datasets extract_text $DATASET_ID $OUTPUT_DIR --output_format parquet  --output_compression zstd
```

### Available datasets

A list or table with all available datasets can be print with the follow command:

```bash
llm-datasets print_stats --print_output md
```
#### Token count by language

| Language   | Tokens   |
|:-----------|:---------|
| bg         | 31 B               |
| ca         | 6 B                |
| code       | 212 B              |
| cs         | 42 B               |
| da         | 13 B               |
| de         | 160 B              |
| el         | 63 B               |
| en         | 1 T                |
| es         | 101 B              |
| et         | 9 B                |
| eu         | 1 B                |
| fi         | 19 B               |
| fr         | 84 B               |
| ga         | 274 M              |
| gl         | 231 M              |
| hr         | 11 B               |
| hu         | 52 B               |
| it         | 61 B               |
| lt         | 7 B                |
| lv         | 5 B                |
| mt         | 4 B                |
| nl         | 44 B               |
| nn         | 76 M               |
| no         | 13 B               |
| pl         | 45 B               |
| pt         | 46 B               |
| ro         | 18 B               |
| sh         | 184 M              |
| sk         | 32 B               |
| sl         | 13 B               |
| sr         | 11 B               |
| sv         | 19 B               |
| uk         | 56 B               |

#### Token count by source

| Source                       | Tokens   |
|:---------------------------------|:---------|
| curlicat                         | 963 M              |
| macocu                           | 74 B               |
| redpajama                        | 44 B               |
| wura                             | N/A                |
| wikihow                          | 99 M               |
| pes2o                            | 57 B               |
| proof_pile                       | 12 B               |
| pile_of_law                      | 111 B              |
| math_amps                        | 7 B                |
| edgarcorpus                      | N/A                |
| bulgarian_news                   | 640 M              |
| bulnc                            | 4 B                |
| openlegaldata                    | 7 B                |
| dewac                            | 3 B                |
| ga_bilingual_legistation         | 4 k                |
| ga_universal_dependencies        | 40 k               |
| hrwac                            | 2 B                |
| styria_news                      | 432 M              |
| croatian_news_engri              | 1 B                |
| itwac                            | 3 B                |
| korpus_malti                     | 816 M              |
| sonar                            | 746 M              |
| cc_gigafida                      | 260 M              |
| academic_slovene_kas             | 3 B                |
| slwac_web                        | 3 B                |
| sk_court_decisions               | 24 B               |
| sk_laws                          | 105 M              |
| syn_v9                           | 13 B               |
| cs_en_parallel                   | 473 M              |
| danish_gigaword                  | 2 B                |
| danewsroom                       | 835 M              |
| dk_clarin                        | 80 M               |
| cabernet                         | 599 M              |
| norwegian_cc                     | 11 B               |
| pl_nkjp                          | 3 M                |
| pl_parliamentary_corpus          | 1 B                |
| parlamento_pt                    | 732 M              |
| brwac                            | 4 B                |
| seimas_lt_en                     | 12 k               |
| state_related_latvian_web        | 52 k               |
| greek_legal_code                 | 80 M               |
| greek_web_corpus                 | 11 B               |
| estonian_reference_corpus        | 481 M              |
| enc2021                          | 3 B                |
| ekspress                         | 723 M              |
| euscrawl                         | 831 M              |
| spanish_legal                    | 1 B                |
| ylenews                          | 286 M              |
| sv_gigaword                      | 528 M              |
| srpkor                           | 866 M              |
| marcell_legislative_subcorpus_v2 | 1 B                |
| uk_laws                          | 2 B                |
| eurlex                           | 41 B               |
| legal_mc4                        | 28 B               |
| wiki                             | 21 B               |
| wikibooks                        | 313 M              |
| wikiquote                        | 247 M              |
| wikinews                         | 90 M               |
| wikisource                       | 2 B                |
| wikivoyage                       | 119 M              |
| colossal_oscar                   | 2 T                |
| starcoder                        | 212 B              |


### Dataset viewer

We provide a Web-based application through streamlit to browse all datasets and their contained text content.
To start the app, first clone this repository, install dependencies, and run the following command:

```bash
# clone is needed since streamlit does not support apps from modules yet
git clone https://github.com/malteos/llm-datasets.git

streamlit run src/lm_datasets/viewer/app.py -- \
    --raw_datasets_dir=$RAW_DATASETS_DIR \
    --output_dir=$PROCESSED_DATASET_DIR
```


## Development & Contributions

### Setup environment

To setup, your local development environment we recommend conda and cloning the repository.
The repository also includes settings and launch scripts for VSCode.

```bash
git clone git@github.com:malteos/llm-datasets.git
cd llm-datasets

conda create -n llm-datasets python=3.10
conda activate llm-datasets

pip install -r requirements.txt
```

Alternatively, you can install the Python package directly from the dev branch:

```bash
pip install git+https://github.com/malteos/llm-datasets.git@dev
```

### Install the pre-commit hooks

This repository uses git hooks to validate code quality and formatting.

```bash
pre-commit install
git config --bool flake8.strict true  # Makes the commit fail if flake8 reports an error
```

To run the hooks:
```bash
pre-commit run --all-files
```

### Testing

The tests can be executed with:
```bash
pytest --doctest-modules --cov-report term --cov=llm_datasets
```

## Acknowledgements

The work on the llm-datasets software is partially funded by the [German Federal Ministry for Economic Affairs and Climate Action (BMWK)](https://www.bmwk.de/Navigation/EN/Home/home.html)
through the project [OpenGPT-X](https://opengpt-x.de/en/) (project no. 68GX21007D).

## License

Apache 2.0

*(Please note that the actual datasets are released with different licenses)*

