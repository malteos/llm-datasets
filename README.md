# lm-datasets

<img align="left" src="https://github.com/malteos/lm-datasets/raw/main/docs/images/A_colorful_parrot_sitting_on_a_pile_of_books__whit-removebg-preview.png" height="200" />

![](https://img.shields.io/pypi/l/lm-datasets?style=flat-square)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

**lm-datasets is a collection of datasets for language model training including scripts for downloading, preprocesssing, and sampling.**

The documentation is available [here](https://malteos.github.io/lm-datasets/).

## Quick start

### Installation

Install the `lm-datasets` package with [pip](https://pypi.org/project/lm-datasets/):

```bash
pip install lm-datasets
```

In order to keep the package minimal by default, `lm-datasets` comes with optional dependencies useful for some use cases.
For example, if you want to have the text extraction for all available datasets, run:

```bash
pip install lm-datasets[datasets]
```

### Download and text extraction

To download and extract the plain-text of one or more datasets, run the following command:

```bash
lm_datasets extract_text $DATASET_ID $OUTPUT_DIR
```

By default, output is saved as JSONL files. To change the output format, you can use the `--output_format` argument as below:

```bash
lm_datasets extract_text $DATASET_ID $OUTPUT_DIR --output_format parquet  --output_compression zstd
```

### Available datasets

A list or table with all available datasets can be print with the follow command:

```bash
lm_datasets print_stats --print_output md
```
#### Token count by language

| Language   | Tokens   |
|:-----------|:---------|
| bg         | 53 B     |
| ca         | 5 B      |
| code       | 250 B    |
| cs         | 128 B    |
| da         | 34 B     |
| de         | 795 B    |
| el         | 108 B    |
| en         | 6 T      |
| es         | 674 B    |
| et         | 15 B     |
| eu         | 696 M    |
| fi         | 55 B     |
| fr         | 655 B    |
| ga         | 767 M    |
| gl         | 70 M     |
| hr         | 8 B      |
| hu         | 179 B    |
| it         | 386 B    |
| lt         | 24 B     |
| lv         | 14 B     |
| mt         | 4 B      |
| nl         | 238 B    |
| nn         | 307 M    |
| no         | 9 B      |
| pl         | 223 B    |
| pt         | 187 B    |
| ro         | 77 B     |
| sh         | 2 M      |
| sk         | 47 B     |
| sl         | 11 B     |
| sr         | 10 B     |
| sv         | 89 B     |
| uk         | 47 B     |

#### Token count by source

| Source                       | Tokens   |
|:---------------------------------|:---------|
| academic_slovene_kas             | 1 B      |
| bgnc_admin_eur                   | 79 M     |
| bgnc_news_corpus                 | 18 M     |
| brwac                            | 3 B      |
| bulgarian_news                   | 283 M    |
| bulnc                            | 567 M    |
| cabernet                         | 712 M    |
| cc_gigafida                      | 127 M    |
| colossal_oscar                   | 208 B    |
| croatian_news_engri              | 695 M    |
| curlicat                         | 410 M    |
| danewsroom                       | 472 M    |
| danish_gigaword                  | 1 B      |
| dewac                            | 2 B      |
| dialogstudio                     | 0        |
| dk_clarin                        | 441 M    |
| enc2021                          | 0        |
| estonian_reference_corpus        | 175 M    |
| eurlex                           | 121 B    |
| euscrawl                         | 423 M    |
| ga_bilingual_legistation         | 4 M      |
| ga_universal_dependencies        | 3 M      |
| greek_legal_code                 | 45 M     |
| greek_web_corpus                 | 3 B      |
| hrwac                            | 1 B      |
| itwac                            | 2 B      |
| korpus_malti                     | 366 M    |
| legal_mc4                        | 29 B     |
| macocu                           | 23 B     |
| marcell_legislative_subcorpus_v2 | 31 M     |
| norwegian_cc                     | 5 B      |
| openlegaldata                    | 10 B     |
| oscar                            | 9 T      |
| oscar_opengptx                   | 245 B    |
| parlamento_pt                    | 819 M    |
| pes2o                            | 42 B     |
| pl_nkjp                          | 1 M      |
| pl_parliamentary_corpus          | 671 M    |
| proof_pile                       | 8 B      |
| redpajama                        | 46 B     |
| seimas_lt_en                     | 48 k     |
| sk_court_decisions               | 11 B     |
| sk_laws                          | 45 M     |
| slwac_web                        | 1 B      |
| sonar                            | 500 M    |
| sonar_new_media                  | 36 M     |
| spanish_legal                    | 3 B      |
| srpkor                           | 0        |
| starcoder                        | 250 B    |
| state_related_latvian_web        | 1 M      |
| styria_news                      | 409 M    |
| sv_gigaword                      | 1 B      |
| syn_v9                           | 5 B      |
| uk_laws                          | 579 M    |
| wiki                             | 12 B     |
| wikibooks                        | 353 M    |
| wikihow                          | 2 M      |
| wikinews                         | 79 M     |
| wikiquote                        | 268 M    |
| wikisource                       | 2 B      |
| wikivoyage                       | 132 M    |
| ylenews                          | 0        |


### Dataset viewer

We provide a Web-based application through streamlit to browse all datasets and their contained text content.
To start the app, first clone this repository, install dependencies, and run the following command:

```bash
# clone is needed since streamlit does not support apps from modules yet
git clone https://github.com/malteos/lm-datasets.git

streamlit run src/lm_datasets/viewer/app.py -- \
    --raw_datasets_dir=$RAW_DATASETS_DIR \
    --output_dir=$PROCESSED_DATASET_DIR
```


## Development & Contributions

### Setup environment

To setup, your local development environment we recommend conda and cloning the repository.
The repository also includes settings and launch scripts for VSCode.

```bash
git clone git@github.com:malteos/lm-datasets.git
cd lm-datasets

conda create -n lm-datasets python=3.10
conda activate lm-datasets

pip install -r requirements.txt
```

Alternatively, you can install the Python package directly from the dev branch:

```bash
pip install git+https://github.com/malteos/lm-datasets.git@dev
```

### Install the pre-commit hooks

This repository uses git hooks to validate code quality and formatting.

```
pre-commit install
git config --bool flake8.strict true  # Makes the commit fail if flake8 reports an error
```

To run the hooks:
```
pre-commit run --all-files
```

### Testing

The tests can be executed with:
```
pytest --doctest-modules --cov-report term --cov=lm_datasets
```

## Acknowledgements

The work on the lm-datasets software is partially funded by the [German Federal Ministry for Economic Affairs and Climate Action (BMWK)](https://www.bmwk.de/Navigation/EN/Home/home.html)
through the project [OpenGPT-X](https://opengpt-x.de/en/) (project no. 68GX21007D).

## License

Apache 2.0

*(Please note that the actual datasets are released with different licenses)*

