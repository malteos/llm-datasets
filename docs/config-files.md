# Config Files

`lm-datasets` allows you to specific general settings through config files so you do not need to specific always the same command line arguments.
Several commands support passing the `--configs` argument which should point to one or more YAML-files on your file system. For example, the text extraction command:

```bash
lm_datasets extract_text ... --configs $PATH_TO_YAML_CONFIG_FILE
```

## Specifing local paths

In the config files, you can store for example system specific settings like the local paths, where the raw dataset files are located:

```yaml
# ./examples/lm_datasets_configs/my_system.yaml
local_dirs_by_source_id:
  redpajama: /my_system_specific_data_directory/redpajama
```

The [RedPajama dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) requires the manual download prior to the text extraction.
With the above config, we tell the extraction command the path where we downloaded the RedPajama data by providing the config file:

```bash
lm_datasets extract_text redpajama_book --configs ./examples/lm_datasets_configs/my_system.yaml
```

## Dataset selection and sampling

The configuration files are also needed for specifying the final dataset composition, including the selection of the datasets and their sampling.
The following examples shows a config for an Italian dataset:

```yaml
# ./examples/lm_datasets_configs/italian_data.yaml

# a fixed random seed for shuffling etc.
seed: 0

selected_dataset_ids:
  # italian subsets
  - itwac
  - eurlex_it
  - wikipedia_20231101_it
  - wikibooks_it
  - wikinews_it
  - colossal_oscar_2023-23_it
  - parlamint_it

# down-sample webcrawled + up-sampled high quality
sampling_factor_by_source_id:
  colossal_oscar: 0.1

sampling_factor_by_dataset_id:
  itwac: 0.5
  eurlex_it: 2
  wikipedia_20231101_it: 3
```

To use this config, provide the path in the `--configs` argument:

```bash
# compose final dataset
lm_datasets compose ... --configs ./examples/lm_datasets_configs/italian_data.yaml

```