# Integrate a custom dataset

- Write a dataset class: `my_datasets/pg19.py`
- Register new dataset classes: `my_datasets/dataset_registry.py`

## Load registry in commands

To load the registerd datasets in the pipeline commands, you need to specify the `--extra_dataset_registries` argument:

```bash
llm-datasets compose ... -extra_dataset_registries=my_datasets.dataset_registry
```