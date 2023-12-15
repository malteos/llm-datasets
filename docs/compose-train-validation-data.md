# Compose dataset

The pipeline step that produces the final training or validation set is the `compose` step.
Before you run this command, you should specify in the [config](config-files.md) files what datasets should be selected and how they should be sampled.

```bash
lm-datasets compose –-split=train –-configs=my_dataset.yaml \
	--text_data_dir=/data/my_text_data \
	--composed_data_dir=/data/my_composed_data/train/
```

Depending on the your system (especially IO-speed) and dataset size this step can take a substantial amount of time (> 24 hours for a 1T token dataset).
