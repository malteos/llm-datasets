# Integration with other frameworks

LLM-Datasets can be used in combination with our own processing pipelines or integration in other frameworks, for example with [Huggingface's DataTrove](https://github.com/huggingface/datatrove).

## DataTrove integration

HuggingFace's DataTrove is a library to process, filter and deduplicate text data at a very large scale.
All datasets implemented within LLM-Dataset can be processed with DataTrove.
To do so, you can use the `LLMDatasetsDatatroveReader` class as input for any DataTrove pipeline.
The `LLMDatasetsDatatroveReader` class takes a list of dataset ID(s) and/or [config files](config-files.md) as arguments, as shown in the example below:

```python
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import JsonlWriter

from llm_datasets.datatrove_reader import LLMDatasetsDatatroveReader
from llm_datasets.utils.config import Config, get_config_from_paths

llmds_config: Config = get_config_from_paths(["path/to/my/config.yaml"])

pipeline = [
    LLMDatasetsDatatroveReader("legal_mc4_en", llmds_config),
    SamplerFilter(rate=0.5),
    JsonlWriter(
        output_folder="/my/output/path"
    )
]
```

