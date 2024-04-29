from datatrove.pipeline.readers.base import BaseReader


from typing import Callable, List

from loguru import logger

from datatrove.data import DocumentsPipeline

from llm_datasets.datasets.base import BaseDataset
from llm_datasets.datasets.dataset_registry import get_dataset_class_by_id, get_datasets_list_from_string
from llm_datasets.utils import get_auto_workers
from llm_datasets.utils.config import Config


class LLMDatasetsDatatroveReader(BaseReader):
    """
    A datatrove-compatible reader for integrated datasets
    """

    name = "🦜 LLM-Datasets"

    def __init__(
        self,
        dataset_ids: List[str] | str,
        config: Config,
        limit: int = -1,
        progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
    ):
        super().__init__(limit, progress, adapter, text_key, id_key, default_metadata)

        # Build list of datasets
        self.datasets_list = get_datasets_list_from_string(dataset_ids, config)
        self.config = config

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """
        Will get this rank's shard and sequentially read each file in the shard, yielding Document.
        Args:
            data: any existing data from previous pipeline stages
            rank: rank of the current task
            world_size: total number of tasks

        Returns:

        """
        if data:
            yield from data

        config = self.config
        li = 0
        # Iterate over datasets
        for i, dataset_id in enumerate(self.datasets_list, 1):
            logger.info(f"Dataset ID: {dataset_id} ({i} / {len(self.datasets_list)})")
            dataset_cls = get_dataset_class_by_id(dataset_id, config.extra_dataset_registries)
            dataset: BaseDataset = dataset_cls(
                raw_datasets_dir=config.raw_datasets_dir,
                # workers=get_auto_workers(config.workers),
                # skip_items=config.skip_items,
                # min_length=config.min_text_length,
                config=config,
                **config.get_extra_dataset_kwargs(dataset_id),
            )
            # Yield documents from each dataset
            for doc in dataset.get_documents():
                if doc is None:
                    # skip empty docs
                    continue

                self.update_doc_stats(doc)
                yield doc

                li += 1

                if self.limit > 0 and li >= self.limit:
                    # doc level break
                    break

            # log stats
            for k, v in dataset.counter.items():
                logger.info("sttats {k}")
                self.stat_update(f"{dataset_id}.{k}", value=v, unit="documents")

            if self.limit > 0 and li >= self.limit:
                # dataset level break
                break

        logger.info("Reader completed")
