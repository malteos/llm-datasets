import json
from pathlib import Path
from lm_datasets.datasets.base import MB, BaseDataset, Availability, License

import logging

logger = logging.getLogger(__name__)


class BulgarianNewsDataset(BaseDataset):
    DATASET_ID = "bulgarian_news"
    TITLE = "Crawl of Bulgarian news websites"
    DOWNLOAD_URLS = ["http://old.dcl.bas.bg/dataset/Bulgarian_news.7z"]
    DESCRIPTION = (
        "The collection was collected by crawling Bulgarian websites in Bulgarian. Text samples are in json format. We"
        " can provide raw tests."
    )
    WEB_CRAWLED = True
    LANGUAGES = ["bg"]
    BYTES = 919 * MB
    AVAILIBILITY = Availability.ON_REQUEST
    LICENSE = License("research only")

    def decompress(self):
        # 7z x Bulgarian_news.7z
        pass

    def get_texts(self):
        # read from extracted JSON files
        for i, file_path in enumerate(Path(self.get_local_dataset_dir()).rglob("*.json")):
            if self.skip_items > 0 and i < self.skip_items:
                continue

            with open(file_path) as f:
                try:
                    doc = json.load(f)
                    if "bg_a_text" in doc:
                        text = self.paragraph_delimiter.join(doc["bg_a_text"])
                        yield text
                    else:
                        logger.warning("JSON has no text field: %s", file_path)

                except ValueError:
                    logger.error("Cannot parse JSON from %s", file_path)
