from lm_datasets.datasets.base import BaseDataset, Availability, License


class CCGigaFidaDataset(BaseDataset):
    DATASET_ID = "cc_gigafida"
    TITLE = "Written corpus ccGigafida 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1035"
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    LICENSE = License(
        "Creative Commons - Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
        url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        commercial_use=False,
        sharealike=True,
        attribution=True,
    )
    LANGUAGES = ["sl"]

    DOWNLOAD_URLS = ["https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1035/ccGigafida-text.zip"]

    TOKENS = 126919097

    def decompress(self):
        # decompress single zip file
        pass

    def get_texts(self):
        # TODO read directly from zip

        file_paths = self.get_dataset_file_paths(needed_suffix=".txt")
        for fp in file_paths:
            with open(fp) as f:
                text = f.read()

                yield text
