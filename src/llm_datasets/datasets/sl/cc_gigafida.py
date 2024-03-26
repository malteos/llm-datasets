from llm_datasets.datasets.base import BaseDataset, Availability, License


class CCGigaFidaDataset(BaseDataset):
    DATASET_ID = "cc_gigafida"
    TITLE = "Written corpus ccGigafida 1.0"
    HOMEPAGE = "https://www.clarin.si/repository/xmlui/handle/11356/1035"
    DESCRIPTION = "Corpus ccGigafida consists of paragraph samples from 31,722 documents, each containing information about the source (e.g. newspapers, magazines), year of publication, text type (fiction, newspaper), the title and author if they are known."
    AVAILIBILITY = Availability.DIRECT_DOWNLOAD
    LICENSE = License(
        "Creative Commons - Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
        url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        commercial_use=False,
        sharealike=True,
        attribution=True,
    )
    CITATION = r"""@misc{11356/1035,
    title = {Written corpus {ccGigafida} 1.0},
    author = {Logar, Nata{\v s}a and Erjavec, Toma{\v z} and Krek, Simon and Gr{\v c}ar, Miha and Holozan, Peter},
    url = {http://hdl.handle.net/11356/1035},
    note = {Slovenian language resource repository {CLARIN}.{SI}},
    copyright = {Creative Commons - Attribution-{NonCommercial}-{ShareAlike} 4.0 International ({CC} {BY}-{NC}-{SA} 4.0)},
    issn = {2820-4042},
    year = {2013} }"""
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
