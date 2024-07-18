import json
import os

import tlsh

from tests.conftest import FIXTURES_DIR


def test_tlsh_hashes_against_oscar():
    """Use the exact same hashing from OSCAR 23.01

    https://oscar-project.github.io/documentation/versions/oscar-2301/#locality-sensitive-hashing

    """
    with open(os.path.join(FIXTURES_DIR, "oscar_2301_texts_and_hashes.json")) as f:
        texts_and_hashes = json.load(f)

    for text, hash in zip(texts_and_hashes["text"], texts_and_hashes["tlsh"]):
        if hash is None or text is None:
            continue

        given_hash = hash[5:]
        computed_hash = tlsh.hash(text.encode("utf-8"))
        assert given_hash == computed_hash


# texts_and_hashes = [
#     "Kategorie 5 kabel, (ook: Cat5 / Cat5e), is 'n gedraaide draadpaarkabel wat ontwerp is om 'n hoë mate van"
#     " integriteit in elektriese seine te verseker. Baie sulke kabels is onafgeskerm, maar kan ook soms met 'n"
#     " afskermingskede voorsien word. Die Kategorie 5-spesifikasie is vervang met die Kategorie 5e spesifikasie. Hierdie"
#     " soort kabel word dikwels gebruik in kabels vir rekenaarnetwerke soos Ethernet en word ook gebruik om baie ander"
#     " seine te dra soos vir basiese telefoniese dienste."
# ]

# # init dataset in streaming mode
# ds = load_dataset("malteos/tmpdata2", name="de", split="2023_23", streaming=True)
# ds_iter = iter(ds)


# for i in range(100):
#     print(i)
#     row = next(ds_iter)
#     if row["tlsh"] is not None:


# print(row)
# print(row["tlsh"])


# print("done")


if __name__ == "__main__":
    test_tlsh_hashes_against_oscar()
