import prevert
import warnings

from typing import TextIO


class PrevertFile(prevert.dataset):
    """
    Work-around for `prevert` dataset be directly read from file handlers
    """

    def __init__(self, file: TextIO, xml=True):
        self.file = file
        if xml:
            line = self.file.readline()
            if not line.startswith("<corpus"):
                warnings.warn(
                    "Warning: your prevert might not have a XML header. Define the xml second parameter as False in the"
                    " constructor if this is the case."
                )
