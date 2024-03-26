import re
import tempfile

import sys

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

from llm_datasets.utils.dataframe import get_datasets_as_dataframe

from llm_datasets.datasets.base import BaseDataset
from llm_datasets.utils.config import Config

from llm_datasets.utils.docs import TokensColumn
from llm_datasets.viewer.viewer_utils import millify

from pathlib import Path
import matplotlib.ticker as ticker

import itertools
import logging
import locale
import math
from numbers import Integral

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms


class CustomEngFormatter(ticker.EngFormatter):
    """
    Format axis values using engineering prefixes to represent powers
    of 1000, plus a specified unit, e.g., 10 MHz instead of 1e7.
    """

    # The SI engineering prefixes
    ENG_PREFIXES = {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "\N{MICRO SIGN}",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "B",  # changed from G to B
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
    }

    def __init__(self, unit="", places=None, sep=" ", *, usetex=None, useMathText=None):
        r"""
        Parameters
        ----------
        unit : str, default: ""
            Unit symbol to use, suitable for use with single-letter
            representations of powers of 1000. For example, 'Hz' or 'm'.

        places : int, default: None
            Precision with which to display the number, specified in
            digits after the decimal point (there will be between one
            and three digits before the decimal point). If it is None,
            the formatting falls back to the floating point format '%g',
            which displays up to 6 *significant* digits, i.e. the equivalent
            value for *places* varies between 0 and 5 (inclusive).

        sep : str, default: " "
            Separator used between the value and the prefix/unit. For
            example, one get '3.14 mV' if ``sep`` is " " (default) and
            '3.14mV' if ``sep`` is "". Besides the default behavior, some
            other useful options may be:

            * ``sep=""`` to append directly the prefix/unit to the value;
            * ``sep="\N{THIN SPACE}"`` (``U+2009``);
            * ``sep="\N{NARROW NO-BREAK SPACE}"`` (``U+202F``);
            * ``sep="\N{NO-BREAK SPACE}"`` (``U+00A0``).

        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            numbers in the formatter.

        useMathText : bool, default: :rc:`axes.formatter.use_mathtext`
            To enable/disable the use mathtext for rendering the numbers in
            the formatter.
        """
        self.unit = unit
        self.places = places
        self.sep = sep
        self.set_usetex(usetex)
        self.set_useMathText(useMathText)

    def get_usetex(self):
        return self._usetex

    def set_usetex(self, val):
        if val is None:
            self._usetex = mpl.rcParams["text.usetex"]
        else:
            self._usetex = val

    usetex = property(fget=get_usetex, fset=set_usetex)

    def get_useMathText(self):
        return self._useMathText

    def set_useMathText(self, val):
        if val is None:
            self._useMathText = mpl.rcParams["axes.formatter.use_mathtext"]
        else:
            self._useMathText = val

    useMathText = property(fget=get_useMathText, fset=set_useMathText)

    def __call__(self, x, pos=None):
        s = "%s%s" % (self.format_eng(x), self.unit)
        # Remove the trailing separator when there is neither prefix nor unit
        if self.sep and s.endswith(self.sep):
            s = s[: -len(self.sep)]
        return self.fix_minus(s)

    def format_eng(self, num):
        """
        Format a number in engineering notation, appending a letter
        representing the power of 1000 of the original number.
        Some examples:

        >>> format_eng(0)        # for self.places = 0
        '0'

        >>> format_eng(1000000)  # for self.places = 1
        '1.0 M'

        >>> format_eng(-1e-6)  # for self.places = 2
        '-1.00 \N{MICRO SIGN}'
        """
        sign = 1
        fmt = "g" if self.places is None else ".{:d}f".format(self.places)

        if num < 0:
            sign = -1
            num = -num

        if num != 0:
            pow10 = int(math.floor(math.log10(num) / 3) * 3)
        else:
            pow10 = 0
            # Force num to zero, to avoid inconsistencies like
            # format_eng(-0) = "0" and format_eng(0.0) = "0"
            # but format_eng(-0.0) = "-0.0"
            num = 0.0

        pow10 = np.clip(pow10, min(self.ENG_PREFIXES), max(self.ENG_PREFIXES))

        mant = sign * num / (10.0**pow10)
        # Taking care of the cases like 999.9..., which may be rounded to 1000
        # instead of 1 k.  Beware of the corner case of values that are beyond
        # the range of SI prefixes (i.e. > 'Y').
        if abs(float(format(mant, fmt))) >= 1000 and pow10 < max(self.ENG_PREFIXES):
            mant /= 1000
            pow10 += 3

        prefix = self.ENG_PREFIXES[int(pow10)]
        if self._usetex or self._useMathText:
            formatted = "${mant:{fmt}}${sep}{prefix}".format(mant=mant, sep=self.sep, prefix=prefix, fmt=fmt)
        else:
            formatted = "{mant:{fmt}}{sep}{prefix}".format(mant=mant, sep=self.sep, prefix=prefix, fmt=fmt)

        return formatted


def plot_tokens_by_language(
    df,
    tokens_col: TokensColumn,
    save_to_path=None,
    save_format="pdf",
    figsize=(14, 3),
    language_col="language",
    top_k=0,
):
    xlabel = "Languages"

    if top_k > 0:
        df = df.head(top_k)
        xlabel += f" (top {top_k})"

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(ax=ax, data=df.reset_index(), x=language_col, y=tokens_col)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(CustomEngFormatter())

    ax.set(xlabel=xlabel, ylabel="Tokens (log scale)")

    plt.xticks(rotation=45)

    if save_to_path:
        plt.savefig(save_to_path, format=save_format, bbox_inches="tight")

    plt.show()

    return fig, ax


def plot_tokens_by_source(
    df,
    tokens_col: TokensColumn,
    save_to_path=None,
    save_format="pdf",
    figsize=(14, 3),
    source_col="source_id",
    top_k=15,
    xticks_rotate=90,  # 45 for latex
):
    if top_k > 0:
        df = df.head(n=top_k)

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(ax=ax, data=df.reset_index(), x=source_col, y=tokens_col)

    ax.set(xlabel=f"Data source (top {top_k})", ylabel="Tokens (log scale)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(CustomEngFormatter())
    plt.xticks(rotation=xticks_rotate)

    if save_to_path:
        plt.savefig(save_to_path, format=save_format, bbox_inches="tight")
    plt.show()

    return fig, ax
