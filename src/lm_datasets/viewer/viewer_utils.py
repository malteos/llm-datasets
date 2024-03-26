import math

millnames = ["", " k", " M", " B", " T"]


def millify(n, na_string="N/A") -> str:
    n = float(n)
    if math.isnan(n):
        return na_string

    millidx = max(0, min(len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

    if n <= 0:
        return na_string
    else:
        return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
