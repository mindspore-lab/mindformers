"""Utils of layers"""
import collections.abc
from itertools import repeat


def _ntuple(n):
    """Tuple for num."""
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
