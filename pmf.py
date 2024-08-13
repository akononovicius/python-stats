from collections import Counter
from typing import Optional

import numpy as np


def make_pmf(
    data: list,
    start: Optional[float] = None,
    stop: Optional[float] = None,
) -> np.ndarray:
    """Extract empirical probability mass function from data."""
    _data = np.array(data)

    if start is None:
        start = np.min(data)
    if stop is None:
        stop = np.max(data)
    _data = _data[(start <= _data) & (_data <= stop)]

    counter_dict = Counter(_data)
    pmf = np.array(
        [list(counter_dict.keys()), list(counter_dict.values())], dtype=float
    ).T
    pmf = pmf[np.argsort(pmf[:, 0])]
    pmf[:, 1] = pmf[:, 1] / len(_data)
    return pmf
