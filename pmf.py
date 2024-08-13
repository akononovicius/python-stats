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
    
    mask_flag = False
    if start is None:
        start = np.min(data)
        mask_flag = True
    if stop is None:
        stop = np.max(data)
        mask_flag = True
    if mask_flag:
        _data = _data[(start <= _data) & (_data <= stop)]
    
    counter_dict = Counter(_data)
    pmf = np.array([list(counter_dict.keys()), list(counter_dict.values())]).T
    pmf = pmf[np.argsort(pmf[:, 0])]
    pmf[:, 1] = pmf[:, 1] / len(_data)
    return pmf
