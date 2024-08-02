from collections import Counter
from typing import Optional, Tuple

import numpy as np


def right_histogram(
    data: np.ndarray,
    start: Optional[float] = None,
    stop: Optional[float] = None,
    out_points: int = 100,
    density: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate right-inclusive histogram.

    Problem: `numpy.histogram` is left-inclusive, which doesn't align well with
    how CDF is defined. Issues arise when there is a degree of discreteness in
    the data.
    """
    if start is None:
        start = np.min(data)
    if stop is None:
        stop = np.max(data)

    _data = np.ceil(
        (
            (data[(start <= data) & (data <= stop)] - start)
            * (out_points - 1)
            / (stop - start)
        )
    ).astype(int)

    counts = Counter(_data)
    hist = np.zeros(out_points)
    for key, val in counts.items():
        hist[key] = val
    if density:
        n_samples = len(_data)
        hist = hist / n_samples
    edges = np.linspace(start, stop, num=out_points)
    return hist, edges
