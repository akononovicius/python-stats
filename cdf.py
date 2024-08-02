from typing import Optional

import numpy as np


def make_cdf(
    data: list,
    start: Optional[float] = None,
    stop: Optional[float] = None,
    out_points: int = 100,
) -> np.ndarray:
    """Extract empirical CDF on lin-lin scale."""
    if start is None:
        start = np.min(data)
    if stop is None:
        stop = np.max(data)
    bins = np.linspace(start, stop, num=out_points)
    histogram, _ = np.histogram(data, bins=bins, density=False)
    cdf = np.vstack([bins, np.append([0], np.cumsum(histogram))]).T
    cdf[:, 1] /= np.sum(histogram)
    return cdf
