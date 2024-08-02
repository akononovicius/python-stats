from typing import Optional

import numpy as np

from .histogram import right_histogram


def make_cdf(
    data: np.ndarray,
    start: Optional[float] = None,
    stop: Optional[float] = None,
    out_points: int = 100,
) -> np.ndarray:
    """Extract empirical CDF on lin-lin scale."""
    if start is None:
        start = np.min(data)
    if stop is None:
        stop = np.max(data)
    histogram, edges = right_histogram(
        data, start=start, stop=stop, out_points=out_points, density=False
    )
    cdf = np.vstack([edges, np.cumsum(histogram)]).T
    cdf[:, 1] /= np.sum(histogram)
    return cdf
