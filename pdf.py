from typing import Optional

import numpy as np


def __make_pdf(
    data: list,
    bin_boundaries: list,
) -> np.ndarray:
    # We usually study distributions, which have single region of support.
    # In other words, most distributions we study should have non-zero
    # density in (start, stop) interval.
    #
    # For this reason we set bins manually, calculate histogram with
    # density=False setting, delete empty bins, and finally calculate the
    # density manually by dividing counts by the width of non-empty bins.
    histogram = np.histogram(data, bins=bin_boundaries, density=False)[0]
    empty_bin_pos = np.where(histogram == 0)
    # For simplicity sake, the bin to the left of non-empty bin is extended.
    histogram = np.delete(histogram, empty_bin_pos)
    bin_boundaries = np.delete(bin_boundaries, empty_bin_pos)
    bin_widths = np.diff(bin_boundaries)
    density = histogram / np.sum(histogram) / bin_widths

    # The reported bin location is centered between the boundaries
    centroids = 0.5 * (bin_boundaries[1:] + bin_boundaries[:-1])

    pdf = np.array([centroids, density]).T
    return pdf


def make_pdf(
    data: list,
    start: Optional[float] = None,
    stop: Optional[float] = None,
    out_points: int = 100,
) -> np.ndarray:
    """Extract empirical PDF on lin-lin scale."""
    if start is None:
        start = np.min(data)
    if stop is None:
        stop = np.max(data)

    bin_boundaries = np.linspace(start, stop, num=out_points)

    return __make_pdf(data, bin_boundaries)


def make_log_pdf(
    data: list,
    start: Optional[float] = None,
    stop: Optional[float] = None,
    out_points: int = 100,
) -> np.ndarray:
    """Extract empirical PDF on log-log scale."""
    if start is None:
        _start = np.min(data)
    else:
        _start = start
    _start = np.log10(_start)
    if stop is None:
        _stop = np.max(data)
    else:
        _stop = stop
    _stop = np.log10(_stop)

    bin_boundaries = np.logspace(_start, _stop, num=out_points)

    return __make_pdf(data, bin_boundaries)


def estimate_cdf_from_pdf(pdf: list) -> np.ndarray:
    """Approximate empirical CDF from empirical PDF."""
    bin_widths = np.diff(pdf[:, 0])
    left_pdf_sum = np.cumsum(pdf[1:, 1] * bin_widths)
    right_pdf_sum = np.cumsum(pdf[:-1, 1] * bin_widths)

    cdf = 0.5 * left_pdf_sum + 0.5 * right_pdf_sum
    cdf = cdf / cdf[-1]
    cdf = np.concatenate(([0], cdf))

    return np.transpose([pdf[:, 0], cdf])
