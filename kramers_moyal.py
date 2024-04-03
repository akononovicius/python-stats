import numpy as np
from numpy.typing import ArrayLike
from scipy.special import factorial


def get_km(order: int, n_bins: int, series: ArrayLike, delta_t: float) -> np.ndarray:
    """Calculate Kramers-Moyal term.

    Input:
        order:
            Which order term to calculate.
        n_bins:
            Number of bins to use when estimating Kramers-Moyal term. Bins are
            used to aggregate initial conditions into manageable ranges.
        series:
            One dimensional array containing time series values at fixed time
            intervals.
        delta_t:
            Sampling period of the time series given as `series` input
            variable.

    Output:
        Two dimensional numpy array. First column contains initial values
        (binned), second column contains respective value of a Kramers-Moyal
        coefficient.
    """
    bins_start = np.min(series)
    bins_end = np.max(series)

    binned_series = series.copy()
    binned_series = (binned_series - bins_start) / (bins_end - bins_start)
    binned_series = np.round(binned_series * n_bins).astype(int)

    initial_bin = np.arange(0, n_bins + 1).astype(int)
    coeff = np.zeros(initial_bin.shape)
    mask_next = np.empty_like(binned_series, dtype=bool)
    # first value will never be x(t+delta_t) candidate
    mask_next[0] = False
    for idx, x in enumerate(initial_bin):
        # pick out when x(t) falls inside the considered bin
        mask_initial = binned_series == x
        # we do not care about last value, because we do not know what follows it
        mask_initial[-1] = False
        if np.any(mask_initial):
            # pick proper x(t+delta_t)
            mask_next[1:] = mask_initial[:-1]
            coeff[idx] = np.mean((series[mask_next] - series[mask_initial]) ** order)
    coeff = coeff / (factorial(order) * delta_t)
    x_0 = (bins_end - bins_start) * initial_bin / n_bins + bins_start

    return np.vstack((x_0, coeff)).T
