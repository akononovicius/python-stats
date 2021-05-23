import matplotlib.pyplot as plt
import numpy as np


def __rescaled_length(series, offset, scale):
    n_full = len(series)
    diffs = np.diff(series[offset::scale])
    n_rescaled = len(diffs) + 1
    sum_term = np.sum(np.abs(diffs))
    return (n_full - 1) / (n_rescaled * (scale ** 2)) * sum_term


def __mean_length(series, scale):
    return np.mean(
        [__rescaled_length(series, offset, scale) for offset in np.arange(0, scale)]
    )


def higuchi_dimension(
    series: list[float], scales: list[int], plot_curves: bool = False
):
    """Calculate Higuchi dimension.

    Args:
        series      - 1D list of real numbers, whose Higuchi dimension needs to
                      be estimated.
        scales      - 1D list of integer numbers describing the scales at which
                      the series ought to be analysed.
        plot_curves - whether to plot the length vs 1/scale plot. (optional)

    Returns:
        An estimate of Higuchi dimension. Optionally one can choose to plot
        the curves, from which the Higuchi dimension is estimated.
    """
    log_lengths = np.log([__mean_length(series, scale) for scale in scales])
    log_scales = -np.log(scales)

    coeffs = np.polyfit(log_scales, log_lengths, 1)

    if plot_curves:
        plt.figure()
        plt.xlabel("log(1/scale)")
        plt.ylabel("log(length)")
        plt.plot(log_scales, log_lengths, "ro")
        plt.plot(log_scales, np.polyval(coeffs, log_scales), "k-")
        plt.show()

    return coeffs[0]
