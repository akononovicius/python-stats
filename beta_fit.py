from typing import Optional, Tuple

import numpy as np


def naive_beta_binomial_fit(
    data: list, n: Optional[int] = None
) -> Tuple[float, float, int]:
    """Estimate Beta-binomial distribution parameters from the mean and variance of the data.

    Input:
        data:
            List (array) containing data values. Logic dictates that it should
            contain positive integer values, as otherwise fitting Beta-binomial
            distribution makes no sense.
        n: (default: None)
            N parameter of the Beta-binomial distribution can be fixed by
            passing this optional value. If `None` is passed (which is the
            default), then the maximum of the `data` will be used as the
            estimate for N parameter.

    Output:
        alpha, beta and N parameter estimates.
    """
    if n is None:
        n = np.max(data)

    mean = np.mean(data)
    variance = np.var(data, ddof=1)

    # these formulas were obtained by inverting the expressions for mean and
    # variance of the Beta-binomial distribution (e.g., see
    # https://en.wikipedia.org/wiki/Beta-binomial_distribution)
    alpha_par = ((mean**2) * n - mean**3 - mean * variance) / (
        mean**2 - mean * n + n * variance
    )
    beta_par = alpha_par * (n / mean - 1)

    return alpha_par, beta_par, n


def naive_beta_fit(data: list) -> Tuple[float, float]:
    """Estimate Beta distribution parameters from the mean and variance of the data.

    Input:
        data:
            List (array) containing data values. Logic dictates that it should
            contain values in (0, 1) range, as otherwise fitting Beta
            distribution might result in nonsensical fits.

    Output:
        alpha, beta parameter estimates.
    """
    mean = np.mean(data)
    variance = np.var(data, ddof=1)

    # these formulas were obtained by inverting the expressions for mean and
    # variance of the Beta-binomial distribution (e.g., see
    # https://en.wikipedia.org/wiki/Beta_distribution)
    alpha_par = ((mean**2) - mean**3 - mean * variance) / variance
    beta_par = alpha_par * (1 / mean - 1)

    return alpha_par, beta_par
