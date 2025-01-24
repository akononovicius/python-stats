from typing import Optional, Tuple

import numpy as np
import scipy.signal as sp  # type: ignore

from .average_over_loglog import average_over_loglog


def make_equilog_psd(
    times: list,
    vals: list,
    min_log_freq: Optional[float] = None,
    max_log_freq: Optional[float] = None,
    out_points: int = 100,
) -> np.ndarray:
    """Calculate equi-log-sampled PSD from non-equi-sampled data.

    Input:
        times:
            Times when the observations stored in `vals` were made.
        vals:
            Values observed at times stored in `times`.
        min_log_freq:
            Set the lower PSD frequency bound. Pass log10 of the
            actual frequency.
        max_log_freq:
            Set the upper PSD frequency bound. Pass log10 of the
            actual frequency.
        out_points:
            Desired number of points in the output PSD.

    Output:
        Two dimensional ndarray. Firt column - frequencies,
        the second column - estimated PSD at those frequencies.
    """
    if max_log_freq is None:
        max_log_freq = np.log10(0.5 / np.mean(np.diff(times)))
    if min_log_freq is None:
        min_log_freq = -np.log10(times[-1] - times[0])
    freqs = np.logspace(min_log_freq, max_log_freq, out_points)
    norm = len(times) / 4
    psd = sp.lombscargle(times, vals, 2 * np.pi * freqs) / norm
    return np.vstack([freqs, psd]).T


def make_log_psd(series: list, fs: float = 1.0, out_points: int = 100) -> np.ndarray:
    """Estimate log-sampled PSD from equi-sampled data.

    Input:
        series:
            Equi-sampled data.
        fs:
            Sampling frequency of the data.
        out_points:
            Desired number of points in the output PSD.
            Lower resolution of the PSD is obtained by
            averaging over binned FFT output.

    Output:
        Two dimensional ndarray. Firt column - frequencies,
        the second column - estimated PSD at those frequencies.
    """
    psd = np.array(sp.periodogram(series, fs=fs)).T
    ids = np.unique(np.logspace(0, np.log10(psd.shape[0]), out_points).astype(int)) - 1
    ids = np.vstack((ids[1:-1], ids[2:])).T
    return np.array(
        [[(psd[i[0], 0] + psd[i[1], 0]) / 2, np.mean(psd[i[0] : i[1], 1])] for i in ids]
    )


def make_seg_log_psd(
    series: list,
    fs: float = 1.0,
    out_points: int = 100,
    segment_len: int = 262144,
    bi_directional_split: bool = True,
) -> np.ndarray:
    """Estimate log-sampled PSD from segmented equi-sampled data.

    Input:
        series:
            Equi-sampled data.
        fs:
            Sampling frequency of the data.
        out_points:
            Desired number of points in the output PSD.
            Lower resolution of the PSD is obtained by
            averaging over binned FFT output.
        segment_len:
            Length of a segment. Data will be split into
            multiple segments of this length. PSD will
            be estimated for each segment, and then it
            will be averaged over the segments.
        bi_directional_split:
            Whether the splits should done both from the
            start and from the end. If segmentation of the
            data is not perfect, it might be wise to obtain
            segments starting both from the start and from
            the end of the series. Value of this parameter
            will be ignored, if the split is perfect.

    Output:
        Two dimensional ndarray. Firt column - frequencies,
        the second column - estimated PSD at those frequencies.
    """

    def _to_pow_2(num: int) -> int:
        # (num & (num - 1)) != 0 check if num is already power of 2
        if num > 0 and (num & (num - 1)) != 0:
            num = 2 ** np.ceil(np.log2(num))
        return num

    series_len = len(series)
    segment_len = _to_pow_2(segment_len)
    if series_len < segment_len:
        segment_len = int(_to_pow_2(series_len) / 2)
    n_splits = int(np.floor(series_len / segment_len))

    psds: Tuple = ()
    # do spliting from the start of the series
    for i in range(n_splits):
        start = i * segment_len
        psd = make_log_psd(
            series[start : start + segment_len], fs=fs, out_points=10 * out_points
        )
        psds = psds + (psd,)

    if bi_directional_split and (n_splits * segment_len < series_len):
        # do spliting from the end of the series
        for i in range(n_splits):
            start = series_len - i * segment_len
            psd = make_log_psd(
                series[start - segment_len : start], fs=fs, out_points=10 * out_points
            )
            psds = psds + (psd,)

    return average_over_loglog(psds, out_points=out_points)
