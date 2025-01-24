from typing import Tuple

import numpy as np


def average_over_loglog(arrs: Tuple[np.ndarray], out_points: int = 100) -> np.ndarray:
    """Average arrays with log-log data."""

    def _get_bin_width(data: np.ndarray, idx: int) -> float:
        # Bin width is estimated to be half of the
        # interval between surrounding points.
        before = data[idx - 1, 0] if (idx > 0) else 0.0
        next = data[idx + 1, 0] if (idx < len(data) - 1) else data[idx, 0]
        return (next - before) / 2.0

    def _resample_area(data: np.ndarray, poles: np.ndarray) -> np.ndarray:
        areas = np.zeros(len(poles))
        data_idx = 0
        for idx, pole in enumerate(poles):
            integral = 0.0
            total_width = 0.0
            while data_idx < len(data) and data[data_idx, 0] < pole:
                width = _get_bin_width(data, data_idx)
                total_width = total_width + width
                integral = integral + data[data_idx, 1] * width
                data_idx = data_idx + 1
            if total_width > 0:
                areas[idx] = integral / total_width
        return areas

    min_value = np.min([np.min(arr[:, 0]) for arr in arrs])
    max_value = np.max([np.max(arr[:, 0]) for arr in arrs])
    poles = np.logspace(np.log10(min_value), np.log10(max_value), out_points)
    resampled = np.mean(np.array([_resample_area(arr, poles) for arr in arrs]), axis=0)
    poles = poles * np.sqrt(poles[0] / poles[1])
    resampled = np.vstack((poles, resampled)).T
    return resampled[resampled[:, 1] > 0]
