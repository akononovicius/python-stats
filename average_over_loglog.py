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

    def _resample_area(data: np.ndarray, x_poles: np.ndarray) -> np.ndarray:
        areas = np.zeros(len(x_poles))
        data_idx = 0
        for idx, pole in enumerate(x_poles):
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

    min_x = np.min([np.min(arr[:, 0]) for arr in arrs])
    max_x = np.max([np.max(arr[:, 0]) for arr in arrs])
    x_poles = np.logspace(np.log10(min_x), np.log10(max_x), out_points)
    resampled = np.mean(
        np.array([_resample_area(arr, x_poles) for arr in arrs]), axis=0
    )
    x_poles = x_poles * np.sqrt(x_poles[0] / x_poles[1])
    resampled = np.vstack((x_poles, resampled)).T
    return resampled[resampled[:, 1] > 0]
