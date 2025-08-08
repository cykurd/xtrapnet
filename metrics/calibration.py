from __future__ import annotations

import numpy as np


def expected_calibration_error(mean: np.ndarray, labels: np.ndarray, variance: np.ndarray, num_bins: int = 10) -> float:
    std = np.sqrt(np.maximum(variance, 1e-12))
    confidence = 1.0 / (std + 1e-8)
    bins = np.quantile(confidence, np.linspace(0, 1, num_bins + 1))
    ece = 0.0
    for i in range(num_bins):
        mask = (confidence >= bins[i]) & (confidence <= bins[i + 1])
        if not np.any(mask):
            continue
        bin_error = np.mean(np.abs(mean[mask] - labels[mask]))
        bin_conf = np.mean(confidence[mask])
        ece += np.abs(bin_error - 1.0 / (bin_conf + 1e-8)) * np.mean(mask)
    return float(ece)


