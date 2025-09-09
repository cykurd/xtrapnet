from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class ConformalCalibrator:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile_: Optional[float] = None

    def fit(self, predictor, features: np.ndarray, labels: np.ndarray):
        n = features.shape[0]
        if n < 10:
            self.quantile_ = 0.0
            return self
        calib_frac = max(0.1, min(0.2, 200 / max(1, n)))
        split_idx = int(n * (1 - calib_frac))
        calib_X = features[split_idx:]
        calib_y = labels[split_idx:]
        pred = predictor.predict(calib_X)
        if isinstance(pred, tuple):
            pred = pred[0]
        residuals = np.abs(calib_y.flatten() - pred.flatten())
        q = np.quantile(residuals, 1 - self.alpha)
        self.quantile_ = float(q)
        return self

    def predict_intervals(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.quantile_ is None:
            raise RuntimeError("ConformalCalibrator not fit")
        q = self.quantile_
        n = features.shape[0]
        zeros = np.zeros((n, 1), dtype=np.float32)
        lower = zeros - q
        upper = zeros + q
        return lower, upper


