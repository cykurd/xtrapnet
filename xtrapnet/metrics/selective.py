from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def risk_coverage_curve(mean: np.ndarray, labels: np.ndarray, variance: Optional[np.ndarray]) -> Dict[str, float]:
    if variance is None:
        errors = np.abs(mean - labels)
        return {"risk_at_full_coverage": float(np.mean(errors))}

    uncertainty = variance.flatten()
    order = np.argsort(uncertainty)
    sorted_errors = np.abs(mean.flatten()[order] - labels.flatten()[order])

    coverages = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    metrics: Dict[str, float] = {}
    n = len(sorted_errors)
    for c in coverages:
        k = max(1, int(n * c))
        risk = float(np.mean(sorted_errors[:k]))
        metrics[f"risk_at_{int(c*100)}pct_coverage"] = risk
    return metrics


