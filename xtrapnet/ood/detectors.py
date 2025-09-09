from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import KDTree


class BaseDetector:
    def fit(self, features: np.ndarray):
        raise NotImplementedError

    def score(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NullDetector(BaseDetector):
    def fit(self, features: np.ndarray):
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return np.zeros((features.shape[0],), dtype=np.float32)


class MahalanobisDetector(BaseDetector):
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray):
        self.mean_ = features.mean(axis=0, keepdims=True)
        cov = np.cov(features.T) + 1e-6 * np.eye(features.shape[1])
        self.cov_inv_ = np.linalg.inv(cov)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.cov_inv_ is None:
            raise RuntimeError("Detector not fit")
        diff = features - self.mean_
        return np.einsum('ij,jk,ik->i', diff, self.cov_inv_, diff)


class KNNDetector(BaseDetector):
    def __init__(self, k: int = 5):
        self.k = k
        self.tree: Optional[KDTree] = None

    def fit(self, features: np.ndarray):
        self.tree = KDTree(features)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.tree is None:
            raise RuntimeError("Detector not fit")
        dists, _ = self.tree.query(features, k=min(self.k, max(1, getattr(self.tree, 'n', 1))))
        if dists.ndim == 1:
            return dists
        return dists.mean(axis=1)


