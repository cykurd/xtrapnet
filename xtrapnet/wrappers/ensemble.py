from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch


class EnsembleWrapper:
    def __init__(self, base_model_constructor: Callable[[], torch.nn.Module], num_members: int = 5, diversity: str = "seed"):
        self.members: List[torch.nn.Module] = [base_model_constructor() for _ in range(num_members)]
        self.diversity = diversity

    def parameters(self):
        for m in self.members:
            yield from m.parameters()

    def train(self):
        for m in self.members:
            m.train()

    def eval(self):
        for m in self.members:
            m.eval()

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        outputs = []
        for m in self.members:
            outputs.append(m(features))
        return torch.stack(outputs, dim=0).mean(dim=0)

    def predict(self, features: np.ndarray, mc_dropout: bool = False, n_samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        for m in self.members:
            out = m.predict(features, mc_dropout=mc_dropout, n_samples=n_samples)
            if isinstance(out, tuple):
                mean, _ = out
            else:
                mean = out
            preds.append(mean)
        stacked = np.stack(preds, axis=0)
        return stacked.mean(axis=0), stacked.var(axis=0)


