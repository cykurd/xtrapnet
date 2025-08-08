from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..config import PolicyConfig


class GuardrailPolicy:
    def __init__(self, config: PolicyConfig):
        self.config = config

    def decide(self, mean: np.ndarray, var: Optional[np.ndarray], ood_scores: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        actions = []
        rationale = []

        unc_thresh = None
        if var is not None:
            unc_thresh = np.quantile(var.flatten(), self.config.abstain_uncertainty_quantile)
        ood_thresh = None
        if ood_scores is not None:
            ood_thresh = np.quantile(ood_scores.flatten(), self.config.abstain_ood_quantile)

        for i in range(mean.shape[0]):
            abstain = False
            reasons = []
            if var is not None and unc_thresh is not None and var.flatten()[i] >= unc_thresh:
                abstain = True
                reasons.append("high_uncertainty")
            if ood_scores is not None and ood_thresh is not None and ood_scores.flatten()[i] >= ood_thresh:
                abstain = True
                reasons.append("ood_risk")
            actions.append("abstain" if abstain else "predict")
            rationale.append(",".join(reasons))

        return {"action": np.array(actions), "rationale": np.array(rationale)}


