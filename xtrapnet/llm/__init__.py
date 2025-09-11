"""
LLM-assisted extrapolation for XtrapNet.

This module provides Large Language Model integration for intelligent
out-of-distribution handling and decision making.
"""

from .llm_assistant import LLMAssistant
from .ood_explainer import OODExplainer
from .decision_maker import LLMDecisionMaker
from .fever_classifier import DistilBertFeverClassifier, FeverTrainingConfig

__all__ = [
    "LLMAssistant",
    "OODExplainer",
    "LLMDecisionMaker",
    "DistilBertFeverClassifier",
    "FeverTrainingConfig",
]
