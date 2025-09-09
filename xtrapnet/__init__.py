from .model import XtrapNet
from .trainer import XtrapTrainer
from .controller import XtrapController
from .config import PipelineConfig, default_config
from .pipeline import XtrapPipeline
from .wrappers.ensemble import EnsembleWrapper

# Bayesian Neural Networks (v0.3.0)
from .bayesian import (
    BayesianNeuralNetwork,
    VariationalBNN,
    UncertaintyDecomposition,
    BayesianConformalPredictor
)

# Physics-Informed Neural Networks (v0.4.0)
from .physics import (
    PhysicsInformedNN,
    PhysicsLoss,
    DomainAwareExtrapolation
)

# LLM-Assisted Extrapolation (v0.5.0)
from .llm import (
    LLMAssistant,
    OODExplainer,
    LLMDecisionMaker
)

# Adaptive Learning & Meta-Learning (v0.6.0)
from .adaptive import (
    MetaLearner,
    OnlineAdaptation,
    ActiveLearning,
    ContinualLearning,
    MemoryBank
)

__all__ = [
    "XtrapNet",
    "XtrapTrainer", 
    "XtrapController",
    "PipelineConfig",
    "default_config",
    "XtrapPipeline",
    "EnsembleWrapper",
    # Bayesian components
    "BayesianNeuralNetwork",
    "VariationalBNN", 
    "UncertaintyDecomposition",
    "BayesianConformalPredictor",
    # Physics components
    "PhysicsInformedNN",
    "PhysicsLoss",
    "DomainAwareExtrapolation",
    # LLM components
    "LLMAssistant",
    "OODExplainer",
    "LLMDecisionMaker",
    # Adaptive learning components
    "MetaLearner",
    "OnlineAdaptation",
    "ActiveLearning",
    "ContinualLearning",
    "MemoryBank",
]