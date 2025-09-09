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
]