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

# Production-Ready Anomaly Detection (v0.7.0)
from .anomaly import (
    MultiModalAnomalyDetector,
    RealTimeMonitor,
    ExplainableAnomalyDetector,
    DeploymentTools,
    AnomalyBenchmark
)

# Comprehensive Benchmarking & Evaluation (v0.8.0)
from .benchmarks import (
    # Evaluation metrics
    EvaluationMetrics,
    OODDetectionMetrics,
    UncertaintyMetrics,
    ExtrapolationMetrics,
    AnomalyDetectionMetrics,
    
    # Benchmark datasets
    BenchmarkDataset,
    SyntheticOODDataset,
    RealWorldOODDataset,
    AnomalyDetectionDataset,
    
    # Benchmark suites
    BenchmarkSuite,
    BenchmarkConfig,
    OODBenchmark,
    UncertaintyBenchmark,
    ExtrapolationBenchmark,
    AnomalyBenchmark as BenchmarkAnomalyBenchmark,
    FullSystemBenchmark,
    
    # Reporting
    BenchmarkReport,
    ComparisonReport,
    PerformanceReport,
    BenchmarkReporter
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
    # Anomaly detection components
    "MultiModalAnomalyDetector",
    "RealTimeMonitor",
    "ExplainableAnomalyDetector",
    "DeploymentTools",
    "AnomalyBenchmark",
    
    # Benchmarking components
    "EvaluationMetrics",
    "OODDetectionMetrics",
    "UncertaintyMetrics",
    "ExtrapolationMetrics",
    "AnomalyDetectionMetrics",
    "BenchmarkDataset",
    "SyntheticOODDataset",
    "RealWorldOODDataset",
    "AnomalyDetectionDataset",
    "BenchmarkSuite",
    "BenchmarkConfig",
    "OODBenchmark",
    "UncertaintyBenchmark",
    "ExtrapolationBenchmark",
    "BenchmarkAnomalyBenchmark",
    "FullSystemBenchmark",
    "BenchmarkReport",
    "ComparisonReport",
    "PerformanceReport",
    "BenchmarkReporter",
]