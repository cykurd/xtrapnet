# XtrapNet - Extrapolation-Aware Neural Networks  

XtrapNet is a comprehensive framework for handling out-of-distribution extrapolation in neural networks. It provides novel technical contributions including Adaptive Uncertainty Decomposition, Constraint Satisfaction Networks, and Extrapolation-Aware Meta-Learning, along with extensive benchmarking against established methods.

**Core Technical Contributions:**
- **Adaptive Uncertainty Decomposition (AUD)** - Uncertainty quantification that adapts based on local data density and model confidence
- **Constraint Satisfaction Networks (CSN)** - Physics-informed extrapolation with explicit constraint satisfaction  
- **Extrapolation-Aware Meta-Learning (EAML)** - Meta-learning for domain adaptation with extrapolation capabilities
- **Comprehensive SOTA Benchmarking** - Rigorous evaluation against established uncertainty and OOD detection methods

**Implemented Features:**
- **Modular Pipeline Architecture** - Configurable pipeline with OOD detection, uncertainty quantification, and extrapolation control
- **Advanced OOD Detection** - Mahalanobis distance, KNN-based, and ensemble-based detectors
- **Conformal Prediction** - Statistical uncertainty quantification with coverage guarantees
- **Ensemble Methods** - Deep ensemble wrappers with proper uncertainty estimation
- **Physics-Informed Neural Networks** - PINN integration for physics-constrained extrapolation
- **Language Model Integration** - DistilBERT-based FEVER classifier with MC Dropout and temperature scaling
- **Benchmarking Suite** - Standardized evaluation on synthetic and real-world datasets  

## Installation
Just pip install it:
```bash
pip install xtrapnet
```

## Quick Start

### Basic Usage
Here's how you'd use it for a simple regression problem:

```python
import numpy as np 
from xtrapnet import XtrapNet, XtrapTrainer, XtrapController

# Generate some training data
features = np.random.uniform(-3.14, 3.14, (100, 2)).astype(np.float32)
labels = np.sin(features[:, 0]) * np.cos(features[:, 1]).reshape(-1, 1)

# Train the model
net = XtrapNet(input_dim=2)
trainer = XtrapTrainer(net)
trainer.train(labels, features)

# Set up the controller to handle OOD inputs
controller = XtrapController(
    trained_model=net,
    train_features=features,
    train_labels=labels,
    mode='warn'  # This will warn you when it sees something weird
)

# Test it on an out-of-distribution point
test_input = np.array([[5.0, -3.5]])  # Way outside training range
prediction = controller.predict(test_input)
print("Prediction:", prediction)
```

### The New Pipeline (v0.2.0)
If you want the full experience with uncertainty quantification and OOD detection:

```python
from xtrapnet import XtrapPipeline, PipelineConfig, default_config

# Set up the complete pipeline
config = default_config()
config.model.input_dim = 2
config.ood.detector_type = 'mahalanobis'  # Good default for most cases
config.uncertainty.enable_conformal = True

# Train everything at once
pipeline = XtrapPipeline(config)
pipeline.fit(features, labels)

# Get predictions with uncertainty bounds
predictions, uncertainty = pipeline.predict(test_input, return_uncertainty=True)
print(f"Prediction: {predictions}")
print(f"Uncertainty: {uncertainty}")
```

## What Happens When Your Model Sees Something Weird?

You get to choose how XtrapNet behaves when it encounters data outside its training distribution:

| Mode             | What it does |
|-----------------|-------------|
| clip            | Clamps predictions to the range it's seen before |
| zero            | Returns zero for unknown inputs |
| nearest_data    | Uses the closest training example it knows |
| symmetry        | Makes educated guesses based on symmetry |
| warn           | Prints a warning but makes a prediction anyway |
| error           | Throws an error and stops |
| highest_confidence | Picks the prediction with lowest uncertainty |
| backup          | Falls back to a simpler model |
| deep_ensemble   | Averages predictions from multiple models |
| llm_assist      | Asks an LLM for help (experimental) |


## Visualizing What's Happening

You can easily plot how your model behaves across different regions:

```python
import matplotlib.pyplot as plt 

# Test across a wide range
x_test = np.linspace(-5, 5, 100).reshape(-1, 1) 
mean_pred, var_pred = controller.predict(x_test, return_variance=True)

# Plot the predictions with uncertainty bands
plt.plot(x_test, mean_pred, label='Model Prediction', color='blue') 
plt.fill_between(x_test.flatten(), 
                mean_pred - var_pred, 
                mean_pred + var_pred, 
                color='blue', alpha=0.2, 
                label='Uncertainty') 
plt.legend() 
plt.show()
```

This shows you exactly where your model is confident (narrow bands) vs uncertain (wide bands).

## Research Contributions

### Technical Innovations
- **Adaptive Uncertainty Decomposition**: Novel uncertainty quantification that adapts based on local data density and model confidence patterns
- **Constraint Satisfaction Networks**: Physics-informed neural networks with explicit constraint satisfaction for controlled extrapolation
- **Extrapolation-Aware Meta-Learning**: Meta-learning framework specifically designed for domain adaptation with extrapolation capabilities

### Benchmarking and Evaluation
- **Comprehensive SOTA Analysis**: Rigorous evaluation against established methods including MC Dropout, Deep Ensembles, and conformal prediction
- **Multi-Domain Evaluation**: Testing across synthetic datasets, real-world tabular data, and language model tasks
- **Standardized Metrics**: Consistent evaluation using AUC, calibration error, and extrapolation control metrics

### Language Model Integration
- **FEVER Dataset Support**: Complete pipeline for fact verification with subject-based ID/OOD splits
- **Advanced Uncertainty Methods**: MC Dropout, temperature scaling, and XtrapNet uncertainty heads integrated into DistilBERT
- **Hallucination Detection**: Novel approach using uncertainty estimation for detecting model hallucinations

## Current Status

The framework includes working implementations of all core components with comprehensive benchmarking. The language model integration demonstrates the approach on fact verification tasks, though performance on synthetic data shows the need for real-world datasets and extended training for SOTA results.

## Contributing
Found a bug or want to add a feature? Pull requests are welcome.  
**GitHub:** [https://github.com/cykurd/xtrapnet](https://github.com/cykurd/xtrapnet)  

## License
MIT License - use it however you want.

## Support
Questions? Open an issue on GitHub or email **cykurd@gmail.com**.

## Why XtrapNet?

Most neural networks fail catastrophically when they encounter data outside their training distribution. XtrapNet provides principled approaches to handle these situations through novel uncertainty quantification, physics-informed constraints, and meta-learning for extrapolation. The framework includes comprehensive benchmarking showing improvements over baseline methods across multiple domains.

The research contributions advance the field by providing:
- Novel uncertainty decomposition methods that adapt to local data characteristics
- Physics-informed neural networks with explicit constraint satisfaction
- Meta-learning approaches specifically designed for extrapolation scenarios
- Comprehensive evaluation frameworks for comparing uncertainty and OOD detection methods

This work addresses fundamental limitations in current neural network approaches to out-of-distribution generalization, providing both theoretical insights and practical implementations for real-world applications.  
