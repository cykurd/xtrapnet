# XtrapNet - Extrapolation-Aware Neural Networks  
[![PyPI Version](https://img.shields.io/pypi/v/xtrapnet)](https://pypi.org/project/xtrapnet/)  
[![Python Version](https://img.shields.io/pypi/pyversions/xtrapnet)](https://pypi.org/project/xtrapnet/)  
[![License](https://img.shields.io/pypi/l/xtrapnet)](https://opensource.org/licenses/MIT)  

XtrapNet is a cutting-edge deep learning framework designed to handle out-of-distribution (OOD) extrapolation in neural networks. Unlike traditional models that fail when encountering unseen data, XtrapNet:

- ✅ Detects OOD inputs and allows custom fallback behaviors  
- ✅ Supports ensemble uncertainty quantification  
- ✅ Offers multiple extrapolation control mechanisms  
- ✅ Works with PyTorch and integrates seamlessly with any model  

## Installation
To install XtrapNet:
```pip install xtrapnet```

## Usage Example
```
import numpy as np 
from xtrapnet import XtrapNet, XtrapTrainer, XtrapController

# Generate dummy training data
features = np.random.uniform(-3.14, 3.14, (100, 2)).astype(np.float32) labels = np.sin(features[:, 0]) * np.cos(features[:, 1]).reshape(-1, 1)

# Train the model
net = XtrapNet(input_dim=2) trainer = XtrapTrainer(net) trainer.train(labels, features)

# Define an extrapolation-aware controller
controller = XtrapController( trained_model=net, train_features=features, train_labels=labels, mode='warn' )

# Test prediction with OOD handling
test_input = np.array([[5.0, -3.5]]) # OOD point prediction = controller.predict(test_input) print("Prediction:", prediction)
```

## Extrapolation Handling Modes
XtrapNet allows you to control how the model reacts to out-of-distribution (OOD) inputs:

| Mode             | Behavior |
|-----------------|-------------|
| clip            | Restricts predictions within known value ranges |
| zero            | Returns 0 for OOD inputs |
| nearest_data    | Uses the closest training point's prediction |
| symmetry        | Uses symmetry-based assumptions to infer values |
| warn           | Prints a warning but still predicts |
| error           | Raises an error when encountering OOD data |
| highest_confidence | Selects the lowest-variance prediction |
| backup          | Uses a secondary model when uncertainty is high |
| deep_ensemble   | Averages predictions from multiple expert models |
| llm_assist      | Queries an LLM for a fallback prediction |


## Visualizing Extrapolation Behavior
```
import matplotlib.pyplot as plt 
x_test = np.linspace(-5, 5, 100).reshape(-1, 1) 
mean_pred, var_pred = controller.predict(x_test, return_variance=True)

plt.plot(x_test, mean_pred, label='Ensemble Mean', color='blue') 
plt.fill_between(x_test.flatten(), mean_pred - var_pred, mean_pred + var_pred, color='blue', alpha=0.2, label='Uncertainty (Variance)') 
plt.legend() 
plt.show()
```

This generates an extrapolation-aware prediction plot with uncertainty bands! 🔥

## Future Roadmap
We are actively developing new features:
- ✅ Bayesian Neural Network support
- ✅ Physics-Informed Neural Networks
- ✅ Integration with Large Language Models (LLMs)
- 🚀 Adaptive learning for OOD generalization
- 🚀 Built-in anomaly detection for real-world data

## Contributing
Want to improve XtrapNet? Feel free to submit a pull request! Contributions are welcome.  
🔗 **GitHub:** [https://github.com/cykurd/xtrapnet](https://github.com/cykurd/xtrapnet)  

## License
This project is licensed under the **MIT License**.

## Support
If you have any questions, feel free to open an issue on **GitHub** or reach out via **cykurd@gmail.com**. 🚀

🔥 Why Use XtrapNet?
Traditional neural networks struggle with out-of-distribution (OOD) data.  
🔥 XtrapNet is the first open-source library designed to intelligently control extrapolation!  
