import numpy as np
from xtrapnet.model import XtrapNet
from xtrapnet.trainer import XtrapTrainer
from xtrapnet.controller import XtrapController

# Function defining ground truth
def ground_truth_function(x1, x2):
    return np.cos(x1) * np.cos(x2)

# Generate training data, excluding certain regions (to create OOD)
def generate_data(num_samples=500):
    features, labels = [], []
    for _ in range(num_samples):
        x1, x2 = np.random.uniform(-np.pi, np.pi, 2)
        # Exclude x1 < 0 and x2 > 0 (OOD region)
        if x1 < 0 and x2 > 0:
            continue  
        features.append([x1, x2])
        labels.append([ground_truth_function(x1, x2)])
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)

# Train a model
print("\nTraining model...")
features, labels = generate_data()
net = XtrapNet(input_dim=2)
trainer = XtrapTrainer(net, num_epochs=100)  # Reduce epochs for faster testing
trainer.train(labels, features)
print("Training complete!")

# Create XtrapController
xtrap_ctrl = XtrapController(trained_model=net, train_features=features, train_labels=labels, mode='predict_anyway')

# Test some predictions
test_points = np.array([
    [0.5, 0.5],   # In-distribution
    [-2, 2],      # OOD (excluded region)
    [np.pi, -np.pi]  # Edge of training range
])

print("\nPredictions (predict_anyway mode):")
predictions = xtrap_ctrl.predict(test_points)
for i, p in enumerate(test_points):
    print(f"Input: {p} → Prediction: {predictions[i]:.4f}")

# Test different fallback strategies
modes = ['clip', 'zero', 'nearest_data', 'symmetry', 'highest_confidence', 'backup']

for mode in modes:
    print(f"\nTesting mode: {mode}")
    xtrap_ctrl = XtrapController(trained_model=net, train_features=features, train_labels=labels, mode=mode)
    predictions = xtrap_ctrl.predict(test_points)
    for i, p in enumerate(test_points):
        print(f"Mode: {mode} | Input: {p} → Prediction: {predictions[i]:.4f}")

print("\n✅ All tests completed. If outputs make sense, you're ready to publish!")
