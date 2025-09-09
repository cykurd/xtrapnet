"""
Bayesian Neural Networks Demo for XtrapNet v0.3.0

This script demonstrates the new Bayesian Neural Network capabilities
including uncertainty quantification, conformal prediction, and OOD detection.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import XtrapNet components
from xtrapnet import (
    BayesianNeuralNetwork,
    VariationalBNN,
    UncertaintyDecomposition,
    BayesianConformalPredictor
)


def generate_synthetic_data(n_samples=1000, noise_level=0.1, ood_ratio=0.2):
    """Generate synthetic data with clear in-distribution and OOD regions."""
    np.random.seed(42)
    
    # In-distribution data: y = sin(x) + noise
    x_id = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
    y_id = np.sin(x_id) + noise_level * np.random.randn(n_samples, 1)
    
    # Out-of-distribution data: y = sin(x) + bias + noise
    n_ood = int(n_samples * ood_ratio)
    x_ood = np.random.uniform(-5, -3, n_ood).reshape(-1, 1)  # OOD region
    y_ood = np.sin(x_ood) + 0.5 + noise_level * np.random.randn(n_ood, 1)
    
    # Combine data
    x = np.vstack([x_id, x_ood])
    y = np.vstack([y_id, y_ood])
    
    # Create labels for OOD detection
    ood_labels = np.zeros(len(x))
    ood_labels[-n_ood:] = 1  # OOD samples
    
    return x, y, ood_labels


def train_bayesian_model(x_train, y_train, x_val, y_val, model_type='variational'):
    """Train a Bayesian Neural Network."""
    print(f"Training {model_type} Bayesian Neural Network...")
    
    # Convert to tensors
    x_train_tensor = torch.FloatTensor(x_train)
    y_train_tensor = torch.FloatTensor(y_train)
    x_val_tensor = torch.FloatTensor(x_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create model
    if model_type == 'variational':
        model = VariationalBNN(
            input_dim=1,
            hidden_dims=[64, 32],
            output_dim=1,
            activation='relu',
            prior_std=1.0,
            vi_method='mean_field',
            temperature=1.0
        )
    else:
        model = BayesianNeuralNetwork(
            input_dim=1,
            hidden_dims=[64, 32],
            output_dim=1,
            activation='relu',
            prior_std=1.0
        )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Compute ELBO loss
        if model_type == 'variational':
            loss = model.elbo_loss(x_train_tensor, y_train_tensor, num_samples=1)
        else:
            loss = model.elbo_loss(x_train_tensor, y_train_tensor, num_samples=1)
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = model.elbo_loss(x_val_tensor, y_val_tensor, num_samples=1)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
    
    return model, train_losses, val_losses


def demonstrate_uncertainty_quantification(model, x_test, y_test):
    """Demonstrate uncertainty quantification capabilities."""
    print("\n=== Uncertainty Quantification Demo ===")
    
    # Get uncertainty decomposition
    uncertainty_decomp = UncertaintyDecomposition(model)
    uncertainty = uncertainty_decomp.decompose_uncertainty(
        torch.FloatTensor(x_test), num_samples=100
    )
    
    # Print uncertainty statistics
    print(f"Mean Epistemic Std: {torch.mean(uncertainty['epistemic_std']):.4f}")
    print(f"Mean Aleatoric Std: {torch.mean(uncertainty['aleatoric_std']):.4f}")
    print(f"Mean Total Std: {torch.mean(uncertainty['total_std']):.4f}")
    print(f"Epistemic Ratio: {torch.mean(uncertainty['epistemic_ratio']):.4f}")
    print(f"Aleatoric Ratio: {torch.mean(uncertainty['aleatoric_ratio']):.4f}")
    
    # Get uncertainty metrics
    metrics = uncertainty_decomp.get_uncertainty_metrics(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test), num_samples=100
    )
    
    print(f"\nUncertainty Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return uncertainty, metrics


def demonstrate_conformal_prediction(model, x_cal, y_cal, x_test, y_test):
    """Demonstrate Bayesian Conformal Prediction."""
    print("\n=== Bayesian Conformal Prediction Demo ===")
    
    # Create conformal predictor
    conformal_predictor = BayesianConformalPredictor(
        model, alpha=0.1, method='bayesian'
    )
    
    # Calibrate
    print("Calibrating conformal predictor...")
    conformal_predictor.calibrate(
        torch.FloatTensor(x_cal), torch.FloatTensor(y_cal), num_samples=100
    )
    
    # Get calibration info
    calib_info = conformal_predictor.get_calibration_info()
    print(f"Calibration Info: {calib_info}")
    
    # Make predictions
    predictions = conformal_predictor.predict_with_uncertainty_decomposition(
        torch.FloatTensor(x_test), num_samples=100
    )
    
    # Evaluate coverage
    coverage_results = conformal_predictor.evaluate_coverage(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test), num_samples=100
    )
    
    print(f"\nCoverage Results:")
    for key, value in coverage_results.items():
        print(f"  {key}: {value:.4f}")
    
    return conformal_predictor, predictions, coverage_results


def plot_results(x_train, y_train, x_test, y_test, uncertainty, predictions, ood_labels):
    """Plot the results of the Bayesian analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sort data for plotting
    sort_idx = np.argsort(x_test.flatten())
    x_sorted = x_test[sort_idx]
    y_sorted = y_test[sort_idx]
    
    # Plot 1: Training data and predictions
    axes[0, 0].scatter(x_train, y_train, alpha=0.6, label='Training Data', s=20)
    axes[0, 0].plot(x_sorted, uncertainty['mean'][sort_idx].numpy(), 'r-', label='Prediction')
    
    # Add uncertainty bands
    mean_sorted = uncertainty['mean'][sort_idx].numpy()
    std_sorted = uncertainty['total_std'][sort_idx].numpy()
    axes[0, 0].fill_between(
        x_sorted.flatten(),
        mean_sorted.flatten() - 1.96 * std_sorted.flatten(),
        mean_sorted.flatten() + 1.96 * std_sorted.flatten(),
        alpha=0.3, label='95% Confidence'
    )
    
    axes[0, 0].set_title('Bayesian Predictions with Uncertainty')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty decomposition
    epistemic_std = uncertainty['epistemic_std'][sort_idx].numpy()
    aleatoric_std = uncertainty['aleatoric_std'][sort_idx].numpy()
    total_std = uncertainty['total_std'][sort_idx].numpy()
    
    axes[0, 1].plot(x_sorted, epistemic_std, 'b-', label='Epistemic (Model)')
    axes[0, 1].plot(x_sorted, aleatoric_std, 'g-', label='Aleatoric (Data)')
    axes[0, 1].plot(x_sorted, total_std, 'r-', label='Total')
    axes[0, 1].set_title('Uncertainty Decomposition')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Uncertainty (Std)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Conformal prediction intervals
    axes[1, 0].scatter(x_test, y_test, alpha=0.6, label='Test Data', s=20)
    axes[1, 0].plot(x_sorted, predictions['prediction'][sort_idx].numpy(), 'r-', label='Prediction')
    
    # Add conformal intervals
    conformal_lower = predictions['conformal_lower'][sort_idx].numpy()
    conformal_upper = predictions['conformal_upper'][sort_idx].numpy()
    axes[1, 0].fill_between(
        x_sorted.flatten(),
        conformal_lower.flatten(),
        conformal_upper.flatten(),
        alpha=0.3, label='Conformal Interval'
    )
    
    axes[1, 0].set_title('Conformal Prediction Intervals')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: OOD detection
    ood_mask = ood_labels == 1
    id_mask = ood_labels == 0
    
    axes[1, 1].scatter(x_test[id_mask], uncertainty['total_std'][id_mask].numpy(), 
                      alpha=0.6, label='In-Distribution', s=20)
    axes[1, 1].scatter(x_test[ood_mask], uncertainty['total_std'][ood_mask].numpy(), 
                      alpha=0.6, label='Out-of-Distribution', s=20, color='red')
    
    axes[1, 1].set_title('OOD Detection via Uncertainty')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Total Uncertainty')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration function."""
    print("XtrapNet v0.3.0: Bayesian Neural Networks Demo")
    print("=" * 50)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    x, y, ood_labels = generate_synthetic_data(n_samples=1000, noise_level=0.1)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    # Normalize data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_train = scaler_x.fit_transform(x_train)
    x_cal = scaler_x.transform(x_cal)
    x_test = scaler_x.transform(x_test)
    y_train = scaler_y.fit_transform(y_train)
    y_cal = scaler_y.transform(y_cal)
    y_test = scaler_y.transform(y_test)
    
    print(f"Training set size: {len(x_train)}")
    print(f"Calibration set size: {len(x_cal)}")
    print(f"Test set size: {len(x_test)}")
    
    # Train Bayesian model
    model, train_losses, val_losses = train_bayesian_model(
        x_train, y_train, x_cal, y_cal, model_type='variational'
    )
    
    # Demonstrate uncertainty quantification
    uncertainty, metrics = demonstrate_uncertainty_quantification(model, x_test, y_test)
    
    # Demonstrate conformal prediction
    conformal_predictor, predictions, coverage_results = demonstrate_conformal_prediction(
        model, x_cal, y_cal, x_test, y_test
    )
    
    # Plot results
    plot_results(x_train, y_train, x_test, y_test, uncertainty, predictions, ood_labels)
    
    print("\n=== Demo Complete ===")
    print("Key Features Demonstrated:")
    print("✓ Bayesian Neural Networks with variational inference")
    print("✓ Epistemic vs Aleatoric uncertainty decomposition")
    print("✓ Bayesian Conformal Prediction with statistical guarantees")
    print("✓ OOD detection via uncertainty quantification")
    print("✓ Comprehensive uncertainty visualization")


if __name__ == "__main__":
    main()
