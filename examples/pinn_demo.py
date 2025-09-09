"""
Physics-Informed Neural Networks Demo for XtrapNet v0.4.0

This script demonstrates the new PINN capabilities including:
- Heat equation solving
- Domain-aware extrapolation
- Physics-constrained uncertainty quantification
- Multi-physics problems
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import XtrapNet components
from xtrapnet import (
    PhysicsInformedNN,
    PhysicsLoss,
    DomainAwareExtrapolation
)


def generate_heat_equation_data(n_samples=1000, noise_level=0.01):
    """Generate synthetic data for 1D heat equation."""
    np.random.seed(42)
    
    # Spatial domain: x ∈ [0, 1]
    # Temporal domain: t ∈ [0, 1]
    x = np.random.uniform(0, 1, n_samples)
    t = np.random.uniform(0, 1, n_samples)
    
    # Analytical solution: u(x,t) = sin(πx) * exp(-π²t)
    u_analytical = np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    
    # Add noise
    u_noisy = u_analytical + noise_level * np.random.randn(n_samples)
    
    # Combine coordinates
    coords = np.column_stack([x, t])
    
    return coords, u_noisy.reshape(-1, 1), u_analytical.reshape(-1, 1)


def generate_boundary_conditions(n_boundary=100):
    """Generate boundary and initial conditions."""
    # Initial condition: u(x,0) = sin(πx)
    x_initial = np.linspace(0, 1, n_boundary)
    t_initial = np.zeros_like(x_initial)
    u_initial = np.sin(np.pi * x_initial)
    
    # Boundary conditions: u(0,t) = u(1,t) = 0
    t_boundary = np.linspace(0, 1, n_boundary)
    x_boundary_left = np.zeros_like(t_boundary)
    x_boundary_right = np.ones_like(t_boundary)
    u_boundary = np.zeros_like(t_boundary)
    
    # Combine
    x_boundary = np.concatenate([x_boundary_left, x_boundary_right])
    t_boundary_full = np.concatenate([t_boundary, t_boundary])
    u_boundary_full = np.concatenate([u_boundary, u_boundary])
    
    initial_coords = np.column_stack([x_initial, t_initial])
    boundary_coords = np.column_stack([x_boundary, t_boundary_full])
    
    return initial_coords, u_initial.reshape(-1, 1), boundary_coords, u_boundary_full.reshape(-1, 1)


def train_pinn_model(x_data, y_data, x_initial, y_initial, x_boundary, y_boundary):
    """Train a Physics-Informed Neural Network."""
    print("Training Physics-Informed Neural Network...")
    
    # Convert to tensors
    x_data_tensor = torch.FloatTensor(x_data)
    y_data_tensor = torch.FloatTensor(y_data)
    x_initial_tensor = torch.FloatTensor(x_initial)
    y_initial_tensor = torch.FloatTensor(y_initial)
    x_boundary_tensor = torch.FloatTensor(x_boundary)
    y_boundary_tensor = torch.FloatTensor(y_boundary)
    
    # Create PINN model
    model = PhysicsInformedNN(
        input_dim=2,  # (x, t)
        hidden_dims=[50, 50, 50],
        output_dim=1,  # u(x,t)
        activation='tanh',
        physics_loss_weight=1.0,
        data_loss_weight=1.0,
        boundary_loss_weight=10.0,
        initial_loss_weight=10.0
    )
    
    # Add physics constraint (heat equation)
    model.add_physics_constraint(
        lambda x, m: PhysicsLoss.heat_equation_1d(x, m, thermal_diffusivity=1.0),
        weight=1.0,
        name="heat_equation"
    )
    
    # Add boundary conditions
    def boundary_condition(x, model):
        """Boundary condition: u(0,t) = u(1,t) = 0"""
        u_pred = model.forward(x)
        return torch.mean(u_pred**2)
    
    model.add_boundary_condition(boundary_condition, weight=10.0, name="dirichlet_boundary")
    
    # Add initial condition
    def initial_condition(x, model):
        """Initial condition: u(x,0) = sin(πx)"""
        u_pred = model.forward(x)
        x_coords = x[:, 0:1]
        u_true = torch.sin(np.pi * x_coords)
        return torch.mean((u_pred - u_true)**2)
    
    model.add_initial_condition(initial_condition, weight=10.0, name="initial_condition")
    
    # Set domain bounds
    model.set_domain_bounds({
        'x': (0.0, 1.0),
        't': (0.0, 1.0)
    })
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000
    
    # Training loop
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Compute total loss
        loss_dict = model.compute_total_loss(
            x_data_tensor, y_data_tensor,
            x_data_tensor,  # Physics points (same as data points)
            x_boundary_tensor,
            x_initial_tensor
        )
        
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        optimizer.step()
        
        losses.append({
            'total': total_loss.item(),
            'data': loss_dict['data_loss'].item(),
            'physics': loss_dict['physics_loss'].item(),
            'boundary': loss_dict['boundary_loss'].item(),
            'initial': loss_dict['initial_loss'].item()
        })
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}")
            print(f"  Data: {loss_dict['data_loss'].item():.6f}, "
                  f"Physics: {loss_dict['physics_loss'].item():.6f}, "
                  f"Boundary: {loss_dict['boundary_loss'].item():.6f}, "
                  f"Initial: {loss_dict['initial_loss'].item():.6f}")
    
    return model, losses


def demonstrate_domain_aware_extrapolation(model, x_train, x_test):
    """Demonstrate domain-aware extrapolation capabilities."""
    print("\n=== Domain-Aware Extrapolation Demo ===")
    
    # Create domain-aware extrapolation
    domain_bounds = {'x': (0.0, 1.0), 't': (0.0, 1.0)}
    physics_constraints = [
        lambda x, m: PhysicsLoss.heat_equation_1d(x, m, thermal_diffusivity=1.0)
    ]
    
    extrapolator = DomainAwareExtrapolation(
        model, domain_bounds, physics_constraints
    )
    
    # Test extrapolation beyond training domain
    x_extrapolation = torch.FloatTensor([
        [1.5, 0.5],  # Beyond spatial domain
        [0.5, 1.5],  # Beyond temporal domain
        [1.2, 0.8],  # Slightly beyond domain
        [0.8, 1.2],  # Slightly beyond domain
    ])
    
    # Extrapolate with uncertainty
    results = extrapolator.extrapolate_with_uncertainty(x_extrapolation, num_samples=50)
    
    print(f"Extrapolation Results:")
    print(f"  Mean predictions: {results['mean'].flatten().numpy()}")
    print(f"  Prediction std: {results['std'].flatten().numpy()}")
    print(f"  Physics uncertainty: {results['physics_uncertainty'].item():.6f}")
    print(f"  Domain violation: {results['domain_violation'].item():.6f}")
    
    # Evaluate extrapolation quality
    quality_metrics = extrapolator.evaluate_extrapolation_quality(x_extrapolation)
    print(f"\nExtrapolation Quality Metrics:")
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return extrapolator, results


def demonstrate_adaptive_extrapolation(extrapolator, x_extrapolation):
    """Demonstrate adaptive extrapolation."""
    print("\n=== Adaptive Extrapolation Demo ===")
    
    # Adaptive extrapolation
    adaptive_results = extrapolator.adaptive_extrapolation(
        x_extrapolation, max_iterations=20, tolerance=1e-4
    )
    
    print(f"Adaptive Extrapolation Results:")
    print(f"  Final predictions: {adaptive_results['predictions'].flatten().numpy()}")
    print(f"  Final physics weight: {adaptive_results['final_physics_weight']:.6f}")
    print(f"  Final violation: {adaptive_results['final_violation']:.6f}")
    print(f"  Iterations: {adaptive_results['iterations']}")
    print(f"  Converged: {adaptive_results['converged']}")
    
    return adaptive_results


def plot_results(x_train, y_train, x_test, y_test, model, losses, extrapolation_results):
    """Plot the results of the PINN analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Training loss
    epochs = range(len(losses))
    axes[0, 0].plot(epochs, [l['total'] for l in losses], label='Total Loss')
    axes[0, 0].plot(epochs, [l['data'] for l in losses], label='Data Loss')
    axes[0, 0].plot(epochs, [l['physics'] for l in losses], label='Physics Loss')
    axes[0, 0].plot(epochs, [l['boundary'] for l in losses], label='Boundary Loss')
    axes[0, 0].plot(epochs, [l['initial'] for l in losses], label='Initial Loss')
    axes[0, 0].set_title('Training Loss Components')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training data
    scatter = axes[0, 1].scatter(x_train[:, 0], x_train[:, 1], c=y_train.flatten(), 
                                cmap='viridis', s=20, alpha=0.7)
    axes[0, 1].set_title('Training Data Distribution')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('t')
    plt.colorbar(scatter, ax=axes[0, 1], label='u(x,t)')
    
    # Plot 3: PINN predictions vs analytical solution
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(x_test)
        y_pred = model.forward(x_test_tensor).numpy()
    
    axes[0, 2].scatter(y_test.flatten(), y_pred.flatten(), alpha=0.6, s=20)
    axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 2].set_title('PINN Predictions vs Analytical Solution')
    axes[0, 2].set_xlabel('Analytical Solution')
    axes[0, 2].set_ylabel('PINN Prediction')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Compute R²
    ss_res = np.sum((y_test.flatten() - y_pred.flatten())**2)
    ss_tot = np.sum((y_test.flatten() - np.mean(y_test.flatten()))**2)
    r2 = 1 - (ss_res / ss_tot)
    axes[0, 2].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 2].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: Solution surface
    x_surface = np.linspace(0, 1, 50)
    t_surface = np.linspace(0, 1, 50)
    X, T = np.meshgrid(x_surface, t_surface)
    
    # PINN predictions
    coords_surface = np.column_stack([X.flatten(), T.flatten()])
    coords_tensor = torch.FloatTensor(coords_surface)
    
    model.eval()
    with torch.no_grad():
        u_surface = model.forward(coords_tensor).numpy().reshape(X.shape)
    
    im = axes[1, 0].contourf(X, T, u_surface, levels=20, cmap='viridis')
    axes[1, 0].set_title('PINN Solution Surface')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('t')
    plt.colorbar(im, ax=axes[1, 0], label='u(x,t)')
    
    # Plot 5: Analytical solution surface
    u_analytical_surface = np.sin(np.pi * X) * np.exp(-np.pi**2 * T)
    
    im2 = axes[1, 1].contourf(X, T, u_analytical_surface, levels=20, cmap='viridis')
    axes[1, 1].set_title('Analytical Solution Surface')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('t')
    plt.colorbar(im2, ax=axes[1, 1], label='u(x,t)')
    
    # Plot 6: Error surface
    error_surface = np.abs(u_surface - u_analytical_surface)
    
    im3 = axes[1, 2].contourf(X, T, error_surface, levels=20, cmap='Reds')
    axes[1, 2].set_title('Absolute Error Surface')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('t')
    plt.colorbar(im3, ax=axes[1, 2], label='|Error|')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration function."""
    print("XtrapNet v0.4.0: Physics-Informed Neural Networks Demo")
    print("=" * 60)
    
    # Generate synthetic data
    print("Generating synthetic heat equation data...")
    x_data, y_data, y_analytical = generate_heat_equation_data(n_samples=1000, noise_level=0.01)
    x_initial, y_initial, x_boundary, y_boundary = generate_boundary_conditions(n_boundary=100)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42
    )
    
    print(f"Training set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print(f"Boundary conditions: {len(x_boundary)}")
    print(f"Initial conditions: {len(x_initial)}")
    
    # Train PINN model
    model, losses = train_pinn_model(
        x_train, y_train, x_initial, y_initial, x_boundary, y_boundary
    )
    
    # Demonstrate domain-aware extrapolation
    extrapolator, extrapolation_results = demonstrate_domain_aware_extrapolation(
        model, x_train, x_test
    )
    
    # Demonstrate adaptive extrapolation
    x_extrapolation = torch.FloatTensor([[1.2, 0.8], [0.8, 1.2]])
    adaptive_results = demonstrate_adaptive_extrapolation(extrapolator, x_extrapolation)
    
    # Plot results
    plot_results(x_train, y_train, x_test, y_test, model, losses, extrapolation_results)
    
    print("\n=== Demo Complete ===")
    print("Key Features Demonstrated:")
    print("✓ Physics-Informed Neural Networks with heat equation")
    print("✓ Domain-aware extrapolation beyond training bounds")
    print("✓ Physics-constrained uncertainty quantification")
    print("✓ Adaptive extrapolation with convergence")
    print("✓ Comprehensive physics loss functions")
    print("✓ Boundary and initial condition enforcement")


if __name__ == "__main__":
    main()
