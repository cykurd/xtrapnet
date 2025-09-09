"""
Adaptive Learning Demo for XtrapNet v0.6.0

This script demonstrates the new adaptive learning capabilities including:
- Meta-learning for rapid OOD adaptation
- Online adaptation from data streams
- Active learning for intelligent data acquisition
- Continual learning without catastrophic forgetting
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

# Import XtrapNet components
from xtrapnet import (
    MetaLearner,
    OnlineAdaptation,
    ActiveLearning,
    ContinualLearning,
    MemoryBank
)


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def generate_task_data(task_id: int, n_samples: int = 50) -> List[tuple]:
    """Generate synthetic task data."""
    # Different tasks have different data distributions
    if task_id == 0:
        # Task 0: Linear relationship
        x = torch.randn(n_samples, 2)
        y = x[:, 0] + 0.5 * x[:, 1] + 0.1 * torch.randn(n_samples, 1)
    elif task_id == 1:
        # Task 1: Quadratic relationship
        x = torch.randn(n_samples, 2)
        y = x[:, 0] ** 2 + 0.3 * x[:, 1] + 0.1 * torch.randn(n_samples, 1)
    elif task_id == 2:
        # Task 2: Sinusoidal relationship
        x = torch.randn(n_samples, 2)
        y = torch.sin(x[:, 0]) + 0.5 * x[:, 1] + 0.1 * torch.randn(n_samples, 1)
    else:
        # Default: Random relationship
        x = torch.randn(n_samples, 2)
        y = torch.randn(n_samples, 1)
    
    return list(zip(x, y))


def demonstrate_meta_learning():
    """Demonstrate meta-learning capabilities."""
    print("=== Meta-Learning Demo ===")
    
    # Create model and meta-learner
    model = SimpleModel()
    meta_learner = MetaLearner(algorithm="maml", inner_lr=0.01, meta_lr=0.001)
    
    # Generate tasks
    tasks = []
    for task_id in range(3):
        support_data = generate_task_data(task_id, 20)
        query_data = generate_task_data(task_id, 10)
        tasks.append({
            'support': support_data,
            'query': query_data
        })
    
    # Meta-learn on tasks
    print("1. Meta-learning on multiple tasks...")
    for i, task_data in enumerate(tasks):
        result = meta_learner.adapt_to_task(model, task_data, f"task_{i}")
        print(f"   Task {i}: Final loss = {result['final_loss']:.4f}")
    
    # Few-shot adaptation
    print("2. Few-shot adaptation to new task...")
    new_task_data = generate_task_data(3, 15)  # New task
    result = meta_learner.few_shot_adaptation(model, new_task_data, n_shots=5)
    print(f"   Few-shot result: Final loss = {result['final_loss']:.4f}")
    
    # Get adaptation summary
    summary = meta_learner.get_adaptation_summary()
    print(f"3. Adaptation summary: {summary}")
    print()
    
    return meta_learner


def demonstrate_online_adaptation():
    """Demonstrate online adaptation capabilities."""
    print("=== Online Adaptation Demo ===")
    
    # Create model and online adapter
    model = SimpleModel()
    online_adapter = OnlineAdaptation(
        model=model,
        learning_rate=0.001,
        memory_size=100,
        adaptation_threshold=0.1
    )
    
    # Simulate online data stream
    print("1. Simulating online data stream...")
    data_stream = []
    for i in range(50):
        # Gradually changing distribution
        x = torch.randn(1, 2) + 0.1 * i  # Drift over time
        y = x[:, 0] + 0.5 * x[:, 1] + 0.1 * torch.randn(1, 1)
        uncertainty = {
            'epistemic_std': 0.1 + 0.05 * np.sin(i * 0.1),
            'aleatoric_std': 0.05,
            'total_std': 0.15 + 0.05 * np.sin(i * 0.1)
        }
        data_stream.append((x, y, uncertainty))
    
    # Process data stream
    adaptation_count = 0
    for i, (x, y, uncertainty) in enumerate(data_stream):
        if online_adapter.detect_adaptation_need(x, model(x), uncertainty):
            result = online_adapter.adapt_online(x, y, uncertainty)
            adaptation_count += 1
            if adaptation_count <= 3:  # Show first few adaptations
                print(f"   Adaptation {adaptation_count}: Loss = {result['final_loss']:.4f}")
    
    print(f"   Total adaptations: {adaptation_count}")
    
    # Get statistics
    stats = online_adapter.get_adaptation_statistics()
    print(f"2. Adaptation statistics: {stats}")
    print()
    
    return online_adapter


def demonstrate_active_learning():
    """Demonstrate active learning capabilities."""
    print("=== Active Learning Demo ===")
    
    # Create model and active learner
    model = SimpleModel()
    active_learner = ActiveLearning(
        acquisition_function="uncertainty",
        query_batch_size=3,
        uncertainty_threshold=0.1
    )
    
    # Generate unlabeled data pool
    print("1. Generating unlabeled data pool...")
    unlabeled_data = torch.randn(100, 2)
    
    # Oracle function (simulates getting true labels)
    def oracle_function(inputs):
        return inputs[:, 0] + 0.5 * inputs[:, 1] + 0.1 * torch.randn(inputs.shape[0], 1)
    
    # Uncertainty function (simulates model uncertainty)
    def uncertainty_function(model, inputs):
        with torch.no_grad():
            outputs = model(inputs)
            # Simulate uncertainty based on prediction variance
            uncertainty = torch.abs(outputs) * 0.1 + 0.05
        return uncertainty.squeeze()
    
    # Performance evaluation function
    def performance_eval(model):
        test_data = torch.randn(20, 2)
        test_targets = oracle_function(test_data)
        with torch.no_grad():
            predictions = model(test_data)
            mse = nn.MSELoss()(predictions, test_targets)
        return mse.item()
    
    # Run active learning loop
    print("2. Running active learning loop...")
    results = active_learner.active_learning_loop(
        model=model,
        initial_unlabeled_data=unlabeled_data,
        oracle_function=oracle_function,
        uncertainty_function=uncertainty_function,
        max_iterations=5,
        performance_eval_function=performance_eval
    )
    
    print(f"   Total queries made: {results['total_queries']}")
    print(f"   Final performance: {results['performance_history'][-1]:.4f}")
    
    # Get query statistics
    stats = active_learner.get_query_statistics()
    print(f"3. Query statistics: {stats}")
    print()
    
    return active_learner


def demonstrate_continual_learning():
    """Demonstrate continual learning capabilities."""
    print("=== Continual Learning Demo ===")
    
    # Create model and continual learner
    model = SimpleModel()
    continual_learner = ContinualLearning(
        model=model,
        learning_rate=0.001,
        memory_size=200,
        method="ewc"
    )
    
    # Generate task stream
    print("1. Generating task stream...")
    task_stream = []
    for task_id in range(3):
        task_data = generate_task_data(task_id, 30)
        task_stream.append((task_id, task_data))
    
    # Run continual learning
    print("2. Running continual learning...")
    results = continual_learner.continual_learning_loop(
        task_stream=task_stream,
        epochs_per_task=5
    )
    
    # Show results
    for i, task_result in enumerate(results['task_results']):
        print(f"   Task {i}: Final loss = {task_result['final_loss']:.4f}")
    
    # Analyze forgetting
    forgetting_analysis = results['forgetting_analysis']
    print(f"3. Forgetting analysis: {forgetting_analysis}")
    
    # Get statistics
    stats = continual_learner.get_continual_learning_statistics()
    print(f"4. Continual learning statistics: {stats}")
    print()
    
    return continual_learner


def demonstrate_memory_bank():
    """Demonstrate memory bank capabilities."""
    print("=== Memory Bank Demo ===")
    
    # Create memory bank
    memory_bank = MemoryBank(
        max_size=100,
        embedding_dim=10,
        similarity_threshold=0.7
    )
    
    # Add experiences
    print("1. Adding experiences to memory bank...")
    for i in range(20):
        input_data = torch.randn(1, 2)
        prediction = torch.randn(1, 1)
        uncertainty = {
            'epistemic_std': 0.1 + 0.05 * np.random.random(),
            'aleatoric_std': 0.05,
            'total_std': 0.15 + 0.05 * np.random.random()
        }
        target = torch.randn(1, 1)
        metadata = {'ood_score': np.random.random(), 'task_id': i % 3}
        
        experience_id = memory_bank.add_experience(
            input_data, prediction, uncertainty, target, metadata
        )
    
    print(f"   Added {memory_bank.total_experiences} experiences")
    
    # Retrieve similar experiences
    print("2. Retrieving similar experiences...")
    query_input = torch.randn(1, 2)
    query_prediction = torch.randn(1, 1)
    query_uncertainty = {'epistemic_std': 0.2, 'aleatoric_std': 0.1, 'total_std': 0.3}
    
    similar_experiences = memory_bank.retrieve_similar_experiences(
        query_input, query_prediction, query_uncertainty, n_experiences=5
    )
    print(f"   Retrieved {len(similar_experiences)} similar experiences")
    
    # Retrieve by uncertainty
    print("3. Retrieving by uncertainty range...")
    uncertainty_experiences = memory_bank.retrieve_by_uncertainty(
        uncertainty_range=(0.1, 0.3), n_experiences=5
    )
    print(f"   Retrieved {len(uncertainty_experiences)} experiences in uncertainty range")
    
    # Retrieve by importance
    print("4. Retrieving most important experiences...")
    important_experiences = memory_bank.retrieve_by_importance(n_experiences=5)
    print(f"   Retrieved {len(important_experiences)} most important experiences")
    
    # Get statistics
    stats = memory_bank.get_memory_statistics()
    print(f"5. Memory bank statistics: {stats}")
    print()
    
    return memory_bank


def demonstrate_integrated_adaptive_learning():
    """Demonstrate integrated adaptive learning system."""
    print("=== Integrated Adaptive Learning Demo ===")
    
    # Create components
    model = SimpleModel()
    meta_learner = MetaLearner(algorithm="reptile")
    online_adapter = OnlineAdaptation(model=model)
    active_learner = ActiveLearning()
    memory_bank = MemoryBank()
    
    print("1. Integrated adaptive learning pipeline...")
    
    # Simulate adaptive learning scenario
    for iteration in range(10):
        # Generate new data
        input_data = torch.randn(1, 2)
        target = torch.randn(1, 1)
        uncertainty = {
            'epistemic_std': 0.1 + 0.05 * np.random.random(),
            'aleatoric_std': 0.05,
            'total_std': 0.15 + 0.05 * np.random.random()
        }
        
        # Store in memory bank
        prediction = model(input_data)
        experience_id = memory_bank.add_experience(
            input_data, prediction, uncertainty, target
        )
        
        # Check if online adaptation is needed
        if online_adapter.detect_adaptation_need(input_data, prediction, uncertainty):
            online_adapter.adapt_online(input_data, target, uncertainty)
        
        # Every 5 iterations, do meta-learning
        if iteration % 5 == 0 and iteration > 0:
            # Retrieve similar experiences for meta-learning
            similar_experiences = memory_bank.retrieve_similar_experiences(
                input_data, prediction, uncertainty, n_experiences=10
            )
            
            if len(similar_experiences) >= 5:
                # Create task data
                support_data = [(exp['input_data'], exp['target']) for exp in similar_experiences[:5]]
                query_data = [(exp['input_data'], exp['target']) for exp in similar_experiences[5:]]
                task_data = {'support': support_data, 'query': query_data}
                
                # Meta-learn
                meta_learner.adapt_to_task(model, task_data, f"meta_task_{iteration}")
    
    print("   Integrated adaptive learning completed")
    
    # Get final statistics
    meta_stats = meta_learner.get_adaptation_summary()
    online_stats = online_adapter.get_adaptation_statistics()
    memory_stats = memory_bank.get_memory_statistics()
    
    print(f"2. Final statistics:")
    print(f"   Meta-learning: {meta_stats}")
    print(f"   Online adaptation: {online_stats}")
    print(f"   Memory bank: {memory_stats}")
    print()


def main():
    """Main demonstration function."""
    print("XtrapNet v0.6.0: Adaptive Learning & Meta-Learning Demo")
    print("=" * 70)
    
    # Demonstrate individual components
    meta_learner = demonstrate_meta_learning()
    online_adapter = demonstrate_online_adaptation()
    active_learner = demonstrate_active_learning()
    continual_learner = demonstrate_continual_learning()
    memory_bank = demonstrate_memory_bank()
    
    # Demonstrate integrated system
    demonstrate_integrated_adaptive_learning()
    
    print("=== Demo Complete ===")
    print("Key Features Demonstrated:")
    print("✓ Meta-learning for rapid OOD adaptation")
    print("✓ Online adaptation from data streams")
    print("✓ Active learning for intelligent data acquisition")
    print("✓ Continual learning without catastrophic forgetting")
    print("✓ Memory bank for experience storage and retrieval")
    print("✓ Integrated adaptive learning pipeline")
    print("✓ Real-time adaptation to changing distributions")
    print("✓ Intelligent sample selection and querying")


if __name__ == "__main__":
    main()
