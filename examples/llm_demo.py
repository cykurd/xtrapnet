"""
LLM-Assisted Extrapolation Demo for XtrapNet v0.5.0

This script demonstrates the new LLM integration capabilities including:
- Natural language OOD explanations
- Intelligent decision making
- Uncertainty analysis in natural language
- Multi-modal OOD handling
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Import XtrapNet components
from xtrapnet import (
    LLMAssistant,
    OODExplainer,
    LLMDecisionMaker
)


def generate_ood_scenarios(n_scenarios=5):
    """Generate various OOD scenarios for demonstration."""
    scenarios = []
    
    # Scenario 1: Extrapolation beyond training range
    scenarios.append({
        'name': 'Extrapolation Beyond Range',
        'input': np.array([5.0, -3.5]),  # Way outside training range
        'prediction': np.array([2.3]),
        'uncertainty': {
            'epistemic_std': 0.8,
            'aleatoric_std': 0.2,
            'total_std': 0.9
        },
        'ood_score': 0.85,
        'ood_type': 'extrapolation',
        'detection_method': 'mahalanobis'
    })
    
    # Scenario 2: Domain shift
    scenarios.append({
        'name': 'Domain Shift',
        'input': np.array([1.2, 0.8]),  # Different domain
        'prediction': np.array([-0.5]),
        'uncertainty': {
            'epistemic_std': 0.6,
            'aleatoric_std': 0.4,
            'total_std': 0.7
        },
        'ood_score': 0.65,
        'ood_type': 'domain_shift',
        'detection_method': 'knn'
    })
    
    # Scenario 3: High uncertainty
    scenarios.append({
        'name': 'High Uncertainty',
        'input': np.array([0.5, 0.5]),  # Within range but uncertain
        'prediction': np.array([0.1]),
        'uncertainty': {
            'epistemic_std': 0.9,
            'aleatoric_std': 0.3,
            'total_std': 1.0
        },
        'ood_score': 0.45,
        'ood_type': 'interpolation_gap',
        'detection_method': 'uncertainty'
    })
    
    # Scenario 4: Novel class
    scenarios.append({
        'name': 'Novel Class',
        'input': np.array([-2.0, 2.0]),  # Novel pattern
        'prediction': np.array([1.8]),
        'uncertainty': {
            'epistemic_std': 0.7,
            'aleatoric_std': 0.1,
            'total_std': 0.8
        },
        'ood_score': 0.75,
        'ood_type': 'novel_class',
        'detection_method': 'isolation_forest'
    })
    
    # Scenario 5: Corrupted input
    scenarios.append({
        'name': 'Corrupted Input',
        'input': np.array([0.0, 0.0]),  # Suspicious pattern
        'prediction': np.array([0.0]),
        'uncertainty': {
            'epistemic_std': 0.5,
            'aleatoric_std': 0.6,
            'total_std': 0.8
        },
        'ood_score': 0.55,
        'ood_type': 'corrupted',
        'detection_method': 'autoencoder'
    })
    
    return scenarios[:n_scenarios]


def demonstrate_llm_assistant():
    """Demonstrate LLM Assistant capabilities."""
    print("=== LLM Assistant Demo ===")
    
    # Create LLM Assistant
    llm_assistant = LLMAssistant()
    
    # Test basic functionality
    print("1. Testing basic LLM generation...")
    response = llm_assistant.provider.generate("Explain what out-of-distribution means in machine learning.")
    print(f"Response: {response}")
    print()
    
    # Test OOD explanation
    print("2. Testing OOD explanation...")
    input_features = np.array([5.0, -3.5])
    prediction = np.array([2.3])
    uncertainty = {'epistemic_std': 0.8, 'aleatoric_std': 0.2, 'total_std': 0.9}
    confidence = 0.3
    
    explanation = llm_assistant.explain_ood_prediction(
        input_features, prediction, uncertainty, confidence
    )
    print(f"OOD Explanation: {explanation}")
    print()
    
    # Test uncertainty analysis
    print("3. Testing uncertainty analysis...")
    analysis = llm_assistant.analyze_uncertainty(
        epistemic_uncertainty=0.8,
        aleatoric_uncertainty=0.2,
        total_uncertainty=0.9,
        prediction=prediction
    )
    print(f"Uncertainty Analysis: {analysis}")
    print()
    
    # Test strategy recommendation
    print("4. Testing strategy recommendation...")
    available_strategies = ['clip', 'warn', 'error', 'backup']
    context = {'domain': 'finance', 'risk_tolerance': 'low'}
    
    strategy, reasoning = llm_assistant.recommend_strategy(
        input_features, available_strategies, context
    )
    print(f"Recommended Strategy: {strategy}")
    print(f"Reasoning: {reasoning}")
    print()
    
    return llm_assistant


def demonstrate_ood_explainer(llm_assistant):
    """Demonstrate OOD Explainer capabilities."""
    print("=== OOD Explainer Demo ===")
    
    # Create OOD Explainer
    ood_explainer = OODExplainer(llm_assistant)
    
    # Test OOD detection explanation
    print("1. Testing OOD detection explanation...")
    input_data = np.array([5.0, -3.5])
    ood_score = 0.85
    detection_method = 'mahalanobis'
    threshold = 0.5
    training_stats = {
        'mean': [0.0, 0.0],
        'std': [1.0, 1.0],
        'n_samples': 1000
    }
    
    detection_explanation = ood_explainer.explain_ood_detection(
        input_data, ood_score, detection_method, threshold, training_stats
    )
    print(f"Detection Explanation: {detection_explanation}")
    print()
    
    # Test OOD impact explanation
    print("2. Testing OOD impact explanation...")
    prediction = np.array([2.3])
    uncertainty = {'epistemic_std': 0.8, 'aleatoric_std': 0.2, 'total_std': 0.9}
    ood_type = 'extrapolation'
    confidence = 0.3
    
    impact_explanation = ood_explainer.explain_ood_impact(
        input_data, prediction, uncertainty, ood_type, confidence
    )
    print(f"Impact Explanation: {impact_explanation}")
    print()
    
    # Test uncertainty explanation in OOD context
    print("3. Testing uncertainty explanation in OOD context...")
    context = {'domain': 'finance', 'risk_tolerance': 'low'}
    
    uncertainty_explanation = ood_explainer.explain_uncertainty_in_ood(
        epistemic_uncertainty=0.8,
        aleatoric_uncertainty=0.2,
        total_uncertainty=0.9,
        ood_score=0.85,
        context=context
    )
    print(f"Uncertainty Explanation: {uncertainty_explanation}")
    print()
    
    # Test OOD handling recommendation
    print("4. Testing OOD handling recommendation...")
    available_strategies = ['clip', 'warn', 'error', 'backup', 'llm_assist']
    domain = 'finance'
    
    strategy, reasoning = ood_explainer.recommend_ood_handling(
        input_data, ood_score, uncertainty, available_strategies, domain
    )
    print(f"Recommended Strategy: {strategy}")
    print(f"Reasoning: {reasoning}")
    print()
    
    return ood_explainer


def demonstrate_decision_maker(llm_assistant):
    """Demonstrate LLM Decision Maker capabilities."""
    print("=== LLM Decision Maker Demo ===")
    
    # Create Decision Maker
    decision_maker = LLMDecisionMaker(llm_assistant)
    
    # Test individual decision making
    print("1. Testing individual decision making...")
    input_data = np.array([5.0, -3.5])
    prediction = np.array([2.3])
    uncertainty = {'epistemic_std': 0.8, 'aleatoric_std': 0.2, 'total_std': 0.9}
    ood_score = 0.85
    context = {
        'domain': 'finance',
        'risk_tolerance': 'low',
        'performance_requirements': 'high',
        'computational_cost': 'medium'
    }
    
    decision = decision_maker.make_decision(
        input_data, prediction, uncertainty, ood_score, context
    )
    
    print(f"Decision: {decision['strategy']}")
    print(f"Confidence: {decision['confidence']:.3f}")
    print(f"Validated: {decision['validated']}")
    print(f"Reasoning: {decision['reasoning']}")
    print()
    
    # Test batch decision making
    print("2. Testing batch decision making...")
    scenarios = generate_ood_scenarios(3)
    
    inputs = []
    for scenario in scenarios:
        inputs.append({
            'input_data': scenario['input'],
            'prediction': scenario['prediction'],
            'uncertainty': scenario['uncertainty'],
            'ood_score': scenario['ood_score']
        })
    
    decisions = decision_maker.batch_decision_making(inputs, context)
    
    print("Batch Decisions:")
    for i, decision in enumerate(decisions):
        print(f"  Scenario {i+1}: {decision['strategy']} (confidence: {decision['confidence']:.3f})")
    print()
    
    # Test decision statistics
    print("3. Testing decision statistics...")
    stats = decision_maker.get_decision_statistics(decisions)
    print(f"Decision Statistics: {stats}")
    print()
    
    # Test decision explanation
    print("4. Testing decision explanation...")
    explanation = decision_maker.explain_decision(decisions[0], context)
    print(f"Decision Explanation: {explanation}")
    print()
    
    return decision_maker


def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive LLM-assisted analysis."""
    print("=== Comprehensive LLM Analysis Demo ===")
    
    # Create all components
    llm_assistant = LLMAssistant()
    ood_explainer = OODExplainer(llm_assistant)
    decision_maker = LLMDecisionMaker(llm_assistant)
    
    # Generate scenarios
    scenarios = generate_ood_scenarios(3)
    
    # Analyze each scenario
    for i, scenario in enumerate(scenarios):
        print(f"--- Scenario {i+1}: {scenario['name']} ---")
        
        # OOD explanation
        detection_explanation = ood_explainer.explain_ood_detection(
            scenario['input'],
            scenario['ood_score'],
            scenario['detection_method'],
            0.5
        )
        print(f"Detection: {detection_explanation}")
        
        # Impact explanation
        impact_explanation = ood_explainer.explain_ood_impact(
            scenario['input'],
            scenario['prediction'],
            scenario['uncertainty'],
            scenario['ood_type'],
            0.3
        )
        print(f"Impact: {impact_explanation}")
        
        # Decision making
        context = {'domain': 'finance', 'risk_tolerance': 'medium'}
        decision = decision_maker.make_decision(
            scenario['input'],
            scenario['prediction'],
            scenario['uncertainty'],
            scenario['ood_score'],
            context
        )
        print(f"Decision: {decision['strategy']} (confidence: {decision['confidence']:.3f})")
        print()


def plot_ood_scenarios(scenarios):
    """Plot OOD scenarios for visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    inputs = [s['input'] for s in scenarios]
    ood_scores = [s['ood_score'] for s in scenarios]
    uncertainties = [s['uncertainty']['total_std'] for s in scenarios]
    names = [s['name'] for s in scenarios]
    
    # Plot 1: OOD scores
    axes[0, 0].bar(range(len(scenarios)), ood_scores, color='red', alpha=0.7)
    axes[0, 0].set_title('OOD Scores by Scenario')
    axes[0, 0].set_xlabel('Scenario')
    axes[0, 0].set_ylabel('OOD Score')
    axes[0, 0].set_xticks(range(len(scenarios)))
    axes[0, 0].set_xticklabels([f'S{i+1}' for i in range(len(scenarios))])
    
    # Plot 2: Uncertainty levels
    axes[0, 1].bar(range(len(scenarios)), uncertainties, color='blue', alpha=0.7)
    axes[0, 1].set_title('Uncertainty Levels by Scenario')
    axes[0, 1].set_xlabel('Scenario')
    axes[0, 1].set_ylabel('Total Uncertainty')
    axes[0, 1].set_xticks(range(len(scenarios)))
    axes[0, 1].set_xticklabels([f'S{i+1}' for i in range(len(scenarios))])
    
    # Plot 3: Input space (2D)
    if len(inputs) > 0 and len(inputs[0]) >= 2:
        x_coords = [inp[0] for inp in inputs]
        y_coords = [inp[1] for inp in inputs]
        
        scatter = axes[1, 0].scatter(x_coords, y_coords, c=ood_scores, 
                                    cmap='Reds', s=100, alpha=0.7)
        axes[1, 0].set_title('OOD Scenarios in Input Space')
        axes[1, 0].set_xlabel('Feature 1')
        axes[1, 0].set_ylabel('Feature 2')
        plt.colorbar(scatter, ax=axes[1, 0], label='OOD Score')
        
        # Add labels
        for i, name in enumerate(names):
            axes[1, 0].annotate(f'S{i+1}', (x_coords[i], y_coords[i]), 
                               xytext=(5, 5), textcoords='offset points')
    
    # Plot 4: Uncertainty vs OOD Score
    axes[1, 1].scatter(ood_scores, uncertainties, s=100, alpha=0.7)
    axes[1, 1].set_title('Uncertainty vs OOD Score')
    axes[1, 1].set_xlabel('OOD Score')
    axes[1, 1].set_ylabel('Total Uncertainty')
    
    # Add labels
    for i, name in enumerate(names):
        axes[1, 1].annotate(f'S{i+1}', (ood_scores[i], uncertainties[i]), 
                           xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration function."""
    print("XtrapNet v0.5.0: LLM-Assisted Extrapolation Demo")
    print("=" * 60)
    
    # Demonstrate LLM Assistant
    llm_assistant = demonstrate_llm_assistant()
    
    # Demonstrate OOD Explainer
    ood_explainer = demonstrate_ood_explainer(llm_assistant)
    
    # Demonstrate Decision Maker
    decision_maker = demonstrate_decision_maker(llm_assistant)
    
    # Demonstrate comprehensive analysis
    demonstrate_comprehensive_analysis()
    
    # Generate and plot scenarios
    scenarios = generate_ood_scenarios(5)
    plot_ood_scenarios(scenarios)
    
    print("=== Demo Complete ===")
    print("Key Features Demonstrated:")
    print("✓ LLM Assistant for natural language generation")
    print("✓ OOD Explainer for intelligent explanations")
    print("✓ LLM Decision Maker for strategy selection")
    print("✓ Comprehensive OOD analysis and reporting")
    print("✓ Multi-scenario decision making")
    print("✓ Natural language uncertainty analysis")
    print("✓ Context-aware decision making")


if __name__ == "__main__":
    main()
