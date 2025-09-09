"""
Simple LLM Demo for XtrapNet v0.5.0

This script demonstrates the LLM integration capabilities without matplotlib.
"""

import numpy as np
from typing import Dict, List, Any

# Import XtrapNet components
from xtrapnet import (
    LLMAssistant,
    OODExplainer,
    LLMDecisionMaker
)


def generate_ood_scenarios(n_scenarios=3):
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
    
    return scenarios[:n_scenarios]


def demonstrate_llm_assistant():
    """Demonstrate LLM Assistant capabilities."""
    print("=== LLM Assistant Demo ===")
    
    # Create LLM Assistant
    llm_assistant = LLMAssistant()
    
    # Test basic functionality
    print("1. Testing basic LLM generation...")
    response = llm_assistant.provider.generate('What is out-of-distribution data in machine learning?')
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
