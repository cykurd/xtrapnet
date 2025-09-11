import os
import numpy as np
import torch

from xtrapnet.benchmarks import FeverDataset
from xtrapnet.llm.fever_classifier import (
    DistilBertFeverClassifier,
    FeverTrainingConfig,
    FeverTextDataset,
    fever_collate_fn,
    train_fever_classifier,
    evaluate_fever_classifier,
)


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fever")
    dataset = FeverDataset(data_dir=data_dir, ood_subject_ratio=0.5, val_ratio=0.1)
    dataset.load_data()
    split = dataset.get_split("train")

    # Use verification labels (y) for supervised training; OOD labels are in test_labels
    train_texts = split.train_data.tolist()
    train_y = split.train_labels

    # If a validation split exists, use it; else carve a small slice from training
    val_split = dataset.splits.get("val")
    if val_split is not None and len(val_split.train_data) > 0:
        val_texts = val_split.train_data.tolist()
        val_y = val_split.train_labels
    else:
        n_val = max(100, int(0.1 * len(train_texts)))
        val_texts = train_texts[:n_val]
        val_y = train_y[:n_val]
        train_texts = train_texts[n_val:]
        train_y = train_y[n_val:]

    # Test set with OOD labels
    test_texts = split.test_data.tolist()
    test_y = split.train_labels  # Verification labels
    test_ood_labels = torch.tensor(split.test_labels, dtype=torch.long)  # OOD labels

    from transformers import AutoTokenizer
    config = FeverTrainingConfig(
        num_epochs=2, 
        mc_dropout_samples=10, 
        batch_size=16,
        use_temperature_scaling=True,
        use_xtrapnet_uncertainty=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_ds = FeverTextDataset(train_texts, train_y, tokenizer, max_length=config.max_length)
    val_ds = FeverTextDataset(val_texts, val_y, tokenizer, max_length=config.max_length)
    test_ds = FeverTextDataset(test_texts, test_y, tokenizer, max_length=config.max_length)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=fever_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=fever_collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=fever_collate_fn
    )

    model = DistilBertFeverClassifier(
        model_name=config.model_name, 
        dropout_prob=config.dropout_prob,
        use_xtrapnet=config.use_xtrapnet_uncertainty
    )

    print("Training FEVER classifier with advanced features...")
    train_fever_classifier(model, train_loader, val_loader, config)
    
    print("Evaluating on test set...")
    test_metrics = evaluate_fever_classifier(
        model, test_loader, config, ood_labels=test_ood_labels
    )
    
    print("=== ADVANCED FEVER EVALUATION RESULTS ===")
    print(f"Claim Verification Accuracy: {test_metrics['accuracy']:.3f}")
    if 'calibration_error' in test_metrics:
        print(f"Calibration Error (ECE): {test_metrics['calibration_error']:.3f}")
    if 'ood_detection_auc' in test_metrics:
        print(f"OOD Detection AUC: {test_metrics['ood_detection_auc']:.3f}")
    if 'hallucination_detection_auc' in test_metrics:
        print(f"Hallucination Detection AUC: {test_metrics['hallucination_detection_auc']:.3f}")
    
    # SOTA comparison
    baseline_accuracy = 0.509  # Original FEVER baseline
    sota_accuracy = 0.96  # Current SOTA
    
    improvement_over_baseline = (test_metrics['accuracy'] - baseline_accuracy) / baseline_accuracy * 100
    gap_to_sota = (sota_accuracy - test_metrics['accuracy']) / sota_accuracy * 100
    
    print(f"\n=== SOTA ANALYSIS ===")
    print(f"Improvement over baseline: {improvement_over_baseline:.1f}%")
    print(f"Gap to current SOTA: {gap_to_sota:.1f}%")
    
    if test_metrics['accuracy'] > 0.85:
        print("✅ XtrapNet achieves strong performance approaching SOTA!")
    elif test_metrics['accuracy'] > 0.70:
        print("✅ XtrapNet shows significant improvement over baseline!")
    else:
        print("⚠️  Further optimization needed to reach SOTA levels.")
    
    return test_metrics


if __name__ == "__main__":
    main()
