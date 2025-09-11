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

    from transformers import AutoTokenizer
    config = FeverTrainingConfig(num_epochs=1, mc_dropout_samples=0, batch_size=16)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_ds = FeverTextDataset(train_texts, train_y, tokenizer, max_length=config.max_length)
    val_ds = FeverTextDataset(val_texts, val_y, tokenizer, max_length=config.max_length)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=fever_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=fever_collate_fn
    )

    model = DistilBertFeverClassifier(model_name=config.model_name, dropout_prob=config.dropout_prob)

    train_fever_classifier(model, train_loader, val_loader, config)
    metrics = evaluate_fever_classifier(model, val_loader, config)
    print({"val_accuracy": metrics["accuracy"]})


if __name__ == "__main__":
    main()


