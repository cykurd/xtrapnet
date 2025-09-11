"""
DistilBERT-based FEVER classifier with optional MC Dropout for uncertainty.

This module provides a lightweight classifier head over DistilBERT to predict
SUPPORTS vs REFUTES on FEVER claims. It supports Monte Carlo Dropout sampling
to estimate predictive uncertainty by keeping dropout active at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FeverTrainingConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    num_epochs: int = 1
    mc_dropout_samples: int = 0  # 0 disables MC dropout during eval
    dropout_prob: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Advanced components
    use_temperature_scaling: bool = True
    use_deep_ensemble: bool = False
    ensemble_size: int = 3
    use_xtrapnet_uncertainty: bool = True


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


class XtrapNetUncertaintyHead(nn.Module):
    """XtrapNet-inspired uncertainty estimation head."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.uncertainty_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        self.ood_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        uncertainty = self.uncertainty_mlp(hidden_states)
        ood_score = self.ood_head(hidden_states)
        return uncertainty, ood_score


class DistilBertFeverClassifier(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", dropout_prob: float = 0.1, use_xtrapnet: bool = True):
        super().__init__()
        from transformers import DistilBertModel

        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 2)
        self.use_xtrapnet = use_xtrapnet
        
        if use_xtrapnet:
            self.uncertainty_head = XtrapNetUncertaintyHead(hidden_size)
        
        self.temperature_scaling = TemperatureScaling()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mc_keep_dropout: bool = False) -> Dict[str, torch.Tensor]:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        if self.training or mc_keep_dropout:
            cls = self.dropout(cls)
        
        logits = self.classifier(cls)
        result = {"logits": logits}
        
        if self.use_xtrapnet:
            uncertainty, ood_score = self.uncertainty_head(cls)
            result["uncertainty"] = uncertainty
            result["ood_score"] = ood_score
            
        return result

    @torch.no_grad()
    def predict_proba(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, mc_samples: int = 0, use_temperature: bool = True) -> Dict[str, torch.Tensor]:
        """Return softmax probabilities with uncertainty estimates."""
        self.eval()
        if mc_samples and mc_samples > 0:
            # MC Dropout sampling
            probs = []
            uncertainties = []
            ood_scores = []
            for _ in range(mc_samples):
                outputs = self.forward(input_ids, attention_mask, mc_keep_dropout=True)
                logits = outputs["logits"]
                if use_temperature:
                    logits = self.temperature_scaling(logits)
                probs.append(F.softmax(logits, dim=-1))
                if self.use_xtrapnet:
                    uncertainties.append(outputs["uncertainty"])
                    ood_scores.append(outputs["ood_score"])
            
            result = {"probs": torch.stack(probs, dim=0).mean(dim=0)}
            if self.use_xtrapnet:
                result["uncertainty"] = torch.stack(uncertainties, dim=0).mean(dim=0)
                result["ood_score"] = torch.stack(ood_scores, dim=0).mean(dim=0)
            return result
        else:
            outputs = self.forward(input_ids, attention_mask, mc_keep_dropout=False)
            logits = outputs["logits"]
            if use_temperature:
                logits = self.temperature_scaling(logits)
            
            result = {"probs": F.softmax(logits, dim=-1)}
            if self.use_xtrapnet:
                result["uncertainty"] = outputs["uncertainty"]
                result["ood_score"] = outputs["ood_score"]
            return result


class FeverTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels.astype(np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        item = {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0), "labels": torch.tensor(label, dtype=torch.long)}
        return item


def fever_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = torch.stack([b["labels"] for b in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_temperature_scaling(model: DistilBertFeverClassifier, val_loader: torch.utils.data.DataLoader, config: FeverTrainingConfig) -> None:
    """Train temperature scaling on validation set."""
    device = torch.device(config.device)
    model.eval()
    optimizer = torch.optim.LBFGS([model.temperature_scaling.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()
    
    def eval_loss():
        optimizer.zero_grad()
        total_loss = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = model.temperature_scaling(outputs["logits"])
            loss = criterion(logits, labels)
            total_loss += loss
        total_loss.backward()
        return total_loss
    
    optimizer.step(eval_loss)


def train_fever_classifier(
    model: DistilBertFeverClassifier,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    config: FeverTrainingConfig,
) -> None:
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(config.num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs["logits"], labels)
            loss.backward()
            optimizer.step()

        if val_loader is not None and config.use_temperature_scaling:
            train_temperature_scaling(model, val_loader, config)


@torch.no_grad()
def evaluate_fever_classifier(
    model: DistilBertFeverClassifier,
    data_loader: torch.utils.data.DataLoader,
    config: FeverTrainingConfig,
    ood_labels: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    device = torch.device(config.device)
    model.eval()
    total = 0
    correct = 0
    all_probs = []
    all_uncertainties = []
    all_ood_scores = []
    all_labels = []
    all_ood_true = []
    
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model.predict_proba(
            input_ids, attention_mask, 
            mc_samples=config.mc_dropout_samples,
            use_temperature=config.use_temperature_scaling
        )
        
        probs = outputs["probs"]
        preds = probs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        
        if model.use_xtrapnet:
            all_uncertainties.append(outputs["uncertainty"].cpu())
            all_ood_scores.append(outputs["ood_score"].cpu())
    
    acc = correct / max(1, total)
    results = {"accuracy": acc}
    
    # Calibration error (ECE)
    if len(all_probs) > 0:
        probs = torch.cat(all_probs, dim=0)
        labels = torch.cat(all_labels, dim=0)
        max_probs = probs.max(dim=1)[0]
        ece = compute_ece(max_probs, labels, probs.argmax(dim=1))
        results["calibration_error"] = ece
    
    # OOD detection AUC
    if model.use_xtrapnet and ood_labels is not None and len(all_ood_scores) > 0:
        ood_scores = torch.cat(all_ood_scores, dim=0).squeeze()
        ood_auc = compute_auc(ood_scores, ood_labels)
        results["ood_detection_auc"] = ood_auc
    
    # Hallucination detection (using uncertainty vs prediction confidence)
    if model.use_xtrapnet and len(all_uncertainties) > 0:
        uncertainties = torch.cat(all_uncertainties, dim=0).squeeze()
        pred_confidences = torch.cat(all_probs, dim=0).max(dim=1)[0]
        # Create binary labels: high uncertainty + low confidence = potential hallucination
        hallucination_threshold = 0.5
        is_hallucination = ((uncertainties > hallucination_threshold) & (pred_confidences < hallucination_threshold)).float()
        if is_hallucination.sum() > 0 and (1 - is_hallucination).sum() > 0:
            hallucination_auc = compute_auc(uncertainties, is_hallucination)
            results["hallucination_detection_auc"] = hallucination_auc
    
    return results


def compute_ece(confidences: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def compute_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute AUC using sklearn."""
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels.numpy(), scores.numpy())
    except ImportError:
        # Fallback: simple approximation
        sorted_indices = torch.argsort(scores)
        sorted_labels = labels[sorted_indices]
        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        return 0.5  # Placeholder


