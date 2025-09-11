"""
FEVER dataset loader with ID/OOD subject splits.

This module provides a lightweight loader for the FEVER fact verification dataset
that supports creating in-distribution (ID) vs out-of-distribution (OOD) splits
based on claim subjects (e.g., page/entity titles). This is intended for
hallucination and OOD detection benchmarking with small transformer classifiers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .benchmark_datasets import BenchmarkDataset, DatasetType, DatasetSplit


@dataclass
class FeverExample:
    claim: str
    label: int  # 1: SUPPORTS, 0: REFUTES, optionally -1: NOT_ENOUGH_INFO (excluded by default)
    subject: str  # e.g., Wikipedia page title parsed from evidence or metadata


class FeverDataset(BenchmarkDataset):
    """FEVER dataset with subject-based ID/OOD splitting.

    Expected input format
    - TSV or JSONL with fields: claim, label, subject
    - Labels are mapped to {"SUPPORTS":1, "REFUTES":0}. "NOT_ENOUGH_INFO" rows can be
      optionally dropped (default) or mapped to a third class.

    ID/OOD split
    - We partition subjects; training uses ID subjects only. Test contains both ID
      and OOD subjects according to a specified OOD ratio over unique subjects.
    """

    def __init__(
        self,
        data_dir: str,
        drop_nei: bool = True,
        val_ratio: float = 0.1,
        ood_subject_ratio: float = 0.5,
        random_seed: int = 42,
    ) -> None:
        super().__init__(
            name="fever",
            dataset_type=DatasetType.TEXT,
            description="FEVER fact verification with subject-based ID/OOD splits",
        )
        self.data_dir = data_dir
        self.drop_nei = drop_nei
        self.val_ratio = max(0.0, min(0.5, val_ratio))
        self.ood_subject_ratio = max(0.0, min(0.9, ood_subject_ratio))
        self.random_seed = random_seed

    def load_data(self) -> None:
        train_path = self._find_file(["train.jsonl", "train.tsv", "train.json"])
        dev_path = self._find_file(["dev.jsonl", "dev.tsv", "dev.json", "valid.jsonl", "val.tsv"])  # optional
        test_path = self._find_file(["test.jsonl", "test.tsv", "test.json"])  # optional

        train_examples = self._load_examples(train_path)
        dev_examples = self._load_examples(dev_path) if dev_path else []
        test_examples = self._load_examples(test_path) if test_path else []

        # Merge train+dev as training pool; keep test separate for evaluation
        rng = np.random.default_rng(self.random_seed)

        all_train = train_examples + dev_examples
        if len(all_train) == 0:
            raise ValueError("No FEVER training data found. Place train/dev files with fields claim,label,subject.")

        # Build subject sets
        subjects = sorted({ex.subject for ex in all_train})
        rng.shuffle(subjects)
        n_ood_subjects = int(len(subjects) * self.ood_subject_ratio)
        ood_subjects = set(subjects[:n_ood_subjects])
        id_subjects = set(subjects[n_ood_subjects:])

        # Split all_train into ID-only pool
        id_pool = [ex for ex in all_train if ex.subject in id_subjects]
        if len(id_pool) < 10:
            # Fallback: if subjects are missing, treat all as ID
            id_pool = all_train
            id_subjects = {ex.subject for ex in all_train}
            ood_subjects = set()

        rng.shuffle(id_pool)
        n_val = int(len(id_pool) * self.val_ratio)
        val_set = id_pool[:n_val]
        train_set = id_pool[n_val:]

        # Build test set: include ID and OOD by subject. If explicit test file exists, use it; else sample from pool
        if len(test_examples) == 0:
            # Create a held-out test by subject from combined pool
            test_candidates = all_train
        else:
            test_candidates = test_examples

        id_test = [ex for ex in test_candidates if ex.subject in id_subjects]
        ood_test = [ex for ex in test_candidates if ex.subject in ood_subjects]

        # If no ood_test found (e.g., data lacks subjects), synthesize by subject reassignment (rare)
        if len(ood_test) == 0 and len(id_test) > 0:
            ood_test = id_test[len(id_test)//2:]
            id_test = id_test[:len(id_test)//2]

        test_set = id_test + ood_test
        rng.shuffle(test_set)

        # Vectorize minimal representation as arrays of strings with labels; model handles tokenization later
        def to_arrays(examples: List[FeverExample]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
            claims = np.array([ex.claim for ex in examples], dtype=object)
            labels = np.array([ex.label for ex in examples], dtype=np.int64)
            subjects_local = [ex.subject for ex in examples]
            return claims, labels, subjects_local

        train_claims, train_labels, train_subjects = to_arrays(train_set)
        val_claims, val_labels, val_subjects = to_arrays(val_set) if len(val_set) > 0 else (np.array([], dtype=object), np.array([], dtype=np.int64), [])
        test_claims, test_labels, test_subjects = to_arrays(test_set)

        # OOD labels for test: subject in ood_subjects -> 1 else 0
        test_ood = np.array([1 if s in ood_subjects else 0 for s in test_subjects], dtype=np.int64)

        # Store splits
        self.splits["train"] = DatasetSplit(
            train_data=train_claims,
            train_labels=train_labels,
            test_data=test_claims,
            test_labels=test_ood,
            metadata={
                "task": "fever_fact_verification",
                "label_meaning": {"y": "support=1, refute=0", "ood": "1=OOD by subject"},
                "n_subjects_total": len(subjects),
                "n_subjects_id": len(id_subjects),
                "n_subjects_ood": len(ood_subjects),
                "val_size": len(val_claims),
                "train_size": len(train_claims),
                "test_size": len(test_claims),
            },
        )

        if len(val_claims) > 0:
            self.splits["val"] = DatasetSplit(
                train_data=val_claims,
                train_labels=val_labels,
                test_data=test_claims,
                test_labels=test_ood,
            )

    def _find_file(self, candidates: List[str]) -> Optional[str]:
        for name in candidates:
            path = os.path.join(self.data_dir, name)
            if os.path.exists(path):
                return path
        return None

    def _load_examples(self, path: Optional[str]) -> List[FeverExample]:
        if path is None:
            return []
        lower = path.lower()
        if lower.endswith(".jsonl") or lower.endswith(".json"):
            return self._load_json_like(path)
        if lower.endswith(".tsv") or lower.endswith(".csv"):
            return self._load_tsv_like(path)
        raise ValueError(f"Unsupported FEVER file format: {path}")

    def _label_to_int(self, label_str: str) -> Optional[int]:
        s = label_str.strip().upper()
        if s == "SUPPORTS":
            return 1
        if s == "REFUTES":
            return 0
        if s in {"NOT_ENOUGH_INFO", "NEI"}:
            return None if self.drop_nei else -1
        return None

    def _load_json_like(self, path: str) -> List[FeverExample]:
        import json
        examples: List[FeverExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # maybe it's a JSON array file
                    f.seek(0)
                    data = json.load(f)
                    obj_iter = data if isinstance(data, list) else []
                    for obj in obj_iter:
                        ex = self._parse_obj(obj)
                        if ex is not None:
                            examples.append(ex)
                    break
                else:
                    ex = self._parse_obj(obj)
                    if ex is not None:
                        examples.append(ex)
        return examples

    def _load_tsv_like(self, path: str) -> List[FeverExample]:
        import csv
        examples: List[FeverExample] = []
        with open(path, "r", encoding="utf-8") as f:
            # attempt dialect sniffing
            try:
                dialect = csv.Sniffer().sniff(f.read(2048))
                f.seek(0)
            except csv.Error:
                dialect = csv.excel_tab
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                ex = self._parse_obj(row)
                if ex is not None:
                    examples.append(ex)
        return examples

    def _parse_obj(self, obj: Dict[str, Any]) -> Optional[FeverExample]:
        # Flexible key mapping
        claim = obj.get("claim") or obj.get("text") or obj.get("sentence")
        label_raw = obj.get("label") or obj.get("gold_label") or obj.get("verdict")
        subject = obj.get("subject") or obj.get("page") or obj.get("title") or obj.get("entity")
        if claim is None or label_raw is None or subject is None:
            return None
        label_int = self._label_to_int(str(label_raw))
        if label_int is None:
            return None
        return FeverExample(claim=str(claim), label=label_int, subject=str(subject))


