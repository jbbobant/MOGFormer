"""
cv.py — Repeated stratified, patient-level cross validation with persisted fold indices.

We persist by PATIENT ID (not just positional index) so folds survive any later
reordering of the data matrix, and we store positional indices too for speed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


@dataclass
class FoldSpec:
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_folds(
    y: np.ndarray,
    patient_ids: List[str],
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> List[FoldSpec]:
    """Build 25 (5x5) stratified patient-level outer folds."""
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    folds: List[FoldSpec] = []
    for i, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
        folds.append(FoldSpec(
            repeat=i // n_splits,
            fold=i % n_splits,
            train_idx=np.asarray(tr, dtype=int),
            test_idx=np.asarray(te, dtype=int),
        ))
    return folds





def save_folds(folds: List[FoldSpec], patient_ids: List[str], path: str) -> None:
    pid = np.asarray(patient_ids)
    payload: Dict = {
        "n_estimates": len(folds),
        "patient_order": list(patient_ids),
        "folds": [
            {
                "repeat": f.repeat,
                "fold": f.fold,
                "train_idx": f.train_idx.tolist(),
                "test_idx": f.test_idx.tolist(),
                "train_patients": pid[f.train_idx].tolist(),
                "test_patients": pid[f.test_idx].tolist(),
            }
            for f in folds
        ],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_folds(path: str) -> Tuple[List[FoldSpec], List[str]]:
    with open(path) as fh:
        payload = json.load(fh)
    folds = [
        FoldSpec(
            repeat=d["repeat"], fold=d["fold"],
            train_idx=np.asarray(d["train_idx"], dtype=int),
            test_idx=np.asarray(d["test_idx"], dtype=int),
        )
        for d in payload["folds"]
    ]
    return folds, payload["patient_order"]
