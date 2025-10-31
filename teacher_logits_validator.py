"""
Validator utilities for teacher logits used in distillation.

Checks:
- Each provided .npy file exists and is loadable
- Shape is [N, 3]
- N matches the expected number of train rows after minimal preprocessing
  (dropna on text columns and label derivation just like load_dataset).

Usage (programmatic):
    from teacher_logits_validator import validate_logits
    info = validate_logits(["path/to/teacher1.npy", "path/to/teacher2.npy"], "./data/train.csv")
    # raises ValueError on mismatch, otherwise returns a summary dict
"""

from typing import List, Dict, Any
import os
import numpy as np
import pandas as pd

LABEL_MAP = {"model_a": 0, "model_b": 1, "tie": 2, "tie (both bad)": 2}


def _expected_train_length(train_csv: str) -> int:
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train_csv not found: {train_csv}")
    df = pd.read_csv(train_csv)
    text_cols = ["prompt", "response_a", "response_b"]
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in train.csv: {c}")

    # derive labels similar to student_train_distill_hf.load_dataset
    if 'winner' in df.columns:
        labels = df['winner'].map(LABEL_MAP)
    else:
        trio = None
        candidates = [
            ['winner_model_a', 'winner_model_b', 'winner_tie'],
            ['winner_model_a_prob', 'winner_model_b_prob', 'winner_tie_prob'],
        ]
        for cand in candidates:
            if all(c in df.columns for c in cand):
                trio = cand
                break
        if trio is None:
            raise ValueError("Could not find labels: expected 'winner' or probability columns")
        labels = df[trio].astype(float).values.argmax(axis=1)

    df = df.dropna(subset=text_cols).copy()
    df['label'] = labels
    df = df.dropna(subset=['label'])
    return int(len(df))


def validate_logits(teacher_paths: List[str], train_csv_path: str, num_classes: int = 3) -> Dict[str, Any]:
    """
    Validate that each teacher logits file matches the expected length (N) and class count.

    Returns a summary dict with expected_length and per-file shapes. Raises ValueError on mismatch.
    """
    issues: List[str] = []
    details: Dict[str, Any] = {}

    if not teacher_paths:
        raise ValueError("No teacher paths provided")

    expected_len = _expected_train_length(train_csv_path)

    for p in teacher_paths:
        if not os.path.exists(p):
            issues.append(f"Missing file: {p}")
            continue
        try:
            arr = np.load(p)
        except Exception as e:
            issues.append(f"Failed to load {p}: {e}")
            continue
        details[p] = {'shape': tuple(arr.shape)}
        if arr.ndim != 2:
            issues.append(f"{p}: expected 2D array, got {arr.ndim}D with shape {arr.shape}")
            continue
        n, c = arr.shape
        if c != num_classes:
            issues.append(f"{p}: expected num_classes={num_classes}, got {c}")
        if n != expected_len:
            issues.append(f"{p}: expected length N={expected_len}, got {n}")

    if issues:
        raise ValueError("Teacher logits validation failed:\n- " + "\n- ".join(issues))

    return {
        'ok': True,
        'expected_length': expected_len,
        'files': details,
    }
