# student_calibrate.py
"""
Temperature scaling for the trained student classifier.
- Splits train.csv into train/val (same as student_train_hf), loads the trained model,
  computes logits on validation, fits a temperature to minimize NLL (log loss),
  saves calibration to ./model_save/student_distilbert/calibration.json.
"""

import json
import os
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import log_loss

LABEL_MAP = {"model_a": 0, "model_b": 1, "tie": 2, "tie (both bad)": 2}


def build_input_text(row: pd.Series) -> str:
    return f"[PROMPT]{str(row['prompt']).strip()}[RESPONSE_A]{str(row['response_a']).strip()}[RESPONSE_B]{str(row['response_b']).strip()}"


def load_dataset_for_cal(train_csv: str, max_samples: int = 4000) -> Dataset:
    df = pd.read_csv(train_csv)
    text_cols = ['prompt', 'response_a', 'response_b']
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in train.csv: {c}")

    if 'winner' in df.columns:
        labels = df['winner'].map(LABEL_MAP)
    else:
        candidates = [
            ['winner_model_a', 'winner_model_b', 'winner_tie'],
            ['winner_model_a_prob', 'winner_model_b_prob', 'winner_tie_prob'],
        ]
        probs = None
        for trio in candidates:
            if all(c in df.columns for c in trio):
                probs = df[trio].astype(float)
                break
        if probs is None:
            raise ValueError("Could not find labels: expected 'winner' or probabilities columns.")
        labels = probs.values.argmax(axis=1)

    df = df.dropna(subset=text_cols).copy()
    df['text'] = df.apply(build_input_text, axis=1)
    df['label'] = labels

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    return Dataset.from_pandas(df[['text', 'label']])


def compute_nll(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> float:
    labels = labels.astype(int)
    scaled = logits / float(T)
    m = scaled.max(axis=1, keepdims=True)
    log_probs = scaled - (m + np.log(np.sum(np.exp(scaled - m), axis=1, keepdims=True)))
    nll = -np.mean([log_probs[i, labels[i]] for i in range(len(labels))])
    return float(nll)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    # Fit temperature by minimizing NLL with simple line search on log T
    # Works well enough and is stable.
    labels = labels.astype(int)
    best_T = 1.0
    best_loss = 1e9
    grid = np.linspace(-2.0, 2.0, 81)  # logT in [-2, 2] => T in [0.135, 7.39]
    for logT in grid:
        T = float(np.exp(logT))
        # stable NLL via helper
        nll = compute_nll(logits, labels, T)
        if nll < best_loss:
            best_loss = nll
            best_T = T
    return best_T


def calibrate_student(
    model_dir: str = './model_save/student_distilbert',
    train_csv: str = './data/train.csv',
    output_json: Optional[str] = None,
    batch_size: int = 16,
    max_length: int = 512,
) -> Dict:
    # Default save path: alongside the provided student model directory
    if not output_json:
        output_json = os.path.join(model_dir, 'calibration.json')
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    dataset = load_dataset_for_cal(train_csv)
    # First split: create a validation pool (20% of data)
    split1 = dataset.train_test_split(test_size=0.2, seed=42)
    val_pool = split1['test']
    # Second split: split validation pool into calibration and holdout (50/50)
    split2 = val_pool.train_test_split(test_size=0.5, seed=123)
    cal_ds = split2['train']
    holdout_ds = split2['test']

    # Robust tokenizer resolution: prefer local tokenizer in model_dir; fall back to env/base model
    def _has_tok_files(p: str) -> bool:
        try:
            if not os.path.isdir(p):
                return False
            names = set(os.listdir(p))
            want = {"tokenizer.model", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
            return len(want.intersection(names)) > 0
        except Exception:
            return False

    tok_candidates: List[str] = []
    # 1) local directory if tokenizer artifacts exist
    if _has_tok_files(model_dir):
        tok_candidates.append(model_dir)
    # 2) explicit env overrides
    env_tok = os.environ.get('TOKENIZER_DIR')
    if env_tok:
        tok_candidates.append(env_tok)
    env_base = os.environ.get('BASE_MODEL') or os.environ.get('FALLBACK_BASE_MODEL_DIR')
    if env_base:
        tok_candidates.append(env_base)
    # 3) sensible default for this project
    tok_candidates.append('google/gemma-2-9b-it')

    tokenizer = None
    last_err: Optional[Exception] = None
    for cand in tok_candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(cand, use_fast=True, trust_remote_code=True)
            break
        except Exception as e1:
            last_err = e1
            try:
                tokenizer = AutoTokenizer.from_pretrained(cand, use_fast=False, trust_remote_code=True)
                break
            except Exception as e2:
                last_err = e2
                continue
    if tokenizer is None:
        raise RuntimeError(f"Failed to load tokenizer. Tried candidates: {tok_candidates}. Last error: {last_err}")
    # Resolve a valid local model directory (the root or a checkpoint subdir) with config and weights
    def _has_weights(names: set) -> bool:
        if any(n in names for n in (
            'pytorch_model.bin', 'model.safetensors', 'pytorch_model.bin.index.json', 'model.safetensors.index.json'
        )):
            return True
        if any(n.startswith('pytorch_model-') and n.endswith('.bin') for n in names):
            return True
        if any(n.startswith('model-') and n.endswith('.safetensors') for n in names):
            return True
        return False

    def _pick_local_model_dir(root: str) -> str:
        # If root has config and weights, use it
        if os.path.isdir(root):
            names = set(os.listdir(root))
            if 'config.json' in names and _has_weights(names):
                return root
        # Otherwise scan common checkpoint subdirs under root
        candidates: list[tuple[float, str]] = []
        try:
            for entry in os.scandir(root):
                if not entry.is_dir():
                    continue
                sub = entry.path
                try:
                    sub_names = set(os.listdir(sub))
                except Exception:
                    continue
                if 'config.json' in sub_names and _has_weights(sub_names):
                    try:
                        mtime = os.path.getmtime(sub)
                    except Exception:
                        mtime = 0.0
                    candidates.append((mtime, sub))
        except Exception:
            pass
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        return ''

    def _is_valid_model_dir(path: str) -> bool:
        try:
            if not os.path.isdir(path):
                return False
            names = set(os.listdir(path))
            if 'config.json' not in names:
                return False
            return _has_weights(names)
        except Exception:
            return False

    # Resolve candidate model dirs in order of preference
    candidates: List[str] = []
    # 1) Given model_dir (root if valid, else latest checkpoint under it)
    if os.path.isdir(model_dir):
        if _is_valid_model_dir(model_dir):
            candidates.append(model_dir)
        else:
            chosen = _pick_local_model_dir(model_dir)
            if chosen:
                candidates.append(chosen)
    # 2) Env override
    env_student = os.environ.get('STUDENT_MODEL_DIR')
    if env_student and _is_valid_model_dir(env_student):
        candidates.append(env_student)
    # 3) Default stable location
    default_dir = os.path.join('.', 'model_save', 'student_distilbert')
    if _is_valid_model_dir(default_dir):
        candidates.append(default_dir)
    # 4) Parent scan: look for a child named 'student_distilbert' next to the given dir
    try:
        parent = os.path.dirname(os.path.abspath(model_dir))
        sibling = os.path.join(parent, 'student_distilbert')
        if _is_valid_model_dir(sibling):
            candidates.append(sibling)
    except Exception:
        pass

    # Dedup while preserving order
    seen = set()
    candidates = [c for c in candidates if c and (c not in seen and not seen.add(c))]

    load_dir = None
    for cand in candidates:
        load_dir = cand
        break
    if load_dir is None:
        # last resort: use given path with local_files_only to yield a clear message
        load_dir = model_dir
    # Try local-only first to avoid treating local paths as HF repo ids
    model = None
    last_err = None
    for local_only in (True, False):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                load_dir,
                local_files_only=local_only,
                trust_remote_code=True,
            )
            break
        except Exception as e:
            last_err = e
            continue
    if model is None:
        raise RuntimeError(
            f"Failed to load student model. Tried candidates={candidates or [model_dir]}. Last error: {last_err}"
        )
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def compute_logits(ds):
        texts = ds['text']
        labels = np.array(ds['label'])
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tokenizer(texts[i:i+batch_size], truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits.cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits, labels

    cal_logits, cal_labels = compute_logits(cal_ds)
    hold_logits, hold_labels = compute_logits(holdout_ds)

    # Fit temperature on calibration split only
    T = fit_temperature(cal_logits, cal_labels)

    # Metrics on calibration split (NLL and Kaggle log loss)
    cal_before = compute_nll(cal_logits, cal_labels, T=1.0)
    cal_after = compute_nll(cal_logits, cal_labels, T=T)
    cal_probs_before = np.exp(cal_logits - cal_logits.max(axis=1, keepdims=True))
    cal_probs_before = cal_probs_before / cal_probs_before.sum(axis=1, keepdims=True)
    cal_probs_after = np.exp(cal_logits / T - (cal_logits / T).max(axis=1, keepdims=True))
    cal_probs_after = cal_probs_after / cal_probs_after.sum(axis=1, keepdims=True)
    cal_ll_before = float(log_loss(cal_labels, cal_probs_before, labels=[0, 1, 2]))
    cal_ll_after = float(log_loss(cal_labels, cal_probs_after, labels=[0, 1, 2]))

    # Metrics on holdout split (reporting set)
    hold_before = compute_nll(hold_logits, hold_labels, T=1.0)
    hold_after = compute_nll(hold_logits, hold_labels, T=T)
    hold_probs_before = np.exp(hold_logits - hold_logits.max(axis=1, keepdims=True))
    hold_probs_before = hold_probs_before / hold_probs_before.sum(axis=1, keepdims=True)
    hold_probs_after = np.exp(hold_logits / T - (hold_logits / T).max(axis=1, keepdims=True))
    hold_probs_after = hold_probs_after / hold_probs_after.sum(axis=1, keepdims=True)
    hold_ll_before = float(log_loss(hold_labels, hold_probs_before, labels=[0, 1, 2]))
    hold_ll_after = float(log_loss(hold_labels, hold_probs_after, labels=[0, 1, 2]))

    payload = {
        'temperature': float(T),
        'cal_size': int(len(cal_labels)),
        'holdout_size': int(len(hold_labels)),
        'nll_cal_before': float(cal_before),
        'nll_cal_after': float(cal_after),
        'nll_cal_improvement': float(cal_before - cal_after),
        'nll_holdout_before': float(hold_before),
        'nll_holdout_after': float(hold_after),
        'nll_holdout_improvement': float(hold_before - hold_after),
        'logloss_cal_before': float(cal_ll_before),
        'logloss_cal_after': float(cal_ll_after),
        'logloss_cal_improvement': float(cal_ll_before - cal_ll_after),
        'logloss_holdout_before': float(hold_ll_before),
        'logloss_holdout_after': float(hold_ll_after),
        'logloss_holdout_improvement': float(hold_ll_before - hold_ll_after),
    }

    with open(output_json, 'w') as f:
        json.dump(payload, f, indent=2)
    # Optional: print save path for visibility when called from scripts
    print(f"[calibrate_student] Saved calibration to {output_json}")

    return payload


if __name__ == '__main__':
    info = calibrate_student()
    print(info)
