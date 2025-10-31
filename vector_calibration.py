import argparse
import json
import os
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

NUM_CLASSES = 3  # A, B, tie


def _labels_from_csv(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if 'winner' in df.columns:
        label_map = {"model_a": 0, "model_b": 1, "tie": 2, "tie (both bad)": 2}
        return df['winner'].map(label_map).astype(int).values
    for trio in (
        ['winner_model_a', 'winner_model_b', 'winner_tie'],
        ['winner_model_a_prob', 'winner_model_b_prob', 'winner_tie_prob'],
    ):
        if all(c in df.columns for c in trio):
            arr = df[trio].astype(float).values
            return arr.argmax(axis=1).astype(int)
    raise ValueError(f"Could not infer labels from {csv_path} — expected 'winner' or probability columns.")


def _load_scores(path: str, kind: str = 'auto', eps: float = 1e-12) -> np.ndarray:
    """Load per-row 3-way scores from a file.
    - .pt expected to contain tensor/array of shape [N,3] as probs or logprobs
    - If kind='probs', converts to log-space;
      if kind='logprobs', uses as-is;
      if kind='auto', detects by checking if rows sum to ~1 and all in [0,1].
    Returns array of shape [N,3] in 'logits-like' space (we use log-probs as surrogate logits).
    """
    if path.endswith('.pt'):
        obj = torch.load(path, map_location='cpu')
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        else:
            arr = np.array(obj)
    elif path.endswith('.npy'):
        arr = np.load(path)
    else:
        raise ValueError(f"Unsupported file extension for scores: {path}")
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Scores wrong shape in {path}: {arr.shape}")

    def to_log_space(a: np.ndarray) -> np.ndarray:
        # clip then log, avoid -inf
        a = np.clip(a, eps, 1.0)
        return np.log(a)

    if kind == 'probs':
        return to_log_space(arr)
    if kind == 'logprobs':
        return arr
    # auto-detect
    row_sums = arr.sum(axis=1, keepdims=True)
    in_01 = (arr >= -1e-9).all() and (arr <= 1+1e-9).all()
    approx_prob = in_01 and np.allclose(row_sums[: min(16, len(arr))], 1.0, atol=1e-3)
    return to_log_space(arr) if approx_prob else arr


def vector_calibration_loss(params: np.ndarray, logits_like: np.ndarray, labels: np.ndarray) -> float:
    a = params[:NUM_CLASSES]
    b = params[NUM_CLASSES:]
    logits_calib = logits_like * a + b  # elementwise scale + bias per class
    log_probs = logits_calib - logsumexp(logits_calib, axis=1, keepdims=True)
    nll = -log_probs[np.arange(len(labels)), labels].mean()
    return float(nll)


def gather_fold_pairs(prefix: str, folds: List[int], fold_dir: str, prefer: str = 'logprobs') -> List[Tuple[str, str]]:
    """Return list of (scores_file, labels_csv) for each fold.
    prefer: 'logprobs' tries llama_fold_k_val_logprobs.pt first, then *_val_probs.pt
    """
    pairs = []
    pred_dir = "model_save/teacher_logits"
    for k in folds:
        base_logprobs = os.path.join(pred_dir, f"{prefix}_fold_{k}_val_logprobs.pt")
        base_probs = os.path.join(pred_dir, f"{prefix}_fold_{k}_val_probs.pt")
        scores = None
        if prefer == 'logprobs' and os.path.isfile(base_logprobs):
            scores = base_logprobs
        elif os.path.isfile(base_probs):
            scores = base_probs
        elif os.path.isfile(base_logprobs):
            scores = base_logprobs
        else:
            raise FileNotFoundError(f"Missing val predictions for fold {k}: {base_logprobs} or {base_probs}")
        labels_csv = os.path.join(fold_dir, f"fold_{k}_val.csv")
        if not os.path.isfile(labels_csv):
            raise FileNotFoundError(f"Missing labels CSV for fold {k}: {labels_csv}")
        pairs.append((scores, labels_csv))
    return pairs


def main():
    ap = argparse.ArgumentParser(description='Vector scaling calibration for 3-way logits using per-fold validation predictions.')
    ap.add_argument('--prefix', default='llama', help='Prediction file prefix, e.g., llama -> llama_fold_k_val_*')
    ap.add_argument('--folds', default='0,1,2', help='Comma-separated fold indices to include')
    ap.add_argument('--fold-dir', default='fold_data', help='Directory containing fold_{k}_val.csv files')
    ap.add_argument('--prefer', choices=['logprobs', 'probs'], default='logprobs', help='Which prediction type to prefer when both exist')
    ap.add_argument('--score-kind', choices=['auto', 'probs', 'logprobs'], default='auto', help='Force interpretation of prediction files')
    ap.add_argument('--out-dir', default='calibration', help='Directory to write calibration artifacts')
    ap.add_argument('--save-json', action='store_true', help='Also save a JSON alongside the .npz')
    args = ap.parse_args()

    fold_ids = [int(s) for s in args.folds.replace(',', ' ').split() if s.strip()]
    pairs = gather_fold_pairs(args.prefix, fold_ids, args.fold_dir, prefer=args.prefer)

    all_logits_like = []
    all_labels = []
    total = 0
    for scores_path, labels_csv in pairs:
        scores = _load_scores(scores_path, kind=args.score_kind)
        labels = _labels_from_csv(labels_csv)
        if scores.shape[0] != labels.shape[0]:
            raise ValueError(f"Length mismatch for {scores_path} vs {labels_csv}: {scores.shape[0]} vs {labels.shape[0]}")
        all_logits_like.append(scores.astype(np.float64))
        all_labels.append(labels.astype(np.int64))
        total += len(labels)
        print(f"[vector_calib] Fold file {os.path.basename(scores_path)} | rows={len(labels)}")

    X = np.concatenate(all_logits_like, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"[vector_calib] Total rows across folds: {total}")

    init_params = np.concatenate([np.ones(NUM_CLASSES), np.zeros(NUM_CLASSES)])
    result = minimize(
        vector_calibration_loss,
        init_params,
        args=(X, y),
        method='L-BFGS-B'
    )
    opt_a = result.x[:NUM_CLASSES]
    opt_b = result.x[NUM_CLASSES:]

    print("Optimal scale factors (a):", opt_a)
    print("Optimal biases (b):", opt_b)
    print("Final NLL:", result.fun)

    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, 'vector_scaling_params.npz')
    np.savez(npz_path, a=opt_a, b=opt_b, meta={
        'prefix': args.prefix,
        'folds': fold_ids,
        'prefer': args.prefer,
        'score_kind': args.score_kind,
        'n_rows': int(total),
        'final_nll': float(result.fun),
    })
    print(f"Saved to {npz_path}")

    if args.save_json:
        json_path = os.path.join(args.out_dir, 'vector_scaling_params.json')
        with open(json_path, 'w') as f:
            json.dump({'a': opt_a.tolist(), 'b': opt_b.tolist()}, f, indent=2)
        print(f"Saved to {json_path}")


def apply_vector_calibration(logits_like: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    logits_calib = logits_like * a + b
    probs = np.exp(logits_calib - logsumexp(logits_calib, axis=1, keepdims=True))
    return probs


if __name__ == '__main__':
    main()
