import argparse
import json
import os
import re
import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _load_cv_metric(path: str) -> float:
    with open(path, 'r') as f:
        m = json.load(f)
    # Prefer eval_log_loss -> log_loss -> eval_loss
    for key in ['eval_log_loss', 'log_loss', 'eval_loss']:
        if key in m and m[key] is not None:
            try:
                return float(m[key])
            except Exception:
                continue
    raise ValueError(f"No usable log loss metric found in {path}")


def _match_fold_from_path(path: str) -> int:
    # Try to extract fold number from common patterns
    for pat in [r"fold_(\d+)_of_\d+", r"fold_(\d+)"]:
        m = re.search(pat, path)
        if m:
            return int(m.group(1))
    raise ValueError(f"Could not infer fold index from path: {path}")


def _resolve_from_run(run: str) -> Tuple[List[str], List[str], List[int]]:
    # Find metric files under model_save/<run>/student_distilbert_fold_*/cv_metrics_*.json
    model_root = os.path.join('model_save', run)
    metric_paths = glob.glob(os.path.join(model_root, 'student_distilbert_fold_*', 'cv_metrics_*.json'))
    if not metric_paths:
        raise FileNotFoundError(f"No cv_metrics json found under {model_root}")
    folds = []
    fold_to_metric = {}
    for p in metric_paths:
        try:
            k = _match_fold_from_path(p)
            fold_to_metric[k] = p
            folds.append(k)
        except Exception:
            continue
    folds = sorted(set(folds))
    if not folds:
        raise FileNotFoundError(f"No fold metrics found under {model_root}")

    # Resolve submissions: sub/<run>_student_submission_fold_<k>.csv
    sub_paths = []
    for k in folds:
        sp = os.path.join('sub', f"{run}_student_submission_fold_{k}.csv")
        if not os.path.exists(sp):
            raise FileNotFoundError(f"Missing submission for fold {k}: {sp}")
        sub_paths.append(sp)
    metric_list = [fold_to_metric[k] for k in folds]
    return metric_list, sub_paths, folds


def ensemble_from_cv(out_path: str, metrics: List[str], submissions: List[str]) -> str:
    if len(metrics) != len(submissions):
        raise ValueError("metrics and submissions must have the same length")

    # Load metrics and derive weights w_k ~ 1 / LL_k
    ll_list = []
    for mp in metrics:
        ll_list.append(_load_cv_metric(mp))
    ll = np.array(ll_list, dtype=float)
    inv = 1.0 / np.clip(ll, 1e-12, None)
    weights = inv / inv.sum()

    # Load submissions and check alignment
    cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
    dfs = [pd.read_csv(p) for p in submissions]
    base = dfs[0][['id']].copy()
    for i, df in enumerate(dfs[1:], start=1):
        if not base['id'].equals(df['id']):
            raise ValueError(f"Submission id columns do not match between {submissions[0]} and {submissions[i]}")

    stack = np.stack([df[cols].values for df in dfs], axis=0)  # [K,N,3]
    avg = np.tensordot(weights, stack, axes=(0,0))            # [N,3]
    avg = np.clip(avg, 1e-9, None)
    avg = avg / avg.sum(axis=1, keepdims=True)

    out = base.assign(
        winner_model_a=avg[:,0],
        winner_model_b=avg[:,1],
        winner_tie=avg[:,2],
    )
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    out.to_csv(out_path, index=False)

    # Also emit the weights for transparency
    meta = {
        'weights': weights.tolist(),
        'metrics': ll_list,
        'submissions': submissions,
        'metrics_files': metrics,
    }
    with open(os.path.splitext(out_path)[0] + '.weights.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return out_path


def main():
    ap = argparse.ArgumentParser(description='Weighted ensemble from per-fold CV metrics')
    ap.add_argument('out', help='Output ensemble CSV path')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--run', help='RUN name to auto-discover metrics and submissions')
    group.add_argument('--pairs', nargs='+', help='Pairs of metrics.json=submission.csv')
    args = ap.parse_args()

    if args.run:
        metrics, subs, folds = _resolve_from_run(args.run)
        print({
            'run': args.run,
            'folds': folds,
            'metrics': metrics,
            'submissions': subs,
        })
        path = ensemble_from_cv(args.out, metrics, subs)
        print({'saved': path})
        return

    # Pairs mode
    m_list, s_list = [], []
    for pair in args.pairs:
        if '=' not in pair:
            raise ValueError('Each --pairs item must be metrics.json=submission.csv')
        m, s = pair.split('=', 1)
        if not (os.path.exists(m) and os.path.exists(s)):
            raise FileNotFoundError(f'Pair paths not found: {pair}')
        m_list.append(m)
        s_list.append(s)
    path = ensemble_from_cv(args.out, m_list, s_list)
    print({'saved': path})


if __name__ == '__main__':
    main()
