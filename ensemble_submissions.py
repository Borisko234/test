import pandas as pd
import numpy as np
import sys
from typing import List, Optional

def ensemble_submissions(paths: List[str], out_path: str, weights: Optional[List[float]] = None) -> str:
    if not paths:
        raise ValueError("No submission paths provided")
    dfs = [pd.read_csv(p) for p in paths]
    base = dfs[0][['id']].copy()
    cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
    arrs = [df[cols].values for df in dfs]
    if weights is not None:
        if len(weights) != len(arrs):
            raise ValueError("weights length must match number of submissions")
        w = np.array(weights, dtype=float)
        w = w / (w.sum() if w.sum() != 0 else 1.0)
        stacked = np.stack(arrs, axis=0)
        avg = np.tensordot(w, stacked, axes=(0,0))
    else:
        avg = np.mean(arrs, axis=0)
    avg = np.clip(avg, 1e-9, None)
    avg = avg / avg.sum(axis=1, keepdims=True)
    out = base.assign(
        winner_model_a=avg[:,0],
        winner_model_b=avg[:,1],
        winner_tie=avg[:,2],
    )
    out.to_csv(out_path, index=False)
    return out_path

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python ensemble_submissions.py out.csv sub1.csv sub2.csv ... [--weights w1,w2,...]")
        sys.exit(1)
    out = sys.argv[1]
    args = sys.argv[2:]
    weights = None
    if '--weights' in args:
        idx = args.index('--weights')
        weight_str = args[idx+1]
        args = args[:idx] + args[idx+2:]
        weights = [float(x) for x in weight_str.split(',')]
    inputs = args
    path = ensemble_submissions(inputs, out, weights=weights)
    print({"saved": path})
