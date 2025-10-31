#!/bin/bash
# Step 2: Create and persist 3-fold splits to reuse across teachers and student.
# Usage:
#   sbatch step2_make_folds.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --time=30
#SBATCH --cpus-per-task=2
#SBATCH --job-name=S2_Folds
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

SCRATCH_BASE=${SCRATCH_BASE:-/scratch-shared/tc1proj005}
FOLDS_LOCAL_DIR="data/processed_data"
FOLDS_LOCAL_FILE="${FOLDS_LOCAL_DIR}/folds_3_seed42.json"
FOLDS_SCRATCH_DIR=${FOLDS_OUT_DIR:-"${SCRATCH_BASE}/folds"}
mkdir -p "${FOLDS_LOCAL_DIR}" || true
mkdir -p "${FOLDS_SCRATCH_DIR}" || true


# Extra 33k dataset path (override via EXTRA_DATA)
EXTRA_DATA_DEFAULT="data/lmsys-33k-deduplicated.csv"
if [ ! -f "$EXTRA_DATA_DEFAULT" ]; then
    EXTRA_DATA_DEFAULT="data/lmsys-33k.csv"
fi
EXTRA_DATA=${EXTRA_DATA:-$EXTRA_DATA_DEFAULT}
export EXTRA_DATA

python - <<'PY'
import json, os, pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

train_csv = 'data/train.csv'
df = pd.read_csv(train_csv)
label_col = None
if 'winner' in df.columns:
    label_col = 'winner'
else:
    for trio in [
        ['winner_model_a','winner_model_b','winner_tie'],
        ['winner_model_a_prob','winner_model_b_prob','winner_tie_prob']
    ]:
        if all(c in df.columns for c in trio):
            label_col = 'argmax'
            df['argmax'] = df[trio].values.argmax(axis=1)
            break
if label_col is None:
    raise SystemExit('No labels/probs found in data/train.csv')

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
labels = df[label_col].astype(str).values
folds = {}
for k, (_, val_idx) in enumerate(skf.split(df, labels)):
    folds[k] = sorted(val_idx.tolist())

os.makedirs('data/processed_data', exist_ok=True)
out_local = os.environ.get('FOLDS_LOCAL_FILE','data/processed_data/folds_3_seed42.json')
with open(out_local,'w') as f:
        json.dump(folds, f)
print(f'[Step2] wrote {out_local}')

# Also materialize per-fold CSVs:
#  - Train = 4/5 Kaggle train + ALL extra 33k rows (if present)
#  - Dev   = 1/5 Kaggle train only
fold_out_dir = 'fold_data'
os.makedirs(fold_out_dir, exist_ok=True)

extra_path = os.environ.get('EXTRA_DATA')
extra_df = None
if extra_path and os.path.isfile(extra_path):
    try:
        extra_df = pd.read_csv(extra_path, low_memory=False)
        print(f"[Step2] Loaded extra 33k dataset: {extra_path} | rows={len(extra_df)} cols={list(extra_df.columns)[:8]}{'...' if len(extra_df.columns)>8 else ''}")
    except Exception as e:
        print(f"[Step2][Warn] Failed to load extra dataset '{extra_path}': {e}")
        extra_df = None
else:
    print(f"[Step2][Info] No extra dataset found at '{extra_path}'. Train sets will include Kaggle only.")

for k, val_idx in folds.items():
    val_mask = np.zeros(len(df), dtype=bool)
    val_mask[val_idx] = True
    kaggle_val = df[val_mask].copy()
    kaggle_train = df[~val_mask].copy()

    # Combine with extra dataset for train if available
    if extra_df is not None:
        combined_train = pd.concat([kaggle_train, extra_df], ignore_index=True, sort=False)
    else:
        combined_train = kaggle_train

    train_path = os.path.join(fold_out_dir, f'fold_{k}_train.csv')
    val_path = os.path.join(fold_out_dir, f'fold_{k}_val.csv')
    combined_train.to_csv(train_path, index=False)
    kaggle_val.to_csv(val_path, index=False)
    print(f"[Step2] Fold {k}: train rows={len(combined_train)} (kaggle={len(kaggle_train)} + extra={0 if extra_df is None else len(extra_df)}), val rows={len(kaggle_val)}")
PY

SCRATCH_FOLDS_FILE="${FOLDS_SCRATCH_DIR}/folds_3_seed42.json"
if [ -f "${FOLDS_LOCAL_FILE}" ]; then
    cp -f "${FOLDS_LOCAL_FILE}" "${SCRATCH_FOLDS_FILE}" && echo "[Step2] Copied folds JSON to ${SCRATCH_FOLDS_FILE}" || echo "[Step2][Warn] Failed to copy folds JSON to scratch (${SCRATCH_FOLDS_FILE})"
else
    echo "[Step2][Warn] Expected local folds file ${FOLDS_LOCAL_FILE} not found after generation." >&2
fi
if [ -f "${SCRATCH_FOLDS_FILE}" ]; then
    echo "[Step2] Scratch folds file present: ${SCRATCH_FOLDS_FILE}";
else
    echo "[Step2][Error] Scratch folds file missing: ${SCRATCH_FOLDS_FILE}" >&2
fi

echo "[Step2] Done"
