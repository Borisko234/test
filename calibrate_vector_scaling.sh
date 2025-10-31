#!/bin/bash
# Run vector scaling calibration across folds using validation predictions.
# Usage:
#   sbatch calibrate_vector_scaling.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --time=30
#SBATCH --cpus-per-task=2
#SBATCH --job-name=VecCal
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load anaconda || true
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate myenv || true
fi

cd "$HOME/exported-assets_sc4000" || cd "$(dirname "$0")"

# Arguments (override via env):
PREFIX=${PREFIX:-llama}
FOLDS=${FOLDS:-0,1,2}
FOLD_DIR=${FOLD_DIR:-fold_data}
OUT_DIR=${OUT_DIR:-calibration}
PREFER=${PREFER:-logprobs}   # logprobs | probs
SCORE_KIND=${SCORE_KIND:-auto}  # auto | probs | logprobs
SAVE_JSON=${SAVE_JSON:-1}

python -u vector_calibration.py \
  --prefix "$PREFIX" \
  --folds "$FOLDS" \
  --fold-dir "$FOLD_DIR" \
  --prefer "$PREFER" \
  --score-kind "$SCORE_KIND" \
  --out-dir "$OUT_DIR" \
  $( [ "$SAVE_JSON" = "1" ] && echo "--save-json" )

echo "[VecCal] Saved calibration artifacts under $OUT_DIR"
