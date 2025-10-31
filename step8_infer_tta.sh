#!/bin/bash
# Step 8: Apply TTA during inference (GPTQ 8-bit CausalLM + classification head)
# Usage:
#   sbatch step8_infer_tta.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --time=120
#SBATCH --cpus-per-task=4
#SBATCH --job-name=S8_TTA
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

if [ -d "$HOME/exported-assets_sc4000" ]; then
  cd "$HOME/exported-assets_sc4000"
else
  cd "$(dirname "$0")"
fi

HEAD_DIR=${HEAD_DIR:-model_save/final_merged_model}
HEAD_OUT=${HEAD_OUT:-${HEAD_DIR}/classifier_head.pt}
if [ ! -f "$HEAD_OUT" ]; then
  if ! python export_classifier_head.py \
    --model-dir "$HEAD_DIR" \
    --out "$HEAD_OUT"; then
    echo "[Step8][Warn] Head export failed; will use --classifier-from-dir=$HEAD_DIR"
    HEAD_OUT=""
  fi
fi

MODEL_DIR=${MODEL_DIR:-./model_save/final_merged_model}
TEST_CSV=${TEST_CSV:-./data/test.csv}
OUT_SUB=${OUT_SUB:-./sub/final_submission.csv}
CALIB_JSON=${CALIB_JSON:-${HEAD_DIR}/calibration.json}
# Optional explicit base model for fallback if GPTQ shards are corrupted
FALLBACK_BASE=${FALLBACK_BASE:-google/gemma-2-9b-it}
# If merged base lacks weights or config, fall back to the original HF id
if [ -d "$FALLBACK_BASE" ]; then
  if ! ls "$FALLBACK_BASE" | grep -E "(pytorch_model\.bin|model\.safetensors|pytorch_model-.*\.bin|model-.*\.safetensors|pytorch_model\.bin\.index\.json|model\.safetensors\.index\.json)" > /dev/null 2>&1 || [ ! -s "$FALLBACK_BASE/config.json" ]; then
    echo "[Step8][Warn] $FALLBACK_BASE missing weights or config.json; using base HF model google/gemma-2-9b-it as fallback."
    FALLBACK_BASE="google/gemma-2-9b-it"
  fi
fi
CMD=(python student_infer_simple.py \
  --model-dir "$MODEL_DIR" \
  --test-csv "$TEST_CSV" \
  --out "$OUT_SUB" \
  --head-path "$HEAD_OUT" \
  --batch-size 4 \
  --max-length 2000)

export BASE_MODEL="$FALLBACK_BASE"
"${CMD[@]}"

echo "[Step8] Wrote $OUT_SUB"
