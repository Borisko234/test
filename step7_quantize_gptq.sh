#!/bin/bash
# Step 7: Quantize final merged model to 8-bit (GPTQ with calibration)
# Usage:
#   sbatch step7_quantize_gptq.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --time=180
#SBATCH --cpus-per-task=4
#SBATCH --job-name=S7_GPTQ
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

MODEL_DIR=${MODEL_DIR:-model_save/final_merged_model}
OUT_DIR=${OUT_DIR:-model_save/final_quantized_model}
CALIB_CSV=${CALIB_CSV:-./data/train.csv}
TOKENIZER_DIR=${TOKENIZER_DIR:-google/gemma-2-9b-it}
# Sanity check: ensure merged model contains weights
if [ -d "$MODEL_DIR" ]; then
  if ! ls "$MODEL_DIR" | grep -E "(pytorch_model\.bin|model\.safetensors|pytorch_model-.*\.bin|model-.*\.safetensors|pytorch_model\.bin\.index\.json|model\.safetensors\.index\.json)" > /dev/null 2>&1; then
    echo "[Step7][Error] No weight files found in $MODEL_DIR. Re-run Step 6 to merge LoRA into a base with standard filenames."
    exit 1
  fi
fi
# Use 8-bit per the summary, keep group size reasonable
python quantize_gptq_calibrated.py \
  --model-dir "$MODEL_DIR" \
  --out-dir "$OUT_DIR" \
  --calib-csv "$CALIB_CSV" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --bits 8 \
  --group-size 128 \
  --max-calib-samples 2048 \
  --max-length 1024

echo "[Step7] Done"

# Ensure config.json is present in the quantized model directory
cp "$MODEL_DIR/config.json" "$OUT_DIR/config.json"
