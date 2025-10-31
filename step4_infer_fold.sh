#!/bin/bash
# Step 4 (per-fold): Infer teacher distributions for ONE fold.
# Usage examples:
#   FOLD=0 sbatch step4_infer_fold.sh
#   sbatch --export=FOLD=3,INFER_MODELS=llama step4_infer_fold.sh
# Or submit an array: sbatch --array=0-4 step4_infer_fold.sh  (script maps SLURM_ARRAY_TASK_ID to FOLD)

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=180   # 3h per fold (adjust as needed)
#SBATCH --job-name=S4_Fold
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

# Determine fold
if [ -z "${FOLD:-}" ]; then
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    FOLD=${SLURM_ARRAY_TASK_ID}
  else
    echo "[FoldInfer][Error] FOLD not set and no SLURM_ARRAY_TASK_ID." >&2
    exit 2
  fi
fi

# Allow override of models & root; otherwise rely on defaults of step4 script
export INFER_FOLDS=${FOLD}
# Optionally set INFER_MODELS or other vars before calling this wrapper.

# Reuse the main step4 script (which now supports single-fold selection)
# We avoid last-token logits by default for speed.
export INFER_SAVE_LASTTOK=${INFER_SAVE_LASTTOK:-0}
# Force regen within the fold so retries reflect changes.
export INFER_FORCE_REGEN=${INFER_FORCE_REGEN:-1}

bash step4_infer_teacher_logits.sh

echo "[FoldInfer] Completed fold ${FOLD}"