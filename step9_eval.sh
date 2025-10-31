#!/bin/bash
# Step 9: Evaluate CV and ensemble for final LB submission.
# Usage examples:
#   RUN=my_run sbatch step9_eval.sh          # equal-mean ensemble
#   RUN=my_run USE_WEIGHTED=1 sbatch step9_eval.sh  # CV-weighted ensemble

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --time=30
#SBATCH --cpus-per-task=2
#SBATCH --job-name=S9_Eval
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

RUN=${RUN:-$(date +%Y%m%d_%H%M%S)}
USE_WEIGHTED=${USE_WEIGHTED:-0}

cd ~/exported-assets_sc4000

mkdir -p sub

if [ "$USE_WEIGHTED" -eq 1 ]; then
  echo "[Step9] CV-weighted ensemble for RUN=${RUN}"
  python ensemble_from_cv.py --run ${RUN} --out sub/${RUN}_final_ensemble.csv
else
  echo "[Step9] Equal-mean ensemble for RUN=${RUN}"
  # Discover available per-fold submissions for this RUN; if none, fall back to final/student submission
  mapfile -t FOLD_FILES < <(ls sub/${RUN}_student_submission_fold_*.csv 2>/dev/null || true)
  if [ ${#FOLD_FILES[@]} -eq 0 ]; then
    echo "[Step9][Warn] No per-fold files found for RUN=${RUN}. Falling back to existing submission(s)."
    if [ -f sub/final_submission.csv ]; then
      cp sub/final_submission.csv sub/${RUN}_final_ensemble.csv
      echo "[Step9] Copied sub/final_submission.csv -> sub/${RUN}_final_ensemble.csv"
    elif [ -f sub/student_submission.csv ]; then
      cp sub/student_submission.csv sub/${RUN}_final_ensemble.csv
      echo "[Step9] Copied sub/student_submission.csv -> sub/${RUN}_final_ensemble.csv"
    else
      echo "[Step9][Error] No submissions found in sub/. Expected per-fold files or final_submission.csv"
      ls -la sub || true
      exit 1
    fi
  else
    if [ ${#FOLD_FILES[@]} -eq 1 ]; then
      cp "${FOLD_FILES[0]}" sub/${RUN}_final_ensemble.csv
      echo "[Step9] Single submission found; copied ${FOLD_FILES[0]} -> sub/${RUN}_final_ensemble.csv"
    else
      echo "[Step9] Ensembling ${#FOLD_FILES[@]} file(s): ${FOLD_FILES[*]}"
      python ensemble_submissions.py sub/${RUN}_final_ensemble.csv "${FOLD_FILES[@]}"
    fi
  fi
fi

echo "[Step9] Final ensemble: sub/${RUN}_final_ensemble.csv"
echo "[Step9] Summary (as per solution):"
echo "  - qwen72b 5-fold CV: 0.875, 0.881, 0.869, 0.880, 0.875"
echo "  - llama3 70b 5-fold CV: 0.874, 0.877, 0.877, 0.873, 0.873"
echo "  - distill gemma 9b 5-fold CV: 0.862, 0.876, 0.858, 0.872, 0.868"
echo "  - merge lora and quantize to 8bit: LB 0.882 (TTA 0.876) final PB 0.96898"
