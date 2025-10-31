#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --cpus-per-task=8
#SBATCH --job-name=DistillAll
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

# Run all folds sequentially and ensemble at the end.
# Usage: sbatch distill_all_folds.sh

RUN=${RUN:-$(date +%Y%m%d_%H%M%S)}
# Set USE_WEIGHTED=1 to use CV-weighted ensembling instead of equal mean
USE_WEIGHTED=${USE_WEIGHTED:-0}

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${PWD}/.hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for FOLD in 0 1 2 3 4; do
  echo "[DistillAll] RUN=${RUN} FOLD=${FOLD}"
  FOLD=${FOLD} RUN=${RUN} sbatch --wait job_distill.sh
  if [ $? -ne 0 ]; then
    echo "[DistillAll] Fold ${FOLD} submission failed" >&2
    exit 1
  fi
done

if [ "$USE_WEIGHTED" -eq 1 ]; then
  echo "[DistillAll] CV-weighted ensemble"
  python ensemble_from_cv.py sub/${RUN}_final_ensemble.csv --run ${RUN}
else
  echo "[DistillAll] Equal-mean ensemble"
  python ensemble_submissions.py sub/${RUN}_final_ensemble.csv \
    sub/${RUN}_student_submission_fold_0.csv \
    sub/${RUN}_student_submission_fold_1.csv \
    sub/${RUN}_student_submission_fold_2.csv \
    sub/${RUN}_student_submission_fold_3.csv \
    sub/${RUN}_student_submission_fold_4.csv
fi
