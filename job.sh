#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=360
# Request more CPU cores to speed up tokenization and dataloading
# (adjust if your cluster has different limits)
#SBATCH --cpus-per-task=8
#SBATCH --job-name=TestJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

# Throughput-oriented env tuning
# Avoid CPU thread oversubscription with multi-process tokenization/dataloaders
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Disable internal tokenizer threading when also using multiprocessing
export TOKENIZERS_PARALLELISM=false
# Cache datasets/tokenization artifacts across runs for reuse
export HF_DATASETS_CACHE="${PWD}/.hf_cache"
# Reduce CUDA allocator fragmentation (can help long jobs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run directly (submit this script with: sbatch job.sh)
python ./main.py --mode student-train \
  --student-model microsoft/deberta-v3-base \
  --student-epochs 2 \
  --student-max-samples 0 \
  --student-label-smoothing 0.1 \
  --student-early-stopping 1 \
  --student-max-length 256 \
  --student-extra-csvs "data/ultrafeedback.csv,data/ultrafeedback_ties.csv,data/lmsys-33k-deduplicated.csv" \
  --student-dedup-by-prompt \
  --student-shuffle-ab \
  --student-fp16 \
  --student-gradient-checkpointing \
  --student-train-batch-size 12 \
  --student-eval-batch-size 12 \
  --student-grad-accum 2 \
  --student-num-workers 8

# Optional: cap samples for a much faster run while iterating
# (uncomment to limit total merged+deduped rows before train/val split)
#   --student-max-samples 120000 \