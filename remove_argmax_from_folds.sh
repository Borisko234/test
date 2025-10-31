#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --time=30
#SBATCH --cpus-per-task=2
#SBATCH --job-name=RemoveArgmax
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

echo "[RemoveArgmax] Removing 'argmax' column from all fold train/val files..."

for f in data/fold_data/fold_*_train.csv data/fold_data/fold_*_val.csv fold_data/fold_*_train.csv fold_data/fold_*_val.csv /scratch-shared/tc1proj005/fold_data/fold_*_train.csv /scratch-shared/tc1proj005/fold_data/fold_*_val.csv; do
  if [ -f "$f" ]; then
    echo "Processing $f"
    python -c "import pandas as pd; df = pd.read_csv('$f'); df = df.drop(columns=['argmax'], errors='ignore'); df.to_csv('$f', index=False)"
  fi
done

echo "[RemoveArgmax] Done."
