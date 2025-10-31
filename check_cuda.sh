#!/bin/bash
#SBATCH --job-name=check_cuda
#SBATCH --output=check_cuda.out
#SBATCH --error=check_cuda.err
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=4         # Number of CPU cores
#SBATCH --mem=8G                  # Memory
#SBATCH --time=00:10:00           # Max runtime
#SBATCH --partition=UGGPU-TC1           # Change if your cluster uses a different GPU partition

# (Optional) Load your Python and CUDA modules
module load python/3.9
module load cuda/11.8

# (Optional) Activate a virtual environment if you have one
# source ~/envs/myenv/bin/activate

# Print diagnostic info
echo "Running on $(hostname)"
nvidia-smi

# Run the Python script
python check_cuda.py