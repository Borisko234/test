#!/bin/bash
# Step 6: Directly average the LoRA adapters from 5 folds and merge into a final base.
# Usage:
#   sbatch step6_lora_ensemble.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --time=30
#SBATCH --cpus-per-task=2
#SBATCH --job-name=S6_LoRAEnsemble
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

# Inputs
BASE_MODEL=${BASE_MODEL:-google/gemma-2-9b-it}
FOLDS=${FOLDS:-"0,1"}
LORA_DIR_PREFIX=${LORA_DIR_PREFIX:-model_save/distilled_gemma2-9b_fold_}
OUT_LORA=${OUT_LORA:-model_save/avg_lora}
OUT_MERGED=${OUT_MERGED:-model_save/final_merged_model}

echo "[Step6] Averaging LoRA adapters across folds: $FOLDS -> $OUT_LORA"
python - <<'PY'
import os, json, shutil, torch
from safetensors.torch import load_file, save_file

folds = [0, 1]  # Only use folds 0 and 1 for averaging; ignore fold 2
pref = os.environ.get('LORA_DIR_PREFIX','model_save/distilled_gemma2-9b_fold_')
out_dir = os.environ.get('OUT_LORA','model_save/avg_lora')
os.makedirs(out_dir, exist_ok=True)

adapters = []
config_src = None
for k in folds:
	fold_dir = f"{pref}{k}"
	cand = os.path.join(fold_dir, "adapter_model.safetensors")
	if os.path.isfile(cand):
		adapters.append(cand)
		cfg = os.path.join(fold_dir, 'adapter_config.json')
		if config_src is None and os.path.isfile(cfg):
			config_src = cfg
	else:
		# try peft default
		alt = os.path.join(fold_dir, "adapter_model.bin")
		if os.path.isfile(alt):
			adapters.append(alt)
			cfg = os.path.join(fold_dir, 'adapter_config.json')
			if config_src is None and os.path.isfile(cfg):
				config_src = cfg
		else:
			print(f"[Step6][Warn] Missing adapter for fold {k}: {cand}")
if not adapters:
	raise SystemExit("No fold adapters found to average.")

print(f"[Step6] Found {len(adapters)} adapters: {adapters}")
acc = {}
for p in adapters:
	if p.endswith('.safetensors'):
		state = load_file(p)
	else:
		state = torch.load(p, map_location='cpu')
	for k,v in state.items():
		if not torch.is_floating_point(v):
			continue
		acc[k] = acc.get(k, torch.zeros_like(v)) + v

avg = {k: (v/len(adapters)).to(v.dtype) for k,v in acc.items()}
save_path = os.path.join(out_dir, 'adapter_model.safetensors')
save_file(avg, save_path)
print(f"[Step6] Wrote averaged adapter -> {save_path}")

# Ensure adapter_config.json is present so PEFT can load locally
if config_src and os.path.isfile(config_src):
	dst = os.path.join(out_dir, 'adapter_config.json')
	try:
		shutil.copy2(config_src, dst)
		print(f"[Step6] Copied adapter_config.json from {config_src} -> {dst}")
	except Exception as e:
		print(f"[Step6][Warn] Failed to copy adapter_config.json: {e}")
else:
	print("[Step6][Warn] No adapter_config.json found in any fold dir; merge may fail. Provide a config or re-run Step5 with PEFT saving.")
PY

echo "[Step6] Merging averaged LoRA into base: $BASE_MODEL -> $OUT_MERGED"
python lora_merge.py --base-model "$BASE_MODEL" --lora-dir "$OUT_LORA" --out-dir "$OUT_MERGED"

echo "[Step6] Done: merged model in $OUT_MERGED"
