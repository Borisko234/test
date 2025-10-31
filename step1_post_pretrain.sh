#!/bin/bash
# Step 1: Post-pretrain large models on UltraFeedback via LoRA/QLoRA, then merge adapters.
# Usage:
#   sbatch step1_post_pretrain.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=360
#SBATCH --cpus-per-task=8
#SBATCH --job-name=S1_PostPretrain
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate myenv

cd ~/exported-assets_sc4000

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Resolve and validate SCRATCH_BASE
echo "[Step1] Resolving SCRATCH_BASE..."

SCRATCH_BASE=$(python - <<'PY'
import os
default = '/scratch-shared/tc1proj005'
print(os.path.abspath(os.environ.get('SCRATCH_BASE', default)))
PY
)

echo "[Debug] SCRATCH_BASE resolved to: ${SCRATCH_BASE}"

if [ ! -d "${SCRATCH_BASE}" ] || [ ! -w "${SCRATCH_BASE}" ]; then
  echo "[Step1] SCRATCH_BASE not usable: ${SCRATCH_BASE}"
  if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}" ] && [ -w "${SLURM_TMPDIR}" ]; then
    echo "[Step1] Using SLURM_TMPDIR fallback: ${SLURM_TMPDIR}"
    SCRATCH_BASE="${SLURM_TMPDIR}"
  elif [ -d "/scratch-shared/tc1proj005" ] && [ -w "/scratch-shared/tc1proj005" ]; then
    echo "[Step1] Using hardcoded fallback: /scratch-shared/tc1proj005"
    SCRATCH_BASE="/scratch-shared/tc1proj005"
  else
    echo "[Step1] ERROR: No usable scratch directory found." >&2
    exit 98
  fi
fi

export HF_HOME="${SCRATCH_BASE}"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_BASE}"
export HF_DATASETS_CACHE="${SCRATCH_BASE}"
export TRANSFORMERS_CACHE="${SCRATCH_BASE}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Tokenizer/model repos (initial defaults; may be auto-downgraded based on GPU memory)
LLAMA_TOK=${LLAMA_TOK:-meta-llama/Llama-3.1-8B-Instruct}
QWEN_TOK=${QWEN_TOK:-Qwen/Qwen2.5-7B-Instruct}
GEMMA_TOK=${GEMMA_TOK:-google/gemma-2-9b}

LLAMA_BASE=${LLAMA_BASE:-meta-llama/Llama-3.1-8B-Instruct}
QWEN_BASE=${QWEN_BASE:-Qwen/Qwen2.5-7B-Instruct}
GEMMA_BASE=${GEMMA_BASE:-google/gemma-2-9b}

# Auto-downgrade very large models if single GPU has <= 32GB unless user overrides
AUTO_DOWNGRADE=${AUTO_DOWNGRADE:-1}
GPU_MEM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)
if [ "${AUTO_DOWNGRADE}" = "1" ] && [ "${GPU_MEM_GB}" -ne 0 ] && [ "${GPU_MEM_GB}" -le 32 ]; then
  # If user did not explicitly override to smaller variants already
  if [[ "${QWEN_BASE}" == *"72B"* ]]; then
    echo "[Step1][Auto] Detected ${GPU_MEM_GB}GB GPU: replacing Qwen2.5-72B with Qwen2.5-14B" >&2
    QWEN_BASE=Qwen/Qwen2.5-7B-Instruct
    QWEN_TOK=Qwen/Qwen2.5-7B-Instruct
  fi
  if [[ "${LLAMA_BASE}" == *"70B"* || "${LLAMA_BASE}" == *"72B"* ]]; then
    echo "[Step1][Auto] Detected ${GPU_MEM_GB}GB GPU: replacing LLaMA 70B with Meta-Llama-3.1-8B" >&2
    LLAMA_BASE=meta-llama/Llama-3.1-8B-Instruct
    LLAMA_TOK=meta-llama/Llama-3.1-8B-Instruct
  fi
fi

export LLAMA_TOK QWEN_TOK GEMMA_TOK

echo "[Step1] Working dir: $(pwd)"
echo "[Step1] Home dir: ${HOME}"
echo "[Step1] SCRATCH_BASE: ${SCRATCH_BASE}"
echo "[Step1] HF_HOME: ${HF_HOME}"
echo "[Step1] HUGGINGFACE_HUB_CACHE: ${HUGGINGFACE_HUB_CACHE}"
echo "[Step1] HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "[Step1] TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"

# Training configs & speed controls (override via environment):
#   POSTPRE_EPOCHS          : Epochs (can be fractional if lora_train.py supports) default=1
#   POSTPRE_LR              : Learning rate (default 1e-5)
#   POSTPRE_MAXLEN          : Max sequence length (default 1024; reduce to 512/384 for speed)
#   POSTPRE_GRAD_ACCUM      : Gradient accumulation steps (default 16 for gemma; optional others)
#   POSTPRE_SUBSET_SIZE     : If >0, cap number of training examples (0 or unset = full dataset)
#   POSTPRE_PER_DEVICE_BS   : Per-GPU batch size (default 1)
#   POSTPRE_MAX_STEPS       : If >0, hard cap total optimizer steps (overrides full epochs runtime)
#   POSTPRE_LORA_R          : LoRA rank (default 64; reduce to 16 or 8 for speed)
#   POSTPRE_LORA_ALPHA      : LoRA alpha (default 128; scale with rank, e.g. 32 when r=16)
#   POSTPRE_DISABLE_CHKPT   : If set to 1, do NOT use --grad-checkpoint (faster if memory ok)
POSTPRE_EPOCHS=${POSTPRE_EPOCHS:-1}
POSTPRE_LR=${POSTPRE_LR:-1e-5}
POSTPRE_MAXLEN=${POSTPRE_MAXLEN:-1024}
POSTPRE_PER_DEVICE_BS=${POSTPRE_PER_DEVICE_BS:-1}
POSTPRE_MAX_STEPS=${POSTPRE_MAX_STEPS:--1}
POSTPRE_LORA_R=${POSTPRE_LORA_R:-64}
POSTPRE_LORA_ALPHA=${POSTPRE_LORA_ALPHA:-128}
UT_DATA=${UT_DATA:-data/ultrafeedback.csv}
echo "[Step1] UT_DATA: ${UT_DATA} | EPOCHS: ${POSTPRE_EPOCHS} | LR: ${POSTPRE_LR} | MAXLEN: ${POSTPRE_MAXLEN} | PER_DEVICE_BS: ${POSTPRE_PER_DEVICE_BS} | MAX_STEPS: ${POSTPRE_MAX_STEPS} | LORA_R: ${POSTPRE_LORA_R} | LORA_ALPHA: ${POSTPRE_LORA_ALPHA}"

# Auto runtime budgeting (enable by leaving POSTPRE_MAX_STEPS=-1). Adjustable via:
#   POSTPRE_TIME_BUDGET_HOURS (default 6)
#   POSTPRE_EST_STEP_SEC (override heuristic seconds/step)
POSTPRE_TIME_BUDGET_HOURS=${POSTPRE_TIME_BUDGET_HOURS:-6}
POSTPRE_EST_STEP_SEC=${POSTPRE_EST_STEP_SEC:-0}
POSTPRE_SUBSET_SIZE=${POSTPRE_SUBSET_SIZE:-0}
POSTPRE_GRAD_ACCUM=${POSTPRE_GRAD_ACCUM:-16}
POSTPRE_FORCE_MAX_STEPS=${POSTPRE_FORCE_MAX_STEPS:-}

if [ "${POSTPRE_MAX_STEPS}" = "-1" ]; then
  TOTAL_LINES=0
  if [ -f "${UT_DATA}" ]; then
    TOTAL_LINES=$(wc -l < "${UT_DATA}" 2>/dev/null || echo 0)
  fi
  if [ "${TOTAL_LINES}" -gt 1 ]; then
    DATA_EXAMPLES=$((TOTAL_LINES - 1))
  else
    DATA_EXAMPLES=${TOTAL_LINES}
  fi
  if [ "${POSTPRE_SUBSET_SIZE}" -gt 0 ] && [ "${POSTPRE_SUBSET_SIZE}" -lt "${DATA_EXAMPLES}" ]; then
    EFFECTIVE_EXAMPLES=${POSTPRE_SUBSET_SIZE}
  else
    EFFECTIVE_EXAMPLES=${DATA_EXAMPLES}
  fi
  EFFECTIVE_BATCH=$(( POSTPRE_PER_DEVICE_BS * POSTPRE_GRAD_ACCUM ))
  if [ "${EFFECTIVE_BATCH}" -lt 1 ]; then EFFECTIVE_BATCH=1; fi
  STEPS_PER_EPOCH=$(python - <<PY
import math
ex=${EFFECTIVE_EXAMPLES}
b=${EFFECTIVE_BATCH}
print(1 if ex<=0 else math.ceil(ex/ b))
PY
  )
  if [ "${POSTPRE_EST_STEP_SEC}" = "0" ] || [ -z "${POSTPRE_EST_STEP_SEC}" ]; then
    case "${RUN_STAGE}" in
      gemma) POSTPRE_EST_STEP_SEC=22 ;;
      qwen)  POSTPRE_EST_STEP_SEC=48 ;;
      llama) POSTPRE_EST_STEP_SEC=50 ;;
      *) POSTPRE_EST_STEP_SEC=40 ;;
    esac
  fi
  PLANNED_STEPS=$(python - <<PY
import math
spp=${STEPS_PER_EPOCH}
epochs=float(${POSTPRE_EPOCHS})
print(int(math.ceil(spp * epochs)))
PY
  )
  if [ "${PLANNED_STEPS}" -lt 1 ]; then PLANNED_STEPS=1; fi
  EST_TOTAL_SEC=$(( PLANNED_STEPS * POSTPRE_EST_STEP_SEC ))
  BUDGET_SEC=$(python - <<PY
print(int(float(${POSTPRE_TIME_BUDGET_HOURS})*3600))
PY
  )
  if [ "${EST_TOTAL_SEC}" -gt "${BUDGET_SEC}" ]; then
    NEW_MAX=$(( BUDGET_SEC / POSTPRE_EST_STEP_SEC ))
    if [ "${NEW_MAX}" -lt 1 ]; then NEW_MAX=1; fi
    echo "[Step1][AutoCap] Projected ${PLANNED_STEPS} steps (~$((EST_TOTAL_SEC/3600))h) exceeds budget ${POSTPRE_TIME_BUDGET_HOURS}h; capping to ${NEW_MAX}."
    POSTPRE_MAX_STEPS=${NEW_MAX}
  else
    echo "[Step1][Estimate] Steps=${PLANNED_STEPS} (~$((EST_TOTAL_SEC/3600))h) within budget ${POSTPRE_TIME_BUDGET_HOURS}h."
  fi
fi

# If user explicitly sets POSTPRE_FORCE_MAX_STEPS, override computed/post values unconditionally
if [ -n "${POSTPRE_FORCE_MAX_STEPS}" ]; then
  echo "[Step1][Force] Overriding max steps to ${POSTPRE_FORCE_MAX_STEPS} (ignoring POSTPRE_MAX_STEPS=${POSTPRE_MAX_STEPS})"
  POSTPRE_MAX_STEPS=${POSTPRE_FORCE_MAX_STEPS}
fi

# Helper: Clean HF cache
clean_hf_cache() {
  echo "[Step1] Cleaning HF cache at ${HUGGINGFACE_HUB_CACHE}"
  if [ "${HUGGINGFACE_HUB_CACHE}" = "${SCRATCH_BASE}" ]; then
    echo "[Step1] Skipping cache clean to avoid wiping SCRATCH_BASE root"
    return 0
  fi
  rm -rf "${HUGGINGFACE_HUB_CACHE}"/* || true
}

# Helper: Ensure disk space
require_space_gb() {
  local need=$1
  local path="${HF_HOME}"
  local avail=$(df -BG "${path}" | awk 'NR==2 {gsub("G","",$4); print $4}')
  echo "[Step1] Disk check for ${path}: need ${need}G, avail ${avail}G"
  if [ "${avail}" -lt "${need}" ]; then
    echo "[Step1] ERROR: Not enough space at ${path}. Required: ${need}G, Available: ${avail}G" >&2
    exit 99
  fi
}

# Check/install dependencies
python - <<'PY'
import importlib, sys
for m in ['torch','transformers','peft','datasets','pandas','numpy','huggingface_hub']:
  try:
    importlib.import_module(m)
  except:
    print('[Step1] Missing', m)
    sys.exit(1)
print('[Step1] Deps OK')
PY

if [ $? -ne 0 ]; then
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r requirements.txt
fi

# HF login and tokenizer prefetch
export TRANSFORMERS_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=0
export HUGGINGFACE_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_ENABLE_HF_XET=0
export HUGGINGFACE_HUB_ENABLE_HF_XET=0
export HF_HUB_DISABLE_XET=1
export HF_HUB_HTTP_TIMEOUT=600
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_PROGRESS_BARS=1
export HF_TOKEN="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"

python - <<'PY'
import os, sys
print("[Step1] Online mode enabled (TRANSFORMERS_OFFLINE=0)")
from huggingface_hub import login
login(token="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", add_to_git_credential=False)

qwen_tok = os.environ.get("QWEN_TOK", "Qwen/Qwen2.5-7B-Instruct")
from transformers import AutoTokenizer
from pathlib import Path

def is_local_dir(p:str)->bool:
  try:
    return Path(p).exists() and Path(p).is_dir()
  except: return False

local_tok_dir = None
if is_local_dir(qwen_tok):
  local_tok_dir = qwen_tok
  print(f"[Step1] Using local tokenizer dir: {local_tok_dir}")
else:
  from huggingface_hub import snapshot_download
  print(f"[Step1] Downloading tokenizer files for {qwen_tok}...")
  snapshot_path = snapshot_download(
    repo_id=qwen_tok,
    allow_patterns=[
      'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
      'vocab.json', 'merges.txt', 'tokenizer.model', '*.model', '*.txt'
    ],
    token="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",
    resume_download=True,
    local_dir_use_symlinks=False,
    max_workers=1,
  )
  local_tok_dir = snapshot_path
  print(f"[Step1] Tokenizer snapshot at: {snapshot_path}")

try:
  tok = AutoTokenizer.from_pretrained(local_tok_dir, trust_remote_code=True, use_fast=True)
  print(f"[Step1] Prefetched tokenizer: vocab size {len(tok)}")
except Exception as e:
  print(f"[Step1] Failed to load tokenizer: {e}")
  sys.exit(2)
PY

echo "[Step1] Starting post-pretraining (LoRA/QLoRA)"

RUN_STAGE=${RUN_STAGE:-gemma}
OUT_BASE=${OUT_BASE:-"${SCRATCH_BASE}"}
echo "[Step1] Selected stage: ${RUN_STAGE}"

case "${RUN_STAGE}" in
  gemma)
    require_space_gb 40
    clean_hf_cache
    echo "[Step1] Gemma2-9B LoRA: starting"
    POSTPRE_MAX_STEPS=100
    GEMMA_ARGS=(
      --base-model "${GEMMA_BASE}"
      --output-dir "${OUT_BASE}/post_pretrain_gemma2-9b_lora"
      --data-path "${UT_DATA}"
      --tokenizer-path "${GEMMA_TOK}"
      --bf16
      --attn-impl eager
      --load-8bit
      --grad-accum "${POSTPRE_GRAD_ACCUM:-16}"
      --subset-size "${POSTPRE_SUBSET_SIZE:-0}"
      --epochs "${POSTPRE_EPOCHS}"
      --lr "${POSTPRE_LR}"
      --max-length "${POSTPRE_MAXLEN}"
      --r "${POSTPRE_LORA_R}"
      --lora-alpha "${POSTPRE_LORA_ALPHA}"
      --per-device-batch "${POSTPRE_PER_DEVICE_BS}"
    )
    if [ "${POSTPRE_DISABLE_CHKPT:-0}" -ne 1 ]; then
      GEMMA_ARGS+=( --grad-checkpoint )
    fi
    if [ "${POSTPRE_MAX_STEPS}" != "-1" ]; then
      GEMMA_ARGS+=( --max-steps "${POSTPRE_MAX_STEPS}" )
    fi
    echo "[Step1] Gemma args: ${GEMMA_ARGS[*]}"
    if [ "${POSTPRE_MERGE_ONLY:-0}" -eq 1 ]; then
      echo "[Step1][Skip] POSTPRE_MERGE_ONLY=1 set; skipping Gemma training." >&2
    elif [ -d "${OUT_BASE}/post_pretrain_gemma2-9b_lora" ] && [ "${POSTPRE_SKIP_IF_EXISTS:-0}" -eq 1 ]; then
      echo "[Step1][Skip] Gemma LoRA dir exists and POSTPRE_SKIP_IF_EXISTS=1; skipping training." >&2
    else
      python lora_train.py "${GEMMA_ARGS[@]}"
    fi

    if [ -d "${OUT_BASE}/post_pretrain_gemma2-9b_lora" ]; then
      python lora_merge.py \
        --base-model "${GEMMA_BASE}" \
        --lora-dir "${OUT_BASE}/post_pretrain_gemma2-9b_lora" \
        --out-dir "${OUT_BASE}/post_pretrain_gemma2-9b_merged"
    else
      echo "[Step1][Warn] Gemma LoRA dir missing; merge skipped." >&2
    fi
    ;;

  qwen)
    # Secondary safeguard: if user didn't force max steps and auto produced >100, clamp to 100
    if [ -z "${POSTPRE_FORCE_MAX_STEPS}" ] && [ "${POSTPRE_MAX_STEPS}" != "-1" ] && [ "${POSTPRE_MAX_STEPS}" -gt 100 ]; then
      echo "[Step1][Clamp] Reducing Qwen max steps from ${POSTPRE_MAX_STEPS} to 100 to fit time budget." >&2
      POSTPRE_MAX_STEPS=100
    fi
    require_space_gb 180
    clean_hf_cache
    echo "[Step1] ${QWEN_BASE} QLoRA: starting"
    QWEN_ARGS=(
      --base-model "${QWEN_BASE}"
      --output-dir "${OUT_BASE}/post_pretrain_qwen2-72b_lora"
      --data-path "${UT_DATA}"
      --tokenizer-path "${QWEN_TOK}"
      --bf16
      --qlora
      --epochs "${POSTPRE_EPOCHS}"
      --lr "${POSTPRE_LR}"
      --max-length "${POSTPRE_MAXLEN}"
      --r "${POSTPRE_LORA_R}"
      --lora-alpha "${POSTPRE_LORA_ALPHA}"
      --per-device-batch "${POSTPRE_PER_DEVICE_BS}"
    )
    if [ -n "${POSTPRE_GRAD_ACCUM:-}" ]; then
      QWEN_ARGS+=( --grad-accum "${POSTPRE_GRAD_ACCUM}" )
    fi
    if [ -n "${POSTPRE_SUBSET_SIZE:-}" ]; then
      QWEN_ARGS+=( --subset-size "${POSTPRE_SUBSET_SIZE}" )
    fi
    if [ "${POSTPRE_MAX_STEPS}" != "-1" ]; then
      QWEN_ARGS+=( --max-steps "${POSTPRE_MAX_STEPS}" )
    fi
    if [ "${POSTPRE_DISABLE_CHKPT:-0}" -ne 1 ]; then
      QWEN_ARGS+=( --grad-checkpoint )
    fi
    echo "[Step1] Qwen args: ${QWEN_ARGS[*]}"
    if [ "${POSTPRE_MERGE_ONLY:-0}" -eq 1 ]; then
      echo "[Step1][Skip] POSTPRE_MERGE_ONLY=1 set; skipping Qwen training." >&2
    elif [ -d "${OUT_BASE}/post_pretrain_qwen2-72b_lora" ] && [ "${POSTPRE_SKIP_IF_EXISTS:-0}" -eq 1 ]; then
      echo "[Step1][Skip] Qwen LoRA dir exists and POSTPRE_SKIP_IF_EXISTS=1; skipping training." >&2
    else
      python lora_train.py "${QWEN_ARGS[@]}"
    fi

    if [ -d "${OUT_BASE}/post_pretrain_qwen2-72b_lora" ]; then
      python lora_merge.py \
        --base-model "${QWEN_BASE}" \
        --lora-dir "${OUT_BASE}/post_pretrain_qwen2-72b_lora" \
        --out-dir "${OUT_BASE}/post_pretrain_qwen2-72b_merged"
    else
      echo "[Step1][Warn] Qwen LoRA dir missing; merge skipped." >&2
    fi
    ;;

  llama)
    # Secondary safeguard for LLaMA
    if [ -z "${POSTPRE_FORCE_MAX_STEPS}" ] && [ "${POSTPRE_MAX_STEPS}" != "-1" ] && [ "${POSTPRE_MAX_STEPS}" -gt 200 ]; then
      echo "[Step1][Clamp] Reducing LLaMA max steps from ${POSTPRE_MAX_STEPS} to 200 to fit time budget." >&2
      POSTPRE_MAX_STEPS=200
    fi
    require_space_gb 180
    clean_hf_cache
    echo "[Step1] ${LLAMA_BASE} QLoRA: starting"
    LLAMA_TOK_RESOLVED="${LLAMA_TOK}"
    if [ ! -d "${LLAMA_TOK}" ]; then
      LLAMA_TOK_RESOLVED=$(python - <<'PY'
import os
from huggingface_hub import snapshot_download
tok=os.environ['LLAMA_TOK']
print(snapshot_download(
  repo_id=tok,
  allow_patterns=['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','tokenizer.model','*.model','*.txt'],
  token="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",
  resume_download=True,
  local_dir_use_symlinks=False,
  max_workers=1,
))
PY
)
      echo "[Step1] LLaMA tokenizer snapshot at: ${LLAMA_TOK_RESOLVED}"
    fi

    # Dynamic output directory naming based on detected model size (legacy name kept for 70B)
    if [[ "${LLAMA_BASE}" == *"8B"* ]]; then
      LLAMA_TAG="llama3-8b"
    elif [[ "${LLAMA_BASE}" == *"70B"* ]]; then
      LLAMA_TAG="llama3-70b"
    else
      # Fallback: derive tag from final component after last slash
      LLAMA_TAG=$(basename "${LLAMA_BASE}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g')
    fi
    LLAMA_LORA_DIR="${OUT_BASE}/post_pretrain_${LLAMA_TAG}_lora"
    LLAMA_MERGED_DIR="${OUT_BASE}/post_pretrain_${LLAMA_TAG}_merged"
    # Backward compatibility: if expecting 8B but only legacy 70b dir exists (user previously trained with old naming), use it.
    if [[ "${LLAMA_TAG}" == "llama3-8b" && ! -d "${LLAMA_LORA_DIR}" && -d "${OUT_BASE}/post_pretrain_llama3-70b_lora" ]]; then
      echo "[Step1][Compat] Using legacy directory name post_pretrain_llama3-70b_lora for 8B model (will save merged likewise)." >&2
      LLAMA_LORA_DIR="${OUT_BASE}/post_pretrain_llama3-70b_lora"
      LLAMA_MERGED_DIR="${OUT_BASE}/post_pretrain_llama3-70b_merged"
    fi

    LLAMA_ARGS=(
      --base-model "${LLAMA_BASE}"
      --output-dir "${LLAMA_LORA_DIR}"
      --data-path "${UT_DATA}"
      --tokenizer-path "${LLAMA_TOK_RESOLVED}"
      --bf16
      --qlora
      --epochs "${POSTPRE_EPOCHS}"
      --lr "${POSTPRE_LR}"
      --max-length "${POSTPRE_MAXLEN}"
      --r "${POSTPRE_LORA_R}"
      --lora-alpha "${POSTPRE_LORA_ALPHA}"
      --per-device-batch "${POSTPRE_PER_DEVICE_BS}"
    )
    if [ -n "${POSTPRE_GRAD_ACCUM:-}" ]; then
      LLAMA_ARGS+=( --grad-accum "${POSTPRE_GRAD_ACCUM}" )
    fi
    if [ -n "${POSTPRE_SUBSET_SIZE:-}" ]; then
      LLAMA_ARGS+=( --subset-size "${POSTPRE_SUBSET_SIZE}" )
    fi
    if [ "${POSTPRE_MAX_STEPS}" != "-1" ]; then
      LLAMA_ARGS+=( --max-steps "${POSTPRE_MAX_STEPS}" )
    fi
    if [ "${POSTPRE_DISABLE_CHKPT:-0}" -ne 1 ]; then
      LLAMA_ARGS+=( --grad-checkpoint )
    fi
    echo "[Step1] LLaMA args: ${LLAMA_ARGS[*]}"
    if [ "${POSTPRE_MERGE_ONLY:-0}" -eq 1 ]; then
      echo "[Step1][Skip] POSTPRE_MERGE_ONLY=1 set; skipping LLaMA training." >&2
    elif [ -d "${LLAMA_LORA_DIR}" ] && [ "${POSTPRE_SKIP_IF_EXISTS:-0}" -eq 1 ]; then
      echo "[Step1][Skip] LLaMA LoRA dir ${LLAMA_LORA_DIR} exists and POSTPRE_SKIP_IF_EXISTS=1; skipping training." >&2
    else
      python lora_train.py "${LLAMA_ARGS[@]}"
    fi

    if [ -d "${LLAMA_LORA_DIR}" ]; then
      python lora_merge.py \
        --base-model "${LLAMA_BASE}" \
        --lora-dir "${LLAMA_LORA_DIR}" \
        --out-dir "${LLAMA_MERGED_DIR}"
    else
      echo "[Step1][Warn] LLaMA LoRA dir missing (${LLAMA_LORA_DIR}); merge skipped." >&2
    fi
    ;;

  *)
    echo "[Step1] Unknown RUN_STAGE='${RUN_STAGE}'. Use one of: gemma | qwen | llama" >&2
    exit 2
    ;;
esac

echo "[Step1] Stage '${RUN_STAGE}' completed"
