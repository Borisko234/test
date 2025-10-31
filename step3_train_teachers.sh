#!/bin/bash
# Step 3: Train teacher models per fold with 4/5 Kaggle train + external 33k augmentation, then merge adapters.
# Usage:
#   sbatch step3_train_teachers.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=360   # 6 hours wall-clock limit
#SBATCH --cpus-per-task=8
#SBATCH --job-name=S3_Teachers
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
export HF_DATASETS_CACHE="${PWD}/.hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resolve SCRATCH_BASE robustly (mirrors step1 logic) to avoid unwritable /scratch permission errors
echo "[Step3] Resolving SCRATCH_BASE..."
SCRATCH_BASE=${SCRATCH_BASE:-/scratch-shared/tc1proj005}
if [ ! -d "${SCRATCH_BASE}" ] || [ ! -w "${SCRATCH_BASE}" ]; then
  if [ -n "${SLURM_TMPDIR:-}" ] && [ -d "${SLURM_TMPDIR}" ] && [ -w "${SLURM_TMPDIR}" ]; then
    echo "[Step3] Primary SCRATCH_BASE not writable; using SLURM_TMPDIR=${SLURM_TMPDIR}" >&2
    SCRATCH_BASE="${SLURM_TMPDIR}"
  elif [ -d "/scratch-shared/tc1proj005" ] && [ -w "/scratch-shared/tc1proj005" ]; then
    echo "[Step3] Falling back to /scratch-shared/tc1proj005" >&2
    SCRATCH_BASE="/scratch-shared/tc1proj005"
  else
    echo "[Step3][Error] No writable scratch directory found (tried '${SCRATCH_BASE}', SLURM_TMPDIR, /scratch-shared/tc1proj005)." >&2
    exit 97
  fi
fi
echo "[Step3] Using SCRATCH_BASE=${SCRATCH_BASE}"

# Hugging Face gated model login (only if token available). You can override HF_TOKEN externally.
export HF_TOKEN="${HF_TOKEN:-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}"
python - <<'PY'
import os
tok=os.environ.get('HF_TOKEN')
if tok:
  try:
    from huggingface_hub import login
    login(token=tok, add_to_git_credential=False)
    print('[Step3] Hugging Face login succeeded (token provided).')
  except Exception as e:
    print('[Step3][Warn] HF login failed:', e)
else:
  print('[Step3] No HF_TOKEN set; proceeding without login.')
PY

KF=data/train.csv
EXT=data/lmsys-33k-deduplicated.csv
FOLDS_JSON=${FOLDS_JSON:-data/processed_data/folds_3_seed42.json}
if [ ! -f "$FOLDS_JSON" ]; then
  # Try scratch locations
  for cand in "${SCRATCH_BASE}/folds/folds_3_seed42.json" "${SCRATCH_BASE}/folds_3_seed42.json"; do
    if [ -f "$cand" ]; then
      echo "[Step3] Found folds file at scratch location: $cand"; FOLDS_JSON="$cand"; break
    fi
  done
fi
if [ ! -f "$FOLDS_JSON" ]; then
  echo "[Step3] Missing folds file (tried: ${FOLDS_JSON} and scratch). Run step2_make_folds.sh first." >&2
  exit 3
fi

echo "[Step3] Using Kaggle train: $KF | External: $EXT"
if [ "${TEACHER_SKIP_PREP:-0}" = "1" ]; then
  echo "[Step3][Prep] Skipping fold CSV generation (TEACHER_SKIP_PREP=1). Assuming existing data/fold_data/*.csv present."
else
  echo "[Step3][Prep] Starting fold CSV generation..."
  python -u <<'PY'
import json, pandas as pd, os
KF='data/train.csv'
EXT='data/lmsys-33k-deduplicated.csv'
FOLDS_JSON=os.environ.get('FOLDS_JSON','data/processed_data/folds_3_seed42.json')
import traceback, sys, time, shutil
sel_spec = os.environ.get('TEACHER_FOLDS','all')
start=time.time()
try:
  if not os.path.isfile(FOLDS_JSON):
    raise FileNotFoundError(f'Folds JSON not found: {FOLDS_JSON}')
  with open(FOLDS_JSON,'r') as f:
    folds=json.load(f)
  print(f"[Step3][Prep] Loaded folds json with keys: {list(folds.keys())}", flush=True)
  if sel_spec != 'all':
    # normalize selection string
    wanted=[s for s in sel_spec.replace(',', ' ').split() if s.strip()!='']
    try:
      wanted_ints=set(int(x) for x in wanted)
    except ValueError:
      raise ValueError(f"Invalid TEACHER_FOLDS specification: {sel_spec}")
    folds={k:v for k,v in folds.items() if int(k) in wanted_ints}
    print(f"[Step3][Prep] Filtering to folds {sorted(folds.keys())} per TEACHER_FOLDS={sel_spec}", flush=True)
  print('[Step3][Prep] Reading Kaggle train CSV...', flush=True)
  train_df=pd.read_csv(KF)
  print(f"[Step3][Prep] Loaded Kaggle train df: {train_df.shape}", flush=True)
  print('[Step3][Prep] Reading external CSV...', flush=True)
  ext_df=pd.read_csv(EXT)
  print(f"[Step3][Prep] Loaded external df: {ext_df.shape}", flush=True)
  def normalize(df):
    cols=df.columns
    out=df.copy()
    if 'prompt' not in cols:
      text_cols=[c for c in cols if df[c].dtype==object]
      if text_cols:
        out['prompt']=df[text_cols[0]]
    if 'response' not in cols:
      cand=[c for c in ['chosen','answer','response','completion'] if c in cols]
      if cand:
        out['response']=df[cand[0]]
    return out
  train_df=normalize(train_df)
  ext_df=normalize(ext_df)
  os.makedirs('data/fold_data', exist_ok=True)
  total=len(folds)
  done=0
  scratch_root=os.environ.get('SCRATCH_BASE','/scratch-shared/tc1proj005')
  write_root=os.environ.get('TEACHER_FOLD_WRITE_ROOT', os.path.join(scratch_root,'fold_data'))
  os.makedirs(write_root, exist_ok=True)
  print(f"[Step3][Prep] Writing fold CSVs initially to {write_root} (will mirror to data/fold_data)", flush=True)
  for k, val_idx in folds.items():
    done+=1
    t0=time.time()
    print(f"[Step3][Prep][Fold {k}] Start generation (val idx count={len(val_idx)})", flush=True)
    # Build validation set
    val_set=train_df.iloc[val_idx]
    print(f"[Step3][Prep][Fold {k}] Built val set shape={val_set.shape}", flush=True)
    # Train indices (avoid set difference overhead for large lists by boolean mask)
    mask=[True]*len(train_df)
    for vi in val_idx:
      mask[vi]=False
    tr_part=train_df[mask]
    print(f"[Step3][Prep][Fold {k}] Train subset shape={tr_part.shape}", flush=True)
    combined=pd.concat([tr_part, ext_df], ignore_index=True)
    print(f"[Step3][Prep][Fold {k}] Combined shape={combined.shape}", flush=True)
    print(f"[Step3][Prep][Fold {k}] Composition: kaggle_train_rows={tr_part.shape[0]} external_rows={ext_df.shape[0]} expected_total={tr_part.shape[0]+ext_df.shape[0]}", flush=True)
    # Write per-fold CSVs with robust fallback if scratch has I/O issues
    proj_train=f'data/fold_data/fold_{k}_train.csv'
    proj_val=f'data/fold_data/fold_{k}_val.csv'
    os.makedirs('data/fold_data', exist_ok=True)

    def try_write(out_dir):
      os.makedirs(out_dir, exist_ok=True)
      t_path=os.path.join(out_dir, f'fold_{k}_train.csv')
      v_path=os.path.join(out_dir, f'fold_{k}_val.csv')
      combined.to_csv(t_path, index=False)
      val_set.to_csv(v_path, index=False)
      return t_path, v_path

    fold_train_tmp = fold_val_tmp = None
    tried = []
    # Attempt 1: configured scratch
    try:
      fold_train_tmp, fold_val_tmp = try_write(write_root)
      print(f"[Step3][Prep][Fold {k}] Wrote to scratch: {write_root}", flush=True)
    except Exception as e1:
      tried.append((write_root, str(e1)))
      # Attempt 2: local project dir
      local_dir='data/fold_data'
      try:
        fold_train_tmp, fold_val_tmp = try_write(local_dir)
        print(f"[Step3][Prep][Fold {k}] Scratch write failed; wrote to local: {local_dir}", flush=True)
      except Exception as e2:
        tried.append((local_dir, str(e2)))
        # Attempt 3: SLURM_TMPDIR if available
        tmpdir=os.environ.get('SLURM_TMPDIR') or ''
        if tmpdir:
          td=os.path.join(tmpdir, 'fold_data')
          try:
            fold_train_tmp, fold_val_tmp = try_write(td)
            print(f"[Step3][Prep][Fold {k}] Local write failed; wrote to SLURM_TMPDIR: {td}", flush=True)
          except Exception as e3:
            tried.append((td, str(e3)))
        if fold_train_tmp is None or fold_val_tmp is None:
          print(f"[Step3][Prep][Fold {k}][ERROR] All write attempts failed: {tried}", flush=True)
          raise

    # Mirror to project data directory (if not already there)
    try:
      if os.path.abspath(os.path.dirname(fold_train_tmp)) != os.path.abspath('data/fold_data'):
        shutil.copy2(fold_train_tmp, proj_train)
        shutil.copy2(fold_val_tmp, proj_val)
      else:
        # Already written to project dir
        pass
    except Exception as e_cp:
      print(f"[Step3][Prep][Fold {k}][Warn] Copy to project dir failed: {e_cp}", flush=True)
    try:
      sz_train=os.path.getsize(fold_train_tmp)
      sz_val=os.path.getsize(fold_val_tmp)
    except Exception:
      # Fallback to project copies for size reporting
      sz_train=os.path.getsize(proj_train) if os.path.isfile(proj_train) else 0
      sz_val=os.path.getsize(proj_val) if os.path.isfile(proj_val) else 0
    print(f"[Step3][Prep][Fold {k}] Wrote train={sz_train/1e6:.2f}MB val={sz_val/1e6:.2f}MB | {done}/{total} | {(time.time()-t0):.2f}s", flush=True)
  print('[Step3][Prep] Completed fold CSV generation in %.1fs' % (time.time()-start), flush=True)
except Exception as e:
  print('[Step3][Prep][ERROR]', e)
  traceback.print_exc()
  sys.exit(91)
PY
fi

# Verify at least one fold CSV exists else abort with context
if [ ! -s data/fold_data/fold_0_train.csv ]; then
  echo "[Step3][Error] fold_0_train.csv not found or empty after prep (TEACHER_SKIP_PREP=${TEACHER_SKIP_PREP:-0})." >&2
  ls -l data/fold_data 2>/dev/null || true
  exit 92
fi

# Copy fold CSVs to scratch if requested / available
SCRATCH_FOLD_DIR="${SCRATCH_BASE}/fold_data"
mkdir -p "${SCRATCH_FOLD_DIR}" || true
for k in 0 1 2 3 4; do
  src_train="data/fold_data/fold_${k}_train.csv"
  src_val="data/fold_data/fold_${k}_val.csv"
  if [ -f "$src_train" ]; then
    cp -f "$src_train" "${SCRATCH_FOLD_DIR}/" && echo "[Step3][Sync] Copied $src_train -> ${SCRATCH_FOLD_DIR}/" || echo "[Step3][Warn] Failed copy $src_train"
  fi
  if [ -f "$src_val" ]; then
    cp -f "$src_val" "${SCRATCH_FOLD_DIR}/" && echo "[Step3][Sync] Copied $src_val -> ${SCRATCH_FOLD_DIR}/" || echo "[Step3][Warn] Failed copy $src_val"
  fi
done

# Robust resolution of local post-pretrained base model directories.
# We search in (1) explicit override, (2) model_save/, (3) repo root, (4) SCRATCH_BASE.
resolve_base_dir() {
  local override="$1"; shift
  local -a names=("$@")
  if [ -n "$override" ] && [ -d "$override" ]; then
    echo "$override"; return 0
  fi
  for n in "${names[@]}"; do
    for base in "." "model_save" "${SCRATCH_BASE}"; do
      if [ -d "${base}/${n}" ]; then
        echo "${base}/${n}"; return 0
      fi
    done
  done
  echo ""  # not found
}

LLAMA_CANDIDATES=(post_pretrain_llama3-8b_merged post_pretrain_llama3-70b_merged)
QWEN_CANDIDATES=(post_pretrain_qwen2-14b_merged post_pretrain_qwen2-72b_merged)

LLAMA_BASE_DIR=$(resolve_base_dir "${LLAMA_FT_BASE:-}" "${LLAMA_CANDIDATES[@]}")
QWEN_BASE_DIR=$(resolve_base_dir "${QWEN_FT_BASE:-}" "${QWEN_CANDIDATES[@]}")

SKIP_LLAMA=${SKIP_LLAMA:-0}
SKIP_QWEN=${SKIP_QWEN:-0}

if [ -z "$LLAMA_BASE_DIR" ]; then
  if [ "$SKIP_LLAMA" = "1" ]; then
    echo "[Step3][Info] SKIP_LLAMA=1 and no local LLaMA base found; LLaMA folds will be skipped." >&2
  else
    echo "[Step3][Error] No local LLaMA post-pretrained directory found. Set LLAMA_FT_BASE=/path/to/post_pretrain_llama*_merged or export SKIP_LLAMA=1 to bypass." >&2
    exit 41
  fi
fi
REQUIRE_LARGE=${TEACHER_REQUIRE_LARGE:-0}
if [ "$REQUIRE_LARGE" = "1" ]; then
  # Enforce that discovered dirs correspond to 70B / 72B variants
  if [[ "$LLAMA_BASE_DIR" != *"70b"* && "$LLAMA_BASE_DIR" != *"70B"* ]]; then
    echo "[Step3][Error] TEACHER_REQUIRE_LARGE=1 but LLaMA base is not a 70B directory: $LLAMA_BASE_DIR" >&2
    exit 61
  fi
  if [[ "$QWEN_BASE_DIR" != *"72b"* && "$QWEN_BASE_DIR" != *"72B"* ]]; then
    echo "[Step3][Error] TEACHER_REQUIRE_LARGE=1 but Qwen base is not a 72B directory: $QWEN_BASE_DIR" >&2
    exit 62
  fi
  echo "[Step3] Large-model enforcement active (70B / 72B confirmed)."
fi
if [ -z "$QWEN_BASE_DIR" ]; then
  if [ "$SKIP_QWEN" = "1" ]; then
    echo "[Step3][Info] SKIP_QWEN=1 and no local Qwen base found; Qwen folds will be skipped." >&2
  else
    echo "[Step3][Error] No local Qwen post-pretrained directory found. Set QWEN_FT_BASE=/path/to/post_pretrain_qwen*_merged or export SKIP_QWEN=1 to bypass." >&2
    exit 42
  fi
fi

# Tokenizer paths (local merged dirs typically lack tokenizer files)
LLAMA_TOKENIZER=${LLAMA_TOKENIZER_PATH:-meta-llama/Meta-Llama-3.1-8B}
QWEN_TOKENIZER=${QWEN_TOKENIZER_PATH:-Qwen/Qwen2.5-14B}

echo "[Step3] LLaMA base: ${LLAMA_BASE_DIR:-<skipped>} | tokenizer: $LLAMA_TOKENIZER | skip=$SKIP_LLAMA"
echo "[Step3] Qwen base:  ${QWEN_BASE_DIR:-<skipped>} | tokenizer: $QWEN_TOKENIZER | skip=$SKIP_QWEN"

# Root directory where trained fold artifacts (LoRA + merged) will be written.
# Default now points to SCRATCH_BASE/folds so heavy outputs live on scratch.
# Override with TEACHER_SAVE_ROOT if needed.
SAVE_ROOT=${TEACHER_SAVE_ROOT:-${SCRATCH_BASE}/folds}
mkdir -p "${SAVE_ROOT}" || { echo "[Step3][Error] Could not create SAVE_ROOT=${SAVE_ROOT}" >&2; exit 50; }
# Export so that inline Python manifest snippets can access it; previously this was not exported causing root=None.
export SAVE_ROOT
echo "[Step3] Teacher artifacts will be saved under: ${SAVE_ROOT}" 
mkdir -p model_save || true  # ensure local dir exists for symlink compatibility

LORA_R=${TEACHER_LORA_R:-16}
LORA_ALPHA=${TEACHER_LORA_ALPHA:-32}
MAXLEN=${TEACHER_MAXLEN:-512}
# For fastest training: use minimal accumulation and a larger per-device batch by default.
# Override with TEACHER_GRAD_ACCUM / TEACHER_PER_DEVICE_BS if you hit OOM on your GPU.
GRAD_ACCUM=${TEACHER_GRAD_ACCUM:-8}
BS=${TEACHER_PER_DEVICE_BS:-2}
EPOCHS=${TEACHER_EPOCHS:-1}
LR=${TEACHER_LR:-1e-5}
SUBSET=${TEACHER_SUBSET_SIZE:--1}
# Default to epoch-driven training: no max-steps unless explicitly provided
MAX_STEPS=${TEACHER_MAX_STEPS:--1}
# Time budget auto-capping disabled by default; set >0 to enable custom logic externally
TIME_BUDGET_HOURS=${TEACHER_TIME_BUDGET_HOURS:--1}

# Epoch-first training: we do not auto-cap by time unless explicitly configured.
# If you want time-based capping, export TEACHER_MAX_STEPS>0 and/or TEACHER_TIME_BUDGET_HOURS>0.

# Build optional extra args for lora_train.py based on user overrides
COMMON_EXTRA_ARGS=()
if [ -n "${MAX_STEPS}" ] 2>/dev/null && [ "${MAX_STEPS}" -gt 0 ] 2>/dev/null; then
  COMMON_EXTRA_ARGS+=(--max-steps "${MAX_STEPS}")
fi
if [ -n "${SUBSET}" ] 2>/dev/null && [ "${SUBSET}" -gt 0 ] 2>/dev/null; then
  COMMON_EXTRA_ARGS+=(--subset-size "${SUBSET}")
fi

###############################################
# Fold selection logic
# Use TEACHER_FOLDS to restrict which folds run.
# Examples:
#   TEACHER_FOLDS=0            -> only fold 0
#   TEACHER_FOLDS="1,3"        -> folds 1 and 3
#   TEACHER_FOLDS="2 4"        -> folds 2 and 4
#   (unset)                    -> all folds 0..4
FOLD_SPEC=${TEACHER_FOLDS:-all}
if [ "$FOLD_SPEC" = "all" ]; then
  FOLD_LIST="0 1 2 3 4"
else
  # Normalize commas to spaces
  FOLD_LIST=$(echo "$FOLD_SPEC" | tr ',' ' ')
fi
echo "[Step3] Selected folds: $FOLD_LIST (from TEACHER_FOLDS='${FOLD_SPEC}')"
EXPOSE_ROOT=${TEACHER_EXPOSE_ROOT:-1}
if [ "$EXPOSE_ROOT" = "1" ]; then
  echo "[Step3] Root-level symlinks for fold models will be created (TEACHER_EXPOSE_ROOT=1)."
else
  echo "[Step3] Skipping root-level symlinks (TEACHER_EXPOSE_ROOT=${EXPOSE_ROOT})."
fi
DISABLE_SYMLINKS=${TEACHER_DISABLE_SYMLINKS:-0}
if [ "$DISABLE_SYMLINKS" = "1" ]; then
  echo "[Step3] Symlink creation disabled (TEACHER_DISABLE_SYMLINKS=1) — artifacts will exist only under SAVE_ROOT=${SAVE_ROOT}."
fi
MIRROR_MODEL_SAVE=${TEACHER_MIRROR_MODEL_SAVE:-0}
REQUIRE_WEIGHTS=${TEACHER_REQUIRE_WEIGHTS:-0}
ACCEPT_SHARDS=${TEACHER_ACCEPT_SHARDS:-1}
if [ "$MIRROR_MODEL_SAVE" = "1" ]; then
  echo "[Step3] Will mirror merged & lora dirs into model_save/ (TEACHER_MIRROR_MODEL_SAVE=1)."
fi
if [ "$REQUIRE_WEIGHTS" = "1" ]; then
  echo "[Step3] Will enforce presence of model.safetensors or pytorch_model.bin (TEACHER_REQUIRE_WEIGHTS=1)."
fi

check_weight_dir() {
  local d="$1"; local label="$2"; local has=0
  if [ -f "$d/model.safetensors" ] || [ -f "$d/pytorch_model.bin" ]; then
    has=1
  fi
  if [ $has -eq 0 ] && [ "$ACCEPT_SHARDS" = "1" ]; then
    # Detect sharded safetensors pattern
    if ls "$d"/model-*-of-*.safetensors >/dev/null 2>&1; then
      has=2
      echo "[Step3][OK] Sharded safetensors detected in $label ($d)"
      # Optionally note shard count
      local shard_count=$(ls "$d"/model-*-of-*.safetensors 2>/dev/null | wc -l | tr -d ' ') 
      echo "[Step3][Info] $label shard count: ${shard_count}"
    elif [ -f "$d/model.safetensors.index.json" ]; then
      has=2
      echo "[Step3][OK] Index file model.safetensors.index.json detected in $label ($d)"
    fi
  fi
  if [ $has -eq 1 ]; then
    echo "[Step3][OK] Weights present in $label ($d)"
  elif [ $has -eq 2 ]; then
    echo "[Step3][OK] Weights present (sharded) in $label ($d)"
  else
    echo "[Step3][Warn] No model.safetensors or pytorch_model.bin found in $label ($d)" >&2
    if [ "$REQUIRE_WEIGHTS" = "1" ]; then
      echo "[Step3][Error] Required weight file missing for $label; aborting." >&2
      exit 60
    fi
  fi
}

mirror_dir_if_requested() {
  local src="$1"; local name="$2"
  if [ "$MIRROR_MODEL_SAVE" = "1" ]; then
    local dst="model_save/${name}"
    rm -rf "${dst}.tmp" 2>/dev/null || true
    mkdir -p model_save || true
    # Prefer rsync if available for speed / incremental sync
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "$src/" "${dst}.tmp/" && mv -f "${dst}.tmp" "$dst" && echo "[Step3][Mirror] rsync $src -> $dst"
    else
      cp -a "$src" "${dst}.tmp" && rm -rf "$dst" && mv "${dst}.tmp" "$dst" && echo "[Step3][Mirror] cp -a $src -> $dst"
    fi
  fi
}

for FOLD in $FOLD_LIST; do
  export FOLD
  TRAIN_CSV=data/fold_data/fold_${FOLD}_train.csv
  if [ "$SKIP_LLAMA" != "1" ]; then
    echo "[Step3] LLaMA fold ${FOLD} train csv: ${TRAIN_CSV}"
    LL_OUT_LORA="${SAVE_ROOT}/llama_fold_${FOLD}_lora"
    LL_OUT_MERGED="${SAVE_ROOT}/llama_fold_${FOLD}"
    if [ "${TEACHER_SKIP_TRAIN:-0}" = "1" ] && [ -d "${LL_OUT_LORA}" ]; then
      echo "[Step3][SkipTrain] Skipping LLaMA training for fold ${FOLD} (TEACHER_SKIP_TRAIN=1 and LoRA dir exists)"
    else
      python lora_train.py \
        --base-model "${LLAMA_BASE_DIR}" \
        --tokenizer-path "${LLAMA_TOKENIZER}" \
        --output-dir "${LL_OUT_LORA}" \
        --data-path "${TRAIN_CSV}" \
        --bf16 \
        --qlora \
        --r "${LORA_R}" \
        --lora-alpha "${LORA_ALPHA}" \
        --max-length "${MAXLEN}" \
        --grad-accum "${GRAD_ACCUM}" \
        --per-device-batch "${BS}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        "${COMMON_EXTRA_ARGS[@]}" || { echo "[Step3][Error] LLaMA fold ${FOLD} failed"; exit 11; }
    fi
    python lora_merge.py \
      --base-model "${LLAMA_BASE_DIR}" \
      --lora-dir "${LL_OUT_LORA}" \
      --out-dir "${LL_OUT_MERGED}" || { echo "[Step3][Error] LLaMA merge fold ${FOLD} failed"; exit 12; }
    # Symlink creation removed as requested
    check_weight_dir "${LL_OUT_MERGED}" "LLaMA merged fold ${FOLD}"
    mirror_dir_if_requested "${LL_OUT_MERGED}" "llama_fold_${FOLD}"
    mirror_dir_if_requested "${LL_OUT_LORA}" "llama_fold_${FOLD}_lora"
    # Append/update manifest
    python - <<PY
import os, json, time, glob, sys
root=os.environ.get('SAVE_ROOT')
fold_env=os.environ.get('FOLD')
if not root:
    print('[Step3][Manifest][Skip] SAVE_ROOT not set in environment; skipping llama manifest update.')
    sys.exit(0)
try:
    fold=int(fold_env)
except (TypeError, ValueError):
    print(f'[Step3][Manifest][Skip] Invalid FOLD={fold_env!r}; skipping llama manifest update.')
    sys.exit(0)
merged=os.path.join(root,f'llama_fold_{fold}')
summary=os.path.join(merged,'training_summary.json')
manifest_path=os.path.join(root,'manifest.json')
entry={
  'model':'llama',
  'fold':fold,
  'merged_dir':merged,
  'lora_dir':os.path.join(root,f'llama_fold_{fold}_lora'),
  'timestamp':time.time(),
  'size_bytes': sum(os.path.getsize(p) for p in glob.glob(os.path.join(merged,'**'), recursive=True) if os.path.isfile(p)) if os.path.isdir(merged) else None,
}
try:
  if os.path.isfile(summary):
    with open(summary,'r') as f: entry['training_summary']=json.load(f)
except Exception as e:
  entry['training_summary_error']=str(e)
data=[]
if os.path.isfile(manifest_path):
  try:
    with open(manifest_path,'r') as f: data=json.load(f)
  except Exception:
    data=[]
data=[d for d in data if not (d.get('model')=='llama' and d.get('fold')==fold)]
data.append(entry)
try:
  with open(manifest_path,'w') as f: json.dump(data,f,indent=2)
  print(f'[Step3][Manifest] Updated {manifest_path} with llama fold {fold}')
except Exception as e:
  print(f'[Step3][Manifest][Warn] Could not write manifest: {e}')
PY
  else
    echo "[Step3][Skip] LLaMA fold ${FOLD} due to SKIP_LLAMA=1"
  fi

  if [ "$SKIP_QWEN" != "1" ]; then
    echo "[Step3] Qwen fold ${FOLD} train csv: ${TRAIN_CSV}"
    QW_OUT_LORA="${SAVE_ROOT}/qwen_fold_${FOLD}_lora"
    QW_OUT_MERGED="${SAVE_ROOT}/qwen_fold_${FOLD}"
    if [ "${TEACHER_SKIP_TRAIN:-0}" = "1" ] && [ -d "${QW_OUT_LORA}" ]; then
      echo "[Step3][SkipTrain] Skipping Qwen training for fold ${FOLD} (TEACHER_SKIP_TRAIN=1 and LoRA dir exists)"
    else
      python lora_train.py \
        --base-model "${QWEN_BASE_DIR}" \
        --tokenizer-path "${QWEN_TOKENIZER}" \
        --output-dir "${QW_OUT_LORA}" \
        --data-path "${TRAIN_CSV}" \
        --bf16 \
        --qlora \
        --r "${LORA_R}" \
        --lora-alpha "${LORA_ALPHA}" \
        --max-length "${MAXLEN}" \
        --grad-accum "${GRAD_ACCUM}" \
        --per-device-batch "${BS}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        "${COMMON_EXTRA_ARGS[@]}" || { echo "[Step3][Error] Qwen fold ${FOLD} failed"; exit 21; }
    fi
    python lora_merge.py \
      --base-model "${QWEN_BASE_DIR}" \
      --lora-dir "${QW_OUT_LORA}" \
      --out-dir "${QW_OUT_MERGED}" || { echo "[Step3][Error] Qwen merge fold ${FOLD} failed"; exit 22; }
    # Symlink creation removed as requested
    check_weight_dir "${QW_OUT_MERGED}" "Qwen merged fold ${FOLD}"
    mirror_dir_if_requested "${QW_OUT_MERGED}" "qwen_fold_${FOLD}"
    mirror_dir_if_requested "${QW_OUT_LORA}" "qwen_fold_${FOLD}_lora"
    # Append/update manifest
    python - <<PY
import os, json, time, glob, sys
root=os.environ.get('SAVE_ROOT')
fold_env=os.environ.get('FOLD')
if not root:
    print('[Step3][Manifest][Skip] SAVE_ROOT not set in environment; skipping qwen manifest update.')
    sys.exit(0)
try:
    fold=int(fold_env)
except (TypeError, ValueError):
    print(f'[Step3][Manifest][Skip] Invalid FOLD={fold_env!r}; skipping qwen manifest update.')
    sys.exit(0)
merged=os.path.join(root,f'qwen_fold_{fold}')
summary=os.path.join(merged,'training_summary.json')
manifest_path=os.path.join(root,'manifest.json')
entry={
  'model':'qwen',
  'fold':fold,
  'merged_dir':merged,
  'lora_dir':os.path.join(root,f'qwen_fold_{fold}_lora'),
  'timestamp':time.time(),
  'size_bytes': sum(os.path.getsize(p) for p in glob.glob(os.path.join(merged,'**'), recursive=True) if os.path.isfile(p)) if os.path.isdir(merged) else None,
}
try:
  if os.path.isfile(summary):
    with open(summary,'r') as f: entry['training_summary']=json.load(f)
except Exception as e:
  entry['training_summary_error']=str(e)
data=[]
if os.path.isfile(manifest_path):
  try:
    with open(manifest_path,'r') as f: data=json.load(f)
  except Exception:
    data=[]
data=[d for d in data if not (d.get('model')=='qwen' and d.get('fold')==fold)]
data.append(entry)
try:
  with open(manifest_path,'w') as f: json.dump(data,f,indent=2)
  print(f'[Step3][Manifest] Updated {manifest_path} with qwen fold {fold}')
except Exception as e:
  print(f'[Step3][Manifest][Warn] Could not write manifest: {e}')
PY
  else
    echo "[Step3][Skip] Qwen fold ${FOLD} due to SKIP_QWEN=1"
  fi
done

echo "[Step3] Done"
