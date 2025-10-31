#!/bin/bash
# Step 4: Infer teacher distributions.
# For each fold model, compute:
#  1) Last-token logits snapshots (diagnostic)
#  2) Full response log-likelihood for response_a / response_b conditioned on prompt
#  3) Probability distribution over {A, B, Tie} via softmax with tie temperature heuristic
# Output per fold/model:
#   *_train_lasttok_logits.pt (existing)
#   *_train_logprobs.pt (tensor [N,3] => logprobs raw: logp_a, logp_b, logp_tie_placeholder)
#   *_train_probs.pt (tensor [N,3] => probs pA,pB,pTie)
# Usage:
#   sbatch step4_infer_teacher_logits.sh

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=360   # 6 hours
#SBATCH --cpus-per-task=8
#SBATCH --job-name=S4_TeacherInfer
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

python - <<'PY'
import os, json, torch, pandas as pd, glob, math, time
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FOLDS_SPEC = os.environ.get('INFER_FOLDS','all')
FOLDS = list(range(5)) if FOLDS_SPEC=='all' else [int(p) for p in FOLDS_SPEC.replace(',',' ').split() if p.isdigit()]
SCRATCH_BASE = os.environ.get('SCRATCH_BASE','/scratch-shared/tc1proj005')
SAVE_ROOT = os.environ.get('TEACHER_SAVE_ROOT', os.path.join(SCRATCH_BASE,'folds'))
DISABLE_SYMLINKS = os.environ.get('TEACHER_DISABLE_SYMLINKS','0')=='1'
INFER_MODELS = [m for m in os.environ.get('INFER_MODELS','llama,qwen').replace(',',' ').split() if m]
ENSEMBLE_OUT = os.environ.get('INFER_ENSEMBLE_OUT','model_save/teacher_logits/ensemble_oof_probs.pt')
OOF_TABLE = os.environ.get('INFER_OOF_TABLE','model_save/teacher_logits/oof_probs.parquet')
FORCE_REGEN = os.environ.get('INFER_FORCE_REGEN','0')=='1'
CALC_VAL = os.environ.get('INFER_INCLUDE_VAL','1')=='1'
SAVE_LASTTOK = os.environ.get('INFER_SAVE_LASTTOK','0')=='1'
LASTTOK_TOPK = int(os.environ.get('INFER_LASTTOK_TOPK','0'))  # 0 means full when saving
print(f"[Step4] SAVE_ROOT={SAVE_ROOT} FOLDS={FOLDS} MODELS={INFER_MODELS} SAVE_LASTTOK={SAVE_LASTTOK} TOPK={LASTTOK_TOPK}")

def pick_model_dir(prefix: str, fold: int):
    cand = os.path.join(SAVE_ROOT, f'{prefix}_fold_{fold}')
    if os.path.isdir(cand):
        return cand
    if not DISABLE_SYMLINKS:
        alt = f'model_save/{prefix}_fold_{fold}'
        if os.path.isdir(alt):
            return alt
    return None

def pick_lora_dir(prefix: str, fold: int):
    # canonical SAVE_ROOT location
    lora = os.path.join(SAVE_ROOT, f'{prefix}_fold_{fold}_lora')
    if os.path.isdir(lora):
        return lora
    if not DISABLE_SYMLINKS:
        alt = f'model_save/{prefix}_fold_{fold}_lora'
        if os.path.isdir(alt):
            return alt
    return None

def has_weight_files(d: str) -> bool:
    if not d: return False
    for f in ('model.safetensors','pytorch_model.bin'):
        if os.path.isfile(os.path.join(d,f)):
            return True
    if glob.glob(os.path.join(d,'model-*.safetensors')):
        return True
    return False

def is_sequence_classification_dir(d: str) -> bool:
    """Heuristic: detect sequence-classification checkpoints which are unsuitable for CausalLM scoring."""
    try:
        if not d or not os.path.isdir(d):
            return False
        cfg_path = os.path.join(d, 'config.json')
        if not os.path.isfile(cfg_path):
            return False
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        arch = cfg.get('architectures') or []
        if any('SequenceClassification' in str(a) for a in arch):
            return True
        if 'num_labels' in cfg:
            return True
    except Exception:
        pass
    return False

STRICT_NONEMPTY = os.environ.get('INFER_STRICT_NONEMPTY','0')=='1'
def load_rows(path: str):
    try:
        size = os.path.getsize(path)
        if size == 0:
            msg = f"[Step4][Warn] Empty CSV {path} (0 bytes)."
            if STRICT_NONEMPTY:
                raise RuntimeError(msg)
            print(msg + " Skipping.")
            return pd.DataFrame()
        # Force all columns to string to avoid mixed-type parser issues
        df = pd.read_csv(path, dtype=str, low_memory=False)
        print(f"[Step4] Loaded {path} rows={len(df)} cols={len(df.columns)}")
        return df
    except pd.errors.EmptyDataError:
        msg = f"[Step4][Warn] EmptyDataError reading {path}"
        if STRICT_NONEMPTY:
            raise
        print(msg + " -> returning empty DataFrame")
        return pd.DataFrame()
    except Exception as e:
        if STRICT_NONEMPTY:
            raise
        print(f"[Step4][Warn] Failed to read {path}: {e} -> empty DataFrame")
        return pd.DataFrame()

def build_prompt(row):
    return str(row.get('prompt') or row.get('question') or '').strip()

def extract_text(row):
    p = build_prompt(row)
    r = row.get('response') or row.get('chosen') or row.get('answer') or ''
    return (p+'\n' if p else '') + str(r).strip()

def compute_pair_loglik(model, tokenizer, prompts, ra_list, rb_list, max_len=1024):
    """Original per-example (kept for fallback)"""
    logp_a, logp_b = [], []
    model.eval()
    with torch.no_grad():
        for p, ra, rb in zip(prompts, ra_list, rb_list):
            for resp, store in [(ra, logp_a), (rb, logp_b)]:
                full = (p+'\n' if p else '') + resp
                enc = tokenizer(full, return_tensors='pt', truncation=True, max_length=max_len)
                input_ids = enc['input_ids'].to(DEVICE)
                attn = enc['attention_mask'].to(DEVICE)
                p_enc = tokenizer(p + ('\n' if p else ''), return_tensors='pt', truncation=True, max_length=max_len)
                p_len = p_enc['input_ids'].shape[-1]
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                logits = out.logits[:, :-1, :]
                target = input_ids[:, 1:]
                resp_slice = slice(p_len-1, target.shape[1])
                lls = torch.nn.functional.log_softmax(logits[:, resp_slice, :], dim=-1)
                tgt = target[:, resp_slice]
                tok_ll = lls.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                store.append(tok_ll.sum().item())
    return torch.tensor(logp_a), torch.tensor(logp_b)

def compute_pair_loglik_batched(model, tokenizer, prompts, ra_list, rb_list, max_len=1024, batch_size=1, progress_every=100):
    """Batched log-likelihood computation with progress logging.
    We tokenize prompt+response in batches for A and then B separately to avoid memory blow-up.
    """
    n = len(prompts)
    logp_a = torch.empty(n, dtype=torch.float32)
    logp_b = torch.empty(n, dtype=torch.float32)
    model.eval()
    start = time.time()
    with torch.no_grad():
        for which, resp_list, target_store in [('A', ra_list, logp_a), ('B', rb_list, logp_b)]:
            for i in range(0, n, batch_size):
                j = i + batch_size
                batch_prompts = prompts[i:j]
                batch_resps = resp_list[i:j]
                full_texts = []
                prompt_lens = []
                for p, resp in zip(batch_prompts, batch_resps):
                    full = (p+'\n' if p else '') + resp
                    full_texts.append(full)
                    p_ids = tokenizer(p + ('\n' if p else ''), return_tensors='pt', truncation=True, max_length=max_len).input_ids
                    prompt_lens.append(p_ids.shape[-1])
                enc = tokenizer(full_texts, return_tensors='pt', truncation=True, max_length=max_len, padding=True)
                input_ids = enc['input_ids'].to(DEVICE)
                attn = enc['attention_mask'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
                logits = outputs.logits[:, :-1, :]
                targets = input_ids[:, 1:]
                lls = torch.nn.functional.log_softmax(logits, dim=-1)
                for row_idx in range(input_ids.size(0)):
                    p_len = prompt_lens[row_idx]
                    # slice ensuring p_len-1 >=0
                    resp_slice = slice(max(p_len-1,0), targets.shape[1])
                    tgt_ids = targets[row_idx:row_idx+1, resp_slice]
                    tok_ll = lls[row_idx:row_idx+1, resp_slice, :].gather(-1, tgt_ids.unsqueeze(-1)).squeeze(-1)
                    target_store[i+row_idx] = tok_ll.sum().float().cpu()
                if (i // batch_size) % max(1, progress_every) == 0:
                    done = min(j, n)
                    elapsed = time.time()-start
                    rate = done/(elapsed+1e-6)
                    print(f"[Step4][Progress] Resp{which} fold batch {done}/{n} ({rate:.2f} ex/s) elapsed={elapsed/60:.1f}m")
    return logp_a, logp_b

def batched_lasttok_repr(model, tokenizer, texts, max_len=512, batch_size=2):
    if not SAVE_LASTTOK:
        return None
    chunks = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, return_tensors='pt', truncation=True, max_length=max_len, padding=True)
            enc = {k: v.to(DEVICE) for k,v in enc.items()}
            out = model(**enc, use_cache=False)
            logits = out.logits[:, -1, :].to(torch.float16)
            if LASTTOK_TOPK > 0:
                k = min(LASTTOK_TOPK, logits.size(-1))
                vals, idx = torch.topk(logits, k=k, dim=-1)
                chunks.append({'topk_values': vals.cpu(), 'topk_indices': idx.cpu()})
            else:
                chunks.append(logits.cpu())
    if LASTTOK_TOPK > 0:
        return {
            'k': LASTTOK_TOPK,
            'topk_values': torch.cat([c['topk_values'] for c in chunks], 0),
            'topk_indices': torch.cat([c['topk_indices'] for c in chunks], 0),
            'dtype': 'float16'
        }
    return torch.cat(chunks, 0)

os.makedirs('model_save/teacher_logits', exist_ok=True)

def save_obj(path, obj):
    if (not FORCE_REGEN) and os.path.isfile(path):
        print(f"[Step4][Skip] {path} exists")
        return
    torch.save(obj, path)
    if isinstance(obj, dict):
        shape = obj['topk_values'].shape
        print(f"[Step4] Saved top-k lasttok -> {path} shape={shape}")
    else:
        print(f"[Step4] Saved tensor -> {path} shape={tuple(obj.shape)}")

oof_rows = []

for fold in FOLDS:
    train_csv = f'data/fold_data/fold_{fold}_train.csv'
    val_csv = f'data/fold_data/fold_{fold}_val.csv'
    if not os.path.isfile(train_csv):
        print(f"[Step4][Skip] Missing {train_csv}")
        continue
    train_df = load_rows(train_csv)
    if train_df.empty:
        print(f"[Step4][Skip] train_df empty for fold {fold}; continuing to next fold.")
        continue
    # Subset controls (global + per-model). We first record original indices for optional mapping.
    train_df['_orig_idx'] = list(range(len(train_df)))
    subset_size_global = int(os.environ.get('INFER_TRAIN_SUBSET_SIZE','0') or '0')
    subset_seed = int(os.environ.get('INFER_SUBSET_SEED','123') or '123')
    strict_subset = os.environ.get('INFER_STRICT_SUBSET','0')=='1'
    # Per-model overrides allow dramatically smaller subsets: INFER_LLAMA_SUBSET, INFER_QWEN_SUBSET
    # Format: either single integer (e.g., 18000) or range "15000-20000" -> sample uniform random size in that range per fold.
    def parse_subset(spec: str):
        if not spec:
            return None
        spec = spec.strip()
        if '-' in spec:
            a,b = spec.split('-',1)
            try:
                lo, hi = int(a), int(b)
                if lo>hi: lo,hi = hi,lo
                import random
                sz = random.Random(subset_seed + fold).randint(lo, hi)
                return sz
            except ValueError:
                return None
        else:
            try:
                return int(spec)
            except ValueError:
                return None
    llama_spec = os.environ.get('INFER_LLAMA_SUBSET','')
    qwen_spec  = os.environ.get('INFER_QWEN_SUBSET','')
    # We will defer actual sampling until inside per-model loop if per-model subset specified; if only global subset, apply once here.
    if subset_size_global > 0 and not (llama_spec or qwen_spec):
        current_n = len(train_df)
        if subset_size_global > current_n:
            msg = f"[Step4][Subset] Requested global subset_size={subset_size_global} exceeds train rows={current_n} (fold {fold})."
            if strict_subset:
                raise RuntimeError(msg + " (INFER_STRICT_SUBSET=1)")
            else:
                print(msg + " Using full train set instead.")
        elif subset_size_global < current_n:
            train_df = train_df.sample(n=subset_size_global, random_state=subset_seed).reset_index(drop=True)
            print(f"[Step4][Subset] Fold {fold}: global sampled {subset_size_global}/{current_n} rows (seed={subset_seed}).")
        else:
            print(f"[Step4][Subset] Fold {fold}: global subset equals train rows ({current_n}).")
    val_df = load_rows(val_csv) if os.path.isfile(val_csv) else None

    model_dirs = {}
    if 'llama' in INFER_MODELS:
        model_dirs['llama'] = pick_model_dir('llama', fold)
    if 'qwen' in INFER_MODELS:
        model_dirs['qwen'] = pick_model_dir('qwen', fold)

    prefer_lora = os.environ.get('INFER_PREFER_LORA','0')=='1'
    base_llama = os.environ.get('BASE_LLAMA_DIR') or os.environ.get('LLAMA_BASE_DIR') or os.path.join(SCRATCH_BASE,'post_pretrain_llama3-8b_merged')
    base_qwen  = os.environ.get('BASE_QWEN_DIR')  or os.environ.get('QWEN_BASE_DIR')  or os.path.join(SCRATCH_BASE,'post_pretrain_qwen2-72b_merged')
    base_map = {'llama': base_llama, 'qwen': base_qwen}

    for name, mdir in model_dirs.items():
        lora_dir = pick_lora_dir(name, fold)
        using_lora = False
        load_dir = mdir
        # Decide whether to use merged or lora
        if prefer_lora or ( (mdir is None or not has_weight_files(mdir)) and lora_dir is not None ):
            using_lora = True
            load_dir = lora_dir
        if using_lora:
            if lora_dir is None:
                print(f"[Step4][Warn] No LoRA dir found for {name} fold {fold}")
                continue
            base_dir = base_map.get(name)
            if not base_dir or not os.path.isdir(base_dir):
                print(f"[Step4][Error] Base model dir {base_dir} missing for {name} needed to apply LoRA {lora_dir}")
                continue
        else:
            if load_dir is None or not has_weight_files(load_dir):
                print(f"[Step4][Warn] Missing merged weights for {name} fold {fold} (dir={load_dir}); consider INFER_PREFER_LORA=1 if LoRA exists.")
                continue

        # If merged dir looks like a sequence-classification checkpoint, fall back to base for causal LM scoring
        if (not using_lora) and is_sequence_classification_dir(load_dir):
            print(f"[Step4][Warn] Detected SequenceClassification checkpoint at {load_dir}. Falling back to base model {base_map.get(name)} for causal LM scoring.")
            base_dir = base_map.get(name)
            if base_dir and os.path.isdir(base_dir):
                load_dir = base_dir
            else:
                print(f"[Step4][Error] Base model dir missing for {name}; cannot score with CausalLM. Skipping.")
                continue

        source_desc = f"LoRA {lora_dir} + base {base_map.get(name)}" if using_lora else load_dir
        print(f"[Step4] Loading {name} fold {fold} from {source_desc}")
        # Robust tokenizer loading with fallback if tokenizer assets were pruned from fold dir.
        def load_tokenizer_with_fallback(model_name:str, model_dir:str):
            # Environment overrides
            env_specific = os.environ.get(f'{model_name.upper()}_TOKENIZER_PATH')
            # Also support *_TOK shorthand commonly used in our envs (e.g., LLAMA_TOK, QWEN_TOK)
            if not env_specific:
                env_specific = os.environ.get(f'{model_name.upper()}_TOK')
            generic_fallback = os.environ.get('FALLBACK_TOKENIZER_PATH')
            base_dir_env = os.environ.get(f'{model_name.upper()}_BASE_DIR')
            tried = []
            candidates = [model_dir]
            # Auto sibling *_lora detection if tokenizer assets absent
            def has_tokenizer_files(path):
                if not path or not os.path.isdir(path):
                    return False
                for fn in ('tokenizer.model','tokenizer.json','tokenizer_config.json'):
                    if os.path.isfile(os.path.join(path, fn)):
                        return True
                return False
            if not has_tokenizer_files(model_dir):
                sib = model_dir + '_lora'
                if has_tokenizer_files(sib) and sib not in candidates:
                    candidates.append(sib)
            # Also consider direct lora_dir if using lora inference
            if using_lora and lora_dir not in candidates:
                candidates.append(lora_dir)
            if env_specific and env_specific not in candidates:
                candidates.append(env_specific)
            if base_dir_env and base_dir_env not in candidates:
                candidates.append(base_dir_env)
            if generic_fallback and generic_fallback not in candidates:
                candidates.append(generic_fallback)
            last_err=None
            for c in candidates:
                try:
                    tried.append(c)
                    tok_local = AutoTokenizer.from_pretrained(c, use_fast=True, trust_remote_code=True)
                    return tok_local, c
                except Exception as e:
                    last_err=e
                    continue
            # Retry slow tokenizer if fast failed everywhere
            for c in candidates:
                try:
                    tried.append(c+'(slow)')
                    tok_local = AutoTokenizer.from_pretrained(c, use_fast=False, trust_remote_code=True)
                    return tok_local, c
                except Exception as e:
                    last_err=e
            raise RuntimeError(f"Tokenizer load failed for {model_name}. Tried: {tried}. Last error: {last_err}")

        tok, tok_src = load_tokenizer_with_fallback(name, load_dir if not using_lora else lora_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"[Step4][Info] Tokenizer source for {name} fold {fold}: {tok_src}")
        if using_lora:
            # load base, then apply LoRA adapter
            base_dir_for_load = base_map.get(name)
            try:
                base_model = AutoModelForCausalLM.from_pretrained(base_dir_for_load, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
            except Exception as e:
                print(f"[Step4][Error] Failed loading base model {base_dir_for_load} for {name} fold {fold}: {e}")
                continue
            try:
                model = PeftModel.from_pretrained(base_model, lora_dir, torch_dtype=torch.float16)
            except Exception as e:
                print(f"[Step4][Error] Failed applying LoRA {lora_dir} to base {base_dir_for_load}: {e}")
                continue
        else:
            # load merged causal LM (or base if we fell back)
            model = AutoModelForCausalLM.from_pretrained(load_dir, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        # Safe placement: if accelerate assigned a device_map (offloading), do NOT call model.to(DEVICE)
        if hasattr(model, 'hf_device_map'):
            print('[Step4][Info] Accelerate device_map detected; skipping explicit model.to().')
        else:
            try:
                print(DEVICE)
                model.to(DEVICE)
            except RuntimeError as e:
                print(f'[Step4][Warn] model.to({DEVICE}) failed (continuing with existing placement): {e}')

        # Apply per-model subset if specified (works off pre-global-sampled or global-sampled frame)
        model_subset_spec = llama_spec if name=='llama' else qwen_spec if name=='qwen' else ''
        model_subset_size = parse_subset(model_subset_spec) if model_subset_spec else None
        model_train_df = train_df
        if model_subset_size and model_subset_size > 0:
            cur_n = len(model_train_df)
            if model_subset_size > cur_n:
                msg = f"[Step4][Subset] {name} fold {fold}: requested {model_subset_size} > available {cur_n}."
                if strict_subset:
                    print(msg + " Aborting per strict mode.")
                    continue
                else:
                    print(msg + " Using full set.")
            elif model_subset_size < cur_n:
                model_train_df = model_train_df.sample(n=model_subset_size, random_state=subset_seed + (11 if name=='qwen' else 0)).reset_index(drop=True)
                print(f"[Step4][Subset] {name} fold {fold}: sampled {model_subset_size}/{cur_n} rows (seed offset).")
            else:
                print(f"[Step4][Subset] {name} fold {fold}: subset matches available {cur_n} rows.")
        records_train = model_train_df.to_dict('records')
        texts_train = [extract_text(r) for r in records_train]
        print(f"[Step4][Debug] Preparing last-token repr on {len(texts_train)} texts (SAVE_LASTTOK={SAVE_LASTTOK})", flush=True)
        rep = batched_lasttok_repr(model, tok, texts_train, max_len=min(getattr(tok,'model_max_length',1024),1024))
        if rep is not None:
            save_obj(f'model_save/teacher_logits/{name}_fold_{fold}_train_lasttok_logits.pt', rep)

        if {'response_a','response_b'} <= set(model_train_df.columns):
            prompts = [build_prompt(r) for r in records_train]
            # Determine batch size with sensible defaults if not provided
            env_bs = int(os.environ.get('INFER_LOGPROB_BATCH','0') or '0')
            if env_bs <= 0:
                try:
                    if torch.cuda.is_available():
                        total_mem = torch.cuda.get_device_properties(0).total_memory
                        # Heuristic defaults by VRAM
                        if total_mem >= 22*(1024**3):
                            batch_sz = 8
                        elif total_mem >= 15*(1024**3):
                            batch_sz = 4
                        elif total_mem >= 10*(1024**3):
                            batch_sz = 3
                        elif total_mem >= 7*(1024**3):
                            batch_sz = 2
                        else:
                            batch_sz = 1
                    else:
                        batch_sz = 1
                except Exception:
                    batch_sz = 2
            else:
                batch_sz = env_bs
            progress_every = int(os.environ.get('INFER_PROGRESS_EVERY','50') or '50')
            print(f"[Step4][Debug] Starting TRAIN log-likelihood (N={len(records_train)}), batch_size={batch_sz}, progress_every={progress_every}", flush=True)
            ra_inputs = [str(r.get('response_a','')) for r in records_train]
            rb_inputs = [str(r.get('response_b','')) for r in records_train]
            if batch_sz > 1:
                print(f"[Step4] Using batched loglik batch_size={batch_sz} progress_every={progress_every}", flush=True)
                logp_a, logp_b = compute_pair_loglik_batched(model, tok, prompts, ra_inputs, rb_inputs, max_len=1024, batch_size=batch_sz, progress_every=progress_every)
            else:
                print(f"[Step4] Using non-batched loglik (very slow). Consider setting INFER_LOGPROB_BATCH>1.", flush=True)
                logp_a, logp_b = compute_pair_loglik(model, tok, prompts, ra_inputs, rb_inputs, max_len=1024)
            logp_t = 0.5*(logp_a + logp_b)
            raw = torch.stack([logp_a, logp_b, logp_t], dim=1)
            probs = torch.softmax(raw, dim=1)
            # Persist with per-model subset awareness: include mapping of original indices if present.
            save_obj(f'model_save/teacher_logits/{name}_fold_{fold}_train_logprobs.pt', raw)
            save_obj(f'model_save/teacher_logits/{name}_fold_{fold}_train_probs.pt', probs)
            orig_idx_series = model_train_df.get('_orig_idx')
            for ridx,(pa,pb,pt) in enumerate(probs.tolist()):
                oof_rows.append({'fold':fold,'split':'train','model':name,'row_id':ridx,'pA':pa,'pB':pb,'pTie':pt,'orig_idx': int(orig_idx_series.iloc[ridx]) if orig_idx_series is not None else ridx})
        else:
            print(f"[Step4][Warn] response_a/response_b missing for train fold {fold} {name}")

        if CALC_VAL and val_df is not None and {'response_a','response_b'} <= set(val_df.columns):
            v_records = val_df.to_dict('records')
            v_prompts = [build_prompt(r) for r in v_records]
            env_bs = int(os.environ.get('INFER_LOGPROB_BATCH','0') or '0')
            if env_bs <= 0:
                try:
                    if torch.cuda.is_available():
                        total_mem = torch.cuda.get_device_properties(0).total_memory
                        if total_mem >= 22*(1024**3):
                            batch_sz = 8
                        elif total_mem >= 15*(1024**3):
                            batch_sz = 4
                        elif total_mem >= 10*(1024**3):
                            batch_sz = 3
                        elif total_mem >= 7*(1024**3):
                            batch_sz = 2
                        else:
                            batch_sz = 1
                    else:
                        batch_sz = 1
                except Exception:
                    batch_sz = 2
            else:
                batch_sz = env_bs
            progress_every = int(os.environ.get('INFER_PROGRESS_EVERY','50') or '50')
            print(f"[Step4][Debug] Starting VAL log-likelihood (N={len(v_records)}), batch_size={batch_sz}, progress_every={progress_every}", flush=True)
            va_inputs = [str(r.get('response_a','')) for r in v_records]
            vb_inputs = [str(r.get('response_b','')) for r in v_records]
            if batch_sz > 1:
                print(f"[Step4] (val) Using batched loglik batch_size={batch_sz} progress_every={progress_every}", flush=True)
                v_logp_a, v_logp_b = compute_pair_loglik_batched(model, tok, v_prompts, va_inputs, vb_inputs, max_len=1024, batch_size=batch_sz, progress_every=progress_every)
            else:
                print(f"[Step4] (val) Using non-batched loglik (very slow). Consider setting INFER_LOGPROB_BATCH>1.", flush=True)
                v_logp_a, v_logp_b = compute_pair_loglik(model, tok, v_prompts, va_inputs, vb_inputs, max_len=1024)
            v_logp_t = 0.5*(v_logp_a + v_logp_b)
            v_raw = torch.stack([v_logp_a, v_logp_b, v_logp_t], dim=1)
            v_probs = torch.softmax(v_raw, dim=1)
            save_obj(f'model_save/teacher_logits/{name}_fold_{fold}_val_logprobs.pt', v_raw)
            save_obj(f'model_save/teacher_logits/{name}_fold_{fold}_val_probs.pt', v_probs)
            for ridx,(pa,pb,pt) in enumerate(v_probs.tolist()):
                oof_rows.append({'fold':fold,'split':'val','model':name,'row_id':ridx,'pA':pa,'pB':pb,'pTie':pt})

        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print('[Step4] Per-model/fold inference done.')

if oof_rows:
    oof_df = pd.DataFrame(oof_rows)
    basis = oof_df[oof_df.split=='val'] if (oof_df['split']=='val').any() else oof_df
    grp = basis.groupby(['fold','row_id'])[['pA','pB','pTie']].mean().reset_index()
    torch.save(torch.tensor(grp[['pA','pB','pTie']].values, dtype=torch.float32), ENSEMBLE_OUT)
    print(f"[Step4][Ensemble] Saved mean probs -> {ENSEMBLE_OUT} shape={(len(grp),3)}")
    if OOF_TABLE.endswith('.parquet'):
        oof_df.to_parquet(OOF_TABLE, index=False)
    else:
        oof_df.to_csv(OOF_TABLE, index=False)
    print(f"[Step4][OOF] Table -> {OOF_TABLE} rows={len(oof_df)}")
else:
    print('[Step4][OOF] No rows; nothing saved.')

print('[Step4] Finished.')
PY
python teacher_logits_validator.py || echo "[Step4][Warn] Validator failed; adjust validator script to new file naming."

echo "[Step4] Done"
