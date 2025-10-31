"""
Inference for GPTQ-quantized CausalLM exports with a classification head adapter.

This script loads a quantized CausalLM (e.g., LLaMA/Qwen) using AutoGPTQ and applies a
lightweight classification head on top of the model's hidden states to produce 3-way
probabilities (winner_model_a, winner_model_b, winner_tie) for the Kaggle submission.

Two ways to provide the classification head:
1) Adapter weights path (preferred): a small torch .pt containing a linear layer weights/bias.
2) Derive from an existing sequence-classifier directory via --classifier-from-dir (extract the head).

Notes:
- We use the CLS/pooled representation if available, else mean-pool last_hidden_state.
- Supports TTA by averaging probabilities across multiple max_length values.
- Applies temperature calibration if calibration.json exists in the classifier dir or provided path.
"""

import argparse
import json
import os
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    import importlib
    _gptq = importlib.import_module('auto_gptq')
    AutoGPTQForCausalLM = getattr(_gptq, 'AutoGPTQForCausalLM', None)
except Exception:
    AutoGPTQForCausalLM = None


def build_input_text(row: pd.Series) -> str:
    return f"[PROMPT]{str(row['prompt']).strip()}[RESPONSE_A]{str(row['response_a']).strip()}[RESPONSE_B]{str(row['response_b']).strip()}"


def load_temperature(calibration_json_path: Optional[str]) -> float:
    if not calibration_json_path:
        return 1.0
    if not os.path.exists(calibration_json_path):
        return 1.0
    try:
        with open(calibration_json_path, 'r') as f:
            data = json.load(f)
        return float(data.get('temperature', 1.0))
    except Exception:
        return 1.0


def _normalize_head_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Prefer original_module.* if present, else modules_to_save.default.*, else identity
    keys = state.keys()
    out: Dict[str, torch.Tensor] = {}
    # Weight
    if 'original_module.weight' in keys:
        out['weight'] = state['original_module.weight']
    elif 'modules_to_save.default.weight' in keys:
        out['weight'] = state['modules_to_save.default.weight']
    elif 'weight' in keys:
        out['weight'] = state['weight']
    # Bias
    if 'original_module.bias' in keys:
        out['bias'] = state['original_module.bias']
    elif 'modules_to_save.default.bias' in keys:
        out['bias'] = state['modules_to_save.default.bias']
    elif 'bias' in keys:
        out['bias'] = state['bias']
    return out if out else state


def extract_classifier_from_dir(classifier_dir: str, num_labels: int = 3) -> nn.Module:
    """
    Attempt to extract a classifier head from a fine-tuned sequence classification directory by
    reading a saved head (if exported) or initializing a new linear layer with the same hidden size.
    We search for hidden_size in config.json and expect classifier weights in classifier_head.pt.
    """
    cfg_path = os.path.join(classifier_dir, 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found in {classifier_dir}")
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    hidden_size = cfg.get('hidden_size') or cfg.get('hidden_sizes', [None])[0]
    if hidden_size is None:
        # fallback to common keys
        hidden_size = cfg.get('d_model') or cfg.get('hidden_size', 1024)
    head = nn.Linear(int(hidden_size), num_labels)

    # If a dumped head exists, load it
    head_path = os.path.join(classifier_dir, 'classifier_head.pt')
    if os.path.exists(head_path):
        state = torch.load(head_path, map_location='cpu')
        state = _normalize_head_state(state)
        try:
            head.load_state_dict(state)
        except Exception:
            pass
    return head


def load_classifier_head(head_path: Optional[str], classifier_from_dir: Optional[str], hidden_size: int, num_labels: int = 3) -> nn.Module:
    if head_path and os.path.exists(head_path):
        state = torch.load(head_path, map_location='cpu')
        state = _normalize_head_state(state)
        has_bias = 'bias' in state
        head = nn.Linear(hidden_size, num_labels, bias=has_bias)
        # Load with strict=False to tolerate missing bias or extra keys
        head.load_state_dict(state, strict=False)
        return head
    if classifier_from_dir:
        head = extract_classifier_from_dir(classifier_from_dir, num_labels=num_labels)
        return head
    # default randomly initialized head
    return nn.Linear(hidden_size, num_labels)


def infer(
    model_dir: str,
    test_csv: str,
    submission_path: str,
    head_path: Optional[str] = None,
    classifier_from_dir: Optional[str] = None,
    tta_lengths: Optional[List[int]] = None,
    batch_size: int = 8,
    max_length: int = 512,
    device: Optional[str] = None,
    temperature_json: Optional[str] = None,
    fallback_base_model_dir: Optional[str] = None,
):
    # We'll try AutoGPTQ, but support a graceful fallback to BitsAndBytes int8 if needed

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    df = pd.read_csv(test_csv)
    for c in ['id', 'prompt', 'response_a', 'response_b']:
        if c not in df.columns:
            raise ValueError(f"Missing column in test.csv: {c}")
    texts = (df.apply(build_input_text, axis=1)).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    # Detect GPTQ vs BNB fallback by metadata files
    use_bnb_int8 = False
    meta_json = os.path.join(model_dir, 'quantization_config.json')
    if os.path.exists(meta_json):
        try:
            with open(meta_json, 'r') as f:
                meta = json.load(f)
            if str(meta.get('method','')).upper() == 'BNB_INT8':
                use_bnb_int8 = True
        except Exception:
            pass

    if use_bnb_int8:
        # Read target base model directory and load with BitsAndBytes int8
        target_file = os.path.join(model_dir, 'target_model_dir.txt')
        if not os.path.exists(target_file):
            raise RuntimeError("BNB_INT8 fallback selected but target_model_dir.txt not found in quantized dir")
        with open(target_file, 'r') as f:
            target_dir = f.read().strip()
        # Build candidate base models list for robust fallback
        candidates: List[str] = [target_dir]
        # 1) Explicit CLI arg
        if 'fallback_base_model_dir' in locals() and fallback_base_model_dir:
            candidates.append(fallback_base_model_dir)
        # 2) Environment override
        env_base = os.environ.get('BASE_MODEL') or os.environ.get('FALLBACK_BASE_MODEL_DIR')
        if env_base:
            candidates.append(env_base)
        # 3) Metadata hint (tokenizer source)
        tok_src = None
        try:
            with open(os.path.join(model_dir, 'quantization_config.json'), 'r') as f:
                meta = json.load(f)
                tok_src = meta.get('tokenizer_source') or meta.get('tokenizer_dir') or meta.get('tokenizer')
        except Exception:
            tok_src = None
        if tok_src:
            candidates.append(str(tok_src))
        # Dedup while preserving order
        seen = set()
        candidates = [c for c in candidates if c and (c not in seen and not seen.add(c))]
        # Optional: log likely-corrupt safetensors files (size==0)
        try:
            st_files = [p for p in os.listdir(target_dir) if p.endswith('.safetensors')]
            for name in st_files:
                fp = os.path.join(target_dir, name)
                try:
                    if os.path.getsize(fp) == 0:
                        print(f"[Warn] Zero-byte safetensors file detected: {fp}; loading may fail. Consider re-merging/re-quantizing.")
                except Exception:
                    pass
        except Exception:
            pass
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        last_err = None
        loaded = False
        for cand in candidates:
            # Quick check: if cand is a local dir without weight files, skip to next
            if os.path.isdir(cand):
                names = os.listdir(cand)
                has_weights = any(
                    n in names for n in (
                        'pytorch_model.bin','model.safetensors','tf_model.h5','model.ckpt.index','flax_model.msgpack','pytorch_model.bin.index.json','model.safetensors.index.json'
                    )
                ) or any(n.startswith('pytorch_model-') and n.endswith('.bin') for n in names) or any(n.startswith('model-') and n.endswith('.safetensors') for n in names)
                if not has_weights:
                    print(f"[Info] Skipping base candidate without weight files: {cand}")
                    continue
            print(f"[Info] Trying BNB int8 load from: {cand}")
            try:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        cand,
                        quantization_config=bnb_cfg,
                        device_map='auto' if dev.type == 'cuda' else None,
                        trust_remote_code=True,
                        use_safetensors=True,
                    )
                except Exception as e1:
                    msg = str(e1)
                    if 'safetensors' in msg.lower() or 'invalid json in header' in msg.lower():
                        print('[Info] Safetensors load failed; retrying with use_safetensors=False to use .bin shards if available...')
                        model = AutoModelForCausalLM.from_pretrained(
                            cand,
                            quantization_config=bnb_cfg,
                            device_map='auto' if dev.type == 'cuda' else None,
                            trust_remote_code=True,
                            use_safetensors=False,
                        )
                    else:
                        raise
                loaded = True
                break
            except Exception as e2:
                last_err = e2
                print(f"[Warn] BNB int8 load failed for {cand}: {e2}")
                continue
        if not loaded:
            raise RuntimeError(
                'Failed to load base model for BNB int8. Provide a valid float base model via --fallback-base-model-dir or BASE_MODEL env var, '
                f'or fix the directory listed in target_model_dir.txt. Last error: {last_err}'
            )
    else:
        if AutoGPTQForCausalLM is None:
            raise RuntimeError("auto-gptq is not installed. Please install auto-gptq or re-run quantization to produce BNB_INT8 fallback.")
        try:
            model = AutoGPTQForCausalLM.from_quantized(
                model_dir,
                device_map='auto' if dev.type == 'cuda' else None,
            )
        except Exception as e:
            msg = str(e)
            # Attempt graceful fallback to BNB int8 if GPTQ shards look corrupted
            if 'safetensors' in msg.lower() or 'invalid json in header' in msg.lower():
                print('[Warn] GPTQ safetensors load failed; attempting BNB int8 fallback using a float base model...')
                # Resolve candidate base model dirs
                candidates: List[str] = []
                if fallback_base_model_dir:
                    candidates.append(fallback_base_model_dir)
                env_base = os.environ.get('BASE_MODEL') or os.environ.get('FALLBACK_BASE_MODEL_DIR')
                if env_base:
                    candidates.append(env_base)
                for name in ('target_model_dir.txt', 'base_model_dir.txt'):
                    fp = os.path.join(model_dir, name)
                    if os.path.exists(fp):
                        try:
                            with open(fp, 'r') as f:
                                candidates.append(f.read().strip())
                        except Exception:
                            pass
                tried = []
                last_err = None
                for cand in candidates:
                    if not cand or cand in tried:
                        continue
                    tried.append(cand)
                    print(f"[Info] Trying BNB int8 load from base model: {cand}")
                    try:
                        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                        try:
                            model = AutoModelForCausalLM.from_pretrained(
                                cand,
                                quantization_config=bnb_cfg,
                                device_map='auto' if dev.type == 'cuda' else None,
                                trust_remote_code=True,
                                use_safetensors=True,
                            )
                        except Exception:
                            model = AutoModelForCausalLM.from_pretrained(
                                cand,
                                quantization_config=bnb_cfg,
                                device_map='auto' if dev.type == 'cuda' else None,
                                trust_remote_code=True,
                                use_safetensors=False,
                            )
                        break
                    except Exception as e2:
                        last_err = e2
                        continue
                else:
                    raise RuntimeError(
                        'Failed to load GPTQ model due to safetensors error and could not fall back to BNB int8. '
                        'Provide --fallback-base-model-dir or set BASE_MODEL to the float base model directory. '
                        f'Original error: {e}\nLast fallback error: {last_err}'
                    )
            else:
                raise
    model.eval()

    # Hidden size discovery via config if available
    hidden_size = getattr(model.config, 'hidden_size', None) or getattr(model.config, 'hidden_sizes', [None])[0]
    if hidden_size is None:
        hidden_size = getattr(model.config, 'd_model', 1024)

    head = load_classifier_head(head_path, classifier_from_dir, hidden_size, num_labels=3).to(dev)
    head.eval()

    # Temperature from provided path or fallback to classifier dir
    temp_path = temperature_json
    if not temp_path and classifier_from_dir:
        temp_path = os.path.join(classifier_from_dir, 'calibration.json')
    temperature = load_temperature(temp_path)

    lengths = tta_lengths if tta_lengths else [max_length]
    probs_accum = None

    with torch.no_grad():
        for L in lengths:
            probs_list = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                enc = tokenizer(batch_texts, truncation=True, padding='max_length', max_length=L, return_tensors='pt')
                enc = {k: v.to(dev) for k, v in enc.items()}

                # forward to get hidden states
                try:
                    outputs = model(**enc, output_hidden_states=True)
                except TypeError:
                    # Some wrappers may require passing output_hidden_states via config; fallback to common submodules
                    if hasattr(model, 'transformer'):
                        outputs = model.transformer(**enc, output_hidden_states=True)
                    else:
                        outputs = model.model(**enc, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]  # [B, T, H]
                # Try CLS token (first token) if present otherwise mean pool
                if tokenizer.cls_token_id is not None:
                    pooled = last_hidden[:, 0]
                else:
                    pooled = (last_hidden * enc['attention_mask'].unsqueeze(-1)).sum(dim=1) / enc['attention_mask'].sum(dim=1, keepdim=True)

                # Align dtypes to avoid matmul Half vs Float errors
                if pooled.dtype != head.weight.dtype:
                    pooled = pooled.to(head.weight.dtype)
                logits = head(pooled)
                if temperature != 1.0:
                    logits = logits / temperature
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                probs_list.append(probs)

            probs = np.vstack(probs_list)
            probs_accum = probs if probs_accum is None else (probs_accum + probs)

    probs = probs_accum / len(lengths)
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    """
    # ========== POST-PROCESSING FIX: Reduce Excessive Tie Predictions ==========
    print(f"\n[INFO] Applying tie prediction correction...")
    print(f"  Before: mean_tie={probs[:, 2].mean():.4f}, mean_a={probs[:, 0].mean():.4f}, mean_b={probs[:, 1].mean():.4f}")

    # Reduce tie predictions by 40% (adjust if needed)
    tie_reduction_factor = 0.6  # Keep 60% of tie probability
    original_tie = probs[:, 2].copy()
    probs[:, 2] = probs[:, 2] * tie_reduction_factor

    # Redistribute the reduced tie probability to model_a and model_b
    reduction_amount = original_tie - probs[:, 2]
    probs[:, 0] += reduction_amount * 0.4  # Give 40% to model_a
    probs[:, 1] += reduction_amount * 0.6  # Give 60% to model_b (historically more common)

    # Renormalize to ensure probabilities sum to 1.0
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums

    print(f"  After:  mean_tie={probs[:, 2].mean():.4f}, mean_a={probs[:, 0].mean():.4f}, mean_b={probs[:, 1].mean():.4f}")
    print(f"[INFO] Tie correction applied successfully.")
    # ===========================================================================
    """
    sub = pd.DataFrame({
        'id': df['id'],
        'winner_model_a': probs[:, 0],
        'winner_model_b': probs[:, 1],
        'winner_tie': probs[:, 2],
    })
    sub.to_csv(submission_path, index=False)
    return submission_path


def main():
    ap = argparse.ArgumentParser(description='Inference with GPTQ-quantized CausalLM using a classification head adapter')
    ap.add_argument('--model-dir', required=True, help='Path to GPTQ quantized model directory')
    ap.add_argument('--test-csv', default='./data/test.csv')
    ap.add_argument('--out', default='./sub/student_submission.csv')
    ap.add_argument('--head-path', default='./model_save/distilled_gemma2-9b_fold_0/classifier_head.pt', help='Path to classifier_head.pt (state_dict for nn.Linear)')
    ap.add_argument('--classifier-from-dir', default='./model_save/distilled_gemma2-9b_fold_0', help='Directory of a seq-classification model to extract head and calibration.json')
    ap.add_argument('--tta-lengths', default='', help='Comma-separated list, e.g. 512,1024')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--max-length', type=int, default=512)
    ap.add_argument('--device', default=None)
    ap.add_argument('--temperature-json', default=None, help='Optional explicit path to calibration.json')
    ap.add_argument('--fallback-base-model-dir', default=None, help='Optional float base model directory to use if GPTQ loading fails')
    args = ap.parse_args()

    tta = [int(x) for x in args.tta_lengths.split(',') if x.strip()] if args.tta_lengths else None

    infer(
        model_dir=args.model_dir,
        test_csv=args.test_csv,
        submission_path=args.out,
        head_path=args.head_path,
        classifier_from_dir=args.classifier_from_dir,
        tta_lengths=tta,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        temperature_json=args.temperature_json,
        fallback_base_model_dir=args.fallback_base_model_dir,
    )


if __name__ == '__main__':
    main()
