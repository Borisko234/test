"""
Simple inference script - loads model directly without GPTQ complications
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def _file_exists(p: str) -> bool:
    try:
        return os.path.exists(p)
    except Exception:
        return False


def _has_weight_files(path: str) -> bool:
    try:
        files = os.listdir(path)
    except Exception:
        return False
    patterns = (
        'pytorch_model.bin',
        'model.safetensors',
        'pytorch_model.bin.index.json',
        'model.safetensors.index.json',
        'tf_model.h5',
        'model.ckpt.index',
        'flax_model.msgpack',
    )
    return any(any(f.startswith(p.split('.')[0]) and p.split('.')[-1] in f for f in files) or (p in files) for p in patterns)


def _resolve_model_load_dir(model_dir: str) -> str:
    """If given a lightweight quantized folder, redirect to the real base model directory."""
    # Check for pointer files created by step7
    for fname in ('target_model_dir.txt', 'base_model_dir.txt'):
        ptr = os.path.join(model_dir, fname)
        if _file_exists(ptr):
            try:
                with open(ptr, 'r') as f:
                    base = f.read().strip()
                if base:
                    return base
            except Exception:
                pass
    # If no weights in folder, honor BASE_MODEL env var as last resort
    if not _has_weight_files(model_dir):
        env_base = os.environ.get('BASE_MODEL', '').strip()
        if env_base:
            return env_base
    return model_dir


def build_input_text(row: pd.Series) -> str:
    return f"[PROMPT]{str(row['prompt']).strip()}[RESPONSE_A]{str(row['response_a']).strip()}[RESPONSE_B]{str(row['response_b']).strip()}"


def load_classifier_head(head_path: str, hidden_size: int, num_labels: int = 3) -> nn.Module:
    """Load the trained 3-class classifier head"""
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"Classifier head not found: {head_path}")
    
    state = torch.load(head_path, map_location='cpu')
    
    # Normalize keys (handle PEFT wrapper keys)
    normalized = {}
    if 'modules_to_save.default.weight' in state:
        normalized['weight'] = state['modules_to_save.default.weight']
    elif 'weight' in state:
        normalized['weight'] = state['weight']
    
    if 'modules_to_save.default.bias' in state:
        normalized['bias'] = state['modules_to_save.default.bias']
    elif 'bias' in state:
        normalized['bias'] = state['bias']
    
    has_bias = 'bias' in normalized
    head = nn.Linear(hidden_size, num_labels, bias=has_bias)
    head.load_state_dict(normalized, strict=False)
    return head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True, help='Path to trained model directory')
    parser.add_argument('--head-path', default='', help='Path to classifier_head.pt; if empty, will try <model-dir>/classifier_head.pt')
    parser.add_argument('--test-csv', default='./data/test.csv')
    parser.add_argument('--out', default='./sub/submission.csv')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=512)
    args = parser.parse_args()

    print(f"[Info] Loading model from: {args.model_dir}")
    print(f"[Info] Loading classifier head from: {args.head_path}")

    # Load test data
    df = pd.read_csv(args.test_csv)
    texts = df.apply(build_input_text, axis=1).tolist()
    print(f"[Info] Loaded {len(texts)} test samples")

    # Determine tokenizer dir and actual model load dir
    tokenizer_dir = args.model_dir
    model_load_dir = _resolve_model_load_dir(args.model_dir)

    # Load tokenizer (prefer provided dir, then fallback to model load dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_load_dir, use_fast=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try 8-bit quantization first, fall back to float16
    try:
        print("[Info] Attempting to load with 8-bit quantization...")
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_load_dir,
            quantization_config=bnb_cfg,
            device_map='auto' if device.type == 'cuda' else None,
            trust_remote_code=True,
        )
        print("[Info] ✅ Loaded with 8-bit quantization")
    except Exception as e:
        print(f"[Warn] 8-bit loading failed: {e}")
        print("[Info] Loading in float16/float32...")
        model = AutoModelForCausalLM.from_pretrained(
            model_load_dir,
            device_map='auto' if device.type == 'cuda' else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        )
        print("[Info] ✅ Loaded in float16/float32")
    
    model.eval()

    # Load classifier head
    hidden_size = model.config.hidden_size
    head_path = args.head_path
    if not head_path:
        candidate = os.path.join(args.model_dir, 'classifier_head.pt')
        if _file_exists(candidate):
            head_path = candidate
    if not head_path or not _file_exists(head_path):
        raise FileNotFoundError(
            f"Classifier head not found. Provide --head-path or place classifier_head.pt in {args.model_dir}"
        )
    head = load_classifier_head(head_path, hidden_size, num_labels=3)
    head = head.to(device)
    head.eval()
    print(f"[Info] ✅ Loaded 3-class classifier head (hidden_size={hidden_size})")

    # Run inference
    print("[Info] Running inference...")
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i : i + args.batch_size]
            enc = tokenizer(
                batch_texts, 
                truncation=True, 
                padding='max_length', 
                max_length=args.max_length, 
                return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            # Get hidden states
            outputs = model(**enc, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # [B, T, H]
            
            # Mean pool (since Gemma doesn't have CLS token)
            pooled = (last_hidden * enc['attention_mask'].unsqueeze(-1)).sum(dim=1) / enc['attention_mask'].sum(dim=1, keepdim=True)

            # Align dtypes
            if pooled.dtype != head.weight.dtype:
                pooled = pooled.to(head.weight.dtype)
            
            # Get predictions
            logits = head(pooled)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            
            if (i // args.batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch_texts)}/{len(texts)} samples...")

    # Combine and normalize
    all_probs = np.vstack(all_probs)
    all_probs = np.clip(all_probs, 1e-9, None)
    all_probs = all_probs / all_probs.sum(axis=1, keepdims=True)

    # Create submission
    submission = pd.DataFrame({
        'id': df['id'],
        'winner_model_a': all_probs[:, 0],
        'winner_model_b': all_probs[:, 1],
        'winner_tie': all_probs[:, 2],
    })
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    submission.to_csv(args.out, index=False)
    
    print(f"\n[Info] ✅ Saved submission to: {args.out}")
    print(f"[Info] Sample predictions:")
    print(submission.head())
    print(f"\n[Info] Mean probabilities:")
    print(f"  model_a: {all_probs[:, 0].mean():.4f}")
    print(f"  model_b: {all_probs[:, 1].mean():.4f}")
    print(f"  tie:     {all_probs[:, 2].mean():.4f}")


if __name__ == '__main__':
    main()
