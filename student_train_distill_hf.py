"""
Student training with knowledge distillation (KL + CE + optional MSE) using Hugging Face Trainer.
- Supports multiple teacher files (.npy logits, .pt logits, or .pt probabilities) and/or a single OOF parquet.
- Two alignment modes:
    1) Direct alignment: provide a fold-specific train CSV (data/fold_data/fold_k_train.csv) whose row order matches teacher outputs
    2) OOF alignment: provide the OOF table path; we'll align teacher probabilities by orig_idx to the fold CSV row order
- Loss: alpha * KL(student||teacher) at T_soft + (1-alpha) * CE(labels) + mse_weight * MSE(student_probs_T, teacher_probs)
- Uses Kaggle-style log loss for evaluation metrics
"""

from typing import Dict, List, Optional, Tuple
import json
import os
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    TrainerCallback,
)
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None
from scipy.special import softmax
from sklearn.metrics import log_loss
import time
import argparse
import re
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

LABEL_MAP = {"model_a": 0, "model_b": 1, "tie": 2, "tie (both bad)": 2}


def build_input_text(row: pd.Series) -> str:
    return f"[PROMPT]{str(row['prompt']).strip()}[RESPONSE_A]{str(row['response_a']).strip()}[RESPONSE_B]{str(row['response_b']).strip()}"


def _label_from_df(df: pd.DataFrame) -> np.ndarray:
    if 'winner' in df.columns:
        return df['winner'].map(LABEL_MAP).values
    trio = None
    candidates = [
        ['winner_model_a', 'winner_model_b', 'winner_tie'],
        ['winner_model_a_prob', 'winner_model_b_prob', 'winner_tie_prob'],
    ]
    for cand in candidates:
        if all(c in df.columns for c in cand):
            trio = cand
            break
    if trio is None:
        raise ValueError("Could not find labels: expected 'winner' or probability columns")
    return df[trio].astype(float).values.argmax(axis=1)


def _load_base_csv_with_idx(train_csv: str) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    text_cols = ['prompt', 'response_a', 'response_b']
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in train.csv: {c}")
    labels = _label_from_df(df)
    df = df.dropna(subset=text_cols).copy()
    df['label'] = labels
    df = df.dropna(subset=['label'])
    # Base index aligned with post-processed order (for teacher logits alignment)
    df['base_idx'] = np.arange(len(df), dtype=np.int64)
    return df[['prompt', 'response_a', 'response_b', 'label', 'base_idx']]


def _load_extra_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize schema similar to baseline
    def _norm(df_in: pd.DataFrame) -> pd.DataFrame:
        df_in = df_in.copy()
        alt_a = ['response_a', 'answer_a', 'assistant_a', 'chosen', 'output_1', 'completion_a']
        alt_b = ['response_b', 'answer_b', 'assistant_b', 'rejected', 'output_2', 'completion_b']
        a_col = next((c for c in alt_a if c in df_in.columns), None)
        b_col = next((c for c in alt_b if c in df_in.columns), None)
        prompt_col = 'prompt' if 'prompt' in df_in.columns else next((c for c in ['question','instruction','query','messages','prompt_text','prompt_a'] if c in df_in.columns), None)
        cols = {}
        if prompt_col: cols['prompt'] = df_in[prompt_col]
        if a_col: cols['response_a'] = df_in[a_col]
        if b_col: cols['response_b'] = df_in[b_col]
        return pd.DataFrame(cols) if cols else df_in
    df = _norm(df)
    text_cols = ['prompt', 'response_a', 'response_b']
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in extra CSV: {path} -> missing {c}; available: {list(df.columns)}")
    try:
        labels = _label_from_df(df)
    except Exception:
        # No labels in extra; assume response_a is preferred if chosen/rejected source
        labels = np.full(len(df), LABEL_MAP['model_a'])
    df = df.dropna(subset=text_cols).copy()
    df['label'] = labels
    df = df.dropna(subset=['label'])
    df['base_idx'] = -1  # no teacher logits available
    return df[['prompt', 'response_a', 'response_b', 'label', 'base_idx']]


def load_dataset(train_csv: str, max_samples: Optional[int] = None, extra_csvs: Optional[List[str]] = None, shuffle_ab: bool = False, dedup_by_prompt: bool = False) -> Dataset:
    base_df = _load_base_csv_with_idx(train_csv)
    dfs = [base_df]
    if extra_csvs:
        for p in extra_csvs:
            if p and os.path.exists(p):
                dfs.append(_load_extra_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    # Deduplicate
    if dedup_by_prompt:
        df = df.drop_duplicates(subset=['prompt']).reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=['prompt', 'response_a', 'response_b']).reset_index(drop=True)

    # Optional A/B shuffle only for extra rows (to avoid needing to swap teacher probs)
    if shuffle_ab:
        mask_extra = df['base_idx'] < 0
        extra_idx = df[mask_extra].sample(frac=0.5, random_state=42).index
        a = df.loc[extra_idx, 'response_a'].copy()
        b = df.loc[extra_idx, 'response_b'].copy()
        df.loc[extra_idx, 'response_a'] = b
        df.loc[extra_idx, 'response_b'] = a
        lab = df.loc[extra_idx, 'label'].astype(int).copy()
        flipped = lab.map({0: 1, 1: 0}).fillna(2).astype(int)
        df.loc[extra_idx, 'label'] = flipped

    df['text'] = df.apply(build_input_text, axis=1)
    # idx used for teacher alignment (-1 means no teacher)
    df['idx'] = df['base_idx'].astype(int)

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).sort_index()

    ds = Dataset.from_pandas(df[['text', 'label', 'idx']])
    # Trainer expects 'labels' key
    if 'label' in ds.column_names and 'labels' not in ds.column_names:
        ds = ds.rename_column('label', 'labels')
    return ds


def tokenize_function(examples: Dict, tokenizer, max_length: int = 512) -> Dict:
    # Dynamic padding via DataCollatorWithPadding (no padding here)
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
    )


class DistillTrainer(Trainer):
    def __init__(self, *args, teacher_probs: np.ndarray, alpha: float = 0.7, T_soft: float = 3.0, label_smoothing: float = 0.0, mse_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_probs = teacher_probs  # shape [N,3] aligned to dataset idx
        self.alpha = float(alpha)
        self.T_soft = float(T_soft)
        self.label_smoothing = float(label_smoothing)
        self.mse_weight = float(mse_weight)
        self.kl = KLDivLoss(reduction='batchmean')

    # Avoid moving models that were loaded with device_map/bitsandbytes which would break offloading and OOM
    def _move_model_to_device(self, model, device):
        try:
            if getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False) or hasattr(model, 'hf_device_map'):
                return model
        except Exception:
            pass
        return super()._move_model_to_device(model, device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pull out indices for teacher alignment (may be absent if removed by collator/args)
        idx = inputs.pop('idx', None)
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.logits  # [B,3]

        # CE loss with optional label smoothing
        ce = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        # KL distillation at T_soft (mask rows without teacher)
        student_logp_T = F.log_softmax(logits / self.T_soft, dim=-1)
        with torch.no_grad():
            if idx is None:
                # Synthesize a mask of invalid indices; no KD will be applied for this batch
                bsz = logits.size(0)
                idx_np = np.full((bsz,), -1, dtype=np.int64)
                idx = torch.from_numpy(idx_np).to(logits.device)
            else:
                idx_np = idx.detach().cpu().numpy()
        # prepare teacher batch aligned to dataset indices (idx expected 0..N-1)
        in_range_mask = (idx >= 0) & (idx < len(self.teacher_probs))
        if in_range_mask.any():
            with torch.no_grad():
                tp_batch_full = torch.from_numpy(self.teacher_probs[idx_np]).to(logits.device)
            # valid if sum > 0 and finite
            valid_mask = in_range_mask & torch.isfinite(tp_batch_full).all(dim=-1) & (tp_batch_full.sum(dim=-1) > 0)
            if valid_mask.any():
                tp_valid = tp_batch_full[valid_mask]
                slp_valid = student_logp_T[valid_mask]
                kl_elem = F.kl_div(slp_valid, tp_valid, reduction='none').sum(dim=-1)  # [n_valid]
                kl = kl_elem.sum() / logits.size(0)
            else:
                kl = torch.tensor(0.0, device=logits.device)
        else:
            kl = torch.tensor(0.0, device=logits.device)
        kl = kl * (self.T_soft ** 2)

        # Optional third loss: MSE between student probs@T and teacher probs
        if self.mse_weight > 0 and in_range_mask.any():
            with torch.no_grad():
                tp_full = torch.from_numpy(self.teacher_probs[idx_np]).to(logits.device)
                valid_mask_for_mse = torch.isfinite(tp_full).all(dim=-1) & (tp_full.sum(dim=-1) > 0)
            sp_T = F.softmax(logits / self.T_soft, dim=-1)
            if valid_mask_for_mse.any():
                mse = F.mse_loss(sp_T[valid_mask_for_mse], tp_full[valid_mask_for_mse], reduction='mean')
            else:
                mse = torch.tensor(0.0, device=logits.device)
        else:
            mse = torch.tensor(0.0, device=logits.device)

        loss = self.alpha * kl + (1.0 - self.alpha) * ce + self.mse_weight * mse
        return (loss, outputs) if return_outputs else loss


def _load_teacher_matrix(path: str) -> np.ndarray:
    """Load a [N,3] array from .npy or .pt. Accepts logits, logprobs, or probs.
    Heuristics:
      - .npy: assume raw logits; softmax applied later by caller
      - .pt: torch.load -> tensor; if values in [0,1] and row-sum ~1 => treat as probs; else if looks like logprobs (mostly negative) => softmax over them.
    Returns float32 probabilities (normalized) when possible. For raw logits, caller can pass through softmax(T) externally.
    """
    if path.endswith('.npy'):
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Teacher file wrong shape: {path} -> {getattr(arr,'shape',None)}")
        return arr.astype(np.float32)
    elif path.endswith('.pt') or path.endswith('.pth'):
        t = torch.load(path, map_location='cpu')
        if isinstance(t, dict) and all(k in t for k in ('topk_values','topk_indices')):
            raise ValueError(f"Unsupported last-token top-k structure in {path}; expected [N,3] array/tensor.")
        if isinstance(t, torch.Tensor):
            arr = t.detach().cpu().numpy()
        else:
            arr = np.array(t)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Teacher file wrong shape: {path} -> {getattr(arr,'shape',None)}")
        return arr.astype(np.float32)
    else:
        raise ValueError(f"Unsupported teacher file extension: {path}")

def average_teacher_probs(teacher_files: List[str], num_classes: int = 3, T_soft: float = 3.0, assume_logits: Optional[bool] = None) -> np.ndarray:
    """Load one or more teacher files and return averaged probabilities.
    - If assume_logits is None: auto-detect per file (.npy => logits; .pt heuristic)
    - For arrays that look like probabilities (sum ~1 and in [0,1]), use directly
    - For arrays that look like logprobs (mostly negative), apply softmax to convert
    - For arrays that look like raw logits (mixed signs), also apply softmax
    """
    probs_list = []
    for path in teacher_files:
        arr = _load_teacher_matrix(path)
        use_softmax = False
        if assume_logits is True:
            use_softmax = True
        elif assume_logits is False:
            use_softmax = False
        else:
            # auto
            row_sum = np.nansum(arr, axis=1)
            in_01 = (arr >= 0).all() and (arr <= 1).all()
            approx_prob = in_01 and np.allclose(row_sum[: min(10, len(row_sum))], 1.0, atol=1e-3)
            mostly_negative = (np.nanmedian(arr) < 0) and (np.nanmax(arr) < 10)
            use_softmax = not approx_prob
            if mostly_negative:
                use_softmax = True
        p = softmax(arr / T_soft, axis=1) if use_softmax else arr
        # normalize just in case
        p = np.clip(p, 1e-9, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        probs_list.append(p.astype(np.float32))
    probs = np.mean(probs_list, axis=0) if len(probs_list) > 1 else probs_list[0]
    return probs.astype(np.float32)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(logits, axis=1)
    ll = float(log_loss(labels, probs, labels=[0, 1, 2]))
    preds = probs.argmax(axis=1)
    acc = float((preds == labels).mean())
    return {'log_loss': ll, 'accuracy': acc}


def _parse_temp_schedule(spec: str, num_epochs: int, T_init: float) -> List[float]:
    """Parse a simple schedule string like 'linear:5,2' -> linear from T_init to 2 over 5 epochs.
    Returns list length num_epochs of per-epoch temperatures. Unknown spec returns constant list.
    """
    temps = [T_init for _ in range(int(num_epochs))]
    if not spec:
        return temps
    try:
        mode, rest = spec.split(':', 1)
        parts = rest.split(',')
        if mode == 'linear' and len(parts) == 2:
            span = int(parts[0])
            T_end = float(parts[1])
            span = max(1, min(span, int(num_epochs)))
            for e in range(span):
                frac = e / max(1, span - 1)
                temps[e] = (1 - frac) * T_init + frac * T_end
            for e in range(span, int(num_epochs)):
                temps[e] = T_end
    except Exception:
        return temps
    return temps


def _load_fold_csv_for_pairwise(fold_train_csv: str) -> pd.DataFrame:
    """Load a fold-specific train CSV that contains pairwise columns and return a cleaned DataFrame
    with columns [prompt, response_a, response_b, label]. Rows lacking A/B are dropped.
    """
    df = pd.read_csv(fold_train_csv)
    # try to locate alt names
    if 'response_a' not in df.columns or 'response_b' not in df.columns:
        # attempt to construct from alternatives; if not present, drop
        alt_a = ['response_a','answer_a','assistant_a','output_1','completion_a']
        alt_b = ['response_b','answer_b','assistant_b','output_2','completion_b']
        a_col = next((c for c in alt_a if c in df.columns), None)
        b_col = next((c for c in alt_b if c in df.columns), None)
        if a_col and b_col:
            df = df.rename(columns={a_col:'response_a', b_col:'response_b'})
    text_cols = ['prompt','response_a','response_b']
    missing = [c for c in text_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Fold CSV missing required columns {missing} in {fold_train_csv}")
    # derive labels
    labels = _label_from_df(df)
    df = df.dropna(subset=text_cols).copy()
    df['label'] = labels[:len(df)]
    df = df.dropna(subset=['label']).copy()
    # ensure int label
    df['label'] = df['label'].astype(int)
    df = df.reset_index(drop=True)
    return df[['prompt','response_a','response_b','label']]


def _teacher_from_oof(oof_path: str, fold_idx: int, model_name: str, n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load teacher probabilities for a given fold and model from an OOF parquet/csv.
    Returns (probs[N,3], present_mask[N]) aligned to row order of the fold CSV.
    Missing rows will have zeros and present_mask False.
    """
    import pandas as pd
    if oof_path.endswith('.parquet'):
        oof = pd.read_parquet(oof_path)
    else:
        oof = pd.read_csv(oof_path)
    oof = oof[(oof['split']=='train') & (oof['fold']==fold_idx) & (oof['model']==model_name)]
    probs = np.zeros((n_rows, 3), dtype=np.float32)
    present = np.zeros((n_rows,), dtype=bool)
    # We expect 'orig_idx' in OOF rows pointing to the row index inside the fold CSV
    if 'orig_idx' not in oof.columns:
        # fallback to row_id if orig_idx absent
        oof['orig_idx'] = oof['row_id']
    for _, r in oof.iterrows():
        i = int(r['orig_idx'])
        if 0 <= i < n_rows:
            probs[i, 0] = float(r['pA'])
            probs[i, 1] = float(r['pB'])
            probs[i, 2] = float(r['pTie'])
            present[i] = True
    # Normalize any non-zero rows
    row_sums = probs.sum(axis=1, keepdims=True)
    nz = row_sums.squeeze(-1) > 0
    probs[nz] = probs[nz] / row_sums[nz]
    return probs, present


def train_student_distill(
    train_csv: str = './data/train.csv',
    output_dir: str = './model_save/student_distilled',
    teacher_logits: Optional[List[str]] = None,
    teacher_oof_table: Optional[str] = None,
    fold_train_csv: Optional[str] = None,
    teacher_model_name: str = 'llama',
    model_name: str = 'google/gemma-2-9b-it',
    max_samples: Optional[int] = None,
    num_epochs: int = 1,
    alpha: float = 0.7,
    T_soft: float = 3.0,
    temp_schedule: str = '',
    label_smoothing: float = 0.0,
    mse_weight: float = 0.0,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.06,
    fp16: bool = False,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    early_stopping_patience: int = 2,
    max_length: int = 512,
    extra_csvs: Optional[List[str]] = None,
    shuffle_ab: bool = False,
    dedup_by_prompt: bool = False,
    use_fast_tokenizer: bool = True,
    dataloader_num_workers: int = 0,
    num_folds: int = 1,
    fold_idx: int = 0,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    max_steps: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
    evaluation_strategy: str = 'epoch',
    save_strategy: str = 'epoch',
    save_steps: Optional[int] = None,
    save_total_limit: int = 1,
    logging_steps: int = 20,
    calibration: Optional[str] = None,
):
    # Validate inputs
    if not teacher_logits and not (teacher_oof_table and fold_train_csv):
        raise ValueError('Provide either teacher_logits files or both teacher_oof_table and fold_train_csv for alignment')
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=bool(use_fast_tokenizer))
    # Capability-based precision handling: V100 doesn't support bf16
    if bf16:
        bf16_supported = False
        try:
            bf16_supported = bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
        except Exception:
            bf16_supported = False
        if not bf16_supported:
            bf16 = False
            fp16 = True
    logger.info(f"Precision config -> bf16={bf16}, fp16={fp16}")
    # Build dataset either from fold CSV (preferred for OOF alignment) or general train.csv + extras
    if fold_train_csv:
        fold_df = _load_fold_csv_for_pairwise(fold_train_csv)
        # Optional max_samples
        if max_samples is not None and len(fold_df) > max_samples:
            fold_df = fold_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        fold_df['text'] = fold_df.apply(build_input_text, axis=1)
        fold_df['idx'] = np.arange(len(fold_df), dtype=np.int64)
        ds = Dataset.from_pandas(fold_df[['text','label','idx']])
        if 'label' in ds.column_names and 'labels' not in ds.column_names:
            ds = ds.rename_column('label','labels')
        # simple split
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds, val_ds = split['train'], split['test']
        logger.info(f"[Distill] Using fold CSV dataset: total={len(ds)} train={len(train_ds)} eval={len(val_ds)}")
    else:
        ds = load_dataset(train_csv, max_samples=max_samples, extra_csvs=extra_csvs, shuffle_ab=shuffle_ab, dedup_by_prompt=dedup_by_prompt)
        # Fold-aware split: deterministic K-fold over dataset order
        if num_folds and num_folds > 1:
            n = len(ds)
            fold_idx = int(fold_idx) % int(num_folds)
            fold_size = n // int(num_folds)
            start = fold_idx * fold_size
            end = n if fold_idx == num_folds - 1 else (start + fold_size)
            indices = np.arange(n)
            val_mask = (indices >= start) & (indices < end)
            train_mask = ~val_mask
            train_ds = ds.select(np.where(train_mask)[0].tolist())
            val_ds = ds.select(np.where(val_mask)[0].tolist())
            logger.info(f"[Distill] Using fold {fold_idx}/{num_folds}: train={len(train_ds)} eval={len(val_ds)}")
        else:
            split = ds.train_test_split(test_size=0.1, seed=42)
            train_ds, val_ds = split['train'], split['test']

    # Multiprocess tokenization + dynamic padding
    _num_proc = max(1, min(8, int(dataloader_num_workers) or 1))
    train_tok = train_ds.map(lambda x: tokenize_function(x, tokenizer, max_length=max_length), batched=True, num_proc=_num_proc)
    val_tok = val_ds.map(lambda x: tokenize_function(x, tokenizer, max_length=max_length), batched=True, num_proc=_num_proc)

    # Build optional quantization config
    quantization_config = None
    if (load_in_4bit or load_in_8bit) and BitsAndBytesConfig is not None:
        if load_in_4bit:
            compute_dtype = torch.bfloat16 if bf16 else torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model (quantized if requested). Use device_map='auto' when quantized to enable offloading/sharding.
    if quantization_config is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            quantization_config=quantization_config,
            device_map='auto',
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            try:
                model.gradient_checkpointing_enable(use_reentrant=False)
            except TypeError:
                model.gradient_checkpointing_enable()

    # Optionally enable LoRA for memory-efficient fine-tuning on large models
    if use_lora and get_peft_model is not None:
        try:
            if (load_in_4bit or load_in_8bit) and prepare_model_for_kbit_training is not None:
                model = prepare_model_for_kbit_training(model)
            # Detect target modules heuristically
            cand = set()
            for n, _m in model.named_modules():
                for key in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'):  # common for Gemma/LLaMA
                    if key in n:
                        cand.add(key)
            target_modules = sorted(list(cand)) or ['q_proj','v_proj','o_proj']
            lora_cfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                bias='none',
                task_type='SEQ_CLS',
                target_modules=target_modules,
            )
            model = get_peft_model(model, lora_cfg)
            try:
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                logger.info(f"[Distill] LoRA enabled. Trainable params={trainable:,} / total={total:,}")
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"[Distill] Failed to enable LoRA: {e}")

    # Load and align teacher probs
    if teacher_logits:
        teacher_probs = average_teacher_probs(teacher_logits, num_classes=3, T_soft=T_soft)
        # a, b = load_calibration(calibration)
        # teacher_probs = apply_calibration(teacher_probs, a, b)
        # If probs look like logits (sum not ~1), they were softmaxed inside average_teacher_probs
        if fold_train_csv:
            # Expect exact alignment
            if len(teacher_probs) != len(ds):
                raise ValueError(f'Teacher probs length {len(teacher_probs)} must equal dataset length {len(ds)} when using fold_train_csv alignment')
        else:
            if len(teacher_probs) < len(ds):
                raise ValueError('Teacher logits length is smaller than dataset; ensure alignment/order match')
            # truncate if longer
            teacher_probs = teacher_probs[:len(ds)]
    elif teacher_oof_table and fold_train_csv:
        # Align via OOF
        n_rows = len(ds)
        teacher_probs, present_mask = _teacher_from_oof(teacher_oof_table, fold_idx, teacher_model_name, n_rows)
        # a, b = load_calibration(calibration)
        # teacher_probs = apply_calibration(teacher_probs, a, b)
        logger.info(f"[Distill] Loaded OOF teacher probs for fold={fold_idx}, model={teacher_model_name} -> present={present_mask.sum()}/{n_rows}")
    else:
        raise ValueError('Invalid teacher specification')

    # Training args
    pin_mem = bool(torch.cuda.is_available() or (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()))
    # Build TrainingArguments with version compatibility: filter unsupported kwargs
    import inspect
    base_kwargs = dict(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        save_total_limit=int(save_total_limit),
        load_best_model_at_end=True,
        metric_for_best_model='log_loss',
        greater_is_better=False,
        logging_steps=int(logging_steps),
        label_smoothing_factor=0.0,  # handled in CE above
        fp16=fp16,
        bf16=bf16,
        warmup_ratio=warmup_ratio,
        dataloader_pin_memory=pin_mem,
        dataloader_num_workers=int(dataloader_num_workers),
        group_by_length=True,
        report_to='none',
        remove_unused_columns=False,  # keep 'idx' for teacher alignment
    )
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    # Add fallbacks for older transformers that don't support evaluation/save strategy
    if 'evaluation_strategy' not in allowed:
        # Use step-based evaluation if available
        if 'do_eval' in allowed:
            base_kwargs['do_eval'] = True
        if 'eval_steps' in allowed:
            # fallback to evaluating ~every epoch assuming ~len(train_ds) steps with bs/accum
            base_kwargs['eval_steps'] = 500
        # Remove unsupported key
        base_kwargs.pop('evaluation_strategy', None)
        # Also avoid mismatch checks in older versions
        if 'load_best_model_at_end' in base_kwargs:
            base_kwargs['load_best_model_at_end'] = False
        # Ensure EarlyStoppingCallback has a metric to monitor
        if 'metric_for_best_model' in allowed:
            base_kwargs['metric_for_best_model'] = 'log_loss'
        if 'greater_is_better' in allowed:
            base_kwargs['greater_is_better'] = False
    if 'save_strategy' not in allowed:
        if 'save_steps' in allowed:
            base_kwargs['save_steps'] = base_kwargs.get('eval_steps', 500)
        base_kwargs.pop('save_strategy', None)
    # If evaluation_strategy isn't allowed but save_strategy is, avoid setting save_strategy to prevent mismatch
    if ('evaluation_strategy' not in allowed) and ('save_strategy' in allowed):
        base_kwargs.pop('save_strategy', None)
    # If both evaluation_strategy and save_strategy are supported, honor provided values
    if ('evaluation_strategy' in allowed) and ('save_strategy' in allowed):
        base_kwargs['evaluation_strategy'] = evaluation_strategy
        base_kwargs['save_strategy'] = save_strategy
        if save_strategy == 'steps' and ('save_steps' in allowed) and (save_steps is not None):
            base_kwargs['save_steps'] = int(save_steps)
    if 'group_by_length' not in allowed:
        base_kwargs.pop('group_by_length', None)
    if 'report_to' not in allowed:
        base_kwargs.pop('report_to', None)
    if 'dataloader_pin_memory' not in allowed:
        base_kwargs.pop('dataloader_pin_memory', None)
    if 'dataloader_num_workers' not in allowed:
        base_kwargs.pop('dataloader_num_workers', None)
    if 'label_smoothing_factor' not in allowed:
        base_kwargs.pop('label_smoothing_factor', None)
    if 'warmup_ratio' not in allowed:
        base_kwargs.pop('warmup_ratio', None)
    if 'remove_unused_columns' not in allowed:
        base_kwargs.pop('remove_unused_columns', None)
    # Optional cap on total training steps
    if max_steps is not None and 'max_steps' in allowed:
        base_kwargs['max_steps'] = int(max_steps)
    if 'metric_for_best_model' not in allowed:
        base_kwargs.pop('metric_for_best_model', None)
    if 'greater_is_better' not in allowed:
        base_kwargs.pop('greater_is_better', None)
    if 'load_best_model_at_end' not in allowed:
        base_kwargs.pop('load_best_model_at_end', None)

    args = TrainingArguments(**{k: v for k, v in base_kwargs.items() if k in allowed})
    _pad_to_multiple = 8 if (fp16 or bf16) else None
    base_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=_pad_to_multiple)
    def _collate_keep_idx(batch):
        # Extract idx and labels; drop raw 'text' prior to padding
        idx_list = [ex.get('idx', -1) for ex in batch]
        labels_list = [ex.get('labels', ex.get('label')) for ex in batch]
        # Strip non-tokenizer fields for padding
        toks_only = []
        for ex in batch:
            ex2 = {k: v for k, v in ex.items() if k not in ('text','idx','label','labels')}
            toks_only.append(ex2)
        out = base_collator(toks_only)
        if labels_list is not None:
            out['labels'] = torch.tensor(labels_list, dtype=torch.long)
        out['idx'] = torch.tensor(idx_list, dtype=torch.long)
        return out
    collator = _collate_keep_idx

    # Throughput logging
    class ThroughputLoggerCallback(TrainerCallback):
        def __init__(self, ds):
            self._ds = ds
            self._epoch_start = None
            # estimate total tokens from attention_mask if present
            self._examples = len(ds)
            toks = 0
            try:
                for ex in ds:
                    am = ex.get('attention_mask')
                    if am is not None:
                        toks += int(sum(am))
            except Exception:
                toks = self._examples * max_length
            self._tokens = toks
            if self._examples > 0:
                avg_len = toks / self._examples
                logger.info(f"[Distill] Train set -> examples={self._examples}, total_tokens={self._tokens}, avg_len={avg_len:.1f}")

        def on_epoch_begin(self, args, state, control, **kwargs):
            self._epoch_start = time.time()

        def on_epoch_end(self, args, state, control, **kwargs):
            if self._epoch_start is None:
                return
            dt = max(1e-6, time.time() - self._epoch_start)
            ex_per_sec = self._examples / dt
            tok_per_sec = self._tokens / dt
            ep = state.epoch if state.epoch is not None else 0
            logger.info(f"[Distill] Epoch {int(ep)} finished in {dt:.2f}s | examples/sec: {ex_per_sec:.2f} | tokens/sec: {tok_per_sec:.2f}")

    # Optional per-epoch temperature schedule (adjusts T_soft used in KL and MSE)
    temps = _parse_temp_schedule(temp_schedule, num_epochs, T_soft)

    # Build callbacks conditionally: EarlyStopping requires an eval strategy.
    support_eval_strategy = 'evaluation_strategy' in allowed
    callbacks_list = [ThroughputLoggerCallback(train_tok)]
    if support_eval_strategy and early_stopping_patience and early_stopping_patience > 0:
        callbacks_list.insert(0, EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = DistillTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks_list,
        teacher_probs=teacher_probs,
        alpha=alpha,
        T_soft=T_soft,
        label_smoothing=label_smoothing,
        mse_weight=mse_weight,
    )

    # Hook to update trainer temperature at each epoch begin
    class _TempUpdater(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            ep = int(state.epoch or 0)
            if 0 <= ep < len(temps):
                new_T = float(temps[ep])
                if getattr(trainer, 'T_soft', None) != new_T:
                    trainer.T_soft = new_T
                    logger.info(f"[Distill] Updated temperature T_soft -> {new_T}")

    trainer.add_callback(_TempUpdater())

    # Train (auto-resume if not explicitly provided)
    def _find_latest_checkpoint(dir_path: str) -> Optional[str]:
        try:
            if not os.path.isdir(dir_path):
                return None
            ckpts = []
            for name in os.listdir(dir_path):
                if name.startswith('checkpoint-'):
                    m = re.match(r'^checkpoint-(\d+)$', name)
                    if m:
                        step = int(m.group(1))
                        ckpt_dir = os.path.join(dir_path, name)
                        # Consider it complete if trainer_state.json exists (most indicative)
                        state_ok = os.path.isfile(os.path.join(ckpt_dir, 'trainer_state.json'))
                        if state_ok:
                            ckpts.append((step, ckpt_dir))
            if not ckpts:
                return None
            ckpts.sort(key=lambda x: x[0], reverse=True)
            return ckpts[0][1]
        except Exception:
            return None

    auto_resume = None
    if resume_from_checkpoint in (None, "", "none"):
        auto_resume = _find_latest_checkpoint(output_dir)
        if auto_resume:
            logger.info(f"[Distill] Auto-resuming from latest checkpoint: {auto_resume}")
            resume_from_checkpoint = auto_resume
        else:
            logger.info("[Distill] No resume checkpoint provided and none found in output_dir; starting fresh.")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = trainer.evaluate()
    # Persist CV metrics for ensembling weights
    try:
        os.makedirs(output_dir, exist_ok=True)
        fold_tag = f"fold_{fold_idx}_of_{num_folds}" if num_folds and num_folds > 1 else "fold_single"
        with open(os.path.join(output_dir, f"cv_metrics_{fold_tag}.json"), "w") as f:
            import json
            json.dump(metrics, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write CV metrics: {e}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return metrics


## Calibration code commented out
# def load_calibration(calib_path):
#     if calib_path is None:
#         return None, None
#     if calib_path.endswith('.npz'):
#         params = np.load(calib_path)
#         a = params['a']
#         b = params['b']
#     elif calib_path.endswith('.json'):
#         with open(calib_path) as f:
#             params = json.load(f)
#         a = np.array(params['a'])
#         b = np.array(params['b'])
#     else:
#         a = b = None
#     return a, b


## Calibration code commented out
# def apply_calibration(logits, a, b):
#     if a is not None and b is not None:
#         logits = logits.astype(np.float32)
#         a = a.astype(np.float32)
#         b = b.astype(np.float32)
#         return logits * a + b
#     return logits.astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3-class student with knowledge distillation')
    parser.add_argument('--train_csv', type=str, default='./data/train.csv')
    parser.add_argument('--fold_train_csv', type=str, default=None, help='Path to data/fold_data/fold_k_train.csv for alignment')
    parser.add_argument('--teacher_logits', type=str, nargs='*', default=None, help='One or more teacher files (.npy/.pt)')
    parser.add_argument('--teacher_oof_table', type=str, default=None, help='OOF parquet/csv path produced by step4')
    parser.add_argument('--teacher_model_name', type=str, default='llama', help='Filter when using OOF table')
    parser.add_argument('--output_dir', type=str, default='./model_save/student_distilled')
    parser.add_argument('--model_name', type=str, default='google/gemma-2-9b-it')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--T_soft', type=float, default=3.0)
    parser.add_argument('--temp_schedule', type=str, default='')
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--mse_weight', type=float, default=0.1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--early_stopping_patience', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--extra_csvs', type=str, nargs='*', default=None)
    parser.add_argument('--shuffle_ab', action='store_true')
    parser.add_argument('--dedup_by_prompt', action='store_true')
    parser.add_argument('--use_fast_tokenizer', action='store_true')
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=1)
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    # Checkpoint/resume controls
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--evaluation_strategy', type=str, default='epoch', choices=['no','steps','epoch'])
    parser.add_argument('--save_strategy', type=str, default='epoch', choices=['no','steps','epoch'])
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=20)
    parser.add_argument('--calibration', type=str, default=None, help='Path to vector scaling calibration file (.json or .npz)')

    args = parser.parse_args()

    metrics = train_student_distill(
        train_csv=args.train_csv,
        fold_train_csv=args.fold_train_csv,
        teacher_logits=args.teacher_logits,
        teacher_oof_table=args.teacher_oof_table,
        teacher_model_name=args.teacher_model_name,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_samples=args.max_samples,
        num_epochs=args.num_epochs,
        alpha=args.alpha,
        T_soft=args.T_soft,
        temp_schedule=args.temp_schedule,
        label_smoothing=args.label_smoothing,
        mse_weight=args.mse_weight,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        early_stopping_patience=args.early_stopping_patience,
        max_length=args.max_length,
        extra_csvs=args.extra_csvs,
        shuffle_ab=args.shuffle_ab,
        dedup_by_prompt=args.dedup_by_prompt,
        use_fast_tokenizer=args.use_fast_tokenizer,
        dataloader_num_workers=args.dataloader_num_workers,
        num_folds=args.num_folds,
        fold_idx=args.fold_idx,
        max_steps=args.max_steps,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        resume_from_checkpoint=args.resume_from_checkpoint,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        calibration=args.calibration,
    )
    print({'eval': metrics})
