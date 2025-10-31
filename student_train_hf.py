# student_train_hf.py
"""
Minimal real training loop for a small student classifier using Hugging Face Trainer.
- Model: distilbert-base-uncased (sequence classification, 3 labels)
- Data: ./data/train.csv with columns prompt,response_a,response_b,winner
- Output: ./model_save/student_distilbert (adapter-free; full finetune on CPU/GPU)
"""

import os
import time
import pandas as pd
from typing import Dict, Optional, List

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
)
from transformers import EarlyStoppingCallback, TrainerCallback
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)
from scipy.special import softmax
from sklearn.metrics import log_loss

LABEL_MAP = {"model_a": 0, "model_b": 1, "tie": 2, "tie (both bad)": 2}


def _make_training_args(**kwargs) -> TrainingArguments:
    """Create TrainingArguments while handling older Transformers that may not accept
    evaluation_strategy. Falls back to eval_strategy if needed.
    """
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        if 'evaluation_strategy' in kwargs:
            # fallback name seen in older releases
            kwargs['eval_strategy'] = kwargs.pop('evaluation_strategy')
        return TrainingArguments(**kwargs)


def dataset_stats(
    train_csv: str,
    max_samples: Optional[int] = None,
    extra_csvs: Optional[List[str]] = None,
    shuffle_ab: bool = False,
    dedup_by_prompt: bool = False,
) -> Dict[str, int]:
    dfs: List[pd.DataFrame] = [_load_and_unify_csv(train_csv)]
    if extra_csvs:
        for p in extra_csvs:
            if p and os.path.exists(p):
                dfs.append(_load_and_unify_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    merged = len(df)
    if dedup_by_prompt:
        df = df.drop_duplicates(subset=['prompt']).reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=['prompt', 'response_a', 'response_b']).reset_index(drop=True)
    deduped = len(df)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    used = len(df)
    # Basic label distribution
    label_counts = df['label'].value_counts(dropna=False).to_dict()
    return {
        'merged': int(merged),
        'deduped': int(deduped),
        'used': int(used),
        'label_0': int(label_counts.get(0, 0)),
        'label_1': int(label_counts.get(1, 0)),
        'label_2': int(label_counts.get(2, 0)),
    }


def build_input_text(row: pd.Series) -> str:
    return f"[PROMPT]{str(row['prompt']).strip()}[RESPONSE_A]{str(row['response_a']).strip()}[RESPONSE_B]{str(row['response_b']).strip()}"


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Map alternative response column names to response_a/response_b
    alt_a = ['response_a', 'answer_a', 'assistant_a', 'chosen', 'output_1', 'completion_a']
    alt_b = ['response_b', 'answer_b', 'assistant_b', 'rejected', 'output_2', 'completion_b']
    a_col = next((c for c in alt_a if c in df.columns), None)
    b_col = next((c for c in alt_b if c in df.columns), None)

    # Sometimes prompts can be under different names
    prompt_col = 'prompt'
    if 'prompt' not in df.columns:
        for c in ['question', 'instruction', 'query', 'messages', 'prompt_text', 'prompt_a']:
            if c in df.columns:
                prompt_col = c
                break

    # If chosen/rejected present, prefer them
    if a_col is None and 'chosen' in df.columns:
        a_col = 'chosen'
    if b_col is None and 'rejected' in df.columns:
        b_col = 'rejected'

    # Write normalized columns but keep all originals
    if prompt_col in df.columns:
        df['prompt'] = df[prompt_col]
    if a_col is not None:
        df['response_a'] = df[a_col]
    if b_col is not None:
        df['response_b'] = df[b_col]
    return df


def _load_and_unify_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_schema(df)
    text_cols = ['prompt', 'response_a', 'response_b']
    for c in text_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in {path}: {c}. Available: {list(df.columns)}")
    # Determine labels
    if 'winner' in df.columns:
        labels = df['winner'].map(LABEL_MAP)
    else:
        prob_cols = None
        candidates = [
            ['winner_model_a', 'winner_model_b', 'winner_tie'],
            ['winner_model_a_prob', 'winner_model_b_prob', 'winner_tie_prob'],
        ]
        for trio in candidates:
            if all(c in df.columns for c in trio):
                prob_cols = trio
                break
        if prob_cols is not None:
            probs = df[prob_cols].astype(float)
            labels = probs.values.argmax(axis=1)
        else:
            # If schema was chosen/rejected without labels, assume response_a is preferred
            if 'response_a' in df.columns and 'response_b' in df.columns:
                labels = np.full(len(df), LABEL_MAP['model_a'])
            else:
                raise ValueError("Could not find labels: expected 'winner' or probability columns like winner_model_a/b/tie")

    df = df.dropna(subset=text_cols).copy()
    df['label'] = labels
    return df


def load_dataset(train_csv: str, max_samples: Optional[int] = 2000, extra_csvs: Optional[List[str]] = None, shuffle_ab: bool = False, dedup_by_prompt: bool = False) -> Dataset:
    dfs: List[pd.DataFrame] = [_load_and_unify_csv(train_csv)]
    if extra_csvs:
        for p in extra_csvs:
            if p and os.path.exists(p):
                dfs.append(_load_and_unify_csv(p))
    df = pd.concat(dfs, ignore_index=True)
    total_before = len(df)
    # Deduplicate
    if dedup_by_prompt:
        df = df.drop_duplicates(subset=['prompt']).reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=['prompt', 'response_a', 'response_b']).reset_index(drop=True)
    total_after_dedup = len(df)

    # Optional A/B shuffle to reduce position bias
    if shuffle_ab and len(df) > 0:
        rng = pd.Series(range(len(df))).sample(frac=0.5, random_state=42).index
        # swap responses for selected rows
        a = df.loc[rng, 'response_a'].copy()
        b = df.loc[rng, 'response_b'].copy()
        df.loc[rng, 'response_a'] = b
        df.loc[rng, 'response_b'] = a
        # flip labels: 0<->1, 2 stays
        lab = df.loc[rng, 'label'].astype(int).copy()
        flipped = lab.map({0: 1, 1: 0}).fillna(2).astype(int)
        df.loc[rng, 'label'] = flipped

    df['text'] = df.apply(build_input_text, axis=1)
    df = df.dropna(subset=['label'])

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    total_after_sample = len(df)

    logger.info(f"Dataset sizes -> merged={total_before}, deduped={total_after_dedup}, used={total_after_sample}")
    if total_after_sample <= 0:
        raise ValueError("No training rows after merging/dedup/sampling. Check your CSV paths and dedup options.")
    return Dataset.from_pandas(df[['text', 'label']])


def tokenize_function(examples: Dict, tokenizer, max_length: int = 512) -> Dict:
    # Dynamic padding via DataCollatorWithPadding (no padding here)
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
    )


def train_student(
    train_csv: str = './data/train.csv',
    output_dir: str = './model_save/student_distilbert',
    max_samples: Optional[int] = 2000,
    num_epochs: int = 1,
    model_name: str = 'distilbert-base-uncased',
    label_smoothing: float = 0.05,
    early_stopping_patience: int = 2,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    fp16: bool = False,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.06,
    max_length: int = 512,
    extra_csvs: Optional[List[str]] = None,
    shuffle_ab: bool = False,
    dedup_by_prompt: bool = False,
    use_fast_tokenizer: bool = True,
    dataloader_num_workers: int = 0,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=bool(use_fast_tokenizer))
    # Capability-based precision handling: V100 doesn't support bf16
    if bf16:
        bf16_supported = False
        try:
            bf16_supported = bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())
        except Exception:
            bf16_supported = False
        if not bf16_supported:
            # Fallback to fp16 if available
            bf16 = False
            fp16 = True
    logger.info(f"Precision config -> bf16={bf16}, fp16={fp16}")
    dataset = load_dataset(train_csv, max_samples=max_samples, extra_csvs=extra_csvs, shuffle_ab=shuffle_ab, dedup_by_prompt=dedup_by_prompt)

    # Simple train/val split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    # Multiprocess tokenization for speed
    _num_proc = max(1, min(8, int(dataloader_num_workers) or 1))
    tokenized_train = dataset['train'].map(lambda x: tokenize_function(x, tokenizer, max_length=max_length), batched=True, num_proc=_num_proc)
    tokenized_val = dataset['test'].map(lambda x: tokenize_function(x, tokenizer, max_length=max_length), batched=True, num_proc=_num_proc)

    # Precompute totals for throughput metrics
    total_train_examples = len(tokenized_train)
    total_train_tokens = 0
    try:
        for ex in tokenized_train:
            am = ex.get('attention_mask')
            if am is not None:
                total_train_tokens += int(sum(am))
    except Exception:
        # Fallback if iteration fails or column missing
        total_train_tokens = total_train_examples * max_length
    if total_train_examples > 0:
        avg_len = total_train_tokens / total_train_examples
        logger.info(f"Train set -> examples={total_train_examples}, total_tokens={total_train_tokens}, avg_len={avg_len:.1f}")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        # Disable cache and reentrant checkpointing to avoid double-backward issues
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            try:
                model.gradient_checkpointing_enable(use_reentrant=False)
            except TypeError:
                model.gradient_checkpointing_enable()

    pin_mem = bool(torch.cuda.is_available() or (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()))
    args = _make_training_args(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='log_loss',
        greater_is_better=False,
        logging_steps=20,
        label_smoothing_factor=label_smoothing,
        fp16=fp16,
        bf16=bf16,
        warmup_ratio=warmup_ratio,
        dataloader_pin_memory=pin_mem,
        dataloader_num_workers=int(dataloader_num_workers),
        group_by_length=True,
        report_to='none',
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = softmax(logits, axis=1)
        # Kaggle-style multiclass log loss
        ll = float(log_loss(labels, probs, labels=[0, 1, 2]))
        preds = probs.argmax(axis=1)
        acc = float((preds == labels).mean())
        return {'log_loss': ll, 'accuracy': acc, 'eval_loss': ll}

    # Use dynamic padding to reduce average sequence length per batch
    _pad_to_multiple = 8 if (fp16 or bf16) else None
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=_pad_to_multiple)

    class ThroughputLoggerCallback(TrainerCallback):
        def __init__(self, total_examples: int, total_tokens: int):
            self.total_examples = int(total_examples)
            self.total_tokens = int(total_tokens)
            self._epoch_start = None

        def on_epoch_begin(self, args, state, control, **kwargs):
            self._epoch_start = time.time()

        def on_epoch_end(self, args, state, control, **kwargs):
            if self._epoch_start is None:
                return
            dt = max(1e-6, time.time() - self._epoch_start)
            ex_per_sec = self.total_examples / dt
            tok_per_sec = self.total_tokens / dt
            ep = state.epoch if state.epoch is not None else 0
            logger.info(
                f"Epoch {int(ep)} finished in {dt:.2f}s | examples/sec: {ex_per_sec:.2f} | tokens/sec: {tok_per_sec:.2f}"
            )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            ThroughputLoggerCallback(total_examples=total_train_examples, total_tokens=total_train_tokens),
        ],
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return metrics


if __name__ == '__main__':
    m = train_student()
    print({"eval": m})
