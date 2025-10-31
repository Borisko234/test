"""
Online distillation: train a 3-class student by computing teacher logits on-the-fly (no .pt files).
- Loads up to two teacher models (e.g., global LLaMA and Qwen) as sequence classifiers in 8-bit for memory efficiency.
- Aggregates teacher signals via temperature-weighted average: softmax((logits1/T1 + logits2/T2)/k), where k is the number of active teachers.
- Student is trained with KL(student||combined_teacher) at T_soft plus CE to labels.
- Supports LoRA on student and gradient checkpointing for memory.
- Optional A/B flip augmentation for robustness and TTA equivalence.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Tuple
import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None
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
        # default to model_a when unknown
        return np.zeros(len(df), dtype=np.int64)
    return df[trio].astype(float).values.argmax(axis=1)


def load_pairwise_dataset(train_csv: str,
                          max_samples: Optional[int] = None,
                          extra_csvs: Optional[List[str]] = None,
                          dedup_by_prompt: bool = False,
                          flip_ab_prob: float = 0.0) -> Dataset:
    def norm_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'prompt' not in df.columns:
            for c in ['question','instruction','query','messages','prompt_text','prompt_a']:
                if c in df.columns:
                    df['prompt'] = df[c]; break
        if 'response_a' not in df.columns or 'response_b' not in df.columns:
            a = next((c for c in ['response_a','answer_a','assistant_a','chosen','output_1','completion_a'] if c in df.columns), None)
            b = next((c for c in ['response_b','answer_b','assistant_b','rejected','output_2','completion_b'] if c in df.columns), None)
            if a and b:
                df = df.rename(columns={a:'response_a', b:'response_b'})
        for c in ['prompt','response_a','response_b']:
            if c not in df.columns:
                df[c] = ''
        lab = _label_from_df(df)
        df['label'] = lab
        df = df.dropna(subset=['prompt','response_a','response_b']).copy()
        return df[['prompt','response_a','response_b','label']]

    base = norm_df(pd.read_csv(train_csv))
    dfs = [base]
    if extra_csvs:
        for p in extra_csvs:
            if p and os.path.exists(p):
                try:
                    dfs.append(norm_df(pd.read_csv(p)))
                except Exception:
                    pass
    df = pd.concat(dfs, ignore_index=True)
    if dedup_by_prompt:
        df = df.drop_duplicates(subset=['prompt']).reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=['prompt','response_a','response_b']).reset_index(drop=True)

    # Optional A/B flip augmentation
    if flip_ab_prob and flip_ab_prob > 0.0:
        rng = np.random.default_rng(42)
        mask = rng.random(len(df)) < float(flip_ab_prob)
        a = df.loc[mask, 'response_a'].copy()
        b = df.loc[mask, 'response_b'].copy()
        df.loc[mask, 'response_a'] = b
        df.loc[mask, 'response_b'] = a
        lab = df.loc[mask, 'label'].astype(int)
        df.loc[mask, 'label'] = lab.map({0:1, 1:0}).fillna(2).astype(int)

    df['text'] = df.apply(build_input_text, axis=1)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    ds = Dataset.from_pandas(df[['text','label']])
    if 'label' in ds.column_names and 'labels' not in ds.column_names:
        ds = ds.rename_column('label','labels')
    return ds


def tokenize_function(examples: Dict, tokenizer, max_length: int = 512) -> Dict:
    return tokenizer(examples['text'], truncation=True, max_length=max_length)


class OnlineKDTrainer(Trainer):
    def __init__(self, *args, teacher_models: List[AutoModelForSequenceClassification], teacher_temps: List[float], kd_alpha: float, T_soft: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_models = [m.eval() for m in (teacher_models or []) if m is not None]
        self.teacher_temps = [float(t) for t in (teacher_temps or [])]
        self.kd_alpha = float(kd_alpha)
        self.T_soft = float(T_soft)
        assert len(self.teacher_models) == len(self.teacher_temps), "teacher_models and teacher_temps length must match"

    # Avoid moving quantized/sharded models
    def _move_model_to_device(self, model, device):
        try:
            if getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False) or hasattr(model, 'hf_device_map'):
                return model
        except Exception:
            pass
        return super()._move_model_to_device(model, device)

    @torch.no_grad()
    def _combined_teacher_probs(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.teacher_models:
            return None
        acc = None
        k = 0
        for model, temp in zip(self.teacher_models, self.teacher_temps):
            try:
                out = model(**inputs)
                logits = out.logits
                scaled = logits / float(temp)
                acc = scaled if acc is None else (acc + scaled)
                k += 1
            except Exception:
                continue
        if acc is None or k == 0:
            return None
        acc = acc / float(k)
        return F.softmax(acc, dim=-1)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get('labels')
        outputs = model(**{k:v for k,v in inputs.items() if k != 'labels'})
        logits = outputs.logits
        ce = F.cross_entropy(logits, labels)
        # KD
        with torch.no_grad():
            t_probs = self._combined_teacher_probs({k:v for k,v in inputs.items() if k != 'labels'})
        if t_probs is not None:
            student_logp_T = F.log_softmax(logits / self.T_soft, dim=-1)
            kl = F.kl_div(student_logp_T, t_probs, reduction='batchmean') * (self.T_soft ** 2)
        else:
            kl = torch.tensor(0.0, device=logits.device)
        loss = self.kd_alpha * kl + (1.0 - self.kd_alpha) * ce
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser(description='Online distillation training with on-the-fly teacher logits (no .pt files)')
    # Data
    ap.add_argument('--train_csv', type=str, default='data/train.csv')
    ap.add_argument('--extra_csvs', type=str, nargs='*', default=None)
    ap.add_argument('--max_samples', type=int, default=None)
    ap.add_argument('--max_length', type=int, default=512)
    ap.add_argument('--num_folds', type=int, default=1)
    ap.add_argument('--fold_idx', type=int, default=0)
    ap.add_argument('--flip_ab_prob', type=float, default=0.0)
    ap.add_argument('--dedup_by_prompt', action='store_true')
    # Student
    ap.add_argument('--student_model', type=str, default='google/gemma-2-9b-it')
    ap.add_argument('--output_dir', type=str, default='model_save/student_online_distilled')
    ap.add_argument('--num_epochs', type=int, default=1)
    ap.add_argument('--learning_rate', type=float, default=5e-5)
    ap.add_argument('--per_device_train_batch_size', type=int, default=1)
    ap.add_argument('--per_device_eval_batch_size', type=int, default=1)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=8)
    ap.add_argument('--warmup_ratio', type=float, default=0.06)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--gradient_checkpointing', action='store_true')
    ap.add_argument('--dataloader_num_workers', type=int, default=0)
    ap.add_argument('--max_steps', type=int, default=None)
    ap.add_argument('--use_lora', action='store_true')
    ap.add_argument('--lora_r', type=int, default=8)
    ap.add_argument('--lora_alpha', type=int, default=16)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--load_in_4bit', action='store_true')
    ap.add_argument('--load_in_8bit', action='store_true')
    # Teachers
    ap.add_argument('--teacher1_dir', type=str, default=None)
    ap.add_argument('--teacher2_dir', type=str, default=None)
    ap.add_argument('--teacher1_temp', type=float, default=3.0)
    ap.add_argument('--teacher2_temp', type=float, default=3.0)
    ap.add_argument('--teacher_load_in_8bit', action='store_true')
    ap.add_argument('--teacher_load_in_4bit', action='store_true')
    # KD Loss
    ap.add_argument('--kd_alpha', type=float, default=0.7)
    ap.add_argument('--T_soft', type=float, default=3.0)
    # Strategies
    ap.add_argument('--evaluation_strategy', type=str, default='epoch', choices=['no','steps','epoch'])
    ap.add_argument('--save_strategy', type=str, default='epoch', choices=['no','steps','epoch'])
    ap.add_argument('--save_total_limit', type=int, default=1)
    ap.add_argument('--save_steps', type=int, default=None)
    ap.add_argument('--logging_steps', type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, use_fast=True)

    # Dataset and fold split
    ds_all = load_pairwise_dataset(
        train_csv=args.train_csv,
        max_samples=args.max_samples,
        extra_csvs=args.extra_csvs,
        dedup_by_prompt=args.dedup_by_prompt,
        flip_ab_prob=args.flip_ab_prob,
    )
    if args.num_folds and args.num_folds > 1:
        n = len(ds_all)
        f = int(args.fold_idx) % int(args.num_folds)
        fold_size = n // int(args.num_folds)
        start = f * fold_size
        end = n if f == args.num_folds - 1 else (start + fold_size)
        idx = np.arange(n)
        val_mask = (idx >= start) & (idx < end)
        train_mask = ~val_mask
        train_ds = ds_all.select(np.where(train_mask)[0].tolist())
        val_ds = ds_all.select(np.where(val_mask)[0].tolist())
    else:
        split = ds_all.train_test_split(test_size=0.1, seed=42)
        train_ds, val_ds = split['train'], split['test']

    # Tokenize
    num_proc = max(1, min(8, int(args.dataloader_num_workers) or 1))
    train_tok = train_ds.map(lambda x: tokenize_function(x, tokenizer, args.max_length), batched=True, num_proc=num_proc)
    val_tok = val_ds.map(lambda x: tokenize_function(x, tokenizer, args.max_length), batched=True, num_proc=num_proc)

    # Student model (with optional quantization)
    student_quant = None
    if (args.load_in_4bit or args.load_in_8bit) and BitsAndBytesConfig is not None:
        if args.load_in_4bit:
            compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
            student_quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
        elif args.load_in_8bit:
            student_quant = BitsAndBytesConfig(load_in_8bit=True)
    if student_quant is not None:
        student = AutoModelForSequenceClassification.from_pretrained(
            args.student_model, num_labels=3, quantization_config=student_quant, device_map='auto')
    else:
        student = AutoModelForSequenceClassification.from_pretrained(args.student_model, num_labels=3)

    if args.gradient_checkpointing and hasattr(student, 'gradient_checkpointing_enable'):
        if hasattr(student.config, 'use_cache'):
            student.config.use_cache = False
        try:
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            try:
                student.gradient_checkpointing_enable(use_reentrant=False)
            except TypeError:
                student.gradient_checkpointing_enable()

    if args.use_lora and get_peft_model is not None:
        try:
            if (args.load_in_4bit or args.load_in_8bit) and prepare_model_for_kbit_training is not None:
                student = prepare_model_for_kbit_training(student)
            cand = set()
            for n,_m in student.named_modules():
                for key in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'):
                    if key in n:
                        cand.add(key)
            target_modules = sorted(list(cand)) or ['q_proj','v_proj','o_proj']
            lcfg = LoraConfig(r=int(args.lora_r), lora_alpha=int(args.lora_alpha), lora_dropout=float(args.lora_dropout), bias='none', task_type='SEQ_CLS', target_modules=target_modules)
            student = get_peft_model(student, lcfg)
        except Exception as e:
            print(f"[OnlineKD] Warn: failed enabling LoRA on student: {e}")

    # Teacher models (8-bit recommended)
    teacher_quant = None
    if BitsAndBytesConfig is not None:
        if args.teacher_load_in_4bit:
            teacher_quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
        elif args.teacher_load_in_8bit or True:
            teacher_quant = BitsAndBytesConfig(load_in_8bit=True)
    teachers = []
    temps = []
    for path, temp in [(args.teacher1_dir, args.teacher1_temp), (args.teacher2_dir, args.teacher2_temp)]:
        if path and os.path.isdir(path):
            try:
                tm = AutoModelForSequenceClassification.from_pretrained(path, quantization_config=teacher_quant, device_map='auto')
                teachers.append(tm.eval())
                temps.append(float(temp))
                print(f"[OnlineKD] Loaded teacher: {path} (temp={temp})")
            except Exception as e:
                print(f"[OnlineKD] Warn: failed to load teacher at {path}: {e}")
        else:
            if path:
                print(f"[OnlineKD] Warn: teacher path missing: {path}")

    # Collator preserving dynamic padding
    pad_to_multiple = 8 if (args.fp16 or args.bf16) else None
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple)

    # Training args
    import inspect
    pin_mem = bool(torch.cuda.is_available())
    base_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=int(args.save_total_limit),
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        logging_steps=int(args.logging_steps),
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        warmup_ratio=args.warmup_ratio,
        dataloader_pin_memory=pin_mem,
        dataloader_num_workers=int(args.dataloader_num_workers),
        group_by_length=True,
        report_to='none',
        remove_unused_columns=False,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    if 'evaluation_strategy' not in allowed:
        base_kwargs.pop('evaluation_strategy', None)
        if 'do_eval' in allowed:
            base_kwargs['do_eval'] = True
    if 'save_strategy' not in allowed:
        base_kwargs.pop('save_strategy', None)
    if 'group_by_length' not in allowed:
        base_kwargs.pop('group_by_length', None)
    if 'report_to' not in allowed:
        base_kwargs.pop('report_to', None)
    if 'dataloader_pin_memory' not in allowed:
        base_kwargs.pop('dataloader_pin_memory', None)
    if 'dataloader_num_workers' not in allowed:
        base_kwargs.pop('dataloader_num_workers', None)
    if 'warmup_ratio' not in allowed:
        base_kwargs.pop('warmup_ratio', None)
    if args.max_steps is not None and 'max_steps' in allowed:
        base_kwargs['max_steps'] = int(args.max_steps)

    training_args = TrainingArguments(**{k:v for k,v in base_kwargs.items() if k in allowed})

    trainer = OnlineKDTrainer(
        model=student,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        teacher_models=teachers,
        teacher_temps=temps,
        kd_alpha=args.kd_alpha,
        T_soft=args.T_soft,
    )

    trainer.train()
    metrics = trainer.evaluate()
    try:
        import json
        with open(os.path.join(args.output_dir, 'cv_metrics_fold_single.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print({'eval': metrics})


if __name__ == '__main__':
    main()
