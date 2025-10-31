import os
import json
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


def train_lora(
    base_model: str,
    output_dir: str,
    data_path: str,
    tokenizer_path: Optional[str] = None,
    num_epochs: float = 1.0,
    lr: float = 1e-5,
    max_length: int = 1024,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    qlora: bool = False,
    bf16: bool = False,
    attn_impl: str = None,
    grad_checkpoint: bool = False,
    load_8bit: bool = False,
    grad_accum: int = 64,
    subset_size: int = None,
    per_device_batch: int = 1,
    max_steps: int = -1,
):
    os.makedirs(output_dir, exist_ok=True)
    # Ensure standard HTTP path; disable xet/transfer accelerations in-process just in case
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    tok_src = tokenizer_path or base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True, local_files_only=False, trust_remote_code=True)
    except Exception as e_fast:
        # Fallback: try slow tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=False, local_files_only=False, trust_remote_code=True)
        except Exception as e_slow:
            raise RuntimeError(
                f"Failed to load tokenizer from '{tok_src}'. If your base model dir lacks tokenizer files, "
                f"pass --tokenizer-path pointing to a compatible model/repo with tokenizer files (e.g., meta-llama/Meta-Llama-3-70B)."
            ) from e_slow
    if tokenizer.pad_token is None:
        # Ensure a valid pad token; for decoder-only LLMs we usually reuse eos as pad
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding is generally safer for decoder-only models when doing classification
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass

    bnb_config = None
    if qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )

    # Allow passing a HF token via env if needed; step1 passes token at login level already
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    from transformers import AutoModelForSequenceClassification
    from functools import partial
    model_kwargs = dict(
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        token=hf_token,
        trust_remote_code=True,
        local_files_only=False,
        num_labels=3,
        id2label={0: "A", 1: "B", 2: "tie"},
        label2id={"A": 0, "B": 1, "tie": 2},
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    if load_8bit and not qlora:
        # Only use 8-bit when not already doing 4bit QLoRA
        model_kwargs["load_in_8bit"] = True
    model = AutoModelForSequenceClassification.from_pretrained(base_model, **model_kwargs)
    # Align pad/eos/bos ids between tokenizer and model to avoid batch>1 errors
    try:
        if getattr(model.config, 'pad_token_id', None) is None or model.config.pad_token_id == -1:
            model.config.pad_token_id = tokenizer.pad_token_id
        # Keep eos/bos consistent if available
        if getattr(model.config, 'eos_token_id', None) is None:
            model.config.eos_token_id = tokenizer.eos_token_id
        if getattr(model.config, 'bos_token_id', None) is None and hasattr(tokenizer, 'bos_token_id'):
            model.config.bos_token_id = tokenizer.bos_token_id
        # Generation config if present
        if hasattr(model, 'generation_config'):
            if getattr(model.generation_config, 'pad_token_id', None) is None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
            if getattr(model.generation_config, 'eos_token_id', None) is None:
                model.generation_config.eos_token_id = tokenizer.eos_token_id
    except Exception as _align_e:
        print(f"[WARN] Could not fully align special token ids: {_align_e}")

    # Prepare model for k-bit training in both 4-bit (QLoRA) and 8-bit LoRA scenarios
    if qlora or load_8bit:
        model = prepare_model_for_kbit_training(model)
    if grad_checkpoint:
        model.gradient_checkpointing_enable()
        if hasattr(model, 'config'):
            model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_cfg)

    # Debug: report trainable vs total parameters to ensure LoRA adapters are active
    try:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = (trainable / total * 100) if total else 0.0
        print(f"[DEBUG] Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
        if trainable == 0:
            raise RuntimeError(
                "No trainable parameters detected after applying LoRA. "
                "If using --load-8bit without --qlora, ensure prepare_model_for_kbit_training is applied."
            )
    except Exception as debug_e:
        print(f"[WARN] Failed to compute trainable parameter stats: {debug_e}")

    # Robust CSV -> Dataset loading. Arrow inference sometimes fails if a column mixes numeric and string tokens
    # (e.g., a mostly numeric column with a stray string like 'evol_instruct'). We attempt standard load first;
    # on failure we fall back to pandas and coerce mixed columns to string to avoid DatasetGenerationError.
    from datasets import Dataset, DatasetDict
    ds = None
    try:
        ds = load_dataset("csv", data_files={"train": data_path})
    except Exception as e_csv:
        print(f"[WARN] load_dataset csv auto-infer failed: {type(e_csv).__name__}: {e_csv}\n[WARN] Falling back to pandas coercion mode.")
        import pandas as pd
        # Read with low_memory=False so pandas infers types in one pass; then coerce object-like / mixed to string.
        pdf = pd.read_csv(data_path, low_memory=False)
        # Force every column to pure string to completely sidestep Arrow casting issues (safe for text-only training).
        for col in pdf.columns:
            pdf[col] = pdf[col].astype(str)
        # Replace any 'nan' textual artifacts after cast.
        pdf = pdf.replace({'nan': ''}).fillna("")
        ds = DatasetDict({"train": Dataset.from_pandas(pdf, preserve_index=False)})
        print(f"[INFO] Fallback dataset constructed: {len(ds['train'])} rows | columns: {list(pdf.columns)[:15]}{'...' if len(pdf.columns)>15 else ''}")
        # Optionally persist a debug sample
        try:
            pdf.head(50).to_csv(os.path.join(output_dir, 'debug_dataset_sample.csv'), index=False)
        except Exception as _e_dbg:
            print(f"[WARN] Could not write debug sample: {_e_dbg}")
    if subset_size is not None:
        if subset_size > 0:
            # Take a deterministic head subset for speed
            orig_len = len(ds["train"])
            capped = min(subset_size, orig_len)
            ds["train"] = ds["train"].select(range(capped))
            print(f"[INFO] Subsetting training data: {capped} / {orig_len} examples (requested {subset_size})")
        else:
            print(f"[INFO] subset_size={subset_size} -> ignoring and using full dataset of {len(ds['train'])} examples")

    # For ultrafeedback_mini.csv: for each row, create two examples (chosen=label 0, rejected=label 1)
    """
    def build_examples_from_row(row):
        prompt = row.get("prompt") or row.get("question") or ""
        chosen = row.get("chosen") or ""
        rejected = row.get("rejected") or ""
        examples = []
        # Detect tie: if chosen == rejected (after stripping whitespace), label=2
        if chosen.strip() and rejected.strip() and chosen.strip() == rejected.strip():
            text = (str(prompt).strip() + "\n" if prompt else "") + str(chosen).strip()
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            encoded["labels"] = 2  # tie
            examples.append(encoded)
        else:
            if chosen:
                text = (str(prompt).strip() + "\n" if prompt else "") + str(chosen).strip()
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
                encoded["labels"] = 0  # chosen is A (label 0)
                examples.append(encoded)
            if rejected:
                text = (str(prompt).strip() + "\n" if prompt else "") + str(rejected).strip()
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
                encoded["labels"] = 1  # rejected is B (label 1)
                examples.append(encoded)
        return examples
    """

    import ast
    import unicodedata

    def _coerce_text(x):
        """
        Ensure x is a clean Python str without invalid surrogate code points.
        - If bytes, decode utf-8 with errors='ignore'.
        - If not str, cast to str.
        - Strip surrounding whitespace.
        - Remove any unencodable characters by round-tripping through utf-8.
        """
        if isinstance(x, bytes):
            try:
                x = x.decode('utf-8', errors='ignore')
            except Exception:
                x = str(x)
        elif not isinstance(x, str):
            x = str(x)
        # Normalize to NFC to avoid odd surrogate combos
        try:
            x = unicodedata.normalize('NFC', x)
        except Exception:
            pass
        # Drop invalid surrogates
        try:
            x = x.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except Exception:
            pass
        return x.strip()

    def build_examples_from_row(row):
        prompts = row.get("prompt")
        responses_a = row.get("response_a")
        responses_b = row.get("response_b")

        # Convert stringified lists to real lists
        for name, val in [("prompt", prompts), ("response_a", responses_a), ("response_b", responses_b)]:
            if isinstance(val, str):
                try:
                    val_parsed = ast.literal_eval(val)
                    if isinstance(val_parsed, list):
                        if name == "prompt": prompts = val_parsed
                        elif name == "response_a": responses_a = val_parsed
                        else: responses_b = val_parsed
                    else:
                        if name == "prompt": prompts = [str(val)]
                        elif name == "response_a": responses_a = [str(val)]
                        else: responses_b = [str(val)]
                except Exception:
                    # just wrap in list if not parseable
                    if name == "prompt": prompts = [str(val)]
                    elif name == "response_a": responses_a = [str(val)]
                    else: responses_b = [str(val)]
        
        # Ensure lists
        if not isinstance(prompts, list): prompts = [str(prompts)]
        if not isinstance(responses_a, list): responses_a = [str(responses_a)]
        if not isinstance(responses_b, list): responses_b = [str(responses_b)]

        winner_a = row.get("winner_model_a", 0)
        winner_b = row.get("winner_model_b", 0)
        winner_tie = row.get("winner_tie", 0)

        # Fallback: Kaggle schemas
        if not (winner_a or winner_b or winner_tie):
            w = row.get("winner")
            # winner can be int label (0/1/2) or string 'A'/'B'/'tie'
            if w is not None and w != "":
                try:
                    wi = int(float(w))
                    if wi in (0,1,2):
                        winner_a = 1 if wi == 0 else 0
                        winner_b = 1 if wi == 1 else 0
                        winner_tie = 1 if wi == 2 else 0
                except Exception:
                    s = str(w).strip().lower()
                    if s in ("a","model_a"): winner_a = 1
                    elif s in ("b","model_b"): winner_b = 1
                    elif s.startswith("tie"): winner_tie = 1
            # One-hot ints
            if not (winner_a or winner_b or winner_tie) and all(k in row for k in ("winner_model_a","winner_model_b","winner_tie")):
                try:
                    a_i = int(float(row.get("winner_model_a") or 0))
                    b_i = int(float(row.get("winner_model_b") or 0))
                    t_i = int(float(row.get("winner_tie") or 0))
                    # pick argmax
                    mx = max([(a_i, 'a'), (b_i, 'b'), (t_i, 't')])
                    winner_a = 1 if mx[1]=='a' else 0
                    winner_b = 1 if mx[1]=='b' else 0
                    winner_tie = 1 if mx[1]=='t' else 0
                except Exception:
                    pass
            # Probability columns
            if not (winner_a or winner_b or winner_tie) and all(k in row for k in ("winner_model_a_prob","winner_model_b_prob","winner_tie_prob")):
                try:
                    a_p = float(row.get("winner_model_a_prob") or 0.0)
                    b_p = float(row.get("winner_model_b_prob") or 0.0)
                    t_p = float(row.get("winner_tie_prob") or 0.0)
                    if a_p >= b_p and a_p >= t_p: winner_a = 1
                    elif b_p >= a_p and b_p >= t_p: winner_b = 1
                    else: winner_tie = 1
                except Exception:
                    pass

        examples = []

        for i, prompt in enumerate(prompts):
            if i >= len(responses_a) or i >= len(responses_b):
                break

            # Pick chosen/rejected based on winner columns
            if winner_tie:
                chosen, rejected, label = responses_a[i], responses_b[i], 2
            elif winner_a:
                chosen, rejected, label = responses_a[i], responses_b[i], 0
            elif winner_b:
                chosen, rejected, label = responses_b[i], responses_a[i], 1
            else:
                continue

            # Guarantee clean strings
            prompt_s = _coerce_text(prompt)
            chosen_s = _coerce_text(chosen)

            # For 33k-style rows we use prompt + chosen; for Kaggle rows it still applies
            text = f"{prompt_s}\n{chosen_s}" if prompt_s else chosen_s

            try:
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )
            except Exception as e:
                # Avoid UnicodeEncodeError when printing problematic text
                try:
                    sample = repr(text)
                except Exception:
                    sample = f"<unprintable type {type(text)}>"
                print("❌ Tokenization error:", e)
                print("Offending text repr (first 500):", sample[:500])
                continue

            encoded["labels"] = label
            examples.append(encoded)

        return examples


    # Flatten all examples from all rows
    all_examples = []
    for ex in ds["train"]:
        all_examples.extend(build_examples_from_row(ex))

    # Convert to HuggingFace Dataset
    from datasets import Dataset
    tokenized = Dataset.from_list(all_examples)

    # If max_steps provided (>0), we let Trainer cap steps and ignore epochs after reaching it.
    # Trainer expects an int for max_steps; use -1 (default behavior) instead of None
    eff_max_steps = max_steps if (isinstance(max_steps, int) and max_steps > 0) else -1
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        max_steps=eff_max_steps,
        learning_rate=lr,
        fp16=not bf16,
        bf16=bf16,
        gradient_checkpointing=grad_checkpoint,
        logging_steps=20,
        save_strategy="epoch" if eff_max_steps == -1 else "steps",
        save_steps=max(100, (eff_max_steps // 5) if eff_max_steps > 0 else 100),
        report_to="none",
    )

    # Timing callback to capture per-step durations
    class StepTimingCallback(TrainerCallback):
        def __init__(self):
            self.step_times = []
            self._last = None
            self._train_start = None
        def on_train_begin(self, args, state, control, **kwargs):
            import time
            self._train_start = time.time()
            self._last = self._train_start
        def on_step_end(self, args, state, control, **kwargs):
            import time
            now = time.time()
            if self._last is not None:
                self.step_times.append(now - self._last)
            self._last = now
        def summary(self):
            import math, time
            if not self.step_times:
                return {}
            return {
                "steps_recorded": len(self.step_times),
                "avg_step_sec": sum(self.step_times)/len(self.step_times),
                "p50_step_sec": sorted(self.step_times)[len(self.step_times)//2],
                "p90_step_sec": sorted(self.step_times)[int(len(self.step_times)*0.9)-1 if len(self.step_times)>1 else 0],
            }

    timing_cb = StepTimingCallback()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if (bf16 or not bf16) else None)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[timing_cb]
    )
    import time, json
    t0 = time.time()
    trainer.train()
    total_time = time.time() - t0
    # Save artifacts
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Export logs & summary
    try:
        log_hist = trainer.state.log_history
        with open(os.path.join(output_dir,'training_log.json'),'w') as f:
            json.dump(log_hist, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not write training_log.json: {e}")
    # Parameter stats captured earlier
    summary = {
        "total_time_sec": total_time,
        "global_steps": getattr(trainer.state,'global_step', None),
        "max_steps_arg": max_steps,
        "effective_max_steps": args.max_steps,
        "trainable_params": trainable if 'trainable' in locals() else None,
        "total_params": total if 'total' in locals() else None,
        "trainable_pct": pct if 'pct' in locals() else None,
        "timing": timing_cb.summary(),
    }
    try:
        with open(os.path.join(output_dir,'training_summary.json'),'w') as f:
            json.dump(summary, f, indent=2)
        print('[INFO] Wrote training_summary.json')
    except Exception as e:
        print(f"[WARN] Could not write training_summary.json: {e}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--epochs", type=float, default=1.0, help="Number of epochs (can be fractional, ignored early if --max-steps hit)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--attn-impl", type=str, default=None, help="Attention implementation override (e.g. eager, sdpa, flash_attention_2)")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit (ignored if --qlora)")
    p.add_argument("--grad-accum", type=int, default=64, help="Gradient accumulation steps (global batch = per_device * grad_accum)")
    p.add_argument("--subset-size", type=int, default=None, help="Optional cap on number of training examples")
    p.add_argument("--per-device-batch", type=int, default=1, help="Per-device (GPU) batch size")
    p.add_argument("--max-steps", type=int, default=-1, help="Stop after this many optimizer steps (overrides epochs if >0)")
    a = p.parse_args()

    train_lora(
        base_model=a.base_model,
        output_dir=a.output_dir,
        data_path=a.data_path,
        tokenizer_path=a.tokenizer_path,
        num_epochs=a.epochs,
        lr=a.lr,
        max_length=a.max_length,
        r=a.r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        qlora=a.qlora,
        bf16=a.bf16,
        attn_impl=a.attn_impl,
        grad_checkpoint=a.grad_checkpoint,
        load_8bit=a.load_8bit,
        grad_accum=a.grad_accum,
        subset_size=a.subset_size,
        per_device_batch=a.per_device_batch,
        max_steps=a.max_steps,
    )
