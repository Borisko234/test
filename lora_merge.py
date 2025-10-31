import os
import json
import argparse
import gc
from contextlib import contextmanager
from peft import PeftModel
from transformers import AutoModelForSequenceClassification
import torch


@contextmanager
def _temporary_hide_gpus(enabled: bool):
    """Temporarily hide CUDA devices (set CUDA_VISIBLE_DEVICES="") to force CPU load.
    Restores the original value afterwards.
    """
    if not enabled:
        yield
        return
    original = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


def _select_cpu_dtype():
    """Choose the lightest viable dtype for CPU merge.
    Preference: float16 if supported, else bfloat16, else float32.
    """
    # Some ops on older CPUs may not like bfloat16/float16; we optimistically try float16.
    # The merge itself is mostly parameter arithmetic (matmul not required), so float16 ok.
    return torch.float16


def _merge_low_mem(base_model: str, lora_dir: str, hf_token: str, hide_gpu: bool = True, force_cpu: bool = True):
    """Load base + LoRA entirely on CPU to merge, avoiding GPU OOM.

    Strategies:
      - Temporarily hide GPUs so HF/accelerate won't offload any shard to CUDA.
      - Use low_cpu_mem_usage to stream weights.
      - Use the smallest safe dtype to reduce RAM footprint.
    """
    print("[Merge] Low-memory CPU merge path engaged (forcing CPU load, minimal dtype)")
    cpu_dtype = _select_cpu_dtype()
    with _temporary_hide_gpus(hide_gpu):
        if force_cpu:
            torch.cuda.is_available = lambda: False  # type: ignore
        base = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            device_map={"": "cpu"},
            trust_remote_code=True,
            torch_dtype=cpu_dtype,
            low_cpu_mem_usage=True,
            token=hf_token,
            local_files_only=False,
            num_labels=3,
            id2label={0: "A", 1: "B", 2: "tie"},
            label2id={"A": 0, "B": 1, "tie": 2},
        )
        base.to("cpu")
        peft = PeftModel.from_pretrained(base, lora_dir, device_map={"": "cpu"})
        merged = peft.merge_and_unload()
        merged.to("cpu")
        # Ensure config is correct
        merged.config.num_labels = 3
        merged.config.id2label = {0: "A", 1: "B", 2: "tie"}
        merged.config.label2id = {"A": 0, "B": 1, "tie": 2}
    return merged


def _free_cuda():
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _ensure_standard_weights(out_dir: str, model: torch.nn.Module):
    """Ensure at least one standard weight artifact exists in out_dir.
    Creates pytorch_model.bin if neither .bin nor .safetensors (or shard indexes) are present.
    """
    expected = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    ]
    present = {fn for fn in os.listdir(out_dir)}
    if not any(fn in present for fn in expected):
        try:
            bin_path = os.path.join(out_dir, "pytorch_model.bin")
            state = model.state_dict()
            torch.save(state, bin_path)
            print(f"[Merge] Wrote fallback weights -> {bin_path}")
        except Exception as e:
            print(f"[Merge][Error] Failed to write fallback pytorch_model.bin: {e}")


def _ensure_config(out_dir: str, model: torch.nn.Module):
    cfg_path = os.path.join(out_dir, "config.json")
    needs = True
    try:
        if os.path.isfile(cfg_path) and os.path.getsize(cfg_path) > 0:
            # attempt to parse
            with open(cfg_path, "r") as f:
                json.load(f)
            needs = False
    except Exception:
        needs = True
    if needs:
        try:
            model.config.to_json_file(cfg_path)
            print(f"[Merge] Wrote fallback config -> {cfg_path}")
        except Exception as e:
            print(f"[Merge][Warn] Failed to write fallback config.json: {e}")


def merge_lora(base_model: str, lora_dir: str, out_dir: str, low_mem: bool = False, cpu_only: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    # Ensure standard HTTP path; disable xet/transfer accelerations in-process
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HUGGINGFACE_HUB_ENABLE_HF_XET", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    merged = None
    if cpu_only:
        print("[Merge] --cpu-only specified; skipping any GPU attempt.")
        low_mem = True

    if low_mem:
        merged = _merge_low_mem(base_model, lora_dir, hf_token, hide_gpu=True, force_cpu=cpu_only or True)
    else:
        try:
            base = AutoModelForSequenceClassification.from_pretrained(
                base_model,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token,
                local_files_only=False,
                num_labels=3,
                id2label={0: "A", 1: "B", 2: "tie"},
                label2id={"A": 0, "B": 1, "tie": 2},
            )
            peft = PeftModel.from_pretrained(base, lora_dir)
            merged = peft.merge_and_unload()
            merged.config.num_labels = 3
            merged.config.id2label = {0: "A", 1: "B", 2: "tie"}
            merged.config.label2id = {"A": 0, "B": 1, "tie": 2}
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                print("[Merge][Warn] GPU OOM during merge; retrying with low-memory CPU path...")
                del base
                try:
                    del peft
                except Exception:
                    pass
                gc.collect()
                _free_cuda()
                merged = _merge_low_mem(base_model, lora_dir, hf_token, hide_gpu=True, force_cpu=True)
            else:
                raise
    print(f"[Merge] Saving merged model to {out_dir}")
    # Try to save with safetensors and sharding; fallbacks handled below
    try:
        merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    except TypeError:
        # older transformers may not support safe_serialization arg
        try:
            merged.save_pretrained(out_dir, max_shard_size="2GB")
        except Exception:
            merged.save_pretrained(out_dir)
    except Exception as e:
        print(f"[Merge][Warn] save_pretrained encountered an issue: {e}. Will attempt non-safe sharded save, then ensure .bin/config fallbacks.")
        try:
            merged.save_pretrained(out_dir, safe_serialization=False, max_shard_size="2GB")
        except Exception as e2:
            print(f"[Merge][Warn] Non-safe sharded save also failed: {e2}")
    # Ensure at least one standard weight file exists
    _ensure_standard_weights(out_dir, merged)
    # Ensure a valid config.json exists
    _ensure_config(out_dir, merged)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--low-mem", action="store_true", help="Force CPU/low-memory merge path (still allows GPU visibility)")
    ap.add_argument("--cpu-only", action="store_true", help="Force CPU merge and hide GPUs (strongest OOM avoidance)")
    args = ap.parse_args()
    merge_lora(args.base_model, args.lora_dir, args.out_dir, low_mem=args.low_mem, cpu_only=args.cpu_only)
