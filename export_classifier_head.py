import argparse
import json
import os
import torch
from transformers import AutoModelForSequenceClassification


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


def _resolve_base_dir(model_dir: str) -> str:
    """If model_dir is a lightweight quantized folder, redirect to the real base model dir."""
    # Preferred pointers written by our quantizer
    for fname in ('target_model_dir.txt', 'base_model_dir.txt'):
        ptr = os.path.join(model_dir, fname)
        if os.path.exists(ptr):
            try:
                with open(ptr, 'r') as f:
                    base = f.read().strip()
                if base:
                    return base
            except Exception:
                pass
    # If no weights in current folder and BASE_MODEL env is set, use it
    if not _has_weight_files(model_dir):
        env_base = os.environ.get('BASE_MODEL', '').strip()
        if env_base:
            return env_base
    return model_dir


def export_head(model_dir: str, out_path: str = None):
    # Try to read num_labels from config; default to 3
    cfg_path = os.path.join(model_dir, 'config.json')
    num_labels = 3
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            num_labels = int(cfg.get('num_labels', 3))
        except Exception:
            pass

    # Ensure model is instantiated with correct head size to avoid PEFT modules_to_save mismatch
    load_dir = _resolve_base_dir(model_dir)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            load_dir,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
    except RuntimeError as e:
        # Try to infer num_labels from adapter_model.safetensors if present
        try:
            from safetensors.torch import load_file as sf_load
            adapter_path = os.path.join(model_dir, 'adapter_model.safetensors')
            if os.path.exists(adapter_path):
                state = sf_load(adapter_path, device='cpu')
                # search for common classifier keys
                keys = [k for k in state.keys() if k.endswith('score.modules_to_save.default.weight') or k.endswith('classifier.modules_to_save.default.weight')]
                if keys:
                    w = state[keys[0]]
                    num_labels = int(w.shape[0])
        except Exception:
            pass
        # Retry with updated num_labels if changed, else re-raise
        model = AutoModelForSequenceClassification.from_pretrained(
            load_dir,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
        )
    # Try common attribute names
    head_module = None
    for name in ['classifier', 'score', 'classification_head', 'lm_head']:
        if hasattr(model, name):
            head_module = getattr(model, name)
            break
    if head_module is None:
        # Try the last module as a heuristic
        head_module = list(model.children())[-1]
    # Normalize state_dict keys to plain weight/bias if possible
    state = head_module.state_dict()
    norm = {}
    if 'weight' in state:
        norm['weight'] = state['weight']
    elif 'modules_to_save.default.weight' in state:
        norm['weight'] = state['modules_to_save.default.weight']
    elif 'original_module.weight' in state:
        norm['weight'] = state['original_module.weight']
    if 'bias' in state:
        norm['bias'] = state['bias']
    elif 'modules_to_save.default.bias' in state:
        norm['bias'] = state['modules_to_save.default.bias']
    elif 'original_module.bias' in state:
        norm['bias'] = state['original_module.bias']
    # If bias missing entirely, create a zero bias for compatibility
    if 'weight' in norm and 'bias' not in norm:
        import torch as _torch
        norm['bias'] = _torch.zeros(norm['weight'].shape[0])
    state = norm or state
    out = out_path or os.path.join(model_dir, 'classifier_head.pt')
    torch.save(state, out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Export classifier head weights from a seq-classification model directory')
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    path = export_head(args.model_dir, args.out)
    print({'saved': path})
