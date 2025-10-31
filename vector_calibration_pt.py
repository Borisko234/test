
import argparse
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.optimize import minimize
import os
import glob

# CONFIGURATION
num_classes = 3  # example: [A, B, tie]
folds = [0, 1, 2]

def get_out_dir():
    parser = argparse.ArgumentParser(description='Vector scaling calibration for .pt logits/labels.')
    parser.add_argument('--out-dir', default='calibration', help='Directory to write calibration artifacts')
    args, _ = parser.parse_known_args()
    return args.out_dir

out_dir = get_out_dir()

# Helper to load .pt and convert to np
def load_pt(filepath):
    tensor = torch.load(filepath, map_location='cpu')
    if isinstance(tensor, torch.Tensor):
        return tensor.numpy()
    if isinstance(tensor, dict) and 'logits' in tensor:
        return tensor['logits'].numpy()
    raise ValueError(f"Unknown format in {filepath}")

def load_labels(filepath):
    arr = torch.load(filepath, map_location='cpu')
    if isinstance(arr, torch.Tensor):
        return arr.numpy().astype(int)
    raise ValueError(f"Unknown label format in {filepath}")

# Calibration loss function
def vector_calibration_loss(params, logits, labels):
    a = params[:num_classes]
    b = params[num_classes:]
    logits_calib = logits * a + b
    log_probs = logits_calib - logsumexp(logits_calib, axis=1, keepdims=True)
    nll = -log_probs[np.arange(len(labels)), labels].mean()
    return nll

# Calibration and saving function for a set of logits/labels files
def calibrate_save(logits_files, labels_files, save_name):
    all_logits = np.concatenate([load_pt(f) for f in logits_files], axis=0)
    all_labels = np.concatenate([load_labels(f) for f in labels_files], axis=0)
    assert all_logits.shape[0] == all_labels.shape[0], f"Mismatch in shapes for {save_name}"

    init_params = np.concatenate([np.ones(num_classes), np.zeros(num_classes)])
    result = minimize(
        vector_calibration_loss,
        init_params,
        args=(all_logits, all_labels),
        method="L-BFGS-B"
    )
    opt_a = result.x[:num_classes]
    opt_b = result.x[num_classes:]

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{save_name}_vector_scaling_params.npz"), a=opt_a, b=opt_b)
    print(f"Calibrated {save_name}:")
    print("  Optimal scale factors (a):", opt_a)
    print("  Optimal biases (b):", opt_b)
    print("  Final NLL:", result.fun)
    print(f"Saved to {os.path.join(out_dir, f'{save_name}_vector_scaling_params.npz')}\n")

# 1. Calibrate each split (val and train for each fold)
for split in ['val', 'train']:
    for fold in folds:
        # Use logprobs if available, else probs
        logits_file = f"llama_fold_{fold}_{split}_logprobs.pt"
        if not os.path.exists(logits_file):
            logits_file = f"llama_fold_{fold}_{split}_probs.pt"
        labels_file = f"llama_fold_{fold}_{split}_labels.pt"  # update if your label naming is different
        if os.path.exists(logits_file) and os.path.exists(labels_file):
            calibrate_save([logits_file], [labels_file], f"fold{fold}_{split}")

# 2. Calibrate ensemble OOF outputs
ensemble_logits_file = "ensemble_oof_probs.pt"
ensemble_labels_file = "ensemble_oof_labels.pt"  # update if needed
if os.path.exists(ensemble_logits_file) and os.path.exists(ensemble_labels_file):
    calibrate_save([ensemble_logits_file], [ensemble_labels_file], "ensemble_oof")

# 3. Calibrate OOF outputs (parquet file not handled here)
oof_logits_file = "oof_probs.pt"
oof_labels_file = "oof_labels.pt"
if os.path.exists(oof_logits_file) and os.path.exists(oof_labels_file):
    calibrate_save([oof_logits_file], [oof_labels_file], "oof")

# Optional: Function for applying calibration
def apply_vector_calibration(logits, a, b):
    logits_calib = logits * a + b
    probs = np.exp(logits_calib - logsumexp(logits_calib, axis=1, keepdims=True))
    return probs

# Example usage:
# raw_logits = load_pt('some_logits.pt')
# calibrated_probs = apply_vector_calibration(raw_logits, opt_a, opt_b)