import argparse
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def quantize_gptq(base_model_dir: str, out_dir: str, bits: int = 8, group_size: int = 128, desc_act: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        base_model_dir,
        quantize_config,
        device_map='auto',
    )

    # Simple quantization without calibration dataset (PTQ). For better accuracy, provide a calibration dataloader.
    model.quantize()

    model.save_quantized(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Quantize a CausalLM model with AutoGPTQ')
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--bits', type=int, default=8)
    ap.add_argument('--group-size', type=int, default=128)
    ap.add_argument('--desc-act', action='store_true')
    args = ap.parse_args()
    quantize_gptq(args.model_dir, args.out_dir, bits=args.bits, group_size=args.group_size, desc_act=args.desc_act)
