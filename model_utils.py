# model_utils.py (mock)
"""
Model utilities for creating and configuring models with LoRA
Supports Llama3-70B, Qwen2-72B, and Gemma2-9B (configuration only)
"""

from typing import List, Optional, Dict

class ModelFactory:
    @staticmethod
    def get_target_modules(model_name: str) -> List[str]:
        m = model_name.lower()
        if 'llama' in m or 'qwen' in m or 'gemma' in m:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    @classmethod
    def create_lora_config(cls, model_name: str, lora_r: int = 64, lora_alpha: int = 128, lora_dropout: float = 0.1, target_modules: Optional[List[str]] = None) -> Dict:
        if target_modules is None:
            target_modules = cls.get_target_modules(model_name)
        return {
            'r': lora_r,
            'lora_alpha': lora_alpha,
            'target_modules': target_modules,
            'lora_dropout': lora_dropout,
            'bias': 'none',
            'task_type': 'SEQ_CLS',
        }
