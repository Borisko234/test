# data_preprocessing.py
"""
Data preprocessing utilities for LMSYS competition (mock)
Handles tokenization, dataset creation, and data splitting
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple

class LMSYSDataProcessor:
    """Data processor for LMSYS Chatbot Arena dataset"""
    def __init__(self):
        self.label_mapping = {"model_a": 0, "model_b": 1, "tie": 2, "tie (both bad)": 2}
    
    def load_and_clean_data(self, data_path: str) -> pd.DataFrame:
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        # Basic cleaning
        cols = [c for c in ['prompt', 'response_a', 'response_b', 'winner'] if c in data.columns]
        data = data.dropna(subset=cols)
        if 'winner' in data.columns:
            data['label'] = data['winner'].map(self.label_mapping)
        else:
            data['label'] = np.nan
        return data
    
    def create_input_text(self, row: pd.Series) -> str:
        prompt = str(row['prompt']).strip()
        response_a = str(row['response_a']).strip()
        response_b = str(row['response_b']).strip()
        return f"[PROMPT]{prompt}[RESPONSE_A]{response_a}[RESPONSE_B]{response_b}"
    
    def prepare_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['input_text'] = data.apply(self.create_input_text, axis=1)
        return data
    
    def create_folds(self, data: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = []
        for train_idx, val_idx in kf.split(data):
            train_fold = data.iloc[train_idx].reset_index(drop=True)
            val_fold = data.iloc[val_idx].reset_index(drop=True)
            folds.append((train_fold, val_fold))
        return folds
    
    def save_fold_data(self, folds: List[Tuple[pd.DataFrame, pd.DataFrame]], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for fold_idx, (train_data, val_data) in enumerate(folds):
            train_path = f"{output_dir}/fold_{fold_idx}_train.csv"
            val_path = f"{output_dir}/fold_{fold_idx}_val.csv"
            train_data.to_csv(train_path, index=False)
            val_data.to_csv(val_path, index=False)
            print(f"Saved fold {fold_idx}: {len(train_data)} train, {len(val_data)} val samples")

if __name__ == "__main__":
    processor = LMSYSDataProcessor()
    # Try both possible paths
    candidates = [
        './data/lmsys-chatbot-arena/train.csv',
        './data/train.csv'
    ]
    train_csv = next((p for p in candidates if os.path.exists(p)), None)
    if train_csv:
        train_data = processor.load_and_clean_data(train_csv)
        train_data = processor.prepare_dataset(train_data)
        folds = processor.create_folds(train_data)
        processor.save_fold_data(folds, './data/processed_data')
    else:
        print("Training CSV not found at ./data/lmsys-chatbot-arena/train.csv or ./data/train.csv")
