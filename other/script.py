# Let me create a comprehensive code implementation based on the first place solution
# I'll create the main components: data preprocessing, model training, distillation, and inference

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import json
from datasets import Dataset
from torch.utils.data import DataLoader
import argparse
import os

# Create the main training script structure
main_script = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import json
from datasets import Dataset
from torch.utils.data import DataLoader
import argparse
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

class LMSYSDataset:
    """Dataset class for LMSYS Chatbot Arena data"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = pd.read_csv(data_path)
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for training"""
        # Convert labels to numerical format
        label_mapping = {
            "model_a": 0,
            "model_b": 1, 
            "tie": 2,
            "tie (both bad)": 2
        }
        self.data['label_numeric'] = self.data['winner'].map(label_mapping)
        
        # Create input text combining prompt and responses
        self.data['input_text'] = (
            "[PROMPT]" + self.data['prompt'].astype(str) + 
            "[RESPONSE_A]" + self.data['response_a'].astype(str) + 
            "[RESPONSE_B]" + self.data['response_b'].astype(str)
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        # Tokenize input
        encoding = self.tokenizer(
            item['input_text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label_numeric'], dtype=torch.long)
        }

class DistillationTrainer:
    """Custom trainer for knowledge distillation"""
    
    def __init__(self, 
                 student_model,
                 teacher_model=None,
                 tokenizer=None,
                 temperature: float = 3.0,
                 alpha: float = 0.7,
                 device='cuda'):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        if self.teacher_model is not None:
            self.teacher_model.eval()
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Compute distillation loss combining soft targets and hard targets
        """
        # Soft target loss (KL divergence)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        kl_loss *= (self.temperature ** 2)
        
        # Hard target loss (cross-entropy)
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        return total_loss

class ModelFactory:
    """Factory class to create and configure models"""
    
    @staticmethod
    def create_lora_model(model_name: str, 
                          num_labels: int = 3,
                          lora_r: int = 64,
                          lora_alpha: int = 128,
                          target_modules: List[str] = None):
        """Create model with LoRA configuration"""
        
        # Default target modules for different model types
        if target_modules is None:
            if "llama" in model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                "gate_proj", "up_proj", "down_proj"]
            elif "qwen" in model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]
            elif "gemma" in model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # Load base model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model

class LMSYSTrainer:
    """Main trainer class implementing the first place solution"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizers for different models
        self.tokenizers = {}
        for model_name in config['models']:
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                config['model_paths'][model_name]
            )
            if self.tokenizers[model_name].pad_token is None:
                self.tokenizers[model_name].pad_token = self.tokenizers[model_name].eos_token
    
    def post_pretrain_phase(self, model_name: str, ut_data_path: str):
        """
        Phase 1: Post-pretrain on UT dataset
        """
        print(f"Starting post-pretrain phase for {model_name}")
        
        # Load model
        model = ModelFactory.create_lora_model(
            self.config['model_paths'][model_name],
            lora_r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha']
        )
        
        # Load UT dataset
        dataset = LMSYSDataset(
            ut_data_path, 
            self.tokenizers[model_name],
            max_length=self.config['max_length']
        )
        
        # Training arguments for post-pretrain
        training_args = TrainingArguments(
            output_dir=f"./model_save/post_pretrain_{model_name}",
            num_train_epochs=1,
            per_device_train_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=1e-5,
            logging_steps=100,
            save_strategy="epoch",
            fp16=True,
            dataloader_pin_memory=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizers[model_name],
        )
        
        # Train
        trainer.train()
        trainer.save_model()
        
        return model
    
    def get_teacher_logits(self, teacher_model, dataloader):
        """
        Get logits distribution from teacher model
        """
        teacher_model.eval()
        all_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = teacher_model(**batch)
                all_logits.append(outputs.logits.cpu())
        
        return torch.cat(all_logits, dim=0)
    
    def cross_validation_training(self, model_name: str, data_path: str, n_folds: int = 5):
        """
        Phase 2: 5-fold cross-validation training for large models
        """
        print(f"Starting 5-fold CV training for {model_name}")
        
        # Load data
        data = pd.read_csv(data_path)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_models = []
        fold_logits = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            print(f"Training fold {fold + 1}/{n_folds}")
            
            # Split data
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            
            # Save fold data
            train_data.to_csv(f"./data/fold_{fold}_train.csv", index=False)
            val_data.to_csv(f"./data/fold_{fold}_val.csv", index=False)
            
            # Create datasets
            train_dataset = LMSYSDataset(
                f"./data/fold_{fold}_train.csv",
                self.tokenizers[model_name],
                max_length=self.config['max_length']
            )
            
            # Load post-pretrained model
            model = ModelFactory.create_lora_model(
                self.config['model_paths'][model_name],
                lora_r=self.config['lora_r'],
                lora_alpha=self.config['lora_alpha']
            )
            
            # Load post-pretrain weights if available
            post_pretrain_path = f"./model_save/post_pretrain_{model_name}"
            if os.path.exists(post_pretrain_path):
                model = PeftModel.from_pretrained(model, post_pretrain_path)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./model_save/{model_name}_fold_{fold}",
                num_train_epochs=2,
                per_device_train_batch_size=self.config['batch_size'],
                gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
                learning_rate=5e-5,
                logging_steps=100,
                save_strategy="epoch",
                evaluation_strategy="epoch",
                fp16=True,
                dataloader_pin_memory=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizers[model_name],
            )
            
            # Train
            trainer.train()
            trainer.save_model()
            
            # Get logits for distillation
            val_dataset = LMSYSDataset(
                f"./data/fold_{fold}_val.csv",
                self.tokenizers[model_name],
                max_length=self.config['max_length']
            )
            val_dataloader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
            
            logits = self.get_teacher_logits(model, val_dataloader)
            fold_logits.append(logits)
            fold_models.append(model)
        
        return fold_models, fold_logits
    
    def distillation_training(self, 
                              student_model_name: str,
                              teacher_logits: List[torch.Tensor],
                              data_paths: List[str]):
        """
        Phase 3: Distillation training for student model
        """
        print(f"Starting distillation training for {student_model_name}")
        
        student_models = []
        
        for fold in range(len(teacher_logits)):
            print(f"Distilling to student model fold {fold + 1}")
            
            # Create student model
            student_model = ModelFactory.create_lora_model(
                self.config['model_paths'][student_model_name],
                lora_r=self.config['lora_r'],
                lora_alpha=self.config['lora_alpha']
            )
            
            # Load post-pretrain weights if available
            post_pretrain_path = f"./model_save/post_pretrain_{student_model_name}"
            if os.path.exists(post_pretrain_path):
                student_model = PeftModel.from_pretrained(student_model, post_pretrain_path)
            
            # Load fold data
            train_dataset = LMSYSDataset(
                data_paths[fold],
                self.tokenizers[student_model_name],
                max_length=self.config['max_length']
            )
            
            # Custom training loop with distillation
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
            student_model.train()
            
            dataloader = DataLoader(train_dataset, 
                                  batch_size=self.config['batch_size'],
                                  shuffle=True)
            
            distill_trainer = DistillationTrainer(
                student_model=student_model,
                tokenizer=self.tokenizers[student_model_name]
            )
            
            # Training loop
            for epoch in range(2):
                total_loss = 0
                for i, batch in enumerate(dataloader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Get student outputs
                    student_outputs = student_model(**batch)
                    student_logits = student_outputs.logits
                    
                    # Get corresponding teacher logits
                    batch_teacher_logits = teacher_logits[fold][i*self.config['batch_size']:
                                                             (i+1)*self.config['batch_size']].to(self.device)
                    
                    # Compute distillation loss
                    loss = distill_trainer.distillation_loss(
                        student_logits, batch_teacher_logits, batch['labels']
                    )
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    if i % 100 == 0:
                        print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
                
                print(f"Epoch {epoch}, Average Loss: {total_loss/len(dataloader):.4f}")
            
            # Save student model
            student_model.save_pretrained(f"./model_save/student_{student_model_name}_fold_{fold}")
            student_models.append(student_model)
        
        return student_models
    
    def merge_lora_weights(self, model_paths: List[str], output_path: str):
        """
        Phase 4: Merge LoRA weights from different folds
        """
        print("Merging LoRA weights from different folds")
        
        # Load first model as base
        base_model = PeftModel.from_pretrained(
            AutoModelForSequenceClassification.from_pretrained(
                self.config['model_paths']['gemma2-9b'],
                num_labels=3,
                torch_dtype=torch.float16
            ),
            model_paths[0]
        )
        
        # Average LoRA weights
        base_lora_state = base_model.peft_config['default']
        avg_weights = {}
        
        # Collect all LoRA weights
        all_weights = []
        for path in model_paths:
            model = PeftModel.from_pretrained(
                AutoModelForSequenceClassification.from_pretrained(
                    self.config['model_paths']['gemma2-9b'],
                    num_labels=3,
                    torch_dtype=torch.float16
                ),
                path
            )
            all_weights.append(model.state_dict())
        
        # Average the weights
        for key in all_weights[0]:
            if 'lora' in key:
                avg_weights[key] = torch.mean(
                    torch.stack([w[key] for w in all_weights]), dim=0
                )
            else:
                avg_weights[key] = all_weights[0][key]
        
        # Load averaged weights
        base_model.load_state_dict(avg_weights)
        base_model.save_pretrained(output_path)
        
        return base_model

def main():
    # Configuration
    config = {
        'models': ['llama3-70b', 'qwen2-72b', 'gemma2-9b'],
        'model_paths': {
            'llama3-70b': './model_path/llama3_70b',
            'qwen2-72b': './model_path/qwen2_72b', 
            'gemma2-9b': './model_path/Gemma2_9b'
        },
        'lora_r': 64,
        'lora_alpha': 128,
        'max_length': 1024,
        'batch_size': 8,
        'gradient_accumulation_steps': 8,
        'data_paths': {
            'kaggle_train': './data/lmsys-chatbot-arena/train.csv',
            'ut_data': './data/ut_data.csv',
            '33k_data': './data/33k_data.csv'
        }
    }
    
    trainer = LMSYSTrainer(config)
    
    # Phase 1: Post-pretrain on UT data
    print("=== Phase 1: Post-pretrain ===")
    for model_name in ['llama3-70b', 'qwen2-72b', 'gemma2-9b']:
        trainer.post_pretrain_phase(model_name, config['data_paths']['ut_data'])
    
    # Phase 2: Cross-validation training for large models
    print("=== Phase 2: Cross-validation training ===")
    teacher_models = {}
    teacher_logits = {}
    
    for model_name in ['llama3-70b', 'qwen2-72b']:
        models, logits = trainer.cross_validation_training(
            model_name, 
            config['data_paths']['kaggle_train']
        )
        teacher_models[model_name] = models
        teacher_logits[model_name] = logits
    
    # Phase 3: Distillation to Gemma2-9B
    print("=== Phase 3: Distillation ===")
    # Combine logits from both teacher models
    combined_logits = []
    for i in range(5):  # 5 folds
        combined = (teacher_logits['llama3-70b'][i] + teacher_logits['qwen2-72b'][i]) / 2
        combined_logits.append(combined)
    
    fold_data_paths = [f"./data/fold_{i}_train.csv" for i in range(5)]
    student_models = trainer.distillation_training(
        'gemma2-9b', 
        combined_logits,
        fold_data_paths
    )
    
    # Phase 4: Merge LoRA weights
    print("=== Phase 4: Merge LoRA weights ===")
    student_model_paths = [f"./model_save/student_gemma2-9b_fold_{i}" for i in range(5)]
    final_model = trainer.merge_lora_weights(
        student_model_paths,
        "./model_save/final_merged_model"
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
'''

print("Created comprehensive training script based on first place solution")
print("Key components implemented:")
print("1. Data preprocessing and tokenization")
print("2. LoRA configuration for all target models")
print("3. Post-pretraining phase")
print("4. 5-fold cross-validation training")
print("5. Knowledge distillation from teacher to student")
print("6. LoRA weight averaging")
print("7. Support for Llama3-70B, Qwen2-72B, and Gemma2-9B")