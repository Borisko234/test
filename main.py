# main.py
import os
import argparse
import logging
from typing import Dict, Any

from training import TrainingOrchestrator
from inference import InferencePipeline
from student_train_hf import train_student, dataset_stats
from student_infer_hf import infer_student
from student_calibrate import calibrate_student
from student_train_distill_hf import train_student_distill
from teacher_logits_validator import validate_logits
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_directories():
    for d in [
        './model_path', './model_save', './model_save_or', './data', './data/processed_data', './data/oof', './sub'
    ]:
        os.makedirs(d, exist_ok=True)
    # Suppress Windows symlink warning from huggingface_hub if desired
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')


def load_config() -> Dict[str, Any]:
    return {
        'models': {
            'llama3-70b': {
                'model_path': './model_path/llama3_70b',
                'max_length': 1024, 'lora_r': 64, 'lora_alpha': 128
            },
            'qwen2-72b': {
                'model_path': './model_path/qwen2_72b',
                'max_length': 1024, 'lora_r': 64, 'lora_alpha': 128
            },
            'gemma2-9b': {
                'model_path': './model_path/Gemma2_9b',
                'max_length': 1024, 'lora_r': 64, 'lora_alpha': 128
            }
        },
        'batch_size': 8,
        'gradient_accumulation_steps': 8,
        'max_length': 1024,
        'lora_r': 64,
        'lora_alpha': 128,
        'temperature': 3.0,
        'alpha': 0.7,
        'use_tta': True,
        'tta_lengths': [1024, 2000],
        'quantization_bits': 8,
        'data_paths': {
            'kaggle_train': './data/lmsys-chatbot-arena/train.csv',
            'ut_data': './data/ut_data.csv',
            '33k_data': './data/33k_data.csv',
            'test_data': './data/test.csv'
        }
    }


def main():
    parser = argparse.ArgumentParser(description='LMSYS Mock Pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'full', 'student-train', 'student-infer', 'student-calibrate', 'student-eval-holdout', 'student-distill-train'], default='full')
    parser.add_argument('--output-dir', type=str, default='./sub')
    parser.add_argument('--student-output-model-dir', type=str, default='./model_save/student_distilbert')
    parser.add_argument('--student-submission-path', type=str, default='./sub/student_submission.csv')
    parser.add_argument('--student-infer-tta-lengths', type=str, default='')  # e.g., '512,2000'
    parser.add_argument('--student-infer-8bit', action='store_true')
    # Student training knobs
    parser.add_argument('--student-epochs', type=int, default=1)
    parser.add_argument('--student-max-samples', type=int, default=2000)
    parser.add_argument('--student-model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--student-label-smoothing', type=float, default=0.05)
    parser.add_argument('--student-early-stopping', type=int, default=2)
    parser.add_argument('--student-max-length', type=int, default=512)
    parser.add_argument('--student-extra-csvs', type=str, default='')  # comma-separated additional train CSVs
    parser.add_argument('--student-shuffle-ab', action='store_true')
    parser.add_argument('--student-dedup-by-prompt', action='store_true')
    # Distillation knobs
    parser.add_argument('--distill-teachers', type=str, default='')  # comma-separated paths or globs to .npy logits
    parser.add_argument('--distill-alpha', type=float, default=0.7)
    parser.add_argument('--distill-temp', type=float, default=3.0)
    parser.add_argument('--distill-mse-weight', type=float, default=0.0)
    parser.add_argument('--distill-temp-schedule', type=str, default='')  # e.g., 'linear:5,2' over epochs
    parser.add_argument('--cv-num-folds', type=int, default=5)
    parser.add_argument('--cv-fold-idx', type=int, default=0)
    # Throughput knobs (optional)
    parser.add_argument('--student-train-batch-size', type=int, default=8)
    parser.add_argument('--student-eval-batch-size', type=int, default=8)
    parser.add_argument('--student-grad-accum', type=int, default=1)
    parser.add_argument('--student-learning-rate', type=float, default=5e-5)
    parser.add_argument('--student-warmup-ratio', type=float, default=0.06)
    parser.add_argument('--student-fp16', action='store_true')
    parser.add_argument('--student-bf16', action='store_true')
    parser.add_argument('--student-gradient-checkpointing', action='store_true')
    parser.add_argument('--student-stats-only', action='store_true')
    parser.add_argument('--student-use-fast-tokenizer', action='store_true')
    parser.add_argument('--student-num-workers', type=int, default=0)
    parser.set_defaults(student_use_fast_tokenizer=True)
    args = parser.parse_args()

    setup_directories()
    config = load_config()

    if args.mode in ['train', 'full']:
        logger.info('Starting training pipeline (mock)')
        orchestrator = TrainingOrchestrator(config)
        orchestrator.run_full_pipeline()
        orchestrator.save_training_summary()

    if args.mode in ['inference', 'full']:
        logger.info('Starting inference pipeline (mock)')
        pipeline = InferencePipeline({
            'max_length': 1024,
            'tta_lengths': [1024, 2000],
            'quantization_bits': 8,
            'model_name': 'gemma2-9b',
            'use_tta': True
        })
        results = pipeline.run_inference(
            test_data_path=config['data_paths']['test_data'],
            merged_model_path='./model_save/final_merged_model',
            output_dir=args.output_dir
        )
        logger.info('Inference completed; submission saved to ./sub/final_submission.csv')

    if args.mode == 'student-train':
        logger.info('Starting real small student training (HF/Transformers)')
        train_csv_candidates = [
            './data/train.csv',
            './data/lmsys-chatbot-arena/train.csv'
        ]
        train_csv = next((p for p in train_csv_candidates if os.path.exists(p)), None)
        if not train_csv:
            raise FileNotFoundError('train.csv not found in ./data or ./data/lmsys-chatbot-arena')
        extra_list = [s.strip() for s in args.student_extra_csvs.split(',') if s.strip()]

        if args.student_stats_only:
            stats = dataset_stats(
                train_csv=train_csv,
                max_samples=(args.student_max_samples if args.student_max_samples > 0 else None),
                extra_csvs=extra_list,
                shuffle_ab=args.student_shuffle_ab,
                dedup_by_prompt=args.student_dedup_by_prompt,
            )
            logger.info(f"Dataset stats -> merged={stats['merged']}, deduped={stats['deduped']}, used={stats['used']} | labels: a={stats['label_0']}, b={stats['label_1']}, tie={stats['label_2']}")
            return
        metrics = train_student(
            train_csv=train_csv,
            output_dir=args.student_output_model_dir,
            max_samples=(args.student_max_samples if args.student_max_samples > 0 else None),
            num_epochs=args.student_epochs,
            model_name=args.student_model,
            label_smoothing=args.student_label_smoothing,
            early_stopping_patience=args.student_early_stopping,
            per_device_train_batch_size=args.student_train_batch_size,
            per_device_eval_batch_size=args.student_eval_batch_size,
            gradient_accumulation_steps=args.student_grad_accum,
            fp16=args.student_fp16,
            bf16=args.student_bf16,
            gradient_checkpointing=args.student_gradient_checkpointing,
            learning_rate=args.student_learning_rate,
            warmup_ratio=args.student_warmup_ratio,
            max_length=args.student_max_length,
            extra_csvs=extra_list,
            shuffle_ab=args.student_shuffle_ab,
            dedup_by_prompt=args.student_dedup_by_prompt,
            use_fast_tokenizer=args.student_use_fast_tokenizer,
            dataloader_num_workers=(args.student_num_workers if args.student_num_workers > 0 else max(2, (os.cpu_count() or 4)//2)),
        )
        logger.info(f'Student training done. Eval metrics: {metrics}')

    if args.mode == 'student-infer':
        logger.info('Running student inference (DistilBERT)')
        test_csv = config['data_paths']['test_data']
        tta_lengths = [int(x) for x in args.student_infer_tta_lengths.split(',') if x.strip()] if args.student_infer_tta_lengths else None
        out = infer_student(
            model_dir=args.student_output_model_dir,
            test_csv=test_csv,
            submission_path=args.student_submission_path,
            max_length=args.student_max_length,
            tta_lengths=tta_lengths,
            load_in_8bit=args.student_infer_8bit,
        )
        logger.info(f'Student submission saved to {out}')

    if args.mode == 'student-calibrate':
        logger.info('Calibrating student probabilities via temperature scaling')
        train_csv_candidates = [
            './data/train.csv',
            './data/lmsys-chatbot-arena/train.csv'
        ]
        train_csv = next((p for p in train_csv_candidates if os.path.exists(p)), None)
        if not train_csv:
            raise FileNotFoundError('train.csv not found in ./data or ./data/lmsys-chatbot-arena')
        info = calibrate_student(model_dir=args.student_output_model_dir, train_csv=train_csv)
        logger.info(f'Calibration saved. Info: {info}')

    if args.mode == 'student-eval-holdout':
        logger.info('Evaluating pre/post calibration log loss on a distinct holdout split')
        train_csv_candidates = [
            './data/train.csv',
            './data/lmsys-chatbot-arena/train.csv'
        ]
        train_csv = next((p for p in train_csv_candidates if os.path.exists(p)), None)
        if not train_csv:
            raise FileNotFoundError('train.csv not found in ./data or ./data/lmsys-chatbot-arena')
        info = calibrate_student(model_dir='./model_save/student_distilbert', train_csv=train_csv)
        logger.info(
            f"Holdout NLL (before/after): {info.get('nll_holdout_before'):.6f} -> {info.get('nll_holdout_after'):.6f} (Δ {info.get('nll_holdout_improvement'):.6f})"
        )
        if 'logloss_holdout_before' in info:
            logger.info(
                f"Holdout LogLoss (before/after): {info.get('logloss_holdout_before'):.6f} -> {info.get('logloss_holdout_after'):.6f} (Δ {info.get('logloss_holdout_improvement'):.6f})"
            )

    if args.mode == 'student-distill-train':
        logger.info('Starting student knowledge distillation training (KL + CE)')
        train_csv_candidates = [
            './data/train.csv',
            './data/lmsys-chatbot-arena/train.csv'
        ]
        train_csv = next((p for p in train_csv_candidates if os.path.exists(p)), None)
        if not train_csv:
            raise FileNotFoundError('train.csv not found in ./data or ./data/lmsys-chatbot-arena')

        if not args.distill_teachers:
            raise ValueError('Please provide --distill-teachers with comma-separated paths or globs to .npy logits files')
        teacher_specs = [s.strip() for s in args.distill_teachers.split(',') if s.strip()]
        teacher_files = []
        for spec in teacher_specs:
            matches = glob.glob(spec)
            if matches:
                teacher_files.extend(matches)
            elif os.path.exists(spec):
                teacher_files.append(spec)
        if not teacher_files:
            raise FileNotFoundError(f'No teacher logits files found from spec: {args.distill_teachers}')
        # Validate teacher logits alignment and shape
        try:
            val_info = validate_logits(teacher_files, train_csv)
            logger.info(f"Validated teacher logits: expected_length={val_info['expected_length']} files={len(val_info['files'])}")
        except Exception as e:
            raise ValueError(f"Teacher logits validation error: {e}")
        logger.info(f'Using {len(teacher_files)} teacher logits files')

        extra_list = [s.strip() for s in args.student_extra_csvs.split(',') if s.strip()]
        metrics = train_student_distill(
            train_csv=train_csv,
            output_dir=args.student_output_model_dir,
            teacher_logits=teacher_files,
            model_name=args.student_model,
            max_samples=args.student_max_samples if args.student_max_samples > 0 else None,
            num_epochs=args.student_epochs,
            alpha=args.distill_alpha,
            T_soft=args.distill_temp,
            mse_weight=args.distill_mse_weight,
            temp_schedule=args.distill_temp_schedule,
            label_smoothing=args.student_label_smoothing,
            per_device_train_batch_size=args.student_train_batch_size,
            per_device_eval_batch_size=args.student_eval_batch_size,
            gradient_accumulation_steps=args.student_grad_accum,
            learning_rate=args.student_learning_rate,
            warmup_ratio=args.student_warmup_ratio,
            fp16=args.student_fp16,
            bf16=args.student_bf16,
            gradient_checkpointing=args.student_gradient_checkpointing,
            early_stopping_patience=args.student_early_stopping,
            max_length=args.student_max_length,
            extra_csvs=extra_list,
            shuffle_ab=args.student_shuffle_ab,
            dedup_by_prompt=args.student_dedup_by_prompt,
            use_fast_tokenizer=args.student_use_fast_tokenizer,
            dataloader_num_workers=(args.student_num_workers if args.student_num_workers > 0 else max(2, (os.cpu_count() or 4)//2)),
            num_folds=args.cv_num_folds,
            fold_idx=args.cv_fold_idx,
        )
        logger.info(f'Distillation training done. Eval metrics: {metrics}')


if __name__ == '__main__':
    main()
