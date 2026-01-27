#!/usr/bin/env python3

import os
import json
import time
import logging
import datetime
import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, DatasetDict
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed
)
from sacrebleu import sentence_bleu
import evaluate

# Configure logging for research reproducibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model and training configuration
MODEL_NAME = "google/byt5-small"
DATASET_PATH = 'dataset/dataset_abbreviation_corrected.csv'
OUTPUT_DIR = './byt5-ocr-correction-2'
MAX_SEQUENCE_LENGTH = 128
TASK_PREFIX = "correct OCR: "

# Training hyperparameters optimized for medieval text correction
TRAINING_CONFIG = {
    'output_dir': OUTPUT_DIR,
    'run_name': "byt5-ocr-correction-cer",
    'overwrite_output_dir': True,
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'gradient_accumulation_steps': 4,
    'learning_rate': 5e-4,
    'warmup_steps': 250,
    'logging_steps': 100,
    'eval_strategy': "steps",
    'eval_steps': 250,
    'num_train_epochs': 4,
    'fp16': False,
    'max_steps': 5000,
    'save_strategy': "steps",
    'save_steps': 500,
    'load_best_model_at_end': True,
    'metric_for_best_model': "cer",
    'greater_is_better': False,
    'report_to': None  # Disable wandb/tensorboard for reproducibility
}


@dataclass
class OCRDatasetConfig:
    """Configuration for OCR correction dataset processing."""
    csv_separator: str = ';'
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    max_input_length: int = MAX_SEQUENCE_LENGTH
    max_target_length: int = MAX_SEQUENCE_LENGTH
    task_prefix: str = TASK_PREFIX


def setup_device_and_seed(seed: int = 42) -> str:
    """
    Configure computing device and set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed for reproducible results
        
    Returns:
        str: Device identifier ('cuda' or 'cpu')
    """
    # Determine optimal device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log hardware information for research documentation
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        logger.info("Using CPU (GPU not available)")
    
    # Set seeds for reproducibility
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    return str(device)


def load_and_validate_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load and validate the OCR correction dataset.
    
    Args:
        dataset_path (str): Path to the CSV dataset file
        
    Returns:
        pd.DataFrame: Validated dataset
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If required columns are missing
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Load dataset with proper encoding for medieval characters
    df = pd.read_csv(dataset_path, sep=';', encoding='utf-8')
    
    # Validate required columns
    required_columns = ['ocr_prediction', 'text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Log dataset statistics
    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  - Total samples: {len(df):,}")
    logger.info(f"  - Columns: {list(df.columns)}")
    logger.info(f"  - Average OCR length: {df['ocr_prediction'].str.len().mean():.1f} chars")
    logger.info(f"  - Average target length: {df['text'].str.len().mean():.1f} chars")
    
    # Remove any null values
    initial_len = len(df)
    df = df.dropna(subset=['ocr_prediction', 'text'])
    if len(df) < initial_len:
        logger.warning(f"Removed {initial_len - len(df)} samples with null values")
    
    return df


def prepare_ocr_correction_dataset(
    df: pd.DataFrame, 
    tokenizer: AutoTokenizer, 
    config: OCRDatasetConfig
) -> HFDataset:
    """
    Prepare dataset for ByT5 fine-tuning with proper formatting.
    
    This function:
    1. Adds task prefix to OCR predictions
    2. Tokenizes input and target sequences
    3. Handles truncation and padding
    4. Prepares labels for training
    
    Args:
        df (pd.DataFrame): Input dataset
        tokenizer (AutoTokenizer): ByT5 tokenizer
        config (OCRDatasetConfig): Dataset configuration
        
    Returns:
        HFDataset: Tokenized dataset ready for training
    """
    def preprocess_function(examples):
        """Tokenize and format examples for ByT5."""
        # Add task prefix to inputs (as used in T5 training)
        inputs = [config.task_prefix + text for text in examples['ocr_prediction']]
        targets = examples['text']

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=config.max_input_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        # Tokenize targets
        labels = tokenizer(
            targets,
            max_length=config.max_target_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        # T5 uses labels for both input and loss calculation
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Convert to HuggingFace dataset format
    dataset = HFDataset.from_pandas(df)

    # Apply tokenization
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing OCR correction dataset"
    )
    
    logger.info(f"Dataset tokenization completed: {len(tokenized_dataset)} samples")
    return tokenized_dataset


def split_dataset(
    df: pd.DataFrame, 
    config: OCRDatasetConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/validation/test sets with logging.
    
    Args:
        df (pd.DataFrame): Complete dataset
        config (OCRDatasetConfig): Dataset configuration
        
    Returns:
        Tuple of train, validation, and test DataFrames
    """
    total_samples = len(df)
    
    # Calculate split sizes
    train_size = int(total_samples * config.train_ratio)
    val_size = int(total_samples * config.validation_ratio)
    
    # Perform splits
    train_df = df[:train_size].reset_index(drop=True)
    val_df = df[train_size:train_size + val_size].reset_index(drop=True)
    test_df = df[train_size + val_size:].reset_index(drop=True)
    
    # Log split information
    logger.info(f"Dataset split completed:")
    logger.info(f"  - Training samples: {len(train_df):,} ({len(train_df)/total_samples*100:.1f}%)")
    logger.info(f"  - Validation samples: {len(val_df):,} ({len(val_df)/total_samples*100:.1f}%)")
    logger.info(f"  - Test samples: {len(test_df):,} ({len(test_df)/total_samples*100:.1f}%)")
    
    return train_df, val_df, test_df


def prepare_complete_dataset(
    df: pd.DataFrame, 
    tokenizer: AutoTokenizer, 
    config: OCRDatasetConfig
) -> DatasetDict:
    """
    Prepare complete dataset with train/validation splits.
    
    Args:
        df (pd.DataFrame): Complete dataset
        tokenizer (AutoTokenizer): ByT5 tokenizer
        config (OCRDatasetConfig): Dataset configuration
        
    Returns:
        DatasetDict: Dictionary containing train and validation datasets
    """
    # Split the data
    train_df, val_df, test_df = split_dataset(df, config)
    
    # Prepare tokenized datasets
    train_dataset = prepare_ocr_correction_dataset(train_df, tokenizer, config)
    val_dataset = prepare_ocr_correction_dataset(val_df, tokenizer, config)

    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    return dataset_dict


def show_dataset_examples(
    dataset: HFDataset, 
    tokenizer: AutoTokenizer, 
    num_examples: int = 3
) -> None:
    """
    Display examples from the dataset for verification.
    
    Args:
        dataset: Tokenized dataset
        tokenizer: ByT5 tokenizer
        num_examples: Number of examples to show
    """
    logger.info(f"Dataset examples (showing {num_examples}):")
    
    for i in range(min(num_examples, len(dataset))):
        print(f"\n--- Example {i+1} ---")

        # Decode input
        input_ids = dataset[i]['input_ids']
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"Input: {input_text}")

        # Decode target (handle -100 labels)
        labels = dataset[i]['labels']
        labels_for_decode = [l if l != -100 else tokenizer.pad_token_id for l in labels]
        target_text = tokenizer.decode(labels_for_decode, skip_special_tokens=True)
        print(f"Target: {target_text}")


def compute_edit_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    
    This is a core metric for OCR evaluation, measuring the minimum
    number of character-level edits (insertions, deletions, substitutions)
    needed to transform one string into another.
    
    Args:
        s1, s2: Input strings
        
    Returns:
        int: Edit distance
    """
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Character Error Rate (CER) for OCR evaluation.
    
    CER is the standard metric for OCR systems, calculated as:
    CER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=total characters
    
    Args:
        predictions: List of predicted text strings
        references: List of ground truth text strings
        
    Returns:
        float: Character Error Rate (0.0 = perfect, 1.0 = completely wrong)
    """
    total_chars = 0
    total_errors = 0

    for pred, ref in zip(predictions, references):
        # Normalize whitespace
        pred = ' '.join(pred.split())
        ref = ' '.join(ref.split())

        total_chars += len(ref)
        edit_distance = compute_edit_distance(pred, ref)
        total_errors += edit_distance

    cer = total_errors / total_chars if total_chars > 0 else 0
    return cer


def compute_metrics(eval_pred, tokenizer: AutoTokenizer) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for model performance.
    
    This function calculates multiple metrics used in OCR research:
    - Character Error Rate (CER): Primary OCR metric
    - BLEU Score: Sequence similarity metric from MT
    - Edit Distance: Raw number of character edits needed
    
    Args:
        eval_pred: Predictions and labels from trainer
        tokenizer: ByT5 tokenizer for decoding
        
    Returns:
        Dict containing computed metrics
    """
    predictions, labels = eval_pred

    # Handle different prediction formats
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert logits to token IDs
    if predictions.ndim == 3:  # (batch_size, seq_len, vocab_size)
        predictions = np.argmax(predictions, axis=-1)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Handle -100 labels (used for padding in loss calculation)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute Character Error Rate (primary metric)
    cer = compute_cer(decoded_preds, decoded_labels)

    # Compute BLEU scores for sequence-level evaluation
    bleu_scores = []
    edit_distances = []
    
    for pred, ref in zip(decoded_preds, decoded_labels):
        # BLEU score (0-1 scale)
        bleu = sentence_bleu(pred, [ref]).score / 100.0
        bleu_scores.append(bleu)
        
        # Edit distance
        edit_dist = compute_edit_distance(pred, ref)
        edit_distances.append(edit_dist)

    # Aggregate metrics
    metrics = {
        "cer": cer,
        "bleu": np.mean(bleu_scores),
        "edit_distance": np.mean(edit_distances),
        "eval_samples": len(decoded_preds)
    }

    # Log metrics for research documentation
    logger.info(f"Evaluation metrics: CER={cer:.4f}, BLEU={metrics['bleu']:.4f}, "
                f"Edit Distance={metrics['edit_distance']:.2f}")

    # Save metrics to CSV for analysis
    write_metrics_to_csv(metrics)

    return metrics


def write_metrics_to_csv(metrics: Dict[str, float]) -> None:
    """
    Write evaluation metrics to CSV file for research tracking.
    
    This enables tracking of model performance over training steps
    and comparison across different experiments.
    
    Args:
        metrics: Dictionary of computed metrics
    """
    csv_file = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
    
    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if file exists to determine headers
    file_exists = os.path.exists(csv_file)
    
    # Prepare row data with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_data = {
        "timestamp": timestamp,
        "cer": metrics["cer"],
        "bleu": metrics["bleu"],
        "edit_distance": metrics.get("edit_distance", 0),
        "eval_samples": metrics.get("eval_samples", 0)
    }
    
    # Write to CSV
    try:
        with open(csv_file, "a", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row_data.keys()))
        
            # Write header if new file
        if not file_exists:
            writer.writeheader()
        
        # Write metrics row
        writer.writerow(row_data)
    
        logger.debug(f"Metrics saved to: {csv_file}")
        
    except Exception as e:
        logger.error(f"Failed to write metrics to CSV: {e}")


def setup_trainer(
    model: T5ForConditionalGeneration,
    tokenizer: AutoTokenizer,
    dataset_dict: DatasetDict,
    training_args: TrainingArguments
) -> Trainer:
    """
    Set up the Hugging Face trainer with custom metrics.
    
    Args:
        model: Fine-tuned T5 model
        tokenizer: ByT5 tokenizer
        dataset_dict: Training and validation datasets
        training_args: Training configuration
        
    Returns:
        Configured Trainer instance
    """
    # Data collator for sequence-to-sequence models
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Create compute_metrics function with tokenizer access
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    # Initialize trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['validation'],
    data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper,
)

    return trainer


def main():
    """
    Main training pipeline for ByT5 OCR correction model.
    
    This function orchestrates the complete training process:
    1. Environment setup and validation
    2. Dataset loading and preprocessing
    3. Model initialization
    4. Training execution
    5. Model saving and evaluation
    """
    logger.info("=" * 60)
    logger.info("ByT5 Medieval OCR Correction - Training Pipeline")
    logger.info("=" * 60)
    
    # Setup environment
    device = setup_device_and_seed()
    
    # Load and validate dataset
    df = load_and_validate_dataset(DATASET_PATH)
    
    # Initialize tokenizer and model
    logger.info(f"Initializing ByT5 model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Configure model for our task
    tokenizer.model_max_length = MAX_SEQUENCE_LENGTH
    model.config.max_length = MAX_SEQUENCE_LENGTH
    
    logger.info(f"Model configuration:")
    logger.info(f"  - Model: {MODEL_NAME}")
    logger.info(f"  - Parameters: {model.num_parameters():,}")
    logger.info(f"  - Max sequence length: {MAX_SEQUENCE_LENGTH}")
    
    # Prepare datasets
    config = OCRDatasetConfig()
    dataset_dict = prepare_complete_dataset(df, tokenizer, config)
    
    # Show examples for verification
    show_dataset_examples(dataset_dict['train'], tokenizer, num_examples=2)
    
    # Setup training arguments
    training_args = TrainingArguments(**TRAINING_CONFIG)
    
    # Initialize trainer
    trainer = setup_trainer(model, tokenizer, dataset_dict, training_args)
    
    # Log training configuration
    logger.info("Training configuration:")
    for key, value in TRAINING_CONFIG.items():
        logger.info(f"  - {key}: {value}")
    
    # Start training
    logger.info("Starting training...")
    start_time = time.time()
    
    try:
        # Train the model
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        
        # Save final model and tokenizer
        logger.info(f"Saving model to: {OUTPUT_DIR}")
trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        
        logger.info("Final evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"  - {key}: {value:.4f}")
        
        # Save training summary
        summary = {
            "model_name": MODEL_NAME,
            "dataset_path": DATASET_PATH,
            "training_samples": len(dataset_dict['train']),
            "validation_samples": len(dataset_dict['validation']),
            "training_time_seconds": training_time,
            "final_training_loss": train_result.training_loss,
            "final_eval_results": eval_results,
            "training_config": TRAINING_CONFIG
        }
        
        with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {OUTPUT_DIR}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()