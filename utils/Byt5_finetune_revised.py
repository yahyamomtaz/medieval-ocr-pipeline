import nlpaug.augmenter.char as nac
from transformers import pipeline
from dataclasses import dataclass, field
from typing import Optional
from transformers import set_seed
from transformers import HfArgumentParser, TrainingArguments
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import Trainer, DataCollatorForSeq2Seq
import argparse
import glob
import os
import json
import time
import logging
import random
import re
import csv
import datetime
from itertools import chain
from string import punctuation
from sacrebleu import sentence_bleu
import evaluate
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.optim import AdamW
from datasets import Dataset, DatasetDict
from torchmetrics.text import CharErrorRate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

training_args = TrainingArguments(
    output_dir='./byt5-ocr-correction-2',
    run_name="byt5-ocr-correction-cer",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    warmup_steps=250,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=250,
    num_train_epochs=4,
    fp16=False,
    max_steps=5000,
)


set_seed(training_args.seed)

model_name_or_path = "google/byt5-small"
max_len = 128
cache_dir = None

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    cache_dir=cache_dir,
)

model = T5ForConditionalGeneration.from_pretrained(
    model_name_or_path,
    cache_dir=cache_dir,
)

tokenizer.model_max_length = max_len
model.config.max_length = max_len

set_seed(training_args.seed)

df = pd.read_csv('dataset/dataset_abbreviation_corrected.csv' , sep=';')

def prepare_ocr_correction_dataset(df, tokenizer, max_length=128):
    def preprocess_function(examples):
        inputs = ["correct OCR: " + text for text in examples['ocr_prediction']]
        targets = examples['text']

        model_inputs = tokenizer(
            inputs,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        labels = tokenizer(
            targets,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        # Set labels
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = Dataset.from_pandas(df)

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    return tokenized_dataset

def split_dataset(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    train_df = df[:train_size].reset_index(drop=True)
    val_df = df[train_size:].reset_index(drop=True)
    return train_df, val_df

def prepare_complete_dataset(df, tokenizer, max_length=128, train_ratio=0.8):
    train_df, val_df = split_dataset(df, train_ratio)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    train_dataset = prepare_ocr_correction_dataset(train_df, tokenizer, max_length)
    val_dataset = prepare_ocr_correction_dataset(val_df, tokenizer, max_length)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    return dataset_dict

def show_dataset_examples(dataset, tokenizer, num_examples=3):
    for i in range(min(num_examples, len(dataset))):
        print(f"\n--- Example {i+1} ---")

        input_ids = dataset[i]['input_ids']
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"Input: {input_text}")

        labels = dataset[i]['labels']
        labels_for_decode = [l if l != -100 else tokenizer.pad_token_id for l in labels]
        target_text = tokenizer.decode(labels_for_decode, skip_special_tokens=True)
        print(f"Target: {target_text}")

dataset_dict = prepare_complete_dataset(df, tokenizer, max_len)
show_dataset_examples(dataset_dict['train'], tokenizer, num_examples=2)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)

def compute_cer(predictions, references):
    total_chars = 0
    total_errors = 0

    for pred, ref in zip(predictions, references):
        pred = ' '.join(pred.split())
        ref = ' '.join(ref.split())

        total_chars += len(ref)

        edit_distance = compute_edit_distance(pred, ref)
        total_errors += edit_distance

    cer = total_errors / total_chars if total_chars > 0 else 0
    return cer

def compute_edit_distance(s1, s2):
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

globals()['tokenizer'] = tokenizer

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Ensure predictions are in the correct format
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert logits to token IDs by taking argmax
    if predictions.ndim == 3:  # (batch_size, seq_len, vocab_size)
        predictions = np.argmax(predictions, axis=-1)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute CER
    cer = compute_cer(decoded_preds, decoded_labels)

    # Optionally, compute other metrics like BLEU
    bleu_scores = []
    for pred, ref in zip(decoded_preds, decoded_labels):
        bleu = sentence_bleu(pred, [ref]).score / 100.0
        bleu_scores.append(bleu)
    avg_bleu = np.mean(bleu_scores)

    # Prepare metrics dictionary
    metrics = {
        "cer": cer,
        "bleu": avg_bleu,
    }

    # Write metrics to CSV file
    write_metrics_to_csv(metrics)

    return metrics

def write_metrics_to_csv(metrics):
    """Write evaluation metrics to a CSV file with timestamp"""
    csv_file = os.path.join(training_args.output_dir, "evaluation_metrics.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_file)
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare row data
    row_data = {
        "timestamp": timestamp,
        "cer": metrics["cer"],
        "bleu": metrics["bleu"]
    }
    
    # Write to CSV
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "cer", "bleu"])
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write metrics row
        writer.writerow(row_data)
    
    print(f"Metrics written to: {csv_file}")

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)