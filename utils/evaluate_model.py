#!/usr/bin/env python3
"""
ByT5 OCR Correction Model Evaluation Script
==========================================

Comprehensive evaluation of the fine-tuned ByT5 model for medieval OCR correction.
This script provides detailed performance analysis including:

- Character Error Rate (CER) - Primary OCR metric
- BLEU Score - Sequence similarity from machine translation
- Edit Distance (Levenshtein) - Character-level edits required
- Word Error Rate (WER) - Word-level accuracy
- Accuracy metrics - Exact match and character-level accuracy

Usage:
    python evaluate_model.py --model_path ./byt5-ocr-correction --test_data dataset/test_split.csv
"""

import os
import argparse
import json
import logging
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sacrebleu import sentence_bleu
from sklearn.metrics import classification_report
import difflib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Evaluation configuration
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OCREvaluator:
    """
    Comprehensive evaluator for OCR correction models.
    
    This class provides a complete suite of evaluation metrics
    commonly used in OCR and text correction research.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the evaluator with a trained model.
        
        Args:
            model_path (str): Path to the fine-tuned ByT5 model
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = DEVICE
        
        logger.info(f"Initializing OCR Evaluator with model: {model_path}")
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✓ Model loaded successfully on {self.device}")
            logger.info(f"  - Parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_single(self, ocr_text: str) -> str:
        """
        Predict correction for a single OCR text.
        
        Args:
            ocr_text (str): Raw OCR text
            
        Returns:
            str: Corrected text
        """
        if not ocr_text.strip():
            return ""
        
        # Prepare input with task prefix
        input_text = f"correct: {ocr_text}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate correction
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=MAX_SEQUENCE_LENGTH,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode result
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text.strip()
    
    def predict_batch(self, ocr_texts: List[str]) -> List[str]:
        """
        Predict corrections for a batch of OCR texts.
        
        Args:
            ocr_texts (List[str]): List of raw OCR texts
            
        Returns:
            List[str]: List of corrected texts
        """
        corrections = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(ocr_texts), BATCH_SIZE):
            batch = ocr_texts[i:i + BATCH_SIZE]
            batch_corrections = []
            
            for ocr_text in batch:
                correction = self.predict_single(ocr_text)
                batch_corrections.append(correction)
            
            corrections.extend(batch_corrections)
            
            if i % (BATCH_SIZE * 10) == 0:
                logger.info(f"Processed {i + len(batch)}/{len(ocr_texts)} samples")
        
        return corrections
    
    @staticmethod
    def compute_edit_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return OCREvaluator.compute_edit_distance(s2, s1)
        
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
    
    @staticmethod
    def compute_cer(predictions: List[str], references: List[str]) -> float:
        """
        Compute Character Error Rate (CER).
        
        CER = (S + D + I) / N
        where S=substitutions, D=deletions, I=insertions, N=total characters
        """
        total_chars = 0
        total_errors = 0
        
        for pred, ref in zip(predictions, references):
            # Normalize whitespace
            pred = ' '.join(pred.split())
            ref = ' '.join(ref.split())
            
            total_chars += len(ref)
            edit_distance = OCREvaluator.compute_edit_distance(pred, ref)
            total_errors += edit_distance
        
        return total_errors / total_chars if total_chars > 0 else 0
    
    @staticmethod
    def compute_wer(predictions: List[str], references: List[str]) -> float:
        """
        Compute Word Error Rate (WER).
        
        WER = (S + D + I) / N
        where S=substitutions, D=deletions, I=insertions, N=total words
        """
        total_words = 0
        total_errors = 0
        
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            
            total_words += len(ref_words)
            edit_distance = OCREvaluator.compute_edit_distance(
                ' '.join(pred_words), ' '.join(ref_words)
            )
            total_errors += edit_distance
        
        return total_errors / total_words if total_words > 0 else 0
    
    @staticmethod
    def compute_bleu_scores(predictions: List[str], references: List[str]) -> List[float]:
        """Compute BLEU scores for each prediction-reference pair."""
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            try:
                bleu = sentence_bleu(pred, [ref]).score / 100.0
                bleu_scores.append(bleu)
            except:
                bleu_scores.append(0.0)
        
        return bleu_scores
    
    @staticmethod
    def compute_accuracy_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute various accuracy metrics."""
        exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        exact_match_accuracy = exact_matches / len(predictions) if predictions else 0
        
        # Character-level accuracy
        total_chars = 0
        correct_chars = 0
        
        for pred, ref in zip(predictions, references):
            pred = pred.strip()
            ref = ref.strip()
            
            # Align strings for character comparison
            max_len = max(len(pred), len(ref))
            pred_padded = pred.ljust(max_len)
            ref_padded = ref.ljust(max_len)
            
            for p_char, r_char in zip(pred_padded, ref_padded):
                total_chars += 1
                if p_char == r_char:
                    correct_chars += 1
        
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        
        return {
            "exact_match_accuracy": exact_match_accuracy,
            "character_accuracy": char_accuracy,
            "exact_matches": exact_matches,
            "total_samples": len(predictions)
        }
    
    def evaluate_dataset(self, test_data: pd.DataFrame) -> Dict:
        """
        Comprehensive evaluation of the model on a test dataset.
        
        Args:
            test_data (pd.DataFrame): DataFrame with 'ocr_prediction' and 'text' columns
            
        Returns:
            Dict: Comprehensive evaluation results
        """
        logger.info(f"Starting evaluation on {len(test_data)} samples...")
        
        start_time = time.time()
        
        # Extract data
        ocr_texts = test_data['ocr_prediction'].tolist()
        ground_truths = test_data['text'].tolist()
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = self.predict_batch(ocr_texts)
        
        evaluation_time = time.time() - start_time
        
        # Compute metrics
        logger.info("Computing evaluation metrics...")
        
        # Core OCR metrics
        cer = self.compute_cer(predictions, ground_truths)
        wer = self.compute_wer(predictions, ground_truths)
        bleu_scores = self.compute_bleu_scores(predictions, ground_truths)
        accuracy_metrics = self.compute_accuracy_metrics(predictions, ground_truths)
        
        # Edit distances
        edit_distances = [
            self.compute_edit_distance(pred, ref)
            for pred, ref in zip(predictions, ground_truths)
        ]
        
        # Aggregate results
        results = {
            # Primary metrics
            "character_error_rate": cer,
            "word_error_rate": wer,
            "bleu_score_mean": np.mean(bleu_scores),
            "bleu_score_std": np.std(bleu_scores),
            
            # Accuracy metrics
            "exact_match_accuracy": accuracy_metrics["exact_match_accuracy"],
            "character_accuracy": accuracy_metrics["character_accuracy"],
            
            # Edit distance statistics
            "edit_distance_mean": np.mean(edit_distances),
            "edit_distance_std": np.std(edit_distances),
            "edit_distance_median": np.median(edit_distances),
            
            # Dataset statistics
            "total_samples": len(test_data),
            "exact_matches": accuracy_metrics["exact_matches"],
            "evaluation_time_seconds": evaluation_time,
            "samples_per_second": len(test_data) / evaluation_time,
            
            # Raw data for analysis
            "predictions": predictions,
            "ground_truths": ground_truths,
            "ocr_inputs": ocr_texts,
            "bleu_scores": bleu_scores,
            "edit_distances": edit_distances
        }
        
        # Log summary
        logger.info("Evaluation completed!")
        logger.info(f"Results Summary:")
        logger.info(f"  - Character Error Rate: {cer:.4f} ({cer*100:.2f}%)")
        logger.info(f"  - Word Error Rate: {wer:.4f} ({wer*100:.2f}%)")
        logger.info(f"  - BLEU Score: {np.mean(bleu_scores):.4f}")
        logger.info(f"  - Exact Match Accuracy: {accuracy_metrics['exact_match_accuracy']:.4f}")
        logger.info(f"  - Character Accuracy: {accuracy_metrics['character_accuracy']:.4f}")
        logger.info(f"  - Average Edit Distance: {np.mean(edit_distances):.2f}")
        logger.info(f"  - Evaluation Time: {evaluation_time:.2f}s ({len(test_data)/evaluation_time:.1f} samples/s)")
        
        return results
    
    def analyze_errors(self, results: Dict, output_dir: str = "evaluation_analysis"):
        """
        Perform detailed error analysis and generate reports.
        
        Args:
            results (Dict): Evaluation results from evaluate_dataset
            output_dir (str): Directory to save analysis files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        predictions = results["predictions"]
        ground_truths = results["ground_truths"]
        ocr_inputs = results["ocr_inputs"]
        edit_distances = results["edit_distances"]
        bleu_scores = results["bleu_scores"]
        
        logger.info(f"Performing error analysis...")
        
        # 1. Error distribution analysis
        plt.figure(figsize=(15, 10))
        
        # Edit distance distribution
        plt.subplot(2, 3, 1)
        plt.hist(edit_distances, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Edit Distance Distribution')
        plt.xlabel('Edit Distance')
        plt.ylabel('Frequency')
        
        # BLEU score distribution
        plt.subplot(2, 3, 2)
        plt.hist(bleu_scores, bins=30, alpha=0.7, edgecolor='black')
        plt.title('BLEU Score Distribution')
        plt.xlabel('BLEU Score')
        plt.ylabel('Frequency')
        
        # Error rate by input length
        input_lengths = [len(text) for text in ocr_inputs]
        plt.subplot(2, 3, 3)
        plt.scatter(input_lengths, edit_distances, alpha=0.5)
        plt.title('Error Rate vs Input Length')
        plt.xlabel('Input Length (characters)')
        plt.ylabel('Edit Distance')
        
        # CER by input length
        cer_by_length = [ed / len(gt) if len(gt) > 0 else 0 
                        for ed, gt in zip(edit_distances, ground_truths)]
        plt.subplot(2, 3, 4)
        plt.scatter(input_lengths, cer_by_length, alpha=0.5)
        plt.title('CER vs Input Length')
        plt.xlabel('Input Length (characters)')
        plt.ylabel('Character Error Rate')
        
        # Improvement analysis (OCR vs Corrected)
        ocr_errors = [self.compute_edit_distance(ocr, gt) 
                     for ocr, gt in zip(ocr_inputs, ground_truths)]
        correction_errors = edit_distances
        improvements = [ocr_err - corr_err for ocr_err, corr_err in zip(ocr_errors, correction_errors)]
        
        plt.subplot(2, 3, 5)
        plt.hist(improvements, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Error Reduction Distribution')
        plt.xlabel('Error Reduction (positive = improvement)')
        plt.ylabel('Frequency')
        
        # Correlation between metrics
        plt.subplot(2, 3, 6)
        plt.scatter(edit_distances, bleu_scores, alpha=0.5)
        plt.title('Edit Distance vs BLEU Score')
        plt.xlabel('Edit Distance')
        plt.ylabel('BLEU Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_analysis_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Generate detailed error report
        error_analysis = []
        
        for i, (ocr, pred, gt, ed, bleu) in enumerate(zip(
            ocr_inputs, predictions, ground_truths, edit_distances, bleu_scores
        )):
            error_analysis.append({
                'sample_id': i,
                'ocr_input': ocr,
                'prediction': pred,
                'ground_truth': gt,
                'edit_distance': ed,
                'bleu_score': bleu,
                'input_length': len(ocr),
                'output_length': len(pred),
                'target_length': len(gt),
                'ocr_errors': self.compute_edit_distance(ocr, gt),
                'improvement': self.compute_edit_distance(ocr, gt) - ed
            })
        
        # Save detailed results
        error_df = pd.DataFrame(error_analysis)
        error_df.to_csv(os.path.join(output_dir, 'detailed_error_analysis.csv'), index=False)
        
        # 3. Find worst and best cases
        worst_cases = error_df.nlargest(10, 'edit_distance')
        best_improvements = error_df.nlargest(10, 'improvement')
        
        # Save examples
        with open(os.path.join(output_dir, 'error_examples.txt'), 'w', encoding='utf-8') as f:
            f.write("WORST CORRECTION CASES (Highest Edit Distance)\n")
            f.write("=" * 60 + "\n\n")
            
            for _, row in worst_cases.iterrows():
                f.write(f"Sample ID: {row['sample_id']}\n")
                f.write(f"OCR Input:    {row['ocr_input']}\n")
                f.write(f"Prediction:   {row['prediction']}\n")
                f.write(f"Ground Truth: {row['ground_truth']}\n")
                f.write(f"Edit Distance: {row['edit_distance']}\n")
                f.write(f"BLEU Score: {row['bleu_score']:.4f}\n")
                f.write("-" * 40 + "\n\n")
            
            f.write("\nBEST IMPROVEMENT CASES\n")
            f.write("=" * 60 + "\n\n")
            
            for _, row in best_improvements.iterrows():
                f.write(f"Sample ID: {row['sample_id']}\n")
                f.write(f"OCR Input:    {row['ocr_input']}\n")
                f.write(f"Prediction:   {row['prediction']}\n")
                f.write(f"Ground Truth: {row['ground_truth']}\n")
                f.write(f"Improvement: {row['improvement']} errors reduced\n")
                f.write(f"BLEU Score: {row['bleu_score']:.4f}\n")
                f.write("-" * 40 + "\n\n")
        
        logger.info(f"✓ Error analysis saved to: {output_dir}")
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file."""
        # Prepare results for JSON serialization
        json_results = {k: v for k, v in results.items() 
                       if k not in ['predictions', 'ground_truths', 'ocr_inputs']}
        
        # Add summary statistics
        json_results['evaluation_summary'] = {
            'model_path': self.model_path,
            'device': str(self.device),
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_parameters': self.model.num_parameters()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Results saved to: {output_path}")


def load_test_data(data_path: str) -> pd.DataFrame:
    """Load and validate test dataset."""
    logger.info(f"Loading test data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data file not found: {data_path}")
    
    # Try different separators
    for sep in [';', ',', '\t']:
        try:
            df = pd.read_csv(data_path, sep=sep, encoding='utf-8')
            if 'ocr_prediction' in df.columns and 'text' in df.columns:
                break
        except:
            continue
    else:
        raise ValueError("Could not load test data. Check file format and columns.")
    
    # Validate columns
    required_columns = ['ocr_prediction', 'text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove null values
    initial_len = len(df)
    df = df.dropna(subset=required_columns)
    if len(df) < initial_len:
        logger.warning(f"Removed {initial_len - len(df)} samples with null values")
    
    logger.info(f"✓ Test data loaded: {len(df)} samples")
    return df


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluate ByT5 OCR Correction Model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to the fine-tuned ByT5 model'
    )
    parser.add_argument(
        '--test_data',
        required=True,
        help='Path to test dataset (CSV file)'
    )
    parser.add_argument(
        '--output_dir',
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--analysis',
        action='store_true',
        help='Perform detailed error analysis'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize evaluator
        evaluator = OCREvaluator(args.model_path)
        
        # Load test data
        test_data = load_test_data(args.test_data)
        
        # Run evaluation
        results = evaluator.evaluate_dataset(test_data)
        
        # Save results
        results_path = os.path.join(args.output_dir, 'evaluation_results.json')
        evaluator.save_results(results, results_path)
        
        # Perform error analysis if requested
        if args.analysis:
            analysis_dir = os.path.join(args.output_dir, 'error_analysis')
            evaluator.analyze_errors(results, analysis_dir)
        
        logger.info("=" * 60)
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 