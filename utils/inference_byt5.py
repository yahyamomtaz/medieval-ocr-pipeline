#!/usr/bin/env python3
"""
Inference script for ByT5 OCR correction model.
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
import json
from typing import List, Dict


class ByT5OCRCorrector:
    """ByT5 OCR Correction model wrapper."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the OCR corrector.
        
        Args:
            model_path (str): Path to the trained model
            device (str): Device to run inference on
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Default task prefix
        self.task_prefix = "correct ocr: "
        
        print("Model loaded successfully!")
    
    def correct_text(self, ocr_text: str, max_length: int = 512, num_beams: int = 4, early_stopping: bool = True) -> str:
        """
        Correct OCR text using the trained model.
        
        Args:
            ocr_text (str): OCR predicted text to correct
            max_length (int): Maximum length for generation
            num_beams (int): Number of beams for beam search
            early_stopping (bool): Whether to use early stopping
            
        Returns:
            str: Corrected text
        """
        # Add task prefix
        input_text = self.task_prefix + ocr_text
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate correction
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                do_sample=False
            )
        
        # Decode output
        corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_text
    
    def correct_batch(self, ocr_texts: List[str], max_length: int = 512, num_beams: int = 4, batch_size: int = 8) -> List[str]:
        """
        Correct multiple OCR texts in batches.
        
        Args:
            ocr_texts (List[str]): List of OCR texts to correct
            max_length (int): Maximum length for generation
            num_beams (int): Number of beams for beam search
            batch_size (int): Batch size for processing
            
        Returns:
            List[str]: List of corrected texts
        """
        corrected_texts = []
        
        for i in range(0, len(ocr_texts), batch_size):
            batch = ocr_texts[i:i + batch_size]
            
            # Add task prefix to all texts in batch
            input_texts = [self.task_prefix + text for text in batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate corrections
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode outputs
            batch_corrected = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            corrected_texts.extend(batch_corrected)
        
        return corrected_texts


def load_test_data(file_path: str) -> List[Dict]:
    """Load test data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def evaluate_corrections(corrector: ByT5OCRCorrector, test_data: List[Dict]) -> Dict:
    """
    Evaluate the model on test data.
    
    Args:
        corrector: ByT5OCRCorrector instance
        test_data: List of test samples
        
    Returns:
        Dict: Evaluation metrics
    """
    print(f"Evaluating on {len(test_data)} samples...")
    
    # Extract OCR texts
    ocr_texts = [item['input'].replace(corrector.task_prefix, '') for item in test_data]
    ground_truth = [item['target'] for item in test_data]
    
    # Generate corrections
    predictions = corrector.correct_batch(ocr_texts, batch_size=8)
    
    # Calculate metrics
    exact_matches = 0
    char_correct = 0
    char_total = 0
    word_correct = 0
    word_total = 0
    
    for pred, gt in zip(predictions, ground_truth):
        # Exact match
        if pred.strip() == gt.strip():
            exact_matches += 1
        
        # Character-level accuracy
        pred_chars = pred.strip()
        gt_chars = gt.strip()
        
        for i in range(min(len(pred_chars), len(gt_chars))):
            if pred_chars[i] == gt_chars[i]:
                char_correct += 1
            char_total += 1
        
        char_total += abs(len(pred_chars) - len(gt_chars))
        
        # Word-level accuracy
        pred_words = pred.strip().split()
        gt_words = gt.strip().split()
        
        for i in range(min(len(pred_words), len(gt_words))):
            if pred_words[i] == gt_words[i]:
                word_correct += 1
            word_total += 1
        
        word_total += abs(len(pred_words) - len(gt_words))
    
    metrics = {
        'exact_match_accuracy': exact_matches / len(test_data),
        'character_accuracy': char_correct / char_total if char_total > 0 else 0,
        'word_accuracy': word_correct / word_total if word_total > 0 else 0,
        'total_samples': len(test_data)
    }
    
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description="ByT5 OCR Correction Inference")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--test_data", help="Path to test data JSONL file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--input_text", help="Single text to correct")
    parser.add_argument("--output_file", help="Output file for corrections")
    parser.add_argument("--max_length", type=int, default=512, help="Max generation length")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize corrector
    corrector = ByT5OCRCorrector(args.model_path, args.device)
    
    if args.interactive:
        # Interactive mode
        print("\n🔧 Interactive OCR Correction Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            ocr_text = input("\nEnter OCR text to correct: ").strip()
            if ocr_text.lower() == 'quit':
                break
            
            if ocr_text:
                corrected = corrector.correct_text(ocr_text, args.max_length, args.num_beams)
                print(f"Original:  {ocr_text}")
                print(f"Corrected: {corrected}")
    
    elif args.input_text:
        # Single text correction
        print(f"\n🔧 Correcting single text...")
        corrected = corrector.correct_text(args.input_text, args.max_length, args.num_beams)
        print(f"Original:  {args.input_text}")
        print(f"Corrected: {corrected}")
    
    elif args.test_data:
        # Evaluate on test data
        print(f"\n📊 Evaluating on test data...")
        test_data = load_test_data(args.test_data)
        metrics, predictions = evaluate_corrections(corrector, test_data)
        
        print(f"\n=== Evaluation Results ===")
        print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
        print(f"Character Accuracy: {metrics['character_accuracy']:.4f}")
        print(f"Word Accuracy: {metrics['word_accuracy']:.4f}")
        print(f"Total Samples: {metrics['total_samples']}")
        
        # Save results if output file specified
        if args.output_file:
            results = []
            for i, (sample, pred) in enumerate(zip(test_data, predictions)):
                results.append({
                    'line_id': sample.get('line_id', f'sample_{i}'),
                    'original_ocr': sample['input'].replace(corrector.task_prefix, ''),
                    'ground_truth': sample['target'],
                    'prediction': pred,
                    'exact_match': pred.strip() == sample['target'].strip()
                })
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics': metrics,
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to: {args.output_file}")
    
    else:
        print("Please specify --interactive, --input_text, or --test_data")


if __name__ == "__main__":
    main() 