# Medieval OCR Text Correction Pipeline using ByT5 and Kraken

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-MDPI%20Electronics-orange)](your-paper-link)

This repository contains the implementation of a complete OCR text correction pipeline specifically designed for medieval manuscripts. The system combines **Kraken** for line segmentation, **TrOCR** for initial text recognition, and a fine-tuned **ByT5** model for OCR error correction.

## 🔍 Overview

Medieval manuscripts present unique challenges for OCR systems due to:
- **Historical writing styles** and letterforms
- **Abbreviations and contractions** common in medieval texts
- **Varying print quality** and document degradation
- **Complex layouts** with irregular line spacing

Our solution addresses these challenges through a multi-stage pipeline that achieves significant improvements in text recognition accuracy.

## 🏗️ Pipeline Architecture

```mermaid
graph TB
    subgraph FT ["🎓 Fine-Tuning Phase"]
        A[📝 Raw OCR Line]
        F[✅ Verified Ground<br/>Truth Line]
        B[🔧 Manual Alignment]
        C[📋 Aligned Line Pairs]
        D[🤖 ByT5 Fine-Tuning]
        M[💾 Trained ByT5<br/>Correction Model]
        
        A --> B
        F --> B
        B --> C
        C --> D
        D --> M
    end
    
    subgraph IP ["🚀 Inference Pipeline"]
        IMG[📄 New Manuscript<br/>Image]
        SEG[🔍 Kraken Line<br/>Segmentation]
        LINES[📝 Line Images]
        OCR[🤖 TrOCR Medieval<br/>Model]
        NOISY[📝 Raw OCR Output]
        CORR[✨ ByT5 Correction]
        FINAL[📖 Corrected<br/>Output]
        
        IMG --> SEG
        SEG --> LINES
        LINES --> OCR
        OCR --> NOISY
        NOISY --> CORR
        CORR --> FINAL
    end
    
    M -.-> CORR
    
    style A fill:#bbdefb,color:#000
    style F fill:#c8e6c9,color:#000
    style B fill:#fff59d,color:#000
    style C fill:#ffcc80,color:#000
    style D fill:#f8bbd9,color:#000
    style M fill:#a5d6a7,color:#000
    style IMG fill:#90caf9,color:#000
    style SEG fill:#81d4fa,color:#000
    style LINES fill:#e0e0e0,color:#000
    style OCR fill:#ffab91,color:#000
    style NOISY fill:#e0e0e0,color:#000
    style CORR fill:#ffcc80,color:#000
    style FINAL fill:#a5d6a7,color:#000
```

### Process Flow

#### Fine-Tuning Phase
1. **Data Preparation**: Raw OCR lines and verified ground truth texts are collected
2. **Manual Alignment**: OCR outputs are aligned with their corresponding ground truth
3. **Dataset Creation**: Aligned line pairs form the training dataset
4. **Model Training**: ByT5 model is fine-tuned on medieval OCR correction patterns

#### Inference Pipeline
1. **Line Segmentation**: Kraken automatically detects and segments text lines
2. **Initial OCR**: TrOCR processes each line with medieval-optimized model
3. **Error Correction**: Fine-tuned ByT5 model corrects OCR errors
4. **Text Assembly**: Individual corrected lines are combined into final output


## 📊 Dataset

Our dataset consists of **10,643 text line pairs** extracted from medieval manuscripts:

| Column | Description | Example |
|--------|-------------|---------|
| `line_id` | Unique identifier | `0033_033_line_30` |
| `image_path` | Path to line image | `/path/to/line.png` |
| `text` | Ground truth text | `Ius dei occultaret. la terza ut tetatis fa` |
| `ocr_prediction` | Raw OCR output | `Ius dei occultare. la terza un tẽtatis fa` |
| `page_id` | Source page identifier | `0033_033` |
| `line_number` | Line position in page | `30` |

### Data Statistics
- **Total lines**: 10,643
- **Average line length**: ~45 characters
- **Character Error Rate** (before correction): ~12.3%
- **Training/Validation/Test split**: 80%/10%/10%

The early printed books used for OCR and post-correction tasks originate from the [MAGIC digital archive](https://www.magic.unina.it), which provides open access to digitized manuscripts. Our training data was created by aligning OCR outputs with manually verified transcriptions based on these sources.

## 🚀 Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/medieval-ocr-correction.git
cd medieval-ocr-correction

# Install dependencies
pip install -r requirements.txt

# Install Kraken for line segmentation
pip install kraken
```

### Model Downloads

The pipeline will automatically download required models:
- **TrOCR Medieval**: `medieval-data/trocr-medieval-print`
- **ByT5 Base**: `google/byt5-small` (fine-tuned version included)

## ⚡ Quick Start

### Complete Pipeline

Process a medieval manuscript image through the entire pipeline:

```bash
python complete_ocr_pipeline.py \
    --image_path dataset/images/manuscript.jpg \
    --output_file corrected_text.txt
```

### Output Files
- `corrected_text.txt`: Final corrected text
- `detailed_results.txt`: Line-by-line comparison
- Individual line images (if `--keep_temp` flag used)

### Example Usage

```python
from complete_ocr_pipeline import process_complete_image

# Process image and get results
final_text, line_results = process_complete_image(
    image_path="path/to/manuscript.jpg",
    output_file="output.txt"
)

print(f"Processed {len(line_results)} lines")
print(f"Final text: {final_text[:100]}...")
```

## 🎯 Training

### Dataset Preparation

If you want to train with your own dataset:

```bash
# Prepare dataset in the required format
python prepare_dataset.py \
    --input_csv your_data.csv \
    --output_dir processed_dataset/
```

### ByT5 Fine-tuning

Train the correction model:

```bash
python Byt5_finetune.py \
    --dataset_path dataset/dataset.csv \
    --output_dir ./byt5-ocr-correction \
    --num_epochs 4 \
    --batch_size 2 \
    --learning_rate 5e-4
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_epochs` | 4 | Number of training epochs |
| `--batch_size` | 2 | Training batch size |
| `--learning_rate` | 5e-4 | Learning rate |
| `--max_length` | 128 | Maximum sequence length |
| `--warmup_steps` | 250 | Warmup steps for scheduler |

## 📈 Evaluation

### Metrics

The system is evaluated using standard OCR metrics:

- **Character Error Rate (CER)**: Character-level accuracy
- **BLEU Score**: Sequence-level similarity
- **Edit Distance**: Number of character edits required

### Running Evaluation

```bash
python evaluate_model.py \
    --model_path ./byt5-ocr-correction \
    --test_data dataset/test_split.csv
```

### Example Corrections

| Original OCR | Corrected Text | Ground Truth |
|--------------|----------------|--------------|
| `Ius dei occultare. la terza un tẽtatis fa` | `Ius dei occultaret. la terza ut tetatis fa` | `Ius dei occultaret. la terza ut tetatis fa` |
| `ti perlo peccato:cessi uolse esser tenta` | `ti per lo peccato: Cossi uolse esser tenta` | `ti per lo peccato: Cossi uolse esser tenta` |
| `Uctus ẽ iesus in desertũ a sai` | `Vctus est iesus in desertum a spi` | `Vctus est iesus in desertum a spi` |

## 📂 Repository Structure

```
medieval-ocr-correction/
├── 📁 dataset/
│   ├── images/                     # Input manuscript images
│   ├── lines/                      # Segmented line images
│   └── dataset_abbreviation_corrected.csv  # Training data
├── 📁 models/
│   └── byt5-ocr-correction/        # Fine-tuned ByT5 model
├── 📁 utils/
│   ├── segmentation_*.py           # Line segmentation utilities
│   ├── dataset.py                  # Dataset processing
│   └── evaluation.py               # Evaluation metrics
├── 📄 complete_ocr_pipeline.py     # Main pipeline script
├── 📄 Byt5_finetune.py            # Model training script
├── 📄 requirements.txt             # Dependencies
├── 📄 README.md                    # This file
└── 📄 LICENSE                      # License file
```

## 🙏 Acknowledgments

- **Kraken** OCR engine for line segmentation
- **TrOCR** team for the medieval manuscript model
- **Google** for the ByT5 architecture
- **MDPI Electronics** for publishing our research

## 📞 Contact

Feel free to contact me via [LinkedIn](https://www.linkedin.com/in/yahya-momtaz-601b34108/)
