# Code Package

## Paper

**Title**: Automatic Candidates Identification for Robotic Process Automation
based on Textual Process Descriptions

---

## Environment Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. For users in mainland China, set HuggingFace mirror:
set HF_ENDPOINT=https://hf-mirror.com          # Windows CMD
export HF_ENDPOINT=https://hf-mirror.com        # Linux/Mac/Git Bash
```

**Requirements**: Python 3.9+, 8 GB RAM (16 GB recommended for transformer baseline).

---

## File Overview

| File | Description |
|------|-------------|
| `data_process.py` | Feature engineering from CSV datasets |
| `model.py` | Main experiment: 10-fold CV, NB/LR/SVM, per-class metrics, stats |
| `baseline_enhanced.py` | Char-ngram baseline + cross-dataset generalization |
| `transformer_baseline.py` | SBERT embedding baseline |
| `data/` | Pre-generated feature files (ready to use) |
| `datasets/` | Training CSVs and gold standards |
| `textual_descriptions/` | Source process description texts |

---

## Dataset Summary

| Dataset | Source Texts | Labeled Activities | Classes (0/2/3) |
|---------|-------------|-------------------|-------------------|
| Old | 33 human-authored | 424 | 250 / 153 / 21 |
| New | 150 NLG-generated | 1,837 | 989 / 677 / 171 |
| Full | 183 (combined) | 2,262 | 1,240 / 830 / 192 |

**Note**: The Old dataset originally contained 47 textual process descriptions.
14 descriptions (Model10-1 through Model10-14) were excluded because they were
produced by machine translation software. The remaining 33 human-authored texts
yield 424 labeled activities.

---

## Quick Start

```bash
# Step 1: (Optional) Regenerate feature files from CSV
python data_process.py

# Step 2: Run main experiments (8 configurations)
python model.py

# Step 3: Run modern baselines
python baseline_enhanced.py
python transformer_baseline.py ./data/Full/No_feature/ sbert
```

---

## Step-by-Step Details

### Step 1: Feature Generation (`data_process.py`)

Generates `.txt` feature files from the CSV datasets in `datasets/`.
Pre-generated files are already provided in `data/`.

```
data/
├── Full/
│   ├── No_feature/          # Activity text only (baseline)
│   ├── All_feature/         # Activity + verb + object + volume + repetitive
│   └── Single_feature/
│       ├── verb/
│       ├── object_/
│       ├── process_volume/
│       └── repetitive/
├── New/All_feature/         # Synthetic dataset
└── Old/All_feature/         # Human-authored dataset
```

### Step 2: Main Experiments (`model.py`)

Runs all 8 configurations with stratified 10-fold CV (random_state=42).
TF-IDF is fitted within each training fold (no leakage).
Reports mean +/- std and paired t-tests.
Saves `results.json` and `.model` files in each data directory.

### Step 3a: Char-ngram Baseline (`baseline_enhanced.py`)

Character n-gram TF-IDF (3-5 char within word boundaries) + Logistic Regression.
Also runs cross-dataset evaluation (New->Old and Old->New).

### Step 3b: SBERT Baseline (`transformer_baseline.py`)

Uses all-MiniLM-L6-v2 (384-dim) via mean pooling over transformer hidden states.
First run downloads ~80 MB model. Set HF_ENDPOINT for mainland China access.

---

## Expected Results (10-fold CV, random_state=42)

### Word TF-IDF (model.py)

| Method | Macro-F1 | Class-3 F1 |
|--------|----------|------------|
| SVM (no features) | ~0.725 | ~0.421 |
| SVM (all features) | ~0.786 | ~0.575 |

### Modern Baselines

| Method | Macro-F1 | Class-3 F1 |
|--------|----------|------------|
| Char-ngram TF-IDF + LR | ~0.855 | ~0.864 |
| SBERT + SVM | ~0.865 | ~0.709 |

### Cross-Dataset Generalization

| Train -> Test | Macro-F1 | Class-3 F1 |
|--------------|----------|------------|
| New -> Old | ~0.480 | ~0.000 |
| Old -> New | ~0.526 | ~0.034 |

---

## Changes from Original Submission

| Item | Original | Revised |
|------|----------|---------|
| Features | 6 (incl. leaked) | 4 (leakage removed) |
| CV protocol | Single split, no seed | Stratified 10-fold, seed=42 |
| TF-IDF fitting | All data | Per-fold training only |
| Metrics | Single values | Mean +/- std + t-tests |
| Per-class results | Not reported | P/R/F1 per class + confusion matrix |
| Modern baselines | None | Char-ngram + SBERT |
