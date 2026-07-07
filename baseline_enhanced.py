"""
Enhanced Baseline + Cross-Dataset Evaluation
- Strong character+word n-gram TF-IDF (approximates subword tokenization)
- Cross-dataset generalization: Train on synthetic → Test on human-authored
- No PyTorch/GPU required
"""

import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, metrics
from sklearn.feature_extraction.text import TfidfVectorizer

RANDOM_STATE = 42
N_SPLITS = 10


def load_activities(data_dir):
    """Load activity texts from feature files, returning just the activity text."""
    texts = []
    labels_list = []
    target_names = []

    txt_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
    for cnt_c, fname in enumerate(txt_files):
        target_names.append(fname.replace('.txt', ''))
        file_path = os.path.join(data_dir, fname)
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Extract activity text (first token before space)
                    parts = line.split(' ', 1)
                    activity = parts[0] if parts else line
                    texts.append(activity)
                    labels_list.append(cnt_c)
    return texts, np.array(labels_list), target_names


def load_activities_from_csv(csv_path, text_col=3, label_col=-1):
    """Load activity texts directly from CSV files."""
    import csv
    texts = []
    labels = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if len(line) <= max(text_col, label_col):
                continue
            activity = line[text_col].strip()
            gs = line[label_col].strip()
            if gs and gs != '?':
                texts.append(activity)
                labels.append(int(float(gs)) if '.' in gs else int(gs))
    return texts, np.array(labels)


def run_char_ngram_kfold(data_dir, n_splits=N_SPLITS):
    """K-fold CV with character n-gram TF-IDF (strong subword baseline)."""
    texts, labels, target_names = load_activities(data_dir)
    labels = np.array(labels)

    # Map labels to 0,1,2 for stratification
    unique_labels = sorted(set(labels))
    label_map = {orig: i for i, orig in enumerate(unique_labels)}
    mapped_labels = np.array([label_map[l] for l in labels])
    reverse_map = {i: orig for orig, i in label_map.items()}

    print(f"\n{'='*60}")
    print(f"Char-ngram Baseline: {data_dir}")
    print(f"  Samples: {len(texts)}, Classes: {target_names}")
    print(f"{'='*60}")

    skf = model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    all_f1_macro = []
    all_f1_weighted = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(texts, mapped_labels)):
        fold += 1
        train_texts = [texts[i] for i in train_idx]
        valid_texts = [texts[i] for i in valid_idx]
        train_lbl = mapped_labels[train_idx]
        valid_lbl = mapped_labels[valid_idx]

        # Character n-grams (3-5) + word n-grams (1-2) = strong subword baseline
        tfidf = TfidfVectorizer(
            min_df=2, max_df=0.8,
            ngram_range=(1, 2),          # word n-grams
            analyzer='char_wb',           # character n-grams within word boundaries
            max_features=5000,
            use_idf=True, smooth_idf=True
        )
        train_x = tfidf.fit_transform(train_texts)
        valid_x = tfidf.transform(valid_texts)

        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=RANDOM_STATE)
        clf.fit(train_x, train_lbl)
        preds = clf.predict(valid_x)

        all_f1_macro.append(metrics.f1_score(valid_lbl, preds, average='macro', zero_division=0))
        all_f1_weighted.append(metrics.f1_score(valid_lbl, preds, average='weighted', zero_division=0))
        all_y_true.extend(valid_lbl.tolist())
        all_y_pred.extend(preds.tolist())

        print(f"  Fold {fold}/{N_SPLITS} M-F1: {all_f1_macro[-1]:.4f} W-F1: {all_f1_weighted[-1]:.4f}", flush=True)

    print(f"\n  Char-ngram Summary:")
    print(f"    Macro-F1:    {np.mean(all_f1_macro):.4f} ± {np.std(all_f1_macro):.4f}")
    print(f"    Weighted-F1: {np.mean(all_f1_weighted):.4f} ± {np.std(all_f1_weighted):.4f}")

    cm = metrics.confusion_matrix(all_y_true, all_y_pred)
    print(f"  Confusion Matrix:\n{cm}")

    per_class = metrics.precision_recall_fscore_support(all_y_true, all_y_pred, zero_division=0)
    mapped_names = [target_names[reverse_map[i]] if i in reverse_map else str(i)
                    for i in range(len(per_class[0]))]
    for i, name in enumerate(mapped_names):
        print(f"    {name}: P={per_class[0][i]:.4f} R={per_class[1][i]:.4f} F1={per_class[2][i]:.4f} S={per_class[3][i]}")

    results = {
        'model': 'Char-ngram TF-IDF + LR',
        'f1_macro_mean': float(np.mean(all_f1_macro)),
        'f1_macro_std': float(np.std(all_f1_macro)),
        'f1_weighted_mean': float(np.mean(all_f1_weighted)),
        'f1_weighted_std': float(np.std(all_f1_weighted)),
        'confusion_matrix': cm.tolist(),
        'target_names': target_names,
        'per_class': {},
    }
    for i, name in enumerate(mapped_names):
        results['per_class'][name] = {
            'precision': float(per_class[0][i]),
            'recall': float(per_class[1][i]),
            'f1': float(per_class[2][i]),
            'support': int(per_class[3][i]),
        }

    json_path = os.path.join(data_dir, 'results_char_ngram.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def cross_dataset_train_test(train_dir, test_dir, label):
    """Train on one dataset, test on another — evaluates generalization."""
    print(f"\n{'='*60}")
    print(f"Cross-Dataset: Train {train_dir} → Test {test_dir} ({label})")
    print(f"{'='*60}")

    train_texts, train_labels, train_names = load_activities(train_dir)
    test_texts, test_labels, test_names = load_activities(test_dir)

    # Map labels to 0,1,2
    unique_lbls = sorted(set(train_labels) | set(test_labels))
    lbl_map = {orig: i for i, orig in enumerate(unique_lbls)}
    train_lbl_mapped = np.array([lbl_map[l] for l in train_labels])
    test_lbl_mapped = np.array([lbl_map[l] for l in test_labels])

    print(f"  Train: {len(train_texts)} samples, Test: {len(test_texts)} samples")
    print(f"  Train class dist: {dict(zip(*np.unique(train_lbl_mapped, return_counts=True)))}")
    print(f"  Test class dist:  {dict(zip(*np.unique(test_lbl_mapped, return_counts=True)))}")

    # Char-ngram TF-IDF
    tfidf = TfidfVectorizer(
        min_df=2, max_df=0.8,
        ngram_range=(1, 2), analyzer='char_wb',
        max_features=5000, use_idf=True, smooth_idf=True
    )
    train_x = tfidf.fit_transform(train_texts)
    test_x = tfidf.transform(test_texts)

    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=RANDOM_STATE)
    clf.fit(train_x, train_lbl_mapped)
    preds = clf.predict(test_x)

    results = {
        'label': label,
        'accuracy': float(metrics.accuracy_score(test_lbl_mapped, preds)),
        'precision_macro': float(metrics.precision_score(test_lbl_mapped, preds, average='macro', zero_division=0)),
        'recall_macro': float(metrics.recall_score(test_lbl_mapped, preds, average='macro', zero_division=0)),
        'f1_macro': float(metrics.f1_score(test_lbl_mapped, preds, average='macro', zero_division=0)),
        'f1_weighted': float(metrics.f1_score(test_lbl_mapped, preds, average='weighted', zero_division=0)),
        'confusion_matrix': metrics.confusion_matrix(test_lbl_mapped, preds).tolist(),
        'target_names': train_names,
        'per_class': {},
    }

    per_class = metrics.precision_recall_fscore_support(test_lbl_mapped, preds, zero_division=0)
    for i, name in enumerate(train_names):
        if i < len(per_class[0]):
            results['per_class'][name] = {
                'precision': float(per_class[0][i]),
                'recall': float(per_class[1][i]),
                'f1': float(per_class[2][i]),
                'support': int(per_class[3][i]),
            }

    print(f"\n  Cross-Dataset Results ({label}):")
    print(f"    Accuracy:    {results['accuracy']:.4f}")
    print(f"    Macro-F1:    {results['f1_macro']:.4f}")
    print(f"    Weighted-F1: {results['f1_weighted']:.4f}")
    print(f"  Confusion Matrix:\n{np.array(results['confusion_matrix'])}")
    for name, pc in results['per_class'].items():
        print(f"    {name}: P={pc['precision']:.4f} R={pc['recall']:.4f} F1={pc['f1']:.4f}")

    json_path = os.path.join(test_dir, f'results_cross_{label}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    import sys
    base = './data/'

    if len(sys.argv) > 1:
        run_char_ngram_kfold(sys.argv[1])
    else:
        print("=" * 70)
        print("ENHANCED BASELINE + CROSS-DATASET EVALUATION")
        print("=" * 70)

        # 1. Char-ngram baseline on each dataset
        for ds in ['Full/No_feature', 'New/All_feature', 'Old/All_feature']:
            path = os.path.join(base, ds)
            if os.path.isdir(path):
                run_char_ngram_kfold(path)

        # 2. Cross-dataset evaluation
        print("\n" + "=" * 70)
        print("CROSS-DATASET GENERALIZATION")
        print("=" * 70)

        new_dir = os.path.join(base, 'New/All_feature')
        old_dir = os.path.join(base, 'Old/All_feature')

        if os.path.isdir(new_dir) and os.path.isdir(old_dir):
            cross_dataset_train_test(new_dir, old_dir, 'New_to_Old')
            cross_dataset_train_test(old_dir, new_dir, 'Old_to_New')

        print("\n" + "=" * 70)
        print("All enhanced experiments complete!")
        print("=" * 70)
