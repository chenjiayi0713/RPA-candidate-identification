"""
Transformer Baseline for RPA Activity Classification
=====================================================
Uses sentence-transformers (all-MiniLM-L6-v2) to generate embeddings,
then trains LR/SVM classifiers under the same 10-fold CV protocol.
Also includes a distilBERT fine-tuning option.

Packages required (all already installed):
    torch, transformers, sentence-transformers, sklearn, numpy, scipy
"""

import numpy as np
from scipy import stats as scipy_stats
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pickle
import os
import json
import sys

# Use HuggingFace mirror for China access
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

RANDOM_STATE = 42
N_SPLITS = 10


def load_data(data_dir):
    """Load text/label data from pickle or raw txt files."""
    texts_path = os.path.join(data_dir, 'texts.pkl')
    labels_path = os.path.join(data_dir, 'labels.pkl')
    target_names_path = os.path.join(data_dir, 'target_names.pkl')

    if os.path.exists(texts_path):
        with open(texts_path, 'rb') as f:
            texts = pickle.load(f)
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
        with open(target_names_path, 'rb') as f:
            target_names = pickle.load(f)
    else:
        cato_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
        texts, labels, target_names = [], [], []
        for cnt_c, cato_file in enumerate(cato_files):
            target_names.append(cato_file.replace('.txt', ''))
            with open(os.path.join(data_dir, cato_file), encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
            last = ''
            for line in lines:
                if line == last:
                    continue
                texts.append(line)
                labels.append(cnt_c)
                last = line
        with open(texts_path, 'wb') as f:
            pickle.dump(texts, f)
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f)
        with open(target_names_path, 'wb') as f:
            pickle.dump(target_names, f)
    return texts, labels, target_names


def get_sbert_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2', batch_size=64):
    """
    Extract sentence embeddings using mean pooling over transformer hidden states.
    Equivalent to Sentence-BERT all-MiniLM-L6-v2: 384-dim, fast on CPU, ~80MB.
    Uses transformers library directly (avoids sentence-transformers version conflicts).
    Returns numpy array of shape (n_samples, 384).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading model: {model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_embeddings = []
    print(f"  Encoding {len(texts)} texts on {device} (batch_size={batch_size}) ...", flush=True)

    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=128,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            # Mean pooling: average all token embeddings (excluding padding)
            attention_mask = encoded['attention_mask'].unsqueeze(-1).float()
            token_embeddings = outputs.last_hidden_state
            mean_embeddings = (token_embeddings * attention_mask).sum(1) / attention_mask.sum(1)
            all_embeddings.append(mean_embeddings.cpu().numpy())

        if (i // batch_size + 1) % 20 == 0:
            print(f"    {i+len(batch_texts)}/{len(texts)}", flush=True)

    embeddings = np.vstack(all_embeddings)
    print(f"  Embeddings shape: {embeddings.shape}", flush=True)
    return embeddings


def get_bert_cls_embeddings(texts, model_name='distilbert-base-uncased', batch_size=32):
    """
    Extract [CLS] token embeddings from a BERT model.
    distilbert-base-uncased: 768-dim, ~260MB, faster than BERT-base.
    Returns numpy array of shape (n_samples, 768).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading model: {model_name} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_embeddings = []
    print(f"  Encoding {len(texts)} texts on {device} ...", flush=True)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128,
                           return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)
        if (i // batch_size) % 50 == 0:
            print(f"    {i}/{len(texts)}", flush=True)
    embeddings = np.vstack(all_embeddings)
    print(f"  Embeddings shape: {embeddings.shape}", flush=True)
    return embeddings


def run_kfold_with_embeddings(embeddings, labels, target_names, data_name):
    """Run 10-fold CV with LR and SVM on pre-computed embeddings."""
    labels = np.array(labels)

    skf = model_selection.StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    results = {'LR': {}, 'SVM': {}}
    for name in results:
        results[name] = {
            'f1_macro': [], 'f1_weighted': [], 'accuracy': [],
            'per_class': [], 'all_y_true': [], 'all_y_pred': []
        }

    fold = 0
    for train_idx, valid_idx in skf.split(embeddings, labels):
        fold += 1
        train_x, valid_x = embeddings[train_idx], embeddings[valid_idx]
        train_y, valid_y = labels[train_idx], labels[valid_idx]

        classifiers = {
            'LR': LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
            'SVM': LinearSVC(C=1.0, random_state=RANDOM_STATE, dual=False, max_iter=2000),
        }

        for name, clf in classifiers.items():
            clf.fit(train_x, train_y)
            preds = clf.predict(valid_x)

            results[name]['all_y_true'].extend(valid_y.tolist())
            results[name]['all_y_pred'].extend(preds.tolist())

            results[name]['accuracy'].append(metrics.accuracy_score(valid_y, preds))
            results[name]['f1_macro'].append(
                metrics.f1_score(valid_y, preds, average='macro', zero_division=0))
            results[name]['f1_weighted'].append(
                metrics.f1_score(valid_y, preds, average='weighted', zero_division=0))
            results[name]['per_class'].append(
                metrics.precision_recall_fscore_support(valid_y, preds,
                                                         labels=sorted(set(labels)),
                                                         zero_division=0))

        print(f"  Fold {fold}/{N_SPLITS} complete", flush=True)

    # Confusion matrices
    for name in results:
        all_y_true = results[name].pop('all_y_true')
        all_y_pred = results[name].pop('all_y_pred')
        results[name]['confusion_matrix'] = metrics.confusion_matrix(all_y_true, all_y_pred)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  {data_name} — Transformer Embedding Results")
    print(f"{'='*70}")
    for name in ['LR', 'SVM']:
        r = results[name]
        print(f"\n  [{name}]")
        print(f"    Macro-F1:    {np.mean(r['f1_macro']):.4f} ± {np.std(r['f1_macro']):.4f}")
        print(f"    Weighted-F1: {np.mean(r['f1_weighted']):.4f} ± {np.std(r['f1_weighted']):.4f}")
        print(f"    Accuracy:    {np.mean(r['accuracy']):.4f} ± {np.std(r['accuracy']):.4f}")

        # Per-class
        n_classes = len(target_names)
        print(f"    Per-class F1 (averaged over 10 folds):")
        for i, cn in enumerate(target_names):
            avg_f = np.mean([pc[2][i] for pc in r['per_class']])
            print(f"      {cn}: {avg_f:.4f}")

        # CM
        print(f"    Confusion Matrix:")
        cm = r['confusion_matrix']
        header = "        " + " ".join(f"{n:>8}" for n in target_names)
        print(header)
        for i, cn in enumerate(target_names):
            print(f"      {cn:>6} " + " ".join(f"{cm[i][j]:>8}" for j in range(len(target_names))))

    return results


def run_zero_shot(texts, labels, target_names, model_name='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'):
    """
    Zero-shot classification using a pre-trained NLI model.
    Maps activity text to candidate labels: manual, user-assisted, automated.
    This approach requires no training data — purely zero-shot.
    """
    from transformers import pipeline
    import torch

    # Map numeric labels to natural language descriptions
    label_descriptions = {
        0: "a manual task requiring human judgment",
        2: "a user-assisted task involving system interaction",
        3: "an automated system task with no human involvement",
    }

    candidate_labels = [label_descriptions[i] for i in sorted(label_descriptions.keys())]
    label_map = {0: 0, 2: 2, 3: 3}  # sorted keys → original labels

    print(f"  Loading zero-shot classifier: {model_name} ...", flush=True)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model=model_name, device=device)

    # Run on a sample (full run would be slow on CPU for 2262 examples)
    sample_size = min(200, len(texts))
    indices = np.random.RandomState(RANDOM_STATE).choice(len(texts), sample_size, replace=False)
    sample_texts = [texts[i] for i in indices]
    sample_labels = [labels[i] for i in indices]

    print(f"  Running zero-shot on {sample_size} samples ...", flush=True)
    predictions = []
    for i, text in enumerate(sample_texts):
        result = classifier(text, candidate_labels, multi_label=False)
        pred_label_idx = candidate_labels.index(result['labels'][0])
        predictions.append(label_map[pred_label_idx])
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{sample_size}", flush=True)

    # Evaluate
    print(f"\n  Zero-Shot Results (n={sample_size}):")
    print(f"    Accuracy: {metrics.accuracy_score(sample_labels, predictions):.4f}")
    for avg_name in ['macro', 'weighted']:
        f1 = metrics.f1_score(sample_labels, predictions, average=avg_name, zero_division=0)
        print(f"    {avg_name}-F1: {f1:.4f}")
    print(f"    Classification Report:")
    print(metrics.classification_report(sample_labels, predictions,
                                         target_names=target_names, zero_division=0))

    return predictions, sample_labels


def save_results(results, target_names, output_path):
    """Save summary JSON."""
    summary = {}
    for name in results:
        r = results[name]
        summary[name] = {
            'f1_macro': {'mean': float(np.mean(r['f1_macro'])),
                         'std': float(np.std(r['f1_macro']))},
            'f1_weighted': {'mean': float(np.mean(r['f1_weighted'])),
                            'std': float(np.std(r['f1_weighted']))},
            'accuracy': {'mean': float(np.mean(r['accuracy'])),
                         'std': float(np.std(r['accuracy']))},
            'confusion_matrix': r['confusion_matrix'].tolist(),
            'per_class': {},
        }
        for i, cn in enumerate(target_names):
            summary[name]['per_class'][cn] = {
                'precision': float(np.mean([pc[0][i] for pc in r['per_class']])),
                'recall': float(np.mean([pc[1][i] for pc in r['per_class']])),
                'f1': float(np.mean([pc[2][i] for pc in r['per_class']])),
            }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './data/Full/All_feature/'
    method = sys.argv[2] if len(sys.argv) > 2 else 'sbert'

    texts, labels, target_names = load_data(data_dir)
    print(f"Data: {len(texts)} texts, {len(target_names)} classes: {target_names}")

    if method == 'sbert':
        # === Method A: Sentence-BERT embeddings + LR/SVM ===
        embeddings = get_sbert_embeddings(texts)
        results = run_kfold_with_embeddings(embeddings, labels, target_names,
                                            data_name="SBERT (all-MiniLM-L6-v2)")
        save_results(results, target_names,
                     os.path.join(data_dir, 'transformer_sbert_results.json'))

    elif method == 'bert':
        # === Method B: DistilBERT [CLS] embeddings + LR/SVM ===
        embeddings = get_bert_cls_embeddings(texts)
        results = run_kfold_with_embeddings(embeddings, labels, target_names,
                                            data_name="DistilBERT [CLS]")
        save_results(results, target_names,
                     os.path.join(data_dir, 'transformer_bert_results.json'))

    elif method == 'zeroshot':
        # === Method C: Zero-shot NLI classifier ===
        run_zero_shot(texts, labels, target_names)

    else:
        print(f"Unknown method: {method}. Choose: sbert, bert, zeroshot")
