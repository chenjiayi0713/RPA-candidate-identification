import numpy as np
from scipy import stats as scipy_stats
from sklearn import model_selection, metrics, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
import json


RANDOM_STATE = 42
N_SPLITS = 10


def preprocess(data_dir):
    """Load pre-processed text and label data from data_dir."""
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
        return texts, labels, target_names

    # First run: read raw .txt files and build pickle cache
    cato_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    texts = []
    labels = []
    target_names = []

    for cnt_c, cato_file in enumerate(sorted(cato_files)):
        target_names.append(cato_file.replace('.txt', ''))

        file_path = os.path.join(data_dir, cato_file)
        with open(file_path, encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Deduplicate consecutive lines
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


def run_kfold_cv(raw_dir, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """
    Run stratified k-fold cross-validation with proper TF-IDF isolation.
    TF-IDF is fitted on training fold only, then transforms both train and test folds.
    """
    texts, labels, target_names = preprocess(raw_dir)
    labels = np.array(labels)

    skf = model_selection.StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    # Store per-fold results for each classifier
    results = {
        'LR': {'accuracy': [], 'precision_macro': [], 'recall_macro': [], 'f1_macro': [],
               'precision_weighted': [], 'recall_weighted': [], 'f1_weighted': [],
               'per_class': [], 'confusion_matrix': None, 'all_y_true': [], 'all_y_pred': []},
        'SVM': {'accuracy': [], 'precision_macro': [], 'recall_macro': [], 'f1_macro': [],
                'precision_weighted': [], 'recall_weighted': [], 'f1_weighted': [],
                'per_class': [], 'confusion_matrix': None, 'all_y_true': [], 'all_y_pred': []},
        'NB': {'accuracy': [], 'precision_macro': [], 'recall_macro': [], 'f1_macro': [],
               'precision_weighted': [], 'recall_weighted': [], 'f1_weighted': [],
               'per_class': [], 'confusion_matrix': None, 'all_y_true': [], 'all_y_pred': []},
    }

    fold = 0
    for train_idx, valid_idx in skf.split(texts, labels):
        fold += 1
        train_texts = [texts[i] for i in train_idx]
        valid_texts = [texts[i] for i in valid_idx]
        train_labels = labels[train_idx]
        valid_labels = labels[valid_idx]

        # TF-IDF: fit ONLY on training fold, transform both
        tfidf_vect = TfidfVectorizer(
            min_df=3, max_df=0.5, max_features=None,
            ngram_range=(1, 2), use_idf=True, smooth_idf=True
        )
        train_x = tfidf_vect.fit_transform(train_texts)
        valid_x = tfidf_vect.transform(valid_texts)

        # Classifiers
        classifiers = {
            'LR': LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                     random_state=random_state),
            'SVM': LinearSVC(random_state=random_state, dual=False, max_iter=2000),
            'NB': MultinomialNB(),
        }

        for name, clf in classifiers.items():
            clf.fit(train_x, train_labels)
            preds = clf.predict(valid_x)

            # Collect all predictions for confusion matrix
            results[name]['all_y_true'].extend(valid_labels.tolist())
            results[name]['all_y_pred'].extend(preds.tolist())

            # Aggregate metrics
            results[name]['accuracy'].append(
                metrics.accuracy_score(valid_labels, preds)
            )
            results[name]['precision_macro'].append(
                metrics.precision_score(valid_labels, preds, average='macro', zero_division=0)
            )
            results[name]['recall_macro'].append(
                metrics.recall_score(valid_labels, preds, average='macro', zero_division=0)
            )
            results[name]['f1_macro'].append(
                metrics.f1_score(valid_labels, preds, average='macro', zero_division=0)
            )
            results[name]['precision_weighted'].append(
                metrics.precision_score(valid_labels, preds, average='weighted', zero_division=0)
            )
            results[name]['recall_weighted'].append(
                metrics.recall_score(valid_labels, preds, average='weighted', zero_division=0)
            )
            results[name]['f1_weighted'].append(
                metrics.f1_score(valid_labels, preds, average='weighted', zero_division=0)
            )

            # Per-class metrics
            per_class = metrics.precision_recall_fscore_support(
                valid_labels, preds, labels=sorted(set(labels)), zero_division=0
            )
            results[name]['per_class'].append(per_class)

        print(f"  Fold {fold}/{n_splits} complete", flush=True)

    # Build confusion matrices from all predictions
    for name in results:
        all_y_true = results[name]['all_y_true']
        all_y_pred = results[name]['all_y_pred']
        results[name]['confusion_matrix'] = metrics.confusion_matrix(all_y_true, all_y_pred)
        # Clean up for serialization
        del results[name]['all_y_true']
        del results[name]['all_y_pred']

    return results, target_names


def print_summary(results, target_names):
    """Print formatted summary with mean ± std and significance tests."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (10-fold CV, mean ± std)")
    print("=" * 80)

    classifier_names = list(results.keys())

    for name in classifier_names:
        r = results[name]
        print(f"\n--- {name} ---")
        print(f"  Accuracy:           {np.mean(r['accuracy']):.4f} ± {np.std(r['accuracy']):.4f}")
        print(f"  Precision (macro):  {np.mean(r['precision_macro']):.4f} ± {np.std(r['precision_macro']):.4f}")
        print(f"  Recall (macro):     {np.mean(r['recall_macro']):.4f} ± {np.std(r['recall_macro']):.4f}")
        print(f"  F1 (macro):         {np.mean(r['f1_macro']):.4f} ± {np.std(r['f1_macro']):.4f}")
        print(f"  Precision (weight): {np.mean(r['precision_weighted']):.4f} ± {np.std(r['precision_weighted']):.4f}")
        print(f"  Recall (weight):    {np.mean(r['recall_weighted']):.4f} ± {np.std(r['recall_weighted']):.4f}")
        print(f"  F1 (weighted):      {np.mean(r['f1_weighted']):.4f} ± {np.std(r['f1_weighted']):.4f}")

        # Per-class averages
        n_classes = len(target_names)
        print(f"\n  Per-class metrics (averaged over 10 folds):")
        print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Support':<10}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        for i, class_name in enumerate(target_names):
            avg_p = np.mean([pc[0][i] for pc in r['per_class']])
            avg_r = np.mean([pc[1][i] for pc in r['per_class']])
            avg_f = np.mean([pc[2][i] for pc in r['per_class']])
            avg_s = np.mean([pc[3][i] for pc in r['per_class']])
            print(f"  {class_name:<15} {avg_p:<12.4f} {avg_r:<12.4f} {avg_f:<12.4f} {avg_s:<10.1f}")

        # Confusion matrix
        print(f"\n  Confusion Matrix (aggregated across all folds):")
        print(f"  {'':>8} " + " ".join(f"{n:>8}" for n in target_names))
        cm = r['confusion_matrix']
        for i, class_name in enumerate(target_names):
            print(f"  {class_name:>8} " + " ".join(f"{cm[i][j]:>8}" for j in range(len(target_names))))

    # Pairwise significance tests on macro-F1
    print(f"\n--- Paired t-tests on macro-F1 (10 folds) ---")
    for i, name_a in enumerate(classifier_names):
        for name_b in classifier_names[i+1:]:
            t_stat, p_val = scipy_stats.ttest_rel(
                results[name_a]['f1_macro'], results[name_b]['f1_macro']
            )
            sig = "SIGNIFICANT" if p_val < 0.05 else "NOT significant"
            print(f"  {name_a} vs {name_b}: t={t_stat:.3f}, p={p_val:.4f} ({sig})")

    print("\n" + "=" * 80)


def save_results(results, target_names, raw_dir):
    """Save results as JSON and model files."""
    # Save summary JSON
    summary = {}
    for name in results:
        r = results[name]
        summary[name] = {
            'accuracy': {'mean': float(np.mean(r['accuracy'])), 'std': float(np.std(r['accuracy']))},
            'precision_macro': {'mean': float(np.mean(r['precision_macro'])), 'std': float(np.std(r['precision_macro']))},
            'recall_macro': {'mean': float(np.mean(r['recall_macro'])), 'std': float(np.std(r['recall_macro']))},
            'f1_macro': {'mean': float(np.mean(r['f1_macro'])), 'std': float(np.std(r['f1_macro']))},
            'precision_weighted': {'mean': float(np.mean(r['precision_weighted'])), 'std': float(np.std(r['precision_weighted']))},
            'recall_weighted': {'mean': float(np.mean(r['recall_weighted'])), 'std': float(np.std(r['recall_weighted']))},
            'f1_weighted': {'mean': float(np.mean(r['f1_weighted'])), 'std': float(np.std(r['f1_weighted']))},
            'per_class': {},
            'confusion_matrix': r['confusion_matrix'].tolist(),
            'target_names': target_names,
        }
        # Per-class means
        n_classes = len(target_names)
        for i, class_name in enumerate(target_names):
            summary[name]['per_class'][class_name] = {
                'precision': float(np.mean([pc[0][i] for pc in r['per_class']])),
                'recall': float(np.mean([pc[1][i] for pc in r['per_class']])),
                'f1': float(np.mean([pc[2][i] for pc in r['per_class']])),
            }

    json_path = os.path.join(raw_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {json_path}")

    # Save a single trained model per classifier (fit on ALL data, for reference)
    texts, labels, target_names_local = preprocess(raw_dir)
    tfidf_all = TfidfVectorizer(min_df=3, max_df=0.5, ngram_range=(1, 2), use_idf=True, smooth_idf=True)
    X_all = tfidf_all.fit_transform(texts)

    classifiers = {
        'LR': LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE),
        'SVM': LinearSVC(random_state=RANDOM_STATE, dual=False, max_iter=2000),
        'NB': MultinomialNB(),
    }
    for name, clf in classifiers.items():
        clf.fit(X_all, labels)
        model_path = os.path.join(raw_dir, f'{name}.model')
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)


def run_experiment(raw_dir):
    """Run full experiment for a given data directory."""
    print(f"\n{'#' * 60}")
    print(f"# Experiment: {raw_dir}")
    print(f"{'#' * 60}")

    results, target_names = run_kfold_cv(raw_dir)
    print_summary(results, target_names)
    save_results(results, target_names, raw_dir)

    return results


if __name__ == '__main__':
    import sys

    # Allow command-line override: python model.py ./data/Full/No_feature/
    if len(sys.argv) > 1:
        run_experiment(sys.argv[1])
    else:
        experiments = [
            './data/Full/No_feature/',
            './data/Full/Single_feature/verb/',
            './data/Full/Single_feature/object_/',
            './data/Full/Single_feature/process_volume/',
            './data/Full/Single_feature/repetitive/',
            './data/Full/All_feature/',
            './data/Old/All_feature/',
            './data/New/All_feature/',
        ]

        for exp in experiments:
            if os.path.isdir(exp):
                run_experiment(exp)
            else:
                print(f"Skipping {exp} — directory not found (run data_process.py first)")

        print("\n" + "=" * 80)
        print("All experiments complete!")
        print("=" * 80)
