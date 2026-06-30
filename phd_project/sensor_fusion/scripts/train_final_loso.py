#!/usr/bin/env python3
"""Train final task-type detectors using LOSO and export per-fold artifacts.

Saves per-fold metrics, per-class reports, confusion matrices and the final
model trained on the full dataset for each candidate algorithm.
"""
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (f1_score, balanced_accuracy_score, accuracy_score,
                             classification_report, confusion_matrix)
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def get_feature_columns(df):
    exclude = {"subject_id", "task_name", "task_file", "split", "window_idx",
               "start_idx", "end_idx", "n_samples", "pseudo_label",
               "eeg_features_5s__split", "n_modalities_present"}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def build_classifiers():
    cls = {
        "logreg": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "rf": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    }
    if HAS_XGB:
        cls["xgboost"] = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    return cls


def save_dict_csv(d, path: Path):
    df = pd.DataFrame([d])
    df.to_csv(path, index=False)


def run_loso(dataset_path, out_root):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path, low_memory=False)
    if 'pseudo_label' not in df.columns:
        raise RuntimeError('expected column pseudo_label in dataset')

    subjects = sorted(df['subject_id'].unique())
    feature_cols = get_feature_columns(df)
    print(f"Found {len(subjects)} subjects, {len(feature_cols)} features")

    classifiers = build_classifiers()

    summary_rows = []

    for name, clf in classifiers.items():
        print('Running LOSO for', name)
        model_dir = out_root / name
        model_dir.mkdir(parents=True, exist_ok=True)
        per_fold_dir = model_dir / 'folds'
        per_fold_dir.mkdir(exist_ok=True)

        fold_metrics = []

        for subj in subjects:
            train = df[df['subject_id'] != subj]
            test = df[df['subject_id'] == subj]
            if test.empty:
                print('skipping subject', subj, 'no test rows')
                continue

            X_train = train[feature_cols].values
            y_train = train['pseudo_label'].values
            X_test = test[feature_cols].values
            y_test = test['pseudo_label'].values

            pipe = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), clf)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            m = {
                'model': name,
                'subject_id': int(subj),
                'n_test': int(len(y_test)),
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'macro_f1': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
                'balanced_acc': float(balanced_accuracy_score(y_test, y_pred)),
            }

            # per-class report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(df['pseudo_label'])))

            save_dict_csv(m, per_fold_dir / f'fold_{int(subj):02d}.csv')
            pd.DataFrame(report).T.to_csv(per_fold_dir / f'fold_{int(subj):02d}_per_class.csv')
            pd.DataFrame(cm, index=sorted(np.unique(df['pseudo_label'])), columns=sorted(np.unique(df['pseudo_label']))).to_csv(per_fold_dir / f'fold_{int(subj):02d}_confusion.csv')

            fold_metrics.append(m)

        # save per-model aggregated fold metrics
        pd.DataFrame(fold_metrics).to_csv(model_dir / 'fold_metrics.csv', index=False)

        # train final model on full dataset and save
        full_pipe = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), clf)
        full_pipe.fit(df[feature_cols].values, df['pseudo_label'].values)
        joblib.dump(full_pipe, model_dir / 'final_model.joblib')

        # baseline summary
        if fold_metrics:
            dfm = pd.DataFrame(fold_metrics)
            summary_rows.append({
                'model': name,
                'n_folds': int(len(dfm)),
                'macro_f1_mean': float(dfm['macro_f1'].mean()),
                'macro_f1_std': float(dfm['macro_f1'].std()),
                'balanced_acc_mean': float(dfm['balanced_acc'].mean()),
                'balanced_acc_std': float(dfm['balanced_acc'].std()),
            })

    pd.DataFrame(summary_rows).to_csv(out_root / 'benchmark_summary_from_loso.csv', index=False)
    print('Done. Outputs written to', out_root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='/home/g0amer/Desktop/thesis/research_outputs/fusion/v1_fusion/fusion_dataset.csv')
    p.add_argument('--out', default='/home/g0amer/Desktop/thesis/research_outputs/fusion_training/final_detector')
    args = p.parse_args()
    run_loso(args.dataset, args.out)


if __name__ == '__main__':
    main()
