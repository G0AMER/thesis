#!/usr/bin/env python
# coding: utf-8

# # Final Task-Type Detector: LOSO Comparison on Full Dataset
# This notebook loads the full fusion dataset, runs leave-one-subject-out (LOSO) evaluation on 10 tabular models, saves per-fold artifacts, and produces a scientific comparison to identify the best performer. The summary uses mean/std across folds and paired statistical tests, not bootstrap confidence intervals.

# In[3]:


import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from scipy.stats import friedmanchisquare, wilcoxon
from tqdm.auto import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RANDOM_STATE = 42
DATASET = Path('/home/g0amer/Desktop/thesis/research_outputs/fusion/v1_fusion/fusion_dataset.csv')
OUT_ROOT = Path('/home/g0amer/Desktop/thesis/research_outputs/fusion_training/final_detector_loso_10models')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print('dataset:', DATASET)
print('out root:', OUT_ROOT)
print('xgboost available:', HAS_XGB)


# In[4]:


df = pd.read_csv(DATASET, low_memory=False)
if 'pseudo_label' not in df.columns:
    raise RuntimeError('expected pseudo_label column in the dataset')

subject_col = 'subject_id'
if subject_col not in df.columns:
    raise RuntimeError('expected subject_id column in the dataset')

exclude = {
    'subject_id', 'task_name', 'task_file', 'split', 'window_idx',
    'start_idx', 'end_idx', 'n_samples', 'pseudo_label', 'label',
    'eeg_features_5s__split', 'n_modalities_present'
}
feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
labels = sorted(df['pseudo_label'].astype(str).unique())
subjects = sorted(df['subject_id'].unique())

print('rows:', len(df))
print('subjects:', len(subjects))
print('classes:', labels)
print('features:', len(feature_cols))
print('class counts:')
print(df['pseudo_label'].value_counts())
print('feature NaNs:', int(df[feature_cols].isna().sum().sum()))


# In[5]:


def make_models():
    models = {
        'logreg': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))
        ]),
        'random_forest': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('clf', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        'extra_trees': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('clf', ExtraTreesClassifier(n_estimators=400, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        'gradient_boosting': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('clf', GradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
        'hist_gradient_boosting': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('clf', HistGradientBoostingClassifier(random_state=RANDOM_STATE))
        ]),
        'svm_rbf': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=3.0, gamma='scale', class_weight='balanced', probability=False, random_state=RANDOM_STATE))
        ]),
        'knn': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=7, weights='distance'))
        ]),
        'decision_tree': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('clf', DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE, max_depth=None))
        ]),
        'gaussian_nb': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('clf', GaussianNB())
        ]),
        'mlp': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=RANDOM_STATE, early_stopping=False))
        ]),
    }
    if HAS_XGB:
        models['xgboost'] = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('clf', XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    return models

def fold_labels():
    return sorted(df['pseudo_label'].astype(str).unique())

def fold_dir(model_name):
    d = OUT_ROOT / model_name / 'folds'
    d.mkdir(parents=True, exist_ok=True)
    return d


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', context='talk')

# Build the summary from whatever model fold files already exist so this cell
# can be run on its own after individual model cells.
summary_rows = []
for model_name in make_models().keys():
    fold_path = OUT_ROOT / model_name / 'fold_metrics.csv'
    if not fold_path.exists():
        continue
    fold_df = pd.read_csv(fold_path)
    if fold_df.empty:
        continue
    summary_rows.append({
        'model': model_name,
        'n_folds': int(len(fold_df)),
        'accuracy_mean': float(fold_df['accuracy'].mean()),
        'accuracy_std': float(fold_df['accuracy'].std()),
        'balanced_acc_mean': float(fold_df['balanced_acc'].mean()),
        'balanced_acc_std': float(fold_df['balanced_acc'].std()),
        'macro_f1_mean': float(fold_df['macro_f1'].mean()),
        'macro_f1_std': float(fold_df['macro_f1'].std()),
    })

if not summary_rows:
    raise RuntimeError('No fold_metrics.csv files found. Run at least one model cell first.')

summary_df = pd.DataFrame(summary_rows).sort_values('macro_f1_mean', ascending=False).reset_index(drop=True)
summary_df.to_csv(OUT_ROOT / 'benchmark_summary.csv', index=False)
summary_df['rank_macro_f1'] = np.arange(1, len(summary_df) + 1)
summary_df['rank_balanced_acc'] = summary_df['balanced_acc_mean'].rank(ascending=False, method='min').astype(int)
summary_df['rank_accuracy'] = summary_df['accuracy_mean'].rank(ascending=False, method='min').astype(int)
summary_df.to_csv(OUT_ROOT / 'benchmark_summary_ranked.csv', index=False)

# Comparison figures for the benchmark summary
summary_plot = summary_df.copy().sort_values('macro_f1_mean', ascending=True).reset_index(drop=True)
summary_plot['macro_f1_sem'] = summary_plot['macro_f1_std'] / np.sqrt(summary_plot['n_folds'].clip(lower=1))
summary_plot['balanced_acc_sem'] = summary_plot['balanced_acc_std'] / np.sqrt(summary_plot['n_folds'].clip(lower=1))
summary_plot['accuracy_sem'] = summary_plot['accuracy_std'] / np.sqrt(summary_plot['n_folds'].clip(lower=1))

fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)

# Panel 1: macro F1 ranking
axes[0].barh(summary_plot['model'], summary_plot['macro_f1_mean'], xerr=summary_plot['macro_f1_std'], color='#1f77b4', alpha=0.9)
axes[0].set_title('Macro F1 across LOSO folds')
axes[0].set_xlabel('Macro F1 mean ± std')
axes[0].set_ylabel('Model')
axes[0].set_xlim(0, 1)

# Panel 2: balanced accuracy ranking
axes[1].barh(summary_plot['model'], summary_plot['balanced_acc_mean'], xerr=summary_plot['balanced_acc_std'], color='#2ca02c', alpha=0.9)
axes[1].set_title('Balanced accuracy across LOSO folds')
axes[1].set_xlabel('Balanced accuracy mean ± std')
axes[1].set_ylabel('')
axes[1].set_xlim(0, 1)

# Panel 3: accuracy ranking
axes[2].barh(summary_plot['model'], summary_plot['accuracy_mean'], xerr=summary_plot['accuracy_std'], color='#ff7f0e', alpha=0.9)
axes[2].set_title('Accuracy across LOSO folds')
axes[2].set_xlabel('Accuracy mean ± std')
axes[2].set_ylabel('')
axes[2].set_xlim(0, 1)

fig.suptitle('LOSO model comparison on the full fusion dataset', fontsize=18, y=1.02)
fig.savefig(OUT_ROOT / 'comparison_metrics_bars.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT_ROOT / 'comparison_metrics_bars.svg', bbox_inches='tight')
plt.show()

# Fold-wise distribution figure for macro F1
fold_long = []
for model_name in summary_df['model'].tolist():
    fold_df = pd.read_csv(OUT_ROOT / model_name / 'fold_metrics.csv')
    fold_df = fold_df[['model', 'subject_id', 'macro_f1', 'balanced_acc', 'accuracy']].copy()
    fold_long.append(fold_df)
fold_long = pd.concat(fold_long, ignore_index=True)

plt.figure(figsize=(16, 7))
sns.boxplot(data=fold_long, x='model', y='macro_f1', order=summary_df['model'].tolist(), color='#c7dcef')
sns.stripplot(data=fold_long, x='model', y='macro_f1', order=summary_df['model'].tolist(), color='black', alpha=0.35, size=3, jitter=0.18)
plt.xticks(rotation=35, ha='right')
plt.ylim(0, 1)
plt.xlabel('Model')
plt.ylabel('Macro F1 per LOSO fold')
plt.title('Fold-wise macro F1 distribution')
plt.tight_layout()
plt.savefig(OUT_ROOT / 'comparison_macro_f1_boxplot.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_ROOT / 'comparison_macro_f1_boxplot.svg', bbox_inches='tight')
plt.show()

# Metric heatmap of model means
heatmap_df = summary_df.set_index('model')[['macro_f1_mean', 'balanced_acc_mean', 'accuracy_mean']]
plt.figure(figsize=(10, 7))
sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='viridis', vmin=0, vmax=1, cbar_kws={'label': 'Score'})
plt.title('Model comparison summary (mean across LOSO folds)')
plt.xlabel('Metric')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig(OUT_ROOT / 'comparison_metric_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT_ROOT / 'comparison_metric_heatmap.svg', bbox_inches='tight')
plt.show()

print('Saved figures to', OUT_ROOT)
print('Available figures: comparison_metrics_bars.png/.svg, comparison_macro_f1_boxplot.png/.svg, comparison_metric_heatmap.png/.svg')


# In[17]:


best_model = summary_df.iloc[0]['model']
best_path = OUT_ROOT / best_model / 'final_model.joblib'
best_model_obj = joblib.load(best_path)
sample_pred = best_model_obj.predict(df[feature_cols].iloc[:10].values)
print('best model:', best_model)
print('best model path:', best_path)
print('sample predictions:', sample_pred)

result_manifest = {
    'dataset': str(DATASET),
    'output_root': str(OUT_ROOT),
    'best_model': str(best_model),
    'summary_csv': str(OUT_ROOT / 'benchmark_summary.csv'),
    'ranked_csv': str(OUT_ROOT / 'benchmark_summary_ranked.csv'),
    'pairwise_csv': str(OUT_ROOT / 'pairwise_vs_best.csv'),
}
with open(OUT_ROOT / 'manifest.json', 'w') as f:
    json.dump(result_manifest, f, indent=2)
print(json.dumps(result_manifest, indent=2))


# ## What this notebook produces
# - Per-model LOSO fold CSVs with accuracy, balanced accuracy, and macro F1.
# - Per-fold classification reports and confusion matrices.
# - A ranked benchmark summary with mean/std aggregation across folds.
# - Comparison figures for the model benchmark: metric bar charts, fold-wise macro F1 boxplots, and a mean-metric heatmap.
# - A saved final model for each candidate classifier.

# In[18]:


def run_loso_model(model_name, df, feature_cols):
    """Run LOSO for a single model and save per-fold artifacts and final model."""
    model_registry = make_models()
    if model_name not in model_registry:
        raise RuntimeError(f'model {model_name} not found in registry')
    pipeline = model_registry[model_name]
    classes = fold_labels()

    model_root = OUT_ROOT / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    model_folds = fold_dir(model_name)

    fold_rows = []
    for subject in tqdm(subjects, desc=f'{model_name} LOSO', total=len(subjects)):
        train_mask = df['subject_id'] != subject
        test_mask = df['subject_id'] == subject
        train_df = df.loc[train_mask]
        test_df = df.loc[test_mask]
        if test_df.empty:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df['pseudo_label'].astype(str).values
        X_test = test_df[feature_cols].values
        y_test = test_df['pseudo_label'].astype(str).values

        model = pipeline
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metric = {
            'model': model_name,
            'subject_id': int(subject),
            'n_test': int(len(y_test)),
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'balanced_acc': float(balanced_accuracy_score(y_test, y_pred)),
            'macro_f1': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        }
        fold_rows.append(fold_metric)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=classes)

        pd.DataFrame([fold_metric]).to_csv(model_folds / f'fold_{int(subject):02d}.csv', index=False)
        pd.DataFrame(report).T.to_csv(model_folds / f'fold_{int(subject):02d}_per_class.csv')
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(model_folds / f'fold_{int(subject):02d}_confusion.csv')

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(model_root / 'fold_metrics.csv', index=False)

    print(f'Fitting final model for {model_name}', flush=True)
    final_model = pipeline
    final_model.fit(df[feature_cols].values, df['pseudo_label'].astype(str).values)
    joblib.dump(final_model, model_root / 'final_model.joblib')

    print(f'Completed LOSO for {model_name}: {len(fold_df)} folds', flush=True)
    return fold_df


# In[19]:


# Run this cell to execute only the logistic regression LOSO benchmark.
logreg_fold_df = run_loso_model('logreg', df, feature_cols)
logreg_fold_df


# In[20]:


# Run this cell to execute only the random forest LOSO benchmark.
random_forest_fold_df = run_loso_model('random_forest', df, feature_cols)
random_forest_fold_df


# In[21]:


# Run this cell to execute only the extra trees LOSO benchmark.
extra_trees_fold_df = run_loso_model('extra_trees', df, feature_cols)
extra_trees_fold_df


# In[22]:


# Run this cell to execute only the gradient boosting LOSO benchmark.
gradient_boosting_fold_df = run_loso_model('gradient_boosting', df, feature_cols)
gradient_boosting_fold_df


# In[23]:


# Run this cell to execute only the histogram gradient boosting LOSO benchmark.
hist_gradient_boosting_fold_df = run_loso_model('hist_gradient_boosting', df, feature_cols)
hist_gradient_boosting_fold_df


# In[24]:


# Run this cell to execute only the svm_rbf LOSO benchmark.
svm_rbf_fold_df = run_loso_model('svm_rbf', df, feature_cols)
svm_rbf_fold_df


# In[25]:


# Run this cell to execute only the knn LOSO benchmark.
knn_fold_df = run_loso_model('knn', df, feature_cols)
knn_fold_df


# In[26]:


# Run this cell to execute only the decision tree LOSO benchmark.
decision_tree_fold_df = run_loso_model('decision_tree', df, feature_cols)
decision_tree_fold_df


# In[27]:


# Run this cell to execute only the gaussian naive bayes LOSO benchmark.
gaussian_nb_fold_df = run_loso_model('gaussian_nb', df, feature_cols)
gaussian_nb_fold_df


# In[28]:


# Run this cell to execute only the mlp LOSO benchmark.
mlp_fold_df = run_loso_model('mlp', df, feature_cols)
mlp_fold_df


# In[7]:


# Aggregate per-model fold_metrics.csv into the benchmark summary (run after you've executed individual model cells)
summary_rows = []
for model_name in make_models().keys():
    fold_path = OUT_ROOT / model_name / 'fold_metrics.csv'
    if not fold_path.exists():
        print('No results for', model_name)
        continue
    fold_df = pd.read_csv(fold_path)
    if fold_df.empty:
        continue
    summary_rows.append({
        'model': model_name,
        'n_folds': int(len(fold_df)),
        'accuracy_mean': float(fold_df['accuracy'].mean()),
        'accuracy_std': float(fold_df['accuracy'].std()),
        'balanced_acc_mean': float(fold_df['balanced_acc'].mean()),
        'balanced_acc_std': float(fold_df['balanced_acc'].std()),
        'macro_f1_mean': float(fold_df['macro_f1'].mean()),
        'macro_f1_std': float(fold_df['macro_f1'].std()),
    })

summary_df = pd.DataFrame(summary_rows).sort_values('macro_f1_mean', ascending=False).reset_index(drop=True)
summary_df.to_csv(OUT_ROOT / 'benchmark_summary.csv', index=False)
print(summary_df)

# Also save ranked
summary_df['rank_macro_f1'] = np.arange(1, len(summary_df) + 1)
summary_df['rank_balanced_acc'] = summary_df['balanced_acc_mean'].rank(ascending=False, method='min').astype(int)
summary_df['rank_accuracy'] = summary_df['accuracy_mean'].rank(ascending=False, method='min').astype(int)
summary_df.to_csv(OUT_ROOT / 'benchmark_summary_ranked.csv', index=False)
print('Wrote benchmark_summary.csv and benchmark_summary_ranked.csv')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', context='talk')

# Aggregate and plot the LOSO confusion matrix for the best available model.
summary_path = OUT_ROOT / 'benchmark_summary.csv'
if 'summary_df' not in globals():
    if not summary_path.exists():
        raise RuntimeError('Run at least one model cell and the summary cell before generating the confusion matrix.')
    summary_df = pd.read_csv(summary_path)

summary_df = summary_df.sort_values('macro_f1_mean', ascending=False).reset_index(drop=True)
best_model = summary_df.iloc[0]['model']
classes = sorted(df['pseudo_label'].astype(str).unique())
model_folds = OUT_ROOT / best_model / 'folds'

cm_sum = None
for cm_path in sorted(model_folds.glob('fold_*_confusion.csv')):
    cm_df = pd.read_csv(cm_path, index_col=0)
    cm_df = cm_df.reindex(index=classes, columns=classes, fill_value=0)
    cm = cm_df.to_numpy(dtype=float)
    cm_sum = cm if cm_sum is None else cm_sum + cm

if cm_sum is None:
    raise RuntimeError(f'No per-fold confusion matrices found under {model_folds}')

row_totals = cm_sum.sum(axis=1, keepdims=True)
cm_norm = np.divide(cm_sum, row_totals, out=np.zeros_like(cm_sum, dtype=float), where=row_totals != 0)

fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)

sns.heatmap(
    cm_sum,
    annot=True,
    fmt='.0f',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes,
    ax=axes[0],
    cbar_kws={'label': 'Count'},
)
axes[0].set_title(f'LOSO Confusion Matrix (counts) - {best_model}')
axes[0].set_xlabel('Predicted label')
axes[0].set_ylabel('True label')

sns.heatmap(
    cm_norm,
    annot=True,
    fmt='.2f',
    cmap='Greens',
    xticklabels=classes,
    yticklabels=classes,
    ax=axes[1],
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Row-normalized proportion'},
)
axes[1].set_title(f'LOSO Confusion Matrix (row-normalized) - {best_model}')
axes[1].set_xlabel('Predicted label')
axes[1].set_ylabel('')

cm_png = OUT_ROOT / f'confusion_matrix_loso_{best_model}.png'
cm_svg = OUT_ROOT / f'confusion_matrix_loso_{best_model}.svg'
fig.savefig(cm_png, dpi=300, bbox_inches='tight')
fig.savefig(cm_svg, bbox_inches='tight')
plt.show()

print('Saved confusion matrix to', cm_png)
print('Saved confusion matrix to', cm_svg)
print('Model used:', best_model)

