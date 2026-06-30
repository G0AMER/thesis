#!/usr/bin/env python3
from pathlib import Path
import sys
import re

BASE = Path('/home/g0amer/Desktop/thesis/research_outputs/fusion_training/v2_full_benchmark')
if not BASE.exists():
    print('Base directory not found:', BASE)
    sys.exit(2)

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print('Required packages missing:', e)
    print('Run: pip install -r /home/g0amer/Desktop/thesis/phd_project/requirements.txt')
    raise

n_boot = 2000
rng = np.random.default_rng(42)

def bootstrap_ci(arr, n_boot=n_boot, rng=rng):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    mean = float(np.mean(arr))
    low = float(np.percentile(means, 2.5))
    high = float(np.percentile(means, 97.5))
    return mean, low, high

model_dirs = [p for p in BASE.iterdir() if p.is_dir() and p.name != 'figures']
print('Found model dirs:', [p.name for p in model_dirs])

summary_rows = []
problems = []

for md in model_dirs:
    # select only files named like fold_01.csv, fold_1.csv, etc.
    fold_files = sorted([f for f in md.glob('*.csv') if re.match(r'^fold_\d+\.csv$', f.name)])
    if len(fold_files) == 0:
        # fallback: any file starting with 'fold_' and ending with .csv
        fold_files = sorted([f for f in md.glob('*.csv') if f.name.startswith('fold_') and f.suffix=='.csv'])
    values_macro = []
    values_bal = []
    for f in fold_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            problems.append((md.name, f.name, 'read_error', str(e)))
            continue
        if 'macro_f1' in df.columns:
            col_m = 'macro_f1'
        elif 'macro-f1' in df.columns:
            col_m = 'macro-f1'
        else:
            problems.append((md.name, f.name, 'no_macro_f1', df.columns.tolist()))
            continue
        if 'balanced_acc' in df.columns:
            col_b = 'balanced_acc'
        elif 'balanced-acc' in df.columns:
            col_b = 'balanced-acc'
        else:
            col_b = None
        try:
            v_m = df[col_m].iloc[0]
            values_macro.append(float(v_m))
        except Exception as e:
            problems.append((md.name, f.name, 'macro_read_error', str(e)))
        if col_b is not None:
            try:
                v_b = df[col_b].iloc[0]
                values_bal.append(float(v_b))
            except Exception as e:
                problems.append((md.name, f.name, 'bal_read_error', str(e)))
    mean_m, low_m, high_m = bootstrap_ci(values_macro)
    mean_b, low_b, high_b = bootstrap_ci(values_bal)
    summary_rows.append({
        'model': md.name,
        'n_folds': len(values_macro),
        'macro_f1_mean': mean_m,
        'macro_f1_ci_low': low_m,
        'macro_f1_ci_high': high_m,
        'balanced_acc_mean': mean_b,
        'balanced_acc_ci_low': low_b,
        'balanced_acc_ci_high': high_b,
    })

    summary_df = pd.DataFrame(summary_rows).sort_values('macro_f1_mean', ascending=False)
print('\nSummary:')
print(summary_df.to_string(index=False))

if problems:
    print('\nProblems encountered (sample 10):')
    for p in problems[:10]:
        print(p)

orig_file = BASE / 'benchmark_summary.csv'
backup_file = BASE / 'benchmark_summary.backup.csv'
if orig_file.exists():
    print('\nBacking up original benchmark_summary.csv ->', backup_file.name)
    orig = pd.read_csv(orig_file)
    orig.to_csv(backup_file, index=False)
    out = orig.copy()
    for _, r in summary_df.iterrows():
        m = r['model']
        idx = out[out['model']==m].index
        if len(idx):
            i = idx[0]
            if 'folds' in out.columns:
                out.at[i, 'folds'] = int(r['n_folds'])
            if 'macro_f1' in out.columns:
                out.at[i, 'macro_f1'] = r['macro_f1_mean']
            if 'macro_f1_ci_low' in out.columns:
                out.at[i, 'macro_f1_ci_low'] = r['macro_f1_ci_low']
            if 'macro_f1_ci_high' in out.columns:
                out.at[i, 'macro_f1_ci_high'] = r['macro_f1_ci_high']
            if 'balanced_acc' in out.columns:
                out.at[i, 'balanced_acc'] = r['balanced_acc_mean']
            if 'balanced_acc_ci_low' in out.columns:
                out.at[i, 'balanced_acc_ci_low'] = r['balanced_acc_ci_low']
            if 'balanced_acc_ci_high' in out.columns:
                out.at[i, 'balanced_acc_ci_high'] = r['balanced_acc_ci_high']
        else:
            new = {c: None for c in out.columns}
            new['model'] = m
            if 'folds' in out.columns:
                new['folds'] = int(r['n_folds'])
            if 'macro_f1' in out.columns:
                new['macro_f1'] = r['macro_f1_mean']
            if 'macro_f1_ci_low' in out.columns:
                new['macro_f1_ci_low'] = r['macro_f1_ci_low']
            if 'macro_f1_ci_high' in out.columns:
                new['macro_f1_ci_high'] = r['macro_f1_ci_high']
            if 'balanced_acc' in out.columns:
                new['balanced_acc'] = r['balanced_acc_mean']
            if 'balanced_acc_ci_low' in out.columns:
                new['balanced_acc_ci_low'] = r['balanced_acc_ci_low']
            if 'balanced_acc_ci_high' in out.columns:
                new['balanced_acc_ci_high'] = r['balanced_acc_ci_high']
            out = pd.concat([out, pd.DataFrame([new])], ignore_index=True)
    new_file = BASE / 'benchmark_summary_validated.csv'
    out.to_csv(new_file, index=False)
    print('\nWrote validated summary to', new_file)
else:
    new_file = BASE / 'benchmark_summary_validated.csv'
    summary_df.to_csv(new_file, index=False)
    print('\nWrote validated summary to', new_file)

rank_rows = []
for _, r in summary_df.iterrows():
    mf = r['macro_f1_mean']
    l, h = r['macro_f1_ci_low'], r['macro_f1_ci_high']
    if np.isnan(mf):
        mf_str = 'nan'
    else:
        mf_str = f"{mf:.3f} [{l:.3f},{h:.3f}]"
    bf = r['balanced_acc_mean']
    bl, bh = r['balanced_acc_ci_low'], r['balanced_acc_ci_high']
    if np.isnan(bf):
        bf_str = 'nan'
    else:
        bf_str = f"{bf:.3f} [{bl:.3f},{bh:.3f}]"
    rank_rows.append({
        'model': r['model'],
        'macro_f1_mean_ci': mf_str,
        'balanced_acc_mean_ci': bf_str,
        'n_folds': int(r['n_folds'])
    })
rank_df = pd.DataFrame(rank_rows)
rank_file = BASE / 'ranking_table_validated.csv'
rank_df.to_csv(rank_file, index=False)
print('\nWrote ranking table to', rank_file)

print('\nDone')
