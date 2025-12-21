"""Generate recommended figures for the final report.
Saves outputs to reports/figures/:
 - fraud_by_country.png
 - class_distribution_before_after.png
 - class_distribution_after_smote.png

This script is defensive: if required columns or packages are missing, it will write informative placeholder PNGs.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = Path('reports/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Helper to create a text placeholder PNG
def save_placeholder(path, text):
    plt.figure(figsize=(8,4))
    plt.text(0.5, 0.5, text, ha='center', va='center', wrap=True)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

# Try to load processed data
candidates = [
    'data/processed/fraud_data_processed.csv',
    'data/processed/fraud_data_features.csv',
    'data/processed/fraud_data_features.csv',
    'data/processed/fraud_data.csv'
]

df = None
for p in candidates:
    if Path(p).exists():
        try:
            df = pd.read_csv(p)
            print(f'Loaded {p}')
            break
        except Exception as e:
            print(f'Failed to read {p}: {e}')

if df is None:
    save_placeholder(OUT_DIR / 'fraud_by_country.png', 'Missing processed dataset: cannot generate fraud-by-country plot.')
    save_placeholder(OUT_DIR / 'class_distribution_before_after.png', 'Missing processed dataset: cannot generate SMOTE distribution plots.')
    print('No processed data found; wrote placeholders.')
    raise SystemExit(0)

# Identify target column
target_candidates = ['class', 'is_fraud', 'isFraud', 'fraud', 'Class', 'label']
Y_COL = None
for c in target_candidates:
    if c in df.columns:
        Y_COL = c
        break

if Y_COL is None:
    save_placeholder(OUT_DIR / 'class_distribution_before_after.png', 'No target column (is_fraud) found in dataset.')
    print('No target column found; wrote placeholder and continuing with country plot if possible.')

# 1) Fraud by country
if 'country' in df.columns:
    try:
        country_stats = (
            df.groupby('country')
              .agg(total=('{}' .format(Y_COL) if Y_COL else df.columns[0], 'size'), frauds=(Y_COL if Y_COL else df.columns[0], 'sum'))
        )
    except Exception:
        # fallback: compute fraud counts if Y_COL present, else counts only
        if Y_COL:
            country_stats = df.groupby('country').agg(total=('{}' .format(Y_COL), 'size'), frauds=(Y_COL, 'sum'))
        else:
            country_stats = df.groupby('country').agg(total=('country','size'))
            country_stats['frauds'] = 0
    if 'frauds' in country_stats.columns:
        country_stats = country_stats.assign(fraud_rate=lambda x: x['frauds'] / x['total'].replace({0: np.nan}))
        country_stats = country_stats.sort_values('fraud_rate', ascending=False).fillna(0)
    else:
        country_stats['fraud_rate'] = 0

    top = country_stats.head(20).reset_index()
    plt.figure(figsize=(10,6))
    sns.barplot(data=top, y='country', x='fraud_rate', color='C0')
    plt.title('Top 20 Countries by Fraud Rate')
    plt.xlabel('Fraud Rate')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fraud_by_country.png', dpi=150)
    plt.close()
    country_stats.to_csv(OUT_DIR / 'country_fraud_rates.csv')
    print('Saved fraud_by_country.png')
else:
    save_placeholder(OUT_DIR / 'fraud_by_country.png', 'No `country` column in processed data; cannot generate fraud-by-country plot.')
    print('No country column; wrote placeholder.')

# 2) SMOTE before/after class distribution
if Y_COL:
    try:
        y = df[Y_COL].astype(int)
    except Exception:
        y = df[Y_COL]
    # before distribution
    before_counts = y.value_counts().sort_index()

    # Save before-only distribution plot
    plt.figure(figsize=(6,4))
    sns.barplot(x=before_counts.index.astype(str), y=before_counts.values, color='C0')
    plt.title('Class distribution before resampling')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'class_distribution_before.png', dpi=150)
    plt.close()

    # create a simple numeric X for SMOTE: use numeric columns or fall back to index
    X = df.select_dtypes(include=[np.number]).drop(columns=[Y_COL], errors='ignore')
    if X.shape[1] == 0:
        X = pd.DataFrame({'dummy': np.zeros(len(df))})

    # Try to apply SMOTE to show the effect; if imblearn fails, save a placeholder for the after-plot
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        after_counts = pd.Series(y_res).value_counts().sort_index()

        # plot before/after side-by-side
        fig, axes = plt.subplots(1,2, figsize=(10,4))
        sns.barplot(x=before_counts.index.astype(str), y=before_counts.values, ax=axes[0], color='C0')
        axes[0].set_title('Before SMOTE')
        sns.barplot(x=after_counts.index.astype(str), y=after_counts.values, ax=axes[1], color='C1')
        axes[1].set_title('After SMOTE')
        for ax in axes:
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'class_distribution_before_after.png', dpi=150)
        plt.close()

        # also save after-only plot
        plt.figure(figsize=(6,4))
        sns.barplot(x=after_counts.index.astype(str), y=after_counts.values, color='C1')
        plt.title('Class distribution after SMOTE')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'class_distribution_after_smote.png', dpi=150)
        plt.close()

        print('Saved class distribution plots (before and after SMOTE).')
    except Exception as e:
        print('SMOTE failed or imblearn not installed:', e)
        # Save placeholder for after-smote to indicate missing dependency, but we still have before-only plot saved
        save_placeholder(OUT_DIR / 'class_distribution_after_smote.png', 'SMOTE not available: install imbalanced-learn to generate after-resampling plots.')
else:
    save_placeholder(OUT_DIR / 'class_distribution_before.png', 'No target column; cannot compute class distributions.')
    save_placeholder(OUT_DIR / 'class_distribution_after_smote.png', 'No target column; cannot compute class distributions.')
    print('No target column; wrote placeholders for SMOTE figures.')

print('Figure generation complete. Files saved to', OUT_DIR)
