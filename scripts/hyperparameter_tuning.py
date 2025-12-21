"""Hyperparameter tuning with Optuna for LightGBM.
Saves best model to `models/best_model.pkl`, best params to `models/best_params.json`, and the Optuna study to `models/optuna_study.pkl`.

This script runs a limited study by default (n_trials=20) — adjust via CLI `--trials` for longer runs.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, precision_recall_curve, average_precision_score

try:
    import optuna
    import lightgbm as lgb
except Exception as e:
    raise SystemExit('Missing required packages: ensure optuna and lightgbm are installed.')

OUT_DIR = Path('models')
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = Path('reports/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    # Prefer processed dataset with target column.
    # Try multiple likely locations so the script works when run from repository root
    # or from the `fraud-detection` subfolder.
    base_paths = [Path('data/processed'), Path('fraud-detection/data/processed'), Path(__file__).resolve().parents[2] / 'data' / 'processed']
    candidates = []
    for base in base_paths:
        candidates.extend([
            base / 'fraud_data_processed.csv',
            base / 'fraud_data_features.csv',
            base / 'fraud_data_processed.csv',
            base / 'fraud_data.csv',
        ])
    # deduplicate while preserving order
    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    df = None
    for p in uniq:
        if p.exists():
            try:
                df = pd.read_csv(p)
                print(f'Loaded {p}')
                break
            except Exception as e:
                print(f'Failed to read {p}: {e}')
    if df is None:
        raise FileNotFoundError('No processed data file found in data/processed/ or fraud-detection/data/processed/')

    # find target
    target_candidates = ['is_fraud', 'isFraud', 'fraud', 'Class', 'class', 'label']
    y_col = next((c for c in target_candidates if c in df.columns), None)
    if y_col is None:
        raise ValueError('No target column found in data. Expected one of: ' + ','.join(target_candidates))

    y = df[y_col].astype(int)
    X = df.drop(columns=[y_col])
    # keep only numeric columns for LightGBM compatibility; categorical can be added if desired
    X = X.select_dtypes(include=[np.number])
    if X.shape[1] == 0:
        raise ValueError('No numeric features found for training after dropping target.')
    return X, y


def objective(trial, X, y, cv):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'max_depth': trial.suggest_int('max_depth', -1, 16),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'random_state': 42,
    }

    model = lgb.LGBMClassifier(**param)
    # use average precision (AUC-PR) as the main metric since dataset is imbalanced
    scores = cross_val_score(model, X, y, cv=cv, scoring='average_precision', n_jobs=1)
    return float(np.mean(scores))


def run_study(trials=20):
    X, y = load_data()
    # stratified K-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _objective(trial):
        return objective(trial, X, y, cv)

    # Make Optuna more verbose and run the study
    optuna.logging.set_verbosity(optuna.logging.INFO)
    print(f'Starting Optuna study with {trials} trials...')
    study = optuna.create_study(direction='maximize', study_name='lgbm_fraud_ap')
    study.optimize(_objective, n_trials=trials, n_jobs=1)

    print('Optuna study complete.')
    print('Best trial params:', study.best_trial.params)
    print('Best average precision (AP):', study.best_value)

    # train final model on full data with best params
    best_params = study.best_trial.params
    best_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42})
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X, y)

    # persist artifacts
    with open(OUT_DIR / 'best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(OUT_DIR / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    with open(OUT_DIR / 'optuna_study.pkl', 'wb') as f:
        pickle.dump(study, f)

    # optional: save cross-validated predictions (leave for now)
    # save simple summary
    summary = {
        'best_value': study.best_value,
        'n_trials': len(study.trials),
    }
    with open(OUT_DIR / 'study_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print('Artifacts saved to', OUT_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20, help='Number of Optuna trials (default=20)')
    args = parser.parse_args()
    run_study(trials=args.trials)
