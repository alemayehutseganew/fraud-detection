"""Optuna hyperparameter tuning for RandomForest and XGBoost.
Saves best models and params, and writes summary JSON in models/.
"""
import json
import os
from pathlib import Path
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import joblib

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / 'data' / 'processed' / 'fraud_data_features.csv'
OUT_DIR = BASE / 'models'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = BASE / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

print('Loading data...')
df = pd.read_csv(DATA_PATH)

y = df['class'].astype(int)
X = df.drop(columns=['class'])
for c in ['user_id','signup_time','purchase_time','ip_address','device_id']:
    if c in X.columns:
        X = X.drop(columns=[c])

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols), ('cat', cat_transformer, cat_cols)], remainder='drop')

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

X_np = X
y_np = y

# Helper: evaluate with CV and SMOTE
def cv_score_rf(params):
    ap_scores = []
    for train_idx, val_idx in skf.split(X_np, y_np):
        X_tr, X_val = X_np.iloc[train_idx], X_np.iloc[val_idx]
        y_tr, y_val = y_np.iloc[train_idx], y_np.iloc[val_idx]
        # transform
        X_tr_t = preprocessor.fit_transform(X_tr)
        X_val_t = preprocessor.transform(X_val)
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_tr_t, y_tr)
        model = RandomForestClassifier(n_estimators=int(params['n_estimators']), max_depth=params['max_depth'], max_features=params['max_features'], random_state=42, n_jobs=-1)
        model.fit(X_res, y_res)
        y_proba = model.predict_proba(X_val_t)[:,1]
        ap = average_precision_score(y_val, y_proba)
        ap_scores.append(ap)
    return np.mean(ap_scores)

# Optuna objective for RF
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [100,200,300]),
        'max_depth': trial.suggest_categorical('max_depth', [6,10,20,None]),
        'max_features': trial.suggest_categorical('max_features', ['sqrt','log2', None])
    }
    score = cv_score_rf(params)
    return score

print('Tuning RandomForest (20 trials)...')
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=20)
print('RF best params:', study_rf.best_params, 'best value:', study_rf.best_value)

# Fit final RF on full training set (we'll do stratified train-test split for final evaluation)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, stratify=y_np, random_state=42)

rf_best = RandomForestClassifier(n_estimators=int(study_rf.best_params['n_estimators']), max_depth=study_rf.best_params['max_depth'], max_features=study_rf.best_params['max_features'], random_state=42, n_jobs=-1)
rf_pipeline = ImbPipeline(steps=[('preprocess', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', rf_best)])
rf_pipeline.fit(X_train, y_train)
joblib.dump(rf_pipeline, OUT_DIR / 'rf_tuned_pipeline.pkl')

# XGBoost tuning
print('Tuning XGBoost (20 trials)...')

def cv_score_xgb(params):
    ap_scores = []
    for train_idx, val_idx in skf.split(X_np, y_np):
        X_tr, X_val = X_np.iloc[train_idx], X_np.iloc[val_idx]
        y_tr, y_val = y_np.iloc[train_idx], y_np.iloc[val_idx]
        X_tr_t = preprocessor.fit_transform(X_tr)
        X_val_t = preprocessor.transform(X_val)
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_tr_t, y_tr)
        model = XGBClassifier(n_estimators=int(params['n_estimators']), max_depth=int(params['max_depth']), learning_rate=params['learning_rate'], subsample=params['subsample'], colsample_bytree=params['colsample_bytree'], use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
        model.fit(X_res, y_res, verbose=False)
        y_proba = model.predict_proba(X_val_t)[:,1]
        ap = average_precision_score(y_val, y_proba)
        ap_scores.append(ap)
    return np.mean(ap_scores)

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [100,200,300]),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    return cv_score_xgb(params)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=20)
print('XGB best params:', study_xgb.best_params, 'best value:', study_xgb.best_value)

xgb_best = XGBClassifier(n_estimators=int(study_xgb.best_params['n_estimators']), max_depth=int(study_xgb.best_params['max_depth']), learning_rate=study_xgb.best_params['learning_rate'], subsample=study_xgb.best_params['subsample'], colsample_bytree=study_xgb.best_params['colsample_bytree'], use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_pipeline = ImbPipeline(steps=[('preprocess', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', xgb_best)])
xgb_pipeline.fit(X_train, y_train)
joblib.dump(xgb_pipeline, OUT_DIR / 'xgb_tuned_pipeline.pkl')

# Save summary
summary = {
    'rf': {'best_params': study_rf.best_params, 'best_value': study_rf.best_value},
    'xgb': {'best_params': study_xgb.best_params, 'best_value': study_xgb.best_value}
}
with open(OUT_DIR / 'tuning_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Tuning complete. Models saved to', OUT_DIR)
