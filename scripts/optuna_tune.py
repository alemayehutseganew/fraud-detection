"""Run Optuna tuning for RandomForest and XGBoost with SMOTE pipelines.
Saves best models to models/tuned_rf.pkl, models/tuned_xgb.pkl and the overall best to models/best_model_pipeline.pkl
Also generates a precision-recall plot and saves SHAP summary for the selected model.
"""
import os
import joblib
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.explain import SHAPExplainer
import matplotlib.pyplot as plt

# Load data
path = 'data/processed/fraud_data_features.csv'
df = pd.read_csv(path)

y = df['class'].astype(int)
X = df.drop(columns=['class'])
for c in ['user_id','signup_time','purchase_time','ip_address','device_id']:
    if c in X.columns:
        X = X.drop(columns=[c])

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_cols), ('cat', cat_transformer, cat_cols)], remainder='drop')

# Split once (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Helper: evaluate pipeline on validation folds (returns mean AP)
def cv_score_pipeline(model_cls, params, n_splits=3, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aps = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = model_cls(**params)
        pipe = ImbPipeline(steps=[('preprocess', preprocessor), ('smote', SMOTE(random_state=random_state)), ('clf', model)])
        pipe.fit(X_tr, y_tr)
        y_proba = pipe.predict_proba(X_val)[:,1]
        ap = average_precision_score(y_val, y_proba)
        aps.append(ap)
    return np.mean(aps)

# Optuna objective for RF
def objective_rf(trial):
    n_estimators = trial.suggest_categorical('n_estimators', [100,200,300])
    max_depth = trial.suggest_categorical('max_depth', [6,10,16,None])
    params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': 42, 'n_jobs': -1}
    score = cv_score_pipeline(RandomForestClassifier, params, n_splits=3)
    return score

# Optuna objective for XGB
def objective_xgb(trial):
    n_estimators = trial.suggest_categorical('n_estimators', [100,200,300])
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
    params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate, 'random_state': 42, 'use_label_encoder': False, 'n_jobs': -1}
    score = cv_score_pipeline(XGBClassifier, params, n_splits=3)
    return score

os.makedirs('models', exist_ok=True)

# Run studies
study_rf = optuna.create_study(direction='maximize', study_name='rf_ap')
study_rf.optimize(objective_rf, n_trials=20)
best_rf_params = study_rf.best_params
print('Best RF params:', best_rf_params)

study_xgb = optuna.create_study(direction='maximize', study_name='xgb_ap')
study_xgb.optimize(objective_xgb, n_trials=20)
best_xgb_params = study_xgb.best_params
print('Best XGB params:', best_xgb_params)

# Fit best models on full training data
best_rf = RandomForestClassifier(random_state=42, n_jobs=-1, **best_rf_params)
rf_pipe = ImbPipeline(steps=[('preprocess', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', best_rf)])
rf_pipe.fit(X_train, y_train)
joblib.dump(rf_pipe, 'models/tuned_rf.pkl')

best_xgb = XGBClassifier(random_state=42, n_jobs=-1, **best_xgb_params, use_label_encoder=False, eval_metric='logloss')
xgb_pipe = ImbPipeline(steps=[('preprocess', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', best_xgb)])
xgb_pipe.fit(X_train, y_train)
joblib.dump(xgb_pipe, 'models/tuned_xgb.pkl')

# Evaluate on test set
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix, precision_recall_curve, auc

models = {'tuned_rf': rf_pipe, 'tuned_xgb': xgb_pipe}
results = {}
for name, pipe in models.items():
    y_proba = pipe.predict_proba(X_test)[:,1]
    y_pred = pipe.predict(X_test)
    ap = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {'ap': float(ap), 'f1': float(f1), 'cm': cm}
    # Save PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label=f'AP={pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — {name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'reports/figures/pr_curve_{name}.png')
    plt.close()

# Choose best model by AP
best_name = max(results.items(), key=lambda kv: kv[1]['ap'])[0]
print('Best tuned model:', best_name, results[best_name])
# Save best pipeline
best_pipeline = models[best_name]
joblib.dump(best_pipeline, 'models/best_model_pipeline.pkl')

# SHAP artifacts for best model
feature_names = X.columns.tolist()
expl = SHAPExplainer(best_pipeline.named_steps['clf'], feature_names)
# Need transformed training data for explainer kernel/tree
X_train_trans = best_pipeline.named_steps['preprocess'].fit_transform(X_train)
# If X_train_trans is numpy, convert to DataFrame
if isinstance(X_train_trans, np.ndarray):
    X_train_df = pd.DataFrame(X_train_trans, columns=(feature_names if len(feature_names)==X_train_trans.shape[1] else [f'feature_{i}' for i in range(X_train_trans.shape[1])]))
else:
    X_train_df = X_train_trans
expl.create_explainer(X_train_df, explainer_type='tree')
# Use a sample to compute SHAP
sample = X_train_df.sample(n=min(500, len(X_train_df)), random_state=42)
expl.calculate_shap_values(sample)
# Save summary plot
plt.figure(figsize=(12,8))
expl.plot_summary(sample, max_display=20)
plt.savefig('reports/figures/shap_summary_tuned.png')
plt.close()
# Save feature importance
importance_df = expl.get_feature_importance()
importance_df.to_csv('reports/figures/shap_feature_importance_tuned.csv', index=False)

# Save results summary
pd.DataFrame.from_dict(results, orient='index').to_csv('reports/figures/tuned_model_results.csv')
print('Tuning complete. Artifacts saved in reports/figures and models/.')
