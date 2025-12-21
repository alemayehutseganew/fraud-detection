"""Train a baseline LightGBM model on processed data and save artifacts.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, average_precision_score

import lightgbm as lgb

OUT_DIR = Path('models')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# load data
p = Path('data/processed/fraud_data_processed.csv')
if not p.exists():
    p = Path('fraud-detection/data/processed/fraud_data_processed.csv')
if not p.exists():
    raise SystemExit('Processed data not found')

df = pd.read_csv(p)
# target is 'class'
if 'class' in df.columns:
    y = df['class'].astype(int)
    X = df.drop(columns=['class'])
else:
    raise SystemExit('Target column `class` not found')

X = X.select_dtypes(include=[np.number])
if X.shape[1] == 0:
    raise SystemExit('No numeric features for baseline training')

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print('Average precision (AP):', average_precision_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

with open(OUT_DIR / 'baseline_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Saved baseline model to', OUT_DIR / 'baseline_model.pkl')
