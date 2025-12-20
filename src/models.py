"""
Machine learning models for fraud detection.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
SMOTE = None
RandomUnderSampler = None
ImbPipeline = None
import joblib
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionModels:
    """Fraud detection machine learning models."""
    
    def __init__(self, random_state=42):
        """Initialize models."""
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        
    def create_baseline_model(self):
        """Create baseline logistic regression model."""
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        return model
    
    def create_random_forest(self, n_estimators=100, max_depth=None):
        """Create random forest model."""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        return model
    
    def create_xgboost(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        """Create XGBoost model."""
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            scale_pos_weight=self.calculate_scale_pos_weight
        )
        return model
    
    def create_lightgbm(self, n_estimators=100, max_depth=-1, learning_rate=0.1):
        """Create LightGBM model."""
        model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        return model
    
    def calculate_scale_pos_weight(self, y):
        """Calculate scale_pos_weight for XGBoost."""
        negative_count = np.sum(y == 0)
        positive_count = np.sum(y == 1)
        return negative_count / positive_count if positive_count > 0 else 1
    
    def handle_imbalance(self, X_train, y_train, method='smote'):
        """Handle class imbalance in training data.

        Tries to use imbalanced-learn samplers; if unavailable or incompatible
        with the installed scikit-learn, falls back to simple sklearn resampling.
        """
        try:
            if method == 'smote':
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=self.random_state)
            elif method == 'undersample':
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(random_state=self.random_state)
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

        except Exception as e:
            # Fallback: use simple up/down-sampling via sklearn.utils.resample
            from sklearn.utils import resample
            import pandas as _pd

            print(f"Imblearn sampler not available or failed ({e}); falling back to simple resampling.")

            # Create combined frame for resampling
            Xy = X_train.copy()
            Xy['__y__'] = y_train

            majority = Xy[Xy['__y__'] == 0]
            minority = Xy[Xy['__y__'] == 1]

            if len(minority) == 0 or len(majority) == 0:
                # Nothing to do
                X_resampled = X_train
                y_resampled = y_train
            elif method == 'undersample':
                # Downsample majority to minority size
                majority_down = resample(majority, replace=False, n_samples=len(minority), random_state=self.random_state)
                resampled = _pd.concat([majority_down, minority])
                y_resampled = resampled['__y__']
                X_resampled = resampled.drop(columns='__y__')
            else:
                # Upsample minority to majority size
                minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=self.random_state)
                resampled = _pd.concat([majority, minority_up])
                y_resampled = resampled['__y__']
                X_resampled = resampled.drop(columns='__y__')

        print(f"Before resampling: {np.bincount(y_train)}")
        print(f"After {method}: {np.bincount(y_resampled)}")

        return X_resampled, y_resampled
    
    def train_model(self, model, X_train, y_train, X_val=None, y_val=None):
        """Train a single model."""
        print(f"Training {model.__class__.__name__}...")
        model.fit(X_train, y_train)
        return model
    
    def train_with_cross_validation(self, model, X, y, cv=5):
        """Train model with cross-validation."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Handle imbalance for this fold
            X_train_resampled, y_train_resampled = self.handle_imbalance(X_train, y_train)
            
            # Train model
            fold_model = self.train_model(model, X_train_resampled, y_train_resampled)
            
            # Evaluate
            score = fold_model.score(X_val, y_val)
            cv_scores.append(score)
            print(f"Fold {fold}: Accuracy = {score:.4f}")
        
        print(f"CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        return cv_scores
    
    def hyperparameter_tuning(self, model, param_grid, X_train, y_train, cv=3):
        """Perform hyperparameter tuning."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=skf,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def save_model(self, model, model_name, filepath='models/'):
        """Save trained model."""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        filename = f"{filepath}/{model_name}.pkl"
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, model_name, filepath='models/'):
        """Load trained model."""
        filename = f"{filepath}/{model_name}.pkl"
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model