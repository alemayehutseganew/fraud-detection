#!/usr/bin/env python3
"""
Train all fraud detection models.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import FraudDetectionModels
from evaluate import ModelEvaluator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def main():
    print("Training Fraud Detection Models...")
    
    # Load processed data
    processed_path = Path('data/processed/fraud_data_processed.csv')
    if not processed_path.exists():
        print("Error: Processed data not found. Run EDA pipeline first.")
        return
    
    df = pd.read_csv(processed_path)
    
    # Prepare features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target and irrelevant columns
    exclude_cols = ['class', 'user_id', 'device_id', 'ip_address', 
                   'signup_time', 'purchase_time', 'ip_int', 'lower_int', 'upper_int']
    feature_cols = [col for col in numerical_features if col not in exclude_cols]
    
    # Prepare X and y
    X = df[feature_cols].fillna(0)
    y = df['class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training on {X_train.shape[0]:,} samples with {X_train.shape[1]} features")
    
    # Initialize
    model_builder = FraudDetectionModels(random_state=42)
    evaluator = ModelEvaluator()
    
    # Handle imbalance
    X_train_resampled, y_train_resampled = model_builder.handle_imbalance(
        X_train, y_train, method='smote'
    )
    
    # Train models
    models = {
        'Logistic_Regression': model_builder.create_baseline_model(),
        'Random_Forest': model_builder.create_random_forest(n_estimators=150, max_depth=15),
        'XGBoost': model_builder.create_xgboost(n_estimators=150, max_depth=5, learning_rate=0.05),
        'LightGBM': model_builder.create_lightgbm(n_estimators=150, max_depth=7, learning_rate=0.05)
    }
    
    # Train and evaluate each model
    trained_models = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print(f"{'='*50}")
        
        # Train
        trained_model = model_builder.train_model(model, X_train_resampled, y_train_resampled)
        trained_models[name] = trained_model
        
        # Evaluate
        evaluator.evaluate_model(trained_model, X_test, y_test, name)
        
        # Save model
        model_builder.save_model(trained_model, name, 'models/')
    
    # Compare all models
    print(f"\n{'='*50}")
    print("MODEL COMPARISON")
    print(f"{'='*50}")
    comparison_df = evaluator.compare_models()
    
    if comparison_df is not None:
        # Find best model
        best_model_row = comparison_df.loc[comparison_df['f1_score'].idxmax()]
        print(f"\nBest Model: {best_model_row['Model']}")
        print(f"Best F1 Score: {best_model_row['f1_score']:.4f}")
        
        # Save comparison results
        comparison_df.to_csv('reports/model_comparison.csv', index=False)
        print(f"\nComparison saved to: reports/model_comparison.csv")
    
    print(f"\nAll models trained and saved in 'models/' directory")


if __name__ == '__main__':
    main()