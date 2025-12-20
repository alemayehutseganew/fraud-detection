#!/usr/bin/env python3
"""
Main pipeline script for fraud detection project.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_loader import DataLoader
from preprocessing import FraudDataPreprocessor, CreditCardPreprocessor
from feature_engineer import FeatureEngineer
from models import FraudDetectionModels
from evaluate import ModelEvaluator
from explain import SHAPExplainer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def run_eda_pipeline():
    """Run exploratory data analysis pipeline."""
    print("\n" + "="*60)
    print("RUNNING EDA PIPELINE")
    print("="*60)
    
    # Load data
    loader = DataLoader('data/raw')
    data_dict = loader.load_all_data()
    
    # Get fraud data
    fraud_df = data_dict['fraud_data']
    ip_df = data_dict['ip_data']
    credit_df = data_dict['credit_data']
    
    # Initial analysis
    print(f"\nFraud Data Info:")
    print(f"Shape: {fraud_df.shape}")
    print(f"Columns: {list(fraud_df.columns)}")
    
    print(f"\nClass Distribution:")
    class_dist = fraud_df['class'].value_counts()
    for cls, count in class_dist.items():
        percentage = (count / len(fraud_df)) * 100
        print(f"  Class {cls}: {count:,} transactions ({percentage:.2f}%)")
    
    # Basic preprocessing
    preprocessor = FraudDataPreprocessor()
    fraud_clean = preprocessor.clean_data(fraud_df)
    
    # Merge with IP data
    fraud_with_country = preprocessor.merge_with_ip_data(fraud_clean, ip_df)
    
    # Create time features
    fraud_with_features = preprocessor.create_time_features(fraud_with_country)
    
    # Save processed data
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    fraud_with_features.to_csv(output_dir / 'fraud_data_processed.csv', index=False)
    print(f"\nProcessed data saved to: {output_dir / 'fraud_data_processed.csv'}")
    
    return fraud_with_features


def run_modeling_pipeline():
    """Run modeling pipeline."""
    print("\n" + "="*60)
    print("RUNNING MODELING PIPELINE")
    print("="*60)
    
    # Load processed data
    processed_path = Path('data/processed/fraud_data_processed.csv')
    if not processed_path.exists():
        print("Processed data not found. Running EDA pipeline first...")
        df = run_eda_pipeline()
    else:
        df = pd.read_csv(processed_path)
    
    # Prepare features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_all_features(df)
    
    # Select features for modeling
    numerical_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target and irrelevant columns
    exclude_cols = ['class', 'user_id', 'device_id', 'ip_address', 
                   'signup_time', 'purchase_time', 'ip_int', 'lower_int', 'upper_int']
    feature_cols = [col for col in numerical_features if col not in exclude_cols]
    
    # Prepare X and y
    X = df_features[feature_cols].fillna(0)
    y = df_features['class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\nData Split:")
    print(f"  Training: {X_train.shape[0]:,} samples")
    print(f"  Testing:  {X_test.shape[0]:,} samples")
    print(f"  Features: {X_train.shape[1]}")
    
    # Initialize models
    model_builder = FraudDetectionModels(random_state=42)
    evaluator = ModelEvaluator()
    
    # Train baseline model
    print("\nTraining Baseline Model (Logistic Regression)...")
    baseline_model = model_builder.create_baseline_model()
    baseline_model = model_builder.train_model(baseline_model, X_train, y_train)
    baseline_metrics = evaluator.evaluate_model(baseline_model, X_test, y_test, 'Baseline')
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = model_builder.create_random_forest(n_estimators=100, max_depth=10)
    rf_model = model_builder.train_model(rf_model, X_train, y_train)
    rf_metrics = evaluator.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model = model_builder.create_xgboost(n_estimators=100, max_depth=3, learning_rate=0.1)
    xgb_model = model_builder.train_model(xgb_model, X_train, y_train)
    xgb_metrics = evaluator.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    
    # Compare models
    comparison_df = evaluator.compare_models()
    print(f"\nModel Comparison:")
    print(comparison_df.to_string())
    
    # Save best model
    best_model_name = comparison_df.loc[comparison_df['f1_score'].idxmax(), 'Model']
    if best_model_name == 'Random Forest':
        best_model = rf_model
    elif best_model_name == 'XGBoost':
        best_model = xgb_model
    else:
        best_model = baseline_model
    
    model_builder.save_model(best_model, 'best_model', 'models/')
    
    return best_model, X_train, X_test, y_test, feature_cols


def run_explainability_pipeline():
    """Run model explainability pipeline."""
    print("\n" + "="*60)
    print("RUNNING EXPLAINABILITY PIPELINE")
    print("="*60)
    
    # Load best model
    from models import FraudDetectionModels
    model_builder = FraudDetectionModels()
    
    try:
        best_model = model_builder.load_model('best_model', 'models/')
    except:
        print("Best model not found. Running modeling pipeline first...")
        best_model, X_train, X_test, y_test, feature_cols = run_modeling_pipeline()
    else:
        # Load data
        processed_path = Path('data/processed/fraud_data_processed.csv')
        df = pd.read_csv(processed_path)
        
        # Prepare features
        from feature_engineer import FeatureEngineer
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.create_all_features(df)
        
        # Select features
        numerical_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['class', 'user_id', 'device_id', 'ip_address', 
                       'signup_time', 'purchase_time', 'ip_int', 'lower_int', 'upper_int']
        feature_cols = [col for col in numerical_features if col not in exclude_cols]
        
        # Prepare X and y
        X = df_features[feature_cols].fillna(0)
        y = df_features['class']
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # SHAP Analysis
    print("\nPerforming SHAP Analysis...")
    explainer = SHAPExplainer(best_model, feature_cols)
    explainer.create_explainer(X_train, explainer_type='tree')
    
    # Calculate SHAP values
    shap_values = explainer.calculate_shap_values(X_test)
    
    # Summary plot
    print("\nGenerating SHAP Summary Plot...")
    explainer.plot_summary(X_test, max_display=15)
    
    # Feature importance
    importance_df = explainer.get_feature_importance()
    print(f"\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string())
    
    # Analyze specific cases
    print("\nAnalyzing Specific Fraud Cases...")
    cases = explainer.analyze_fraud_cases(X_test, y_test, y_pred, n_cases=2)
    
    # Business recommendations
    print("\nGenerating Business Recommendations...")
    recommendations = explainer.generate_business_recommendations(X_test, top_n=5)
    
    return best_model, explainer, recommendations


def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--task', choices=['eda', 'model', 'explain', 'all'],
                       default='all', help='Task to run')
    
    args = parser.parse_args()
    
    if args.task == 'eda' or args.task == 'all':
        run_eda_pipeline()
    
    if args.task == 'model' or args.task == 'all':
        run_modeling_pipeline()
    
    if args.task == 'explain' or args.task == 'all':
        run_explainability_pipeline()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == '__main__':
    main()