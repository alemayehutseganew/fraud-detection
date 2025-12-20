#!/usr/bin/env python3
"""
Generate SHAP explanations for fraud detection models.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from explain import SHAPExplainer
from models import FraudDetectionModels
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def main():
    print("Generating SHAP Explanations...")
    
    # Load processed data
    processed_path = Path('data/processed/fraud_data_processed.csv')
    if not processed_path.exists():
        print("Error: Processed data not found.")
        return
    
    df = pd.read_csv(processed_path)
    
    # Prepare features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
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
    
    # Load best model
    model_builder = FraudDetectionModels()
    try:
        model = model_builder.load_model('best_model', 'models/')
    except:
        # Try loading XGBoost if best_model not found
        try:
            model = model_builder.load_model('XGBoost', 'models/')
        except:
            print("Error: No trained model found. Train models first.")
            return
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create SHAP explainer
    print("\nCreating SHAP Explainer...")
    explainer = SHAPExplainer(model, feature_cols)
    explainer.create_explainer(X_train, explainer_type='tree')
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values = explainer.calculate_shap_values(X_test)
    
    # Generate plots
    print("\nGenerating SHAP Plots...")
    
    # 1. Summary plot
    plt.figure(figsize=(14, 8))
    explainer.plot_summary(X_test, max_display=20)
    plt.savefig('reports/shap_summary.png', dpi=300, bbox_inches='tight')
    
    # 2. Feature importance
    importance_df = explainer.get_feature_importance()
    print(f"\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string())
    
    # Save feature importance
    importance_df.to_csv('reports/shap_feature_importance.csv', index=False)
    
    # 3. Dependence plots for top 3 features
    top_features = importance_df.head(3)['feature'].tolist()
    for feature in top_features:
        print(f"\nGenerating dependence plot for: {feature}")
        explainer.plot_dependence(X_test, feature)
        plt.savefig(f'reports/shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
    
    # 4. Analyze specific cases
    print("\nAnalyzing Specific Fraud Cases...")
    cases = explainer.analyze_fraud_cases(X_test, y_test, y_pred, n_cases=3)
    
    # 5. Business recommendations
    print("\nGenerating Business Recommendations...")
    recommendations = explainer.generate_business_recommendations(X_test, top_n=5)
    
    # Save recommendations
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv('reports/business_recommendations.csv', index=False)
    
    print(f"\nAll SHAP analysis saved to 'reports/' directory")
    print(f"1. shap_summary.png - Feature importance summary")
    print(f"2. shap_feature_importance.csv - Feature importance values")
    print(f"3. shap_dependence_*.png - Dependence plots")
    print(f"4. business_recommendations.csv - Business recommendations")


if __name__ == '__main__':
    main()