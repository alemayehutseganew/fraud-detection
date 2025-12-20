"""
Model evaluation utilities for fraud detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluate fraud detection models."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate all evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name='Model'):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name='Model'):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (AP={pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return precision, recall, pr_auc
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name='Model'):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fpr, tpr, roc_auc
    
    def plot_feature_importance(self, model, feature_names, top_n=20):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(feature_importance_df)), 
                    feature_importance_df['importance'])
            plt.yticks(range(len(feature_importance_df)), 
                      feature_importance_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
        else:
            print("Model doesn't have feature_importances_ attribute")
            return None
    
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """Comprehensive model evaluation."""
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Print metrics
        print(f"\n{'='*50}")
        print(f"Evaluation Results for {model_name}")
        print(f"{'='*50}")
        for metric_name, value in metrics.items():
            print(f"{metric_name:15}: {value:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        # Plot metrics
        if y_pred_proba is not None:
            self.plot_confusion_matrix(y_test, y_pred, model_name)
            self.plot_precision_recall_curve(y_test, y_pred_proba, model_name)
            self.plot_roc_curve(y_test, y_pred_proba, model_name)
        
        return metrics
    
    def compare_models(self):
        """Compare all evaluated models."""
        if not self.results:
            print("No models evaluated yet.")
            return None
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
        
        if available_metrics:
            fig, axes = plt.subplots(1, len(available_metrics), figsize=(15, 5))
            if len(available_metrics) == 1:
                axes = [axes]
            
            for idx, metric in enumerate(available_metrics):
                ax = axes[idx]
                comparison_df.plot(kind='bar', x='Model', y=metric, ax=ax, legend=False)
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
        
        return comparison_df