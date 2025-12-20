"""
Model explainability using SHAP.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """SHAP explainability for fraud detection models."""
    
    def __init__(self, model, feature_names):
        """Initialize SHAP explainer."""
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_train, explainer_type='tree'):
        """Create SHAP explainer."""
        if explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_train)
        elif explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_train)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        print(f"Created {explainer_type} explainer")
        return self.explainer
    
    def calculate_shap_values(self, X):
        """Calculate SHAP values and standardize output to numpy arrays."""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")

        res = self.explainer.shap_values(X)
        # Handle SHAP Explanation objects (newer API)
        if hasattr(res, 'values'):
            res = res.values

        try:
            res = np.array(res)
        except Exception:
            pass

        self.shap_values = res
        print(f"Calculated SHAP values for {len(X)} samples")
        return self.shap_values
    
    def plot_summary(self, X, max_display=20):
        """Plot SHAP summary plot."""
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        
        # For binary classification
        if isinstance(self.shap_values, list):
            shap_values = self.shap_values[1]  # Use values for class 1 (fraud)
        else:
            shap_values = self.shap_values
        
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Summary Plot - Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_force_plot(self, X, sample_idx, expected_value=None):
        """Plot individual force plot."""
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        if expected_value is None:
            expected_value = self.explainer.expected_value
        
        # For binary classification
        if isinstance(self.shap_values, list):
            shap_values = self.shap_values[1]
            expected_value = expected_value[1] if isinstance(expected_value, list) else expected_value
        else:
            shap_values = self.shap_values
        
        plt.figure(figsize=(12, 4))
        shap.force_plot(
            expected_value,
            shap_values[sample_idx, :],
            X.iloc[sample_idx, :],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Force Plot - Sample {sample_idx}')
        plt.tight_layout()
        plt.show()
    
    def plot_dependence(self, X, feature_name, interaction_index='auto'):
        """Plot SHAP dependence plot."""
        if self.shap_values is None:
            self.calculate_shap_values(X)
        
        # For binary classification
        if isinstance(self.shap_values, list):
            shap_values = self.shap_values[1]
        else:
            shap_values = self.shap_values
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_index,
            show=False
        )
        plt.title(f'SHAP Dependence Plot - {feature_name}')
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self):
        """Get SHAP feature importance (robust to various SHAP output shapes)."""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated.")

        sv = self.shap_values

        # If list, select class 1 if available (binary classification), else first element
        if isinstance(sv, list):
            if len(sv) > 1:
                sv = sv[1]
            else:
                sv = sv[0]

        sv = np.array(sv)

        # Handle 3D arrays (several possible axis orders):
        # - (n_samples, n_features, n_classes)
        # - (n_samples, n_classes, n_features)
        # - (n_classes, n_samples, n_features)
        n_feat = len(self.feature_names)
        if sv.ndim == 3:
            s0, s1, s2 = sv.shape
            # case: (n_samples, n_features, n_classes)
            if s1 == n_feat:
                # choose class index 1 when available (fraud class)
                class_idx = 1 if s2 > 1 else 0
                sv = sv[:, :, class_idx]
            # case: (n_samples, n_classes, n_features)
            elif s2 == n_feat:
                class_idx = 1 if s1 > 1 else 0
                sv = sv[:, class_idx, :]
            # case: (n_classes, n_samples, n_features)
            elif s0 > 1 and s2 == n_feat:
                sv = sv[1]
            else:
                # fallback: collapse everything except last axis if it matches features
                if sv.shape[-1] == n_feat:
                    sv = sv.reshape(-1, sv.shape[-1])
                else:
                    # last resort: flatten to 2D and hope last axis is features
                    sv = sv.reshape(-1, n_feat)

        # Now sv may be 1D (per-feature), 2D (n_samples, n_features) or 3D. Compute per-feature importance robustly.
        if sv.ndim == 1:
            shap_importance = np.abs(sv)
        elif sv.ndim == 2:
            # average over samples
            shap_importance = np.abs(sv).mean(axis=0)
        elif sv.ndim == 3:
            # find which axis corresponds to features by matching length
            shapes = np.array(sv.shape)
            feat_axes = np.where(shapes == len(self.feature_names))[0]
            if len(feat_axes) > 0:
                feat_axis = int(feat_axes[0])
                # average over all axes except the feature axis
                avg_axes = tuple(i for i in range(sv.ndim) if i != feat_axis)
                shap_importance = np.abs(sv).mean(axis=avg_axes)
            else:
                # fallback: average over sample and class axes assuming features in last axis
                shap_importance = np.abs(sv).mean(axis=(0, 1))
        else:
            # unexpected shape; flatten to features
            shap_importance = np.abs(sv).mean(axis=tuple(range(sv.ndim - 1)))

        # Ensure shap_importance is a 1D numeric array
        shap_importance = np.array(shap_importance)
        # If 2D (n_features, n_classes) prefer class index 1 when available, else average across class axis
        if shap_importance.ndim == 2:
            if shap_importance.shape[1] > 1:
                # choose class 1 (fraud) if possible
                if shap_importance.shape[1] > 1:
                    shap_importance = shap_importance[:, 1]
                else:
                    shap_importance = shap_importance.mean(axis=1)
            else:
                shap_importance = shap_importance[:, 0]

        # Final check: ensure we have a numeric 1D array
        shap_importance = np.array(shap_importance)
        print(f"[SHAP DEBUG] shap_importance initial shape={shap_importance.shape}, dtype={shap_importance.dtype}")

        if shap_importance.ndim == 2:
            # Prefer selecting class index 1 if present, else average columns
            if shap_importance.shape[1] > 1 and shap_importance.shape[1] > 1:
                shap_importance = shap_importance[:, 1]
            else:
                shap_importance = shap_importance.mean(axis=1)

        # Flatten and ensure float
        shap_importance = np.asarray(shap_importance).astype(float).ravel()
        print(f"[SHAP DEBUG] shap_importance final shape={shap_importance.shape}")

        # Align feature names with shap_importance length
        n_feat = len(shap_importance)
        if n_feat != len(self.feature_names):
            print(f"Warning: number of features in SHAP ({n_feat}) does not match provided feature_names ({len(self.feature_names)}). Aligning to SHAP output.")
            if len(self.feature_names) >= n_feat:
                aligned_names = self.feature_names[:n_feat]
            else:
                # Create generic feature names if we don't have enough
                aligned_names = [f'feature_{i}' for i in range(n_feat)]
        else:
            aligned_names = self.feature_names

        importance_df = pd.DataFrame({
            'feature': aligned_names,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)

        return importance_df
    
    def analyze_fraud_cases(self, X, y_true, y_pred, n_cases=3):
        """Analyze specific fraud cases."""
        cases = {}
        
        # Find indices for different case types
        tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
        fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
        fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
        
        if len(tp_idx) > 0:
            print(f"\nTrue Positive Cases (Correctly identified fraud):")
            for i in range(min(n_cases, len(tp_idx))):
                idx = tp_idx[i]
                cases[f'TP_{i}'] = {'idx': idx, 'type': 'True Positive'}
                self.plot_force_plot(X, idx)
        
        if len(fp_idx) > 0:
            print(f"\nFalse Positive Cases (Legitimate flagged as fraud):")
            for i in range(min(n_cases, len(fp_idx))):
                idx = fp_idx[i]
                cases[f'FP_{i}'] = {'idx': idx, 'type': 'False Positive'}
                self.plot_force_plot(X, idx)
        
        if len(fn_idx) > 0:
            print(f"\nFalse Negative Cases (Missed fraud):")
            for i in range(min(n_cases, len(fn_idx))):
                idx = fn_idx[i]
                cases[f'FN_{i}'] = {'idx': idx, 'type': 'False Negative'}
                self.plot_force_plot(X, idx)
        
        return cases
    
    def generate_business_recommendations(self, X, top_n=5):
        """Generate business recommendations from SHAP analysis."""
        importance_df = self.get_feature_importance().head(top_n)
        
        print(f"\n{'='*60}")
        print("BUSINESS RECOMMENDATIONS")
        print(f"{'='*60}")
        
        recommendations = []
        
        for idx, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['shap_importance']
            
            # Get basic statistics for the feature
            if feature in X.columns:
                feature_values = X[feature]
                fraud_corr = X[feature].corr(pd.Series(self.model.predict(X)) if hasattr(self.model, 'predict') else pd.Series([0]*len(X)))
                
                recommendation = {
                    'feature': feature,
                    'importance': importance,
                    'recommendation': self._generate_single_recommendation(feature, feature_values, fraud_corr)
                }
                recommendations.append(recommendation)
                
                print(f"\nRecommendation based on '{feature}' (Importance: {importance:.3f}):")
                print(f"  → {recommendation['recommendation']}")
        
        return recommendations
    
    def _generate_single_recommendation(self, feature, values, fraud_corr):
        """Generate a single business recommendation."""
        recommendations_map = {
            'time_since_signup': "Implement additional verification for transactions within 2 hours of account creation.",
            'purchase_value': "Flag high-value transactions (> $500) for manual review.",
            'purchase_hour': "Increase monitoring during non-business hours (10 PM - 6 AM).",
            'country_fraud_rate': "Apply enhanced verification for transactions from high-risk countries.",
            'device_user_count': "Flag transactions from devices used by multiple users.",
            'transaction_velocity': "Monitor accounts with unusually high transaction frequency.",
            'same_day_purchase': "Require additional verification for same-day purchases.",
            'is_weekend': "Increase fraud monitoring during weekends.",
            'high_value_purchase': "Implement additional checks for purchases above $1000.",
            'user_transaction_count': "Monitor new users for unusual transaction patterns."
        }
        
        # Check if we have a specific recommendation
        for key in recommendations_map:
            if key in feature.lower():
                return recommendations_map[key]
        
        # Default recommendation
        if abs(fraud_corr) > 0.1:
            direction = "increases" if fraud_corr > 0 else "decreases"
            return f"Monitor transactions where '{feature}' has extreme values, as it {direction} fraud probability."
        else:
            return f"Include '{feature}' in the fraud risk scoring model due to its predictive importance."