"""
Tests for model training and evaluation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import FraudDetectionModels
from evaluate import ModelEvaluator


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],  # 90% class 0, 10% class 1
        random_state=42
    )
    
    # Convert to DataFrame for realistic testing
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


class TestFraudDetectionModels:
    """Test FraudDetectionModels class."""
    
    
    def test_create_baseline_model(self):
        """Test baseline model creation."""
        model_builder = FraudDetectionModels()
        model = model_builder.create_baseline_model()
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
    
    def test_create_random_forest(self):
        """Test random forest model creation."""
        model_builder = FraudDetectionModels()
        model = model_builder.create_random_forest(n_estimators=50, max_depth=5)
        
        assert model is not None
        assert model.n_estimators == 50
        assert model.max_depth == 5
    
    def test_create_xgboost(self):
        """Test XGBoost model creation."""
        model_builder = FraudDetectionModels()
        model = model_builder.create_xgboost(n_estimators=50, max_depth=3, learning_rate=0.1)
        
        assert model is not None
        assert model.n_estimators == 50
        assert model.max_depth == 3
    
    def test_handle_imbalance_smote(self, sample_data):
        """Test SMOTE for handling class imbalance."""
        X, y = sample_data
        model_builder = FraudDetectionModels()
        
        X_resampled, y_resampled = model_builder.handle_imbalance(X, y, method='smote')
        
        # Check shapes
        assert X_resampled.shape[0] > X.shape[0]  # SMOTE increases samples
        assert X_resampled.shape[1] == X.shape[1]
        
        # Check class balance
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1]  # Should be balanced
    
    def test_handle_imbalance_undersample(self, sample_data):
        """Test undersampling for handling class imbalance."""
        X, y = sample_data
        model_builder = FraudDetectionModels()
        
        X_resampled, y_resampled = model_builder.handle_imbalance(X, y, method='undersample')
        
        # Check shapes
        assert X_resampled.shape[0] < X.shape[0]  # Undersampling decreases samples
        
        # Check class balance
        unique, counts = np.unique(y_resampled, return_counts=True)
        assert len(unique) == 2
        assert counts[0] == counts[1]  # Should be balanced
    
    def test_train_model(self, sample_data):
        """Test model training."""
        X, y = sample_data
        
        # Split data
        X_train = X.iloc[:800]
        y_train = y.iloc[:800]
        
        model_builder = FraudDetectionModels()
        model = model_builder.create_baseline_model()
        
        trained_model = model_builder.train_model(model, X_train, y_train)
        
        assert trained_model is not None
        assert hasattr(trained_model, 'predict')
        
        # Test predictions
        predictions = trained_model.predict(X_train.head(10))
        assert len(predictions) == 10
        assert predictions.dtype in [np.int64, np.int32]
    
    def test_calculate_scale_pos_weight(self):
        """Test scale_pos_weight calculation for XGBoost."""
        model_builder = FraudDetectionModels()
        
        # Test with balanced data
        y_balanced = np.array([0, 1, 0, 1])
        weight = model_builder.calculate_scale_pos_weight(y_balanced)
        assert weight == 1.0
        
        # Test with imbalanced data
        y_imbalanced = np.array([0, 0, 0, 0, 1])  # 4:1 ratio
        weight = model_builder.calculate_scale_pos_weight(y_imbalanced)
        assert weight == 4.0


class TestModelEvaluator:
    """Test ModelEvaluator class."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        
        y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 1])
        y_pred_proba = np.random.rand(10)
        
        return y_true, y_pred, y_pred_proba
    
    def test_calculate_metrics(self, sample_predictions):
        """Test metrics calculation."""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Check all metrics are calculated
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if y_pred_proba is not None:
            expected_metrics.extend(['roc_auc', 'pr_auc'])
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
    
    def test_confusion_matrix_calculation(self, sample_predictions):
        """Test confusion matrix calculation."""
        y_true, y_pred, _ = sample_predictions
        
        evaluator = ModelEvaluator()
        
        # This is a visual test, but we can test the function doesn't crash
        try:
            cm = evaluator.plot_confusion_matrix(y_true, y_pred, 'Test Model')
            # If function returns something, check it's a valid confusion matrix
            if cm is not None:
                assert cm.shape == (2, 2)
        except Exception as e:
            pytest.fail(f"plot_confusion_matrix failed: {e}")
    
    def test_precision_recall_curve(self, sample_predictions):
        """Test precision-recall curve calculation."""
        y_true, _, y_pred_proba = sample_predictions
        
        evaluator = ModelEvaluator()
        
        try:
            precision, recall, pr_auc = evaluator.plot_precision_recall_curve(
                y_true, y_pred_proba, 'Test Model'
            )
            
            # Check returned values
            assert len(precision) == len(recall)
            assert 0 <= pr_auc <= 1
        except Exception as e:
            pytest.fail(f"plot_precision_recall_curve failed: {e}")
    
    def test_evaluate_model_with_classifier(self, sample_data):
        """Test complete model evaluation."""
        X, y = sample_data
        
        # Split data
        X_train = X.iloc[:800]
        X_test = X.iloc[800:]
        y_train = y.iloc[:800]
        y_test = y.iloc[800:]
        
        # Train a simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, X_test, y_test, 'Test Model')
        
        # Check metrics were calculated
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])