"""
Tests for metrics module
========================
Tests for detect_task and calculate_metrics functions.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.metrics import (
    detect_task, 
    calculate_metrics,
    compare_models_metrics,
    perform_mcnemar_test,
    perform_paired_t_test,
    calculate_feature_importance_scores,
    calculate_model_confidence
)


class TestDetectTask:
    """Tests for detect_task function"""
    
    def test_detect_binary_classification(self):
        """Test detection of binary classification task"""
        # Binary labels
        y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        assert detect_task(y) == 'classification'
        
        # String labels
        y = pd.Series(['yes', 'no', 'yes', 'no', 'yes'])
        assert detect_task(y) == 'classification'
    
    def test_detect_multiclass_classification(self):
        """Test detection of multiclass classification task"""
        # Integer classes
        y = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2])
        assert detect_task(y) == 'classification'
        
        # String classes
        y = pd.Series(['cat', 'dog', 'bird', 'cat', 'dog'])
        assert detect_task(y) == 'classification'
    
    def test_detect_regression(self):
        """Test detection of regression task"""
        # Continuous values
        y = np.array([1.5, 2.3, 4.7, 3.2, 5.8, 2.1])
        assert detect_task(y) == 'regression'
        
        # Many unique values
        y = np.random.randn(1000)
        assert detect_task(y) == 'regression'
        
        # Large range of integers
        y = np.arange(100, 200)
        assert detect_task(y) == 'regression'
    
    def test_edge_cases(self):
        """Test edge cases for task detection"""
        # Small number of unique floats (could be classification)
        y = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0])
        assert detect_task(y) == 'classification'
        
        # Mixed with NaN
        y = pd.Series([1, 2, np.nan, 1, 2, 3, np.nan])
        result = detect_task(y)
        assert result in ['classification', 'regression']
        
        # All NaN - should handle gracefully
        y = pd.Series([np.nan, np.nan, np.nan])
        result = detect_task(y)
        assert result in ['classification', 'regression']
    
    def test_ambiguous_input_handling(self):
        """Test handling of ambiguous inputs"""
        # Empty array
        y = np.array([])
        result = detect_task(y)
        assert result in ['classification', 'regression']
        
        # Single value
        y = np.array([1])
        result = detect_task(y)
        assert result in ['classification', 'regression']


class TestCalculateMetrics:
    """Tests for calculate_metrics function"""
    
    def test_classification_metrics(self):
        """Test classification metrics calculation"""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        y_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7],
                            [0.6, 0.4], [0.9, 0.1], [0.1, 0.9], [0.4, 0.6]])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba, task='classification')
        
        # Check required metrics exist
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 'log_loss' in metrics
        
        # Check value ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        assert metrics['log_loss'] >= 0
    
    def test_multiclass_classification_metrics(self):
        """Test multiclass classification metrics"""
        y_true = np.array([0, 1, 2, 0, 1, 2, 1, 0])
        y_pred = np.array([0, 2, 2, 0, 1, 1, 1, 0])
        
        # Create probability matrix for 3 classes
        n_samples = len(y_true)
        n_classes = 3
        y_proba = np.random.dirichlet(np.ones(n_classes), size=n_samples)
        
        metrics = calculate_metrics(y_true, y_pred, y_proba, task='classification')
        
        # Check multiclass specific metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
    
    def test_regression_metrics(self):
        """Test regression metrics calculation"""
        y_true = np.array([1.5, 2.3, 4.7, 3.2, 5.8, 2.1])
        y_pred = np.array([1.4, 2.5, 4.5, 3.0, 6.0, 2.0])
        
        metrics = calculate_metrics(y_true, y_pred, task='regression')
        
        # Check required metrics exist
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'median_ae' in metrics
        assert 'max_error' in metrics
        assert 'explained_variance' in metrics
        
        # Check value constraints
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['median_ae'] >= 0
        assert metrics['max_error'] >= 0
        assert -1 <= metrics['r2'] <= 1
    
    def test_unsupported_task_type(self):
        """Test error handling for unsupported task type"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        
        # Should not raise error, but handle gracefully
        metrics = calculate_metrics(y_true, y_pred, task='clustering')
        
        # Should return empty dict or handle the unsupported task
        assert isinstance(metrics, dict)
    
    def test_metrics_with_missing_probabilities(self):
        """Test metrics when probabilities are not provided"""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 0])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba=None, task='classification')
        
        # Basic metrics should still be calculated
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Probability-based metrics may be missing or None
        # ROC AUC requires probabilities


class TestCompareModelsMetrics:
    """Tests for model comparison functions"""
    
    def test_compare_classification_models(self):
        """Test comparison of two classification models"""
        np.random.seed(42)
        n_samples = 100
        
        y_true = np.random.randint(0, 2, n_samples)
        y_pred_a = np.random.randint(0, 2, n_samples)
        y_pred_b = np.random.randint(0, 2, n_samples)
        
        # Create probability matrices
        y_proba_a = np.random.rand(n_samples, 2)
        y_proba_a = y_proba_a / y_proba_a.sum(axis=1, keepdims=True)
        
        y_proba_b = np.random.rand(n_samples, 2)
        y_proba_b = y_proba_b / y_proba_b.sum(axis=1, keepdims=True)
        
        comparison = compare_models_metrics(
            y_true, y_pred_a, y_pred_b,
            y_proba_a, y_proba_b,
            task='classification',
            model_names=('Model A', 'Model B')
        )
        
        # Check structure
        assert 'model_a' in comparison
        assert 'model_b' in comparison
        assert 'comparison' in comparison
        assert 'statistical_tests' in comparison
        assert 'visualizations' in comparison
        
        # Check metrics exist for both models
        assert 'metrics' in comparison['model_a']
        assert 'metrics' in comparison['model_b']
        
        # Check statistical test
        assert 'test' in comparison['statistical_tests']
        assert 'p_value' in comparison['statistical_tests']
        assert 'significant' in comparison['statistical_tests']
    
    def test_mcnemar_test(self):
        """Test McNemar's test implementation"""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred_a = np.array([0, 1, 0, 1, 0, 0, 1, 1])
        y_pred_b = np.array([0, 1, 1, 1, 1, 0, 1, 0])
        
        result = perform_mcnemar_test(y_true, y_pred_a, y_pred_b)
        
        assert 'test' in result
        assert result['test'] == 'McNemar'
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'n_01' in result
        assert 'n_10' in result
        assert 'significant' in result
        assert isinstance(result['p_value'], float)
        assert 0 <= result['p_value'] <= 1
    
    def test_paired_t_test(self):
        """Test paired t-test for regression"""
        residuals_a = np.random.randn(100)
        residuals_b = np.random.randn(100) + 0.5  # Slightly different
        
        result = perform_paired_t_test(residuals_a, residuals_b)
        
        assert 'test' in result
        assert result['test'] == 'Paired t-test'
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'effect_size' in result
        assert 'mean_difference' in result
        assert 'confidence_interval' in result
        assert 'significant' in result


class TestFeatureImportance:
    """Tests for feature importance functions"""
    
    def test_calculate_feature_importance_scores(self):
        """Test feature importance calculation"""
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        importance_values = np.array([0.3, 0.1, 0.4, 0.15, 0.05])
        
        result = calculate_feature_importance_scores(
            feature_names, 
            importance_values, 
            top_k=3
        )
        
        # Check that we get top 3 features
        assert len(result) == 3
        
        # Check that features are sorted by importance
        sorted_features = list(result.keys())
        assert sorted_features[0] == 'feature_3'  # Highest importance
        assert sorted_features[1] == 'feature_1'  # Second highest
        
        # Check values are floats
        for value in result.values():
            assert isinstance(value, float)
    
    def test_feature_importance_mismatch(self):
        """Test handling of mismatched lengths"""
        feature_names = ['f1', 'f2', 'f3']
        importance_values = np.array([0.5, 0.3])  # Different length
        
        # Should handle gracefully
        result = calculate_feature_importance_scores(
            feature_names,
            importance_values,
            top_k=5
        )
        
        # Should return at most 2 features (min of lengths)
        assert len(result) <= 2


class TestModelConfidence:
    """Tests for model confidence calculation"""
    
    def test_calculate_model_confidence_binary(self):
        """Test confidence calculation for binary classification"""
        # Binary probabilities (just positive class)
        y_proba = np.array([0.9, 0.8, 0.6, 0.3, 0.95, 0.1])
        
        confidence = calculate_model_confidence(y_proba, threshold=0.7)
        
        assert 'mean_confidence' in confidence
        assert 'std_confidence' in confidence
        assert 'min_confidence' in confidence
        assert 'max_confidence' in confidence
        assert 'high_confidence_ratio' in confidence
        assert 'low_confidence_ratio' in confidence
        
        # Check value ranges
        assert 0 <= confidence['mean_confidence'] <= 1
        assert 0 <= confidence['high_confidence_ratio'] <= 1
        assert 0 <= confidence['low_confidence_ratio'] <= 1
        assert confidence['high_confidence_ratio'] + confidence['low_confidence_ratio'] <= 1.01  # Allow small float error
    
    def test_calculate_model_confidence_multiclass(self):
        """Test confidence calculation for multiclass"""
        # Multiclass probabilities
        y_proba = np.array([
            [0.7, 0.2, 0.1],
            [0.4, 0.4, 0.2],
            [0.9, 0.05, 0.05],
            [0.33, 0.33, 0.34]
        ])
        
        confidence = calculate_model_confidence(y_proba, threshold=0.5)
        
        assert 'mean_confidence' in confidence
        assert confidence['mean_confidence'] > 0
        assert confidence['max_confidence'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
