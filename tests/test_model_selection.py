"""Tests for model selection module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.model_selection import (
    get_available_models,
    get_cv_splitter,
    get_param_grid,
    tune_model,
    try_optuna
)


class TestModelSelection:
    """Test model selection functions."""
    
    def test_get_available_models_classification(self):
        """Test getting classification models."""
        models = get_available_models('classification')
        
        assert len(models) > 0
        assert any(name in models for name in ['LogisticRegression', 'RandomForestClassifier', 'DecisionTreeClassifier'])
        
        # Check that models can be instantiated
        for name, model in list(models.items())[:3]:  # Test first 3 models
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
    
    def test_get_available_models_regression(self):
        """Test getting regression models."""
        models = get_available_models('regression')
        
        assert len(models) > 0
        assert any(name in models for name in ['LinearRegression', 'RandomForestRegressor', 'Ridge'])
        
        # Check that models can be instantiated
        for name, model in list(models.items())[:3]:  # Test first 3 models
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
    
    def test_get_cv_splitter_classification(self):
        """Test CV splitter for classification."""
        cv = get_cv_splitter('classification', n_splits=5, random_state=42)
        
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        
        splits = list(cv.split(X, y))
        assert len(splits) == 5
        
        # Check stratification (approximately equal class distribution)
        for train_idx, test_idx in splits:
            train_ratio = np.mean(y[train_idx])
            test_ratio = np.mean(y[test_idx])
            # Allow some tolerance for small datasets
            assert abs(train_ratio - test_ratio) < 0.3
    
    def test_get_cv_splitter_regression(self):
        """Test CV splitter for regression."""
        cv = get_cv_splitter('regression', n_splits=5, random_state=42)
        
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        
        splits = list(cv.split(X, y))
        assert len(splits) == 5
        
        # Check that all indices are used
        all_test_indices = []
        for train_idx, test_idx in splits:
            all_test_indices.extend(test_idx)
        assert len(set(all_test_indices)) == len(X)
    
    def test_get_cv_splitter_timeseries(self):
        """Test CV splitter for time series."""
        cv = get_cv_splitter('timeseries', n_splits=3)
        
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        splits = list(cv.split(X, y))
        assert len(splits) == 3
        
        # Check that test sets are sequential and non-overlapping
        prev_test_max = -1
        for train_idx, test_idx in splits:
            assert min(test_idx) > prev_test_max
            assert max(train_idx) < min(test_idx)  # Training is before test
            prev_test_max = max(test_idx)
    
    def test_get_param_grid(self):
        """Test parameter grid generation."""
        # Test known models
        grid_rf = get_param_grid('RandomForestClassifier')
        assert 'n_estimators' in grid_rf
        assert 'max_depth' in grid_rf
        assert isinstance(grid_rf['n_estimators'], list)
        
        grid_lr = get_param_grid('LogisticRegression')
        assert 'C' in grid_lr
        assert 'penalty' in grid_lr
        
        # Test unknown model
        grid_unknown = get_param_grid('UnknownModel')
        assert grid_unknown == {}
    
    def test_tune_model_no_grid(self):
        """Test tune_model with no parameter grid."""
        from sklearn.linear_model import LogisticRegression
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = LogisticRegression(max_iter=100)
        cv = get_cv_splitter('classification', n_splits=3)
        
        tuned_model, params = tune_model(
            model, X, y, param_grid={}, cv=cv, scoring='accuracy', n_iter=5
        )
        
        assert tuned_model is not None
        assert params == {}
    
    def test_tune_model_with_grid(self):
        """Test tune_model with parameter grid."""
        from sklearn.tree import DecisionTreeClassifier
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
        cv = get_cv_splitter('classification', n_splits=3)
        
        tuned_model, params = tune_model(
            model, X, y, param_grid=param_grid, cv=cv, scoring='accuracy', n_iter=5
        )
        
        assert tuned_model is not None
        assert len(params) > 0
        assert 'max_depth' in params
        assert 'min_samples_split' in params


class TestOptuna:
    """Test Optuna integration."""
    
    def test_try_optuna_classification(self):
        """Test Optuna for classification if available."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        cv = get_cv_splitter('classification', n_splits=3)
        
        # Test with RandomForest
        best_model, best_params = try_optuna(
            'RandomForestClassifier', X, y, 'classification', 
            cv, 'accuracy', n_trials=5
        )
        
        # If Optuna is available, should return tuned model
        # If not, should return None
        if best_model is not None:
            assert hasattr(best_model, 'fit')
            assert hasattr(best_model, 'predict')
            assert len(best_params) > 0
    
    def test_try_optuna_regression(self):
        """Test Optuna for regression if available."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        cv = get_cv_splitter('regression', n_splits=3)
        
        # Test with GradientBoosting
        best_model, best_params = try_optuna(
            'GradientBoostingRegressor', X, y, 'regression',
            cv, 'neg_mean_squared_error', n_trials=5
        )
        
        # If Optuna is available, should return tuned model
        # If not, should return None
        if best_model is not None:
            assert hasattr(best_model, 'fit')
            assert hasattr(best_model, 'predict')
            assert len(best_params) > 0
    
    def test_try_optuna_unsupported_model(self):
        """Test Optuna with unsupported model."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        cv = get_cv_splitter('classification', n_splits=3)
        
        # Test with unsupported model
        best_model, best_params = try_optuna(
            'LogisticRegression', X, y, 'classification',
            cv, 'accuracy', n_trials=5
        )
        
        # Should return None for unsupported models
        assert best_model is None or best_params == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
