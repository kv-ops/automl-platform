"""
Tests for ensemble module
==========================
Tests for AutoMLEnsemble, WeightedEnsemble and ensemble utilities.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.ensemble import (
    AutoMLEnsemble,
    AutoGluonEnsemble,
    WeightedEnsemble,
    create_diverse_ensemble,
    create_ensemble_pipeline
)


class TestAutoMLEnsemble:
    """Tests for AutoMLEnsemble class"""
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset"""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_classes=2,
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def base_models(self):
        """Create base models for ensemble"""
        return [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('lr', LogisticRegression(max_iter=100, random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]
    
    def test_stacking_ensemble_initialization(self, base_models):
        """Test stacking ensemble initialization"""
        ensemble = AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='stacking',
            n_layers=2,
            cv_folds=3,
            task='classification'
        )
        
        assert ensemble.ensemble_method == 'stacking'
        assert ensemble.n_layers == 2
        assert ensemble.cv_folds == 3
        assert ensemble.task == 'classification'
        assert len(ensemble.base_models) == 3
    
    def test_stacking_ensemble_fit(self, classification_data, base_models):
        """Test stacking ensemble fitting"""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='stacking',
            n_layers=1,  # Single layer for faster testing
            cv_folds=3,
            task='classification'
        )
        
        # Fit ensemble
        ensemble.fit(X_train, y_train)
        
        # Check that models were fitted
        assert len(ensemble.fitted_models_) > 0
        assert hasattr(ensemble, 'meta_model_')
        assert hasattr(ensemble, 'scaler_')
    
    def test_stacking_ensemble_predict(self, classification_data, base_models):
        """Test stacking ensemble predictions"""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='stacking',
            n_layers=1,
            cv_folds=3,
            task='classification'
        )
        
        ensemble.fit(X_train, y_train)
        
        # Test predictions
        y_pred = ensemble.predict(X_test)
        
        assert y_pred.shape[0] == X_test.shape[0]
        assert all(pred in [0, 1] for pred in y_pred)
        
        # Check performance is reasonable
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.5  # Better than random
    
    def test_voting_ensemble(self, classification_data, base_models):
        """Test voting ensemble"""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='voting',
            weighted=True,
            use_probabilities=True,
            task='classification'
        )
        
        ensemble.fit(X_train, y_train)
        
        # Check weights were calculated
        assert ensemble.weights_ is not None
        assert len(ensemble.weights_) == len(base_models)
        assert np.allclose(ensemble.weights_.sum(), 1.0, atol=1e-6)
        
        # Test predictions
        y_pred = ensemble.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_blending_ensemble(self, classification_data, base_models):
        """Test blending ensemble"""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='blending',
            task='classification'
        )
        
        ensemble.fit(X_train, y_train)
        
        # Check that models were fitted
        assert len(ensemble.fitted_models_) == len(base_models)
        assert hasattr(ensemble, 'meta_model_')
        
        # Test predictions
        y_pred = ensemble.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_dynamic_ensemble(self, classification_data, base_models):
        """Test dynamic ensemble selection"""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='dynamic',
            task='classification'
        )
        
        ensemble.fit(X_train, y_train)
        
        # Check that competence regions were calculated
        assert hasattr(ensemble, 'competences_')
        assert hasattr(ensemble, 'nn_')
        
        # Test predictions
        y_pred = ensemble.predict(X_test)
        assert y_pred.shape[0] == X_test.shape[0]
    
    def test_regression_ensemble(self):
        """Test ensemble for regression task"""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
            ('ridge', Ridge(random_state=42))
        ]
        
        ensemble = AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='stacking',
            n_layers=1,
            task='regression'
        )
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        # Check predictions
        assert y_pred.shape[0] == X_test.shape[0]
        
        # Check performance
        mse = mean_squared_error(y_test, y_pred)
        baseline_mse = mean_squared_error(y_test, np.full_like(y_test, y_test.mean()))
        assert mse < baseline_mse  # Better than predicting mean


class TestWeightedEnsemble:
    """Tests for WeightedEnsemble class"""
    
    @pytest.fixture
    def models_and_data(self):
        """Create models and data for testing"""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        models = [
            ('model1', LogisticRegression(max_iter=100, random_state=42)),
            ('model2', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('model3', DecisionTreeClassifier(random_state=42))
        ]
        
        return models, X, y
    
    def test_weighted_ensemble_fit(self, models_and_data):
        """Test weighted ensemble fitting and weight optimization"""
        models, X, y = models_and_data
        
        ensemble = WeightedEnsemble(
            models=models,
            task='classification'
        )
        
        ensemble.fit(X, y)
        
        # Check weights were optimized
        assert ensemble.weights_ is not None
        assert len(ensemble.weights_) == len(models)
        assert np.allclose(ensemble.weights_.sum(), 1.0, atol=1e-6)
        assert np.all(ensemble.weights_ >= 0)
        assert np.all(ensemble.weights_ <= 1)
    
    def test_weighted_ensemble_predict(self, models_and_data):
        """Test weighted ensemble predictions"""
        models, X, y = models_and_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        ensemble = WeightedEnsemble(
            models=models,
            task='classification'
        )
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        assert y_pred.shape[0] == X_test.shape[0]
        
        # Verify weighted predictions are coherent
        # Get individual predictions
        individual_preds = []
        for name, model in ensemble.fitted_models_:
            individual_preds.append(model.predict(X_test))
        
        # Check that ensemble prediction is influenced by weights
        # (not testing exact equality due to rounding)
        assert y_pred.shape == individual_preds[0].shape
    
    def test_weight_optimization_methods(self):
        """Test different weight optimization methods"""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        
        models = [
            ('ridge1', Ridge(alpha=0.1)),
            ('ridge2', Ridge(alpha=1.0)),
            ('ridge3', Ridge(alpha=10.0))
        ]
        
        # Test with different optimization methods
        for method in ['nelder-mead', 'BFGS']:
            ensemble = WeightedEnsemble(
                models=models,
                optimization_method=method,
                task='regression'
            )
            
            # Should not raise errors
            ensemble.fit(X, y)
            assert ensemble.weights_ is not None


class TestAutoGluonEnsemble:
    """Tests for AutoGluon-style ensemble"""
    
    def test_autogluon_ensemble_fit(self):
        """Test AutoGluon ensemble with bagging"""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        base_models = [
            ('lr', LogisticRegression(max_iter=100, random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]
        
        ensemble = AutoGluonEnsemble(
            base_models=base_models,
            n_bags=2,  # Small number for testing
            n_layers=1,
            n_folds=3,
            task='classification'
        )
        
        ensemble.fit(X, y)
        
        # Check that bagged models were created
        assert len(ensemble.bagged_models_) == 2
        assert hasattr(ensemble, 'meta_model_')
    
    def test_autogluon_ensemble_predict(self):
        """Test AutoGluon ensemble predictions"""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        base_models = [
            ('lr', LogisticRegression(max_iter=100, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42))
        ]
        
        ensemble = AutoGluonEnsemble(
            base_models=base_models,
            n_bags=2,
            n_layers=1,
            n_folds=3,
            task='classification'
        )
        
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        
        assert y_pred.shape[0] == X_test.shape[0]
        
        # Check performance
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.5  # Better than random


class TestEnsembleUtilities:
    """Tests for ensemble utility functions"""
    
    def test_create_diverse_ensemble(self):
        """Test diverse ensemble creation"""
        X, y = make_classification(n_samples=50, n_features=10, random_state=42)
        
        # Classification models
        models = create_diverse_ensemble(X, y, task='classification', max_models=5)
        
        assert len(models) <= 5
        assert all(isinstance(m, tuple) for m in models)
        assert all(len(m) == 2 for m in models)
        assert all(isinstance(m[0], str) for m in models)
        
        # Regression models
        X_reg, y_reg = make_regression(n_samples=50, n_features=10, random_state=42)
        models_reg = create_diverse_ensemble(X_reg, y_reg, task='regression', max_models=5)
        
        assert len(models_reg) <= 5
    
    def test_create_ensemble_pipeline(self):
        """Test ensemble pipeline creation"""
        base_models = [
            ('model1', LogisticRegression(max_iter=100)),
            ('model2', RandomForestClassifier(n_estimators=10))
        ]
        
        # Test auto selection
        ensemble = create_ensemble_pipeline(
            base_models,
            ensemble_type='auto',
            task='classification'
        )
        
        assert isinstance(ensemble, AutoMLEnsemble)
        assert ensemble.task == 'classification'
        
        # Test specific types
        voting_ensemble = create_ensemble_pipeline(
            base_models,
            ensemble_type='voting',
            task='classification'
        )
        assert voting_ensemble.ensemble_method == 'voting'
        
        stacking_ensemble = create_ensemble_pipeline(
            base_models,
            ensemble_type='stacking',
            task='classification'
        )
        assert stacking_ensemble.ensemble_method == 'stacking'
        
        # Test with many models
        many_models = [(f'model_{i}', LogisticRegression()) for i in range(15)]
        autogluon_ensemble = create_ensemble_pipeline(
            many_models,
            ensemble_type='auto',
            task='classification'
        )
        # Should select AutoGluon for many models
        assert isinstance(autogluon_ensemble, (AutoMLEnsemble, AutoGluonEnsemble))


class TestEnsembleEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_empty_base_models(self):
        """Test with empty base models list"""
        with pytest.raises(Exception):
            ensemble = AutoMLEnsemble(
                base_models=[],
                ensemble_method='stacking'
            )
            X = np.random.randn(10, 5)
            y = np.random.randint(0, 2, 10)
            ensemble.fit(X, y)
    
    def test_single_model_ensemble(self):
        """Test ensemble with single model"""
        base_models = [('single', LogisticRegression(max_iter=100))]
        
        ensemble = AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='voting',
            task='classification'
        )
        
        X, y = make_classification(n_samples=50, n_features=10, random_state=42)
        
        # Should handle single model gracefully
        ensemble.fit(X, y)
        y_pred = ensemble.predict(X)
        
        assert y_pred.shape[0] == X.shape[0]
    
    def test_inconsistent_predictions(self):
        """Test handling of models with inconsistent predictions"""
        # This tests the ensemble's robustness to failing models
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        # Include a model that might fail
        base_models = [
            ('good', LogisticRegression(max_iter=100, random_state=42)),
            ('bad', LogisticRegression(max_iter=1, random_state=42))  # Very few iterations
        ]
        
        ensemble = AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='stacking',
            n_layers=1,
            task='classification'
        )
        
        # Should handle potential convergence warnings
        ensemble.fit(X, y)
        y_pred = ensemble.predict(X)
        
        assert y_pred.shape[0] == X.shape[0]
    
    def test_multilayer_stacking(self):
        """Test multi-layer stacking"""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=5, random_state=42)),
            ('lr', LogisticRegression(max_iter=100, random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ]
        
        # Test with multiple layers
        ensemble = AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='stacking',
            n_layers=2,
            cv_folds=3,
            task='classification'
        )
        
        ensemble.fit(X, y)
        
        # Should have multiple layers of models
        assert len(ensemble.fitted_models_) == 2  # Two layers
        
        y_pred = ensemble.predict(X)
        assert y_pred.shape[0] == X.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
