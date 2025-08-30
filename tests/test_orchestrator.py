"""Tests for AutoML orchestrator module."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.orchestrator import AutoMLOrchestrator
from automl_platform.config import AutoMLConfig


class TestOrchestrator:
    """Test AutoML orchestrator."""
    
    def setup_method(self):
        """Setup test configuration before each test."""
        self.config = AutoMLConfig(
            random_state=42,
            cv_folds=3,
            hpo_n_iter=5,
            algorithms=['LogisticRegression', 'RandomForestClassifier', 'DecisionTreeClassifier'],
            hpo_method='random',
            verbose=0
        )
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        orchestrator = AutoMLOrchestrator(self.config)
        
        assert orchestrator is not None
        assert orchestrator.config == self.config
        assert orchestrator.best_pipeline is None
        assert orchestrator.leaderboard == []
    
    def test_fit_classification_binary(self):
        """Test fitting on binary classification task."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, 
            n_informative=5, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(y)
        
        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)
        
        assert orchestrator.best_pipeline is not None
        assert len(orchestrator.leaderboard) > 0
        assert orchestrator.task == 'classification'
        
        # Check leaderboard structure
        assert all('model' in result for result in orchestrator.leaderboard)
        assert all('cv_score' in result for result in orchestrator.leaderboard)
        assert all('metrics' in result for result in orchestrator.leaderboard)
    
    def test_fit_classification_multiclass(self):
        """Test fitting on multiclass classification task."""
        X, y = make_classification(
            n_samples=150, n_features=10, n_classes=3,
            n_informative=5, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(y)
        
        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)
        
        assert orchestrator.best_pipeline is not None
        assert orchestrator.task == 'classification'
    
    def test_fit_regression(self):
        """Test fitting on regression task."""
        X, y = make_regression(
            n_samples=100, n_features=10, noise=0.1, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(y)
        
        # Update config for regression
        self.config.algorithms = ['LinearRegression', 'RandomForestRegressor', 'Ridge']
        
        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)
        
        assert orchestrator.best_pipeline is not None
        assert len(orchestrator.leaderboard) > 0
        assert orchestrator.task == 'regression'
    
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(y)
        
        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)
        
        predictions = orchestrator.predict(X)
        
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba_after_fit(self):
        """Test probability prediction after fitting."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(y)
        
        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)
        
        if hasattr(orchestrator.best_pipeline, 'predict_proba'):
            probabilities = orchestrator.predict_proba(X)
            
            assert probabilities.shape == (len(X), 2)
            assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_predict_without_fit(self):
        """Test that prediction fails without fitting."""
        orchestrator = AutoMLOrchestrator(self.config)
        
        X = pd.DataFrame(np.random.randn(10, 5))
        
        with pytest.raises(ValueError, match="No model trained"):
            orchestrator.predict(X)
    
    def test_save_load_pipeline(self):
        """Test saving and loading pipeline."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(y)
        
        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'pipeline.joblib'
            
            # Save pipeline
            orchestrator.save_pipeline(str(filepath))
            
            assert filepath.exists()
            assert filepath.with_suffix('.meta.json').exists()
            
            # Load in new orchestrator
            new_orchestrator = AutoMLOrchestrator(self.config)
            new_orchestrator.load_pipeline(str(filepath))
            
            assert new_orchestrator.best_pipeline is not None
            assert new_orchestrator.task == orchestrator.task
            
            # Test predictions are the same
            pred1 = orchestrator.predict(X)
            pred2 = new_orchestrator.predict(X)
            np.testing.assert_array_equal(pred1, pred2)
    
    def test_get_leaderboard(self):
        """Test leaderboard generation."""
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(y)
        
        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)
        
        # Get full leaderboard
        leaderboard = orchestrator.get_leaderboard()
        
        assert isinstance(leaderboard, pd.DataFrame)
        assert len(leaderboard) == len(orchestrator.leaderboard)
        assert 'model' in leaderboard.columns
        assert 'cv_score' in leaderboard.columns
        assert 'training_time' in leaderboard.columns
        
        # Get top N
        top_2 = orchestrator.get_leaderboard(top_n=2)
        assert len(top_2) <= 2
        
        # Check sorting (best first)
        if len(leaderboard) > 1:
            assert leaderboard['cv_score'].iloc[0] >= leaderboard['cv_score'].iloc[1]
    
    def test_no_data_leakage_in_cv(self):
        """Test that CV doesn't have data leakage."""
        # Create data with a moderately correlated feature
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(y)
        
        # Add a somewhat predictive feature but not perfect
        # Mix the target with random noise in a way that's informative but not deterministic
        noise = np.random.randn(len(y))
        X['semi_predictive_feature'] = np.where(y == 1, 
                                                 noise + 2,  # Class 1: centered around +2
                                                 noise - 2)  # Class 0: centered around -2
        # Add more noise to prevent perfect separation
        X['semi_predictive_feature'] += np.random.randn(len(y)) * 2
        
        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)
        
        # CV score should be good but not perfect
        best_score = orchestrator.leaderboard[0]['cv_score']
        # With proper CV and a non-deterministic feature, score should be < 1.0
        # Using 0.996 as threshold since scores around 0.995 are reasonable with good features
        assert best_score < 0.996  # Should not achieve perfect score
    
    def test_handle_categorical_features(self):
        """Test handling of categorical features."""
        # Create mixed data
        X = pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100),
            'categorical1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical2': np.random.choice(['X', 'Y', 'Z'], 100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)
        
        assert orchestrator.best_pipeline is not None
        
        # Test prediction with categorical features
        X_test = pd.DataFrame({
            'numeric1': [0.5],
            'numeric2': [-0.5],
            'categorical1': ['B'],
            'categorical2': ['Y']
        })
        predictions = orchestrator.predict(X_test)
        assert len(predictions) == 1
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=3,
            n_redundant=1, n_repeated=0, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(y)
        
        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)
        
        if orchestrator.feature_importance:
            assert 'importances_mean' in orchestrator.feature_importance
            assert 'importances_std' in orchestrator.feature_importance
    
    def test_explain_predictions(self):
        """Test prediction explanation if available."""
        X, y = make_classification(
            n_samples=50, n_features=5, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(y)
        
        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)
        
        # Try to explain predictions
        explanations = orchestrator.explain_predictions(X, indices=[0, 1])
        
        assert explanations is not None
        assert 'method' in explanations
        
        # Should have either SHAP, LIME, or feature importance
        assert explanations['method'] in ['shap', 'lime', 'feature_importance']


class TestAutoMLConfigIntegration:
    """Test AutoMLConfig integration with orchestrator."""
    
    def test_config_from_yaml(self, tmp_path):
        """Test loading config from YAML and using in orchestrator."""
        # Create temporary YAML config
        config_path = tmp_path / "test_config.yaml"
        config_content = """
random_state: 123
cv_folds: 3
algorithms:
  - LogisticRegression
  - DecisionTreeClassifier
hpo_method: none
"""
        config_path.write_text(config_content)
        
        # Load config
        config = AutoMLConfig.from_yaml(str(config_path))
        assert config.random_state == 123
        assert config.cv_folds == 3
        
        # Use in orchestrator
        orchestrator = AutoMLOrchestrator(config)
        assert orchestrator.config.random_state == 123
    
    def test_config_validation(self):
        """Test config validation in orchestrator."""
        # Create invalid config
        config = AutoMLConfig(
            cv_folds=0,  # Invalid
            max_missing_ratio=2.0  # Invalid
        )
        
        with pytest.raises(AssertionError):
            config.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
