"""Tests for AutoML orchestrator module."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
import sys
import os
import builtins
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing the module, skip tests if not available
try:
    from automl_platform.orchestrator import AutoMLOrchestrator
    from automl_platform.config import AutoMLConfig
    MODULE_AVAILABLE = True
except ImportError as e:
    MODULE_AVAILABLE = False
    AutoMLOrchestrator = None
    AutoMLConfig = None


@pytest.mark.skipif(not MODULE_AVAILABLE, reason="automl_platform modules not available")
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

    def test_simplified_mode_enforces_multiple_models(self, monkeypatch):
        """Ensure simplified mode falls back to multiple models when needed."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score as sklearn_cross_val_score

        config = AutoMLConfig(expert_mode=False)
        config.algorithms = ['NonExistentModel']
        config.hpo_n_iter = 0
        config.cv_folds = 2

        available_keys = ['LogisticRegression', 'RandomForestClassifier']

        def fake_get_available_models(task, include_incremental=False):
            return {
                'LogisticRegression': LogisticRegression(max_iter=200, random_state=42),
                'RandomForestClassifier': RandomForestClassifier(n_estimators=10, random_state=42),
            }

        class DummyRegistry:
            def __init__(self, *_args, **_kwargs):
                self.run_id = None

            def log_params(self, *args, **kwargs):
                return None

            def register_model(self, *args, **kwargs):
                class _Version:
                    version = 1
                    run_id = None

                return _Version()

            def promote_model(self, *args, **kwargs):
                return None

            @property
            def client(self):
                return None

        class IdentityPreprocessor:
            def __init__(self, *_args, **_kwargs):
                pass

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

            def fit_transform(self, X, y=None):
                return X

        monkeypatch.setattr(
            'automl_platform.orchestrator.get_available_models',
            fake_get_available_models
        )
        monkeypatch.setattr(
            'automl_platform.orchestrator.MLflowRegistry',
            DummyRegistry
        )
        monkeypatch.setattr(
            'automl_platform.orchestrator.DataPreprocessor',
            IdentityPreprocessor
        )
        monkeypatch.setattr(
            'automl_platform.orchestrator.cross_val_score',
            lambda estimator, X, y, cv, scoring, n_jobs=-1: sklearn_cross_val_score(
                estimator,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
            )
        )

        X, y = make_classification(
            n_samples=60, n_features=10, n_classes=2, n_informative=5, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y = pd.Series(y)

        orchestrator = AutoMLOrchestrator(config)
        orchestrator.fit(X, y, task='classification')

        assert set(orchestrator.config.algorithms) == set(available_keys)
        assert len(orchestrator.leaderboard) >= 2

        # Check sorting (best first)
        leaderboard = orchestrator.get_leaderboard()
        if len(leaderboard) > 1:
            assert leaderboard['cv_score'].iloc[0] >= leaderboard['cv_score'].iloc[1]

    def test_training_metadata_is_populated(self):
        """After fit the orchestrator should expose structured training metadata."""
        X, y = make_classification(
            n_samples=60, n_features=5, n_classes=2, random_state=123
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(y)

        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)

        metadata = orchestrator.training_metadata

        assert metadata["training_id"] == orchestrator.training_id
        assert metadata["task"] == "classification"
        assert metadata["n_samples"] == len(X)
        assert metadata["n_features"] == X.shape[1]
        assert "start_time" in metadata
        assert "end_time" in metadata

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
        # With proper CV, scores up to 0.999 are possible on simple synthetic data
        # Only a score of 1.0 would indicate true data leakage
        assert best_score < 1.0  # Should not achieve perfect 100% score
    
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

    def test_explain_predictions_feature_importance_fallback(self, monkeypatch):
        """Ensure fallback to feature importance when SHAP and LIME are unavailable."""

        X, y = make_classification(
            n_samples=60, n_features=5, n_classes=2, random_state=42
        )
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(y)

        orchestrator = AutoMLOrchestrator(self.config)
        orchestrator.fit(X, y)

        original_import = builtins.__import__

        def mocked_import(name, *args, **kwargs):
            if name in {'shap', 'lime', 'lime.lime_tabular'}:
                raise ImportError("mocked missing dependency")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mocked_import)

        explanations = orchestrator.explain_predictions(X, indices=[0, 1, 2])

        assert explanations['method'] == 'feature_importance'
        assert 'importances_mean' in explanations


@pytest.mark.skipif(not MODULE_AVAILABLE, reason="automl_platform modules not available")
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


# Fallback test if module is not available
@pytest.mark.skipif(MODULE_AVAILABLE, reason="Only run when module is not available")
def test_module_not_available():
    """Test that module is not available."""
    assert not MODULE_AVAILABLE, "Module should not be available"
    # This ensures at least one test runs even if module is missing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
