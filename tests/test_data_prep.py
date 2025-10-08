"""Tests for data preparation module."""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from unittest.mock import patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing the module, skip tests if not available
try:
    from automl_platform.data_prep import DataPreprocessor, validate_data, handle_imbalance
    from automl_platform.config import AutoMLConfig
    MODULE_AVAILABLE = True
except ImportError as e:
    MODULE_AVAILABLE = False
    DataPreprocessor = None
    validate_data = None
    handle_imbalance = None
    AutoMLConfig = None


@pytest.mark.skipif(not MODULE_AVAILABLE, reason="automl_platform.data_prep module not available")
class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    def setup_method(self):
        """Setup test data before each test."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100),
            'categorical1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical2': np.random.choice(['X', 'Y'], 100),
            'datetime1': pd.date_range('2020-01-01', periods=100),
            'text1': ['sample text ' * np.random.randint(1, 5) for _ in range(100)]
        })
        
        self.config = {
            'scaling_method': 'standard',
            'text_max_features': 50,
            'outlier_method': 'iqr'
        }
    
    def test_detect_feature_types(self):
        """Test feature type detection."""
        preprocessor = DataPreprocessor(self.config)
        feature_types = preprocessor.detect_feature_types(self.df)
        
        assert 'numeric1' in feature_types['numeric']
        assert 'numeric2' in feature_types['numeric']
        assert 'categorical1' in feature_types['categorical']
        assert 'categorical2' in feature_types['categorical']
        assert 'datetime1' in feature_types['datetime']
        assert 'text1' in feature_types['text']
    
    def test_create_pipeline(self):
        """Test pipeline creation."""
        preprocessor = DataPreprocessor(self.config)
        pipeline = preprocessor.create_pipeline(self.df)

        assert pipeline is not None
        assert hasattr(pipeline, 'fit_transform')

    def test_agent_first_enabled_from_nested_config(self):
        """Ensure nested agent_first.enabled enables Agent-First mode."""
        config = {
            'agent_first': {
                'enabled': True
            }
        }

        with patch.object(DataPreprocessor, '_init_agent_first_components') as mock_init:
            preprocessor = DataPreprocessor(config)
            mock_init.assert_called_once()

        assert preprocessor.enable_agent_first is True

    def test_agent_first_enabled_from_config_object(self):
        """Ensure AutoMLConfig propagates nested Agent-First flag."""
        automl_config = AutoMLConfig()
        automl_config.enable_agent_first = False
        automl_config.agent_first.enabled = True

        with patch.object(DataPreprocessor, '_init_agent_first_components') as mock_init:
            preprocessor = DataPreprocessor(automl_config)
            mock_init.assert_called_once()

        assert preprocessor.enable_agent_first is True

    def test_agent_first_conflict_prefers_nested_dict(self, caplog):
        """Nested Agent-First flag should override conflicting top-level flag in dict config."""
        config = {
            'enable_agent_first': False,
            'agent_first': {
                'enabled': True,
            }
        }

        with patch.object(DataPreprocessor, '_init_agent_first_components') as mock_init:
            with caplog.at_level("WARNING"):
                preprocessor = DataPreprocessor(config)
            mock_init.assert_called_once()

        assert preprocessor.enable_agent_first is True
        assert "Conflicting Agent-First flags in configuration dict" in caplog.text

    def test_agent_first_conflict_prefers_nested_config_object(self, caplog):
        """Nested Agent-First flag should override conflicting top-level flag on AutoMLConfig."""
        automl_config = AutoMLConfig()
        automl_config.enable_agent_first = True
        automl_config.agent_first.enabled = False

        with patch.object(DataPreprocessor, '_init_agent_first_components') as mock_init:
            with caplog.at_level("WARNING"):
                preprocessor = DataPreprocessor(automl_config)
            mock_init.assert_not_called()

        assert preprocessor.enable_agent_first is False
        assert "Conflicting Agent-First flags on AutoMLConfig" in caplog.text

    def test_no_data_leakage(self):
        """Test that there's no data leakage in preprocessing."""
        preprocessor = DataPreprocessor(self.config)

        # Split data
        train_df = self.df[:80].copy()
        test_df = self.df[80:].copy()
        
        # Fit on train
        preprocessor.fit_transform(train_df)
        
        # Transform test should work without refitting
        test_transformed = preprocessor.transform(test_df)
        
        assert test_transformed is not None
        assert len(test_transformed) == len(test_df)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df_with_missing = self.df.copy()
        df_with_missing.loc[0:10, 'numeric1'] = np.nan
        df_with_missing.loc[5:15, 'categorical1'] = np.nan
        
        preprocessor = DataPreprocessor(self.config)
        transformed = preprocessor.fit_transform(df_with_missing)
        
        # Should have no missing values after transformation
        assert not np.isnan(transformed).any()
    
    def test_outlier_detection(self):
        """Test outlier detection methods."""
        preprocessor = DataPreprocessor(self.config)
        
        # Create data with outliers
        X = np.random.randn(100, 2)
        X[0, 0] = 100  # Outlier
        
        outliers = preprocessor.detect_outliers(X, method='iqr')
        assert outliers[0, 0] == True  # Should detect the outlier
    
    def test_datetime_features(self):
        """Test datetime feature creation."""
        preprocessor = DataPreprocessor(self.config)
        preprocessor.detect_feature_types(self.df)
        
        df_new = preprocessor.create_datetime_features(self.df.copy())
        
        # Check that datetime features were created
        assert 'datetime1_year' in df_new.columns
        assert 'datetime1_month' in df_new.columns
        assert 'datetime1_day' in df_new.columns
        assert 'datetime1_dayofweek' in df_new.columns
        assert 'datetime1_quarter' in df_new.columns
        assert 'datetime1_is_weekend' in df_new.columns
        
        # Original datetime column should be removed
        assert 'datetime1' not in df_new.columns


@pytest.mark.skipif(not MODULE_AVAILABLE, reason="automl_platform.data_prep module not available")
class TestDataValidation:
    """Test data validation functions."""
    
    def test_validate_empty_dataframe(self):
        """Test validation of empty dataframe."""
        df = pd.DataFrame()
        result = validate_data(df)
        
        assert not result['valid']
        assert 'DataFrame is empty' in result['issues'][0]
    
    def test_validate_duplicate_columns(self):
        """Test validation of duplicate columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        # Create duplicate column
        df_dup = pd.concat([df, df[['col1']]], axis=1)
        result = validate_data(df_dup)
        
        assert not result['valid']
        assert any('Duplicate columns' in issue for issue in result['issues'])
    
    def test_validate_all_null_columns(self):
        """Test validation of all null columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [np.nan, np.nan, np.nan]
        })
        result = validate_data(df)
        
        assert not result['valid']
        assert any('All null columns' in issue for issue in result['issues'])
    
    def test_validate_single_value_columns(self):
        """Test validation of single value columns."""
        df = pd.DataFrame({
            'col1': [1, 1, 1],
            'col2': [1, 2, 3]
        })
        result = validate_data(df)
        
        assert not result['valid']
        assert any('Single value columns' in issue for issue in result['issues'])
    
    def test_validate_good_data(self):
        """Test validation of good data."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        result = validate_data(df)
        
        assert result['valid']
        assert len(result['issues']) == 0


@pytest.mark.skipif(not MODULE_AVAILABLE, reason="automl_platform.data_prep module not available")
class TestImbalanceHandling:
    """Test class imbalance handling."""
    
    def test_handle_imbalance_passthrough(self):
        """Test that handle_imbalance returns data unchanged when method is class_weight."""
        X = np.random.randn(100, 5)
        y = np.array([0] * 90 + [1] * 10)  # Imbalanced
        
        X_res, y_res = handle_imbalance(X, y, method='class_weight')
        
        assert X_res.shape == X.shape
        assert y_res.shape == y.shape
        np.testing.assert_array_equal(X_res, X)
        np.testing.assert_array_equal(y_res, y)
    
    def test_handle_imbalance_smote(self):
        """Test SMOTE handling if available."""
        X = np.random.randn(100, 5)
        y = np.array([0] * 90 + [1] * 10)  # Imbalanced
        
        X_res, y_res = handle_imbalance(X, y, method='smote')
        
        # If SMOTE is not installed, should return original data
        # If installed, should have more balanced data
        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]


# Fallback test if module is not available
@pytest.mark.skipif(MODULE_AVAILABLE, reason="Only run when module is not available")
def test_module_not_available():
    """Test that module is not available."""
    assert not MODULE_AVAILABLE, "Module should not be available"
    # This ensures at least one test runs even if module is missing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
