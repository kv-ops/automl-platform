"""
Unit tests for MLflow Registry Integration
==========================================
Tests for model registration, versioning, and promotion
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.mlflow_registry import MLflowRegistry, ModelStage


class TestMLflowRegistry(unittest.TestCase):
    """Test cases for MLflow Registry integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.mlflow_tracking_uri = "sqlite:///test_mlflow.db"
        self.config.mlflow_experiment_name = "test_experiments"
        
        # Mock MLflow client
        with patch('automl_platform.mlflow_registry.MlflowClient'):
            self.registry = MLflowRegistry(self.config)
            self.registry.client = Mock()
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_initialization(self):
        """Test MLflow registry initialization."""
        with patch('automl_platform.mlflow_registry.mlflow.set_tracking_uri') as mock_set_uri:
            with patch('automl_platform.mlflow_registry.mlflow.set_experiment') as mock_set_exp:
                with patch('automl_platform.mlflow_registry.MlflowClient'):
                    registry = MLflowRegistry(self.config)
                    
                    mock_set_uri.assert_called_once_with("sqlite:///test_mlflow.db")
                    mock_set_exp.assert_called_once_with("test_experiments")
                    self.assertIsNotNone(registry.client)
    
    @patch('automl_platform.mlflow_registry.mlflow.start_run')
    @patch('automl_platform.mlflow_registry.mlflow.sklearn.log_model')
    def test_register_model(self, mock_log_model, mock_start_run):
        """Test model registration with MLflow."""
        # Setup
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        # Create mock model
        model = Mock()
        model.predict.return_value = np.array([1, 0, 1])
        
        # Create mock version
        mock_version = Mock()
        mock_version.version = "1"
        mock_version.run_id = "test_run_id"
        self.registry.client.get_latest_versions.return_value = [mock_version]
        
        # Test data
        X_sample = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_sample = pd.Series([1, 0, 1])
        
        # Register model
        result = self.registry.register_model(
            model=model,
            model_name="test_model",
            metrics={"accuracy": 0.95, "f1": 0.93},
            params={"max_depth": 5, "n_estimators": 100},
            X_sample=X_sample,
            y_sample=y_sample,
            description="Test model",
            tags={"team": "data_science"}
        )
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.version, "1")
        mock_log_model.assert_called_once()
        self.registry.client.update_model_version.assert_called_once_with(
            name="test_model",
            version="1",
            description="Test model"
        )
    
    def test_promote_model(self):
        """Test model promotion to different stages."""
        # Test promotion to production
        self.registry.client.transition_model_version_stage.return_value = None
        
        result = self.registry.promote_model(
            model_name="test_model",
            version=1,
            stage=ModelStage.PRODUCTION
        )
        
        self.assertTrue(result)
        self.registry.client.transition_model_version_stage.assert_called_once_with(
            name="test_model",
            version=1,
            stage="Production"
        )
    
    def test_promote_model_failure(self):
        """Test model promotion failure handling."""
        self.registry.client.transition_model_version_stage.side_effect = Exception("Promotion failed")
        
        result = self.registry.promote_model(
            model_name="test_model",
            version=1,
            stage=ModelStage.PRODUCTION
        )
        
        self.assertFalse(result)
    
    def test_get_model_history(self):
        """Test retrieving model version history."""
        # Create mock versions
        mock_version1 = Mock()
        mock_version1.version = "2"
        mock_version1.current_stage = "Production"
        mock_version1.creation_timestamp = 1234567890000
        mock_version1.last_updated_timestamp = 1234567891000
        mock_version1.description = "Version 2"
        mock_version1.run_id = "run_2"
        
        mock_version2 = Mock()
        mock_version2.version = "1"
        mock_version2.current_stage = "Archived"
        mock_version2.creation_timestamp = 1234567880000
        mock_version2.last_updated_timestamp = 1234567881000
        mock_version2.description = "Version 1"
        mock_version2.run_id = "run_1"
        
        self.registry.client.search_model_versions.return_value = [mock_version1, mock_version2]
        
        # Create mock runs
        mock_run1 = Mock()
        mock_run1.data.metrics = {"accuracy": 0.95}
        mock_run1.data.params = {"max_depth": "5"}
        mock_run1.data.tags = {"team": "ds"}
        
        mock_run2 = Mock()
        mock_run2.data.metrics = {"accuracy": 0.92}
        mock_run2.data.params = {"max_depth": "3"}
        mock_run2.data.tags = {"team": "ds"}
        
        self.registry.client.get_run.side_effect = [mock_run1, mock_run2]
        
        # Get history
        history = self.registry.get_model_history("test_model", limit=10)
        
        # Assertions
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['version'], "2")
        self.assertEqual(history[0]['stage'], "Production")
        self.assertEqual(history[0]['metrics']['accuracy'], 0.95)
        self.assertEqual(history[1]['version'], "1")
        self.assertEqual(history[1]['stage'], "Archived")
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Create mock versions
        mock_v1 = Mock()
        mock_v1.current_stage = "Staging"
        mock_v1.creation_timestamp = 1234567890000
        mock_v1.run_id = "run_1"
        
        mock_v2 = Mock()
        mock_v2.current_stage = "Production"
        mock_v2.creation_timestamp = 1234567900000
        mock_v2.run_id = "run_2"
        
        self.registry.client.get_model_version.side_effect = [mock_v1, mock_v2]
        
        # Create mock runs
        mock_run1 = Mock()
        mock_run1.data.metrics = {"accuracy": 0.90, "f1": 0.88}
        mock_run1.data.params = {"max_depth": "3"}
        
        mock_run2 = Mock()
        mock_run2.data.metrics = {"accuracy": 0.95, "f1": 0.93}
        mock_run2.data.params = {"max_depth": "5"}
        
        self.registry.client.get_run.side_effect = [mock_run1, mock_run2]
        
        # Compare models
        comparison = self.registry.compare_models("test_model", 1, 2)
        
        # Assertions
        self.assertIn('version1', comparison)
        self.assertIn('version2', comparison)
        self.assertIn('metrics_diff', comparison)
        self.assertEqual(comparison['version1']['metrics']['accuracy'], 0.90)
        self.assertEqual(comparison['version2']['metrics']['accuracy'], 0.95)
        self.assertEqual(comparison['metrics_diff']['accuracy']['diff'], 0.05)
        self.assertAlmostEqual(comparison['metrics_diff']['accuracy']['pct_change'], 5.56, places=1)
    
    def test_rollback_model(self):
        """Test model rollback functionality."""
        # Create mock production version
        mock_current = Mock()
        mock_current.version = "3"
        
        self.registry.client.get_latest_versions.return_value = [mock_current]
        
        # Perform rollback
        result = self.registry.rollback_model("test_model", target_version=2)
        
        # Assertions
        self.assertTrue(result)
        
        # Check that current production was archived
        calls = self.registry.client.transition_model_version_stage.call_args_list
        self.assertEqual(len(calls), 2)
        
        # First call: archive current production
        self.assertEqual(calls[0], call(
            name="test_model",
            version="3",
            stage="Archived"
        ))
        
        # Second call: promote target to production
        self.assertEqual(calls[1], call(
            name="test_model",
            version=2,
            stage="Production"
        ))
    
    @patch('automl_platform.mlflow_registry.mlflow.sklearn.load_model')
    def test_load_model(self, mock_load_model):
        """Test model loading from registry."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        # Test loading by version
        model = self.registry.load_model("test_model", version=2)
        self.assertEqual(model, mock_model)
        mock_load_model.assert_called_with("models:/test_model/2")
        
        # Test loading by stage
        mock_load_model.reset_mock()
        model = self.registry.load_model("test_model", stage="Production")
        self.assertEqual(model, mock_model)
        mock_load_model.assert_called_with("models:/test_model/Production")
        
        # Test default loading (Production)
        mock_load_model.reset_mock()
        model = self.registry.load_model("test_model")
        self.assertEqual(model, mock_model)
        mock_load_model.assert_called_with("models:/test_model/Production")
    
    def test_delete_model_version(self):
        """Test model version deletion."""
        self.registry.client.delete_model_version.return_value = None
        
        result = self.registry.delete_model_version("test_model", version=1)
        
        self.assertTrue(result)
        self.registry.client.delete_model_version.assert_called_once_with(
            name="test_model",
            version=1
        )
    
    def test_search_models(self):
        """Test model search functionality."""
        # Create mock models
        mock_model1 = Mock()
        mock_model1.name = "model_1"
        mock_model1.creation_timestamp = 1234567890000
        mock_model1.last_updated_timestamp = 1234567891000
        mock_model1.description = "First model"
        
        mock_version = Mock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.description = "Production version"
        mock_model1.latest_versions = [mock_version]
        
        mock_model2 = Mock()
        mock_model2.name = "model_2"
        mock_model2.creation_timestamp = 1234567900000
        mock_model2.last_updated_timestamp = 1234567901000
        mock_model2.description = "Second model"
        mock_model2.latest_versions = []
        
        self.registry.client.search_registered_models.return_value = [mock_model1, mock_model2]
        
        # Search models
        results = self.registry.search_models(filter_string="name LIKE 'model%'", max_results=10)
        
        # Assertions
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['name'], "model_1")
        self.assertEqual(results[0]['description'], "First model")
        self.assertEqual(len(results[0]['latest_versions']), 1)
        self.assertEqual(results[0]['latest_versions'][0]['stage'], "Production")
        self.assertEqual(results[1]['name'], "model_2")
        self.assertEqual(len(results[1]['latest_versions']), 0)


class TestModelStageEnum(unittest.TestCase):
    """Test cases for ModelStage enum."""
    
    def test_model_stages(self):
        """Test all model stage values."""
        self.assertEqual(ModelStage.NONE.value, "None")
        self.assertEqual(ModelStage.STAGING.value, "Staging")
        self.assertEqual(ModelStage.PRODUCTION.value, "Production")
        self.assertEqual(ModelStage.ARCHIVED.value, "Archived")
        self.assertEqual(ModelStage.DEVELOPMENT.value, "Development")


if __name__ == "__main__":
    unittest.main()
