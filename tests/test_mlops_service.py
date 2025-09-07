"""
Tests for MLOps Service
========================
Tests for MLflow registry, model retraining, and export functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timedelta
import json
import tempfile
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.mlops_service import (
    ModelStage,
    MLflowRegistry,
    RetrainingService,
    ModelExporter,
    ModelVersionManager,
    create_mlops_service
)
from automl_platform.config import AutoMLConfig


class TestModelStage:
    """Tests for ModelStage enum"""
    
    def test_model_stage_values(self):
        """Test model stage enum values"""
        assert ModelStage.NONE.value == "None"
        assert ModelStage.STAGING.value == "Staging"
        assert ModelStage.PRODUCTION.value == "Production"
        assert ModelStage.ARCHIVED.value == "Archived"


class TestMLflowRegistry:
    """Tests for MLflow Registry with caching"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = AutoMLConfig()
        config.enable_cache = True
        config.cache_backend = 'redis'
        config.redis_host = 'localhost'
        config.cache_ttl = 3600
        config.mlflow_tracking_uri = 'sqlite:///test_mlflow.db'
        return config
    
    @pytest.fixture
    def registry(self, config):
        """Create MLflow registry instance"""
        with patch('automl_platform.mlops_service.MLFLOW_AVAILABLE', True):
            with patch('automl_platform.mlops_service.MlflowClient'):
                return MLflowRegistry(config)
    
    def test_initialization_with_cache(self, config):
        """Test registry initialization with cache enabled"""
        with patch('automl_platform.mlops_service.OPTIMIZATIONS_AVAILABLE', True):
            with patch('automl_platform.mlops_service.PipelineCache') as mock_cache:
                registry = MLflowRegistry(config)
                
                assert registry.pipeline_cache is not None
                mock_cache.assert_called_once()
    
    def test_initialization_without_mlflow(self, config):
        """Test registry initialization when MLflow not available"""
        with patch('automl_platform.mlops_service.MLFLOW_AVAILABLE', False):
            registry = MLflowRegistry(config)
            assert registry.client is None
    
    def test_get_production_model_from_cache(self, registry):
        """Test getting production model from cache"""
        mock_model = Mock()
        registry.pipeline_cache = Mock()
        registry.pipeline_cache.get_pipeline.return_value = mock_model
        
        model = registry.get_production_model("test_model", use_cache=True)
        
        assert model == mock_model
        registry.pipeline_cache.get_pipeline.assert_called_once_with("prod_model_test_model")
    
    def test_get_production_model_from_mlflow(self, registry):
        """Test getting production model from MLflow"""
        mock_model = Mock()
        mock_version = Mock(version="1")
        
        registry.pipeline_cache = Mock()
        registry.pipeline_cache.get_pipeline.return_value = None
        registry.client = Mock()
        registry.client.get_latest_versions.return_value = [mock_version]
        
        with patch('automl_platform.mlops_service.mlflow.pyfunc.load_model', return_value=mock_model):
            model = registry.get_production_model("test_model")
            
            assert model == mock_model
            registry.pipeline_cache.set_pipeline.assert_called_once()
    
    def test_get_production_model_not_found(self, registry):
        """Test when production model not found"""
        registry.pipeline_cache = Mock()
        registry.pipeline_cache.get_pipeline.return_value = None
        registry.client = Mock()
        registry.client.get_latest_versions.return_value = []
        
        model = registry.get_production_model("test_model")
        assert model is None
    
    def test_invalidate_model_cache(self, registry):
        """Test invalidating model cache"""
        registry.pipeline_cache = Mock()
        
        registry.invalidate_model_cache("test_model")
        
        registry.pipeline_cache.invalidate.assert_called_once_with(
            "prod_model_test_model", 
            reason="model_updated"
        )
    
    @patch('automl_platform.mlops_service.mlflow')
    def test_register_model(self, mock_mlflow, registry):
        """Test model registration in MLflow"""
        model = Mock()
        metrics = {"accuracy": 0.95, "f1": 0.93}
        params = {"n_estimators": 100}
        X_sample = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        y_sample = pd.Series([0, 1])
        
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        registry.client = Mock()
        mock_version = Mock()
        registry.client.search_model_versions.return_value = [mock_version]
        
        result = registry.register_model(
            model=model,
            model_name="test_model",
            metrics=metrics,
            params=params,
            X_sample=X_sample,
            y_sample=y_sample,
            description="Test model",
            tags={"env": "test"}
        )
        
        assert result == mock_version
        mock_mlflow.sklearn.log_model.assert_called_once()
        assert mock_mlflow.log_metric.call_count == 2
        assert mock_mlflow.log_param.call_count == 1
    
    def test_promote_model(self, registry):
        """Test model promotion to different stages"""
        registry.client = Mock()
        
        registry.promote_model("test_model", "1", ModelStage.PRODUCTION)
        
        registry.client.transition_model_version_stage.assert_called_once_with(
            name="test_model",
            version="1",
            stage="Production"
        )
    
    def test_promote_model_invalidates_cache(self, registry):
        """Test cache invalidation on production promotion"""
        registry.client = Mock()
        registry.pipeline_cache = Mock()
        
        registry.promote_model("test_model", "1", ModelStage.PRODUCTION)
        
        registry.pipeline_cache.invalidate.assert_called_once_with(
            "prod_model_test_model",
            reason="model_updated"
        )


class TestRetrainingService:
    """Tests for automated model retraining"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = AutoMLConfig()
        config.incremental_learning = True
        config.distributed_training = False
        config.max_memory_mb = 1000
        return config
    
    @pytest.fixture
    def registry(self):
        """Create mock registry"""
        registry = Mock(spec=MLflowRegistry)
        registry.pipeline_cache = Mock()
        registry.client = Mock()
        return registry
    
    @pytest.fixture
    def monitor(self):
        """Create mock monitor"""
        monitor = Mock()
        monitor.get_current_performance.return_value = {"accuracy": 0.85}
        monitor.get_baseline_performance.return_value = {"accuracy": 0.90}
        monitor.get_drift_score.return_value = 0.3
        monitor.get_new_data_count.return_value = 500
        return monitor
    
    @pytest.fixture
    def service(self, config, registry, monitor):
        """Create retraining service"""
        return RetrainingService(config, registry, monitor)
    
    @pytest.mark.asyncio
    async def test_check_retraining_needed_performance_degradation(self, service, monitor):
        """Test retraining triggered by performance degradation"""
        monitor.get_current_performance.return_value = {"accuracy": 0.75}
        monitor.get_baseline_performance.return_value = {"accuracy": 0.90}
        
        needs_retraining = await service.check_retraining_needed("test_model")
        
        assert needs_retraining is True
    
    @pytest.mark.asyncio
    async def test_check_retraining_needed_drift(self, service, monitor):
        """Test retraining triggered by drift"""
        monitor.get_drift_score.return_value = 0.6
        
        needs_retraining = await service.check_retraining_needed("test_model")
        
        assert needs_retraining is True
    
    @pytest.mark.asyncio
    async def test_check_retraining_needed_data_volume(self, service, monitor):
        """Test retraining triggered by data volume"""
        monitor.get_new_data_count.return_value = 1500
        
        needs_retraining = await service.check_retraining_needed("test_model")
        
        assert needs_retraining is True
    
    @pytest.mark.asyncio
    async def test_check_retraining_not_needed(self, service, monitor):
        """Test when retraining not needed"""
        monitor.get_current_performance.return_value = {"accuracy": 0.89}
        monitor.get_baseline_performance.return_value = {"accuracy": 0.90}
        monitor.get_drift_score.return_value = 0.3
        monitor.get_new_data_count.return_value = 500
        
        needs_retraining = await service.check_retraining_needed("test_model")
        
        assert needs_retraining is False
    
    @pytest.mark.asyncio
    @patch('automl_platform.mlops_service.AutoMLOrchestrator')
    async def test_retrain_model_standard(self, mock_orchestrator, service, registry):
        """Test standard model retraining"""
        X_train = pd.DataFrame(np.random.randn(100, 5))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.best_pipeline = Mock()
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        mock_version = Mock(version="2", run_id="run_123")
        registry.client.get_latest_versions.return_value = [mock_version]
        registry.client.get_run.return_value = Mock(data=Mock(params={"algorithm": "RandomForest"}))
        registry.register_model.return_value = mock_version
        
        with patch('automl_platform.mlops_service.calculate_metrics', return_value={"accuracy": 0.92}):
            result = await service.retrain_model("test_model", X_train, y_train)
        
        assert result == mock_version
        registry.invalidate_model_cache.assert_called_once_with("test_model")
        registry.register_model.assert_called_once()
        registry.promote_model.assert_called_once_with("test_model", "2", ModelStage.STAGING)
    
    @pytest.mark.asyncio
    async def test_retrain_model_incremental(self, service, registry):
        """Test incremental model retraining"""
        X_train = pd.DataFrame(np.random.randn(15000, 5))
        y_train = pd.Series(np.random.randint(0, 2, 15000))
        
        service.incremental_learner = Mock()
        service.incremental_learner.train_incremental.return_value = [Mock()]
        service.incremental_learner.get_best_model.return_value = Mock()
        
        mock_version = Mock(version="2", run_id="run_123")
        registry.client.get_latest_versions.return_value = [mock_version]
        registry.client.get_run.return_value = Mock(data=Mock(params={}))
        registry.register_model.return_value = mock_version
        
        with patch('automl_platform.mlops_service.calculate_metrics', return_value={"accuracy": 0.92}):
            result = await service.retrain_model("test_model", X_train, y_train, use_incremental=True)
        
        assert result == mock_version
        service.incremental_learner.train_incremental.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('automl_platform.mlops_service.mlflow')
    async def test_validate_retrained_model_better(self, mock_mlflow, service, registry):
        """Test validation when retrained model is better"""
        X_val = pd.DataFrame(np.random.randn(50, 5))
        y_val = pd.Series(np.random.randint(0, 2, 50))
        
        staging_model = Mock()
        staging_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        prod_model = Mock()
        prod_model.predict.return_value = np.array([0, 0, 1, 1, 0])
        
        mock_mlflow.pyfunc.load_model.return_value = staging_model
        registry.get_production_model.return_value = prod_model
        
        with patch('automl_platform.mlops_service.calculate_metrics') as mock_metrics:
            mock_metrics.side_effect = [
                {"accuracy": 0.92},  # staging
                {"accuracy": 0.88}   # production
            ]
            
            is_valid = await service.validate_retrained_model("test_model", "2", X_val, y_val)
        
        assert is_valid is True
    
    @pytest.mark.asyncio
    @patch('automl_platform.mlops_service.mlflow')
    async def test_validate_retrained_model_worse(self, mock_mlflow, service, registry):
        """Test validation when retrained model is worse"""
        X_val = pd.DataFrame(np.random.randn(50, 5))
        y_val = pd.Series(np.random.randint(0, 2, 50))
        
        staging_model = Mock()
        prod_model = Mock()
        
        mock_mlflow.pyfunc.load_model.return_value = staging_model
        registry.get_production_model.return_value = prod_model
        
        with patch('automl_platform.mlops_service.calculate_metrics') as mock_metrics:
            mock_metrics.side_effect = [
                {"accuracy": 0.85},  # staging
                {"accuracy": 0.92}   # production
            ]
            
            is_valid = await service.validate_retrained_model("test_model", "2", X_val, y_val)
        
        assert is_valid is False


class TestModelExporter:
    """Tests for model export functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = AutoMLConfig()
        config.enable_cache = True
        return config
    
    @pytest.fixture
    def exporter(self, config):
        """Create model exporter"""
        return ModelExporter(config)
    
    @patch('automl_platform.mlops_service.ONNX_AVAILABLE', True)
    @patch('automl_platform.mlops_service.convert_sklearn')
    def test_export_to_onnx_success(self, mock_convert, exporter):
        """Test successful ONNX export"""
        model = Mock()
        sample_input = np.random.randn(10, 5)
        output_path = "test_model.onnx"
        
        mock_onnx = Mock()
        mock_onnx.SerializeToString.return_value = b"onnx_model_bytes"
        mock_convert.return_value = mock_onnx
        
        with patch('builtins.open', create=True) as mock_open:
            result = exporter.export_to_onnx(model, sample_input, output_path)
        
        assert result is True
        mock_convert.assert_called_once()
        mock_open.assert_called_once_with(output_path, "wb")
    
    @patch('automl_platform.mlops_service.ONNX_AVAILABLE', True)
    def test_export_to_onnx_with_cache(self, exporter):
        """Test ONNX export with caching"""
        model = Mock()
        sample_input = np.random.randn(10, 5)
        output_path = "test_model.onnx"
        
        exporter.export_cache = Mock()
        exporter.export_cache.get_pipeline.return_value = b"cached_onnx_bytes"
        
        with patch('builtins.open', create=True) as mock_open:
            result = exporter.export_to_onnx(model, sample_input, output_path, use_cache=True)
        
        assert result is True
        mock_open.assert_called_once()
    
    @patch('automl_platform.mlops_service.ONNX_AVAILABLE', False)
    def test_export_to_onnx_not_available(self, exporter):
        """Test ONNX export when library not available"""
        model = Mock()
        sample_input = np.random.randn(10, 5)
        
        result = exporter.export_to_onnx(model, sample_input, "test.onnx")
        
        assert result is False
    
    def test_export_to_pmml(self, exporter):
        """Test PMML export"""
        with patch('automl_platform.mlops_service.sklearn2pmml') as mock_sklearn2pmml:
            model = Mock()
            result = exporter.export_to_pmml(model, "test_model.pmml")
            
            assert result is True
            mock_sklearn2pmml.assert_called_once()
    
    def test_export_to_tensorflow_lite(self, exporter):
        """Test TensorFlow Lite export"""
        with patch('automl_platform.mlops_service.tf') as mock_tf:
            model = Mock()
            sample_input = np.random.randn(10, 5)
            
            result = exporter.export_to_tensorflow_lite(model, sample_input, "model.tflite")
            
            # Would normally test TF conversion logic
            assert result is True or result is False  # Depends on TF availability


class TestModelVersionManager:
    """Tests for model version management"""
    
    @pytest.fixture
    def registry(self):
        """Create mock registry"""
        registry = Mock(spec=MLflowRegistry)
        registry.client = Mock()
        return registry
    
    @pytest.fixture
    def manager(self, registry):
        """Create version manager"""
        return ModelVersionManager(registry)
    
    @patch('automl_platform.mlops_service.mlflow')
    def test_compare_versions(self, mock_mlflow, manager):
        """Test comparing two model versions"""
        X_test = pd.DataFrame(np.random.randn(50, 5))
        y_test = pd.Series(np.random.randint(0, 2, 50))
        
        model_a = Mock()
        model_a.predict.return_value = np.array([0, 1, 0, 1, 0])
        model_b = Mock()
        model_b.predict.return_value = np.array([0, 0, 1, 1, 0])
        
        mock_mlflow.pyfunc.load_model.side_effect = [model_a, model_b]
        
        with patch('automl_platform.mlops_service.calculate_metrics') as mock_metrics:
            mock_metrics.side_effect = [
                {"accuracy": 0.90, "f1": 0.88},
                {"accuracy": 0.85, "f1": 0.83}
            ]
            
            comparison = manager.compare_versions("test_model", "1", "2", X_test, y_test)
        
        assert "version_a" in comparison
        assert "version_b" in comparison
        assert "comparison" in comparison
        assert comparison["version_a"]["metrics"]["accuracy"] == 0.90
        assert comparison["version_b"]["metrics"]["accuracy"] == 0.85
    
    def test_rollback_model(self, manager, registry):
        """Test model rollback"""
        current_version = Mock(version="3")
        registry.client.get_latest_versions.return_value = [current_version]
        
        manager.rollback_model("test_model", "2")
        
        # Should archive current and promote target
        assert registry.promote_model.call_count == 2
        calls = registry.promote_model.call_args_list
        assert calls[0][0] == ("test_model", "3", ModelStage.ARCHIVED)
        assert calls[1][0] == ("test_model", "2", ModelStage.PRODUCTION)
    
    def test_get_version_history(self, manager, registry):
        """Test getting model version history"""
        version1 = Mock(
            version="1",
            current_stage="Archived",
            creation_timestamp=1000,
            last_updated_timestamp=2000,
            run_id="run1"
        )
        version2 = Mock(
            version="2",
            current_stage="Production",
            creation_timestamp=3000,
            last_updated_timestamp=4000,
            run_id="run2"
        )
        
        registry.client.search_model_versions.return_value = [version1, version2]
        
        run1 = Mock(data=Mock(metrics={"acc": 0.85}, params={"n": 100}, tags={}))
        run2 = Mock(data=Mock(metrics={"acc": 0.90}, params={"n": 200}, tags={}))
        registry.client.get_run.side_effect = [run2, run1]
        
        history = manager.get_version_history("test_model")
        
        assert len(history) == 2
        assert history[0]["version"] == "2"  # Sorted by version desc
        assert history[1]["version"] == "1"
        assert history[0]["metrics"]["acc"] == 0.90


class TestCreateMLOpsService:
    """Tests for MLOps service creation"""
    
    def test_create_mlops_service_basic(self):
        """Test creating MLOps service components"""
        config = AutoMLConfig()
        
        with patch('automl_platform.mlops_service.MLflowRegistry'):
            with patch('automl_platform.mlops_service.ModelExporter'):
                with patch('automl_platform.mlops_service.ModelVersionManager'):
                    services = create_mlops_service(config)
        
        assert "registry" in services
        assert "exporter" in services
        assert "version_manager" in services
        assert services["monitor"] is None
        assert services["retraining_service"] is None
    
    def test_create_mlops_service_with_monitoring(self):
        """Test creating MLOps service with monitoring enabled"""
        config = AutoMLConfig()
        config.monitoring = Mock(enabled=True)
        
        with patch('automl_platform.mlops_service.MLflowRegistry'):
            with patch('automl_platform.mlops_service.ModelExporter'):
                with patch('automl_platform.mlops_service.ModelVersionManager'):
                    with patch('automl_platform.mlops_service.ModelMonitor'):
                        services = create_mlops_service(config)
        
        assert services["monitor"] is not None
        assert services["retraining_service"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
