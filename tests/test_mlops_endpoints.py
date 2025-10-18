"""
Tests for MLOps API Endpoints
==============================
Tests for model registry, retraining, export, and A/B testing endpoints.
"""

import pytest
import json
import importlib
import types
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from starlette.responses import Response
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Minimal configuration stub to satisfy storage initialisation during import


@dataclass
class _StubStorageConfig:
    backend: str = "local"
    local_base_path: str = "/tmp"


class _StubAutoMLConfig:
    def __init__(self):
        self.storage = _StubStorageConfig()


# Mock dependencies before importing
_stub_config_module = types.ModuleType("automl_platform.config")
_stub_config_module.AutoMLConfig = _StubAutoMLConfig


def _stub_load_config(*args, **kwargs):
    return _StubAutoMLConfig()


_stub_config_module.load_config = _stub_load_config

with patch.dict('sys.modules', {
    'automl_platform.mlflow_registry': MagicMock(),
    'automl_platform.retraining_service': MagicMock(),
    'automl_platform.export_service': MagicMock(),
    'automl_platform.orchestrator': MagicMock(),
    'automl_platform.config': _stub_config_module,
    'automl_platform.storage': MagicMock(),
    'automl_platform.monitoring': MagicMock(),
}):
    from automl_platform.api.mlops_endpoints import router, ModelStage
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(router)


class TestModelRegistryEndpoints:
    """Tests for model registry endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_register_model_success(self, client):
        """Test successful model registration"""
        with patch('automl_platform.api.mlops_endpoints.registry') as mock_registry:
            mock_version = Mock()
            mock_version.model_name = "test_model"
            mock_version.version = "1"
            mock_version.stage = Mock(value="None")
            mock_version.run_id = "run_123"
            mock_registry.register_model.return_value = mock_version
            
            request_data = {
                "model_name": "test_model",
                "description": "Test model",
                "tags": {"env": "test"},
                "metrics": {"accuracy": 0.95},
                "params": {"n_estimators": 100}
            }
            
            response = client.post("/api/v1/mlops/models/register", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["model_name"] == "test_model"
            assert data["version"] == "1"
    
    def test_register_model_error(self, client):
        """Test model registration error handling"""
        with patch('automl_platform.api.mlops_endpoints.registry') as mock_registry:
            mock_registry.register_model.side_effect = Exception("Registration failed")
            
            request_data = {"model_name": "test_model"}
            
            response = client.post("/api/v1/mlops/models/register", json=request_data)
            
            assert response.status_code == 500
            assert "Registration failed" in response.json()["detail"]
    
    def test_promote_model_success(self, client):
        """Test successful model promotion"""
        with patch('automl_platform.api.mlops_endpoints.registry') as mock_registry:
            mock_registry.promote_model.return_value = True
            
            request_data = {
                "model_name": "test_model",
                "version": 1,
                "target_stage": "Production"
            }
            
            response = client.post("/api/v1/mlops/models/promote", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["new_stage"] == "Production"
    
    def test_promote_model_invalid_stage(self, client):
        """Test model promotion with invalid stage"""
        request_data = {
            "model_name": "test_model",
            "version": 1,
            "target_stage": "InvalidStage"
        }
        
        response = client.post("/api/v1/mlops/models/promote", json=request_data)
        
        assert response.status_code == 500
        assert "Invalid stage" in response.json()["detail"]
    
    def test_get_model_versions(self, client):
        """Test getting model version history"""
        with patch('automl_platform.api.mlops_endpoints.registry') as mock_registry:
            mock_history = [
                {"version": "2", "stage": "Production", "metrics": {"accuracy": 0.95}},
                {"version": "1", "stage": "Archived", "metrics": {"accuracy": 0.92}}
            ]
            mock_registry.get_model_history.return_value = mock_history
            
            response = client.get("/api/v1/mlops/models/test_model/versions?limit=10")
            
            assert response.status_code == 200
            data = response.json()
            assert data["model_name"] == "test_model"
            assert len(data["versions"]) == 2
            assert data["total"] == 2
    
    def test_compare_model_versions(self, client):
        """Test comparing model versions"""
        with patch('automl_platform.api.mlops_endpoints.registry') as mock_registry:
            mock_comparison = {
                "version1": {"metrics": {"accuracy": 0.92}},
                "version2": {"metrics": {"accuracy": 0.95}},
                "metrics_diff": {"accuracy": 0.03}
            }
            mock_registry.compare_models.return_value = mock_comparison
            
            response = client.get("/api/v1/mlops/models/test_model/compare?version1=1&version2=2")
            
            assert response.status_code == 200
            data = response.json()
            assert "version1" in data
            assert "version2" in data
            assert "metrics_diff" in data
    
    def test_rollback_model(self, client):
        """Test model rollback"""
        with patch('automl_platform.api.mlops_endpoints.registry') as mock_registry:
            mock_registry.rollback_model.return_value = True
            
            response = client.post("/api/v1/mlops/models/test_model/rollback?target_version=1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["rolled_back_to"] == 1


class TestABTestingEndpoints:
    """Tests for A/B testing endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_create_ab_test(self, client):
        """Test creating A/B test"""
        with patch('automl_platform.api.mlops_endpoints.ab_testing') as mock_ab:
            mock_ab.create_ab_test.return_value = "test_123"
            
            request_data = {
                "model_name": "test_model",
                "champion_version": 1,
                "challenger_version": 2,
                "traffic_split": 0.2,
                "min_samples": 100
            }
            
            response = client.post("/api/v1/mlops/ab-tests/create", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["test_id"] == "test_123"
            assert data["traffic_split"] == 0.2
    
    def test_get_ab_test_results(self, client):
        """Test getting A/B test results"""
        with patch('automl_platform.api.mlops_endpoints.ab_testing') as mock_ab:
            mock_results = {
                "test_id": "test_123",
                "status": "active",
                "champion_samples": 500,
                "challenger_samples": 100,
                "p_value": 0.03
            }
            mock_ab.get_test_results.return_value = mock_results
            
            response = client.get("/api/v1/mlops/ab-tests/test_123/results")
            
            assert response.status_code == 200
            data = response.json()
            assert data["test_id"] == "test_123"
            assert data["p_value"] == 0.03
    
    def test_get_ab_test_results_not_found(self, client):
        """Test getting results for non-existent test"""
        with patch('automl_platform.api.mlops_endpoints.ab_testing') as mock_ab:
            mock_ab.get_test_results.return_value = None
            
            response = client.get("/api/v1/mlops/ab-tests/nonexistent/results")
            
            assert response.status_code == 404
            assert "Test not found" in response.json()["detail"]
    
    def test_conclude_ab_test(self, client):
        """Test concluding A/B test"""
        with patch('automl_platform.api.mlops_endpoints.ab_testing') as mock_ab:
            mock_ab.conclude_test.return_value = {
                "test_id": "test_123",
                "winner": "challenger",
                "promoted": True
            }
            
            response = client.post("/api/v1/mlops/ab-tests/test_123/conclude?promote_winner=true")
            
            assert response.status_code == 200
            data = response.json()
            assert data["winner"] == "challenger"
            assert data["promoted"] is True
    
    def test_get_active_ab_tests(self, client):
        """Test getting active A/B tests"""
        with patch('automl_platform.api.mlops_endpoints.ab_testing') as mock_ab:
            mock_tests = [
                {"test_id": "test_1", "status": "active"},
                {"test_id": "test_2", "status": "active"}
            ]
            mock_ab.get_active_tests.return_value = mock_tests
            
            response = client.get("/api/v1/mlops/ab-tests/active")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["active_tests"]) == 2
            assert data["total"] == 2


class TestModelExportEndpoints:
    """Tests for model export endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_export_model_onnx(self, client):
        """Test ONNX model export"""
        with patch('automl_platform.api.mlops_endpoints.ModelExporter') as mock_exporter_class:
            mock_exporter = Mock()
            mock_exporter.export_to_onnx.return_value = {
                "success": True,
                "path": "/models/test_model.onnx",
                "size_mb": 10.5
            }
            mock_exporter_class.return_value = mock_exporter
            
            request_data = {
                "model_name": "test_model",
                "version": 1,
                "format": "onnx",
                "quantize": True,
                "optimize_for_edge": False
            }
            
            response = client.post("/api/v1/mlops/models/export", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["size_mb"] == 10.5
    
    def test_export_model_pmml(self, client):
        """Test PMML model export"""
        with patch('automl_platform.api.mlops_endpoints.ModelExporter') as mock_exporter_class:
            mock_exporter = Mock()
            mock_exporter.export_to_pmml.return_value = {
                "success": True,
                "path": "/models/test_model.pmml"
            }
            mock_exporter_class.return_value = mock_exporter
            
            request_data = {
                "model_name": "test_model",
                "version": 1,
                "format": "pmml"
            }
            
            response = client.post("/api/v1/mlops/models/export", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_export_model_edge(self, client):
        """Test edge deployment export"""
        with patch('automl_platform.api.mlops_endpoints.ModelExporter') as mock_exporter_class:
            mock_exporter = Mock()
            mock_exporter.export_for_edge.return_value = {
                "success": True,
                "package_dir": "/models/edge/test_model",
                "files": ["model.onnx", "inference.py", "requirements.txt"]
            }
            mock_exporter_class.return_value = mock_exporter
            
            request_data = {
                "model_name": "test_model",
                "version": 1,
                "format": "edge",
                "optimize_for_edge": True
            }
            
            response = client.post("/api/v1/mlops/models/export", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "files" in data
    
    def test_export_model_unsupported_format(self, client):
        """Test export with unsupported format"""
        request_data = {
            "model_name": "test_model",
            "version": 1,
            "format": "unsupported"
        }
        
        response = client.post("/api/v1/mlops/models/export", json=request_data)
        
        assert response.status_code == 500
        assert "Unsupported format" in response.json()["detail"]
    
    def test_download_exported_model(self, client):
        """Test downloading exported model"""
        with patch('automl_platform.api.mlops_endpoints.Path') as mock_path:
            mock_export_dir = MagicMock()
            mock_file = MagicMock()
            mock_file.exists.return_value = True
            mock_file.name = "test_model.onnx"
            mock_export_dir.__truediv__.return_value = mock_file
            mock_path.return_value = mock_export_dir

            with patch('automl_platform.api.mlops_endpoints.FileResponse') as mock_response:
                mock_response.return_value = Response(media_type="application/octet-stream")

                response = client.get("/api/v1/mlops/models/export/test_model/1/download?format=onnx")

                # FileResponse would be called
                assert mock_response.called
    
    def test_download_exported_model_not_found(self, client):
        """Test downloading non-existent exported model"""
        with patch('automl_platform.api.mlops_endpoints.Path') as mock_path:
            mock_export_dir = MagicMock()
            mock_file = MagicMock()
            mock_file.exists.return_value = False
            mock_export_dir.__truediv__.return_value = mock_file
            mock_path.return_value = mock_export_dir

            response = client.get("/api/v1/mlops/models/export/test_model/1/download")

            assert response.status_code == 404
            assert "Exported model not found" in response.json()["detail"]


class TestRetrainingEndpoints:
    """Tests for automated retraining endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_check_retraining_needed(self, client):
        """Test checking if retraining is needed"""
        with patch('automl_platform.api.mlops_endpoints.retraining_service') as mock_service:
            mock_service.should_retrain.return_value = (
                True,
                "High drift detected",
                {"drift_score": 0.7, "accuracy": 0.82}
            )
            
            request_data = {
                "model_name": "test_model",
                "check_drift": True,
                "check_performance": True,
                "check_data_volume": True
            }
            
            response = client.post("/api/v1/mlops/retraining/check", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["needs_retraining"] is True
            assert data["reason"] == "High drift detected"
            assert data["metrics"]["drift_score"] == 0.7
    
    def test_trigger_retraining(self, client):
        """Test triggering manual retraining"""
        with patch('automl_platform.api.mlops_endpoints.BackgroundTasks') as mock_bg:
            mock_bg_instance = Mock()
            
            response = client.post("/api/v1/mlops/retraining/trigger/test_model")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["status"] == "queued"
    
    def test_get_retraining_history(self, client):
        """Test getting retraining history"""
        with patch('automl_platform.api.mlops_endpoints.retraining_service') as mock_service:
            mock_history = [
                {"model_name": "model1", "end_time": "2024-01-01T10:00:00", "status": "success"},
                {"model_name": "model2", "end_time": "2024-01-02T10:00:00", "status": "failed"}
            ]
            mock_service.get_retraining_history.return_value = mock_history
            
            response = client.get("/api/v1/mlops/retraining/history?limit=10")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["history"]) == 2
            assert data["total"] == 2
    
    def test_get_retraining_stats(self, client):
        """Test getting retraining statistics"""
        with patch('automl_platform.api.mlops_endpoints.retraining_service') as mock_service:
            mock_stats = {
                "total_retrainings": 10,
                "successful": 8,
                "failed": 2,
                "success_rate": 0.8,
                "average_duration_seconds": 1800
            }
            mock_service.get_retraining_stats.return_value = mock_stats
            
            response = client.get("/api/v1/mlops/retraining/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_retrainings"] == 10
            assert data["success_rate"] == 0.8
    
    def test_create_retraining_schedule(self, client):
        """Test creating automated retraining schedule"""
        with patch('automl_platform.api.mlops_endpoints.retraining_service') as mock_service:
            mock_schedule = Mock()
            mock_service.create_retraining_schedule.return_value = mock_schedule
            
            response = client.post("/api/v1/mlops/retraining/schedule")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "framework" in data
    
    def test_create_retraining_schedule_no_framework(self, client):
        """Test schedule creation when no framework available"""
        with patch('automl_platform.api.mlops_endpoints.retraining_service') as mock_service:
            mock_service.create_retraining_schedule.return_value = None
            
            response = client.post("/api/v1/mlops/retraining/schedule")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "No scheduling framework available" in data["message"]


class TestPredictionWithMLOps:
    """Tests for prediction endpoints with MLOps features"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_predict_with_ab_test(self, client):
        """Test prediction with A/B testing"""
        with patch('automl_platform.api.mlops_endpoints.ab_testing') as mock_ab:
            mock_ab.route_prediction.return_value = ("challenger", 2)
            
            request_data = {
                "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "use_ab_test": True,
                "test_id": "test_123"
            }
            
            response = client.post("/api/v1/mlops/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert data["model_info"]["model_type"] == "challenger"
            assert data["model_info"]["version"] == 2
            mock_ab.record_result.assert_called()
    
    def test_predict_with_specific_version(self, client):
        """Test prediction with specific model version"""
        request_data = {
            "features": [[1.0, 2.0, 3.0]],
            "model_name": "test_model",
            "version": 3
        }
        
        response = client.post("/api/v1/mlops/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["model_info"]["model_name"] == "test_model"
        assert data["model_info"]["version"] == 3
    
    def test_predict_default_production(self, client):
        """Test prediction with default production model"""
        request_data = {
            "features": [[1.0, 2.0, 3.0]]
        }
        
        response = client.post("/api/v1/mlops/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["model_info"]["model_type"] == "production"
        assert data["model_info"]["version"] == "latest"


class TestHealthCheck:
    """Tests for MLOps health check endpoint"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_mlops_health_check(self, client):
        """Test MLOps services health check"""
        with patch('automl_platform.api.mlops_endpoints.registry') as mock_registry:
            with patch('automl_platform.api.mlops_endpoints.ab_testing') as mock_ab:
                mock_registry.client = Mock()
                mock_ab.active_tests = []

                response = client.get("/api/v1/mlops/health")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert "services" in data
                assert data["services"]["mlflow"] is True
                assert data["services"]["export"] is True
                assert "timestamp" in data


def test_storage_service_initialization_uses_storage_config():
    """Ensure StorageService receives the storage configuration details."""

    module_name = "automl_platform.api.mlops_endpoints"
    dependencies = [
        module_name,
        "automl_platform.config",
        "automl_platform.storage",
        "automl_platform.mlflow_registry",
        "automl_platform.retraining_service",
        "automl_platform.export_service",
        "automl_platform.orchestrator",
        "automl_platform.monitoring",
    ]

    saved_modules = {name: sys.modules.get(name) for name in dependencies}
    for name in dependencies:
        if name in sys.modules:
            del sys.modules[name]

    storage_call: Dict[str, Any] = {}

    @dataclass
    class DummyStorageConfig:
        backend: str = "s3"
        endpoint: str = "s3.amazonaws.com"
        access_key: str = "AKIAEXAMPLE"
        secret_key: str = "SECRETKEY"
        secure: bool = True
        region: str = "eu-west-3"
        local_base_path: str = "/mnt/mlops"
        project_id: str = "demo-project"
        credentials_path: str = "/secure/creds.json"
        models_bucket: str = "models"
        datasets_bucket: str = "datasets"
        artifacts_bucket: str = "artifacts"
        knowledge_bucket: str = "knowledge"
        enable_feature_store: bool = True
        feature_cache_size: int = 128
        auto_versioning: bool = True
        max_versions_per_model: int = 5
        cleanup_old_versions: bool = False
        tenant_id: str = "tenant-a"
        isolate_buckets_per_tenant: bool = True

    class DummyAutoMLConfig:
        def __init__(self):
            self.storage = DummyStorageConfig()
            # Sensitive fields may be attached dynamically in production.
            setattr(self.storage, "encryption_key", "very-secret")

    class DummyStorageService:
        def __init__(self, *args, **kwargs):
            storage_call["args"] = args
            if "backend" in kwargs:
                storage_call["backend"] = kwargs["backend"]
            elif args:
                storage_call["backend"] = args[0]
            else:
                storage_call["backend"] = None
            sanitized_kwargs = dict(kwargs)
            sanitized_kwargs.pop("backend", None)
            storage_call["kwargs"] = sanitized_kwargs

    config_module = types.ModuleType("automl_platform.config")
    config_module.AutoMLConfig = DummyAutoMLConfig

    storage_module = types.ModuleType("automl_platform.storage")
    storage_module.StorageService = DummyStorageService

    mlflow_module = types.ModuleType("automl_platform.mlflow_registry")

    class DummyMLflowRegistry:
        def __init__(self, config):
            self.config = config

    class DummyABTestingService:
        def __init__(self, registry):
            self.registry = registry

    mlflow_module.MLflowRegistry = DummyMLflowRegistry
    mlflow_module.ABTestingService = DummyABTestingService
    mlflow_module.ModelStage = Enum("ModelStage", "NONE")

    retraining_module = types.ModuleType("automl_platform.retraining_service")

    class DummyRetrainingService:
        def __init__(self, config, registry, monitor, storage):
            self.args = (config, registry, monitor, storage)

    retraining_module.RetrainingService = DummyRetrainingService
    retraining_module.RetrainingConfig = object

    export_module = types.ModuleType("automl_platform.export_service")

    class DummyModelExporter:
        def __init__(self):
            pass

    export_module.ModelExporter = DummyModelExporter
    export_module.ExportConfig = object

    orchestrator_module = types.ModuleType("automl_platform.orchestrator")
    orchestrator_module.AutoMLOrchestrator = object

    monitoring_module = types.ModuleType("automl_platform.monitoring")

    class DummyModelMonitor:
        def __init__(self, config):
            self.config = config

    monitoring_module.ModelMonitor = DummyModelMonitor

    try:
        with patch.dict(
            sys.modules,
            {
                "automl_platform.config": config_module,
                "automl_platform.storage": storage_module,
                "automl_platform.mlflow_registry": mlflow_module,
                "automl_platform.retraining_service": retraining_module,
                "automl_platform.export_service": export_module,
                "automl_platform.orchestrator": orchestrator_module,
                "automl_platform.monitoring": monitoring_module,
            },
        ):
            importlib.import_module(module_name)

            assert storage_call["args"] == ()
            assert storage_call["backend"] == "s3"
            kwargs = storage_call["kwargs"]
            assert kwargs["endpoint"] == "s3.amazonaws.com"
            assert kwargs["access_key"] == "AKIAEXAMPLE"
            assert kwargs["secret_key"] == "SECRETKEY"
            assert kwargs["secure"] is True
            assert kwargs["region"] == "eu-west-3"
            assert kwargs["encryption_key"] == b"very-secret"
            assert "local_base_path" not in kwargs
            assert "project_id" not in kwargs
            assert "credentials_path" not in kwargs
            assert "models_bucket" not in kwargs
            assert "datasets_bucket" not in kwargs
            assert "artifacts_bucket" not in kwargs
            assert "knowledge_bucket" not in kwargs
    finally:
        for name in dependencies:
            if name in sys.modules:
                del sys.modules[name]

        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
