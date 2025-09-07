"""
Tests for API endpoints
========================
Tests for FastAPI endpoints in api.py
"""

import pytest
import json
import tempfile
import os
import sys
from unittest.mock import Mock, MagicMock, patch
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies before importing the app
with patch.dict('sys.modules', {
    'automl_platform.auth': MagicMock(),
    'automl_platform.billing': MagicMock(),
    'automl_platform.billing_middleware': MagicMock(),
    'automl_platform.scheduler': MagicMock(),
    'automl_platform.data_prep': MagicMock(),
    'automl_platform.inference': MagicMock(),
    'automl_platform.streaming': MagicMock(),
    'automl_platform.export_service': MagicMock(),
    'automl_platform.ab_testing': MagicMock(),
    'automl_platform.auth_endpoints': MagicMock(),
    'automl_platform.connectors': MagicMock(),
    'automl_platform.feature_store': MagicMock(),
    'automl_platform.orchestrator': MagicMock(),
    'automl_platform.config': MagicMock(),
}):
    from automl_platform.api.api import app, TrainRequest, PredictRequest, BatchPredictRequest


class TestHealthEndpoints:
    """Tests for health and status endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert data['message'] == 'AutoML Platform API'
        assert 'version' in data
        assert 'status' in data
        assert 'available_features' in data
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'services' in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        with patch('automl_platform.api.api.PROMETHEUS_AVAILABLE', True):
            with patch('automl_platform.api.api.generate_latest') as mock_generate:
                mock_generate.return_value = b"# HELP test_metric Test metric\n# TYPE test_metric gauge\ntest_metric 1.0\n"
                
                response = client.get("/metrics")
                
                assert response.status_code == 200
                assert b"test_metric" in response.content
    
    def test_metrics_not_available(self, client):
        """Test metrics endpoint when Prometheus is not available"""
        with patch('automl_platform.api.api.PROMETHEUS_AVAILABLE', False):
            response = client.get("/metrics")
            
            assert response.status_code == 503
            assert 'Prometheus client not installed' in response.json()['detail']
    
    def test_metrics_status(self, client):
        """Test metrics status endpoint"""
        response = client.get("/metrics/status")
        
        assert response.status_code == 200
        data = response.json()
        assert 'prometheus_available' in data
        assert 'sources' in data
        assert 'summary' in data


class TestDataManagementEndpoints:
    """Tests for data management endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client with auth mocked"""
        with patch('automl_platform.api.api.AUTH_AVAILABLE', True):
            with patch('automl_platform.api.api.get_current_user') as mock_user:
                mock_user.return_value = Mock(
                    id='user123',
                    username='testuser',
                    tenant_id='tenant123'
                )
                return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user"""
        return Mock(
            id='user123',
            username='testuser',
            tenant_id='tenant123',
            plan_type='pro',
            organization='TestOrg'
        )
    
    def test_upload_dataset_success(self, client, mock_user):
        """Test successful dataset upload"""
        with patch('automl_platform.api.api.DATA_PREP_AVAILABLE', True):
            with patch('automl_platform.api.api.get_current_user', return_value=mock_user):
                with patch('automl_platform.api.api.validate_data') as mock_validate:
                    mock_validate.return_value = {'status': 'valid', 'issues': []}
                    
                    # Create test CSV file
                    csv_content = "col1,col2,col3\n1,2,3\n4,5,6\n"
                    files = {'file': ('test.csv', csv_content, 'text/csv')}
                    
                    response = client.post(
                        "/api/upload",
                        files=files,
                        data={'name': 'Test Dataset', 'description': 'Test description'}
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert 'dataset_id' in data
                    assert data['filename'] == 'test.csv'
                    assert data['rows'] == 2
                    assert data['columns'] == 3
    
    def test_upload_dataset_unsupported_format(self, client, mock_user):
        """Test upload with unsupported file format"""
        with patch('automl_platform.api.api.DATA_PREP_AVAILABLE', True):
            with patch('automl_platform.api.api.get_current_user', return_value=mock_user):
                files = {'file': ('test.txt', 'some text', 'text/plain')}
                
                response = client.post("/api/upload", files=files)
                
                assert response.status_code == 400
                assert 'Unsupported file format' in response.json()['detail']
    
    def test_upload_dataset_no_auth(self, client):
        """Test upload without authentication"""
        with patch('automl_platform.api.api.AUTH_AVAILABLE', True):
            with patch('automl_platform.api.api.get_current_user', return_value=None):
                files = {'file': ('test.csv', 'col1\n1\n', 'text/csv')}
                
                response = client.post("/api/upload", files=files)
                
                assert response.status_code == 401
    
    def test_list_datasets(self, client, mock_user):
        """Test listing datasets"""
        with patch('automl_platform.api.api.DATA_PREP_AVAILABLE', True):
            with patch('automl_platform.api.api.get_current_user', return_value=mock_user):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.glob') as mock_glob:
                        # Mock file paths
                        mock_file1 = Mock()
                        mock_file1.name = 'dataset1.csv'
                        mock_file1.stat.return_value = Mock(
                            st_size=1024 * 1024,  # 1 MB
                            st_ctime=1640995200  # 2022-01-01
                        )
                        
                        mock_file2 = Mock()
                        mock_file2.name = 'dataset2.csv'
                        mock_file2.stat.return_value = Mock(
                            st_size=2 * 1024 * 1024,  # 2 MB
                            st_ctime=1641081600  # 2022-01-02
                        )
                        
                        mock_glob.return_value = [mock_file1, mock_file2]
                        
                        response = client.get("/api/datasets")
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert 'datasets' in data
                        assert 'total' in data
                        assert data['total'] == 2
                        assert len(data['datasets']) == 2


class TestModelTrainingEndpoints:
    """Tests for model training endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user"""
        return Mock(
            id='user123',
            username='testuser',
            tenant_id='tenant123',
            plan_type='pro'
        )
    
    @pytest.fixture
    def train_request(self):
        """Create train request"""
        return {
            'dataset_id': 'dataset_123',
            'task': 'classification',
            'target_column': 'target',
            'time_limit': 600,
            'enable_gpu': False,
            'include_neural': True,
            'max_models': 10
        }
    
    def test_train_model_success(self, client, mock_user, train_request):
        """Test successful model training"""
        with patch('automl_platform.api.api.SCHEDULER_AVAILABLE', True):
            with patch('automl_platform.api.api.AUTH_AVAILABLE', True):
                with patch('automl_platform.api.api.get_current_user', return_value=mock_user):
                    with patch('automl_platform.api.api.scheduler') as mock_scheduler:
                        mock_scheduler.submit_job = Mock(return_value='job_123')
                        
                        response = client.post(
                            "/api/train",
                            json=train_request
                        )
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert data['job_id'] == 'job_123'
                        assert data['status'] == 'submitted'
                        assert 'queue' in data
                        assert 'estimated_time_minutes' in data
    
    def test_train_model_no_scheduler(self, client, mock_user, train_request):
        """Test training when scheduler is not configured"""
        with patch('automl_platform.api.api.SCHEDULER_AVAILABLE', True):
            with patch('automl_platform.api.api.AUTH_AVAILABLE', True):
                with patch('automl_platform.api.api.get_current_user', return_value=mock_user):
                    with patch('automl_platform.api.api.scheduler', None):
                        response = client.post("/api/train", json=train_request)
                        
                        assert response.status_code == 503
                        assert 'Scheduler not configured' in response.json()['detail']
    
    def test_get_job_status(self, client, mock_user):
        """Test getting job status"""
        with patch('automl_platform.api.api.SCHEDULER_AVAILABLE', True):
            with patch('automl_platform.api.api.AUTH_AVAILABLE', True):
                with patch('automl_platform.api.api.get_current_user', return_value=mock_user):
                    with patch('automl_platform.api.api.scheduler') as mock_scheduler:
                        mock_job = Mock(
                            tenant_id='tenant123',
                            status=Mock(value='running'),
                            created_at=Mock(isoformat=lambda: '2024-01-01T10:00:00'),
                            started_at=Mock(isoformat=lambda: '2024-01-01T10:05:00'),
                            completed_at=None,
                            result=None,
                            error_message=None
                        )
                        mock_scheduler.get_job_status = Mock(return_value=mock_job)
                        
                        response = client.get("/api/jobs/job_123")
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert data['job_id'] == 'job_123'
                        assert data['status'] == 'running'
                        assert data['created_at'] == '2024-01-01T10:00:00'
    
    def test_get_job_status_not_found(self, client, mock_user):
        """Test getting status for non-existent job"""
        with patch('automl_platform.api.api.SCHEDULER_AVAILABLE', True):
            with patch('automl_platform.api.api.AUTH_AVAILABLE', True):
                with patch('automl_platform.api.api.get_current_user', return_value=mock_user):
                    with patch('automl_platform.api.api.scheduler') as mock_scheduler:
                        mock_scheduler.get_job_status = Mock(return_value=None)
                        
                        response = client.get("/api/jobs/nonexistent")
                        
                        assert response.status_code == 404
                        assert 'Job not found' in response.json()['detail']
    
    def test_get_job_status_wrong_tenant(self, client, mock_user):
        """Test accessing job from different tenant"""
        with patch('automl_platform.api.api.SCHEDULER_AVAILABLE', True):
            with patch('automl_platform.api.api.AUTH_AVAILABLE', True):
                with patch('automl_platform.api.api.get_current_user', return_value=mock_user):
                    with patch('automl_platform.api.api.scheduler') as mock_scheduler:
                        mock_job = Mock(tenant_id='different_tenant')
                        mock_scheduler.get_job_status = Mock(return_value=mock_job)
                        
                        response = client.get("/api/jobs/job_123")
                        
                        assert response.status_code == 403
                        assert 'Access denied' in response.json()['detail']


class TestStatusEndpoint:
    """Tests for platform status endpoint"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user"""
        return Mock(
            id='user123',
            username='testuser',
            tenant_id='tenant123',
            plan_type='enterprise',
            organization='TestCorp'
        )
    
    def test_get_status_with_billing(self, client, mock_user):
        """Test status endpoint with billing information"""
        with patch('automl_platform.api.api.AUTH_AVAILABLE', True):
            with patch('automl_platform.api.api.get_current_user', return_value=mock_user):
                with patch('automl_platform.api.api.BILLING_AVAILABLE', True):
                    with patch('automl_platform.api.api.billing_manager') as mock_billing:
                        mock_billing.get_subscription = Mock(return_value={
                            'plan': 'enterprise',
                            'status': 'active',
                            'credits': 10000
                        })
                        mock_billing.usage_tracker.get_usage = Mock(return_value={
                            'models': 5,
                            'predictions': 1000,
                            'storage_gb': 2.5
                        })
                        
                        response = client.get("/api/status")
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert data['user']['username'] == 'testuser'
                        assert data['user']['plan'] == 'enterprise'
                        assert 'subscription' in data
                        assert 'usage' in data
    
    def test_get_status_with_scheduler(self, client, mock_user):
        """Test status endpoint with scheduler stats"""
        with patch('automl_platform.api.api.AUTH_AVAILABLE', True):
            with patch('automl_platform.api.api.get_current_user', return_value=mock_user):
                with patch('automl_platform.api.api.SCHEDULER_AVAILABLE', True):
                    with patch('automl_platform.api.api.scheduler') as mock_scheduler:
                        mock_scheduler.get_queue_stats = Mock(return_value={
                            'pending': 5,
                            'running': 2,
                            'completed': 100
                        })
                        
                        response = client.get("/api/status")
                        
                        assert response.status_code == 200
                        data = response.json()
                        assert 'queue_stats' in data
                        assert data['queue_stats']['pending'] == 5
    
    def test_get_status_no_auth(self, client):
        """Test status endpoint without authentication"""
        with patch('automl_platform.api.api.AUTH_AVAILABLE', False):
            response = client.get("/api/status")
            
            assert response.status_code == 503
            assert 'Authentication service is not available' in response.json()['detail']


class TestErrorHandling:
    """Tests for error handling"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_404_error(self, client):
        """Test 404 error for non-existent endpoint"""
        response = client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 error for wrong HTTP method"""
        response = client.get("/api/train")  # Should be POST
        
        assert response.status_code == 405
    
    def test_validation_error(self, client):
        """Test validation error for invalid request body"""
        with patch('automl_platform.api.api.AUTH_AVAILABLE', True):
            with patch('automl_platform.api.api.get_current_user', return_value=Mock()):
                # Missing required fields
                response = client.post("/api/train", json={})
                
                assert response.status_code == 422  # Validation error


class TestRequestModels:
    """Tests for request model validation"""
    
    def test_train_request_validation(self):
        """Test TrainRequest model validation"""
        # Valid request
        request = TrainRequest(
            task='classification',
            target_column='target',
            time_limit=300
        )
        
        assert request.task == 'classification'
        assert request.target_column == 'target'
        assert request.time_limit == 300
        assert request.enable_gpu is False  # Default value
    
    def test_predict_request_validation(self):
        """Test PredictRequest model validation"""
        request = PredictRequest(
            model_id='model_123',
            data={'feature1': 1.0, 'feature2': 'A'},
            return_probabilities=True
        )
        
        assert request.model_id == 'model_123'
        assert request.data == {'feature1': 1.0, 'feature2': 'A'}
        assert request.return_probabilities is True
    
    def test_batch_predict_request_validation(self):
        """Test BatchPredictRequest model validation"""
        request = BatchPredictRequest(
            model_id='model_123',
            data=[{'f1': 1}, {'f1': 2}],
            batch_size=500,
            output_format='csv'
        )
        
        assert request.model_id == 'model_123'
        assert len(request.data) == 2
        assert request.batch_size == 500
        assert request.output_format == 'csv'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
