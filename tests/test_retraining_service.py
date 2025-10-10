"""
Tests for retraining service
=============================
Tests for automated model retraining based on drift and performance.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.retraining_service import (
    RetrainingConfig,
    RetrainingService
)
from automl_platform.mlflow_registry import ModelStage


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for RetrainingService"""
    config = Mock()
    registry = Mock()
    monitor = Mock()
    storage_service = Mock()

    production_version = MagicMock()
    production_version.version = "1"
    production_version.current_stage = "Production"
    production_version.last_updated_timestamp = datetime.utcnow().timestamp() * 1000
    registry.get_production_model_metadata = Mock(return_value=production_version)
    registry.get_latest_production_version = Mock(return_value=production_version)

    return config, registry, monitor, storage_service


class TestRetrainingConfig:
    """Tests for RetrainingConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = RetrainingConfig()
        
        assert config.drift_threshold == 0.5
        assert config.performance_degradation_threshold == 0.1
        assert config.min_data_points == 1000
        assert config.min_accuracy_threshold == 0.8
        assert config.check_frequency == "daily"
        assert config.retrain_hour == 2
        assert config.max_retrain_per_day == 5
        assert config.use_gpu is False
        assert config.max_workers == 4
        assert config.timeout_minutes == 120
        assert config.notify_on_drift is True
        assert config.notify_on_retrain is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = RetrainingConfig()
        config.drift_threshold = 0.7
        config.check_frequency = "weekly"
        config.notification_emails = ["test@example.com"]
        config.slack_webhook = "https://hooks.slack.com/test"
        
        assert config.drift_threshold == 0.7
        assert config.check_frequency == "weekly"
        assert config.notification_emails == ["test@example.com"]
        assert config.slack_webhook == "https://hooks.slack.com/test"


class TestRetrainingService:
    """Tests for RetrainingService class"""

    @pytest.fixture
    def retraining_service(self, mock_dependencies):
        """Create RetrainingService instance with mocks"""
        config, registry, monitor, storage = mock_dependencies
        service = RetrainingService(config, registry, monitor, storage)
        return service
    
    def test_initialization(self, retraining_service):
        """Test service initialization"""
        assert isinstance(retraining_service.retrain_config, RetrainingConfig)
        assert retraining_service.retrain_history == []
        assert isinstance(retraining_service.last_check, datetime)
    
    def test_should_retrain_high_drift(self, retraining_service):
        """Test retraining triggered by high drift"""
        # Setup mocks
        retraining_service.monitor.get_drift_score = Mock(return_value=0.7)  # Above threshold
        retraining_service.monitor.get_performance_metrics = Mock(return_value={
            'baseline_accuracy': 0.9,
            'current_accuracy': 0.88
        })
        retraining_service.monitor.get_baseline_performance = Mock(return_value={'accuracy': 0.9})
        retraining_service.monitor.get_current_performance = Mock(return_value={'accuracy': 0.88})
        retraining_service.monitor.get_new_data_count = Mock(return_value=500)
        retraining_service.registry.get_model_history = Mock(return_value=[])

        stale_version = MagicMock()
        stale_version.version = "1"
        stale_version.current_stage = "Production"
        stale_version.last_updated_timestamp = (datetime.utcnow() - timedelta(days=45)).timestamp() * 1000
        retraining_service.registry.get_production_model_metadata.return_value = stale_version

        should_retrain, reason, metrics = retraining_service.should_retrain('test_model')

        assert should_retrain is True
        assert 'High drift detected' in reason
        assert metrics['drift_score'] == 0.7
        retraining_service.registry.get_production_model_metadata.assert_called_once_with('test_model')
        retraining_service.monitor.get_drift_score.assert_called_once_with('test_model')
        retraining_service.monitor.get_performance_metrics.assert_called_once_with('test_model')
        retraining_service.monitor.get_baseline_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_current_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_new_data_count.assert_called_once_with('test_model')

    def test_should_retrain_performance_degradation(self, retraining_service):
        """Test retraining triggered by performance degradation"""
        # Setup mocks
        retraining_service.monitor.get_drift_score = Mock(return_value=0.3)  # Below threshold
        retraining_service.monitor.get_performance_metrics = Mock(return_value={
            'baseline_accuracy': 0.9,
            'current_accuracy': 0.75  # 16.7% degradation
        })
        retraining_service.monitor.get_baseline_performance = Mock(return_value={'accuracy': 0.9})
        retraining_service.monitor.get_current_performance = Mock(return_value={'accuracy': 0.75})
        retraining_service.monitor.get_new_data_count = Mock(return_value=500)
        retraining_service.registry.get_model_history = Mock(return_value=[])

        should_retrain, reason, metrics = retraining_service.should_retrain('test_model')

        assert should_retrain is True
        assert 'Performance degradation' in reason
        assert metrics['degradation'] > 0.1
        retraining_service.monitor.get_drift_score.assert_called_once_with('test_model')
        retraining_service.monitor.get_performance_metrics.assert_called_once_with('test_model')
        retraining_service.monitor.get_baseline_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_current_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_new_data_count.assert_called_once_with('test_model')

    def test_should_retrain_low_accuracy(self, retraining_service):
        """Test retraining triggered by low absolute accuracy"""
        # Setup mocks
        retraining_service.monitor.get_drift_score = Mock(return_value=0.3)
        retraining_service.monitor.get_performance_metrics = Mock(return_value={
            'baseline_accuracy': 0.85,
            'current_accuracy': 0.7  # Below min threshold
        })
        retraining_service.monitor.get_baseline_performance = Mock(return_value={'accuracy': 0.85})
        retraining_service.monitor.get_current_performance = Mock(return_value={'accuracy': 0.7})
        retraining_service.monitor.get_new_data_count = Mock(return_value=500)
        retraining_service.registry.get_model_history = Mock(return_value=[])

        should_retrain, reason, metrics = retraining_service.should_retrain('test_model')

        assert should_retrain is True
        assert 'Accuracy below threshold' in reason
        assert metrics['current_accuracy'] == 0.7
        retraining_service.monitor.get_drift_score.assert_called_once_with('test_model')
        retraining_service.monitor.get_performance_metrics.assert_called_once_with('test_model')
        retraining_service.monitor.get_baseline_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_current_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_new_data_count.assert_called_once_with('test_model')

    def test_should_retrain_sufficient_new_data(self, retraining_service):
        """Test retraining triggered by sufficient new data"""
        # Setup mocks
        retraining_service.monitor.get_drift_score = Mock(return_value=0.3)
        retraining_service.monitor.get_performance_metrics = Mock(return_value={
            'baseline_accuracy': 0.9,
            'current_accuracy': 0.88
        })
        retraining_service.monitor.get_baseline_performance = Mock(return_value={'accuracy': 0.9})
        retraining_service.monitor.get_current_performance = Mock(return_value={'accuracy': 0.88})
        retraining_service.monitor.get_new_data_count = Mock(return_value=1500)  # Above threshold
        retraining_service.registry.get_model_history = Mock(return_value=[])

        should_retrain, reason, metrics = retraining_service.should_retrain('test_model')

        assert should_retrain is True
        assert 'Sufficient new data' in reason
        assert metrics['new_data_count'] == 1500
        retraining_service.monitor.get_drift_score.assert_called_once_with('test_model')
        retraining_service.monitor.get_performance_metrics.assert_called_once_with('test_model')
        retraining_service.monitor.get_baseline_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_current_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_new_data_count.assert_called_once_with('test_model')

    def test_should_retrain_old_model(self, retraining_service):
        """Test retraining triggered by model age"""
        # Setup mocks
        retraining_service.monitor.get_drift_score = Mock(return_value=0.3)
        retraining_service.monitor.get_performance_metrics = Mock(return_value={
            'baseline_accuracy': 0.9,
            'current_accuracy': 0.88
        })
        retraining_service.monitor.get_baseline_performance = Mock(return_value={'accuracy': 0.9})
        retraining_service.monitor.get_current_performance = Mock(return_value={'accuracy': 0.88})
        retraining_service.monitor.get_new_data_count = Mock(return_value=500)

        stale_version = MagicMock()
        stale_version.version = "1"
        stale_version.current_stage = "Production"
        stale_version.last_updated_timestamp = (datetime.utcnow() - timedelta(days=45)).timestamp() * 1000
        retraining_service.registry.get_production_model_metadata.return_value = stale_version

        # Model trained 40 days ago
        old_date = (datetime.utcnow() - timedelta(days=40)).isoformat()
        retraining_service.registry.get_model_history = Mock(return_value=[
            {'created_at': old_date}
        ])

        should_retrain, reason, metrics = retraining_service.should_retrain('test_model')

        assert should_retrain is True
        assert 'days old' in reason
        assert metrics['days_since_training'] >= 30
        retraining_service.monitor.get_drift_score.assert_called_once_with('test_model')
        retraining_service.monitor.get_performance_metrics.assert_called_once_with('test_model')
        retraining_service.monitor.get_baseline_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_current_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_new_data_count.assert_called_once_with('test_model')

    def test_should_handle_integer_timestamps(self, retraining_service):
        """Ensure integer timestamps from MLflow history are parsed safely."""
        retraining_service.monitor.get_drift_score = Mock(return_value=0.3)
        retraining_service.monitor.get_performance_metrics = Mock(return_value={
            'baseline_accuracy': 0.9,
            'current_accuracy': 0.88
        })
        retraining_service.monitor.get_baseline_performance = Mock(return_value={'accuracy': 0.9})
        retraining_service.monitor.get_current_performance = Mock(return_value={'accuracy': 0.88})
        retraining_service.monitor.get_new_data_count = Mock(return_value=500)

        retraining_service.registry.get_production_model_metadata.return_value.last_updated_timestamp = None

        old_datetime = datetime.utcnow() - timedelta(days=35)
        integer_timestamp = int(old_datetime.timestamp() * 1000)  # milliseconds since epoch
        retraining_service.registry.get_model_history = Mock(return_value=[
            {'created_at': integer_timestamp}
        ])

        should_retrain, reason, metrics = retraining_service.should_retrain('test_model')

        assert should_retrain is True
        assert 'days old' in reason
        assert metrics['days_since_training'] >= 30
        retraining_service.monitor.get_drift_score.assert_called_once_with('test_model')
        retraining_service.monitor.get_performance_metrics.assert_called_once_with('test_model')
        retraining_service.monitor.get_baseline_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_current_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_new_data_count.assert_called_once_with('test_model')

    def test_should_not_retrain(self, retraining_service):
        """Test when retraining should not be triggered"""
        # Setup mocks - all conditions are good
        retraining_service.monitor.get_drift_score = Mock(return_value=0.3)
        retraining_service.monitor.get_performance_metrics = Mock(return_value={
            'baseline_accuracy': 0.9,
            'current_accuracy': 0.88
        })
        retraining_service.monitor.get_baseline_performance = Mock(return_value={'accuracy': 0.9})
        retraining_service.monitor.get_current_performance = Mock(return_value={'accuracy': 0.88})
        retraining_service.monitor.get_new_data_count = Mock(return_value=500)

        # Model trained recently
        retraining_service.registry.get_production_model_metadata.return_value.last_updated_timestamp = None

        recent_date = (datetime.utcnow() - timedelta(days=5)).isoformat()
        retraining_service.registry.get_model_history = Mock(return_value=[
            {'created_at': recent_date}
        ])

        should_retrain, reason, metrics = retraining_service.should_retrain('test_model')

        assert should_retrain is False
        assert reason == "No retraining needed"
        retraining_service.monitor.get_drift_score.assert_called_once_with('test_model')
        retraining_service.monitor.get_performance_metrics.assert_called_once_with('test_model')
        retraining_service.monitor.get_baseline_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_current_performance.assert_called_once_with('test_model')
        retraining_service.monitor.get_new_data_count.assert_called_once_with('test_model')

    @pytest.mark.asyncio
    async def test_retrain_model_promotes_with_valid_stage(self):
        """Retraining should promote using a valid ModelStage enum"""
        config = Mock()
        registry = Mock()
        monitor = Mock()
        storage = Mock()

        service = RetrainingService(config, registry, monitor, storage)
        service._validate_model = AsyncMock(return_value=True)
        service._send_notification = AsyncMock()

        registry.client = None

        mock_version = Mock()
        mock_version.version = 1
        registry.register_model = Mock(return_value=mock_version)
        registry.promote_model = Mock(return_value=True)

        leaderboard = pd.DataFrame([{'accuracy': 0.9}])

        mock_orchestrator = MagicMock()
        mock_orchestrator.fit.return_value = None
        mock_orchestrator.best_pipeline = Mock()
        mock_orchestrator.get_leaderboard.return_value = leaderboard

        X_train = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
        y_train = pd.Series(np.random.randint(0, 2, size=20))

        with patch('automl_platform.orchestrator.AutoMLOrchestrator', return_value=mock_orchestrator):
            result = await service.retrain_model('test_model', X_train, y_train, 'automated test')

        registry.promote_model.assert_called_once_with('test_model', mock_version.version, ModelStage.STAGING)
        assert result['status'] == 'success'
    
    def test_should_retrain_no_production_model(self, retraining_service):
        """Test when no production model exists"""
        retraining_service.registry.get_production_model_metadata.return_value = None

        should_retrain, reason, metrics = retraining_service.should_retrain('test_model')

        assert should_retrain is False
        assert reason == "No production model found"
        assert metrics == {}
        retraining_service.registry.get_production_model_metadata.assert_called_once_with('test_model')
        retraining_service.monitor.get_drift_score.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_retrain_model_success(self, retraining_service):
        """Test successful model retraining"""
        # Create test data
        X_train = pd.DataFrame(np.random.randn(100, 5))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        # Setup mocks
        retraining_service.registry.client = Mock()
        retraining_service.registry.client.get_latest_versions = Mock(return_value=[
            Mock(run_id='run_123')
        ])
        retraining_service.registry.client.get_run = Mock(return_value=Mock(
            data=Mock(params={'algorithm': 'RandomForestClassifier'})
        ))
        
        # Mock orchestrator
        with patch('automl_platform.orchestrator.AutoMLOrchestrator') as mock_orchestrator:
            mock_instance = Mock()
            mock_instance.best_pipeline = Mock()
            mock_instance.get_leaderboard = Mock(return_value=pd.DataFrame({
                'accuracy': [0.9],
                'f1': [0.88]
            }))
            mock_orchestrator.return_value = mock_instance
            
            # Mock registry methods
            mock_version = Mock(version=2)
            retraining_service.registry.register_model = Mock(return_value=mock_version)
            retraining_service.registry.promote_model = Mock(return_value=True)
            
            # Mock validation
            retraining_service._validate_model = AsyncMock(return_value=True)
            retraining_service._send_notification = AsyncMock()
            
            result = await retraining_service.retrain_model(
                'test_model',
                X_train,
                y_train,
                reason='Test retraining'
            )
        
        assert result['status'] == 'success'
        assert result['model_name'] == 'test_model'
        assert result['new_version'] == 2
        assert result['promoted_to'] == 'staging'
        assert 'metrics' in result
        
        # Verify registry methods were called
        retraining_service.registry.register_model.assert_called_once()
        retraining_service.registry.promote_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrain_model_validation_failed(self, retraining_service):
        """Test model retraining with validation failure"""
        X_train = pd.DataFrame(np.random.randn(100, 5))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        # Setup mocks for failed validation
        with patch('automl_platform.orchestrator.AutoMLOrchestrator') as mock_orchestrator:
            mock_instance = Mock()
            mock_instance.best_pipeline = Mock()
            mock_instance.get_leaderboard = Mock(return_value=pd.DataFrame({
                'accuracy': [0.6],  # Low accuracy
                'f1': [0.5]
            }))
            mock_orchestrator.return_value = mock_instance
            
            mock_version = Mock(version=2)
            retraining_service.registry.register_model = Mock(return_value=mock_version)
            retraining_service.registry.client = None  # Use local registry
            
            # Mock validation to fail
            retraining_service._validate_model = AsyncMock(return_value=False)
            retraining_service._send_notification = AsyncMock()
            
            result = await retraining_service.retrain_model(
                'test_model',
                X_train,
                y_train,
                reason='Test retraining'
            )
        
        assert result['status'] == 'validation_failed'
        assert result['error'] == 'Model did not pass validation'
        
        # Verify model was not promoted
        retraining_service.registry.promote_model.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_retrain_model_exception(self, retraining_service):
        """Test model retraining with exception"""
        X_train = pd.DataFrame(np.random.randn(100, 5))
        y_train = pd.Series(np.random.randint(0, 2, 100))

        # Setup mocks to raise exception
        retraining_service.registry.client = Mock()
        retraining_service.registry.client.get_latest_versions = Mock(return_value=[])

        with patch('automl_platform.orchestrator.AutoMLOrchestrator') as mock_orchestrator:
            mock_orchestrator.side_effect = Exception("Training failed")

            retraining_service._send_notification = AsyncMock()

            result = await retraining_service.retrain_model(
                'test_model',
                X_train,
                y_train,
                reason='Test retraining'
            )
        
        assert result['status'] == 'failed'
        assert 'Training failed' in result['error']
    
    @pytest.mark.asyncio
    async def test_validate_model_success(self, retraining_service):
        """Test successful model validation"""
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
        
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        metrics = {
            'accuracy': 0.85,
            'cv_score': 0.83,
            'train_score': 0.87
        }
        
        is_valid = await retraining_service._validate_model(model, X, y, metrics)
        
        assert is_valid is True
    
    @pytest.mark.asyncio
    async def test_validate_model_low_accuracy(self, retraining_service):
        """Test validation failure due to low accuracy"""
        model = Mock()
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        metrics = {
            'accuracy': 0.6,  # Below threshold
            'cv_score': 0.58,
            'train_score': 0.62
        }
        
        is_valid = await retraining_service._validate_model(model, X, y, metrics)
        
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_validate_model_overfitting(self, retraining_service):
        """Test validation failure due to overfitting"""
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
        
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        metrics = {
            'accuracy': 0.85,
            'cv_score': 0.70,  # Large gap with train_score
            'train_score': 0.95
        }
        
        is_valid = await retraining_service._validate_model(model, X, y, metrics)
        
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_send_notification_success(self, retraining_service):
        """Test successful notification sending"""
        result = {
            'model_name': 'test_model',
            'reason': 'High drift',
            'status': 'success',
            'new_version': 2,
            'promoted_to': 'staging'
        }
        
        # Mock requests for Slack
        with patch('automl_platform.retraining_service.requests', create=True) as mock_requests:
            retraining_service.retrain_config.slack_webhook = 'https://hooks.slack.com/test'
            
            await retraining_service._send_notification(result)
            
            mock_requests.post.assert_called_once()
            call_args = mock_requests.post.call_args
            assert 'https://hooks.slack.com/test' in call_args[0]
    
    def test_get_retraining_history(self, retraining_service):
        """Test getting retraining history"""
        # Add some history
        retraining_service.retrain_history = [
            {'model_name': 'model1', 'end_time': '2024-01-01T10:00:00'},
            {'model_name': 'model2', 'end_time': '2024-01-02T10:00:00'},
            {'model_name': 'model3', 'end_time': '2024-01-03T10:00:00'},
        ]
        
        history = retraining_service.get_retraining_history(limit=2)
        
        assert len(history) == 2
        assert history[0]['model_name'] == 'model3'  # Most recent first
        assert history[1]['model_name'] == 'model2'
    
    def test_get_retraining_stats(self, retraining_service):
        """Test getting retraining statistics"""
        # Add some history with different statuses
        retraining_service.retrain_history = [
            {
                'status': 'success',
                'start_time': '2024-01-01T10:00:00',
                'end_time': '2024-01-01T10:30:00'
            },
            {
                'status': 'success',
                'start_time': '2024-01-02T10:00:00',
                'end_time': '2024-01-02T10:45:00'
            },
            {
                'status': 'failed',
                'start_time': '2024-01-03T10:00:00',
                'end_time': '2024-01-03T10:15:00'
            },
        ]
        
        stats = retraining_service.get_retraining_stats()
        
        assert stats['total_retrainings'] == 3
        assert stats['successful'] == 2
        assert stats['failed'] == 1
        assert stats['success_rate'] == 2/3
        assert stats['average_duration_seconds'] > 0
        assert stats['last_check'] is not None


class TestSchedulingIntegration:
    """Tests for scheduling integration (Airflow/Prefect)"""
    
    @pytest.fixture
    def service_with_scheduler(self, mock_dependencies):
        """Create service with scheduling capabilities"""
        config, registry, monitor, storage = mock_dependencies
        service = RetrainingService(config, registry, monitor, storage)
        return service
    
    @patch('automl_platform.retraining_service.AIRFLOW_AVAILABLE', True)
    @patch('automl_platform.retraining_service.DAG', create=True)
    @patch('automl_platform.retraining_service.PythonOperator', create=True)
    @patch('automl_platform.retraining_service.days_ago', create=True)
    def test_create_airflow_dag(self, mock_days_ago, mock_operator, mock_dag, service_with_scheduler):
        """Test Airflow DAG creation"""
        service_with_scheduler.retrain_config.check_frequency = 'daily'
        
        dag = service_with_scheduler.create_retraining_schedule()

        # Verify DAG was created
        mock_dag.assert_called_once()
        args, kwargs = mock_dag.call_args
        assert args[0] == 'model_retraining'
        assert kwargs['schedule_interval'] == '@daily'
        
        # Verify operators were created
        assert mock_operator.call_count >= 3  # At least 3 tasks
    
    @patch('automl_platform.retraining_service.PREFECT_AVAILABLE', True)
    @patch('automl_platform.retraining_service.AIRFLOW_AVAILABLE', False)
    @patch('automl_platform.retraining_service.flow', create=True)
    @patch('automl_platform.retraining_service.task', create=True)
    @patch('automl_platform.retraining_service.Deployment', create=True)
    @patch('automl_platform.retraining_service.ConcurrentTaskRunner', create=True)
    @patch('automl_platform.retraining_service.CronSchedule', create=True)
    @patch('automl_platform.retraining_service.IntervalSchedule', create=True)
    def test_create_prefect_flow(self, mock_interval, mock_cron, mock_runner, mock_deployment, mock_task, mock_flow, service_with_scheduler):
        """Test Prefect flow creation"""
        service_with_scheduler.retrain_config.check_frequency = 'weekly'
        
        deployment = service_with_scheduler.create_retraining_schedule()
        
        # Verify deployment was created
        mock_deployment.build_from_flow.assert_called_once()
    
    @patch('automl_platform.retraining_service.AIRFLOW_AVAILABLE', False)
    @patch('automl_platform.retraining_service.PREFECT_AVAILABLE', False)
    def test_no_scheduler_available(self, service_with_scheduler):
        """Test when no scheduler is available"""
        result = service_with_scheduler.create_retraining_schedule()
        
        assert result is None


class TestRetrainingConfigInterpretation:
    """Tests for correct interpretation of RetrainingConfig"""
    
    def test_config_affects_should_retrain(self):
        """Test that config thresholds are correctly used"""
        config = Mock()
        registry = Mock()
        monitor = Mock()
        storage = Mock()
        
        service = RetrainingService(config, registry, monitor, storage)
        
        # Modify config
        service.retrain_config.drift_threshold = 0.3
        service.retrain_config.performance_degradation_threshold = 0.05
        service.retrain_config.min_data_points = 500
        
        # Setup mocks
        production_version = MagicMock()
        production_version.version = "2"
        production_version.current_stage = "Production"
        production_version.last_updated_timestamp = datetime.utcnow().timestamp() * 1000
        registry.get_production_model_metadata = Mock(return_value=production_version)
        registry.get_latest_production_version = Mock(return_value=production_version)
        monitor.get_drift_score = Mock(return_value=0.35)  # Above new threshold
        monitor.get_performance_metrics = Mock(return_value={
            'baseline_accuracy': 0.9,
            'current_accuracy': 0.84  # 6.7% degradation, above new threshold
        })
        monitor.get_baseline_performance = Mock(return_value={'accuracy': 0.9})
        monitor.get_current_performance = Mock(return_value={'accuracy': 0.84})
        monitor.get_new_data_count = Mock(return_value=600)  # Above new threshold
        registry.get_model_history = Mock(return_value=[])
        
        should_retrain, reason, metrics = service.should_retrain('test_model')

        assert should_retrain is True
        assert 'High drift detected' in reason
        assert 'Performance degradation' in reason
        assert 'Sufficient new data' in reason
        monitor.get_drift_score.assert_called_once_with('test_model')
        monitor.get_performance_metrics.assert_called_once_with('test_model')
        monitor.get_new_data_count.assert_called_once_with('test_model')
    
    def test_config_max_retrain_per_day(self):
        """Test that max_retrain_per_day is respected"""
        config = Mock()
        registry = Mock()
        monitor = Mock()
        storage = Mock()
        
        service = RetrainingService(config, registry, monitor, storage)
        service.retrain_config.max_retrain_per_day = 2
        
        # This would be tested in the actual scheduling implementation
        # Here we just verify the config is accessible
        assert service.retrain_config.max_retrain_per_day == 2
    
    def test_config_notification_settings(self):
        """Test notification configuration"""
        config = Mock()
        registry = Mock()
        monitor = Mock()
        storage = Mock()
        
        service = RetrainingService(config, registry, monitor, storage)
        
        # Test email configuration
        service.retrain_config.notification_emails = ['admin@example.com', 'ml@example.com']
        assert len(service.retrain_config.notification_emails) == 2
        
        # Test Slack configuration
        service.retrain_config.slack_webhook = 'https://hooks.slack.com/services/T00/B00/XXX'
        assert 'hooks.slack.com' in service.retrain_config.slack_webhook
        
        # Test notification flags
        service.retrain_config.notify_on_drift = False
        service.retrain_config.notify_on_retrain = False
        assert service.retrain_config.notify_on_drift is False
        assert service.retrain_config.notify_on_retrain is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
