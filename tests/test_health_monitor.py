"""
Tests for HealthMonitor
=======================
Comprehensive tests for the health monitoring service.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
import time

from automl_platform.core.health_monitor import (
    HealthMonitor,
    HealthCheck,
    HealthStatus
)


class TestHealthMonitor:
    """Test suite for HealthMonitor."""
    
    def test_initialization_with_different_intervals(self):
        """Test HealthMonitor initialization with different check intervals."""
        # Test default interval
        monitor = HealthMonitor()
        assert monitor.check_interval == 30
        assert monitor.is_monitoring == False
        assert monitor.monitoring_task is None
        
        # Test custom interval
        monitor = HealthMonitor(check_interval_seconds=60)
        assert monitor.check_interval == 60
        
        # Test with very short interval
        monitor = HealthMonitor(check_interval_seconds=5)
        assert monitor.check_interval == 5
    
    def test_default_thresholds(self):
        """Test default health check thresholds."""
        monitor = HealthMonitor()
        thresholds = monitor._default_thresholds()
        
        assert thresholds["max_latency_ms"] == 5000
        assert thresholds["min_memory_available_gb"] == 1.0
        assert thresholds["max_cpu_usage_percent"] == 90.0
        assert thresholds["max_disk_usage_percent"] == 85.0
        assert thresholds["max_queue_size"] == 1000
        assert thresholds["max_error_rate"] == 0.05
    
    def test_register_check(self):
        """Test registering custom health check functions."""
        monitor = HealthMonitor()
        
        async def custom_check():
            return HealthCheck(
                service="custom_service",
                status=HealthStatus.HEALTHY,
                message="Custom service is healthy"
            )
        
        monitor.register_check("custom_service", custom_check)
        assert "custom_service" in monitor.custom_checks
        assert monitor.custom_checks["custom_service"] == custom_check
    
    @pytest.mark.asyncio
    async def test_check_scheduler_healthy(self):
        """Test scheduler health check - healthy state."""
        monitor = HealthMonitor()
        
        # Mock healthy scheduler
        scheduler = Mock()
        scheduler.get_queue_stats.return_value = {
            'workers': 5,
            'queued_jobs': 10,
            'active_jobs': 3,
            'gpu_workers': 2
        }
        
        result = await monitor.check_scheduler(scheduler)
        
        assert result.service == "scheduler"
        assert result.status == HealthStatus.HEALTHY
        assert "5 workers, 3 active jobs" in result.message
        assert result.details['workers'] == 5
        assert result.details['queued_jobs'] == 10
        assert result.details['active_jobs'] == 3
        assert result.details['gpu_workers'] == 2
        assert result.latency_ms >= 0
    
    @pytest.mark.asyncio
    async def test_check_scheduler_degraded(self):
        """Test scheduler health check - degraded state (high queue size)."""
        monitor = HealthMonitor()
        
        # Mock scheduler with high queue
        scheduler = Mock()
        scheduler.get_queue_stats.return_value = {
            'workers': 5,
            'queued_jobs': 1500,  # Above threshold
            'active_jobs': 3,
            'gpu_workers': 2
        }
        
        result = await monitor.check_scheduler(scheduler)
        
        assert result.service == "scheduler"
        assert result.status == HealthStatus.DEGRADED
        assert "High queue size: 1500" in result.message
    
    @pytest.mark.asyncio
    async def test_check_scheduler_unhealthy(self):
        """Test scheduler health check - unhealthy state (no workers)."""
        monitor = HealthMonitor()
        
        # Mock scheduler with no workers
        scheduler = Mock()
        scheduler.get_queue_stats.return_value = {
            'workers': 0,
            'queued_jobs': 100,
            'active_jobs': 0,
            'gpu_workers': 0
        }
        
        result = await monitor.check_scheduler(scheduler)
        
        assert result.service == "scheduler"
        assert result.status == HealthStatus.UNHEALTHY
        assert "No workers available" in result.message
    
    @pytest.mark.asyncio
    async def test_check_scheduler_exception(self):
        """Test scheduler health check - exception handling."""
        monitor = HealthMonitor()
        
        # Mock scheduler that raises exception
        scheduler = Mock()
        scheduler.get_queue_stats.side_effect = Exception("Connection failed")
        
        result = await monitor.check_scheduler(scheduler)
        
        assert result.service == "scheduler"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in result.message
        assert result.latency_ms >= 0
    
    @pytest.mark.asyncio
    async def test_check_scheduler_unknown(self):
        """Test scheduler health check - unknown state (not configured)."""
        monitor = HealthMonitor()
        
        result = await monitor.check_scheduler(None)
        
        assert result.service == "scheduler"
        assert result.status == HealthStatus.UNKNOWN
        assert "Scheduler not configured" in result.message
    
    @pytest.mark.asyncio
    async def test_check_storage_healthy(self):
        """Test storage health check - healthy state."""
        monitor = HealthMonitor()
        
        # Mock healthy storage
        storage = Mock()
        storage.backend_type = "s3"
        storage.list_models.return_value = ["model1", "model2"]
        
        result = await monitor.check_storage(storage)
        
        assert result.service == "storage"
        assert result.status == HealthStatus.HEALTHY
        assert "s3 storage operational" in result.message
        assert result.details['backend'] == "s3"
        assert result.details['test_successful'] == True
    
    @pytest.mark.asyncio
    async def test_check_storage_degraded_local(self):
        """Test storage health check - degraded state (high disk usage for local)."""
        monitor = HealthMonitor()
        
        # Mock local storage with high disk usage
        storage = Mock()
        storage.backend_type = "local"
        storage.list_models.return_value = []
        
        with patch('psutil.disk_usage') as mock_disk:
            mock_disk.return_value = Mock(percent=90.0)  # High disk usage
            
            result = await monitor.check_storage(storage)
            
            assert result.service == "storage"
            assert result.status == HealthStatus.DEGRADED
            assert "High disk usage: 90.0%" in result.message
            assert result.details['backend'] == "local"
            assert result.details['disk_usage_percent'] == 90.0
    
    @pytest.mark.asyncio
    async def test_check_storage_unhealthy(self):
        """Test storage health check - unhealthy state."""
        monitor = HealthMonitor()
        
        # Mock storage that fails
        storage = Mock()
        storage.backend_type = "s3"
        storage.list_models.side_effect = Exception("S3 connection timeout")
        
        result = await monitor.check_storage(storage)
        
        assert result.service == "storage"
        assert result.status == HealthStatus.UNHEALTHY
        assert "S3 connection timeout" in result.message
    
    @pytest.mark.asyncio
    async def test_check_storage_unknown(self):
        """Test storage health check - unknown state (not configured)."""
        monitor = HealthMonitor()
        
        result = await monitor.check_storage(None)
        
        assert result.service == "storage"
        assert result.status == HealthStatus.UNKNOWN
        assert "Storage not configured" in result.message
    
    @pytest.mark.asyncio
    async def test_check_billing_healthy(self):
        """Test billing health check - healthy state."""
        monitor = HealthMonitor()
        
        # Mock healthy billing manager
        billing_manager = Mock()
        billing_manager.get_system_summary.return_value = {
            "total_active_subscriptions": 42,
            "total_revenue": 10000
        }
        
        result = await monitor.check_billing(billing_manager)
        
        assert result.service == "billing"
        assert result.status == HealthStatus.HEALTHY
        assert "42 active subscriptions" in result.message
    
    @pytest.mark.asyncio
    async def test_check_billing_unhealthy(self):
        """Test billing health check - unhealthy state."""
        monitor = HealthMonitor()
        
        # Mock billing manager with error
        billing_manager = Mock()
        billing_manager.get_system_summary.return_value = {
            "error": "Database connection failed"
        }
        
        result = await monitor.check_billing(billing_manager)
        
        assert result.service == "billing"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Database connection failed" in result.message
    
    @pytest.mark.asyncio
    async def test_check_streaming_healthy(self):
        """Test streaming health check - healthy state."""
        monitor = HealthMonitor()
        
        # Mock healthy streaming orchestrators
        orchestrator1 = Mock()
        orchestrator1.get_metrics.return_value = {
            "status": "running",
            "messages_processed": 1000,
            "messages_failed": 10
        }
        
        orchestrator2 = Mock()
        orchestrator2.get_metrics.return_value = {
            "status": "running",
            "messages_processed": 2000,
            "messages_failed": 20
        }
        
        streaming_orchestrators = {
            "stream1": orchestrator1,
            "stream2": orchestrator2
        }
        
        result = await monitor.check_streaming(streaming_orchestrators)
        
        assert result.service == "streaming"
        assert result.status == HealthStatus.HEALTHY
        assert "2/2 pipelines healthy" in result.message
        assert result.details['active_pipelines'] == 2
        assert result.details['healthy_pipelines'] == 2
        assert result.details['total_messages_processed'] == 3000
        assert result.details['total_messages_failed'] == 30
    
    @pytest.mark.asyncio
    async def test_check_streaming_degraded(self):
        """Test streaming health check - degraded state (high error rate)."""
        monitor = HealthMonitor()
        
        # Mock streaming with high error rate
        orchestrator = Mock()
        orchestrator.get_metrics.return_value = {
            "status": "running",
            "messages_processed": 100,
            "messages_failed": 10  # 9% error rate
        }
        
        streaming_orchestrators = {"stream1": orchestrator}
        
        result = await monitor.check_streaming(streaming_orchestrators)
        
        assert result.service == "streaming"
        assert result.status == HealthStatus.DEGRADED
        assert "High error rate" in result.message
    
    @pytest.mark.asyncio
    async def test_check_system_resources_healthy(self):
        """Test system resources health check - healthy state."""
        monitor = HealthMonitor()
        
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_cpu.return_value = 50.0
            mock_memory.return_value = Mock(
                percent=60.0,
                available=8 * 1024**3  # 8GB available
            )
            mock_disk.return_value = Mock(
                percent=70.0,
                free=100 * 1024**3  # 100GB free
            )
            
            result = await monitor.check_system_resources()
            
            assert result.service == "system"
            assert result.status == HealthStatus.HEALTHY
            assert "System resources within normal range" in result.message
    
    @pytest.mark.asyncio
    async def test_check_system_resources_degraded(self):
        """Test system resources health check - degraded state."""
        monitor = HealthMonitor()
        
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_cpu.return_value = 95.0  # High CPU
            mock_memory.return_value = Mock(
                percent=90.0,
                available=0.5 * 1024**3  # Low memory
            )
            mock_disk.return_value = Mock(
                percent=90.0,  # High disk usage
                free=10 * 1024**3
            )
            
            result = await monitor.check_system_resources()
            
            assert result.service == "system"
            assert result.status == HealthStatus.DEGRADED
            assert "High CPU" in result.message
            assert "Low memory" in result.message
            assert "High disk usage" in result.message
    
    @pytest.mark.asyncio
    async def test_check_all_with_custom_checks(self):
        """Test running all health checks including custom ones."""
        monitor = HealthMonitor()
        
        # Register custom check
        async def custom_check():
            return HealthCheck(
                service="custom",
                status=HealthStatus.HEALTHY,
                message="Custom service OK"
            )
        
        monitor.register_check("custom", custom_check)
        
        # Mock app state
        app_state = Mock()
        app_state.scheduler = Mock()
        app_state.scheduler.get_queue_stats.return_value = {
            'workers': 5,
            'queued_jobs': 10,
            'active_jobs': 3,
            'gpu_workers': 0
        }
        
        with patch.object(monitor, 'check_system_resources') as mock_system:
            mock_system.return_value = HealthCheck(
                service="system",
                status=HealthStatus.HEALTHY,
                message="System OK"
            )
            
            result = await monitor.check_all(app_state)
            
            assert result['overall'] == 'healthy'
            assert 'system' in result['services']
            assert 'scheduler' in result['services']
            assert 'custom' in result['services']
            assert result['services']['custom']['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping continuous monitoring."""
        monitor = HealthMonitor(check_interval_seconds=0.1)  # Short interval for testing
        
        app_state = Mock()
        
        with patch.object(monitor, 'check_all', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = {"overall": "healthy"}
            
            # Start monitoring
            await monitor.start_monitoring(app_state)
            assert monitor.is_monitoring == True
            assert monitor.monitoring_task is not None
            
            # Wait a bit to ensure monitoring runs
            await asyncio.sleep(0.3)
            
            # Stop monitoring
            await monitor.stop_monitoring()
            assert monitor.is_monitoring == False
            
            # Verify check_all was called multiple times
            assert mock_check.call_count >= 2
    
    def test_set_threshold(self):
        """Test updating health check thresholds."""
        monitor = HealthMonitor()
        
        # Set new threshold
        monitor.set_threshold("max_cpu_usage_percent", 95.0)
        assert monitor.thresholds["max_cpu_usage_percent"] == 95.0
        
        # Set another threshold
        monitor.set_threshold("custom_threshold", 100)
        assert monitor.thresholds["custom_threshold"] == 100
    
    def test_get_latest_status(self):
        """Test getting latest health check results."""
        monitor = HealthMonitor()
        
        # No checks performed yet
        result = monitor.get_latest_status()
        assert result['overall'] == 'unknown'
        assert 'No health checks performed yet' in result['message']
        
        # Add some check results
        monitor.checks = {
            'service1': HealthCheck(
                service='service1',
                status=HealthStatus.HEALTHY,
                message='OK',
                latency_ms=10
            ),
            'service2': HealthCheck(
                service='service2',
                status=HealthStatus.DEGRADED,
                message='Warning',
                latency_ms=20
            )
        }
        
        result = monitor.get_latest_status()
        assert result['overall'] == 'degraded'
        assert 'service1' in result['services']
        assert 'service2' in result['services']
        assert result['services']['service1']['status'] == 'healthy'
        assert result['services']['service2']['status'] == 'degraded'
    
    def test_get_unhealthy_and_degraded_services(self):
        """Test getting lists of unhealthy and degraded services."""
        monitor = HealthMonitor()
        
        monitor.checks = {
            'healthy_service': HealthCheck(
                service='healthy_service',
                status=HealthStatus.HEALTHY,
                message='OK'
            ),
            'degraded_service': HealthCheck(
                service='degraded_service',
                status=HealthStatus.DEGRADED,
                message='Warning'
            ),
            'unhealthy_service': HealthCheck(
                service='unhealthy_service',
                status=HealthStatus.UNHEALTHY,
                message='Error'
            )
        }
        
        unhealthy = monitor.get_unhealthy_services()
        assert unhealthy == ['unhealthy_service']
        
        degraded = monitor.get_degraded_services()
        assert degraded == ['degraded_service']
    
    def test_export_metrics_prometheus(self):
        """Test exporting metrics in Prometheus format."""
        monitor = HealthMonitor()
        
        monitor.checks = {
            'service1': HealthCheck(
                service='service1',
                status=HealthStatus.HEALTHY,
                message='OK',
                latency_ms=15.5,
                details={'cpu_usage': 45.2, 'memory_mb': 1024}
            ),
            'service2': HealthCheck(
                service='service2',
                status=HealthStatus.DEGRADED,
                message='Warning',
                latency_ms=100.0,
                details={'error_count': 5}
            )
        }
        
        metrics = monitor.export_metrics_prometheus()
        
        # Check overall health metric
        assert 'automl_health_overall 0.5' in metrics  # degraded = 0.5
        
        # Check service metrics
        assert 'automl_health_service{service="service1"} 1' in metrics
        assert 'automl_health_service{service="service2"} 0.5' in metrics
        
        # Check latency metrics
        assert 'automl_health_latency_ms{service="service1"} 15.5' in metrics
        assert 'automl_health_latency_ms{service="service2"} 100.0' in metrics
        
        # Check detail metrics
        assert 'automl_health_cpu_usage{service="service1"} 45.2' in metrics
        assert 'automl_health_memory_mb{service="service1"} 1024' in metrics
        assert 'automl_health_error_count{service="service2"} 5' in metrics
