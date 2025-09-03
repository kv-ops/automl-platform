"""
Health Monitor for AutoML Platform
===================================
Comprehensive health monitoring for all platform services.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import time

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result."""
    service: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0
    details: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.details is None:
            self.details = {}


class HealthMonitor:
    """
    Comprehensive health monitoring for all platform services.
    Performs regular health checks and aggregates status.
    """
    
    def __init__(self, check_interval_seconds: int = 30):
        """
        Initialize health monitor.
        
        Args:
            check_interval_seconds: Interval between health checks
        """
        self.check_interval = check_interval_seconds
        self.checks: Dict[str, HealthCheck] = {}
        self.custom_checks: Dict[str, Callable] = {}
        self.thresholds = self._default_thresholds()
        self.monitoring_task = None
        self.is_monitoring = False
        
        logger.info(f"HealthMonitor initialized with {check_interval_seconds}s interval")
    
    def _default_thresholds(self) -> Dict[str, Any]:
        """Default health check thresholds."""
        return {
            "max_latency_ms": 5000,
            "min_memory_available_gb": 1.0,
            "max_cpu_usage_percent": 90.0,
            "max_disk_usage_percent": 85.0,
            "max_queue_size": 1000,
            "max_error_rate": 0.05
        }
    
    def register_check(self, service: str, check_func: Callable) -> None:
        """
        Register a custom health check function.
        
        Args:
            service: Service name
            check_func: Async function that returns HealthCheck
        """
        self.custom_checks[service] = check_func
        logger.info(f"Registered custom health check for {service}")
    
    async def check_scheduler(self, scheduler: Any) -> HealthCheck:
        """Check scheduler health."""
        start_time = time.time()
        
        try:
            if not scheduler:
                return HealthCheck(
                    service="scheduler",
                    status=HealthStatus.UNKNOWN,
                    message="Scheduler not configured"
                )
            
            # Get queue statistics
            stats = {}
            if hasattr(scheduler, 'get_queue_stats'):
                stats = scheduler.get_queue_stats()
            
            workers = stats.get('workers', 0)
            queued_jobs = stats.get('queued_jobs', 0)
            active_jobs = stats.get('active_jobs', 0)
            
            # Determine health status
            if workers == 0:
                status = HealthStatus.UNHEALTHY
                message = "No workers available"
            elif queued_jobs > self.thresholds["max_queue_size"]:
                status = HealthStatus.DEGRADED
                message = f"High queue size: {queued_jobs}"
            else:
                status = HealthStatus.HEALTHY
                message = f"{workers} workers, {active_jobs} active jobs"
            
            latency_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                service="scheduler",
                status=status,
                message=message,
                latency_ms=latency_ms,
                details={
                    "workers": workers,
                    "queued_jobs": queued_jobs,
                    "active_jobs": active_jobs,
                    "gpu_workers": stats.get('gpu_workers', 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Scheduler health check failed: {e}")
            return HealthCheck(
                service="scheduler",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def check_storage(self, storage: Any) -> HealthCheck:
        """Check storage health."""
        start_time = time.time()
        
        try:
            if not storage:
                return HealthCheck(
                    service="storage",
                    status=HealthStatus.UNKNOWN,
                    message="Storage not configured"
                )
            
            # Test storage connectivity
            test_successful = False
            backend_type = getattr(storage, 'backend_type', 'unknown')
            
            # Try to list models (quick operation)
            if hasattr(storage, 'list_models'):
                models = storage.list_models()
                test_successful = True
            
            # Check disk usage for local storage
            disk_usage = None
            if backend_type == "local":
                disk_usage = psutil.disk_usage('/').percent
                if disk_usage > self.thresholds["max_disk_usage_percent"]:
                    status = HealthStatus.DEGRADED
                    message = f"High disk usage: {disk_usage:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Storage operational"
            else:
                status = HealthStatus.HEALTHY if test_successful else HealthStatus.UNHEALTHY
                message = f"{backend_type} storage {'operational' if test_successful else 'unavailable'}"
            
            latency_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                service="storage",
                status=status,
                message=message,
                latency_ms=latency_ms,
                details={
                    "backend": backend_type,
                    "disk_usage_percent": disk_usage,
                    "test_successful": test_successful
                }
            )
            
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return HealthCheck(
                service="storage",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def check_billing(self, billing_manager: Any) -> HealthCheck:
        """Check billing service health."""
        start_time = time.time()
        
        try:
            if not billing_manager:
                return HealthCheck(
                    service="billing",
                    status=HealthStatus.UNKNOWN,
                    message="Billing not configured"
                )
            
            # Check database connectivity
            summary = {}
            if hasattr(billing_manager, 'get_system_summary'):
                summary = billing_manager.get_system_summary()
            
            if "error" in summary:
                status = HealthStatus.UNHEALTHY
                message = summary["error"]
            else:
                active_subscriptions = summary.get("total_active_subscriptions", 0)
                status = HealthStatus.HEALTHY
                message = f"{active_subscriptions} active subscriptions"
            
            latency_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                service="billing",
                status=status,
                message=message,
                latency_ms=latency_ms,
                details=summary
            )
            
        except Exception as e:
            logger.error(f"Billing health check failed: {e}")
            return HealthCheck(
                service="billing",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def check_streaming(self, streaming_orchestrators: Dict) -> HealthCheck:
        """Check streaming services health."""
        start_time = time.time()
        
        try:
            if not streaming_orchestrators:
                return HealthCheck(
                    service="streaming",
                    status=HealthStatus.UNKNOWN,
                    message="No streaming pipelines active"
                )
            
            active_count = len(streaming_orchestrators)
            healthy_count = 0
            total_processed = 0
            total_failed = 0
            
            for stream_id, orchestrator in streaming_orchestrators.items():
                if hasattr(orchestrator, 'get_metrics'):
                    metrics = orchestrator.get_metrics()
                    if metrics.get("status") == "running":
                        healthy_count += 1
                    total_processed += metrics.get("messages_processed", 0)
                    total_failed += metrics.get("messages_failed", 0)
            
            # Calculate error rate
            error_rate = total_failed / (total_processed + total_failed) if (total_processed + total_failed) > 0 else 0
            
            if healthy_count == 0 and active_count > 0:
                status = HealthStatus.UNHEALTHY
                message = "No healthy streaming pipelines"
            elif error_rate > self.thresholds["max_error_rate"]:
                status = HealthStatus.DEGRADED
                message = f"High error rate: {error_rate:.2%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"{healthy_count}/{active_count} pipelines healthy"
            
            latency_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                service="streaming",
                status=status,
                message=message,
                latency_ms=latency_ms,
                details={
                    "active_pipelines": active_count,
                    "healthy_pipelines": healthy_count,
                    "total_messages_processed": total_processed,
                    "total_messages_failed": total_failed,
                    "error_rate": error_rate
                }
            )
            
        except Exception as e:
            logger.error(f"Streaming health check failed: {e}")
            return HealthCheck(
                service="streaming",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def check_system_resources(self) -> HealthCheck:
        """Check system resources (CPU, Memory, Disk)."""
        start_time = time.time()
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            issues = []
            
            if cpu_percent > self.thresholds["max_cpu_usage_percent"]:
                issues.append(f"High CPU: {cpu_percent:.1f}%")
            
            memory_available_gb = memory.available / (1024**3)
            if memory_available_gb < self.thresholds["min_memory_available_gb"]:
                issues.append(f"Low memory: {memory_available_gb:.1f}GB available")
            
            if disk.percent > self.thresholds["max_disk_usage_percent"]:
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            if issues:
                status = HealthStatus.DEGRADED
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "System resources within normal range"
            
            latency_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                service="system",
                status=status,
                message=message,
                latency_ms=latency_ms,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory_available_gb,
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                }
            )
            
        except Exception as e:
            logger.error(f"System resources health check failed: {e}")
            return HealthCheck(
                service="system",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )
    
    async def check_all(self, app_state: Any) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Args:
            app_state: FastAPI app.state containing services
        
        Returns:
            Dictionary with overall health and individual service health
        """
        results = {}
        
        # System resources
        results['system'] = await self.check_system_resources()
        
        # Scheduler
        if hasattr(app_state, 'scheduler'):
            results['scheduler'] = await self.check_scheduler(app_state.scheduler)
        
        # Storage
        if hasattr(app_state, 'storage'):
            results['storage'] = await self.check_storage(app_state.storage)
        
        # Billing
        if hasattr(app_state, 'billing_manager'):
            results['billing'] = await self.check_billing(app_state.billing_manager)
        
        # Streaming
        if hasattr(app_state, 'streaming_orchestrators'):
            results['streaming'] = await self.check_streaming(app_state.streaming_orchestrators)
        
        # Run custom checks
        for service_name, check_func in self.custom_checks.items():
            try:
                results[service_name] = await check_func()
            except Exception as e:
                logger.error(f"Custom health check for {service_name} failed: {e}")
                results[service_name] = HealthCheck(
                    service=service_name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e)
                )
        
        # Store results
        self.checks = results
        
        # Determine overall status
        statuses = [check.status for check in results.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = "healthy"
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return {
            "overall": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "latency_ms": check.latency_ms,
                    "details": check.details
                }
                for name, check in results.items()
            }
        }
    
    async def start_monitoring(self, app_state: Any):
        """
        Start continuous health monitoring.
        
        Args:
            app_state: FastAPI app.state containing services
        """
        self.is_monitoring = True
        
        async def monitor_loop():
            while self.is_monitoring:
                try:
                    await self.check_all(app_state)
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                
                await asyncio.sleep(self.check_interval)
        
        self.monitoring_task = asyncio.create_task(monitor_loop())
        logger.info("Started continuous health monitoring")
    
    async def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            await self.monitoring_task
        logger.info("Stopped health monitoring")
    
    def get_latest_status(self) -> Dict[str, Any]:
        """Get latest health check results."""
        if not self.checks:
            return {"overall": "unknown", "message": "No health checks performed yet"}
        
        return {
            "overall": self._calculate_overall_status(),
            "last_check": max(
                (check.timestamp for check in self.checks.values()),
                default=datetime.utcnow()
            ).isoformat(),
            "services": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "latency_ms": check.latency_ms
                }
                for name, check in self.checks.items()
            }
        }
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall system status."""
        if not self.checks:
            return "unknown"
        
        statuses = [check.status for check in self.checks.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return "healthy"
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return "unhealthy"
        else:
            return "degraded"
    
    def get_unhealthy_services(self) -> List[str]:
        """Get list of unhealthy services."""
        return [
            name for name, check in self.checks.items()
            if check.status == HealthStatus.UNHEALTHY
        ]
    
    def get_degraded_services(self) -> List[str]:
        """Get list of degraded services."""
        return [
            name for name, check in self.checks.items()
            if check.status == HealthStatus.DEGRADED
        ]
    
    def set_threshold(self, key: str, value: Any):
        """
        Update a health check threshold.
        
        Args:
            key: Threshold key
            value: New threshold value
        """
        self.thresholds[key] = value
        logger.info(f"Updated threshold {key} to {value}")
    
    def export_metrics_prometheus(self) -> str:
        """Export health metrics in Prometheus format."""
        lines = []
        
        # Overall health (1 = healthy, 0.5 = degraded, 0 = unhealthy)
        overall = self._calculate_overall_status()
        overall_value = {"healthy": 1, "degraded": 0.5, "unhealthy": 0}.get(overall, 0)
        lines.append(f"automl_health_overall {overall_value}")
        
        # Service health
        for name, check in self.checks.items():
            status_value = {
                HealthStatus.HEALTHY: 1,
                HealthStatus.DEGRADED: 0.5,
                HealthStatus.UNHEALTHY: 0,
                HealthStatus.UNKNOWN: -1
            }.get(check.status, -1)
            
            lines.append(f'automl_health_service{{service="{name}"}} {status_value}')
            lines.append(f'automl_health_latency_ms{{service="{name}"}} {check.latency_ms}')
            
            # Add details as metrics
            for key, value in check.details.items():
                if isinstance(value, (int, float)):
                    lines.append(f'automl_health_{key}{{service="{name}"}} {value}')
        
        return "\n".join(lines)
