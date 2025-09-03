"""
Job Scheduler with CPU/GPU Queue Management & Autoscaling
==========================================================
Place in: automl_platform/scheduler.py

Implements intelligent job scheduling with plan-based limits,
GPU/CPU queue separation, and autoscaling capabilities.
Inspired by DataRobot (4 workers trial) and H2O.ai Cloud patterns.
"""

import os
import json
import logging
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import redis
import psutil
import threading
from queue import PriorityQueue
import time
import pickle

# Celery imports
try:
    from celery import Celery, Task, group, chain, chord, signature
    from celery.result import AsyncResult
    from kombu import Queue, Exchange
    from celery.app.control import Control
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# Ray imports
try:
    import ray
    from ray import serve
    from ray.util.queue import Queue as RayQueue
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Internal imports
from .config import AutoMLConfig, WorkerConfig, BillingConfig
from .api.billing import BillingManager, PlanType

logger = logging.getLogger(__name__)


# ============================================================================
# Job Status and Priority
# ============================================================================

class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"


class QueueType(Enum):
    """Queue types with priority"""
    GPU_TRAINING = ("gpu_training", 100)      # Highest priority
    GPU_INFERENCE = ("gpu_inference", 90)
    CPU_PRIORITY = ("cpu_priority", 80)       # Pro/Enterprise users
    LLM = ("llm", 70)
    CPU_DEFAULT = ("cpu_default", 50)         # Free/Trial users
    BATCH = ("batch", 10)                     # Lowest priority
    
    def __init__(self, queue_name: str, priority: int):
        self.queue_name = queue_name
        self.priority = priority


@dataclass
class JobRequest:
    """Job request with metadata"""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default"
    user_id: str = ""
    plan_type: str = PlanType.FREE.value
    
    # Job details
    task_type: str = "train"  # train, predict, explain, optimize
    queue_type: QueueType = QueueType.CPU_DEFAULT
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Resource requirements
    estimated_memory_gb: float = 1.0
    estimated_time_minutes: int = 10
    requires_gpu: bool = False
    num_gpus: int = 0
    gpu_memory_gb: float = 0
    
    # Scheduling
    priority: int = 50
    created_at: datetime = field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Status
    status: JobStatus = JobStatus.PENDING
    worker_id: Optional[str] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


# ============================================================================
# Plan-based Limits Configuration
# ============================================================================

PLAN_LIMITS = {
    PlanType.FREE.value: {
        "max_concurrent_jobs": 1,
        "max_workers": 1,
        "gpu_access": False,
        "max_gpu_hours_per_month": 0,
        "queue_priority": 10,
        "max_job_duration_minutes": 30,
        "max_memory_gb": 4,
        "queues_allowed": [QueueType.CPU_DEFAULT, QueueType.BATCH],
        "api_rate_limit": 10,
        "llm_calls_per_month": 0,
        "max_api_calls_per_day": 100,
        "max_predictions_per_month": 1000
    },
    PlanType.STARTER.value: {
        "max_concurrent_jobs": 2,
        "max_workers": 4,  # DataRobot trial: 4 workers
        "gpu_access": False,
        "max_gpu_hours_per_month": 0,
        "queue_priority": 30,
        "max_job_duration_minutes": 60,
        "max_memory_gb": 8,
        "queues_allowed": [QueueType.CPU_DEFAULT, QueueType.CPU_PRIORITY, QueueType.BATCH],
        "api_rate_limit": 60,
        "llm_calls_per_month": 100,
        "max_api_calls_per_day": 1000,
        "max_predictions_per_month": 10000
    },
    PlanType.PROFESSIONAL.value: {
        "max_concurrent_jobs": 5,
        "max_workers": 8,
        "gpu_access": True,
        "max_gpu_hours_per_month": 10,
        "queue_priority": 70,
        "max_job_duration_minutes": 180,
        "max_memory_gb": 16,
        "queues_allowed": [QueueType.CPU_DEFAULT, QueueType.CPU_PRIORITY, 
                          QueueType.GPU_INFERENCE, QueueType.LLM, QueueType.BATCH],
        "api_rate_limit": 100,
        "llm_calls_per_month": 1000,
        "max_api_calls_per_day": 10000,
        "max_predictions_per_month": 100000
    },
    PlanType.ENTERPRISE.value: {
        "max_concurrent_jobs": 20,
        "max_workers": 50,
        "gpu_access": True,
        "max_gpu_hours_per_month": -1,  # Unlimited
        "queue_priority": 100,
        "max_job_duration_minutes": -1,  # Unlimited
        "max_memory_gb": 64,
        "queues_allowed": "all",  # All queues
        "api_rate_limit": 1000,
        "llm_calls_per_month": -1,  # Unlimited
        "max_api_calls_per_day": -1,  # Unlimited
        "max_predictions_per_month": -1  # Unlimited
    }
}


# ============================================================================
# Celery-based Scheduler
# ============================================================================

class CeleryScheduler:
    """Celery-based job scheduler with queue management"""
    
    def __init__(self, config: AutoMLConfig, billing_manager: Optional[BillingManager] = None):
        self.config = config
        self.billing_manager = billing_manager
        self.redis_client = redis.from_url(config.worker.broker_url)
        
        # Initialize Celery app
        self.app = Celery('automl_scheduler')
        self._configure_celery()
        
        # Job tracking
        self.active_jobs: Dict[str, JobRequest] = {}
        self.job_queue = PriorityQueue()
        
        # Worker tracking
        self.worker_stats: Dict[str, Dict] = {}
        self.gpu_workers: List[str] = []
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_workers, daemon=True)
        self.monitor_thread.start()
    
    def _configure_celery(self):
        """Configure Celery with separate queues"""
        
        # Define exchanges
        default_exchange = Exchange('automl', type='direct')
        gpu_exchange = Exchange('gpu', type='direct')
        
        # Define queues with routing
        queues = [
            Queue('cpu_default', exchange=default_exchange, routing_key='cpu.default'),
            Queue('cpu_priority', exchange=default_exchange, routing_key='cpu.priority'),
            Queue('gpu_training', exchange=gpu_exchange, routing_key='gpu.training'),
            Queue('gpu_inference', exchange=gpu_exchange, routing_key='gpu.inference'),
            Queue('llm', exchange=default_exchange, routing_key='llm.process'),
            Queue('batch', exchange=default_exchange, routing_key='batch.process'),
        ]
        
        # Configure Celery
        self.app.conf.update(
            broker_url=self.config.worker.broker_url,
            result_backend=self.config.worker.result_backend,
            task_serializer='pickle',
            accept_content=['pickle', 'json'],
            result_serializer='pickle',
            timezone='UTC',
            enable_utc=True,
            task_queues=queues,
            task_routes=self._get_task_routes(),
            task_time_limit=self.config.worker.task_time_limit,
            task_soft_time_limit=self.config.worker.task_soft_time_limit,
            worker_prefetch_multiplier=self.config.worker.worker_prefetch_multiplier,
            worker_max_tasks_per_child=100,
            task_acks_late=True,
            task_reject_on_worker_lost=True,
        )
    
    def _get_task_routes(self) -> Dict:
        """Get task routing configuration"""
        return {
            'automl.tasks.train_model': {'queue': 'cpu_priority'},
            'automl.tasks.train_gpu_model': {'queue': 'gpu_training'},
            'automl.tasks.predict': {'queue': 'cpu_default'},
            'automl.tasks.predict_gpu': {'queue': 'gpu_inference'},
            'automl.tasks.process_llm': {'queue': 'llm'},
            'automl.tasks.batch_process': {'queue': 'batch'},
        }
    
    def submit_job(self, job_request: JobRequest) -> str:
        """Submit a job to the appropriate queue"""
        
        # Check quotas
        if not self._check_quotas(job_request):
            job_request.status = JobStatus.RATE_LIMITED
            job_request.error_message = "Quota exceeded for plan"
            return job_request.job_id
        
        # Check if queue is allowed for plan
        plan_limits = PLAN_LIMITS.get(job_request.plan_type, PLAN_LIMITS[PlanType.FREE.value])
        allowed_queues = plan_limits.get("queues_allowed", [])
        
        if allowed_queues != "all" and job_request.queue_type not in allowed_queues:
            # Downgrade to default queue
            job_request.queue_type = QueueType.CPU_DEFAULT
            logger.warning(f"Queue type not allowed for plan {job_request.plan_type}, using default")
        
        # Add priority based on plan
        job_request.priority = plan_limits.get("queue_priority", 10)
        
        # Track job
        self.active_jobs[job_request.job_id] = job_request
        
        # Route to appropriate Celery task
        task_name = self._get_task_name(job_request)
        
        # Prepare task signature
        task = self.app.signature(
            task_name,
            args=[job_request.payload],
            kwargs={
                'job_id': job_request.job_id,
                'tenant_id': job_request.tenant_id,
                'user_id': job_request.user_id,
            },
            queue=job_request.queue_type.queue_name,
            priority=job_request.priority,
            time_limit=plan_limits.get("max_job_duration_minutes", 30) * 60,
        )
        
        # Submit task
        result = task.apply_async()
        job_request.status = JobStatus.QUEUED
        job_request.scheduled_at = datetime.utcnow()
        
        # Store in Redis for persistence
        self._persist_job(job_request)
        
        logger.info(f"Job {job_request.job_id} submitted to queue {job_request.queue_type.queue_name}")
        
        return job_request.job_id
    
    def _check_quotas(self, job_request: JobRequest) -> bool:
        """Check if job can be submitted based on quotas"""
        
        if not self.billing_manager:
            return True
        
        # Check concurrent jobs limit
        tenant_jobs = [j for j in self.active_jobs.values() 
                      if j.tenant_id == job_request.tenant_id and 
                      j.status in [JobStatus.RUNNING, JobStatus.QUEUED]]
        
        plan_limits = PLAN_LIMITS.get(job_request.plan_type, PLAN_LIMITS[PlanType.FREE.value])
        max_concurrent = plan_limits.get("max_concurrent_jobs", 1)
        
        if len(tenant_jobs) >= max_concurrent:
            logger.warning(f"Tenant {job_request.tenant_id} exceeded concurrent job limit")
            return False
        
        # Check GPU quota
        if job_request.requires_gpu:
            if not plan_limits.get("gpu_access", False):
                logger.warning(f"GPU access not allowed for plan {job_request.plan_type}")
                return False
            
            # Check GPU hours
            gpu_hours = plan_limits.get("max_gpu_hours_per_month", 0)
            if gpu_hours != -1:  # Not unlimited
                used_hours = self._get_gpu_usage_hours(job_request.tenant_id)
                estimated_hours = job_request.estimated_time_minutes / 60
                if used_hours + estimated_hours > gpu_hours:
                    logger.warning(f"Tenant {job_request.tenant_id} would exceed GPU hours limit")
                    return False
        
        return True
    
    def _get_gpu_usage_hours(self, tenant_id: str) -> float:
        """Get GPU usage hours for current month"""
        if not self.billing_manager:
            return 0.0
        
        usage = self.billing_manager.usage_tracker.get_usage(tenant_id, "gpu_hours")
        month_key = f"{tenant_id}:gpu_hours:{datetime.now().strftime('%Y-%m')}"
        return usage.get(month_key, 0.0)
    
    def _get_task_name(self, job_request: JobRequest) -> str:
        """Get Celery task name based on job type"""
        task_mapping = {
            ("train", False): "automl.tasks.train_model",
            ("train", True): "automl.tasks.train_gpu_model",
            ("predict", False): "automl.tasks.predict",
            ("predict", True): "automl.tasks.predict_gpu",
            ("llm", False): "automl.tasks.process_llm",
            ("batch", False): "automl.tasks.batch_process",
        }
        
        key = (job_request.task_type, job_request.requires_gpu)
        return task_mapping.get(key, "automl.tasks.default")
    
    def _persist_job(self, job_request: JobRequest):
        """Persist job to Redis"""
        key = f"job:{job_request.job_id}"
        value = pickle.dumps(asdict(job_request))
        self.redis_client.setex(key, 86400, value)  # 24 hour TTL
    
    def get_job_status(self, job_id: str) -> Optional[JobRequest]:
        """Get job status"""
        
        # Check in-memory first
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Check Redis
        key = f"job:{job_id}"
        value = self.redis_client.get(key)
        if value:
            job_dict = pickle.loads(value)
            return JobRequest(**job_dict)
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        
        job = self.get_job_status(job_id)
        if not job:
            return False
        
        # Revoke Celery task
        self.app.control.revoke(job_id, terminate=True)
        
        # Update status
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        
        # Update persistence
        self._persist_job(job)
        
        # Remove from active jobs
        if job_id in self.active_jobs:
            del self.active_jobs[job_id]
        
        logger.info(f"Job {job_id} cancelled")
        return True
    
    def _monitor_workers(self):
        """Monitor worker status and health"""
        
        while True:
            try:
                # Get worker stats
                inspect = self.app.control.inspect()
                stats = inspect.stats()
                
                if stats:
                    self.worker_stats = stats
                    
                    # Identify GPU workers
                    self.gpu_workers = []
                    for worker_name, worker_info in stats.items():
                        if worker_info.get('gpu_available', False):
                            self.gpu_workers.append(worker_name)
                
                # Check for stalled jobs
                self._check_stalled_jobs()
                
                # Autoscaling check
                if self.config.worker.autoscale_enabled:
                    self._check_autoscaling()
                
            except Exception as e:
                logger.error(f"Error in worker monitoring: {e}")
            
            time.sleep(30)  # Check every 30 seconds
    
    def _check_stalled_jobs(self):
        """Check for stalled jobs and retry if needed"""
        
        timeout_threshold = timedelta(hours=1)
        now = datetime.utcnow()
        
        for job_id, job in list(self.active_jobs.items()):
            if job.status == JobStatus.RUNNING:
                if job.started_at and (now - job.started_at) > timeout_threshold:
                    logger.warning(f"Job {job_id} appears stalled, checking...")
                    
                    # Check if task is actually running
                    result = AsyncResult(job_id, app=self.app)
                    if result.state == 'FAILURE' or result.state == 'REVOKED':
                        job.status = JobStatus.FAILED
                        job.error_message = "Job timed out or failed"
                        
                        # Retry if under limit
                        if job.retry_count < job.max_retries:
                            job.retry_count += 1
                            job.status = JobStatus.PENDING
                            self.submit_job(job)
                            logger.info(f"Retrying job {job_id} (attempt {job.retry_count})")
    
    def _check_autoscaling(self):
        """Check if autoscaling is needed"""
        
        # Get queue sizes
        inspect = self.app.control.inspect()
        reserved = inspect.reserved()
        
        if not reserved:
            return
        
        total_queued = sum(len(tasks) for tasks in reserved.values())
        active_workers = len(self.worker_stats)
        
        # Scale up if queue is large
        if total_queued > active_workers * 5:  # More than 5 tasks per worker
            self._scale_workers(1)
        
        # Scale down if queue is empty
        elif total_queued == 0 and active_workers > self.config.worker.autoscale_min_workers:
            self._scale_workers(-1)
    
    def _scale_workers(self, delta: int):
        """Scale workers up or down"""
        
        current_workers = len(self.worker_stats)
        target_workers = current_workers + delta
        
        # Apply limits
        target_workers = max(self.config.worker.autoscale_min_workers, target_workers)
        target_workers = min(self.config.worker.autoscale_max_workers, target_workers)
        
        if target_workers != current_workers:
            logger.info(f"Scaling workers from {current_workers} to {target_workers}")
            
            # In production, this would trigger Kubernetes scaling or cloud autoscaling
            # For now, just log the intent
            if delta > 0:
                # Scale up
                self._start_workers(delta)
            else:
                # Scale down
                self._stop_workers(-delta)
    
    def _start_workers(self, count: int):
        """Start additional workers (placeholder for actual implementation)"""
        # In production, this would:
        # 1. Start new Celery worker processes
        # 2. Or trigger Kubernetes to scale the worker deployment
        # 3. Or start new cloud instances
        logger.info(f"Would start {count} new workers")
    
    def _stop_workers(self, count: int):
        """Stop workers (placeholder for actual implementation)"""
        # In production, this would:
        # 1. Gracefully shutdown Celery workers
        # 2. Or scale down Kubernetes deployment
        # 3. Or terminate cloud instances
        logger.info(f"Would stop {count} workers")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        
        inspect = self.app.control.inspect()
        
        # Get various stats
        stats = {
            'workers': len(self.worker_stats),
            'gpu_workers': len(self.gpu_workers),
            'active_jobs': len([j for j in self.active_jobs.values() 
                              if j.status == JobStatus.RUNNING]),
            'queued_jobs': len([j for j in self.active_jobs.values() 
                              if j.status == JobStatus.QUEUED]),
            'reserved_tasks': {},
            'active_queues': inspect.active_queues() or {},
        }
        
        # Get reserved tasks per queue
        reserved = inspect.reserved()
        if reserved:
            for worker, tasks in reserved.items():
                for task in tasks:
                    queue = task.get('delivery_info', {}).get('routing_key', 'unknown')
                    if queue not in stats['reserved_tasks']:
                        stats['reserved_tasks'][queue] = 0
                    stats['reserved_tasks'][queue] += 1
        
        return stats


# ============================================================================
# Ray-based Scheduler (Alternative Implementation)
# ============================================================================

class RayScheduler:
    """Ray-based job scheduler for distributed computing"""
    
    def __init__(self, config: AutoMLConfig, billing_manager: Optional[BillingManager] = None):
        self.config = config
        self.billing_manager = billing_manager
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                num_cpus=psutil.cpu_count(),
                num_gpus=self._detect_gpus(),
                dashboard_host="0.0.0.0",
                dashboard_port=8265,
            )
        
        # Job tracking
        self.active_jobs: Dict[str, JobRequest] = {}
        self.job_futures: Dict[str, ray.ObjectRef] = {}
    
    def _detect_gpus(self) -> int:
        """Detect number of available GPUs"""
        try:
            import torch
            return torch.cuda.device_count()
        except:
            return 0
    
    def submit_job(self, job_request: JobRequest) -> str:
        """Submit a job using Ray"""
        
        # Check quotas
        if not self._check_quotas(job_request):
            job_request.status = JobStatus.RATE_LIMITED
            return job_request.job_id
        
        # Track job
        self.active_jobs[job_request.job_id] = job_request
        
        # Create Ray task with resource requirements
        if job_request.requires_gpu:
            # GPU task
            task = ray.remote(num_gpus=job_request.num_gpus)(self._execute_task)
        else:
            # CPU task
            task = ray.remote(num_cpus=1)(self._execute_task)
        
        # Submit task
        future = task.remote(job_request)
        self.job_futures[job_request.job_id] = future
        
        job_request.status = JobStatus.QUEUED
        job_request.scheduled_at = datetime.utcnow()
        
        logger.info(f"Job {job_request.job_id} submitted to Ray")
        
        return job_request.job_id
    
    def _execute_task(self, job_request: JobRequest):
        """Execute task in Ray worker"""
        
        try:
            job_request.status = JobStatus.RUNNING
            job_request.started_at = datetime.utcnow()
            
            # Execute based on task type
            if job_request.task_type == "train":
                from .orchestrator import AutoMLOrchestrator
                orchestrator = AutoMLOrchestrator(self.config)
                result = orchestrator.fit(**job_request.payload)
            else:
                result = {"status": "completed"}
            
            job_request.status = JobStatus.COMPLETED
            job_request.completed_at = datetime.utcnow()
            job_request.result = result
            
            return job_request
            
        except Exception as e:
            job_request.status = JobStatus.FAILED
            job_request.error_message = str(e)
            return job_request
    
    def _check_quotas(self, job_request: JobRequest) -> bool:
        """Check quotas (same as Celery implementation)"""
        # Implementation same as CeleryScheduler
        return True
    
    def get_job_status(self, job_id: str) -> Optional[JobRequest]:
        """Get job status"""
        
        if job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[job_id]
        
        # Check Ray future if exists
        if job_id in self.job_futures:
            future = self.job_futures[job_id]
            if ray.get(future):
                updated_job = ray.get(future)
                self.active_jobs[job_id] = updated_job
                return updated_job
        
        return job
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a Ray job"""
        
        if job_id in self.job_futures:
            ray.cancel(self.job_futures[job_id])
            
            if job_id in self.active_jobs:
                self.active_jobs[job_id].status = JobStatus.CANCELLED
            
            return True
        
        return False


# ============================================================================
# Scheduler Factory
# ============================================================================

class SchedulerFactory:
    """Factory to create appropriate scheduler based on configuration"""
    
    @staticmethod
    def create_scheduler(config: AutoMLConfig, 
                        billing_manager: Optional[BillingManager] = None) -> Union[CeleryScheduler, RayScheduler]:
        """Create scheduler based on backend configuration"""
        
        backend = config.worker.backend.lower()
        
        if backend == "celery" and CELERY_AVAILABLE:
            logger.info("Using Celery scheduler")
            return CeleryScheduler(config, billing_manager)
        
        elif backend == "ray" and RAY_AVAILABLE:
            logger.info("Using Ray scheduler")
            return RayScheduler(config, billing_manager)
        
        else:
            # Fallback to simple local scheduler
            logger.warning(f"Backend {backend} not available, using local scheduler")
            return LocalScheduler(config, billing_manager)


class LocalScheduler:
    """Simple local scheduler for development/testing"""
    
    def __init__(self, config: AutoMLConfig, billing_manager: Optional[BillingManager] = None):
        self.config = config
        self.billing_manager = billing_manager
        self.executor = ThreadPoolExecutor(max_workers=config.worker.max_workers)
        self.active_jobs: Dict[str, JobRequest] = {}
    
    def submit_job(self, job_request: JobRequest) -> str:
        """Submit job to thread pool"""
        
        self.active_jobs[job_request.job_id] = job_request
        job_request.status = JobStatus.QUEUED
        
        # Submit to executor
        future = self.executor.submit(self._execute_job, job_request)
        
        return job_request.job_id
    
    def _execute_job(self, job_request: JobRequest):
        """Execute job locally"""
        
        try:
            job_request.status = JobStatus.RUNNING
            job_request.started_at = datetime.utcnow()
            
            # Simulate work
            time.sleep(1)
            
            job_request.status = JobStatus.COMPLETED
            job_request.completed_at = datetime.utcnow()
            
        except Exception as e:
            job_request.status = JobStatus.FAILED
            job_request.error_message = str(e)
    
    def get_job_status(self, job_id: str) -> Optional[JobRequest]:
        """Get job status"""
        return self.active_jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel job"""
        if job_id in self.active_jobs:
            self.active_jobs[job_id].status = JobStatus.CANCELLED
            return True
        return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            'workers': self.executor._max_workers,
            'active_jobs': len([j for j in self.active_jobs.values() 
                              if j.status == JobStatus.RUNNING]),
            'queued_jobs': len([j for j in self.active_jobs.values() 
                              if j.status == JobStatus.QUEUED])
        }


# ============================================================================
# Celery Tasks Definition
# ============================================================================

if CELERY_AVAILABLE:
    app = Celery('automl_tasks')
    
    @app.task(name='automl.tasks.train_model')
    def train_model_task(payload: Dict, job_id: str, tenant_id: str, user_id: str):
        """Celery task for model training"""
        from .orchestrator import AutoMLOrchestrator
        from .config import load_config
        
        config = load_config()
        config.tenant_id = tenant_id
        config.user_id = user_id
        
        orchestrator = AutoMLOrchestrator(config)
        result = orchestrator.fit(**payload)
        
        return {"job_id": job_id, "status": "completed", "result": result}
    
    @app.task(name='automl.tasks.train_gpu_model')
    def train_gpu_model_task(payload: Dict, job_id: str, tenant_id: str, user_id: str):
        """Celery task for GPU model training"""
        # Set CUDA device
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        return train_model_task(payload, job_id, tenant_id, user_id)
    
    @app.task(name='automl.tasks.predict')
    def predict_task(payload: Dict, job_id: str, tenant_id: str, user_id: str):
        """Celery task for predictions"""
        from .orchestrator import AutoMLOrchestrator
        
        model_path = payload.get('model_path')
        data = payload.get('data')
        
        orchestrator = AutoMLOrchestrator(None)
        orchestrator.load_pipeline(model_path)
        predictions = orchestrator.predict(data)
        
        return {"job_id": job_id, "predictions": predictions.tolist()}


# ============================================================================
# Usage Example
# ============================================================================

def main():
    """Example usage of the scheduler"""
    
    from .config import load_config
    from .api.billing import BillingManager
    
    # Load configuration
    config = load_config()
    
    # Initialize billing manager
    billing_manager = BillingManager()
    
    # Create scheduler
    scheduler = SchedulerFactory.create_scheduler(config, billing_manager)
    
    # Create a job request
    job = JobRequest(
        tenant_id="tenant_123",
        user_id="user_456",
        plan_type=PlanType.PROFESSIONAL.value,
        task_type="train",
        queue_type=QueueType.CPU_PRIORITY,
        payload={
            "X": "data",
            "y": "labels",
            "task": "classification"
        },
        estimated_memory_gb=2.0,
        estimated_time_minutes=30,
        requires_gpu=False
    )
    
    # Submit job
    job_id = scheduler.submit_job(job)
    print(f"Submitted job: {job_id}")
    
    # Check status
    status = scheduler.get_job_status(job_id)
    print(f"Job status: {status.status if status else 'Not found'}")
    
    # Get queue stats
    stats = scheduler.get_queue_stats() if hasattr(scheduler, 'get_queue_stats') else {}
    print(f"Queue stats: {stats}")


if __name__ == "__main__":
    main()
