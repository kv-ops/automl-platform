"""
Test Suite for Job Scheduler Module
====================================
Tests for job scheduling with CPU/GPU queue management, quotas, and autoscaling.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
from datetime import datetime, timedelta
import pickle
import uuid
from typing import Dict, Any
from enum import Enum

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.scheduler import (
    JobStatus,
    QueueType,
    JobRequest,
    PLAN_LIMITS,
    CeleryScheduler,
    RayScheduler,
    LocalScheduler,
    SchedulerFactory,
    PlanType
)
from automl_platform.config import AutoMLConfig, WorkerConfig


class TestJobStatus(unittest.TestCase):
    """Test job status enum."""
    
    def test_status_values(self):
        """Test job status enum values."""
        self.assertEqual(JobStatus.PENDING.value, "pending")
        self.assertEqual(JobStatus.QUEUED.value, "queued")
        self.assertEqual(JobStatus.RUNNING.value, "running")
        self.assertEqual(JobStatus.COMPLETED.value, "completed")
        self.assertEqual(JobStatus.FAILED.value, "failed")
        self.assertEqual(JobStatus.CANCELLED.value, "cancelled")
        self.assertEqual(JobStatus.RATE_LIMITED.value, "rate_limited")


class TestQueueType(unittest.TestCase):
    """Test queue type enum with priorities."""
    
    def test_queue_priorities(self):
        """Test queue type priorities."""
        self.assertEqual(QueueType.GPU_TRAINING.priority, 100)
        self.assertEqual(QueueType.GPU_INFERENCE.priority, 90)
        self.assertEqual(QueueType.CPU_PRIORITY.priority, 80)
        self.assertEqual(QueueType.LLM.priority, 70)
        self.assertEqual(QueueType.CPU_DEFAULT.priority, 50)
        self.assertEqual(QueueType.BATCH.priority, 10)
    
    def test_queue_names(self):
        """Test queue names."""
        self.assertEqual(QueueType.GPU_TRAINING.queue_name, "gpu_training")
        self.assertEqual(QueueType.CPU_DEFAULT.queue_name, "cpu_default")


class TestJobRequest(unittest.TestCase):
    """Test job request dataclass."""
    
    def test_default_job_request(self):
        """Test default job request creation."""
        job = JobRequest()
        
        self.assertIsNotNone(job.job_id)
        self.assertEqual(job.tenant_id, "default")
        self.assertEqual(job.plan_type, PlanType.FREE.value)
        self.assertEqual(job.task_type, "train")
        self.assertEqual(job.queue_type, QueueType.CPU_DEFAULT)
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertEqual(job.retry_count, 0)
        self.assertEqual(job.max_retries, 3)
    
    def test_custom_job_request(self):
        """Test custom job request."""
        job = JobRequest(
            tenant_id="company_123",
            user_id="user_456",
            plan_type=PlanType.PRO.value,
            task_type="predict",
            queue_type=QueueType.GPU_INFERENCE,
            requires_gpu=True,
            num_gpus=2,
            estimated_memory_gb=16.0
        )
        
        self.assertEqual(job.tenant_id, "company_123")
        self.assertEqual(job.plan_type, PlanType.PRO.value)
        self.assertEqual(job.queue_type, QueueType.GPU_INFERENCE)
        self.assertTrue(job.requires_gpu)
        self.assertEqual(job.num_gpus, 2)
        self.assertEqual(job.estimated_memory_gb, 16.0)


class TestPlanLimits(unittest.TestCase):
    """Test plan-based limits configuration."""
    
    def test_free_plan_limits(self):
        """Test free plan limits."""
        limits = PLAN_LIMITS[PlanType.FREE.value]
        
        self.assertEqual(limits["max_concurrent_jobs"], 1)
        self.assertEqual(limits["max_workers"], 1)
        self.assertFalse(limits["gpu_access"])
        self.assertEqual(limits["max_gpu_hours_per_month"], 0)
        self.assertEqual(limits["max_job_duration_minutes"], 30)
    
    def test_pro_plan_limits(self):
        """Test pro plan limits."""
        limits = PLAN_LIMITS[PlanType.PRO.value]
        
        self.assertEqual(limits["max_concurrent_jobs"], 5)
        self.assertEqual(limits["max_workers"], 8)
        self.assertTrue(limits["gpu_access"])
        self.assertEqual(limits["max_gpu_hours_per_month"], 10)
        self.assertIn(QueueType.GPU_INFERENCE, limits["queues_allowed"])
    
    def test_enterprise_plan_limits(self):
        """Test enterprise plan limits."""
        limits = PLAN_LIMITS[PlanType.ENTERPRISE.value]
        
        self.assertEqual(limits["max_concurrent_jobs"], 20)
        self.assertEqual(limits["max_workers"], 50)
        self.assertEqual(limits["max_gpu_hours_per_month"], -1)  # Unlimited
        self.assertEqual(limits["queues_allowed"], "all")


class TestCeleryScheduler(unittest.TestCase):
    """Test Celery-based scheduler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AutoMLConfig()
        self.config.worker = WorkerConfig(
            backend="celery",
            broker_url="redis://localhost:6379/0",
            result_backend="redis://localhost:6379/0"
        )
        
        # Mock billing manager
        self.billing_manager = Mock()
        self.billing_manager.usage_tracker = Mock()
    
    @patch('automl_platform.scheduler.redis')
    @patch('automl_platform.scheduler.Celery')
    @patch('automl_platform.scheduler.threading.Thread')
    def test_initialization(self, mock_thread, mock_celery, mock_redis):
        """Test Celery scheduler initialization."""
        scheduler = CeleryScheduler(self.config, self.billing_manager)
        
        self.assertEqual(scheduler.config, self.config)
        self.assertEqual(scheduler.billing_manager, self.billing_manager)
        mock_celery.assert_called_once_with('automl_scheduler')
        mock_thread.assert_called_once()  # Monitor thread
    
    @patch('automl_platform.scheduler.redis')
    @patch('automl_platform.scheduler.Celery')
    @patch('automl_platform.scheduler.threading.Thread')
    def test_submit_job_success(self, mock_thread, mock_celery, mock_redis):
        """Test successful job submission."""
        scheduler = CeleryScheduler(self.config, self.billing_manager)
        
        # Mock Celery app
        mock_app = Mock()
        mock_task = Mock()
        mock_app.signature.return_value = mock_task
        mock_task.apply_async.return_value = Mock(id="celery_task_id")
        scheduler.app = mock_app
        
        # Mock quota check
        scheduler._check_quotas = Mock(return_value=True)
        
        # Create job request
        job = JobRequest(
            tenant_id="tenant_123",
            plan_type=PlanType.PRO.value,
            task_type="train",
            queue_type=QueueType.CPU_PRIORITY
        )
        
        # Submit job
        job_id = scheduler.submit_job(job)
        
        self.assertEqual(job_id, job.job_id)
        self.assertEqual(job.status, JobStatus.QUEUED)
        self.assertIsNotNone(job.scheduled_at)
        mock_app.signature.assert_called_once()
        mock_task.apply_async.assert_called_once()
    
    @patch('automl_platform.scheduler.redis')
    @patch('automl_platform.scheduler.Celery')
    @patch('automl_platform.scheduler.threading.Thread')
    def test_submit_job_rate_limited(self, mock_thread, mock_celery, mock_redis):
        """Test job submission when rate limited."""
        scheduler = CeleryScheduler(self.config, self.billing_manager)
        
        # Mock quota check to fail
        scheduler._check_quotas = Mock(return_value=False)
        
        job = JobRequest(
            tenant_id="tenant_123",
            plan_type=PlanType.FREE.value
        )
        
        job_id = scheduler.submit_job(job)
        
        self.assertEqual(job.status, JobStatus.RATE_LIMITED)
        self.assertEqual(job.error_message, "Quota exceeded for plan")
    
    @patch('automl_platform.scheduler.redis')
    @patch('automl_platform.scheduler.Celery')
    @patch('automl_platform.scheduler.threading.Thread')
    def test_check_quotas_concurrent_limit(self, mock_thread, mock_celery, mock_redis):
        """Test quota check for concurrent job limit."""
        scheduler = CeleryScheduler(self.config, self.billing_manager)
        
        # Add active jobs for tenant
        for i in range(2):
            job = JobRequest(
                job_id=f"job_{i}",
                tenant_id="tenant_123",
                status=JobStatus.RUNNING
            )
            scheduler.active_jobs[job.job_id] = job
        
        # Try to submit another job (Free plan allows only 1 concurrent)
        new_job = JobRequest(
            tenant_id="tenant_123",
            plan_type=PlanType.FREE.value
        )
        
        result = scheduler._check_quotas(new_job)
        self.assertFalse(result)
    
    @patch('automl_platform.scheduler.redis')
    @patch('automl_platform.scheduler.Celery')
    @patch('automl_platform.scheduler.threading.Thread')
    def test_check_quotas_gpu_access(self, mock_thread, mock_celery, mock_redis):
        """Test quota check for GPU access."""
        scheduler = CeleryScheduler(self.config, self.billing_manager)
        
        # Free plan trying to use GPU
        job = JobRequest(
            tenant_id="tenant_123",
            plan_type=PlanType.FREE.value,
            requires_gpu=True
        )
        
        result = scheduler._check_quotas(job)
        self.assertFalse(result)
        
        # Pro plan using GPU
        job.plan_type = PlanType.PRO.value
        result = scheduler._check_quotas(job)
        self.assertTrue(result)
    
    @patch('automl_platform.scheduler.redis')
    @patch('automl_platform.scheduler.Celery')
    @patch('automl_platform.scheduler.threading.Thread')
    def test_get_gpu_usage_hours(self, mock_thread, mock_celery, mock_redis):
        """Test getting GPU usage hours."""
        scheduler = CeleryScheduler(self.config, self.billing_manager)
        
        # Mock billing manager response
        usage = {f"tenant_123:gpu_hours:2024-01": 5.5}
        scheduler.billing_manager.usage_tracker.get_usage.return_value = usage
        
        hours = scheduler._get_gpu_usage_hours("tenant_123")
        self.assertEqual(hours, 5.5)
    
    @patch('automl_platform.scheduler.redis')
    @patch('automl_platform.scheduler.Celery')
    @patch('automl_platform.scheduler.threading.Thread')
    def test_get_task_name(self, mock_thread, mock_celery, mock_redis):
        """Test task name mapping."""
        scheduler = CeleryScheduler(self.config, self.billing_manager)
        
        # Test different task types
        job = JobRequest(task_type="train", requires_gpu=False)
        self.assertEqual(scheduler._get_task_name(job), "automl.tasks.train_model")
        
        job = JobRequest(task_type="train", requires_gpu=True)
        self.assertEqual(scheduler._get_task_name(job), "automl.tasks.train_gpu_model")
        
        job = JobRequest(task_type="predict", requires_gpu=False)
        self.assertEqual(scheduler._get_task_name(job), "automl.tasks.predict")
        
        job = JobRequest(task_type="llm", requires_gpu=False)
        self.assertEqual(scheduler._get_task_name(job), "automl.tasks.process_llm")
    
    @patch('automl_platform.scheduler.redis')
    @patch('automl_platform.scheduler.Celery')
    @patch('automl_platform.scheduler.threading.Thread')
    def test_cancel_job(self, mock_thread, mock_celery, mock_redis):
        """Test job cancellation."""
        scheduler = CeleryScheduler(self.config, self.billing_manager)
