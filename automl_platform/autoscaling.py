"""
Autoscaling Service - Resource Management and Job Scheduling
============================================================
Place in: automl_platform/autoscaling.py

Manages dynamic resource allocation, GPU scheduling, and job orchestration
Compatible with existing infrastructure.py, scheduler.py and config.py
"""

import os
import json
import logging
import psutil
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import heapq
from collections import defaultdict, deque
import threading
import time
import pickle

# Kubernetes for autoscaling
try:
    from kubernetes import client, config as k8s_config
    from kubernetes.client import V1ResourceRequirements
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

# GPU management
try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_AVAILABLE = True
except:
    NVIDIA_AVAILABLE = False

# Ray for distributed computing
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

import redis
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Float, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Import from existing modules
from .config import AutoMLConfig, WorkerConfig, BillingConfig, PlanType
from .scheduler import JobRequest, JobStatus, QueueType, PLAN_LIMITS
from .api.infrastructure import TenantManager, ResourceMonitor
from .api.billing import BillingManager

logger = logging.getLogger(__name__)

Base = declarative_base()

# ============================================================================
# Data Models
# ============================================================================

class ResourceType(Enum):
    """Resource types for allocation"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


class ScalingStrategy(Enum):
    """Autoscaling strategies"""
    REACTIVE = "reactive"      # Scale based on current load
    PREDICTIVE = "predictive"  # Scale based on predicted load
    SCHEDULED = "scheduled"     # Scale based on schedule
    HYBRID = "hybrid"          # Combine multiple strategies


@dataclass
class ResourceAllocation:
    """Resource allocation for a job or worker"""
    allocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default"
    job_id: Optional[str] = None
    worker_id: Optional[str] = None
    
    # Resource limits
    cpu_cores: float = 1.0
    memory_gb: float = 4.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 10.0
    
    # Timing
    allocated_at: datetime = field(default_factory=datetime.utcnow)
    released_at: Optional[datetime] = None
    ttl_seconds: Optional[int] = 3600
    
    # Status
    is_active: bool = True
    utilization: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClusterNode:
    """Represents a compute node in the cluster"""
    node_id: str
    node_type: str = "cpu"  # cpu, gpu, mixed
    
    # Capacity
    total_cpu_cores: int = 8
    total_memory_gb: float = 32.0
    total_gpu_count: int = 0
    total_gpu_memory_gb: float = 0.0
    
    # Current usage
    used_cpu_cores: float = 0.0
    used_memory_gb: float = 0.0
    used_gpu_count: int = 0
    used_gpu_memory_gb: float = 0.0
    
    # Status
    is_available: bool = True
    is_schedulable: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    taints: List[str] = field(default_factory=list)
    
    # Metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    
    def available_resources(self) -> Dict[str, float]:
        """Get available resources on the node"""
        return {
            "cpu_cores": self.total_cpu_cores - self.used_cpu_cores,
            "memory_gb": self.total_memory_gb - self.used_memory_gb,
            "gpu_count": self.total_gpu_count - self.used_gpu_count,
            "gpu_memory_gb": self.total_gpu_memory_gb - self.used_gpu_memory_gb
        }
    
    def can_allocate(self, allocation: ResourceAllocation) -> bool:
        """Check if node can handle the allocation"""
        available = self.available_resources()
        return (
            available["cpu_cores"] >= allocation.cpu_cores and
            available["memory_gb"] >= allocation.memory_gb and
            available["gpu_count"] >= allocation.gpu_count and
            available["gpu_memory_gb"] >= allocation.gpu_memory_gb
        )


# ============================================================================
# Resource Manager
# ============================================================================

class ResourceManager:
    """Manages cluster resources and allocations"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.nodes: Dict[str, ClusterNode] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_lock = threading.Lock()
        
        # Initialize cluster discovery
        self._discover_nodes()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def _discover_nodes(self):
        """Discover available compute nodes"""
        
        if KUBERNETES_AVAILABLE:
            self._discover_k8s_nodes()
        else:
            # Create local node
            local_node = ClusterNode(
                node_id="local",
                node_type="cpu",
                total_cpu_cores=psutil.cpu_count(),
                total_memory_gb=psutil.virtual_memory().total / (1024**3),
                total_gpu_count=self._get_gpu_count()
            )
            self.nodes["local"] = local_node
    
    def _discover_k8s_nodes(self):
        """Discover Kubernetes nodes"""
        try:
            k8s_config.load_incluster_config()
        except:
            k8s_config.load_kube_config()
        
        v1 = client.CoreV1Api()
        nodes = v1.list_node()
        
        for node in nodes.items:
            # Parse node capacity
            capacity = node.status.capacity
            
            k8s_node = ClusterNode(
                node_id=node.metadata.name,
                node_type="gpu" if "nvidia.com/gpu" in capacity else "cpu",
                total_cpu_cores=int(capacity.get("cpu", 0)),
                total_memory_gb=self._parse_memory(capacity.get("memory", "0")),
                total_gpu_count=int(capacity.get("nvidia.com/gpu", 0)),
                labels=node.metadata.labels or {},
                taints=[t.key for t in node.spec.taints] if node.spec.taints else []
            )
            
            # Check if node is schedulable
            for condition in node.status.conditions:
                if condition.type == "Ready":
                    k8s_node.is_schedulable = condition.status == "True"
            
            self.nodes[k8s_node.node_id] = k8s_node
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse Kubernetes memory string to GB"""
        if memory_str.endswith("Ki"):
            return float(memory_str[:-2]) / (1024**2)
        elif memory_str.endswith("Mi"):
            return float(memory_str[:-2]) / 1024
        elif memory_str.endswith("Gi"):
            return float(memory_str[:-2])
        else:
            return float(memory_str) / (1024**3)
    
    def _get_gpu_count(self) -> int:
        """Get number of GPUs on local machine"""
        if NVIDIA_AVAILABLE:
            return pynvml.nvmlDeviceGetCount()
        return 0
    
    def allocate_resources(self, job_request: JobRequest) -> Optional[ResourceAllocation]:
        """Allocate resources for a job"""
        
        with self.allocation_lock:
            # Create allocation request
            allocation = ResourceAllocation(
                tenant_id=job_request.tenant_id,
                job_id=job_request.job_id,
                cpu_cores=job_request.estimated_memory_gb / 4,  # Rough estimate
                memory_gb=job_request.estimated_memory_gb,
                gpu_count=job_request.num_gpus,
                gpu_memory_gb=job_request.gpu_memory_gb
            )
            
            # Find suitable node
            node = self._find_best_node(allocation)
            if not node:
                logger.warning(f"No suitable node found for job {job_request.job_id}")
                return None
            
            # Allocate on node
            node.used_cpu_cores += allocation.cpu_cores
            node.used_memory_gb += allocation.memory_gb
            node.used_gpu_count += allocation.gpu_count
            node.used_gpu_memory_gb += allocation.gpu_memory_gb
            
            # Track allocation
            self.allocations[allocation.allocation_id] = allocation
            
            logger.info(f"Allocated resources for job {job_request.job_id} on node {node.node_id}")
            
            return allocation
    
    def _find_best_node(self, allocation: ResourceAllocation) -> Optional[ClusterNode]:
        """Find best node for allocation using bin packing"""
        
        suitable_nodes = []
        
        for node in self.nodes.values():
            if node.is_schedulable and node.can_allocate(allocation):
                # Calculate fit score (lower is better)
                available = node.available_resources()
                fit_score = (
                    (available["cpu_cores"] - allocation.cpu_cores) +
                    (available["memory_gb"] - allocation.memory_gb)
                )
                suitable_nodes.append((fit_score, node))
        
        if not suitable_nodes:
            return None
        
        # Return node with best fit
        suitable_nodes.sort(key=lambda x: x[0])
        return suitable_nodes[0][1]
    
    def release_resources(self, allocation_id: str):
        """Release allocated resources"""
        
        with self.allocation_lock:
            if allocation_id not in self.allocations:
                return
            
            allocation = self.allocations[allocation_id]
            allocation.is_active = False
            allocation.released_at = datetime.utcnow()
            
            # Find node and release resources
            for node in self.nodes.values():
                # This is simplified - in production, track which node has the allocation
                if node.used_cpu_cores >= allocation.cpu_cores:
                    node.used_cpu_cores -= allocation.cpu_cores
                    node.used_memory_gb -= allocation.memory_gb
                    node.used_gpu_count -= allocation.gpu_count
                    node.used_gpu_memory_gb -= allocation.gpu_memory_gb
                    break
            
            logger.info(f"Released resources for allocation {allocation_id}")
    
    def _monitor_resources(self):
        """Monitor resource usage and clean up expired allocations"""
        
        while True:
            try:
                # Update node metrics
                for node in self.nodes.values():
                    if node.total_cpu_cores > 0:
                        node.cpu_utilization = node.used_cpu_cores / node.total_cpu_cores * 100
                    if node.total_memory_gb > 0:
                        node.memory_utilization = node.used_memory_gb / node.total_memory_gb * 100
                    if node.total_gpu_count > 0:
                        node.gpu_utilization = node.used_gpu_count / node.total_gpu_count * 100
                
                # Clean up expired allocations
                now = datetime.utcnow()
                expired = []
                
                for alloc_id, allocation in self.allocations.items():
                    if allocation.ttl_seconds and allocation.is_active:
                        age = (now - allocation.allocated_at).total_seconds()
                        if age > allocation.ttl_seconds:
                            expired.append(alloc_id)
                
                for alloc_id in expired:
                    self.release_resources(alloc_id)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
            
            time.sleep(30)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster resource status"""
        
        total_cpu = sum(n.total_cpu_cores for n in self.nodes.values())
        used_cpu = sum(n.used_cpu_cores for n in self.nodes.values())
        
        total_memory = sum(n.total_memory_gb for n in self.nodes.values())
        used_memory = sum(n.used_memory_gb for n in self.nodes.values())
        
        total_gpu = sum(n.total_gpu_count for n in self.nodes.values())
        used_gpu = sum(n.used_gpu_count for n in self.nodes.values())
        
        return {
            "nodes": len(self.nodes),
            "schedulable_nodes": sum(1 for n in self.nodes.values() if n.is_schedulable),
            "total_resources": {
                "cpu_cores": total_cpu,
                "memory_gb": total_memory,
                "gpu_count": total_gpu
            },
            "used_resources": {
                "cpu_cores": used_cpu,
                "memory_gb": used_memory,
                "gpu_count": used_gpu
            },
            "utilization": {
                "cpu_percent": (used_cpu / total_cpu * 100) if total_cpu > 0 else 0,
                "memory_percent": (used_memory / total_memory * 100) if total_memory > 0 else 0,
                "gpu_percent": (used_gpu / total_gpu * 100) if total_gpu > 0 else 0
            },
            "active_allocations": sum(1 for a in self.allocations.values() if a.is_active)
        }


# ============================================================================
# GPU Scheduler
# ============================================================================

class GPUScheduler:
    """Specialized scheduler for GPU workloads"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.gpu_queue = deque()
        self.gpu_allocations: Dict[int, ResourceAllocation] = {}
        self.gpu_metrics: Dict[int, Dict] = {}
        
        if NVIDIA_AVAILABLE:
            self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring"""
        
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        logger.info(f"Found {self.num_gpus} GPUs")
        
        # Get GPU info
        for i in range(self.num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode()
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            self.gpu_metrics[i] = {
                "name": name,
                "total_memory_gb": memory.total / (1024**3),
                "used_memory_gb": memory.used / (1024**3),
                "free_memory_gb": memory.free / (1024**3),
                "utilization": 0
            }
    
    def schedule_gpu_job(self, job_request: JobRequest) -> Optional[int]:
        """Schedule a job on GPU"""
        
        if not NVIDIA_AVAILABLE:
            logger.error("No GPUs available")
            return None
        
        # Check plan allows GPU
        plan_limits = PLAN_LIMITS.get(job_request.plan_type, {})
        if not plan_limits.get("gpu_access", False):
            logger.warning(f"Plan {job_request.plan_type} does not allow GPU access")
            return None
        
        # Find available GPU
        for gpu_id in range(self.num_gpus):
            if gpu_id not in self.gpu_allocations:
                # Allocate GPU
                allocation = ResourceAllocation(
                    tenant_id=job_request.tenant_id,
                    job_id=job_request.job_id,
                    gpu_count=1,
                    gpu_memory_gb=job_request.gpu_memory_gb or 8.0
                )
                
                self.gpu_allocations[gpu_id] = allocation
                logger.info(f"Allocated GPU {gpu_id} to job {job_request.job_id}")
                
                return gpu_id
        
        # No GPU available, queue the job
        self.gpu_queue.append(job_request)
        logger.info(f"Queued GPU job {job_request.job_id}, queue size: {len(self.gpu_queue)}")
        
        return None
    
    def release_gpu(self, gpu_id: int):
        """Release GPU allocation"""
        
        if gpu_id in self.gpu_allocations:
            allocation = self.gpu_allocations.pop(gpu_id)
            logger.info(f"Released GPU {gpu_id} from job {allocation.job_id}")
            
            # Check queue for waiting jobs
            if self.gpu_queue:
                next_job = self.gpu_queue.popleft()
                return self.schedule_gpu_job(next_job)
    
    def get_gpu_status(self) -> List[Dict]:
        """Get status of all GPUs"""
        
        if not NVIDIA_AVAILABLE:
            return []
        
        gpu_status = []
        
        for i in range(self.num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Update metrics
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            status = {
                "gpu_id": i,
                "name": self.gpu_metrics[i]["name"],
                "allocated": i in self.gpu_allocations,
                "allocation": asdict(self.gpu_allocations[i]) if i in self.gpu_allocations else None,
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "free_gb": memory.free / (1024**3)
                },
                "utilization": {
                    "gpu": utilization.gpu,
                    "memory": utilization.memory
                }
            }
            
            gpu_status.append(status)
        
        return gpu_status


# ============================================================================
# Job Scheduler (Enhanced version compatible with scheduler.py)
# ============================================================================

class JobScheduler:
    """Enhanced job scheduler with autoscaling capabilities"""
    
    def __init__(self, config: AutoMLConfig, 
                 resource_manager: ResourceManager,
                 gpu_scheduler: GPUScheduler,
                 billing_manager: Optional[BillingManager] = None):
        
        self.config = config
        self.resource_manager = resource_manager
        self.gpu_scheduler = gpu_scheduler
        self.billing_manager = billing_manager
        
        # Redis for state persistence
        self.redis_client = redis.from_url(
            config.worker.broker_url if hasattr(config.worker, 'broker_url') 
            else "redis://localhost:6379"
        )
        
        # Job tracking
        self.pending_jobs = []  # Min heap by priority
        self.running_jobs: Dict[str, JobRequest] = {}
        self.completed_jobs: Dict[str, JobRequest] = {}
        
        # Scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduling_loop, daemon=True)
        self.scheduler_thread.start()
    
    def submit_job(self, job_request: JobRequest) -> str:
        """Submit a job for scheduling"""
        
        # Check quotas using billing manager
        if self.billing_manager:
            if not self.billing_manager.check_quota(
                job_request.tenant_id,
                "concurrent_jobs",
                len(self.running_jobs)
            ):
                job_request.status = JobStatus.RATE_LIMITED
                job_request.error_message = "Concurrent job quota exceeded"
                return job_request.job_id
        
        # Add to pending queue with priority
        heapq.heappush(self.pending_jobs, (-job_request.priority, job_request))
        
        # Persist to Redis
        self._persist_job(job_request)
        
        logger.info(f"Job {job_request.job_id} submitted to scheduler")
        
        return job_request.job_id
    
    def _scheduling_loop(self):
        """Main scheduling loop"""
        
        while True:
            try:
                # Process pending jobs
                while self.pending_jobs:
                    # Peek at highest priority job
                    _, job = self.pending_jobs[0]
                    
                    # Try to allocate resources
                    if job.requires_gpu:
                        gpu_id = self.gpu_scheduler.schedule_gpu_job(job)
                        if gpu_id is None:
                            break  # No GPU available
                        job.worker_id = f"gpu_{gpu_id}"
                    
                    allocation = self.resource_manager.allocate_resources(job)
                    if not allocation:
                        break  # No resources available
                    
                    # Remove from pending and mark as running
                    heapq.heappop(self.pending_jobs)
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.utcnow()
                    self.running_jobs[job.job_id] = job
                    
                    # Execute job (would trigger actual execution)
                    self._execute_job(job, allocation)
                
                # Check for completed jobs
                self._check_completed_jobs()
                
                # Autoscaling check
                self._check_autoscaling()
                
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
            
            time.sleep(5)  # Check every 5 seconds
    
    def _execute_job(self, job: JobRequest, allocation: ResourceAllocation):
        """Execute a scheduled job"""
        
        # In production, this would trigger actual job execution
        # For now, simulate with threading
        
        def run_job():
            try:
                # Simulate job execution
                time.sleep(min(job.estimated_time_minutes * 60, 10))  # Cap at 10s for demo
                
                # Mark as completed
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                
                # Release resources
                self.resource_manager.release_resources(allocation.allocation_id)
                
                if job.requires_gpu and job.worker_id:
                    gpu_id = int(job.worker_id.split("_")[1])
                    self.gpu_scheduler.release_gpu(gpu_id)
                
                # Move to completed
                self.completed_jobs[job.job_id] = job
                del self.running_jobs[job.job_id]
                
                logger.info(f"Job {job.job_id} completed successfully")
                
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                logger.error(f"Job {job.job_id} failed: {e}")
        
        # Start job execution thread
        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()
    
    def _check_completed_jobs(self):
        """Check for jobs that have completed or timed out"""
        
        now = datetime.utcnow()
        timeout_jobs = []
        
        for job_id, job in self.running_jobs.items():
            if job.started_at:
                runtime = (now - job.started_at).total_seconds() / 60
                if runtime > job.estimated_time_minutes * 2:  # 2x estimate as timeout
                    timeout_jobs.append(job_id)
        
        for job_id in timeout_jobs:
            job = self.running_jobs[job_id]
            job.status = JobStatus.FAILED
            job.error_message = "Job timed out"
            job.completed_at = now
            
            self.completed_jobs[job_id] = job
            del self.running_jobs[job_id]
            
            logger.warning(f"Job {job_id} timed out")
    
    def _check_autoscaling(self):
        """Check if autoscaling is needed"""
        
        cluster_status = self.resource_manager.get_cluster_status()
        
        # Scale up if high utilization
        if cluster_status["utilization"]["cpu_percent"] > 80:
            self._trigger_scale_up("cpu", 1)
        
        if cluster_status["utilization"]["memory_percent"] > 80:
            self._trigger_scale_up("memory", 1)
        
        # Scale down if low utilization
        if cluster_status["utilization"]["cpu_percent"] < 20 and len(self.pending_jobs) == 0:
            self._trigger_scale_down("cpu", 1)
    
    def _trigger_scale_up(self, resource_type: str, count: int):
        """Trigger scale up of resources"""
        
        if KUBERNETES_AVAILABLE:
            # In production, would trigger K8s HPA or cluster autoscaler
            logger.info(f"Would scale up {count} {resource_type} nodes")
        else:
            logger.info(f"Autoscaling not available - would scale up {resource_type}")
    
    def _trigger_scale_down(self, resource_type: str, count: int):
        """Trigger scale down of resources"""
        
        if KUBERNETES_AVAILABLE:
            # In production, would trigger K8s HPA or cluster autoscaler
            logger.info(f"Would scale down {count} {resource_type} nodes")
        else:
            logger.info(f"Autoscaling not available - would scale down {resource_type}")
    
    def _persist_job(self, job: JobRequest):
        """Persist job to Redis"""
        key = f"autoscale_job:{job.job_id}"
        value = pickle.dumps(asdict(job))
        self.redis_client.setex(key, 86400, value)  # 24 hour TTL
    
    def get_job_status(self, job_id: str) -> Optional[JobRequest]:
        """Get job status"""
        
        # Check in memory
        if job_id in self.running_jobs:
            return self.running_jobs[job_id]
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        
        # Check pending
        for _, job in self.pending_jobs:
            if job.job_id == job_id:
                return job
        
        # Check Redis
        key = f"autoscale_job:{job_id}"
        value = self.redis_client.get(key)
        if value:
            job_dict = pickle.loads(value)
            return JobRequest(**job_dict)
        
        return None
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        
        return {
            "pending_jobs": len(self.pending_jobs),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs),
            "cluster_status": self.resource_manager.get_cluster_status(),
            "gpu_status": self.gpu_scheduler.get_gpu_status()
        }


# ============================================================================
# Autoscaling Service (Main Interface)
# ============================================================================

class AutoscalingService:
    """Main autoscaling service integrating all components"""
    
    def __init__(self, config: AutoMLConfig, 
                 tenant_manager: Optional[TenantManager] = None,
                 billing_manager: Optional[BillingManager] = None):
        
        self.config = config
        self.tenant_manager = tenant_manager
        self.billing_manager = billing_manager
        
        # Initialize components
        self.resource_manager = ResourceManager(config)
        self.gpu_scheduler = GPUScheduler(config)
        self.job_scheduler = JobScheduler(
            config, 
            self.resource_manager, 
            self.gpu_scheduler,
            billing_manager
        )
        
        logger.info("Autoscaling service initialized")
    
    def submit_job(self, job_request: JobRequest) -> str:
        """Submit a job with autoscaling support"""
        return self.job_scheduler.submit_job(job_request)
    
    def get_job_status(self, job_id: str) -> Optional[JobRequest]:
        """Get job status"""
        return self.job_scheduler.get_job_status(job_id)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "scheduler": self.job_scheduler.get_scheduler_status(),
            "cluster": self.resource_manager.get_cluster_status(),
            "gpus": self.gpu_scheduler.get_gpu_status(),
            "config": {
                "autoscaling_enabled": self.config.worker.autoscale_enabled,
                "max_workers": self.config.worker.autoscale_max_workers,
                "min_workers": self.config.worker.autoscale_min_workers
            }
        }
    
    async def scale_workers(self, target_count: int) -> bool:
        """Manually scale workers"""
        
        current = len(self.resource_manager.nodes)
        
        if target_count > current:
            # Scale up
            for i in range(target_count - current):
                # In production, would create new nodes/pods
                logger.info(f"Scaling up: adding worker {i+1}/{target_count-current}")
        
        elif target_count < current:
            # Scale down
            for i in range(current - target_count):
                # In production, would remove nodes/pods
                logger.info(f"Scaling down: removing worker {i+1}/{current-target_count}")
        
        return True
    
    async def optimize_resources(self) -> Dict[str, Any]:
        """Optimize resource allocation across the cluster"""
        
        optimizations = {
            "timestamp": datetime.utcnow().isoformat(),
            "actions": []
        }
        
        cluster_status = self.resource_manager.get_cluster_status()
        
        # Check for underutilized nodes
        for node_id, node in self.resource_manager.nodes.items():
            if node.cpu_utilization < 10 and node.memory_utilization < 10:
                optimizations["actions"].append({
                    "type": "consolidate",
                    "node": node_id,
                    "reason": "underutilized"
                })
        
        # Check for resource fragmentation
        total_free_cpu = sum(n.available_resources()["cpu_cores"] 
                            for n in self.resource_manager.nodes.values())
        total_free_memory = sum(n.available_resources()["memory_gb"] 
                               for n in self.resource_manager.nodes.values())
        
        if total_free_cpu > 8 and total_free_memory > 32:
            # Could potentially consolidate
            optimizations["actions"].append({
                "type": "defragment",
                "reason": "resource fragmentation",
                "potential_savings": {
                    "cpu_cores": total_free_cpu,
                    "memory_gb": total_free_memory
                }
            })
        
        return optimizations


# ============================================================================
# Usage Example
# ============================================================================

def main():
    """Example usage of autoscaling service"""
    
    from .config import load_config
    
    # Load configuration
    config = load_config()
    
    # Initialize services
    autoscaling = AutoscalingService(config)
    
    # Create a job request
    job = JobRequest(
        tenant_id="tenant_123",
        user_id="user_456",
        plan_type=PlanType.PRO.value,
        task_type="train",
        queue_type=QueueType.GPU_TRAINING,
        requires_gpu=True,
        num_gpus=1,
        gpu_memory_gb=8.0,
        estimated_memory_gb=16.0,
        estimated_time_minutes=30,
        priority=80
    )
    
    # Submit job
    job_id = autoscaling.submit_job(job)
    print(f"Submitted job: {job_id}")
    
    # Check status
    status = autoscaling.get_job_status(job_id)
    print(f"Job status: {status.status if status else 'Not found'}")
    
    # Get system status
    system_status = autoscaling.get_system_status()
    print(f"System status: {json.dumps(system_status, indent=2, default=str)}")


if __name__ == "__main__":
    main()
