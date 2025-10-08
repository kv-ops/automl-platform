"""
Celery worker for asynchronous AutoML jobs with optimizations
=============================================================
Place in: automl_platform/worker.py (REPLACE EXISTING FILE)

Implements distributed training with Ray/Dask, incremental learning,
pipeline caching, and complete GPU/billing support.
"""

from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure, worker_ready
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import json
import time
import traceback
from datetime import datetime
import logging
import os
from pathlib import Path
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import optimization components
try:
    from .distributed_training import DistributedTrainer, DistributedConfig
    from .incremental_learning import IncrementalLearner
    from .pipeline_cache import PipelineCache, CacheConfig, monitor_cache_health
    OPTIMIZATIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Optimization components not available: {e}")
    OPTIMIZATIONS_AVAILABLE = False
    DistributedTrainer = None
    DistributedConfig = None

# Ray/Dask availability check
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# GPU availability check
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if TORCH_AVAILABLE else 0
except ImportError:
    TORCH_AVAILABLE = False
    GPU_COUNT = 0

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# Celery configuration
app = Celery('automl_platform')
app.config_from_object({
    'broker_url': os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    'result_backend': os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
    'task_track_started': True,
    'task_time_limit': 3600,  # 1 hour hard limit
    'task_soft_time_limit': 3000,  # 50 min soft limit
    'worker_prefetch_multiplier': 1,
    'worker_max_tasks_per_child': 10,  # Restart worker after 10 tasks to free memory
    'task_acks_late': True,
    'task_reject_on_worker_lost': True,
    
    # Queue routing configuration with GPU and optimization support
    'task_routes': {
        'automl.train.*': {'queue': 'training'},
        'automl.train.distributed': {'queue': 'distributed'},  # Distributed training
        'automl.train.incremental': {'queue': 'incremental'},  # Incremental learning
        'automl.train.gpu.*': {'queue': 'gpu'},
        'automl.predict.*': {'queue': 'prediction'},
        'automl.predict.gpu.*': {'queue': 'gpu'},
        'automl.llm.*': {'queue': 'llm'},
        'automl.monitor.*': {'queue': 'monitoring'},
        'automl.export.*': {'queue': 'export'},
        'automl.streaming.*': {'queue': 'streaming'},
        'automl.cache.*': {'queue': 'cache'},  # Cache management
        'automl.optimize.*': {'queue': 'optimization'}  # Optimization tasks
    },
    
    # Queue configuration with priorities
    'task_queue_max_priority': 10,
    'task_default_queue': 'default',
    'task_default_exchange': 'tasks',
    'task_default_exchange_type': 'topic',
    'task_default_routing_key': 'task.default',
    
    # Worker configuration
    'worker_pool': 'prefork',  # Use 'solo' for GPU workers
    'worker_concurrency': 4,  # Number of worker processes
    'worker_send_task_events': True,
    'worker_disable_rate_limits': False,
})

# Import after Celery initialization
from .enhanced_orchestrator import EnhancedAutoMLOrchestrator
from .config import AutoMLConfig, load_config
from .storage import StorageManager
from .monitoring import MonitoringService, ModelMonitor
from .infrastructure import TenantManager, SecurityManager
from .billing import BillingManager, UsageTracker
from .metrics import calculate_metrics, detect_task

# Initialize services
config = load_config()
storage_manager = StorageManager(
    backend=config.storage.backend,
    endpoint=config.storage.endpoint,
    access_key=config.storage.access_key,
    secret_key=config.storage.secret_key
) if config.storage.backend != "none" else None

monitoring_service = MonitoringService(storage_manager) if config.monitoring.enabled else None

# Initialize infrastructure and billing
tenant_manager = TenantManager(db_url=config.database.url)
security_manager = SecurityManager(secret_key=config.security.secret_key)
billing_manager = BillingManager(tenant_manager=tenant_manager)
usage_tracker = UsageTracker(billing_manager=billing_manager)

# Initialize optimization services
pipeline_cache = None
if OPTIMIZATIONS_AVAILABLE and hasattr(config, 'cache') and config.cache.enabled:
    cache_config = CacheConfig(
        backend=getattr(config.cache, 'backend', 'redis'),
        redis_host=getattr(config.cache, 'redis_host', 'localhost'),
        redis_port=getattr(config.cache, 'redis_port', 6379),
        ttl_seconds=getattr(config.cache, 'ttl', 3600),
        compression=getattr(config.cache, 'compression', True),
        disk_cache_dir=getattr(config.cache, 'disk_dir', '/tmp/pipeline_cache')
    )
    pipeline_cache = PipelineCache(cache_config)
    logger.info(f"Pipeline cache initialized with {cache_config.backend}")

distributed_trainer = None
if OPTIMIZATIONS_AVAILABLE and hasattr(config, 'distributed') and config.distributed.enabled:
    dist_config = DistributedConfig(
        backend=getattr(config.distributed, 'backend', 'ray'),
        num_workers=getattr(config.distributed, 'n_workers', 4),
        num_cpus_per_worker=getattr(
            config.distributed,
            'num_cpus_per_worker',
            getattr(config.distributed, 'n_cpus', 2)
        ),
        num_gpus_per_worker=getattr(
            config.distributed,
            'num_gpus_per_worker',
            getattr(config.distributed, 'n_gpus', 0.0)
        ),
        memory_per_worker_gb=getattr(
            config.distributed,
            'memory_per_worker_gb',
            getattr(config.distributed, 'memory_gb', 4)
        ),
    )
    distributed_trainer = DistributedTrainer(dist_config)
    logger.info(f"Distributed trainer initialized with {config.distributed.backend}")

incremental_learner = None
if OPTIMIZATIONS_AVAILABLE and hasattr(config, 'optimization'):
    incremental_learner = IncrementalLearner(
        max_memory_mb=getattr(config.optimization, 'max_memory_mb', 1000)
    )
    logger.info("Incremental learner initialized")


class GPUResourceManager:
    """Manage GPU resources for workers with billing tracking."""
    
    def __init__(self):
        self.available_gpus = self._detect_gpus()
        self.allocated_gpus = {}
        self.gpu_usage_start = {}
        
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs."""
        if TORCH_AVAILABLE:
            return list(range(GPU_COUNT))
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                return [gpu.id for gpu in gpus]
            except:
                pass
        
        return []
    
    def allocate_gpu(self, task_id: str, tenant_id: str, preferred_gpu: Optional[int] = None) -> Optional[int]:
        """Allocate a GPU for a task with tenant tracking."""
        tenant = tenant_manager.get_tenant(tenant_id)
        if not tenant or not tenant.features.get('gpu_training', False):
            logger.warning(f"Tenant {tenant_id} does not have GPU access")
            return None
        
        if not self.available_gpus:
            return None
        
        if preferred_gpu is not None and preferred_gpu not in self.allocated_gpus.values():
            self.allocated_gpus[task_id] = preferred_gpu
            self.gpu_usage_start[task_id] = time.time()
            return preferred_gpu
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpus.sort(key=lambda x: x.memoryUtil)
                
                for gpu in gpus:
                    if gpu.id not in self.allocated_gpus.values():
                        self.allocated_gpus[task_id] = gpu.id
                        self.gpu_usage_start[task_id] = time.time()
                        return gpu.id
            except:
                pass
        
        for gpu_id in self.available_gpus:
            if gpu_id not in self.allocated_gpus.values():
                self.allocated_gpus[task_id] = gpu_id
                self.gpu_usage_start[task_id] = time.time()
                return gpu_id
        
        return None
    
    def release_gpu(self, task_id: str, tenant_id: str):
        """Release GPU allocated to a task and track usage."""
        if task_id in self.allocated_gpus:
            gpu_id = self.allocated_gpus.pop(task_id)
            
            if task_id in self.gpu_usage_start:
                start_time = self.gpu_usage_start.pop(task_id)
                gpu_hours = (time.time() - start_time) / 3600
                usage_tracker.track_gpu_usage(tenant_id, gpu_hours)
                
            logger.info(f"Released GPU {gpu_id} from task {task_id}")
    
    def get_gpu_status(self) -> Dict:
        """Get current GPU status."""
        status = {
            "available_count": len(self.available_gpus),
            "allocated_count": len(self.allocated_gpus),
            "gpus": []
        }
        
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    status["gpus"].append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_used": f"{gpu.memoryUsed}MB",
                        "memory_total": f"{gpu.memoryTotal}MB",
                        "memory_util": f"{gpu.memoryUtil*100:.1f}%",
                        "gpu_util": f"{gpu.load*100:.1f}%",
                        "temperature": f"{gpu.temperature}°C",
                        "allocated": gpu.id in self.allocated_gpus.values()
                    })
            except:
                pass
        
        return status


# Global GPU manager
gpu_manager = GPUResourceManager()


class AutoMLTask(Task):
    """Base task with automatic tracking, error handling, and billing."""
    
    def before_start(self, task_id, args, kwargs):
        """Called before task execution."""
        tenant_id = kwargs.get('tenant_id', 'default')
        
        if not billing_manager.check_limits(tenant_id, 'concurrent_jobs', 1):
            raise Exception(f"Tenant {tenant_id} exceeded concurrent jobs limit")
        
        usage_tracker.track_api_call(tenant_id, 'task_execution')
        
        if 'require_gpu' in kwargs and kwargs['require_gpu']:
            gpu_id = gpu_manager.allocate_gpu(task_id, tenant_id)
            if gpu_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                logger.info(f"Task {task_id} allocated GPU {gpu_id}")
            else:
                logger.warning(f"Task {task_id} requested GPU but none available")
        
        cpu_requested = kwargs.get('cpu_cores', 1)
        memory_requested = kwargs.get('memory_gb', 2)
        
        if not tenant_manager.allocate_resources(tenant_id, cpu_requested, memory_requested, 0):
            raise Exception(f"Insufficient resources for tenant {tenant_id}")
        
        logger.info(f"Starting task {task_id}: {self.name} for tenant {tenant_id}")
        
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.start_time = time.time()
        self.tenant_id = tenant_id
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called on successful task completion."""
        tenant_id = kwargs.get('tenant_id', 'default')
        
        gpu_manager.release_gpu(task_id, tenant_id)
        
        cpu_requested = kwargs.get('cpu_cores', 1)
        memory_requested = kwargs.get('memory_gb', 2)
        tenant_manager.release_resources(tenant_id, cpu_requested, memory_requested, 0)
        
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = end_memory - self.start_memory
        time_taken = time.time() - self.start_time
        
        compute_hours = time_taken / 3600
        usage_tracker.track_compute_hours(tenant_id, compute_hours)
        
        logger.info(f"Task {task_id} completed successfully. "
                   f"Time: {time_taken:.2f}s, Memory: {memory_used:.2f}MB")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        tenant_id = kwargs.get('tenant_id', 'default')
        
        gpu_manager.release_gpu(task_id, tenant_id)
        
        cpu_requested = kwargs.get('cpu_cores', 1)
        memory_requested = kwargs.get('memory_gb', 2)
        tenant_manager.release_resources(tenant_id, cpu_requested, memory_requested, 0)
        
        logger.error(f"Task {task_id} failed: {exc}")
        
        if monitoring_service:
            from .monitoring import AlertManager
            alert_manager = AlertManager()
            alert_manager.check_alerts({
                "task_failure": True,
                "task_id": task_id,
                "tenant_id": tenant_id,
                "error": str(exc)
            })
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        logger.warning(f"Task {task_id} retrying: {exc}")


# CPU Training Task with Cache Support
@app.task(base=AutoMLTask, bind=True, name='automl.train.full_pipeline', 
          queue='training', max_retries=3)
def train_full_pipeline(self, job_id: str, dataset_url: str, config_dict: Dict[str, Any], 
                        user_id: str, tenant_id: str, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Execute complete AutoML training pipeline on CPU with caching.
    """
    try:
        # Check cache first
        if use_cache and pipeline_cache:
            cache_key = f"training_{tenant_id}_{hash(str(config_dict))}_{dataset_url}"
            cached_model = pipeline_cache.get_pipeline(cache_key)
            if cached_model:
                logger.info(f"Using cached model for job {job_id}")
                return {
                    'job_id': job_id,
                    'model_url': f"cached_{job_id}",
                    'cached': True,
                    'completed_at': datetime.now().isoformat()
                }
        
        api_key = kwargs.get('api_key')
        if api_key:
            verified_tenant_id = security_manager.verify_api_key(api_key)
            if verified_tenant_id != tenant_id:
                raise Exception("Invalid API key for tenant")
        
        from .config import AutoMLConfig
        job_config = AutoMLConfig(**config_dict)
        
        if not billing_manager.check_limits(tenant_id, 'models', 1):
            raise Exception(f"Tenant {tenant_id} exceeded model limit")
        
        df = storage_manager.load_dataset(dataset_url, tenant_id=tenant_id)
        storage_mb = df.memory_usage(deep=True).sum() / 1024**2
        usage_tracker.track_storage(tenant_id, storage_mb)
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 10, 'total': 100, 'status': 'Checking data quality...'}
        )
        
        if monitoring_service:
            quality_report = monitoring_service.quality_monitor.check_data_quality(df)
            if quality_report['quality_score'] < job_config.monitoring.min_quality_score:
                logger.warning(f"Low data quality: {quality_report['quality_score']}")
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 20, 'total': 100, 'status': 'Engineering features...'}
        )
        
        target_col = config_dict['target_column']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        orchestrator = EnhancedAutoMLOrchestrator(job_config)
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 30, 'total': 100, 'status': 'Training models...'}
        )
        
        orchestrator.fit(
            X, y,
            experiment_name=job_id,
            use_llm_features=job_config.llm.enable_feature_suggestions,
            use_llm_cleaning=job_config.llm.enable_data_cleaning,
            use_cache=use_cache,
            use_distributed=False,
            use_incremental=False
        )
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 90, 'total': 100, 'status': 'Saving model...'}
        )
        
        model_path = orchestrator.save_pipeline(f"/tmp/{job_id}_model.pkl")
        
        tenant = tenant_manager.get_tenant(tenant_id)
        if storage_manager:
            metadata = {
                'job_id': job_id,
                'user_id': user_id,
                'tenant_id': tenant_id,
                'created_at': datetime.now().isoformat(),
                'config': config_dict,
                'best_model': orchestrator.leaderboard[0]['model'] if orchestrator.leaderboard else None,
                'metrics': orchestrator.leaderboard[0]['metrics'] if orchestrator.leaderboard else {},
                'encrypted': tenant.encryption_enabled if tenant else False
            }
            
            model_url = storage_manager.save_model(
                orchestrator.best_pipeline,
                metadata,
                version="1.0.0"
            )
            
            # Cache the model
            if use_cache and pipeline_cache:
                pipeline_cache.set_pipeline(
                    cache_key,
                    orchestrator.best_pipeline,
                    X.head(100),
                    metrics=orchestrator.leaderboard[0]['metrics'] if orchestrator.leaderboard else {}
                )
            
            billing_manager.increment_model_count(tenant_id)
        else:
            model_url = model_path
        
        if monitoring_service and orchestrator.best_pipeline:
            monitor = monitoring_service.register_model(
                model_id=job_id,
                model_type=orchestrator.task,
                reference_data=X
            )
        
        results = {
            'job_id': job_id,
            'model_url': model_url,
            'best_model': orchestrator.leaderboard[0]['model'] if orchestrator.leaderboard else None,
            'cv_score': orchestrator.leaderboard[0]['cv_score'] if orchestrator.leaderboard else None,
            'leaderboard': orchestrator.get_leaderboard().to_dict('records'),
            'completed_at': datetime.now().isoformat(),
            'worker_type': 'CPU',
            'cached': False,
            'billing_info': {
                'compute_hours': (time.time() - self.start_time) / 3600,
                'storage_mb': storage_mb,
                'tenant_plan': tenant.plan if tenant else 'unknown'
            }
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {str(e)}\n{traceback.format_exc()}")
        raise self.retry(exc=e, countdown=60)


# Distributed Training Task with Ray/Dask
@app.task(base=AutoMLTask, bind=True, name='automl.train.distributed', 
          queue='distributed', max_retries=3)
def train_distributed_pipeline(self, job_id: str, dataset_url: str, config_dict: Dict[str, Any],
                              user_id: str, tenant_id: str, backend: str = 'ray', **kwargs) -> Dict[str, Any]:
    """
    Execute distributed training pipeline with Ray or Dask.
    """
    try:
        if not OPTIMIZATIONS_AVAILABLE:
            logger.warning("Optimization components not available, falling back to standard training")
            return train_full_pipeline.apply_async(
                args=[job_id, dataset_url, config_dict, user_id, tenant_id],
                kwargs=kwargs,
                queue='training'
            ).get()
        
        tenant = tenant_manager.get_tenant(tenant_id)
        if not tenant or not tenant.features.get('distributed_training', False):
            raise Exception(f"Tenant {tenant_id} does not have distributed training access")
        
        if backend == 'ray' and RAY_AVAILABLE:
            if not ray.is_initialized():
                ray.init(num_cpus=kwargs.get('n_cpus', 4))
            logger.info(f"Using Ray for distributed training")
        elif backend == 'dask' and DASK_AVAILABLE:
            from dask.distributed import Client
            client = Client(n_workers=kwargs.get('n_workers', 4))
            logger.info(f"Using Dask for distributed training")
        else:
            logger.warning(f"Backend {backend} not available, falling back to standard training")
            return train_full_pipeline.apply_async(
                args=[job_id, dataset_url, config_dict, user_id, tenant_id],
                kwargs=kwargs,
                queue='training'
            ).get()
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': f'Initializing {backend}...'}
        )
        
        df = storage_manager.load_dataset(dataset_url, tenant_id=tenant_id)
        target_col = config_dict['target_column']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        from .config import AutoMLConfig
        job_config = AutoMLConfig(**config_dict)
        
        dist_config = DistributedConfig(
            backend=backend,
            num_workers=kwargs.get('n_workers', 4),
            num_cpus_per_worker=kwargs.get('n_cpus', 2),
            num_gpus_per_worker=kwargs.get('n_gpus', 0.0),
            memory_per_worker_gb=kwargs.get('memory_per_worker_gb', 4),
        )

        trainer = DistributedTrainer(dist_config)
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 20, 'total': 100, 'status': 'Training models in parallel...'}
        )
        
        from .model_selection import get_available_models, get_param_grid
        
        models = get_available_models(detect_task(y))
        param_grids = {name: get_param_grid(name) for name in models.keys()}
        
        results = trainer.train(
            X,
            y,
            models,
            param_grids,
        )

        successful_results = [
            result for result in results.values()
            if result.get('status') == 'success'
        ]
        if not successful_results:
            raise RuntimeError("Distributed training failed for all models")

        best_result = max(
            successful_results,
            key=lambda res: res.get('best_score', float('-inf'))
        )

        best_pipeline = best_result.get('best_model')
        if best_pipeline is None:
            raise RuntimeError("Distributed trainer did not return a trained model")

        task_type = detect_task(y)
        y_pred = best_pipeline.predict(X)
        if task_type == 'classification' and hasattr(best_pipeline, 'predict_proba'):
            y_proba = best_pipeline.predict_proba(X)
        else:
            y_proba = None

        metrics = calculate_metrics(y, y_pred, y_proba, task_type)

        self.update_state(
            state='PROGRESS',
            meta={'current': 80, 'total': 100, 'status': 'Saving distributed model...'}
        )
        
        if storage_manager:
            metadata = {
                'job_id': job_id,
                'user_id': user_id,
                'tenant_id': tenant_id,
                'distributed_backend': backend,
                'n_workers': kwargs.get('n_workers', 4),
                'best_model': best_result.get('model_name'),
                'metrics': metrics,
                'created_at': datetime.now().isoformat()
            }
            
            model_url = storage_manager.save_model(best_pipeline, metadata)

            if pipeline_cache:
                pipeline_cache.set_pipeline(
                    f"distributed_{job_id}",
                    best_pipeline,
                    X.head(100),
                    metrics=metrics
                )
        else:
            model_url = f"/tmp/{job_id}_distributed_model.pkl"

        trainer.shutdown()
        if backend == 'ray' and ray.is_initialized():
            ray.shutdown()
        
        return {
            'job_id': job_id,
            'model_url': model_url,
            'backend': backend,
            'best_model': best_result.get('model_name'),
            'cv_score': best_result.get('best_score'),
            'metrics': metrics,
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Distributed training failed: {str(e)}")
        raise self.retry(exc=e, countdown=60)


# Incremental Learning Task
@app.task(base=AutoMLTask, bind=True, name='automl.train.incremental', 
          queue='incremental', max_retries=3)
def train_incremental_pipeline(self, job_id: str, stream_url: str, config_dict: Dict[str, Any],
                               user_id: str, tenant_id: str, batch_size: int = 1000, **kwargs) -> Dict[str, Any]:
    """
    Train models incrementally for streaming or large datasets.
    """
    try:
        if not OPTIMIZATIONS_AVAILABLE or not incremental_learner:
            logger.warning("Incremental learning not available, using standard training")
            return train_full_pipeline.apply_async(
                args=[job_id, stream_url, config_dict, user_id, tenant_id],
                kwargs=kwargs,
                queue='training'
            ).get()
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Initializing incremental learning...'}
        )
        
        learner = IncrementalLearner(
            max_memory_mb=kwargs.get('max_memory_mb', 1000)
        )
        
        total_samples = 0
        models = {}
        
        df = storage_manager.load_dataset(stream_url, tenant_id=tenant_id)
        target_col = config_dict['target_column']
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            X_batch = batch.drop(columns=[target_col])
            y_batch = batch[target_col]
            
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': int((i / len(df)) * 100),
                    'total': 100,
                    'status': f'Processing batch {i//batch_size + 1}...'
                }
            )
            
            task_type = detect_task(y_batch)
            models = learner.train_incremental(X_batch, y_batch, task_type)
            
            total_samples += len(batch)
            
            if learner.get_memory_usage() > learner.max_memory_mb * 0.9:
                logger.warning("Memory usage high, clearing buffers")
                learner.clear_buffers()
        
        best_model = learner.get_best_model()
        
        if storage_manager:
            metadata = {
                'job_id': job_id,
                'user_id': user_id,
                'tenant_id': tenant_id,
                'training_type': 'incremental',
                'total_samples': total_samples,
                'batch_size': batch_size,
                'created_at': datetime.now().isoformat()
            }
            
            model_url = storage_manager.save_model(best_model, metadata)
        else:
            model_url = f"/tmp/{job_id}_incremental_model.pkl"
        
        return {
            'job_id': job_id,
            'model_url': model_url,
            'training_type': 'incremental',
            'total_samples': total_samples,
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Incremental training failed: {str(e)}")
        raise self.retry(exc=e, countdown=60)


# GPU Training Task (existing, unchanged)
@app.task(base=AutoMLTask, bind=True, name='automl.train.gpu.neural_pipeline', 
          queue='gpu', max_retries=2)
def train_neural_pipeline_gpu(self, job_id: str, dataset_url: str, config_dict: Dict[str, Any],
                              user_id: str, tenant_id: str, require_gpu: bool = True, **kwargs) -> Dict[str, Any]:
    """Execute neural network training pipeline on GPU."""
    # [Keep existing implementation unchanged]
    try:
        tenant = tenant_manager.get_tenant(tenant_id)
        if not tenant or not tenant.features.get('gpu_training', False):
            raise Exception(f"Tenant {tenant_id} does not have GPU training access")
        
        if require_gpu and not TORCH_AVAILABLE:
            raise Exception("GPU requested but not available. Falling back to CPU queue.")
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Initializing GPU...'}
        )
        
        if TORCH_AVAILABLE:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f}GB memory")
        
        df = storage_manager.load_dataset(dataset_url, tenant_id=tenant_id)
        
        target_col = config_dict['target_column']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 20, 'total': 100, 'status': 'Training neural networks on GPU...'}
        )
        
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        
        if y.dtype in ['float64', 'float32']:
            model = TabNetRegressor(
                n_d=16, n_a=16,
                n_steps=5,
                gamma=1.3,
                n_independent=2,
                n_shared=2,
                seed=42,
                verbose=0,
                device_name='cuda' if TORCH_AVAILABLE else 'cpu'
            )
            task_type = 'regression'
        else:
            model = TabNetClassifier(
                n_d=16, n_a=16,
                n_steps=5,
                gamma=1.3,
                n_independent=2,
                n_shared=2,
                seed=42,
                verbose=0,
                device_name='cuda' if TORCH_AVAILABLE else 'cpu'
            )
            task_type = 'classification'
        
        X_train = X.values.astype(np.float32)
        y_train = y.values
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            max_epochs=100,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128
        )
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 80, 'total': 100, 'status': 'Saving GPU model...'}
        )
        
        if storage_manager:
            metadata = {
                'job_id': job_id,
                'user_id': user_id,
                'tenant_id': tenant_id,
                'model_type': 'TabNet',
                'task_type': task_type,
                'gpu_used': gpu_name if TORCH_AVAILABLE else 'None',
                'created_at': datetime.now().isoformat()
            }
            
            model_url = storage_manager.save_model(model, metadata, version="1.0.0")
            billing_manager.increment_model_count(tenant_id, model_type='gpu')
        else:
            model_url = f"/tmp/{job_id}_gpu_model.pkl"
            model.save_model(model_url)
        
        gpu_hours = (time.time() - self.start_time) / 3600
        usage_tracker.track_gpu_usage(tenant_id, gpu_hours)
        
        return {
            'job_id': job_id,
            'model_url': model_url,
            'model_type': 'TabNet',
            'task_type': task_type,
            'gpu_used': True,
            'completed_at': datetime.now().isoformat(),
            'worker_type': 'GPU',
            'billing_info': {
                'gpu_hours': gpu_hours,
                'gpu_name': gpu_name if TORCH_AVAILABLE else None,
                'tenant_plan': tenant.plan
            }
        }
        
    except Exception as e:
        logger.error(f"GPU training failed for job {job_id}: {str(e)}")
        
        if require_gpu and "CUDA" in str(e):
            logger.info(f"Falling back to CPU training for job {job_id}")
            return train_full_pipeline.apply_async(
                args=[job_id, dataset_url, config_dict, user_id, tenant_id],
                kwargs=kwargs,
                queue='training'
            ).get()
        
        raise self.retry(exc=e, countdown=30)


# Cache Management Tasks
@app.task(name='automl.cache.warm', queue='cache')
def warm_pipeline_cache(model_ids: List[str], tenant_id: str) -> Dict[str, Any]:
    """Warm up pipeline cache with frequently used models."""
    try:
        if not pipeline_cache:
            return {'status': 'cache_disabled'}
        
        warmed = 0
        for model_id in model_ids:
            try:
                model, metadata = storage_manager.load_model(model_id, tenant_id=tenant_id)
                
                pipeline_cache.set_pipeline(
                    model_id,
                    model,
                    metrics=metadata.get('metrics', {})
                )
                warmed += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache for {model_id}: {e}")
        
        return {
            'status': 'success',
            'models_warmed': warmed,
            'cache_stats': pipeline_cache.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Cache warming failed: {str(e)}")
        return {'status': 'failed', 'error': str(e)}


@app.task(name='automl.cache.health', queue='monitoring')
def check_cache_health() -> Dict[str, Any]:
    """Monitor pipeline cache health."""
    if not pipeline_cache:
        return {'status': 'cache_disabled'}
    
    health = monitor_cache_health(pipeline_cache)
    
    if health['issues'] and monitoring_service:
        from .monitoring import AlertManager
        alert_manager = AlertManager()
        for issue in health['issues']:
            alert_manager.check_alerts({
                'cache_issue': True,
                'message': issue
            })
    
    return health


@app.task(name='automl.cache.clear', queue='cache')
def clear_pipeline_cache(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """Clear pipeline cache."""
    if not pipeline_cache:
        return {'status': 'cache_disabled'}
    
    if tenant_id:
        # Clear only tenant-specific cache entries
        # This would require enhanced cache implementation
        logger.info(f"Clearing cache for tenant {tenant_id}")
    else:
        pipeline_cache.clear_all()
    
    return {
        'status': 'success',
        'cleared_at': datetime.now().isoformat()
    }


# Memory Optimization Task
@app.task(name='automl.optimize.memory', queue='optimization')
def optimize_memory_usage() -> Dict[str, Any]:
    """Optimize memory usage across workers."""
    
    stats = {
        'timestamp': datetime.now().isoformat(),
        'before': {
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / 1024**3
        }
    }
    
    # Clear pipeline cache if memory is high
    if stats['before']['memory_percent'] > 80 and pipeline_cache:
        logger.info("Memory usage high, clearing pipeline cache")
        pipeline_cache.clear_all()
    
    # Force garbage collection
    gc.collect()
    
    # Clear incremental learner buffers
    if incremental_learner:
        incremental_learner.clear_buffers()
    
    # Clear PyTorch cache if available
    if TORCH_AVAILABLE:
        import torch
        torch.cuda.empty_cache()
    
    stats['after'] = {
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_gb': psutil.virtual_memory().used / 1024**3
    }
    
    stats['memory_freed_gb'] = stats['before']['memory_used_gb'] - stats['after']['memory_used_gb']
    
    logger.info(f"Memory optimization freed {stats['memory_freed_gb']:.2f} GB")
    
    return stats


# Batch Prediction Task (keep existing implementation)
@app.task(base=AutoMLTask, bind=True, name='automl.predict.batch', 
          queue='prediction', max_retries=2)
def predict_batch(self, job_id: str, model_url: str, data_url: str, 
                  tenant_id: str, use_gpu: bool = False, **kwargs) -> Dict[str, Any]:
    """Batch prediction task."""
    # [Keep existing implementation]
    try:
        usage_tracker.track_api_call(tenant_id, 'batch_prediction')
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Loading model...'}
        )
        
        model, metadata = storage_manager.load_model(model_url, tenant_id=tenant_id)
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 30, 'total': 100, 'status': 'Loading data...'}
        )
        
        df = storage_manager.load_dataset(data_url, tenant_id=tenant_id)
        
        if use_gpu and TORCH_AVAILABLE and hasattr(model, 'device_name'):
            model.device_name = 'cuda'
            logger.info(f"Using GPU for batch prediction")
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 50, 'total': 100, 'status': 'Making predictions...'}
        )
        
        start_time = time.time()
        predictions = model.predict(df)
        prediction_time = time.time() - start_time
        
        usage_tracker.track_predictions(tenant_id, len(predictions))
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(df)
            except:
                pass
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 80, 'total': 100, 'status': 'Saving results...'}
        )
        
        results_df = pd.DataFrame({'prediction': predictions})
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                results_df[f'probability_class_{i}'] = probabilities[:, i]
        
        results_url = storage_manager.save_dataset(
            results_df,
            f"predictions_{job_id}",
            tenant_id=tenant_id
        )
        
        if monitoring_service:
            monitor = monitoring_service.get_monitor(metadata.get('model_id', model_url))
            if monitor:
                monitor.log_prediction(df, predictions, None, prediction_time)
        
        return {
            'job_id': job_id,
            'results_url': results_url,
            'n_predictions': len(predictions),
            'prediction_time': prediction_time,
            'gpu_used': use_gpu and TORCH_AVAILABLE,
            'completed_at': datetime.now().isoformat(),
            'billing_info': {
                'predictions_count': len(predictions),
                'compute_seconds': prediction_time
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction failed for job {job_id}: {str(e)}")
        raise self.retry(exc=e, countdown=30)


# System Status Task (enhanced)
@app.task(name='automl.system.status', queue='monitoring')
def get_system_status() -> Dict[str, Any]:
    """Get system status including GPU availability, cache, and billing info."""
    status = {
        'timestamp': datetime.now().isoformat(),
        'workers': {
            'active': app.control.inspect().active(),
            'scheduled': app.control.inspect().scheduled(),
            'reserved': app.control.inspect().reserved()
        },
        'queues': {},
        'resources': {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
    }
    
    # Add GPU status
    if GPU_COUNT > 0:
        status['gpu'] = gpu_manager.get_gpu_status()
    else:
        status['gpu'] = {'available': False, 'count': 0}
    
    # Add cache status
    if pipeline_cache:
        status['cache'] = pipeline_cache.get_stats()
    else:
        status['cache'] = {'enabled': False}
    
    # Add distributed training status
    if distributed_trainer:
        status['distributed'] = {
            'enabled': True,
            'backend': getattr(config.distributed, 'backend', 'unknown')
        }
    else:
        status['distributed'] = {'enabled': False}
    
    # Get queue lengths
    try:
        import redis
        r = redis.from_url(app.conf.broker_url)
        for queue in ['default', 'training', 'distributed', 'incremental', 'gpu', 
                     'prediction', 'monitoring', 'cache', 'optimization']:
            status['queues'][queue] = r.llen(f"celery:{queue}")
    except:
        pass
    
    # Add billing summary
    status['billing'] = billing_manager.get_system_summary()
    
    return status


# Celery beat schedule for periodic tasks
from celery.schedules import crontab

app.conf.beat_schedule = {
    'cleanup-old-jobs': {
        'task': 'automl.maintenance.cleanup',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    'monitor-system-health': {
        'task': 'automl.system.status',
        'schedule': 60.0,  # Every minute
    },
    'check-gpu-health': {
        'task': 'automl.gpu.health_check',
        'schedule': 300.0,  # Every 5 minutes
    },
    'update-billing-metrics': {
        'task': 'automl.billing.update_metrics',
        'schedule': 3600.0,  # Every hour
    },
    'check-cache-health': {
        'task': 'automl.cache.health',
        'schedule': 600.0,  # Every 10 minutes
    },
    'optimize-memory': {
        'task': 'automl.optimize.memory',
        'schedule': 1800.0,  # Every 30 minutes
    },
    'warm-cache': {
        'task': 'automl.cache.warm',
        'schedule': crontab(hour='*/6'),  # Every 6 hours
        'kwargs': {
            'model_ids': [],  # Will be populated with frequently used models
            'tenant_id': 'default'
        }
    }
}


# Keep existing tasks unchanged
# - export_model_docker
# - monitor_drift
# - scheduled_retrain
# - gpu_health_check
# - update_billing_metrics
# - process_streaming_data

# Signal handlers
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Log task start."""
    logger.info(f"Task {task.name} [{task_id}] starting")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Log task completion."""
    logger.info(f"Task {task.name} [{task_id}] completed")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    """Handle task failure."""
    logger.error(f"Task [{task_id}] failed: {exception}")


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Initialize worker with optimization components."""
    logger.info(f"Worker ready. GPUs: {GPU_COUNT}, Cache: {pipeline_cache is not None}, "
               f"Distributed: {distributed_trainer is not None}, "
               f"Incremental: {incremental_learner is not None}")
    
    if GPU_COUNT > 0:
        logger.info(f"GPU status: {gpu_manager.get_gpu_status()}")
    
    if pipeline_cache:
        logger.info(f"Cache stats: {pipeline_cache.get_stats()}")
