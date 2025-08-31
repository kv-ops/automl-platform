"""
Celery worker for asynchronous AutoML jobs
Implements distributed training with proper isolation and GPU support
WITH COMPLETE GPU QUEUE CONFIGURATION
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
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU availability check
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if TORCH_AVAILABLE else 0
except ImportError:
    TORCH_AVAILABLE = False
    GPU_COUNT = 0

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
    
    # Queue routing configuration with GPU support
    'task_routes': {
        'automl.train.*': {'queue': 'training'},
        'automl.train.gpu.*': {'queue': 'gpu'},  # GPU training tasks
        'automl.predict.*': {'queue': 'prediction'},
        'automl.predict.gpu.*': {'queue': 'gpu'},  # GPU prediction tasks
        'automl.llm.*': {'queue': 'llm'},
        'automl.monitor.*': {'queue': 'monitoring'},
        'automl.export.*': {'queue': 'export'}
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

# Initialize services
config = load_config()
storage_manager = StorageManager(
    backend=config.storage.backend,
    endpoint=config.storage.endpoint,
    access_key=config.storage.access_key,
    secret_key=config.storage.secret_key
) if config.storage.backend != "none" else None

monitoring_service = MonitoringService(storage_manager) if config.monitoring.enabled else None


class GPUResourceManager:
    """Manage GPU resources for workers"""
    
    def __init__(self):
        self.available_gpus = self._detect_gpus()
        self.allocated_gpus = {}
        
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs"""
        if TORCH_AVAILABLE:
            return list(range(GPU_COUNT))
        
        try:
            gpus = GPUtil.getGPUs()
            return [gpu.id for gpu in gpus]
        except:
            return []
    
    def allocate_gpu(self, task_id: str, preferred_gpu: Optional[int] = None) -> Optional[int]:
        """Allocate a GPU for a task"""
        if not self.available_gpus:
            return None
        
        # Try to allocate preferred GPU
        if preferred_gpu is not None and preferred_gpu not in self.allocated_gpus.values():
            self.allocated_gpus[task_id] = preferred_gpu
            return preferred_gpu
        
        # Find least loaded GPU
        try:
            gpus = GPUtil.getGPUs()
            gpus.sort(key=lambda x: x.memoryUtil)  # Sort by memory usage
            
            for gpu in gpus:
                if gpu.id not in self.allocated_gpus.values():
                    self.allocated_gpus[task_id] = gpu.id
                    return gpu.id
        except:
            # Fallback to simple allocation
            for gpu_id in self.available_gpus:
                if gpu_id not in self.allocated_gpus.values():
                    self.allocated_gpus[task_id] = gpu_id
                    return gpu_id
        
        return None
    
    def release_gpu(self, task_id: str):
        """Release GPU allocated to a task"""
        if task_id in self.allocated_gpus:
            gpu_id = self.allocated_gpus.pop(task_id)
            logger.info(f"Released GPU {gpu_id} from task {task_id}")
    
    def get_gpu_status(self) -> Dict:
        """Get current GPU status"""
        status = {
            "available_count": len(self.available_gpus),
            "allocated_count": len(self.allocated_gpus),
            "gpus": []
        }
        
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
    """Base task with automatic tracking, error handling, and resource management"""
    
    def before_start(self, task_id, args, kwargs):
        """Called before task execution"""
        # Check if GPU is required
        if 'require_gpu' in kwargs and kwargs['require_gpu']:
            gpu_id = gpu_manager.allocate_gpu(task_id)
            if gpu_id is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                logger.info(f"Task {task_id} allocated GPU {gpu_id}")
            else:
                logger.warning(f"Task {task_id} requested GPU but none available")
        
        # Log task start
        logger.info(f"Starting task {task_id}: {self.name}")
        
        # Monitor resource usage
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.time()
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called on successful task completion"""
        # Release GPU if allocated
        gpu_manager.release_gpu(task_id)
        
        # Log resource usage
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = end_memory - self.start_memory
        time_taken = time.time() - self.start_time
        
        logger.info(f"Task {task_id} completed successfully. "
                   f"Time: {time_taken:.2f}s, Memory: {memory_used:.2f}MB")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure"""
        # Release GPU if allocated
        gpu_manager.release_gpu(task_id)
        
        logger.error(f"Task {task_id} failed: {exc}")
        
        # Send alert if monitoring enabled
        if monitoring_service:
            from .monitoring import AlertManager
            alert_manager = AlertManager()
            alert_manager.check_alerts({
                "task_failure": True,
                "task_id": task_id,
                "error": str(exc)
            })
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(f"Task {task_id} retrying: {exc}")


# CPU Training Task
@app.task(base=AutoMLTask, bind=True, name='automl.train.full_pipeline', 
          queue='training', max_retries=3)
def train_full_pipeline(self, job_id: str, dataset_url: str, config_dict: Dict[str, Any], 
                        user_id: str, tenant_id: str) -> Dict[str, Any]:
    """
    Execute complete AutoML training pipeline on CPU.
    
    Args:
        job_id: Unique job identifier
        dataset_url: S3/MinIO URL to dataset
        config_dict: Training configuration
        user_id: User identifier for isolation
        tenant_id: Tenant identifier for multi-tenancy
    
    Returns:
        Training results with model URL and metrics
    """
    try:
        # Check quota
        from .config import AutoMLConfig
        job_config = AutoMLConfig(**config_dict)
        
        if not job_config.check_quota('max_concurrent_jobs', 1):
            raise Exception(f"Quota exceeded for plan {job_config.billing.plan_type}")
        
        # Update job status
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Loading dataset...'}
        )
        
        # Load dataset from storage
        df = storage_manager.load_dataset(dataset_url, tenant_id=tenant_id)
        
        # Data quality check
        self.update_state(
            state='PROGRESS',
            meta={'current': 10, 'total': 100, 'status': 'Checking data quality...'}
        )
        
        if monitoring_service:
            quality_report = monitoring_service.quality_monitor.check_data_quality(df)
            if quality_report['quality_score'] < job_config.monitoring.min_quality_score:
                logger.warning(f"Low data quality: {quality_report['quality_score']}")
        
        # Feature engineering
        self.update_state(
            state='PROGRESS',
            meta={'current': 20, 'total': 100, 'status': 'Engineering features...'}
        )
        
        # Split features and target
        target_col = config_dict['target_column']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Initialize orchestrator
        orchestrator = EnhancedAutoMLOrchestrator(job_config)
        
        # Training with progress updates
        self.update_state(
            state='PROGRESS',
            meta={'current': 30, 'total': 100, 'status': 'Training models...'}
        )
        
        # Train
        orchestrator.fit(
            X, y,
            experiment_name=job_id,
            use_llm_features=job_config.llm.enable_feature_suggestions,
            use_llm_cleaning=job_config.llm.enable_data_cleaning
        )
        
        # Save model
        self.update_state(
            state='PROGRESS',
            meta={'current': 90, 'total': 100, 'status': 'Saving model...'}
        )
        
        model_path = orchestrator.save_pipeline(f"/tmp/{job_id}_model.pkl")
        
        # Save to storage
        if storage_manager:
            metadata = {
                'job_id': job_id,
                'user_id': user_id,
                'tenant_id': tenant_id,
                'created_at': datetime.now().isoformat(),
                'config': config_dict,
                'best_model': orchestrator.leaderboard[0]['model'] if orchestrator.leaderboard else None,
                'metrics': orchestrator.leaderboard[0]['metrics'] if orchestrator.leaderboard else {}
            }
            
            model_url = storage_manager.save_model(
                orchestrator.best_pipeline,
                metadata,
                version="1.0.0"
            )
        else:
            model_url = model_path
        
        # Register monitoring
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
            'worker_type': 'CPU'
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {str(e)}\n{traceback.format_exc()}")
        raise self.retry(exc=e, countdown=60)


# GPU Training Task
@app.task(base=AutoMLTask, bind=True, name='automl.train.gpu.neural_pipeline', 
          queue='gpu', max_retries=2)
def train_neural_pipeline_gpu(self, job_id: str, dataset_url: str, config_dict: Dict[str, Any],
                              user_id: str, tenant_id: str, require_gpu: bool = True) -> Dict[str, Any]:
    """
    Execute neural network training pipeline on GPU.
    
    Args:
        job_id: Unique job identifier
        dataset_url: S3/MinIO URL to dataset
        config_dict: Training configuration
        user_id: User identifier
        tenant_id: Tenant identifier
        require_gpu: Whether GPU is required
    
    Returns:
        Training results
    """
    try:
        # Verify GPU is available
        if require_gpu and not TORCH_AVAILABLE:
            raise Exception("GPU requested but not available. Falling back to CPU queue.")
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Initializing GPU...'}
        )
        
        # Log GPU info
        if TORCH_AVAILABLE:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f}GB memory")
        
        # Load dataset
        df = storage_manager.load_dataset(dataset_url, tenant_id=tenant_id)
        
        target_col = config_dict['target_column']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 20, 'total': 100, 'status': 'Training neural networks on GPU...'}
        )
        
        # Train TabNet or other neural models
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
        
        # Determine task type
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
        
        # Convert to numpy
        X_train = X.values.astype(np.float32)
        y_train = y.values
        
        # Train model
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
        
        # Save model
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
        else:
            model_url = f"/tmp/{job_id}_gpu_model.pkl"
            model.save_model(model_url)
        
        results = {
            'job_id': job_id,
            'model_url': model_url,
            'model_type': 'TabNet',
            'task_type': task_type,
            'gpu_used': True,
            'completed_at': datetime.now().isoformat(),
            'worker_type': 'GPU'
        }
        
        return results
        
    except Exception as e:
        logger.error(f"GPU training failed for job {job_id}: {str(e)}")
        
        # Fallback to CPU if GPU fails
        if require_gpu and "CUDA" in str(e):
            logger.info(f"Falling back to CPU training for job {job_id}")
            return train_full_pipeline.apply_async(
                args=[job_id, dataset_url, config_dict, user_id, tenant_id],
                queue='training'
            )
        
        raise self.retry(exc=e, countdown=30)


# Batch Prediction Task
@app.task(base=AutoMLTask, bind=True, name='automl.predict.batch', 
          queue='prediction', max_retries=2)
def predict_batch(self, job_id: str, model_url: str, data_url: str, 
                  tenant_id: str, use_gpu: bool = False) -> Dict[str, Any]:
    """
    Batch prediction task with optional GPU acceleration.
    
    Args:
        job_id: Prediction job ID
        model_url: S3/MinIO URL to model
        data_url: S3/MinIO URL to prediction data
        tenant_id: Tenant identifier
        use_gpu: Whether to use GPU for prediction
    
    Returns:
        Prediction results with URL to output file
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Loading model...'}
        )
        
        # Load model
        model, metadata = storage_manager.load_model(model_url, tenant_id=tenant_id)
        
        # Load data
        self.update_state(
            state='PROGRESS',
            meta={'current': 30, 'total': 100, 'status': 'Loading data...'}
        )
        
        df = storage_manager.load_dataset(data_url, tenant_id=tenant_id)
        
        # Check if GPU should be used
        if use_gpu and TORCH_AVAILABLE and hasattr(model, 'device_name'):
            model.device_name = 'cuda'
            logger.info(f"Using GPU for batch prediction")
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 50, 'total': 100, 'status': 'Making predictions...'}
        )
        
        # Make predictions
        start_time = time.time()
        predictions = model.predict(df)
        prediction_time = time.time() - start_time
        
        # Add probabilities if classification
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
        
        # Save predictions
        results_df = pd.DataFrame({'prediction': predictions})
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                results_df[f'probability_class_{i}'] = probabilities[:, i]
        
        # Save to storage
        results_url = storage_manager.save_dataset(
            results_df,
            f"predictions_{job_id}",
            tenant_id=tenant_id
        )
        
        # Log to monitoring
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
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction failed for job {job_id}: {str(e)}")
        raise self.retry(exc=e, countdown=30)


# Model Export Task
@app.task(base=AutoMLTask, bind=True, name='automl.export.docker', 
          queue='export', max_retries=2)
def export_model_docker(self, model_id: str, tenant_id: str, 
                       output_format: str = "docker") -> Dict[str, Any]:
    """
    Export model to Docker, ONNX, or PMML format.
    
    Args:
        model_id: Model identifier
        tenant_id: Tenant identifier
        output_format: Export format (docker, onnx, pmml)
    
    Returns:
        Export results with download URL
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': f'Exporting to {output_format}...'}
        )
        
        if output_format == "docker":
            export_path = storage_manager.export_model_to_docker(
                model_id, tenant_id=tenant_id
            )
        elif output_format == "onnx":
            export_path = storage_manager.export_model_to_onnx(
                model_id, tenant_id=tenant_id
            )
        elif output_format == "pmml":
            export_path = storage_manager.export_model_to_pmml(
                model_id, tenant_id=tenant_id
            )
        else:
            raise ValueError(f"Unsupported export format: {output_format}")
        
        return {
            'model_id': model_id,
            'export_format': output_format,
            'export_path': export_path,
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Export failed for model {model_id}: {str(e)}")
        raise self.retry(exc=e, countdown=30)


# Monitoring Task
@app.task(base=AutoMLTask, name='automl.monitor.drift', queue='monitoring')
def monitor_drift(tenant_id: str, model_id: str, 
                  reference_data_url: str, current_data_url: str) -> Dict[str, Any]:
    """
    Monitor data/model drift.
    
    Args:
        tenant_id: Tenant identifier
        model_id: Model identifier
        reference_data_url: URL to reference dataset
        current_data_url: URL to current dataset
    
    Returns:
        Drift analysis results
    """
    try:
        # Load datasets
        ref_df = storage_manager.load_dataset(reference_data_url, tenant_id=tenant_id)
        curr_df = storage_manager.load_dataset(current_data_url, tenant_id=tenant_id)
        
        # Get monitor
        monitor = monitoring_service.get_monitor(model_id)
        if not monitor:
            monitor = monitoring_service.register_model(
                model_id=model_id,
                model_type='unknown',
                reference_data=ref_df
            )
        
        # Check drift
        drift_results = monitor.check_drift(curr_df)
        
        # Send alert if significant drift
        if drift_results['drift_detected']:
            from .monitoring import MonitoringIntegration
            
            alert = {
                'type': 'data_drift',
                'severity': 'high' if len(drift_results['drifted_features']) > 5 else 'medium',
                'message': f"Drift detected in {len(drift_results['drifted_features'])} features",
                'model_id': model_id,
                'tenant_id': tenant_id,
                'drifted_features': drift_results['drifted_features'][:10]
            }
            
            # Send to configured channels
            if config.monitoring.slack_webhook_url:
                MonitoringIntegration.send_to_slack(alert, config.monitoring.slack_webhook_url)
            
            if config.monitoring.email_recipients:
                smtp_config = {
                    'host': config.monitoring.email_smtp_host,
                    'port': config.monitoring.email_smtp_port,
                    'from_email': config.monitoring.email_from,
                    'username': os.getenv('SMTP_USERNAME'),
                    'password': os.getenv('SMTP_PASSWORD')
                }
                MonitoringIntegration.send_to_email(
                    alert, smtp_config, config.monitoring.email_recipients
                )
        
        return drift_results
        
    except Exception as e:
        logger.error(f"Drift monitoring failed: {str(e)}")
        raise


# Scheduled Retraining Task
@app.task(base=AutoMLTask, name='automl.retrain.scheduled', queue='training')
def scheduled_retrain(tenant_id: str, model_id: str, 
                      dataset_url: str, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scheduled model retraining.
    
    Args:
        tenant_id: Tenant identifier
        model_id: Model to retrain
        dataset_url: New training data URL
        config_dict: Training configuration
    
    Returns:
        New model information
    """
    try:
        # Generate new job ID
        job_id = f"retrain_{model_id}_{int(time.time())}"
        
        # Check if GPU model
        original_model, metadata = storage_manager.load_model(model_id, tenant_id=tenant_id)
        use_gpu = metadata.get('gpu_used', False)
        
        # Trigger appropriate training
        if use_gpu and GPU_COUNT > 0:
            result = train_neural_pipeline_gpu.apply_async(
                args=[job_id, dataset_url, config_dict, 'system', tenant_id],
                kwargs={'require_gpu': True},
                queue='gpu'
            )
        else:
            result = train_full_pipeline.apply_async(
                args=[job_id, dataset_url, config_dict, 'system', tenant_id],
                queue='training'
            )
        
        return {
            'job_id': job_id,
            'task_id': result.id,
            'status': 'scheduled',
            'scheduled_at': datetime.now().isoformat(),
            'use_gpu': use_gpu
        }
        
    except Exception as e:
        logger.error(f"Scheduled retrain failed: {str(e)}")
        raise


# System Status Task
@app.task(name='automl.system.status', queue='monitoring')
def get_system_status() -> Dict[str, Any]:
    """Get system status including GPU availability"""
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
    
    # Get queue lengths
    import redis
    r = redis.from_url(app.conf.broker_url)
    for queue in ['default', 'training', 'gpu', 'prediction', 'llm', 'monitoring', 'export']:
        status['queues'][queue] = r.llen(f"celery:{queue}")
    
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
    }
}


# GPU Health Check Task
@app.task(name='automl.gpu.health_check', queue='monitoring')
def gpu_health_check() -> Dict[str, Any]:
    """Check GPU health and availability"""
    health = {
        'timestamp': datetime.now().isoformat(),
        'gpu_available': GPU_COUNT > 0,
        'gpu_count': GPU_COUNT,
        'issues': []
    }
    
    if GPU_COUNT > 0:
        try:
            import torch
            
            # Check each GPU
            for i in range(GPU_COUNT):
                torch.cuda.set_device(i)
                
                # Check memory
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                if mem_allocated / mem_total > 0.9:
                    health['issues'].append(f"GPU {i} memory usage critical: {mem_allocated:.2f}/{mem_total:.2f} GB")
                
                # Check temperature if available
                try:
                    gpu = GPUtil.getGPUs()[i]
                    if gpu.temperature > 80:
                        health['issues'].append(f"GPU {i} temperature high: {gpu.temperature}°C")
                except:
                    pass
                    
        except Exception as e:
            health['issues'].append(f"GPU health check failed: {str(e)}")
    
    # Send alert if issues
    if health['issues'] and monitoring_service:
        from .monitoring import AlertManager
        alert_manager = AlertManager()
        for issue in health['issues']:
            alert_manager.check_alerts({
                'gpu_issue': True,
                'message': issue
            })
    
    return health


# Signal handlers for monitoring
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Log task start"""
    logger.info(f"Task {task.name} [{task_id}] starting")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Log task completion"""
    logger.info(f"Task {task.name} [{task_id}] completed")


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    """Handle task failure"""
    logger.error(f"Task [{task_id}] failed: {exception}")


@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Initialize worker with GPU detection"""
    logger.info(f"Worker ready. GPUs available: {GPU_COUNT}")
    if GPU_COUNT > 0:
        logger.info(f"GPU status: {gpu_manager.get_gpu_status()}")
