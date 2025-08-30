"""
Celery worker for asynchronous AutoML jobs
Implements distributed training with proper isolation
"""

from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    'task_routes': {
        'automl.train.*': {'queue': 'training'},
        'automl.predict.*': {'queue': 'prediction'},
        'automl.llm.*': {'queue': 'llm'},
    },
})

# Import after Celery initialization
from .orchestrator import AutoMLOrchestrator
from .config import AutoMLConfig
from .storage import StorageManager
from .monitoring import MetricsCollector
from .llm_enhanced import LLMEnhancedProcessor

# Initialize services
storage_manager = StorageManager()
metrics_collector = MetricsCollector()

class AutoMLTask(Task):
    """Base task with automatic tracking and error handling."""
    
    def before_start(self, task_id, args, kwargs):
        """Called before task execution."""
        metrics_collector.record_task_start(task_id, self.name)
        logger.info(f"Starting task {task_id}: {self.name}")
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called on successful task completion."""
        metrics_collector.record_task_success(task_id, self.name)
        logger.info(f"Task {task_id} completed successfully")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        metrics_collector.record_task_failure(task_id, self.name, str(exc))
        logger.error(f"Task {task_id} failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        metrics_collector.record_task_retry(task_id, self.name)
        logger.warning(f"Task {task_id} retrying: {exc}")

@app.task(base=AutoMLTask, bind=True, name='automl.train.full_pipeline', max_retries=3)
def train_full_pipeline(self, job_id: str, dataset_url: str, config: Dict[str, Any], 
                        user_id: str, tenant_id: str) -> Dict[str, Any]:
    """
    Execute complete AutoML training pipeline with isolation.
    
    Args:
        job_id: Unique job identifier
        dataset_url: S3/MinIO URL to dataset
        config: Training configuration
        user_id: User identifier for isolation
        tenant_id: Tenant identifier for multi-tenancy
    
    Returns:
        Training results with model URL and metrics
    """
    try:
        # Update job status
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Downloading dataset...'}
        )
        
        # Download dataset from storage
        dataset_path = storage_manager.download_dataset(dataset_url, tenant_id, job_id)
        df = pd.read_csv(dataset_path)
        
        # Data quality check with LLM
        self.update_state(
            state='PROGRESS',
            meta={'current': 10, 'total': 100, 'status': 'Analyzing data quality...'}
        )
        
        llm_processor = LLMEnhancedProcessor()
        quality_report = llm_processor.analyze_data_quality(df)
        
        # Auto data cleaning if enabled
        if config.get('auto_clean', True):
            self.update_state(
                state='PROGRESS',
                meta={'current': 20, 'total': 100, 'status': 'Cleaning data...'}
            )
            df = llm_processor.auto_clean_data(df, quality_report)
        
        # Feature engineering
        self.update_state(
            state='PROGRESS',
            meta={'current': 30, 'total': 100, 'status': 'Engineering features...'}
        )
        
        if config.get('auto_feature_engineering', True):
            df = llm_processor.generate_features(df, config.get('target'))
        
        # Split features and target
        target_col = config['target']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Initialize orchestrator
        automl_config = AutoMLConfig(**config.get('automl_config', {}))
        orchestrator = AutoMLOrchestrator(automl_config)
        
        # Training with progress updates
        self.update_state(
            state='PROGRESS',
            meta={'current': 40, 'total': 100, 'status': 'Training models...'}
        )
        
        # Custom callback for progress
        def progress_callback(current_model, total_models):
            progress = 40 + int((current_model / total_models) * 40)
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': progress, 
                    'total': 100, 
                    'status': f'Training model {current_model}/{total_models}...'
                }
            )
        
        # Train with callback
        orchestrator.fit(X, y, progress_callback=progress_callback)
        
        # Save model to storage
        self.update_state(
            state='PROGRESS',
            meta={'current': 85, 'total': 100, 'status': 'Saving model...'}
        )
        
        model_path = f"/tmp/{job_id}_model.pkl"
        orchestrator.save_pipeline(model_path)
        
        # Upload to MinIO/S3
        model_url = storage_manager.upload_model(
            model_path, 
            tenant_id, 
            job_id,
            metadata={
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'config': config
            }
        )
        
        # Register in MLflow
        import mlflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        
        with mlflow.start_run(run_name=f"automl_{job_id}"):
            mlflow.log_params(config)
            mlflow.log_metrics({
                'best_cv_score': orchestrator.leaderboard[0]['cv_score'],
                'n_models_tested': len(orchestrator.leaderboard)
            })
            mlflow.sklearn.log_model(
                orchestrator.best_pipeline,
                artifact_path="model",
                registered_model_name=f"automl_{tenant_id}_{job_id}"
            )
        
        # Generate report with LLM
        self.update_state(
            state='PROGRESS',
            meta={'current': 95, 'total': 100, 'status': 'Generating report...'}
        )
        
        report = llm_processor.generate_model_report(
            orchestrator.leaderboard,
            orchestrator.feature_importance,
            quality_report
        )
        
        # Prepare results
        results = {
            'job_id': job_id,
            'model_url': model_url,
            'best_model': orchestrator.leaderboard[0]['model'],
            'cv_score': orchestrator.leaderboard[0]['cv_score'],
            'leaderboard': orchestrator.get_leaderboard().to_dict('records'),
            'feature_importance': orchestrator.feature_importance,
            'report': report,
            'quality_report': quality_report,
            'completed_at': datetime.now().isoformat()
        }
        
        # Record metrics
        metrics_collector.record_training_metrics(
            job_id=job_id,
            tenant_id=tenant_id,
            user_id=user_id,
            model_type=results['best_model'],
            cv_score=results['cv_score'],
            n_models=len(orchestrator.leaderboard),
            training_time=(datetime.now() - self.request.started).total_seconds()
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {str(e)}\n{traceback.format_exc()}")
        raise self.retry(exc=e, countdown=60)

@app.task(base=AutoMLTask, bind=True, name='automl.predict.batch', max_retries=2)
def predict_batch(self, job_id: str, model_url: str, data_url: str, 
                  tenant_id: str) -> Dict[str, Any]:
    """
    Batch prediction task.
    
    Args:
        job_id: Prediction job ID
        model_url: S3/MinIO URL to model
        data_url: S3/MinIO URL to prediction data
        tenant_id: Tenant identifier
    
    Returns:
        Prediction results with URL to output file
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Loading model...'}
        )
        
        # Download model
        model_path = storage_manager.download_model(model_url, tenant_id)
        
        # Load pipeline
        from .inference import load_pipeline
        pipeline, metadata = load_pipeline(model_path)
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 30, 'total': 100, 'status': 'Loading data...'}
        )
        
        # Download data
        data_path = storage_manager.download_dataset(data_url, tenant_id, job_id)
        df = pd.read_csv(data_path)
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 50, 'total': 100, 'status': 'Making predictions...'}
        )
        
        # Make predictions
        predictions = pipeline.predict(df)
        
        # Add probabilities if classification
        probabilities = None
        if hasattr(pipeline, 'predict_proba'):
            try:
                probabilities = pipeline.predict_proba(df)
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
        
        output_path = f"/tmp/{job_id}_predictions.csv"
        results_df.to_csv(output_path, index=False)
        
        # Upload results
        results_url = storage_manager.upload_predictions(
            output_path,
            tenant_id,
            job_id
        )
        
        # Record metrics
        metrics_collector.record_prediction_metrics(
            job_id=job_id,
            tenant_id=tenant_id,
            n_predictions=len(predictions),
            prediction_time=(datetime.now() - self.request.started).total_seconds()
        )
        
        return {
            'job_id': job_id,
            'results_url': results_url,
            'n_predictions': len(predictions),
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction failed for job {job_id}: {str(e)}")
        raise self.retry(exc=e, countdown=30)

@app.task(base=AutoMLTask, bind=True, name='automl.llm.analyze', max_retries=2)
def analyze_with_llm(self, job_id: str, data_url: str, analysis_type: str,
                     tenant_id: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    LLM-powered analysis task.
    
    Args:
        job_id: Analysis job ID
        data_url: S3/MinIO URL to data
        analysis_type: Type of analysis (quality, features, insights)
        tenant_id: Tenant identifier
        config: Additional configuration
    
    Returns:
        Analysis results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Loading data...'}
        )
        
        # Download data
        data_path = storage_manager.download_dataset(data_url, tenant_id, job_id)
        df = pd.read_csv(data_path)
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 30, 'total': 100, 'status': f'Performing {analysis_type} analysis...'}
        )
        
        llm_processor = LLMEnhancedProcessor()
        
        if analysis_type == 'quality':
            results = llm_processor.analyze_data_quality(df)
        elif analysis_type == 'features':
            results = llm_processor.suggest_features(df, config.get('target'))
        elif analysis_type == 'insights':
            results = llm_processor.generate_insights(df)
        elif analysis_type == 'cleaning':
            cleaned_df = llm_processor.auto_clean_data(df)
            output_path = f"/tmp/{job_id}_cleaned.csv"
            cleaned_df.to_csv(output_path, index=False)
            cleaned_url = storage_manager.upload_dataset(
                output_path,
                tenant_id,
                f"{job_id}_cleaned"
            )
            results = {
                'cleaned_data_url': cleaned_url,
                'changes_made': llm_processor.get_cleaning_report()
            }
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 90, 'total': 100, 'status': 'Finalizing...'}
        )
        
        return {
            'job_id': job_id,
            'analysis_type': analysis_type,
            'results': results,
            'completed_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"LLM analysis failed for job {job_id}: {str(e)}")
        raise self.retry(exc=e, countdown=30)

@app.task(base=AutoMLTask, name='automl.monitor.drift')
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
        # Download datasets
        ref_path = storage_manager.download_dataset(reference_data_url, tenant_id, "ref")
        curr_path = storage_manager.download_dataset(current_data_url, tenant_id, "curr")
        
        ref_df = pd.read_csv(ref_path)
        curr_df = pd.read_csv(curr_path)
        
        # Calculate drift metrics
        from .monitoring import DriftDetector
        detector = DriftDetector()
        
        drift_report = detector.detect_drift(ref_df, curr_df)
        
        # Record in monitoring system
        metrics_collector.record_drift_metrics(
            tenant_id=tenant_id,
            model_id=model_id,
            drift_score=drift_report['overall_drift_score'],
            drifted_features=drift_report['drifted_features']
        )
        
        # Alert if significant drift
        if drift_report['overall_drift_score'] > 0.5:
            send_drift_alert(tenant_id, model_id, drift_report)
        
        return drift_report
        
    except Exception as e:
        logger.error(f"Drift monitoring failed: {str(e)}")
        raise

@app.task(base=AutoMLTask, name='automl.retrain.scheduled')
def scheduled_retrain(tenant_id: str, model_id: str, 
                      dataset_url: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scheduled model retraining.
    
    Args:
        tenant_id: Tenant identifier
        model_id: Model to retrain
        dataset_url: New training data URL
        config: Training configuration
    
    Returns:
        New model information
    """
    try:
        # Generate new job ID
        job_id = f"retrain_{model_id}_{int(time.time())}"
        
        # Trigger training
        result = train_full_pipeline.apply_async(
            args=[job_id, dataset_url, config, 'system', tenant_id],
            queue='training'
        )
        
        return {
            'job_id': job_id,
            'task_id': result.id,
            'status': 'scheduled',
            'scheduled_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Scheduled retrain failed: {str(e)}")
        raise

# Celery beat schedule for periodic tasks
from celery.schedules import crontab

app.conf.beat_schedule = {
    'cleanup-old-jobs': {
        'task': 'automl.maintenance.cleanup',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    'monitor-system-health': {
        'task': 'automl.maintenance.health_check',
        'schedule': 60.0,  # Every minute
    },
}

def send_drift_alert(tenant_id: str, model_id: str, drift_report: Dict[str, Any]):
    """Send drift alert via configured channels."""
    # Implement notification logic (email, Slack, etc.)
    logger.warning(f"Drift detected for tenant {tenant_id}, model {model_id}: {drift_report}")

# Signal handlers for monitoring
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