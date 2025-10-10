"""
Automated Retraining Service with Airflow/Prefect
==================================================
Place in: automl_platform/retraining_service.py

Implements automated model retraining based on drift detection and performance degradation.
Supports both Airflow and Prefect for workflow orchestration.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import json

from .mlflow_registry import ModelStage

# Airflow imports for scheduling
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.providers.celery.operators.celery import CeleryOperator
    from airflow.utils.dates import days_ago
    from airflow.models import Variable
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    logging.warning("Airflow not installed. Install with: pip install apache-airflow")

# Prefect alternative
try:
    from prefect import flow, task, get_run_logger
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule
    from prefect.task_runners import SequentialTaskRunner, ConcurrentTaskRunner
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    logging.warning("Prefect not installed. Install with: pip install prefect")

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining"""
    # Thresholds
    drift_threshold: float = 0.5
    performance_degradation_threshold: float = 0.1  # 10% degradation
    min_data_points: int = 1000
    min_accuracy_threshold: float = 0.8
    
    # Schedule
    check_frequency: str = "daily"  # daily, weekly, monthly
    retrain_hour: int = 2  # 2 AM
    max_retrain_per_day: int = 5
    
    # Resources
    use_gpu: bool = False
    max_workers: int = 4
    timeout_minutes: int = 120
    
    # Notifications
    notify_on_drift: bool = True
    notify_on_retrain: bool = True
    notification_emails: List[str] = None
    slack_webhook: Optional[str] = None


class RetrainingService:
    """Automated model retraining based on drift and performance"""

    def __init__(self, config, registry, monitor, storage_service):
        self.config = config
        self.registry = registry
        self.monitor = monitor
        self.storage = storage_service
        self.retrain_config = RetrainingConfig()
        
        # Track retraining history
        self.retrain_history = []
        self.last_check = datetime.utcnow()
        
        logger.info("Retraining service initialized")

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        """Safely parse timestamp values from various formats."""

        if value in (None, ""):
            return None

        # Handle numeric timestamps (seconds or milliseconds since epoch)
        try:
            if isinstance(value, (int, float)):
                timestamp = float(value)
                if timestamp > 1e12:  # Likely in milliseconds
                    timestamp /= 1000.0
                return datetime.utcfromtimestamp(timestamp)

            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return None

                # Try ISO format first
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    pass

                # Fall back to numeric parsing
                timestamp = float(value)
                if timestamp > 1e12:
                    timestamp /= 1000.0
                return datetime.utcfromtimestamp(timestamp)
        except (ValueError, TypeError, OSError, OverflowError):
            return None

        return None

    def should_retrain(self, model_name: str) -> Tuple[bool, str, Dict]:
        """Check if model should be retrained with detailed reasoning"""

        reasons = []
        metrics = {}
        
        # Get current production model metadata
        prod_version = None
        metadata_fn = getattr(self.registry, "get_production_model_metadata", None)
        if callable(metadata_fn):
            prod_version = metadata_fn(model_name)
        else:
            get_version_fn = getattr(self.registry, "get_latest_production_version", None)
            if callable(get_version_fn):
                prod_version = get_version_fn(model_name)
            else:
                # Fallback for older registries that only expose the loaded model helper
                prod_model = getattr(self.registry, "get_production_model", lambda *_: None)(model_name)
                if prod_model:
                    metrics['production_model_loaded'] = True

        if prod_version is None and not metrics.get('production_model_loaded'):
            return False, "No production model found", {}

        if prod_version is not None:
            metrics['production_version'] = str(getattr(prod_version, 'version', ''))
            metrics['production_stage'] = getattr(prod_version, 'current_stage', None)

        # Check drift
        drift_score = self.monitor.get_drift_score(model_name)
        metrics['drift_score'] = drift_score
        
        if drift_score and drift_score > self.retrain_config.drift_threshold:
            reasons.append(f"High drift detected: {drift_score:.2f}")
        
        # Check performance degradation using monitor helpers
        perf_metrics = self.monitor.get_performance_metrics(model_name) or {}
        baseline_metrics = perf_metrics.get('baseline_metrics') or {}
        current_metrics = perf_metrics.get('current_metrics') or {}

        # Always retrieve detailed snapshots to avoid AttributeError regressions
        detailed_baseline = self.monitor.get_baseline_performance(model_name) or {}
        detailed_current = self.monitor.get_current_performance(model_name) or {}

        if detailed_baseline:
            baseline_metrics = {**detailed_baseline, **baseline_metrics}
        if detailed_current:
            current_metrics = {**detailed_current, **current_metrics}

        baseline_accuracy = (
            perf_metrics.get('baseline_accuracy')
            if 'baseline_accuracy' in perf_metrics else baseline_metrics.get('accuracy')
        )
        current_accuracy = (
            perf_metrics.get('current_accuracy')
            if 'current_accuracy' in perf_metrics else current_metrics.get('accuracy')
        )

        if baseline_accuracy is not None and current_accuracy is not None:
            degradation = (
                (baseline_accuracy - current_accuracy) / baseline_accuracy
                if baseline_accuracy > 0 else 0
            )

            metrics['baseline_accuracy'] = baseline_accuracy
            metrics['current_accuracy'] = current_accuracy
            metrics['degradation'] = degradation

            if degradation > self.retrain_config.performance_degradation_threshold:
                reasons.append(f"Performance degradation: {degradation:.2%}")

            if current_accuracy < self.retrain_config.min_accuracy_threshold:
                reasons.append(f"Accuracy below threshold: {current_accuracy:.2f}")

        # Track additional degradation insights for observability
        degradation_metrics = {}
        for metric_name in ['auc', 'f1', 'precision', 'recall', 'r2', 'rmse', 'mae']:
            baseline_value = baseline_metrics.get(metric_name)
            current_value = current_metrics.get(metric_name)
            if baseline_value is None or current_value is None:
                continue

            degradation_metrics[f'{metric_name}_degradation'] = baseline_value - current_value

        if degradation_metrics:
            metrics.update(degradation_metrics)
        
        # Check data volume
        new_data_count = self.monitor.get_new_data_count(model_name)
        metrics['new_data_count'] = new_data_count
        
        if new_data_count > self.retrain_config.min_data_points:
            reasons.append(f"Sufficient new data: {new_data_count} samples")
        
        # Check time since last training
        last_training = None
        if prod_version is not None:
            last_training = self._parse_timestamp(
                getattr(prod_version, 'last_updated_timestamp', None)
                or getattr(prod_version, 'creation_timestamp', None)
            )

        if last_training is None:
            model_history = self.registry.get_model_history(model_name, limit=1)
            if model_history:
                history_entry = model_history[0]
                for field in ('created_at', 'creation_timestamp', 'creation_time'):
                    last_training = self._parse_timestamp(history_entry.get(field))
                    if last_training:
                        break

        if last_training:
            days_since_training = (datetime.utcnow() - last_training).days
            metrics['days_since_training'] = days_since_training

            if days_since_training > 30:  # Retrain monthly at minimum
                reasons.append(f"Model is {days_since_training} days old")

        should_retrain = len(reasons) > 0
        reason_text = "; ".join(reasons) if reasons else "No retraining needed"

        return should_retrain, reason_text, metrics
    
    async def retrain_model(self, 
                           model_name: str,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           reason: str = "Scheduled retraining") -> Dict:
        """Retrain a model with new data"""
        
        logger.info(f"Starting retraining for {model_name}: {reason}")
        
        retrain_result = {
            "model_name": model_name,
            "start_time": datetime.utcnow().isoformat(),
            "reason": reason,
            "status": "started"
        }
        
        try:
            # Get current production model configuration
            prod_versions = self.registry.client.get_latest_versions(
                model_name,
                stages=["Production"]
            ) if self.registry.client else []
            
            if not prod_versions:
                # Fallback to local registry
                current_params = {}
                algorithm = "RandomForestClassifier"
            else:
                current_version = prod_versions[0]
                run = self.registry.client.get_run(current_version.run_id)
                current_params = run.data.params
                algorithm = current_params.get('algorithm', 'RandomForestClassifier')
            
            # Import orchestrator for retraining
            from .orchestrator import AutoMLOrchestrator
            
            # Configure for retraining with optimized settings
            retrain_config = self.config
            retrain_config.algorithms = [algorithm]
            retrain_config.hpo_n_iter = 20  # Less HPO for retraining
            retrain_config.max_time_minutes = self.retrain_config.timeout_minutes
            
            # Train new model
            orchestrator = AutoMLOrchestrator(retrain_config)
            orchestrator.fit(X_train, y_train)
            
            # Get best model and metrics
            best_model = orchestrator.best_pipeline
            leaderboard = orchestrator.get_leaderboard()
            
            if not leaderboard.empty:
                best_metrics = leaderboard.iloc[0].to_dict()
            else:
                best_metrics = {}
            
            # Register new version
            new_version = self.registry.register_model(
                model=best_model,
                model_name=model_name,
                metrics=best_metrics,
                params=current_params,
                X_sample=X_train.head(100),
                y_sample=y_train.head(100),
                description=f"Automated retraining - {reason}",
                tags={
                    "retrained": "true",
                    "trigger": "automated",
                    "reason": reason[:100]
                }
            )
            
            # Validate new model before promotion
            validation_passed = await self._validate_model(
                best_model, X_train, y_train, best_metrics
            )
            
            if validation_passed:
                # Promote to staging for further validation
                self.registry.promote_model(
                    model_name,
                    new_version.version,
                    ModelStage.STAGING
                )
                
                retrain_result["status"] = "success"
                retrain_result["new_version"] = new_version.version
                retrain_result["metrics"] = best_metrics
                retrain_result["promoted_to"] = "staging"
            else:
                retrain_result["status"] = "validation_failed"
                retrain_result["error"] = "Model did not pass validation"
            
        except Exception as e:
            logger.error(f"Retraining failed for {model_name}: {e}")
            retrain_result["status"] = "failed"
            retrain_result["error"] = str(e)
        
        retrain_result["end_time"] = datetime.utcnow().isoformat()
        
        # Store in history
        self.retrain_history.append(retrain_result)
        
        # Send notification
        if self.retrain_config.notify_on_retrain:
            await self._send_notification(retrain_result)
        
        logger.info(f"Retraining completed for {model_name}: {retrain_result['status']}")
        
        return retrain_result
    
    async def _validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                             metrics: Dict) -> bool:
        """Validate retrained model meets quality standards"""
        
        # Check minimum performance thresholds
        accuracy = metrics.get('accuracy', 0)
        if accuracy < self.retrain_config.min_accuracy_threshold:
            logger.warning(f"Model accuracy {accuracy:.2f} below threshold")
            return False
        
        # Check for overfitting
        cv_score = metrics.get('cv_score', 0)
        train_score = metrics.get('train_score', accuracy)
        
        if train_score - cv_score > 0.1:  # 10% gap indicates overfitting
            logger.warning(f"Potential overfitting detected: train={train_score:.2f}, cv={cv_score:.2f}")
            return False
        
        # Check prediction sanity
        try:
            sample_predictions = model.predict(X.head(10))
            if len(sample_predictions) != 10:
                return False
        except:
            return False
        
        return True
    
    async def _send_notification(self, result: Dict):
        """Send notification about retraining result"""
        
        message = f"""
        Model Retraining {'Completed' if result['status'] == 'success' else 'Failed'}
        
        Model: {result['model_name']}
        Reason: {result['reason']}
        Status: {result['status']}
        """
        
        if result['status'] == 'success':
            message += f"""
        New Version: {result.get('new_version', 'N/A')}
        Promoted To: {result.get('promoted_to', 'N/A')}
        """
        else:
            message += f"""
        Error: {result.get('error', 'Unknown error')}
        """
        
        # Send to Slack if configured
        if self.retrain_config.slack_webhook:
            requests_module = requests
            if requests_module is None:
                try:
                    import requests as requests_module  # type: ignore
                except ImportError:
                    requests_module = None

            if requests_module is not None:
                try:
                    requests_module.post(
                        self.retrain_config.slack_webhook,
                        json={"text": message}
                    )
                except Exception:
                    pass

        # Log notification
        logger.info(f"Notification: {message}")
    
    def create_retraining_schedule(self):
        """Create automated retraining schedule using available framework"""
        
        if AIRFLOW_AVAILABLE:
            return self._create_airflow_dag()
        elif PREFECT_AVAILABLE:
            return self._create_prefect_flow()
        else:
            logger.warning("No scheduling framework available. Install Airflow or Prefect.")
            return None
    
    def _create_airflow_dag(self):
        """Create Airflow DAG for automated retraining"""
        
        # Define schedule based on frequency
        schedule_interval = {
            "daily": "@daily",
            "weekly": "@weekly", 
            "monthly": "@monthly"
        }.get(self.retrain_config.check_frequency, "@daily")
        
        default_args = {
            'owner': 'automl',
            'depends_on_past': False,
            'start_date': days_ago(1),
            'email': self.retrain_config.notification_emails or [],
            'email_on_failure': True,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }
        
        dag = DAG(
            'model_retraining',
            default_args=default_args,
            description='Automated model retraining based on drift and performance',
            schedule_interval=schedule_interval,
            catchup=False,
            max_active_runs=1,
            tags=['mlops', 'retraining']
        )
        
        def check_models_for_retraining(**context):
            """Check all models and identify which need retraining"""
            models_to_retrain = []
            
            # Get all registered models
            all_models = self.registry.client.list_registered_models() if self.registry.client else []
            
            for model in all_models:
                should_retrain, reason, metrics = self.should_retrain(model.name)
                
                if should_retrain:
                    models_to_retrain.append({
                        "model_name": model.name,
                        "reason": reason,
                        "metrics": metrics
                    })
                    
                    logger.info(f"Model {model.name} needs retraining: {reason}")
            
            # Store in XCom for next task
            context['task_instance'].xcom_push(key='models_to_retrain', value=models_to_retrain)
            
            return len(models_to_retrain)
        
        def retrain_model_task(**context):
            """Retrain models identified in previous task"""
            models_to_retrain = context['task_instance'].xcom_pull(
                task_ids='check_models',
                key='models_to_retrain'
            )
            
            if not models_to_retrain:
                logger.info("No models need retraining")
                return
            
            # Limit retraining to max per day
            models_to_retrain = models_to_retrain[:self.retrain_config.max_retrain_per_day]
            
            results = []
            for model_info in models_to_retrain:
                try:
                    # Load training data from storage
                    X_train, y_train = self.storage.load_training_data(
                        model_info['model_name']
                    )
                    
                    # Retrain model
                    import asyncio
                    result = asyncio.run(self.retrain_model(
                        model_info['model_name'],
                        X_train,
                        y_train,
                        model_info['reason']
                    ))
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to retrain {model_info['model_name']}: {e}")
                    results.append({
                        "model_name": model_info['model_name'],
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Store results
            context['task_instance'].xcom_push(key='retrain_results', value=results)
            
            return results
        
        def validate_and_promote(**context):
            """Validate retrained models and promote if successful"""
            results = context['task_instance'].xcom_pull(
                task_ids='retrain_models',
                key='retrain_results'
            )
            
            if not results:
                return
            
            promoted = []
            for result in results:
                if result['status'] == 'success':
                    model_name = result['model_name']
                    new_version = result['new_version']
                    
                    # Run A/B test or directly promote based on config
                    if getattr(self.config, 'enable_ab_testing', False):
                        # Create A/B test
                        from .mlflow_registry import ABTestingService
                        ab_service = ABTestingService(self.registry)
                        
                        # Get current production version
                        prod_versions = self.registry.client.get_latest_versions(
                            model_name,
                            stages=["Production"]
                        )
                        
                        if prod_versions:
                            test_id = ab_service.create_ab_test(
                                model_name,
                                prod_versions[0].version,
                                new_version,
                                traffic_split=0.1
                            )
                            
                            promoted.append({
                                "model_name": model_name,
                                "action": "ab_test_created",
                                "test_id": test_id
                            })
                    else:
                        # Direct promotion to production
                        success = self.registry.promote_model(
                            model_name,
                            new_version,
                            ModelStage.PRODUCTION
                        )
                        
                        if success:
                            promoted.append({
                                "model_name": model_name,
                                "action": "promoted_to_production",
                                "version": new_version
                            })
            
            return promoted
        
        # Define tasks
        check_task = PythonOperator(
            task_id='check_models',
            python_callable=check_models_for_retraining,
            dag=dag,
        )
        
        retrain_task = PythonOperator(
            task_id='retrain_models',
            python_callable=retrain_model_task,
            dag=dag,
            pool='model_training_pool',  # Use resource pool
            pool_slots=self.retrain_config.max_workers,
        )
        
        validate_task = PythonOperator(
            task_id='validate_and_promote',
            python_callable=validate_and_promote,
            dag=dag,
        )
        
        # Set dependencies
        check_task >> retrain_task >> validate_task
        
        logger.info(f"Created Airflow DAG for model retraining with schedule: {schedule_interval}")
        
        return dag
    
    def _create_prefect_flow(self):
        """Create Prefect flow for automated retraining"""
        
        @task(name="Check Models for Retraining", retries=2)
        def check_models():
            """Check which models need retraining"""
            logger = get_run_logger()
            models_to_retrain = []
            
            # Get all registered models from registry
            if hasattr(self.registry, 'local_registry'):
                all_models = list(self.registry.local_registry.keys())
            else:
                all_models = []
            
            for model_name in all_models:
                should_retrain, reason, metrics = self.should_retrain(model_name)
                
                if should_retrain:
                    models_to_retrain.append({
                        "model_name": model_name,
                        "reason": reason,
                        "metrics": metrics
                    })
                    
                    logger.info(f"Model {model_name} needs retraining: {reason}")
            
            return models_to_retrain
        
        @task(name="Retrain Model", retries=1)
        async def retrain_model_task(model_info: Dict):
            """Retrain a specific model"""
            logger = get_run_logger()
            
            try:
                # Load training data
                X_train, y_train = self.storage.load_training_data(
                    model_info['model_name']
                )
                
                # Retrain
                result = await self.retrain_model(
                    model_info['model_name'],
                    X_train,
                    y_train,
                    model_info['reason']
                )
                
                logger.info(f"Retrained {model_info['model_name']}: {result['status']}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to retrain {model_info['model_name']}: {e}")
                return {
                    "model_name": model_info['model_name'],
                    "status": "failed",
                    "error": str(e)
                }
        
        @task(name="Validate and Promote")
        def validate_and_promote(results: List[Dict]):
            """Validate and promote successful retraining"""
            logger = get_run_logger()
            promoted = []
            
            for result in results:
                if result['status'] == 'success':
                    # Promote to production or create A/B test
                    model_name = result['model_name']
                    new_version = result['new_version']
                    
                    success = self.registry.promote_model(
                        model_name,
                        new_version,
                        ModelStage.PRODUCTION
                    )
                    
                    if success:
                        promoted.append(model_name)
                        logger.info(f"Promoted {model_name} v{new_version} to production")
            
            return promoted
        
        @flow(name="Model Retraining Pipeline",
              task_runner=ConcurrentTaskRunner())
        async def retraining_flow():
            """Main retraining flow"""
            logger = get_run_logger()
            logger.info("Starting model retraining check")
            
            # Check which models need retraining
            models_to_retrain = await check_models()
            
            if not models_to_retrain:
                logger.info("No models need retraining")
                return
            
            # Limit to max retraining per day
            models_to_retrain = models_to_retrain[:self.retrain_config.max_retrain_per_day]
            
            # Retrain models in parallel
            results = []
            for model_info in models_to_retrain:
                result = await retrain_model_task(model_info)
                results.append(result)
            
            # Validate and promote
            promoted = await validate_and_promote(results)
            
            logger.info(f"Retraining complete. Promoted {len(promoted)} models")
            
            return {
                "retrained": len(results),
                "promoted": len(promoted),
                "results": results
            }
        
        # Create deployment with schedule
        schedule = {
            "daily": CronSchedule(cron=f"0 {self.retrain_config.retrain_hour} * * *"),
            "weekly": CronSchedule(cron=f"0 {self.retrain_config.retrain_hour} * * 0"),
            "monthly": CronSchedule(cron=f"0 {self.retrain_config.retrain_hour} 1 * *")
        }.get(self.retrain_config.check_frequency, 
              IntervalSchedule(interval=timedelta(days=1)))
        
        deployment = Deployment.build_from_flow(
            flow=retraining_flow,
            name="automated-model-retraining",
            schedule=schedule,
            work_queue_name="ml-retraining-queue",
            tags=["mlops", "retraining"],
            parameters={},
            infra_overrides={
                "env": {
                    "PREFECT_LOGGING_LEVEL": "INFO"
                }
            }
        )
        
        logger.info(f"Created Prefect deployment for model retraining")
        
        return deployment
    
    def get_retraining_history(self, limit: int = 10) -> List[Dict]:
        """Get recent retraining history"""
        
        # Sort by end time descending
        sorted_history = sorted(
            self.retrain_history,
            key=lambda x: x.get('end_time', ''),
            reverse=True
        )
        
        return sorted_history[:limit]
    
    def get_retraining_stats(self) -> Dict:
        """Get retraining statistics"""
        
        total = len(self.retrain_history)
        successful = sum(1 for r in self.retrain_history if r['status'] == 'success')
        failed = sum(1 for r in self.retrain_history if r['status'] == 'failed')
        
        # Calculate average retraining time
        durations = []
        for record in self.retrain_history:
            if 'start_time' in record and 'end_time' in record:
                start = datetime.fromisoformat(record['start_time'])
                end = datetime.fromisoformat(record['end_time'])
                durations.append((end - start).total_seconds())
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_retrainings": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "average_duration_seconds": avg_duration,
            "last_check": self.last_check.isoformat() if self.last_check else None
        }
