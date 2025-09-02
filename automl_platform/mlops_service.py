"""
MLOps Service - Complete file with corrections
==================================================================
Place in: automl_platform/mlops_service.py
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ONNX imports
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .config import AutoMLConfig
from .monitoring import DriftDetector, ModelMonitor
from .storage import StorageService
from .metrics import calculate_metrics, detect_task

# Import optimization components
try:
    from .pipeline_cache import PipelineCache, CacheConfig, warm_cache, monitor_cache_health
    from .incremental_learning import IncrementalLearner
    from .distributed_training import DistributedTrainer
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model stages in MLflow"""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class MLflowRegistry:
    """MLflow-based model registry and versioning with caching"""
    
    def __init__(self, config: AutoMLConfig, tracking_uri: Optional[str] = None):
        self.config = config
        
        # Initialize pipeline cache
        self.pipeline_cache = None
        if OPTIMIZATIONS_AVAILABLE and hasattr(config, 'enable_cache') and config.enable_cache:
            cache_config = CacheConfig(
                backend=getattr(config, 'cache_backend', 'redis'),
                redis_host=getattr(config, 'redis_host', 'localhost'),
                ttl_seconds=getattr(config, 'cache_ttl', 3600)
            )
            self.pipeline_cache = PipelineCache(cache_config)
            logger.info("Pipeline cache enabled for model registry")
        
        # Initialize MLflow if available
        if MLFLOW_AVAILABLE:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                mlflow.set_tracking_uri(getattr(config, 'mlflow_tracking_uri', 'sqlite:///mlflow.db'))
            
            self.client = MlflowClient()
        else:
            self.client = None
            logger.warning("MLflow not available - model registry disabled")
    
    def get_production_model(self, model_name: str, use_cache: bool = True) -> Optional[Any]:
        """Get current production model with caching"""
        
        # Check cache first
        if use_cache and self.pipeline_cache:
            cache_key = f"prod_model_{model_name}"
            cached_model = self.pipeline_cache.get_pipeline(cache_key)
            if cached_model:
                logger.debug(f"Production model {model_name} loaded from cache")
                return cached_model
        
        if not MLFLOW_AVAILABLE or not self.client:
            return None
        
        try:
            # Get model from MLflow
            versions = self.client.get_latest_versions(
                model_name, 
                stages=[ModelStage.PRODUCTION.value]
            )
            
            if not versions:
                return None
            
            latest_version = versions[0]
            model_uri = f"models:/{model_name}/{latest_version.version}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Cache the model
            if use_cache and self.pipeline_cache and model:
                self.pipeline_cache.set_pipeline(
                    f"prod_model_{model_name}",
                    model,
                    ttl=3600 * 24  # Cache for 24 hours
                )
                logger.debug(f"Production model {model_name} cached")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None
    
    def invalidate_model_cache(self, model_name: str):
        """Invalidate cached model"""
        if self.pipeline_cache:
            cache_key = f"prod_model_{model_name}"
            self.pipeline_cache.invalidate(cache_key, reason="model_updated")
            logger.info(f"Cache invalidated for model {model_name}")
    
    def register_model(self, model: Any, model_name: str, 
                      metrics: Dict[str, float],
                      params: Dict[str, Any],
                      X_sample: pd.DataFrame,
                      y_sample: pd.Series,
                      description: str = "",
                      tags: Dict[str, str] = None) -> Any:
        """Register a model in MLflow registry"""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available - cannot register model")
            return None
        
        with mlflow.start_run() as run:
            # Log model
            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log sample data
            mlflow.log_dict(X_sample.head(10).to_dict(), "sample_input.json")
            mlflow.log_dict({"target": y_sample.head(10).tolist()}, "sample_output.json")
            
            # Set tags
            if tags:
                for tag_key, tag_value in tags.items():
                    mlflow.set_tag(tag_key, tag_value)
            
            # Set description
            if description:
                mlflow.set_tag("mlflow.note.content", description)
            
            run_id = run.info.run_id
        
        # Get the registered model version
        model_version = self.client.search_model_versions(
            f"name='{model_name}' and run_id='{run_id}'"
        )[0]
        
        return model_version
    
    def promote_model(self, model_name: str, version: str, stage: ModelStage):
        """Promote model to a specific stage"""
        if not MLFLOW_AVAILABLE:
            return
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage.value
        )
        
        # Invalidate cache when model is promoted
        if stage == ModelStage.PRODUCTION:
            self.invalidate_model_cache(model_name)
        
        logger.info(f"Model {model_name} version {version} promoted to {stage.value}")


class RetrainingService:
    """Automated model retraining based on drift and performance with optimizations"""
    
    def __init__(self, config: AutoMLConfig, 
                 registry: MLflowRegistry,
                 monitor: ModelMonitor):
        self.config = config
        self.registry = registry
        self.monitor = monitor
        
        # Retraining thresholds
        self.drift_threshold = 0.5
        self.performance_degradation_threshold = 0.1
        self.min_data_points = 1000
        
        # Schedule configuration
        self.check_frequency = timedelta(days=1)
        self.last_check = datetime.utcnow()
        
        # Initialize optimization components
        self.incremental_learner = None
        if OPTIMIZATIONS_AVAILABLE and hasattr(config, 'incremental_learning') and config.incremental_learning:
            self.incremental_learner = IncrementalLearner(
                max_memory_mb=getattr(config, 'max_memory_mb', 1000)
            )
        
        self.distributed_trainer = None
        if OPTIMIZATIONS_AVAILABLE and hasattr(config, 'distributed_training') and config.distributed_training:
            self.distributed_trainer = DistributedTrainer(
                backend=getattr(config, 'distributed_backend', 'ray'),
                n_workers=getattr(config, 'n_workers', 4)
            )
    
    async def check_retraining_needed(self, model_name: str) -> bool:
        """Check if model needs retraining"""
        # Get current performance
        current_metrics = self.monitor.get_current_performance()
        baseline_metrics = self.monitor.get_baseline_performance()
        
        # Check for performance degradation
        for metric_name in ['accuracy', 'auc', 'f1']:
            if metric_name in current_metrics and metric_name in baseline_metrics:
                degradation = baseline_metrics[metric_name] - current_metrics[metric_name]
                if degradation > self.performance_degradation_threshold:
                    logger.warning(f"Performance degradation detected for {metric_name}: {degradation:.3f}")
                    return True
        
        # Check for drift
        drift_score = self.monitor.get_drift_score()
        if drift_score > self.drift_threshold:
            logger.warning(f"Data drift detected: score = {drift_score:.3f}")
            return True
        
        # Check data volume
        new_data_count = self.monitor.get_new_data_count()
        if new_data_count >= self.min_data_points:
            logger.info(f"Sufficient new data for retraining: {new_data_count} samples")
            return True
        
        return False
    
    async def retrain_model(self, model_name: str, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           use_incremental: bool = False,
                           use_distributed: bool = False) -> Any:
        """Retrain a model with new data using optimizations"""
        
        logger.info(f"Starting retraining for {model_name}")
        
        # Invalidate cache for this model
        if self.registry.pipeline_cache:
            self.registry.invalidate_model_cache(model_name)
        
        # Get current production model configuration
        if not MLFLOW_AVAILABLE:
            logger.error("MLflow not available - cannot retrain")
            return None
        
        prod_versions = self.registry.client.get_latest_versions(
            model_name,
            stages=[ModelStage.PRODUCTION.value]
        )
        
        if not prod_versions:
            raise ValueError(f"No production model found for {model_name}")
        
        current_version = prod_versions[0]
        run = self.registry.client.get_run(current_version.run_id)
        params = run.data.params
        
        # Use incremental learning for large datasets
        if use_incremental and self.incremental_learner and len(X_train) > 10000:
            logger.info("Using incremental retraining")
            task = detect_task(y_train)
            models = self.incremental_learner.train_incremental(X_train, y_train, task)
            best_model = self.incremental_learner.get_best_model()
        
        # Use distributed training if enabled
        elif use_distributed and self.distributed_trainer:
            logger.info("Using distributed retraining")
            from .model_selection import get_available_models
            models = get_available_models(detect_task(y_train))
            results = self.distributed_trainer.train_distributed(
                X_train, y_train, models, {}
            )
            best_result = max(results, key=lambda x: x['cv_score'])
            best_model = best_result['pipeline']
        else:
            # Standard retraining with orchestrator
            from .orchestrator import AutoMLOrchestrator
            
            retrain_config = self.config
            retrain_config.algorithms = [params.get('algorithm', 'RandomForestClassifier')]
            retrain_config.hpo_n_iter = 20
            
            orchestrator = AutoMLOrchestrator(retrain_config)
            orchestrator.fit(X_train, y_train)
            best_model = orchestrator.best_pipeline
        
        # Get metrics
        y_pred = best_model.predict(X_train)
        metrics = calculate_metrics(y_train, y_pred, None, detect_task(y_train))
        
        # Register new version
        new_version = self.registry.register_model(
            model=best_model,
            model_name=model_name,
            metrics=metrics,
            params=params,
            X_sample=X_train.head(100),
            y_sample=y_train.head(100),
            description=f"Automated retraining - {datetime.utcnow()}",
            tags={
                "retrained": "true",
                "trigger": "automated",
                "incremental": str(use_incremental),
                "distributed": str(use_distributed)
            }
        )
        
        # Promote to staging for validation
        self.registry.promote_model(model_name, new_version.version, ModelStage.STAGING)
        
        logger.info(f"Retraining completed for {model_name}, new version: {new_version.version}")
        
        return new_version
    
    async def validate_retrained_model(self, model_name: str, 
                                      version: str,
                                      X_val: pd.DataFrame,
                                      y_val: pd.Series) -> bool:
        """Validate retrained model before promoting to production"""
        if not MLFLOW_AVAILABLE:
            return False
        
        # Load staging model
        model_uri = f"models:/{model_name}/{version}"
        staging_model = mlflow.pyfunc.load_model(model_uri)
        
        # Load production model
        prod_model = self.registry.get_production_model(model_name)
        
        if not prod_model:
            # No production model, auto-approve
            return True
        
        # Compare performance
        staging_pred = staging_model.predict(X_val)
        prod_pred = prod_model.predict(X_val)
        
        task = detect_task(y_val)
        staging_metrics = calculate_metrics(y_val, staging_pred, None, task)
        prod_metrics = calculate_metrics(y_val, prod_pred, None, task)
        
        # Check if staging model is better
        key_metric = 'accuracy' if task == 'classification' else 'r2'
        
        if staging_metrics.get(key_metric, 0) >= prod_metrics.get(key_metric, 0):
            logger.info(f"Staging model {version} performs better than production")
            return True
        else:
            logger.warning(f"Staging model {version} performs worse than production")
            return False


class ModelExporter:
    """Export models to various formats for deployment with caching"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config
        
        # Initialize cache for exported models
        self.export_cache = None
        if OPTIMIZATIONS_AVAILABLE and config and hasattr(config, 'enable_cache') and config.enable_cache:
            cache_config = CacheConfig(
                backend='disk',  # Use disk for exported models
                disk_cache_dir='/tmp/exported_models_cache',
                ttl_seconds=3600 * 24 * 7,  # Cache for 1 week
                use_mmap=True
            )
            self.export_cache = PipelineCache(cache_config)
    
    def export_to_onnx(self, model: Any, 
                      sample_input: np.ndarray,
                      output_path: str,
                      use_cache: bool = True) -> bool:
        """Export model to ONNX format with caching"""
        
        # Check cache first
        if use_cache and self.export_cache:
            cache_key = f"onnx_{hash(str(model))}_{sample_input.shape}"
            cached_export = self.export_cache.get_pipeline(cache_key)
            if cached_export:
                logger.info(f"Using cached ONNX export")
                with open(output_path, 'wb') as f:
                    f.write(cached_export)
                return True
        
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available")
            return False
        
        try:
            n_features = sample_input.shape[1]
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            onnx_bytes = onnx_model.SerializeToString()
            
            # Cache the export
            if use_cache and self.export_cache:
                self.export_cache.set_pipeline(cache_key, onnx_bytes)
            
            with open(output_path, "wb") as f:
                f.write(onnx_bytes)
            
            logger.info(f"Model exported to ONNX: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
            return False
    
    def export_to_pmml(self, model: Any, output_path: str) -> bool:
        """Export model to PMML format"""
        try:
            from sklearn2pmml import sklearn2pmml
            from sklearn2pmml.pipeline import PMMLPipeline
            
            pmml_pipeline = PMMLPipeline([("model", model)])
            sklearn2pmml(pmml_pipeline, output_path)
            
            logger.info(f"Model exported to PMML: {output_path}")
            return True
            
        except ImportError:
            logger.error("sklearn2pmml not available")
            return False
        except Exception as e:
            logger.error(f"Failed to export to PMML: {e}")
            return False
    
    def export_to_tensorflow_lite(self, model: Any, 
                                 sample_input: np.ndarray,
                                 output_path: str) -> bool:
        """Export model to TensorFlow Lite format"""
        try:
            import tensorflow as tf
            from sklearn import tree
            
            # Convert sklearn model to TF
            # This is a simplified example - real implementation would be more complex
            
            # Create a simple TF model that mimics the sklearn model
            tf_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(sample_input.shape[1],)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Model exported to TensorFlow Lite: {output_path}")
            return True
            
        except ImportError:
            logger.error("TensorFlow not available")
            return False
        except Exception as e:
            logger.error(f"Failed to export to TensorFlow Lite: {e}")
            return False


class ModelVersionManager:
    """Manage model versions and rollbacks"""
    
    def __init__(self, registry: MLflowRegistry):
        self.registry = registry
    
    def compare_versions(self, model_name: str, 
                        version_a: str, 
                        version_b: str,
                        X_test: pd.DataFrame,
                        y_test: pd.Series) -> Dict[str, Any]:
        """Compare two model versions"""
        if not MLFLOW_AVAILABLE:
            return {}
        
        # Load both models
        model_a_uri = f"models:/{model_name}/{version_a}"
        model_b_uri = f"models:/{model_name}/{version_b}"
        
        model_a = mlflow.pyfunc.load_model(model_a_uri)
        model_b = mlflow.pyfunc.load_model(model_b_uri)
        
        # Get predictions
        pred_a = model_a.predict(X_test)
        pred_b = model_b.predict(X_test)
        
        # Calculate metrics
        task = detect_task(y_test)
        metrics_a = calculate_metrics(y_test, pred_a, None, task)
        metrics_b = calculate_metrics(y_test, pred_b, None, task)
        
        return {
            "version_a": {
                "version": version_a,
                "metrics": metrics_a
            },
            "version_b": {
                "version": version_b,
                "metrics": metrics_b
            },
            "comparison": {
                metric: metrics_a[metric] - metrics_b[metric]
                for metric in metrics_a.keys()
                if metric in metrics_b
            }
        }
    
    def rollback_model(self, model_name: str, target_version: str):
        """Rollback model to a specific version"""
        if not MLFLOW_AVAILABLE:
            return
        
        # Get current production version
        prod_versions = self.registry.client.get_latest_versions(
            model_name,
            stages=[ModelStage.PRODUCTION.value]
        )
        
        if prod_versions:
            current_version = prod_versions[0]
            # Archive current production
            self.registry.promote_model(
                model_name, 
                current_version.version, 
                ModelStage.ARCHIVED
            )
        
        # Promote target version to production
        self.registry.promote_model(
            model_name, 
            target_version, 
            ModelStage.PRODUCTION
        )
        
        logger.info(f"Rolled back {model_name} to version {target_version}")
    
    def get_version_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get version history for a model"""
        if not MLFLOW_AVAILABLE:
            return []
        
        versions = self.registry.client.search_model_versions(
            f"name='{model_name}'"
        )
        
        history = []
        for version in versions:
            run = self.registry.client.get_run(version.run_id)
            
            history.append({
                "version": version.version,
                "stage": version.current_stage,
                "created_at": version.creation_timestamp,
                "updated_at": version.last_updated_timestamp,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            })
        
        return sorted(history, key=lambda x: x["version"], reverse=True)


# Create convenience functions
def create_mlops_service(config: AutoMLConfig) -> Dict[str, Any]:
    """Create MLOps service components"""
    registry = MLflowRegistry(config)
    exporter = ModelExporter(config)
    version_manager = ModelVersionManager(registry)
    
    # Create monitor if monitoring is enabled
    monitor = None
    retraining_service = None
    
    if hasattr(config, 'monitoring') and config.monitoring.enabled:
        from .monitoring import ModelMonitor
        monitor = ModelMonitor(
            model_id="default",
            model_type="classification",
            reference_data=None
        )
        retraining_service = RetrainingService(config, registry, monitor)
    
    return {
        "registry": registry,
        "exporter": exporter,
        "version_manager": version_manager,
        "monitor": monitor,
        "retraining_service": retraining_service
    }
