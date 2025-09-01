"""
MLOps Service - Model Registry, Versioning, and Automated Retraining
=====================================================================
Place in: automl_platform/mlops_service.py

Implements MLflow integration, model versioning, A/B testing, and automated retraining.
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

# MLflow imports
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models import infer_signature
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Airflow imports for scheduling
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.providers.celery.operators.celery import CeleryOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

# Prefect alternative
try:
    from prefect import flow, task, get_run_logger
    from prefect.deployments import Deployment
    from prefect.orion.schemas.schedules import CronSchedule
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

# ONNX exports
try:
    import onnx
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# PMML exports
try:
    from sklearn2pmml import sklearn2pmml, PMMLPipeline
    PMML_AVAILABLE = True
except ImportError:
    PMML_AVAILABLE = False

from .config import AutoMLConfig
from .monitoring import DriftDetector, ModelMonitor
from .storage import StorageService

logger = logging.getLogger(__name__)


# ============================================================================
# Model Registry with MLflow
# ============================================================================

class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "Development"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class ModelVersion:
    """Model version metadata"""
    model_name: str
    version: int
    run_id: str
    stage: ModelStage
    
    # Metrics
    metrics: Dict[str, float]
    params: Dict[str, Any]
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    created_by: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Performance tracking
    production_metrics: Optional[Dict[str, float]] = None
    drift_score: Optional[float] = None
    
    # A/B testing
    traffic_percentage: float = 0.0
    is_champion: bool = False
    is_challenger: bool = False


class MLflowRegistry:
    """MLflow-based model registry and versioning"""
    
    def __init__(self, config: AutoMLConfig, tracking_uri: Optional[str] = None):
        self.config = config
        
        if MLFLOW_AVAILABLE:
            # Set tracking URI
            self.tracking_uri = tracking_uri or config.mlflow_tracking_uri or "http://localhost:5000"
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Initialize client
            self.client = MlflowClient()
            
            # Set experiment
            experiment_name = f"automl_{config.tenant_id}"
            try:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            except:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                self.experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_name)
        else:
            logger.warning("MLflow not available, using local registry")
            self.client = None
            self.experiment_id = None
    
    def register_model(self, 
                      model: Any,
                      model_name: str,
                      metrics: Dict[str, float],
                      params: Dict[str, Any],
                      X_sample: Optional[pd.DataFrame] = None,
                      y_sample: Optional[pd.Series] = None,
                      description: str = "",
                      tags: Dict[str, str] = None) -> ModelVersion:
        """Register a model in MLflow"""
        
        if not MLFLOW_AVAILABLE:
            # Fallback to local storage
            return self._register_local(model, model_name, metrics, params)
        
        with mlflow.start_run() as run:
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Infer signature if samples provided
            signature = None
            if X_sample is not None and y_sample is not None:
                signature = infer_signature(X_sample, y_sample)
            
            # Log model based on type
            model_type = type(model).__name__
            
            if "sklearn" in str(type(model).__module__):
                mlflow.sklearn.log_model(
                    model, 
                    "model",
                    signature=signature,
                    registered_model_name=model_name
                )
            elif "xgboost" in str(type(model).__module__):
                mlflow.xgboost.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=model_name
                )
            elif "lightgbm" in str(type(model).__module__):
                mlflow.lightgbm.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=model_name
                )
            elif "torch" in str(type(model).__module__):
                mlflow.pytorch.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=model_name
                )
            else:
                # Generic model logging
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=model,
                    signature=signature,
                    registered_model_name=model_name
                )
            
            run_id = run.info.run_id
        
        # Get version number
        versions = self.client.search_model_versions(f"name='{model_name}'")
        version_number = len(versions)
        
        # Create ModelVersion object
        model_version = ModelVersion(
            model_name=model_name,
            version=version_number,
            run_id=run_id,
            stage=ModelStage.DEVELOPMENT,
            metrics=metrics,
            params=params,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by=self.config.user_id or "system",
            description=description,
            tags=tags or {}
        )
        
        logger.info(f"Registered model {model_name} version {version_number}")
        
        return model_version
    
    def promote_model(self, model_name: str, version: int, 
                     target_stage: ModelStage) -> bool:
        """Promote model to a different stage"""
        
        if not MLFLOW_AVAILABLE:
            return False
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=target_stage.value
            )
            
            logger.info(f"Promoted {model_name} v{version} to {target_stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def get_production_model(self, model_name: str) -> Optional[Any]:
        """Get current production model"""
        
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            # Get production version
            versions = self.client.get_latest_versions(
                model_name, 
                stages=[ModelStage.PRODUCTION.value]
            )
            
            if not versions:
                return None
            
            latest_version = versions[0]
            
            # Load model
            model_uri = f"models:/{model_name}/{latest_version.version}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None
    
    def compare_models(self, model_name: str, version1: int, version2: int) -> Dict:
        """Compare two model versions"""
        
        if not MLFLOW_AVAILABLE:
            return {}
        
        try:
            # Get run IDs for versions
            v1 = self.client.get_model_version(model_name, version1)
            v2 = self.client.get_model_version(model_name, version2)
            
            # Get metrics for both runs
            run1 = self.client.get_run(v1.run_id)
            run2 = self.client.get_run(v2.run_id)
            
            comparison = {
                "version1": {
                    "version": version1,
                    "metrics": run1.data.metrics,
                    "params": run1.data.params,
                    "stage": v1.current_stage
                },
                "version2": {
                    "version": version2,
                    "metrics": run2.data.metrics,
                    "params": run2.data.params,
                    "stage": v2.current_stage
                },
                "metric_diff": {}
            }
            
            # Calculate metric differences
            for metric in run1.data.metrics:
                if metric in run2.data.metrics:
                    diff = run2.data.metrics[metric] - run1.data.metrics[metric]
                    comparison["metric_diff"][metric] = {
                        "absolute": diff,
                        "relative": (diff / run1.data.metrics[metric]) * 100 if run1.data.metrics[metric] != 0 else 0
                    }
