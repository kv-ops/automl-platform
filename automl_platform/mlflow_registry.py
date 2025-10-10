"""
MLflow Model Registry Integration
==================================
Place in: automl_platform/mlflow_registry.py

Provides MLflow model registry with A/B testing support.
"""

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models import ModelSignature, infer_signature
from mlflow.entities.model_registry import ModelVersion
from typing import Dict, Any, Optional, List
import logging
import os
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np

# Import A/B testing service
from .ab_testing import ABTestingService

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model stages in MLflow"""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"
    DEVELOPMENT = "Development"


class MLflowRegistry:
    """MLflow model registry management."""
    
    def __init__(self, config):
        """
        Initialize MLflow registry.
        
        Args:
            config: AutoML configuration
        """
        self.config = config
        
        # Set tracking URI
        tracking_uri = getattr(config, 'mlflow_tracking_uri', None) or os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Initialize client
        self.client = MlflowClient()
        
        # Set experiment
        experiment_name = getattr(config, 'mlflow_experiment_name', 'automl_experiments')
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"MLflow registry initialized with tracking URI: {tracking_uri}")
    
    def register_model(self,
                       model: Any,
                       model_name: str,
                       metrics: Dict[str, float] = None,
                       params: Dict[str, Any] = None,
                       X_sample: pd.DataFrame = None,
                       y_sample: pd.Series = None,
                       description: str = "",
                       tags: Dict[str, str] = None) -> Any:
        """
        Register model in MLflow.
        
        Args:
            model: Trained model
            model_name: Name for the model
            metrics: Model metrics
            params: Model parameters
            X_sample: Sample input for signature
            y_sample: Sample output for signature
            description: Model description
            tags: Additional tags
            
        Returns:
            Model version object
        """
        with mlflow.start_run() as run:
            # Log parameters
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, value)
            
            # Log metrics
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
            
            # Log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Infer signature if sample data provided
            signature = None
            if X_sample is not None and y_sample is not None:
                try:
                    predictions = model.predict(X_sample)
                    signature = infer_signature(X_sample, predictions)
                except:
                    logger.warning("Could not infer model signature")
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=model_name
            )
            
            # Get run ID
            run_id = run.info.run_id
        
        # Get latest version
        latest_versions = self.client.get_latest_versions(
            model_name,
            stages=["None"]
        )

        if latest_versions:
            latest_version = latest_versions[0]
            # Update description
            self.client.update_model_version(
                name=model_name,
                version=latest_version.version,
                description=description
            )

            logger.info(f"Model {model_name} version {latest_version.version} registered")

            # Store additional metadata
            latest_version.run_id = run_id
            latest_version.model_name = model_name
            latest_version.stage = ModelStage.NONE

            return latest_version

        logger.warning(
            "Model registration for %s (run %s) did not return any version",
            model_name,
            run_id,
        )

        return None
    
    def promote_model(self,
                     model_name: str,
                     version: int,
                     stage: ModelStage) -> bool:
        """
        Promote model to different stage.
        
        Args:
            model_name: Model name
            version: Model version
            stage: Target stage
            
        Returns:
            Success status
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage.value if isinstance(stage, ModelStage) else stage
            )
            
            logger.info(f"Model {model_name} version {version} promoted to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def get_model_history(self,
                         model_name: str,
                         limit: int = 10) -> List[Dict]:
        """
        Get model version history.
        
        Args:
            model_name: Model name
            limit: Maximum versions to return
            
        Returns:
            List of model version info
        """
        try:
            versions = self.client.search_model_versions(
                f"name='{model_name}'",
                max_results=limit,
                order_by=["version_number DESC"]
            )
            
            history = []
            for version in versions:
                # Get run details
                run = self.client.get_run(version.run_id)
                
                history.append({
                    'version': version.version,
                    'stage': version.current_stage,
                    'created_at': version.creation_timestamp,
                    'updated_at': version.last_updated_timestamp,
                    'description': version.description,
                    'run_id': version.run_id,
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'tags': run.data.tags
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get model history: {e}")
            return []
    
    def compare_models(self,
                      model_name: str,
                      version1: int,
                      version2: int) -> Dict:
        """
        Compare two model versions.
        
        Args:
            model_name: Model name
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        try:
            # Get model versions
            v1 = self.client.get_model_version(model_name, version1)
            v2 = self.client.get_model_version(model_name, version2)
            
            # Get runs
            run1 = self.client.get_run(v1.run_id)
            run2 = self.client.get_run(v2.run_id)
            
            comparison = {
                'version1': {
                    'version': version1,
                    'stage': v1.current_stage,
                    'metrics': run1.data.metrics,
                    'params': run1.data.params,
                    'created_at': v1.creation_timestamp
                },
                'version2': {
                    'version': version2,
                    'stage': v2.current_stage,
                    'metrics': run2.data.metrics,
                    'params': run2.data.params,
                    'created_at': v2.creation_timestamp
                },
                'metrics_diff': {}
            }
            
            # Calculate metric differences
            for metric in run1.data.metrics:
                if metric in run2.data.metrics:
                    diff = run2.data.metrics[metric] - run1.data.metrics[metric]
                    comparison['metrics_diff'][metric] = {
                        'diff': diff,
                        'pct_change': (diff / run1.data.metrics[metric] * 100) if run1.data.metrics[metric] != 0 else 0
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {}
    
    def rollback_model(self,
                      model_name: str,
                      target_version: int) -> bool:
        """
        Rollback to previous model version.
        
        Args:
            model_name: Model name
            target_version: Version to rollback to
            
        Returns:
            Success status
        """
        try:
            # Get current production version
            prod_versions = self.client.get_latest_versions(
                model_name,
                stages=["Production"]
            )
            
            if prod_versions:
                current_prod = prod_versions[0]
                
                # Archive current production
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=current_prod.version,
                    stage="Archived"
                )
            
            # Promote target version to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage="Production"
            )
            
            logger.info(f"Rolled back {model_name} to version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback model: {e}")
            return False
    
    def load_model(self,
                  model_name: str,
                  version: Optional[int] = None,
                  stage: Optional[str] = None) -> Any:
        """
        Load model from registry.
        
        Args:
            model_name: Model name
            version: Specific version (optional)
            stage: Stage to load from (optional)
            
        Returns:
            Loaded model
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                model_uri = f"models:/{model_name}/Production"
            
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from {model_uri}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def delete_model_version(self,
                           model_name: str,
                           version: int) -> bool:
        """
        Delete model version.
        
        Args:
            model_name: Model name
            version: Version to delete
            
        Returns:
            Success status
        """
        try:
            self.client.delete_model_version(
                name=model_name,
                version=version
            )
            
            logger.info(f"Deleted {model_name} version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            return False

    def get_latest_production_version(self, model_name: str) -> Optional[ModelVersion]:
        """Return the latest production model version metadata if available."""

        if not self.client:
            logger.warning("MLflow client not initialized - cannot fetch production version")
            return None

        try:
            versions = self.client.get_latest_versions(
                model_name,
                stages=[ModelStage.PRODUCTION.value]
            )

            if not versions:
                logger.info("No production version found for model %s", model_name)
                return None

            return versions[0]

        except Exception as exc:
            logger.error("Failed to fetch production version for %s: %s", model_name, exc)
            return None

    def get_production_model_metadata(self, model_name: str) -> Optional[ModelVersion]:
        """Return metadata for the latest production model version if available."""

        return self.get_latest_production_version(model_name)

    def load_production_model(self, model_name: str) -> Optional[Any]:
        """Load the current production model for the provided model name."""

        version = self.get_production_model_metadata(model_name)
        if not version:
            return None

        model_uri = f"models:/{model_name}/{version.version}"

        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(
                "Loaded production model for %s (version %s)",
                model_name,
                version.version,
            )
            return model
        except Exception as exc:
            logger.error(
                "Failed to load production model %s (version %s): %s",
                model_name,
                version.version,
                exc,
            )
            return None

    def get_production_model(self, model_name: str) -> Optional[Any]:
        """Backward compatible alias for :meth:`load_production_model`."""

        return self.load_production_model(model_name)
    
    def search_models(self,
                     filter_string: str = "",
                     max_results: int = 100) -> List[Dict]:
        """
        Search for models.
        
        Args:
            filter_string: MLflow filter string
            max_results: Maximum results
            
        Returns:
            List of models
        """
        try:
            models = self.client.search_registered_models(
                filter_string=filter_string,
                max_results=max_results
            )
            
            results = []
            for model in models:
                results.append({
                    'name': model.name,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp,
                    'description': model.description,
                    'latest_versions': [
                        {
                            'version': v.version,
                            'stage': v.current_stage,
                            'description': v.description
                        }
                        for v in model.latest_versions
                    ]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []
