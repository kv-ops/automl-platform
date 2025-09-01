"""
MLflow Registry - Model Registry and Versioning with A/B Testing
================================================================
Place in: automl_platform/mlflow_registry.py

Complete MLflow integration for model registry, versioning, and A/B testing.
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
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
    logging.warning("MLflow not installed. Install with: pip install mlflow")

from scipy import stats

logger = logging.getLogger(__name__)


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
    metrics: Dict[str, float]
    params: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    created_by: str
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    production_metrics: Optional[Dict[str, float]] = None
    drift_score: Optional[float] = None
    traffic_percentage: float = 0.0
    is_champion: bool = False
    is_challenger: bool = False


class MLflowRegistry:
    """MLflow-based model registry and versioning"""
    
    def __init__(self, config, tracking_uri: Optional[str] = None):
        self.config = config
        
        if MLFLOW_AVAILABLE:
            # Set tracking URI from config or environment
            self.tracking_uri = (
                tracking_uri or 
                getattr(config, 'mlflow_tracking_uri', None) or 
                os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Initialize client
            self.client = MlflowClient()
            
            # Set experiment
            tenant_id = getattr(config, 'tenant_id', 'default')
            experiment_name = f"automl_{tenant_id}"
            
            try:
                self.experiment_id = mlflow.create_experiment(experiment_name)
            except:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                self.experiment_id = experiment.experiment_id if experiment else None
            
            if experiment_name:
                mlflow.set_experiment(experiment_name)
                
            logger.info(f"MLflow registry initialized with tracking URI: {self.tracking_uri}")
        else:
            logger.warning("MLflow not available, using local registry")
            self.client = None
            self.experiment_id = None
            self.local_registry = {}  # Fallback storage
    
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
            return self._register_local(model, model_name, metrics, params, description, tags)
        
        with mlflow.start_run() as run:
            # Log parameters
            for key, value in params.items():
                if value is not None:
                    mlflow.log_param(key, str(value)[:250])  # MLflow param limit
            
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            # Log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Add description as tag
            if description:
                mlflow.set_tag("description", description)
            
            # Infer signature if samples provided
            signature = None
            if X_sample is not None and y_sample is not None:
                try:
                    signature = infer_signature(X_sample, y_sample)
                except:
                    pass
            
            # Log model based on type
            model_type = type(model).__name__
            model_module = str(type(model).__module__)
            
            try:
                if "sklearn" in model_module:
                    mlflow.sklearn.log_model(
                        model, 
                        "model",
                        signature=signature,
                        registered_model_name=model_name
                    )
                elif "xgboost" in model_module:
                    mlflow.xgboost.log_model(
                        model,
                        "model",
                        signature=signature,
                        registered_model_name=model_name
                    )
                elif "lightgbm" in model_module:
                    mlflow.lightgbm.log_model(
                        model,
                        "model",
                        signature=signature,
                        registered_model_name=model_name
                    )
                elif "torch" in model_module:
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
            except Exception as e:
                logger.error(f"Failed to log model: {e}")
                # Fallback to pickle
                mlflow.log_artifact(self._pickle_model(model), "model")
            
            run_id = run.info.run_id
        
        # Get version number
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            version_number = len(versions) + 1
        except:
            version_number = 1
        
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
            created_by=getattr(self.config, 'user_id', 'system'),
            description=description,
            tags=tags or {}
        )
        
        logger.info(f"Registered model {model_name} version {version_number}")
        
        return model_version
    
    def promote_model(self, model_name: str, version: int, 
                     target_stage: ModelStage) -> bool:
        """Promote model to a different stage"""
        
        if not MLFLOW_AVAILABLE:
            return self._promote_local(model_name, version, target_stage)
        
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
            return self._get_local_model(model_name, ModelStage.PRODUCTION)
        
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
            return self._compare_local(model_name, version1, version2)
        
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
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {}
    
    def rollback_model(self, model_name: str, target_version: int) -> bool:
        """Rollback to a previous model version"""
        
        if not MLFLOW_AVAILABLE:
            return False
        
        try:
            # Demote current production model
            prod_versions = self.client.get_latest_versions(
                model_name,
                stages=[ModelStage.PRODUCTION.value]
            )
            
            for version in prod_versions:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage=ModelStage.ARCHIVED.value
                )
            
            # Promote target version to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage=ModelStage.PRODUCTION.value
            )
            
            logger.info(f"Rolled back {model_name} to version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback model: {e}")
            return False
    
    def get_model_history(self, model_name: str, limit: int = 10) -> List[Dict]:
        """Get model version history"""
        
        if not MLFLOW_AVAILABLE:
            return []
        
        try:
            versions = self.client.search_model_versions(
                f"name='{model_name}'",
                order_by=["version DESC"],
                max_results=limit
            )
            
            history = []
            for v in versions:
                run = self.client.get_run(v.run_id)
                history.append({
                    "version": v.version,
                    "stage": v.current_stage,
                    "created_at": v.creation_timestamp,
                    "updated_at": v.last_updated_timestamp,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get model history: {e}")
            return []
    
    def _pickle_model(self, model: Any) -> str:
        """Pickle model to temporary file"""
        import tempfile
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        with open(temp_file.name, 'wb') as f:
            pickle.dump(model, f)
        
        return temp_file.name
    
    def _register_local(self, model: Any, model_name: str, 
                       metrics: Dict, params: Dict,
                       description: str, tags: Dict) -> ModelVersion:
        """Fallback local registration when MLflow not available"""
        
        # Create local directory structure
        output_dir = getattr(self.config, 'output_dir', './automl_output')
        base_path = Path(output_dir) / "model_registry" / model_name
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Get version number
        existing_versions = list(base_path.glob("v*"))
        version = len(existing_versions) + 1
        
        # Save model
        version_path = base_path / f"v{version}"
        version_path.mkdir(exist_ok=True)
        
        model_path = version_path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "metrics": metrics,
            "params": params,
            "description": description,
            "tags": tags,
            "created_at": datetime.utcnow().isoformat(),
            "created_by": getattr(self.config, 'user_id', 'system'),
            "stage": ModelStage.DEVELOPMENT.value
        }
        
        metadata_path = version_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Store in local registry
        if model_name not in self.local_registry:
            self.local_registry[model_name] = []
        
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            run_id=str(uuid.uuid4()),
            stage=ModelStage.DEVELOPMENT,
            metrics=metrics,
            params=params,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by=getattr(self.config, 'user_id', 'system'),
            description=description,
            tags=tags or {}
        )
        
        self.local_registry[model_name].append(model_version)
        
        return model_version
    
    def _promote_local(self, model_name: str, version: int, 
                      target_stage: ModelStage) -> bool:
        """Promote model in local registry"""
        
        if model_name in self.local_registry:
            for model_version in self.local_registry[model_name]:
                if model_version.version == version:
                    model_version.stage = target_stage
                    model_version.updated_at = datetime.utcnow()
                    return True
        
        return False
    
    def _get_local_model(self, model_name: str, stage: ModelStage) -> Optional[Any]:
        """Get model from local registry"""
        
        output_dir = getattr(self.config, 'output_dir', './automl_output')
        base_path = Path(output_dir) / "model_registry" / model_name
        
        if model_name in self.local_registry:
            # Find model with target stage
            for model_version in self.local_registry[model_name]:
                if model_version.stage == stage:
                    model_path = base_path / f"v{model_version.version}" / "model.pkl"
                    if model_path.exists():
                        with open(model_path, 'rb') as f:
                            return pickle.load(f)
        
        return None
    
    def _compare_local(self, model_name: str, version1: int, version2: int) -> Dict:
        """Compare models in local registry"""
        
        comparison = {"version1": {}, "version2": {}, "metric_diff": {}}
        
        if model_name in self.local_registry:
            for model_version in self.local_registry[model_name]:
                if model_version.version == version1:
                    comparison["version1"] = {
                        "version": version1,
                        "metrics": model_version.metrics,
                        "params": model_version.params,
                        "stage": model_version.stage.value
                    }
                elif model_version.version == version2:
                    comparison["version2"] = {
                        "version": version2,
                        "metrics": model_version.metrics,
                        "params": model_version.params,
                        "stage": model_version.stage.value
                    }
            
            # Calculate differences
            if comparison["version1"] and comparison["version2"]:
                for metric in comparison["version1"]["metrics"]:
                    if metric in comparison["version2"]["metrics"]:
                        v1_val = comparison["version1"]["metrics"][metric]
                        v2_val = comparison["version2"]["metrics"][metric]
                        diff = v2_val - v1_val
                        comparison["metric_diff"][metric] = {
                            "absolute": diff,
                            "relative": (diff / v1_val) * 100 if v1_val != 0 else 0
                        }
        
        return comparison


class ABTestingService:
    """A/B testing for model comparison in production"""
    
    def __init__(self, registry: MLflowRegistry):
        self.registry = registry
        self.active_tests: Dict[str, Dict] = {}
    
    def create_ab_test(self, 
                       model_name: str,
                       champion_version: int,
                       challenger_version: int,
                       traffic_split: float = 0.1,
                       min_samples: int = 100) -> str:
        """Create A/B test between two model versions"""
        
        test_id = str(uuid.uuid4())
        
        self.active_tests[test_id] = {
            "model_name": model_name,
            "champion_version": champion_version,
            "challenger_version": challenger_version,
            "traffic_split": traffic_split,
            "min_samples": min_samples,
            "start_time": datetime.utcnow(),
            "metrics": {
                "champion": {"predictions": 0, "successes": 0, "errors": []},
                "challenger": {"predictions": 0, "successes": 0, "errors": []}
            }
        }
        
        logger.info(f"Created A/B test {test_id} for {model_name}")
        logger.info(f"Champion v{champion_version} vs Challenger v{challenger_version}")
        logger.info(f"Traffic split: {traffic_split*100:.0f}% to challenger")
        
        return test_id
    
    def route_prediction(self, test_id: str) -> Tuple[str, int]:
        """Route prediction request to champion or challenger"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        
        # Random routing based on traffic split
        if np.random.random() < test.traffic_split:
            return "challenger", test["challenger_version"]
        else:
            return "champion", test["champion_version"]
    
    def record_result(self, test_id: str, model_type: str, 
                     success: bool, error_msg: str = None):
        """Record prediction result for A/B test"""
        
        if test_id in self.active_tests:
            metrics = self.active_tests[test_id]["metrics"][model_type]
            metrics["predictions"] += 1
            
            if success:
                metrics["successes"] += 1
            elif error_msg:
                metrics["errors"].append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": error_msg
                })
    
    def get_test_results(self, test_id: str) -> Dict:
        """Get current A/B test results with statistical analysis"""
        
        if test_id not in self.active_tests:
            return {}
        
        test = self.active_tests[test_id]
        duration_hours = (datetime.utcnow() - test["start_time"]).total_seconds() / 3600
        
        # Calculate success rates
        champion_metrics = test["metrics"]["champion"]
        challenger_metrics = test["metrics"]["challenger"]
        
        champion_rate = (champion_metrics["successes"] / 
                        max(1, champion_metrics["predictions"]))
        challenger_rate = (challenger_metrics["successes"] / 
                          max(1, challenger_metrics["predictions"]))
        
        results = {
            "test_id": test_id,
            "model_name": test["model_name"],
            "duration_hours": round(duration_hours, 2),
            "champion": {
                "version": test["champion_version"],
                "predictions": champion_metrics["predictions"],
                "success_rate": round(champion_rate, 4),
                "error_count": len(champion_metrics["errors"])
            },
            "challenger": {
                "version": test["challenger_version"],
                "predictions": challenger_metrics["predictions"],
                "success_rate": round(challenger_rate, 4),
                "error_count": len(challenger_metrics["errors"])
            },
            "improvement": round((challenger_rate - champion_rate) * 100, 2)
        }
        
        # Statistical significance test if enough samples
        min_samples = test.get("min_samples", 30)
        if (champion_metrics["predictions"] >= min_samples and 
            challenger_metrics["predictions"] >= min_samples):
            
            # Perform chi-square test
            champion_success = champion_metrics["successes"]
            champion_fail = champion_metrics["predictions"] - champion_success
            challenger_success = challenger_metrics["successes"]
            challenger_fail = challenger_metrics["predictions"] - challenger_success
            
            contingency_table = [
                [champion_success, champion_fail],
                [challenger_success, challenger_fail]
            ]
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            results["statistical_significance"] = {
                "chi2_statistic": round(chi2, 4),
                "p_value": round(p_value, 4),
                "significant_at_95": p_value < 0.05,
                "significant_at_99": p_value < 0.01
            }
            
            # Confidence interval for difference
            from statsmodels.stats.proportion import confint_proportions_2indep
            
            low, high = confint_proportions_2indep(
                challenger_success, challenger_metrics["predictions"],
                champion_success, champion_metrics["predictions"],
                method='wald'
            )
            
            results["confidence_interval_95"] = {
                "lower": round(low * 100, 2),
                "upper": round(high * 100, 2)
            }
        else:
            results["note"] = f"Need at least {min_samples} samples per variant for statistical significance"
        
        return results
    
    def conclude_test(self, test_id: str, promote_winner: bool = False) -> Dict:
        """Conclude A/B test and optionally promote winner"""
        
        results = self.get_test_results(test_id)
        
        if not results:
            return {"error": "Test not found"}
        
        # Determine winner
        champion_rate = results["champion"]["success_rate"]
        challenger_rate = results["challenger"]["success_rate"]
        
        winner = "challenger" if challenger_rate > champion_rate else "champion"
        results["winner"] = winner
        results["concluded_at"] = datetime.utcnow().isoformat()
        
        # Check if result is statistically significant
        is_significant = False
        if "statistical_significance" in results:
            is_significant = results["statistical_significance"]["significant_at_95"]
        
        results["recommendation"] = {
            "winner": winner,
            "is_significant": is_significant,
            "action": "promote" if is_significant and winner == "challenger" else "keep_current"
        }
        
        # Promote winner if requested and significant
        if promote_winner and winner == "challenger" and is_significant:
            success = self.registry.promote_model(
                results["model_name"],
                results["challenger"]["version"],
                ModelStage.PRODUCTION
            )
            
            if success:
                # Demote champion
                self.registry.promote_model(
                    results["model_name"],
                    results["champion"]["version"],
                    ModelStage.STAGING
                )
                
                results["promoted"] = True
                logger.info(f"Promoted challenger v{results['challenger']['version']} to production")
            else:
                results["promoted"] = False
                results["promotion_error"] = "Failed to promote model"
        else:
            results["promoted"] = False
            if not is_significant:
                results["promotion_reason"] = "Results not statistically significant"
        
        # Archive test
        test_data = self.active_tests[test_id]
        test_data["results"] = results
        
        # Remove from active tests
        del self.active_tests[test_id]
        
        return results
    
    def get_active_tests(self) -> List[Dict]:
        """Get list of active A/B tests"""
        
        active = []
        for test_id, test_data in self.active_tests.items():
            active.append({
                "test_id": test_id,
                "model_name": test_data["model_name"],
                "champion_version": test_data["champion_version"],
                "challenger_version": test_data["challenger_version"],
                "traffic_split": test_data["traffic_split"],
                "start_time": test_data["start_time"].isoformat(),
                "champion_predictions": test_data["metrics"]["champion"]["predictions"],
                "challenger_predictions": test_data["metrics"]["challenger"]["predictions"]
            })
        
        return active
