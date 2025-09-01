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
    
    def _register_local(self, model: Any, model_name: str, 
                       metrics: Dict, params: Dict) -> ModelVersion:
        """Fallback local registration when MLflow not available"""
        
        # Create local directory structure
        base_path = Path(self.config.output_dir) / "model_registry" / model_name
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
            "created_at": datetime.utcnow().isoformat(),
            "created_by": self.config.user_id or "system"
        }
        
        metadata_path = version_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return ModelVersion(
            model_name=model_name,
            version=version,
            run_id=str(uuid.uuid4()),
            stage=ModelStage.DEVELOPMENT,
            metrics=metrics,
            params=params,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by=self.config.user_id or "system"
        )


# ============================================================================
# Automated Retraining Service
# ============================================================================

class RetrainingService:
    """Automated model retraining based on drift and performance"""
    
    def __init__(self, config: AutoMLConfig, 
                 registry: MLflowRegistry,
                 monitor: ModelMonitor):
        self.config = config
        self.registry = registry
        self.monitor = monitor
        
        # Retraining thresholds
        self.drift_threshold = 0.5
        self.performance_degradation_threshold = 0.1  # 10% degradation
        self.min_data_points = 1000
        
        # Schedule configuration
        self.check_frequency = timedelta(days=1)
        self.last_check = datetime.utcnow()
    
    def should_retrain(self, model_name: str) -> Tuple[bool, str]:
        """Check if model should be retrained"""
        
        # Get current production model
        prod_model = self.registry.get_production_model(model_name)
        if not prod_model:
            return False, "No production model found"
        
        # Check drift
        drift_score = self.monitor.get_drift_score(model_name)
        if drift_score and drift_score > self.drift_threshold:
            return True, f"High drift detected: {drift_score:.2f}"
        
        # Check performance degradation
        perf_metrics = self.monitor.get_performance_metrics(model_name)
        if perf_metrics:
            baseline = perf_metrics.get('baseline_accuracy', 1.0)
            current = perf_metrics.get('current_accuracy', 1.0)
            degradation = (baseline - current) / baseline
            
            if degradation > self.performance_degradation_threshold:
                return True, f"Performance degradation: {degradation:.2%}"
        
        # Check data volume
        new_data_count = self.monitor.get_new_data_count(model_name)
        if new_data_count > self.min_data_points:
            return True, f"Sufficient new data: {new_data_count} samples"
        
        return False, "No retraining needed"
    
    async def retrain_model(self, model_name: str, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series) -> ModelVersion:
        """Retrain a model with new data"""
        
        logger.info(f"Starting retraining for {model_name}")
        
        # Get current production model configuration
        prod_versions = self.registry.client.get_latest_versions(
            model_name,
            stages=[ModelStage.PRODUCTION.value]
        )
        
        if not prod_versions:
            raise ValueError(f"No production model found for {model_name}")
        
        current_version = prod_versions[0]
        run = self.registry.client.get_run(current_version.run_id)
        params = run.data.params
        
        # Import orchestrator for retraining
        from .orchestrator import AutoMLOrchestrator
        
        # Configure for retraining
        retrain_config = self.config
        retrain_config.algorithms = [params.get('algorithm', 'RandomForestClassifier')]
        retrain_config.hpo_n_iter = 20  # Less HPO for retraining
        
        # Train new model
        orchestrator = AutoMLOrchestrator(retrain_config)
        orchestrator.fit(X_train, y_train)
        
        # Get best model
        best_model = orchestrator.best_pipeline
        metrics = orchestrator.leaderboard[0]['metrics'] if orchestrator.leaderboard else {}
        
        # Register new version
        new_version = self.registry.register_model(
            model=best_model,
            model_name=model_name,
            metrics=metrics,
            params=params,
            X_sample=X_train.head(100),
            y_sample=y_train.head(100),
            description=f"Automated retraining - {datetime.utcnow()}",
            tags={"retrained": "true", "trigger": "automated"}
        )
        
        # Promote to staging for validation
        self.registry.promote_model(model_name, new_version.version, ModelStage.STAGING)
        
        logger.info(f"Retraining completed for {model_name}, new version: {new_version.version}")
        
        return new_version
    
    def create_retraining_schedule(self):
        """Create automated retraining schedule using Airflow or Prefect"""
        
        if AIRFLOW_AVAILABLE:
            return self._create_airflow_dag()
        elif PREFECT_AVAILABLE:
            return self._create_prefect_flow()
        else:
            logger.warning("No scheduling framework available")
            return None
    
    def _create_airflow_dag(self):
        """Create Airflow DAG for retraining"""
        
        default_args = {
            'owner': 'automl',
            'depends_on_past': False,
            'start_date': days_ago(1),
            'email_on_failure': True,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        }
        
        dag = DAG(
            'model_retraining',
            default_args=default_args,
            description='Automated model retraining',
            schedule_interval=timedelta(days=1),
            catchup=False,
        )
        
        def check_and_retrain(**context):
            """Task to check and retrain models"""
            # This would be implemented to check all models
            pass
        
        retrain_task = PythonOperator(
            task_id='check_and_retrain',
            python_callable=check_and_retrain,
            dag=dag,
        )
        
        return dag
    
    def _create_prefect_flow(self):
        """Create Prefect flow for retraining"""
        
        @task
        def check_models():
            """Check which models need retraining"""
            models_to_retrain = []
            # Implementation
            return models_to_retrain
        
        @task
        def retrain_model_task(model_name: str):
            """Retrain a specific model"""
            # Implementation
            pass
        
        @flow(name="Model Retraining")
        def retraining_flow():
            """Main retraining flow"""
            logger = get_run_logger()
            logger.info("Starting retraining check")
            
            models = check_models()
            for model in models:
                retrain_model_task(model)
        
        # Create deployment with schedule
        deployment = Deployment.build_from_flow(
            flow=retraining_flow,
            name="daily-retraining",
            schedule=CronSchedule(cron="0 2 * * *"),  # 2 AM daily
            work_queue_name="ml-queue",
        )
        
        return deployment


# ============================================================================
# Model Export Service
# ============================================================================

class ModelExporter:
    """Export models to various formats for deployment"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
    
    def export_to_onnx(self, model: Any, 
                      sample_input: np.ndarray,
                      output_path: str) -> bool:
        """Export model to ONNX format"""
        
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available")
            return False
        
        try:
            # Determine input shape and type
            n_features = sample_input.shape[1]
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            
            # Convert to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Save model
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info(f"Model exported to ONNX: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
            return False
    
    def export_to_pmml(self, pipeline: Any, output_path: str) -> bool:
        """Export model to PMML format"""
        
        if not PMML_AVAILABLE:
            logger.error("PMML export not available")
            return False
        
        try:
            # Create PMML pipeline
            pmml_pipeline = PMMLPipeline([
                ("pipeline", pipeline)
            ])
            
            # Export to PMML
            sklearn2pmml(pmml_pipeline, output_path)
            
            logger.info(f"Model exported to PMML: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to PMML: {e}")
            return False
    
    def export_for_edge(self, model: Any, 
                       output_dir: str,
                       quantize: bool = True) -> Dict[str, str]:
        """Export model for edge deployment with optional quantization"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exports = {}
        
        # Export to ONNX for edge
        onnx_path = output_dir / "model.onnx"
        sample_input = np.random.randn(1, 10).astype(np.float32)  # Adjust size
        
        if self.export_to_onnx(model, sample_input, str(onnx_path)):
            exports["onnx"] = str(onnx_path)
            
            if quantize:
                # Quantize ONNX model for smaller size
                quantized_path = output_dir / "model_quantized.onnx"
                if self._quantize_onnx(onnx_path, quantized_path):
                    exports["onnx_quantized"] = str(quantized_path)
        
        # Export to TensorFlow Lite if possible
        tflite_path = output_dir / "model.tflite"
        if self._export_to_tflite(model, tflite_path):
            exports["tflite"] = str(tflite_path)
        
        # Create deployment package
        self._create_edge_package(output_dir, exports)
        
        return exports
    
    def _quantize_onnx(self, input_path: Path, output_path: Path) -> bool:
        """Quantize ONNX model to INT8"""
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantize_dynamic(
                str(input_path),
                str(output_path),
                weight_type=QuantType.QInt8
            )
            
            # Check size reduction
            original_size = input_path.stat().st_size
            quantized_size = output_path.stat().st_size
            reduction = (1 - quantized_size/original_size) * 100
            
            logger.info(f"Model quantized, size reduced by {reduction:.1f}%")
            return True
            
        except Exception as e:
            logger.error(f"Failed to quantize model: {e}")
            return False
    
    def _export_to_tflite(self, model: Any, output_path: Path) -> bool:
        """Export to TensorFlow Lite format"""
        
        try:
            import tensorflow as tf
            
            # Convert sklearn model to TF
            # This is simplified - actual implementation would be more complex
            
            logger.warning("TFLite export not fully implemented")
            return False
            
        except Exception as e:
            logger.error(f"Failed to export to TFLite: {e}")
            return False
    
    def _create_edge_package(self, output_dir: Path, exports: Dict):
        """Create deployment package for edge"""
        
        # Create inference script
        inference_script = output_dir / "inference.py"
        
        script_content = '''
import numpy as np
import onnxruntime as ort

class EdgeModel:
    def __init__(self, model_path="model_quantized.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
    def predict(self, X):
        return self.session.run(None, {self.input_name: X.astype(np.float32)})[0]

if __name__ == "__main__":
    model = EdgeModel()
    # Example inference
    sample = np.random.randn(1, 10).astype(np.float32)
    prediction = model.predict(sample)
    print(f"Prediction: {prediction}")
'''
        
        with open(inference_script, 'w') as f:
            f.write(script_content)
        
        # Create requirements file
        requirements = output_dir / "requirements_edge.txt"
        with open(requirements, 'w') as f:
            f.write("numpy>=1.20.0\nonnxruntime>=1.12.0\n")
        
        # Create README
        readme = output_dir / "README.md"
        with open(readme, 'w') as f:
            f.write(f"""# Edge Deployment Package

## Contents
- `model.onnx`: Original ONNX model
- `model_quantized.onnx`: Quantized model (smaller, faster)
- `inference.py`: Inference script
- `requirements_edge.txt`: Python dependencies

## Usage
```python
from inference import EdgeModel
model = EdgeModel()
predictions = model.predict(your_data)
```

## Model Formats Available
{json.dumps(exports, indent=2)}

## Deployment
1. Install dependencies: `pip install -r requirements_edge.txt`
2. Run inference: `python inference.py`
""")
        
        logger.info(f"Edge deployment package created in {output_dir}")


# ============================================================================
# A/B Testing Service
# ============================================================================

class ABTestingService:
    """A/B testing for model comparison in production"""
    
    def __init__(self, registry: MLflowRegistry):
        self.registry = registry
        self.active_tests: Dict[str, Dict] = {}
    
    def create_ab_test(self, 
                       model_name: str,
                       champion_version: int,
                       challenger_version: int,
                       traffic_split: float = 0.1) -> str:
        """Create A/B test between two model versions"""
        
        test_id = str(uuid.uuid4())
        
        self.active_tests[test_id] = {
            "model_name": model_name,
            "champion_version": champion_version,
            "challenger_version": challenger_version,
            "traffic_split": traffic_split,
            "start_time": datetime.utcnow(),
            "metrics": {
                "champion": {"predictions": 0, "successes": 0},
                "challenger": {"predictions": 0, "successes": 0}
            }
        }
        
        logger.info(f"Created A/B test {test_id} for {model_name}")
        
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
    
    def record_result(self, test_id: str, model_type: str, success: bool):
        """Record prediction result for A/B test"""
        
        if test_id in self.active_tests:
            metrics = self.active_tests[test_id]["metrics"][model_type]
            metrics["predictions"] += 1
            if success:
                metrics["successes"] += 1
    
    def get_test_results(self, test_id: str) -> Dict:
        """Get current A/B test results"""
        
        if test_id not in self.active_tests:
            return {}
        
        test = self.active_tests[test_id]
        
        # Calculate success rates
        results = {
            "test_id": test_id,
            "model_name": test["model_name"],
            "duration": (datetime.utcnow() - test["start_time"]).total_seconds() / 3600,
            "champion": {
                "version": test["champion_version"],
                "predictions": test["metrics"]["champion"]["predictions"],
                "success_rate": test["metrics"]["champion"]["successes"] / max(1, test["metrics"]["champion"]["predictions"])
            },
            "challenger": {
                "version": test["challenger_version"],
                "predictions": test["metrics"]["challenger"]["predictions"],
                "success_rate": test["metrics"]["challenger"]["successes"] / max(1, test["metrics"]["challenger"]["predictions"])
            }
        }
        
        # Statistical significance test
        from scipy import stats
        
        if results["champion"]["predictions"] > 30 and results["challenger"]["predictions"] > 30:
            # Perform chi-square test
            champion_success = test["metrics"]["champion"]["successes"]
            champion_fail = test["metrics"]["champion"]["predictions"] - champion_success
            challenger_success = test["metrics"]["challenger"]["successes"]
            challenger_fail = test["metrics"]["challenger"]["predictions"] - challenger_success
            
            chi2, p_value = stats.chi2_contingency([
                [champion_success, champion_fail],
                [challenger_success, challenger_fail]
            ])[:2]
            
            results["statistical_significance"] = {
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        
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
        
        # Promote winner if requested
        if promote_winner and winner == "challenger":
            self.registry.promote_model(
                results["model_name"],
                results["challenger"]["version"],
                ModelStage.PRODUCTION
            )
            
            # Demote champion
            self.registry.promote_model(
                results["model_name"],
                results["champion"]["version"],
                ModelStage.STAGING
            )
            
            results["promoted"] = True
            logger.info(f"Promoted challenger v{results['challenger']['version']} to production")
        
        # Remove test
        del self.active_tests[test_id]
        
        return results
