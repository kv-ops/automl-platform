"""
Model Deployment and Export Service
====================================

Based on best practices from Dataiku (Docker/ONNX/PMML export) and H2O MLOps (K8s deployment).
Implements model packaging, serving (REST/gRPC), edge deployment, and production monitoring.

Author: MLOps Platform Team
Date: 2024
"""

import os
import json
import pickle
import joblib
import shutil
import tempfile
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import asyncio
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
import tensorflow as tf
import torch
import docker
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Response

from automl_platform.config import (
    InsecureEnvironmentVariableError,
    MissingEnvironmentVariableError,
    require_secret,
)
from fastapi.responses import FileResponse, StreamingResponse
import grpc
from concurrent import futures
import yaml
from kubernetes import client, config as k8s_config
from minio import Minio
import mlflow
import boto3
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis
import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from scipy import stats
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

# Import from automl_platform modules
try:
    from automl_platform.config import Config as MLOpsConfig
    from automl_platform.model_selection import ModelSelector
    from automl_platform.storage import StorageService
    from automl_platform.monitoring import MonitoringService
    from automl_platform.metrics import MetricsTracker
except ImportError:
    # Fallback if modules not yet created
    MLOpsConfig = None
    ModelSelector = None
    StorageService = None
    MonitoringService = None
    MetricsTracker = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class DeploymentConfig:
    """Deployment configuration aligned with enterprise standards"""
    
    # Model Registry (MLflow style)
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MODEL_REGISTRY_BACKEND = os.getenv("MODEL_REGISTRY_BACKEND", "mlflow")  # mlflow, custom
    
    # Storage (MinIO/S3 for model artifacts like H2O MLOps)
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")

    @staticmethod
    def _get_required_env(var_name: str) -> str:
        try:
            return require_secret(var_name)
        except MissingEnvironmentVariableError as exc:
            raise RuntimeError(
                f"Environment variable {var_name} must be defined to enable deployment storage integration."
            ) from exc
        except InsecureEnvironmentVariableError as exc:
            raise RuntimeError(
                f"Environment variable {var_name} is set to an insecure default. "
                "Generate a new credential before deploying."
            ) from exc

    @classmethod
    def minio_access_key(cls) -> str:
        return cls._get_required_env("MINIO_ACCESS_KEY")

    @classmethod
    def minio_secret_key(cls) -> str:
        return cls._get_required_env("MINIO_SECRET_KEY")
    MODEL_BUCKET = os.getenv("MODEL_BUCKET", "models")
    
    # Kubernetes deployment (H2O AI Cloud style)
    K8S_ENABLED = os.getenv("K8S_ENABLED", "false").lower() == "true"
    K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "mlops")
    K8S_IMAGE_REGISTRY = os.getenv("K8S_IMAGE_REGISTRY", "localhost:5000")
    
    # Docker configuration
    DOCKER_REGISTRY = os.getenv("DOCKER_REGISTRY", "localhost:5000")
    BASE_IMAGE = os.getenv("BASE_IMAGE", "python:3.9-slim")
    
    # Serving configuration
    DEFAULT_SERVING_PORT = 8501
    GRPC_PORT = 50051
    MAX_BATCH_SIZE = 32
    MODEL_CACHE_SIZE = 10  # Number of models to keep in memory
    
    # Edge deployment
    EDGE_OPTIMIZATION_ENABLED = True
    QUANTIZATION_ENABLED = True
    
    # Monitoring
    PROMETHEUS_ENABLED = True
    DRIFT_DETECTION_ENABLED = True
    PERFORMANCE_TRACKING_ENABLED = True
    
    # Redis for caching
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


# ============================================================================
# Data Models
# ============================================================================

class ExportFormat(str, Enum):
    """Supported model export formats (Dataiku-style)"""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    ONNX = "onnx"
    PMML = "pmml"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    DOCKER = "docker"
    PYTHON = "python"
    JAVA = "java"
    MLFLOW = "mlflow"
    TFLITE = "tflite"  # For edge deployment
    COREML = "coreml"  # For iOS


class ServingMode(str, Enum):
    """Model serving modes"""
    REST = "rest"
    GRPC = "grpc"
    BATCH = "batch"
    STREAMING = "streaming"
    EDGE = "edge"


class DeploymentTarget(str, Enum):
    """Deployment targets"""
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    SERVERLESS = "serverless"
    EDGE = "edge"
    CLOUD = "cloud"  # AWS, GCP, Azure


@dataclass
class ModelMetadata:
    """Model metadata for deployment"""
    model_id: str
    name: str
    version: str
    framework: str  # sklearn, tensorflow, pytorch, xgboost, etc.
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: datetime
    created_by: str
    tags: List[str]
    dependencies: List[str]  # Package dependencies
    model_size_mb: float
    inference_time_ms: float


@dataclass
class DeploymentSpec:
    """Deployment specification"""
    deployment_id: str
    model_id: str
    model_version: str
    target: DeploymentTarget
    serving_mode: ServingMode
    replicas: int = 1
    cpu_request: str = "500m"
    memory_request: str = "512Mi"
    gpu_request: int = 0
    autoscaling_enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    environment_variables: Optional[Dict[str, str]] = None
    health_check_path: str = "/health"
    rollback_on_failure: bool = True


# ============================================================================
# Model Export Service
# ============================================================================

class ModelExportService:
    """Service for exporting models in various formats (Dataiku-style)"""
    
    def __init__(self):
        access_key = DeploymentConfig.minio_access_key()
        secret_key = DeploymentConfig.minio_secret_key()

        self.minio_client = Minio(
            DeploymentConfig.MINIO_ENDPOINT,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """Ensure model bucket exists"""
        if not self.minio_client.bucket_exists(DeploymentConfig.MODEL_BUCKET):
            self.minio_client.make_bucket(DeploymentConfig.MODEL_BUCKET)
    
    async def export_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        format: ExportFormat,
        output_path: Optional[str] = None
    ) -> str:
        """Export model in specified format"""
        
        if output_path is None:
            output_path = f"/tmp/{metadata.model_id}_{metadata.version}"
        
        logger.info(f"Exporting model {metadata.model_id} in {format} format")
        
        if format == ExportFormat.PICKLE:
            return await self._export_pickle(model, metadata, output_path)
        elif format == ExportFormat.JOBLIB:
            return await self._export_joblib(model, metadata, output_path)
        elif format == ExportFormat.ONNX:
            return await self._export_onnx(model, metadata, output_path)
        elif format == ExportFormat.PMML:
            return await self._export_pmml(model, metadata, output_path)
        elif format == ExportFormat.DOCKER:
            return await self._export_docker(model, metadata, output_path)
        elif format == ExportFormat.MLFLOW:
            return await self._export_mlflow(model, metadata, output_path)
        elif format == ExportFormat.TFLITE:
            return await self._export_tflite(model, metadata, output_path)
        elif format == ExportFormat.PYTHON:
            return await self._export_python_code(model, metadata, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _export_pickle(self, model: Any, metadata: ModelMetadata, output_path: str) -> str:
        """Export model as pickle"""
        pickle_path = f"{output_path}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Upload to MinIO
        self.minio_client.fput_object(
            DeploymentConfig.MODEL_BUCKET,
            f"{metadata.model_id}/{metadata.version}/model.pkl",
            pickle_path
        )
        
        return pickle_path
    
    async def _export_joblib(self, model: Any, metadata: ModelMetadata, output_path: str) -> str:
        """Export model using joblib"""
        joblib_path = f"{output_path}.joblib"
        joblib.dump(model, joblib_path)
        
        # Upload to MinIO
        self.minio_client.fput_object(
            DeploymentConfig.MODEL_BUCKET,
            f"{metadata.model_id}/{metadata.version}/model.joblib",
            joblib_path
        )
        
        return joblib_path
    
    async def _export_onnx(self, model: Any, metadata: ModelMetadata, output_path: str) -> str:
        """Export sklearn model to ONNX format"""
        if metadata.framework != "sklearn":
            raise ValueError("ONNX export currently only supports sklearn models")
        
        # Get input types from schema
        n_features = len(metadata.input_schema.get('features', []))
        if n_features == 0:
            n_features = 10  # Default fallback
            
        initial_types = [
            ('input', 
             np.float32 if metadata.input_schema.get('dtype') == 'float32' else np.float64,
             [None, n_features])
        ]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(model, initial_types=initial_types)
        
        # Save ONNX model
        onnx_path = f"{output_path}.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        # Upload to MinIO
        self.minio_client.fput_object(
            DeploymentConfig.MODEL_BUCKET,
            f"{metadata.model_id}/{metadata.version}/model.onnx",
            onnx_path
        )
        
        return onnx_path
    
    async def _export_pmml(self, model: Any, metadata: ModelMetadata, output_path: str) -> str:
        """Export model to PMML format"""
        pmml_path = f"{output_path}.pmml"
        
        # Wrap model in PMML pipeline if needed
        if not isinstance(model, PMMLPipeline):
            pipeline = PMMLPipeline([
                ("model", model)
            ])
        else:
            pipeline = model
        
        # Export to PMML
        sklearn2pmml(pipeline, pmml_path)
        
        # Upload to MinIO
        self.minio_client.fput_object(
            DeploymentConfig.MODEL_BUCKET,
            f"{metadata.model_id}/{metadata.version}/model.pmml",
            pmml_path
        )
        
        return pmml_path
    
    async def _export_docker(self, model: Any, metadata: ModelMetadata, output_path: str) -> str:
        """Export model as Docker container with FastAPI serving"""
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save model
        model_path = os.path.join(temp_dir, "model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Create FastAPI serving app
        app_code = self._generate_serving_app(metadata)
        with open(os.path.join(temp_dir, "app.py"), "w") as f:
            f.write(app_code)
        
        # Create requirements.txt
        requirements = self._generate_requirements(metadata)
        with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
            f.write(requirements)
        
        # Create Dockerfile
        dockerfile = self._generate_dockerfile(metadata)
        with open(os.path.join(temp_dir, "Dockerfile"), "w") as f:
            f.write(dockerfile)
        
        # Build Docker image
        docker_client = docker.from_env()
        image_tag = f"{DeploymentConfig.DOCKER_REGISTRY}/{metadata.model_id}:{metadata.version}"
        
        logger.info(f"Building Docker image: {image_tag}")
        image, logs = docker_client.images.build(
            path=temp_dir,
            tag=image_tag,
            rm=True
        )
        
        # Push to registry
        if DeploymentConfig.DOCKER_REGISTRY != "localhost:5000":
            logger.info(f"Pushing image to registry: {image_tag}")
            docker_client.images.push(image_tag)
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return image_tag
    
    async def _export_mlflow(self, model: Any, metadata: ModelMetadata, output_path: str) -> str:
        """Export model using MLflow"""
        mlflow.set_tracking_uri(DeploymentConfig.MLFLOW_TRACKING_URI)
        
        with mlflow.start_run():
            # Log model
            if metadata.framework == "sklearn":
                mlflow.sklearn.log_model(model, "model")
            elif metadata.framework == "tensorflow":
                mlflow.tensorflow.log_model(model, "model")
            elif metadata.framework == "pytorch":
                mlflow.pytorch.log_model(model, "model")
            else:
                mlflow.pyfunc.log_model("model", python_model=model)
            
            # Log metrics
            for metric_name, metric_value in metadata.metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log parameters
            mlflow.log_param("model_id", metadata.model_id)
            mlflow.log_param("version", metadata.version)
            
            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, metadata.name)
        
        return model_uri
    
    async def _export_tflite(self, model: Any, metadata: ModelMetadata, output_path: str) -> str:
        """Export TensorFlow model to TFLite for edge deployment"""
        if metadata.framework != "tensorflow":
            raise ValueError("TFLite export only supports TensorFlow models")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(model)
        
        # Apply optimizations for edge
        if DeploymentConfig.QUANTIZATION_ENABLED:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = f"{output_path}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Upload to MinIO
        self.minio_client.fput_object(
            DeploymentConfig.MODEL_BUCKET,
            f"{metadata.model_id}/{metadata.version}/model.tflite",
            tflite_path
        )
        
        return tflite_path
    
    async def _export_python_code(self, model: Any, metadata: ModelMetadata, output_path: str) -> str:
        """Generate Python code for model inference"""
        code = self._generate_python_inference_code(model, metadata)
        
        py_path = f"{output_path}_inference.py"
        with open(py_path, 'w') as f:
            f.write(code)
        
        # Upload to MinIO
        self.minio_client.fput_object(
            DeploymentConfig.MODEL_BUCKET,
            f"{metadata.model_id}/{metadata.version}/inference.py",
            py_path
        )
        
        return py_path
    
    def _generate_serving_app(self, metadata: ModelMetadata) -> str:
        """Generate FastAPI serving application code"""
        return f'''
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging

app = FastAPI(title="{metadata.name} Model Server")
logger = logging.getLogger(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    features: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str = "{metadata.version}"

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        X = np.array(request.features)
        predictions = model.predict(X)
        return PredictionResponse(predictions=predictions.tolist())
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {{"status": "healthy", "model_id": "{metadata.model_id}"}}

@app.get("/metrics")
async def metrics():
    return {metadata.metrics}
'''
    
    def _generate_requirements(self, metadata: ModelMetadata) -> str:
        """Generate requirements.txt for Docker image"""
        base_requirements = [
            "fastapi==0.104.1",
            "uvicorn==0.24.0",
            "numpy==1.24.3",
            "scikit-learn==1.3.0",
            "pandas==2.0.3",
            "pydantic==2.0.0"
        ]
        
        # Add framework-specific requirements
        if metadata.framework == "tensorflow":
            base_requirements.append("tensorflow==2.13.0")
        elif metadata.framework == "pytorch":
            base_requirements.append("torch==2.0.1")
        elif metadata.framework == "xgboost":
            base_requirements.append("xgboost==1.7.6")
        
        # Add custom dependencies
        if metadata.dependencies:
            base_requirements.extend(metadata.dependencies)
        
        return "\n".join(base_requirements)
    
    def _generate_dockerfile(self, metadata: ModelMetadata) -> str:
        """Generate Dockerfile for model serving"""
        return f'''
FROM {DeploymentConfig.BASE_IMAGE}

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and app
COPY model.pkl .
COPY app.py .

# Expose port
EXPOSE {DeploymentConfig.DEFAULT_SERVING_PORT}

# Run server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "{DeploymentConfig.DEFAULT_SERVING_PORT}"]
'''
    
    def _generate_python_inference_code(self, model: Any, metadata: ModelMetadata) -> str:
        """Generate standalone Python inference code"""
        n_features = len(metadata.input_schema.get('features', []))
        if n_features == 0:
            n_features = 10  # Default fallback
            
        return f'''
"""
Standalone inference code for {metadata.name}
Model ID: {metadata.model_id}
Version: {metadata.version}
Generated: {datetime.utcnow().isoformat()}
"""

import pickle
import numpy as np
from typing import Union, List

class ModelInference:
    def __init__(self, model_path: str = "model.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.metadata = {str(metadata.__dict__)}
    
    def predict(self, features: Union[List, np.ndarray]) -> np.ndarray:
        """Make predictions"""
        X = np.array(features)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)
    
    def predict_proba(self, features: Union[List, np.ndarray]) -> np.ndarray:
        """Get prediction probabilities (if supported)"""
        X = np.array(features)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Model does not support probability predictions")

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = ModelInference()
    
    # Make prediction
    sample_features = {list(range({n_features}))}
    predictions = model.predict(sample_features)
    print(f"Predictions: {{predictions}}")
'''
