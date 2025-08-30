"""
Storage module for ML Platform
Handles model versioning, dataset storage, and artifact management
Compatible with S3/MinIO for cloud-native deployment
"""

import os
import json
import hashlib
import pickle
import joblib
from typing import Any, Dict, List, Optional, Union, BinaryIO, Tuple
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata for versioning and tracking"""
    model_id: str
    version: str
    model_type: str
    algorithm: str
    created_at: str
    updated_at: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    feature_names: List[str]
    target_name: str
    dataset_hash: str
    pipeline_hash: str
    tags: List[str] = None
    description: str = ""
    author: str = ""
    tenant_id: str = "default"
    
    def to_dict(self):
        return asdict(self)


class StorageBackend:
    """Abstract base class for storage backends"""
    
    def save_model(self, model: Any, metadata: ModelMetadata) -> str:
        raise NotImplementedError
    
    def load_model(self, model_id: str, version: str = None) -> tuple:
        raise NotImplementedError
    
    def save_dataset(self, data: pd.DataFrame, dataset_id: str) -> str:
        raise NotImplementedError
    
    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        raise NotImplementedError
    
    def list_models(self, tenant_id: str = None) -> List[Dict]:
        raise NotImplementedError
    
    def delete_model(self, model_id: str, version: str = None) -> bool:
        raise NotImplementedError


class MinIOStorage(StorageBackend):
    """MinIO/S3 compatible storage backend"""
    
    def __init__(self, 
                 endpoint: str = "localhost:9000",
                 access_key: str = "minioadmin",
                 secret_key: str = "minioadmin",
                 secure: bool = False,
                 region: str = "us-east-1"):
        
        if MINIO_AVAILABLE:
            self.client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                region=region
            )
        elif S3_AVAILABLE:
            # Fallback to boto3 for AWS S3
            self.client = boto3.client(
                's3',
                endpoint_url=f"http://{endpoint}" if not secure else f"https://{endpoint}",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
            self.use_boto = True
        else:
            raise ImportError("Neither minio nor boto3 is installed")
        
        self.models_bucket = "models"
        self.datasets_bucket = "datasets"
        self.artifacts_bucket = "artifacts"
        
        # Create buckets if they don't exist
        self._ensure_buckets()
    
    def _ensure_buckets(self):
        """Create required buckets if they don't exist"""
        buckets = [self.models_bucket, self.datasets_bucket, self.artifacts_bucket]
        
        for bucket in buckets:
            try:
                if MINIO_AVAILABLE:
                    if not self.client.bucket_exists(bucket):
                        self.client.make_bucket(bucket)
                        logger.info(f"Created bucket: {bucket}")
                elif S3_AVAILABLE and hasattr(self, 'use_boto'):
                    try:
                        self.client.head_bucket(Bucket=bucket)
                    except ClientError:
                        self.client.create_bucket(Bucket=bucket)
                        logger.info(f"Created bucket: {bucket}")
            except Exception as e:
                logger.warning(f"Could not create bucket {bucket}: {e}")
    
    def save_model(self, model: Any, metadata: ModelMetadata) -> str:
        """Save model to MinIO with versioning"""
        try:
            # Serialize model
            model_bytes = pickle.dumps(model)
            
            # Create object path
            object_name = f"{metadata.tenant_id}/{metadata.model_id}/v{metadata.version}/model.pkl"
            metadata_name = f"{metadata.tenant_id}/{metadata.model_id}/v{metadata.version}/metadata.json"
            
            # Save model
            if MINIO_AVAILABLE:
                from io import BytesIO
                self.client.put_object(
                    self.models_bucket,
                    object_name,
                    BytesIO(model_bytes),
                    len(model_bytes)
                )
                
                # Save metadata
                metadata_bytes = json.dumps(metadata.to_dict()).encode()
                self.client.put_object(
                    self.models_bucket,
                    metadata_name,
                    BytesIO(metadata_bytes),
                    len(metadata_bytes)
                )
            else:
                # Use boto3
                self.client.put_object(
                    Bucket=self.models_bucket,
                    Key=object_name,
                    Body=model_bytes
                )
                self.client.put_object(
                    Bucket=self.models_bucket,
                    Key=metadata_name,
                    Body=json.dumps(metadata.to_dict())
                )
            
            logger.info(f"Model saved: {metadata.model_id} v{metadata.version}")
            return f"{self.models_bucket}/{object_name}"
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, model_id: str, version: str = None, tenant_id: str = "default") -> tuple:
        """Load model from MinIO"""
        try:
            # If no version specified, get latest
            if version is None:
                version = self._get_latest_version(model_id, tenant_id)
            
            object_name = f"{tenant_id}/{model_id}/v{version}/model.pkl"
            metadata_name = f"{tenant_id}/{model_id}/v{version}/metadata.json"
            
            if MINIO_AVAILABLE:
                # Load model
                response = self.client.get_object(self.models_bucket, object_name)
                model = pickle.loads(response.read())
                response.close()
                
                # Load metadata
                response = self.client.get_object(self.models_bucket, metadata_name)
                metadata = json.loads(response.read())
                response.close()
            else:
                # Use boto3
                response = self.client.get_object(Bucket=self.models_bucket, Key=object_name)
                model = pickle.loads(response['Body'].read())
                
                response = self.client.get_object(Bucket=self.models_bucket, Key=metadata_name)
                metadata = json.loads(response['Body'].read())
            
            logger.info(f"Model loaded: {model_id} v{version}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_latest_version(self, model_id: str, tenant_id: str) -> str:
        """Get latest version of a model"""
        prefix = f"{tenant_id}/{model_id}/"
        versions = []
        
        try:
            if MINIO_AVAILABLE:
                objects = self.client.list_objects(self.models_bucket, prefix=prefix)
                for obj in objects:
                    parts = obj.object_name.split('/')
                    if len(parts) >= 3 and parts[2].startswith('v'):
                        versions.append(parts[2][1:])  # Remove 'v' prefix
            else:
                response = self.client.list_objects_v2(
                    Bucket=self.models_bucket,
                    Prefix=prefix
                )
                for obj in response.get('Contents', []):
                    parts = obj['Key'].split('/')
                    if len(parts) >= 3 and parts[2].startswith('v'):
                        versions.append(parts[2][1:])
            
            if versions:
                return sorted(versions, key=lambda x: tuple(map(int, x.split('.'))))[-1]
            else:
                return "1.0.0"
                
        except Exception as e:
            logger.error(f"Failed to get latest version: {e}")
            return "1.0.0"
    
    def save_dataset(self, data: pd.DataFrame, dataset_id: str, tenant_id: str = "default") -> str:
        """Save dataset to MinIO in Parquet format"""
        try:
            # Convert to parquet
            parquet_buffer = BytesIO()
            data.to_parquet(parquet_buffer, engine='pyarrow', compression='snappy')
            parquet_bytes = parquet_buffer.getvalue()
            
            # Calculate hash for deduplication
            data_hash = hashlib.sha256(parquet_bytes).hexdigest()[:16]
            
            # Create object path
            object_name = f"{tenant_id}/{dataset_id}_{data_hash}.parquet"
            metadata_name = f"{tenant_id}/{dataset_id}_{data_hash}_metadata.json"
            
            # Dataset metadata
            dataset_metadata = {
                "dataset_id": dataset_id,
                "hash": data_hash,
                "shape": list(data.shape),
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "created_at": datetime.now().isoformat(),
                "tenant_id": tenant_id
            }
            
            if MINIO_AVAILABLE:
                # Save dataset
                self.client.put_object(
                    self.datasets_bucket,
                    object_name,
                    BytesIO(parquet_bytes),
                    len(parquet_bytes)
                )
                
                # Save metadata
                metadata_bytes = json.dumps(dataset_metadata).encode()
                self.client.put_object(
                    self.datasets_bucket,
                    metadata_name,
                    BytesIO(metadata_bytes),
                    len(metadata_bytes)
                )
            else:
                self.client.put_object(
                    Bucket=self.datasets_bucket,
                    Key=object_name,
                    Body=parquet_bytes
                )
                self.client.put_object(
                    Bucket=self.datasets_bucket,
                    Key=metadata_name,
                    Body=json.dumps(dataset_metadata)
                )
            
            logger.info(f"Dataset saved: {dataset_id} (hash: {data_hash})")
            return f"{self.datasets_bucket}/{object_name}"
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
    
    def load_dataset(self, dataset_id: str, tenant_id: str = "default") -> pd.DataFrame:
        """Load dataset from MinIO"""
        try:
            # Find dataset file
            prefix = f"{tenant_id}/{dataset_id}"
            
            if MINIO_AVAILABLE:
                objects = list(self.client.list_objects(self.datasets_bucket, prefix=prefix))
                parquet_files = [obj.object_name for obj in objects if obj.object_name.endswith('.parquet')]
            else:
                response = self.client.list_objects_v2(
                    Bucket=self.datasets_bucket,
                    Prefix=prefix
                )
                parquet_files = [obj['Key'] for obj in response.get('Contents', []) 
                               if obj['Key'].endswith('.parquet')]
            
            if not parquet_files:
                raise FileNotFoundError(f"Dataset {dataset_id} not found")
            
            # Load the most recent one
            object_name = sorted(parquet_files)[-1]
            
            if MINIO_AVAILABLE:
                response = self.client.get_object(self.datasets_bucket, object_name)
                data = pd.read_parquet(BytesIO(response.read()))
                response.close()
            else:
                response = self.client.get_object(Bucket=self.datasets_bucket, Key=object_name)
                data = pd.read_parquet(BytesIO(response['Body'].read()))
            
            logger.info(f"Dataset loaded: {dataset_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def list_models(self, tenant_id: str = None) -> List[Dict]:
        """List all models for a tenant"""
        models = []
        prefix = f"{tenant_id}/" if tenant_id else ""
        
        try:
            if MINIO_AVAILABLE:
                objects = self.client.list_objects(self.models_bucket, prefix=prefix, recursive=True)
                for obj in objects:
                    if obj.object_name.endswith('metadata.json'):
                        response = self.client.get_object(self.models_bucket, obj.object_name)
                        metadata = json.loads(response.read())
                        models.append(metadata)
                        response.close()
            else:
                response = self.client.list_objects_v2(
                    Bucket=self.models_bucket,
                    Prefix=prefix
                )
                for obj in response.get('Contents', []):
                    if obj['Key'].endswith('metadata.json'):
                        resp = self.client.get_object(Bucket=self.models_bucket, Key=obj['Key'])
                        metadata = json.loads(resp['Body'].read())
                        models.append(metadata)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


class LocalStorage(StorageBackend):
    """Local filesystem storage backend for development"""
    
    def __init__(self, base_path: str = "./ml_storage"):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.datasets_path = self.base_path / "datasets"
        self.artifacts_path = self.base_path / "artifacts"
        
        # Create directories
        for path in [self.models_path, self.datasets_path, self.artifacts_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Any, metadata: ModelMetadata) -> str:
        """Save model to local filesystem"""
        try:
            model_dir = self.models_path / metadata.tenant_id / metadata.model_id / f"v{metadata.version}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pkl"
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            logger.info(f"Model saved locally: {metadata.model_id} v{metadata.version}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, model_id: str, version: str = None, tenant_id: str = "default") -> tuple:
        """Load model from local filesystem"""
        try:
            if version is None:
                version = self._get_latest_version(model_id, tenant_id)
            
            model_dir = self.models_path / tenant_id / model_id / f"v{version}"
            
            # Load model
            model_path = model_dir / "model.pkl"
            model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Model loaded: {model_id} v{version}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_latest_version(self, model_id: str, tenant_id: str) -> str:
        """Get latest version of a model"""
        model_base = self.models_path / tenant_id / model_id
        if not model_base.exists():
            return "1.0.0"
        
        versions = []
        for version_dir in model_base.iterdir():
            if version_dir.is_dir() and version_dir.name.startswith('v'):
                versions.append(version_dir.name[1:])
        
        if versions:
            return sorted(versions, key=lambda x: tuple(map(int, x.split('.'))))[-1]
        return "1.0.0"
    
    def save_dataset(self, data: pd.DataFrame, dataset_id: str, tenant_id: str = "default") -> str:
        """Save dataset to local filesystem"""
        try:
            dataset_dir = self.datasets_path / tenant_id
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet
            data_hash = hashlib.sha256(data.to_csv().encode()).hexdigest()[:16]
            dataset_path = dataset_dir / f"{dataset_id}_{data_hash}.parquet"
            data.to_parquet(dataset_path, engine='pyarrow', compression='snappy')
            
            # Save metadata
            metadata = {
                "dataset_id": dataset_id,
                "hash": data_hash,
                "shape": list(data.shape),
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "created_at": datetime.now().isoformat(),
                "tenant_id": tenant_id
            }
            
            metadata_path = dataset_dir / f"{dataset_id}_{data_hash}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Dataset saved: {dataset_id} (hash: {data_hash})")
            return str(dataset_path)
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
    
    def load_dataset(self, dataset_id: str, tenant_id: str = "default") -> pd.DataFrame:
        """Load dataset from local filesystem"""
        try:
            dataset_dir = self.datasets_path / tenant_id
            
            # Find dataset file
            parquet_files = list(dataset_dir.glob(f"{dataset_id}_*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"Dataset {dataset_id} not found")
            
            # Load the most recent one
            dataset_path = sorted(parquet_files, key=lambda x: x.stat().st_mtime)[-1]
            data = pd.read_parquet(dataset_path)
            
            logger.info(f"Dataset loaded: {dataset_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def list_models(self, tenant_id: str = None) -> List[Dict]:
        """List all models for a tenant"""
        models = []
        
        if tenant_id:
            search_path = self.models_path / tenant_id
        else:
            search_path = self.models_path
        
        try:
            for metadata_file in search_path.rglob("metadata.json"):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    models.append(metadata)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def delete_model(self, model_id: str, version: str = None, tenant_id: str = "default") -> bool:
        """Delete a model version or all versions"""
        try:
            if version:
                model_dir = self.models_path / tenant_id / model_id / f"v{version}"
                if model_dir.exists():
                    import shutil
                    shutil.rmtree(model_dir)
                    logger.info(f"Deleted model: {model_id} v{version}")
                    return True
            else:
                model_base = self.models_path / tenant_id / model_id
                if model_base.exists():
                    import shutil
                    shutil.rmtree(model_base)
                    logger.info(f"Deleted all versions of model: {model_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False


class StorageManager:
    """Main storage manager that handles backend selection"""
    
    def __init__(self, backend: str = "local", **kwargs):
        """
        Initialize storage manager
        
        Args:
            backend: Storage backend type ('local', 'minio', 's3')
            **kwargs: Backend-specific configuration
        """
        self.backend_type = backend
        
        if backend == "minio" or backend == "s3":
            self.backend = MinIOStorage(**kwargs)
        else:
            self.backend = LocalStorage(**kwargs)
    
    def save_model(self, model: Any, metadata: Dict, version: str = None) -> str:
        """Save a model with metadata"""
        if version is None:
            version = "1.0.0"
        
        # Create metadata object
        model_metadata = ModelMetadata(
            model_id=metadata.get('model_id', 'unnamed'),
            version=version,
            model_type=metadata.get('model_type', 'unknown'),
            algorithm=metadata.get('algorithm', 'unknown'),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metrics=metadata.get('metrics', {}),
            parameters=metadata.get('parameters', {}),
            feature_names=metadata.get('feature_names', []),
            target_name=metadata.get('target_name', 'target'),
            dataset_hash=metadata.get('dataset_hash', ''),
            pipeline_hash=metadata.get('pipeline_hash', ''),
            tags=metadata.get('tags', []),
            description=metadata.get('description', ''),
            author=metadata.get('author', 'unknown'),
            tenant_id=metadata.get('tenant_id', 'default')
        )
        
        return self.backend.save_model(model, model_metadata)
    
    def load_model(self, model_id: str, version: str = None, tenant_id: str = "default"):
        """Load a model"""
        return self.backend.load_model(model_id, version, tenant_id)
    
    def save_dataset(self, data: pd.DataFrame, dataset_id: str, tenant_id: str = "default"):
        """Save a dataset"""
        return self.backend.save_dataset(data, dataset_id, tenant_id)
    
    def load_dataset(self, dataset_id: str, tenant_id: str = "default"):
        """Load a dataset"""
        return self.backend.load_dataset(dataset_id, tenant_id)
    
    def list_models(self, tenant_id: str = None):
        """List all models"""
        return self.backend.list_models(tenant_id)
    
    def delete_model(self, model_id: str, version: str = None, tenant_id: str = "default"):
        """Delete a model"""
        if hasattr(self.backend, 'delete_model'):
            return self.backend.delete_model(model_id, version, tenant_id)
        return False
    
    def get_model_history(self, model_id: str, tenant_id: str = "default") -> List[Dict]:
        """Get version history of a model"""
        all_models = self.list_models(tenant_id)
        return [m for m in all_models if m.get('model_id') == model_id]


# Feature Store functionality
class FeatureStore:
    """Simple feature store for feature reusability"""
    
    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
        self.feature_cache = {}
    
    def save_features(self, 
                     features: pd.DataFrame,
                     feature_set_name: str,
                     version: str = "1.0.0",
                     metadata: Dict = None) -> str:
        """Save computed features for reuse"""
        
        feature_metadata = {
            "feature_set_name": feature_set_name,
            "version": version,
            "columns": list(features.columns),
            "shape": list(features.shape),
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Save as dataset with special naming
        dataset_id = f"features_{feature_set_name}_v{version}"
        path = self.storage.save_dataset(features, dataset_id)
        
        # Cache for quick access
        cache_key = f"{feature_set_name}:{version}"
        self.feature_cache[cache_key] = features
        
        logger.info(f"Features saved: {feature_set_name} v{version}")
        return path
    
    def load_features(self, 
                     feature_set_name: str,
                     version: str = None) -> pd.DataFrame:
        """Load features from store"""
        
        # Check cache first
        cache_key = f"{feature_set_name}:{version}" if version else feature_set_name
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Load from storage
        dataset_id = f"features_{feature_set_name}"
        if version:
            dataset_id += f"_v{version}"
        
        features = self.storage.load_dataset(dataset_id)
        
        # Update cache
        self.feature_cache[cache_key] = features
        
        return features
    
    def list_feature_sets(self) -> List[str]:
        """List available feature sets"""
        # This would need to be implemented based on storage backend
        # For now, return cached keys
        return list(set(k.split(':')[0] for k in self.feature_cache.keys()))


# Example usage
if __name__ == "__main__":
    # Example with local storage
    storage = StorageManager(backend="local", base_path="./ml_storage")
    
    # Example with MinIO
    # storage = StorageManager(
    #     backend="minio",
    #     endpoint="localhost:9000",
    #     access_key="minioadmin",
    #     secret_key="minioadmin"
    # )
    
    # Save a dummy model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10)
    
    metadata = {
        "model_id": "test_model",
        "model_type": "classification",
        "algorithm": "RandomForest",
        "metrics": {"accuracy": 0.95, "f1": 0.93},
        "parameters": {"n_estimators": 10},
        "feature_names": ["feature1", "feature2"],
        "target_name": "target"
    }
    
    path = storage.save_model(model, metadata, version="1.0.0")
    print(f"Model saved at: {path}")
    
    # Load model
    loaded_model, loaded_metadata = storage.load_model("test_model", version="1.0.0")
    print(f"Model loaded: {loaded_metadata['model_id']}")
    
    # Feature store example
    feature_store = FeatureStore(storage)
    
    # Save features
    import numpy as np
    features_df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    
    feature_store.save_features(
        features_df,
        "test_features",
        version="1.0.0",
        metadata={"description": "Test feature set"}
    )
    
    # Load features
    loaded_features = feature_store.load_features("test_features", version="1.0.0")
    print(f"Features loaded: {loaded_features.shape}")
