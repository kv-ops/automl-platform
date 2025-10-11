"""
Storage module for ML Platform
Handles model versioning, dataset storage, and artifact management
Compatible with S3/MinIO for cloud-native deployment
WITH DOCKER/ONNX/PMML EXPORT SUPPORT AND CONNECTORS INTEGRATION
"""

import os
import json
import hashlib
import pickle
import joblib
from typing import Any, Callable, Dict, List, Optional, Union, BinaryIO, Tuple
from datetime import datetime
from pathlib import Path
import logging
import warnings
from dataclasses import dataclass, asdict
from collections import OrderedDict
import time
import threading
import pandas as pd
import numpy as np
from io import BytesIO, StringIO

from automl_platform.config import (
    InsecureEnvironmentVariableError,
    validate_secret_value,
)

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None  # type: ignore[assignment]
    S3Error = Exception  # type: ignore[assignment]

try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    from google.cloud import storage as gcs_storage
    from google.oauth2 import service_account
    GCS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    gcs_storage = None
    service_account = None
    GCS_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    from skl2onnx import convert_sklearn  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    convert_sklearn = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from sklearn2pmml import sklearn2pmml  # type: ignore[import]
    from sklearn2pmml.pipeline import PMMLPipeline  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    sklearn2pmml = None  # type: ignore[assignment]
    PMMLPipeline = None  # type: ignore[assignment]

ONNX_AVAILABLE = convert_sklearn is not None
PMML_AVAILABLE = sklearn2pmml is not None

logger = logging.getLogger(__name__)


class StorageDisabledError(RuntimeError):
    """Raised when persistence is requested while storage is disabled."""


@dataclass
class ModelMetadata:
    """Model metadata for versioning and tracking."""
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
    encrypted: bool = False
    compression: str = "none"
    
    def to_dict(self):
        return asdict(self)


class StorageBackend:
    """Abstract base class for storage backends."""
    
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


class NullStorage(StorageBackend):
    """Storage backend that disables persistence when `backend` is set to ``"none"``.

    The orchestrator and higher level services can use this backend to operate in
    environments where persisting artefacts is either unsupported or explicitly
    disabled.  Any persistence method invoked on this backend fails fast with a
    :class:`StorageDisabledError` to avoid silently discarding user artefacts.
    """

    _MESSAGE = (
        "Persistent storage is disabled (storage.backend='none'); the %s operation "
        "is unavailable. Configure a supported backend such as 'local' or 's3' to "
        "enable persistence."
    )

    def _raise(self, operation: str) -> None:
        raise StorageDisabledError(self._MESSAGE % operation)

    def save_model(self, model: Any, metadata: ModelMetadata) -> str:
        self._raise("save_model")

    def load_model(
        self,
        model_id: str,
        version: str = None,
        tenant_id: str = "default",
    ) -> tuple:
        self._raise("load_model")

    def save_dataset(
        self,
        data: pd.DataFrame,
        dataset_id: str,
        tenant_id: str = "default",
        format: str = "parquet",
        compression: str = "snappy",
    ) -> str:
        self._raise("save_dataset")

    def load_dataset(self, dataset_id: str, tenant_id: str = "default") -> pd.DataFrame:
        self._raise("load_dataset")

    def list_models(self, tenant_id: str = None) -> List[Dict]:
        self._raise("list_models")

    def delete_model(
        self,
        model_id: str,
        version: str = None,
        tenant_id: str = "default",
    ) -> bool:
        self._raise("delete_model")


class MinIOStorage(StorageBackend):
    """MinIO/S3 compatible storage backend with encryption support."""
    
    def __init__(self,
                 endpoint: str = "localhost:9000",
                 access_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 secure: bool = False,
                 region: str = "us-east-1",
                 encryption_key: bytes = None):

        self.encryption_key = encryption_key

        resolved_access_key = access_key or os.getenv("MINIO_ACCESS_KEY")
        resolved_secret_key = secret_key or os.getenv("MINIO_SECRET_KEY")

        if not resolved_access_key or not resolved_secret_key:
            raise RuntimeError(
                "MinIO access and secret keys must be provided via arguments or environment variables."
            )

        try:
            validate_secret_value("MINIO_ACCESS_KEY", resolved_access_key)
            validate_secret_value("MINIO_SECRET_KEY", resolved_secret_key)
        except InsecureEnvironmentVariableError as exc:
            raise RuntimeError(
                "MinIO credentials use an insecure default value. "
                "Rotate the access and secret keys before enabling MinIO storage."
            ) from exc

        self._uses_minio = False

        if MINIO_AVAILABLE:
            self.client = Minio(
                endpoint,
                access_key=resolved_access_key,
                secret_key=resolved_secret_key,
                secure=secure,
                region=region
            )
            self._uses_minio = True
        elif S3_AVAILABLE:
            # Fallback to boto3 for AWS S3
            self.client = boto3.client(
                's3',
                endpoint_url=f"http://{endpoint}" if not secure else f"https://{endpoint}",
                aws_access_key_id=resolved_access_key,
                aws_secret_access_key=resolved_secret_key,
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
        """Create required buckets if they don't exist."""
        buckets = [self.models_bucket, self.datasets_bucket, self.artifacts_bucket]
        
        for bucket in buckets:
            try:
                if self._uses_minio:
                    if not self.client.bucket_exists(bucket):
                        self.client.make_bucket(bucket)
                        logger.info(f"Created bucket: {bucket}")
                elif hasattr(self, 'use_boto'):
                    try:
                        self.client.head_bucket(Bucket=bucket)
                    except ClientError:
                        self.client.create_bucket(Bucket=bucket)
                        logger.info(f"Created bucket: {bucket}")
            except Exception as e:
                logger.warning(f"Could not create bucket {bucket}: {e}")
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data if encryption key is provided."""
        if self.encryption_key:
            from cryptography.fernet import Fernet
            f = Fernet(self.encryption_key)
            return f.encrypt(data)
        return data
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data if encryption key is provided."""
        if self.encryption_key:
            from cryptography.fernet import Fernet
            f = Fernet(self.encryption_key)
            return f.decrypt(data)
        return data
    
    def save_model(self, model: Any, metadata: ModelMetadata) -> str:
        """Save model to MinIO with versioning and optional encryption."""
        try:
            # Serialize model
            model_bytes = pickle.dumps(model)
            
            # Encrypt if needed
            if metadata.encrypted and self.encryption_key:
                model_bytes = self._encrypt_data(model_bytes)
            
            # Compress if requested
            if metadata.compression == "gzip":
                import gzip
                model_bytes = gzip.compress(model_bytes)
            elif metadata.compression == "lz4":
                import lz4.frame
                model_bytes = lz4.frame.compress(model_bytes)
            
            # Create object path
            object_name = f"{metadata.tenant_id}/{metadata.model_id}/v{metadata.version}/model.pkl"
            metadata_name = f"{metadata.tenant_id}/{metadata.model_id}/v{metadata.version}/metadata.json"
            
            # Save model
            if self._uses_minio:
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
        """Load model from MinIO with decryption support."""
        try:
            # If no version specified, get latest
            if version is None:
                version = self._get_latest_version(model_id, tenant_id)
            
            object_name = f"{tenant_id}/{model_id}/v{version}/model.pkl"
            metadata_name = f"{tenant_id}/{model_id}/v{version}/metadata.json"
            
            if self._uses_minio:
                # Load model
                response = self.client.get_object(self.models_bucket, object_name)
                model_bytes = response.read()
                response.close()
                
                # Load metadata
                response = self.client.get_object(self.models_bucket, metadata_name)
                metadata = json.loads(response.read())
                response.close()
            else:
                # Use boto3
                response = self.client.get_object(Bucket=self.models_bucket, Key=object_name)
                model_bytes = response['Body'].read()
                
                response = self.client.get_object(Bucket=self.models_bucket, Key=metadata_name)
                metadata = json.loads(response['Body'].read())
            
            # Decompress if needed
            if metadata.get('compression') == "gzip":
                import gzip
                model_bytes = gzip.decompress(model_bytes)
            elif metadata.get('compression') == "lz4":
                import lz4.frame
                model_bytes = lz4.frame.decompress(model_bytes)
            
            # Decrypt if needed
            if metadata.get('encrypted') and self.encryption_key:
                model_bytes = self._decrypt_data(model_bytes)
            
            # Deserialize model
            model = pickle.loads(model_bytes)
            
            logger.info(f"Model loaded: {model_id} v{version}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_latest_version(self, model_id: str, tenant_id: str) -> str:
        """Get latest version of a model."""
        prefix = f"{tenant_id}/{model_id}/"
        versions = []
        
        try:
            if self._uses_minio:
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
    
    def save_dataset(self, data: pd.DataFrame, dataset_id: str, tenant_id: str = "default",
                    format: str = "parquet", compression: str = "snappy") -> str:
        """Save dataset to MinIO in specified format."""
        try:
            # Convert to bytes based on format
            if format == "parquet":
                buffer = BytesIO()
                data.to_parquet(buffer, engine='pyarrow', compression=compression)
                data_bytes = buffer.getvalue()
                extension = "parquet"
            elif format == "csv":
                data_bytes = data.to_csv(index=False).encode()
                extension = "csv"
            elif format == "json":
                data_bytes = data.to_json(orient='records').encode()
                extension = "json"
            elif format == "feather":
                buffer = BytesIO()
                data.to_feather(buffer)
                data_bytes = buffer.getvalue()
                extension = "feather"
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Calculate hash for deduplication
            data_hash = hashlib.sha256(data_bytes).hexdigest()[:16]
            
            # Create object path
            object_name = f"{tenant_id}/{dataset_id}_{data_hash}.{extension}"
            metadata_name = f"{tenant_id}/{dataset_id}_{data_hash}_metadata.json"
            
            # Dataset metadata
            dataset_metadata = {
                "dataset_id": dataset_id,
                "hash": data_hash,
                "shape": list(data.shape),
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "format": format,
                "compression": compression,
                "created_at": datetime.now().isoformat(),
                "tenant_id": tenant_id,
                "size_bytes": len(data_bytes)
            }
            
            if self._uses_minio:
                # Save dataset
                self.client.put_object(
                    self.datasets_bucket,
                    object_name,
                    BytesIO(data_bytes),
                    len(data_bytes)
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
                    Body=data_bytes
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
        """Load dataset from MinIO."""
        try:
            # Find dataset file
            prefix = f"{tenant_id}/{dataset_id}"
            
            if self._uses_minio:
                objects = list(self.client.list_objects(self.datasets_bucket, prefix=prefix))
                dataset_files = [obj.object_name for obj in objects 
                               if not obj.object_name.endswith('_metadata.json')]
            else:
                response = self.client.list_objects_v2(
                    Bucket=self.datasets_bucket,
                    Prefix=prefix
                )
                dataset_files = [obj['Key'] for obj in response.get('Contents', []) 
                               if not obj['Key'].endswith('_metadata.json')]
            
            if not dataset_files:
                raise FileNotFoundError(f"Dataset {dataset_id} not found")
            
            # Load the most recent one
            object_name = sorted(dataset_files)[-1]
            
            if self._uses_minio:
                response = self.client.get_object(self.datasets_bucket, object_name)
                data_bytes = response.read()
                response.close()
            else:
                response = self.client.get_object(Bucket=self.datasets_bucket, Key=object_name)
                data_bytes = response['Body'].read()
            
            # Load based on extension
            if object_name.endswith('.parquet'):
                data = pd.read_parquet(BytesIO(data_bytes))
            elif object_name.endswith('.csv'):
                data = pd.read_csv(BytesIO(data_bytes))
            elif object_name.endswith('.json'):
                data = pd.read_json(BytesIO(data_bytes))
            elif object_name.endswith('.feather'):
                data = pd.read_feather(BytesIO(data_bytes))
            else:
                raise ValueError(f"Unknown file format: {object_name}")
            
            logger.info(f"Dataset loaded: {dataset_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def list_models(self, tenant_id: str = None) -> List[Dict]:
        """List all models for a tenant."""
        models = []
        prefix = f"{tenant_id}/" if tenant_id else ""
        
        try:
            if self._uses_minio:
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
    """Local filesystem storage backend for development."""
    
    def __init__(self, base_path: str = "./ml_storage", encryption_key: bytes = None):
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.datasets_path = self.base_path / "datasets"
        self.artifacts_path = self.base_path / "artifacts"
        self.encryption_key = encryption_key
        
        # Create directories
        for path in [self.models_path, self.datasets_path, self.artifacts_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data if encryption key is provided."""
        if self.encryption_key:
            from cryptography.fernet import Fernet
            f = Fernet(self.encryption_key)
            return f.encrypt(data)
        return data
    
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data if encryption key is provided."""
        if self.encryption_key:
            from cryptography.fernet import Fernet
            f = Fernet(self.encryption_key)
            return f.decrypt(data)
        return data
    
    def save_model(self, model: Any, metadata: ModelMetadata) -> str:
        """Save model to local filesystem with encryption support."""
        try:
            model_dir = self.models_path / metadata.tenant_id / metadata.model_id / f"v{metadata.version}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Serialize model
            model_bytes = pickle.dumps(model)
            
            # Encrypt if needed
            if metadata.encrypted and self.encryption_key:
                model_bytes = self._encrypt_data(model_bytes)
            
            # Save model
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                f.write(model_bytes)
            
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
        """Load model from local filesystem."""
        try:
            if version is None:
                version = self._get_latest_version(model_id, tenant_id)
            
            model_dir = self.models_path / tenant_id / model_id / f"v{version}"
            
            # Load metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load model
            model_path = model_dir / "model.pkl"
            with open(model_path, 'rb') as f:
                model_bytes = f.read()
            
            # Decrypt if needed
            if metadata.get('encrypted') and self.encryption_key:
                model_bytes = self._decrypt_data(model_bytes)
            
            # Deserialize model
            model = pickle.loads(model_bytes)
            
            logger.info(f"Model loaded: {model_id} v{version}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_latest_version(self, model_id: str, tenant_id: str) -> str:
        """Get latest version of a model."""
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
    
    def save_dataset(self, data: pd.DataFrame, dataset_id: str, tenant_id: str = "default",
                    format: str = "parquet", compression: str = "snappy") -> str:
        """Save dataset to local filesystem."""
        try:
            dataset_dir = self.datasets_path / tenant_id
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Save based on format
            data_hash = hashlib.sha256(data.to_csv().encode()).hexdigest()[:16]
            
            if format == "parquet":
                dataset_path = dataset_dir / f"{dataset_id}_{data_hash}.parquet"
                data.to_parquet(dataset_path, engine='pyarrow', compression=compression)
            elif format == "csv":
                dataset_path = dataset_dir / f"{dataset_id}_{data_hash}.csv"
                data.to_csv(dataset_path, index=False)
            elif format == "json":
                dataset_path = dataset_dir / f"{dataset_id}_{data_hash}.json"
                data.to_json(dataset_path, orient='records')
            elif format == "feather":
                dataset_path = dataset_dir / f"{dataset_id}_{data_hash}.feather"
                data.to_feather(dataset_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Save metadata
            metadata = {
                "dataset_id": dataset_id,
                "hash": data_hash,
                "shape": list(data.shape),
                "columns": list(data.columns),
                "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
                "format": format,
                "compression": compression,
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
        """Load dataset from local filesystem."""
        try:
            dataset_dir = self.datasets_path / tenant_id
            
            # Find dataset file
            dataset_files = list(dataset_dir.glob(f"{dataset_id}_*"))
            dataset_files = [f for f in dataset_files if not f.name.endswith('_metadata.json')]
            
            if not dataset_files:
                raise FileNotFoundError(f"Dataset {dataset_id} not found")
            
            # Load the most recent one
            dataset_path = sorted(dataset_files, key=lambda x: x.stat().st_mtime)[-1]
            
            # Load based on extension
            if dataset_path.suffix == '.parquet':
                data = pd.read_parquet(dataset_path)
            elif dataset_path.suffix == '.csv':
                data = pd.read_csv(dataset_path)
            elif dataset_path.suffix == '.json':
                data = pd.read_json(dataset_path)
            elif dataset_path.suffix == '.feather':
                data = pd.read_feather(dataset_path)
            else:
                raise ValueError(f"Unknown file format: {dataset_path}")
            
            logger.info(f"Dataset loaded: {dataset_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def list_models(self, tenant_id: str = None) -> List[Dict]:
        """List all models for a tenant."""
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
        """Delete a model version or all versions."""
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


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        models_bucket: str = "models",
        datasets_bucket: str = "datasets",
        artifacts_bucket: str = "artifacts",
        encryption_key: Optional[bytes] = None,
        client: Optional[Any] = None,
    ) -> None:
        if not GCS_AVAILABLE and client is None:
            raise ImportError("google-cloud-storage is required for the GCS backend")

        self.encryption_key = encryption_key

        if client is not None:
            self.client = client
        else:  # pragma: no cover - requires external dependency
            credentials = None
            if credentials_path and service_account is not None:
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = gcs_storage.Client(project=project_id, credentials=credentials)

        self.models_bucket = models_bucket
        self.datasets_bucket = datasets_bucket
        self.artifacts_bucket = artifacts_bucket

        self._ensure_buckets()

    def _get_bucket(self, bucket_name: str):
        if hasattr(self.client, "lookup_bucket"):
            bucket = self.client.lookup_bucket(bucket_name)
        else:
            bucket = None

        if bucket is None and hasattr(self.client, "create_bucket"):
            try:
                bucket = self.client.create_bucket(bucket_name)
                logger.info(f"Created GCS bucket: {bucket_name}")
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Could not create bucket {bucket_name}: {exc}")
        if bucket is None and hasattr(self.client, "bucket"):
            bucket = self.client.bucket(bucket_name)
        return bucket

    def _ensure_buckets(self) -> None:
        for bucket_name in [self.models_bucket, self.datasets_bucket, self.artifacts_bucket]:
            bucket = None
            try:
                bucket = self._get_bucket(bucket_name)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Failed to initialize bucket {bucket_name}: {exc}")
            if bucket is None:
                logger.error(f"Bucket {bucket_name} could not be initialized for GCS backend")

    def _encrypt_data(self, data: bytes) -> bytes:
        if self.encryption_key:
            from cryptography.fernet import Fernet
            fernet = Fernet(self.encryption_key)
            return fernet.encrypt(data)
        return data

    def _decrypt_data(self, data: bytes) -> bytes:
        if self.encryption_key:
            from cryptography.fernet import Fernet
            fernet = Fernet(self.encryption_key)
            return fernet.decrypt(data)
        return data

    def save_model(self, model: Any, metadata: ModelMetadata) -> str:
        try:
            bucket = self._get_bucket(self.models_bucket)
            if bucket is None:
                raise RuntimeError("Models bucket is not available for GCS backend")

            model_bytes = pickle.dumps(model)
            if metadata.encrypted and self.encryption_key:
                model_bytes = self._encrypt_data(model_bytes)

            if metadata.compression == "gzip":
                import gzip
                model_bytes = gzip.compress(model_bytes)
            elif metadata.compression == "lz4":
                import lz4.frame
                model_bytes = lz4.frame.compress(model_bytes)

            object_name = f"{metadata.tenant_id}/{metadata.model_id}/v{metadata.version}/model.pkl"
            metadata_name = f"{metadata.tenant_id}/{metadata.model_id}/v{metadata.version}/metadata.json"

            model_blob = bucket.blob(object_name)
            model_buffer = BytesIO(model_bytes)
            model_buffer.seek(0)
            model_blob.upload_from_file(model_buffer, rewind=True)

            metadata_blob = bucket.blob(metadata_name)
            metadata_blob.upload_from_string(json.dumps(metadata.to_dict()), content_type="application/json")

            logger.info(f"Model saved to GCS: {metadata.model_id} v{metadata.version}")
            return object_name
        except Exception as exc:
            logger.error(f"Failed to save model to GCS: {exc}")
            raise

    def _get_latest_version(self, bucket, model_id: str, tenant_id: str) -> Optional[str]:
        prefix = f"{tenant_id}/{model_id}/"
        versions: List[str] = []
        try:
            for blob in bucket.list_blobs(prefix=prefix):
                if blob.name.endswith("metadata.json"):
                    parts = blob.name.split("/")
                    if len(parts) >= 3 and parts[-2].startswith("v"):
                        versions.append(parts[-2][1:])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to determine latest version for {model_id}: {exc}")

        if versions:
            return sorted(versions, key=lambda x: tuple(int(p) for p in x.split('.')))[-1]
        return None

    def load_model(self, model_id: str, version: str = None, tenant_id: str = "default") -> tuple:
        bucket = self._get_bucket(self.models_bucket)
        if bucket is None:
            raise RuntimeError("Models bucket is not available for GCS backend")

        if version is None:
            version = self._get_latest_version(bucket, model_id, tenant_id)
            if version is None:
                raise FileNotFoundError(f"No versions found for model {model_id}")

        metadata_name = f"{tenant_id}/{model_id}/v{version}/metadata.json"
        model_name = f"{tenant_id}/{model_id}/v{version}/model.pkl"

        metadata_blob = bucket.blob(metadata_name)
        model_blob = bucket.blob(model_name)

        metadata_bytes = metadata_blob.download_as_bytes()
        metadata = json.loads(metadata_bytes.decode())

        model_bytes = model_blob.download_as_bytes()

        if metadata.get("compression") == "gzip":
            import gzip
            model_bytes = gzip.decompress(model_bytes)
        elif metadata.get("compression") == "lz4":
            import lz4.frame
            model_bytes = lz4.frame.decompress(model_bytes)

        if metadata.get("encrypted") and self.encryption_key:
            model_bytes = self._decrypt_data(model_bytes)

        model = pickle.loads(model_bytes)
        logger.info(f"Model loaded from GCS: {model_id} v{version}")
        return model, metadata

    def save_dataset(
        self,
        data: pd.DataFrame,
        dataset_id: str,
        tenant_id: str = "default",
        format: str = "parquet",
        compression: str = "snappy",
    ) -> str:
        bucket = self._get_bucket(self.datasets_bucket)
        if bucket is None:
            raise RuntimeError("Datasets bucket is not available for GCS backend")

        data_hash = hashlib.sha256(data.to_csv().encode()).hexdigest()[:16]
        extension_map = {
            "parquet": "parquet",
            "csv": "csv",
            "json": "json",
            "feather": "feather",
        }

        if format not in extension_map:
            raise ValueError(f"Unsupported format: {format}")

        extension = extension_map[format]
        object_name = f"{tenant_id}/{dataset_id}_{data_hash}.{extension}"
        metadata_name = f"{tenant_id}/{dataset_id}_{data_hash}_metadata.json"

        if format == "parquet":
            buffer = BytesIO()
            data.to_parquet(buffer, engine="pyarrow", compression=compression)
        elif format == "csv":
            text_buffer = StringIO()
            data.to_csv(text_buffer, index=False)
            buffer = BytesIO(text_buffer.getvalue().encode())
        elif format == "json":
            text_buffer = StringIO()
            data.to_json(text_buffer, orient="records")
            buffer = BytesIO(text_buffer.getvalue().encode())
        elif format == "feather":
            buffer = BytesIO()
            data.to_feather(buffer)
        else:  # pragma: no cover - guarded above
            raise ValueError(f"Unsupported format: {format}")

        buffer.seek(0)

        dataset_blob = bucket.blob(object_name)
        dataset_blob.upload_from_file(buffer, rewind=True)

        metadata = {
            "dataset_id": dataset_id,
            "hash": data_hash,
            "shape": list(data.shape),
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "format": format,
            "compression": compression,
            "created_at": datetime.now().isoformat(),
            "tenant_id": tenant_id,
        }

        metadata_blob = bucket.blob(metadata_name)
        metadata_blob.upload_from_string(json.dumps(metadata), content_type="application/json")

        logger.info(f"Dataset saved to GCS: {dataset_id} (hash: {data_hash})")
        return object_name

    def load_dataset(self, dataset_id: str, tenant_id: str = "default") -> pd.DataFrame:
        bucket = self._get_bucket(self.datasets_bucket)
        if bucket is None:
            raise RuntimeError("Datasets bucket is not available for GCS backend")

        prefix = f"{tenant_id}/{dataset_id}_"
        metadata_blobs = [blob for blob in bucket.list_blobs(prefix=prefix) if blob.name.endswith("_metadata.json")]
        if not metadata_blobs:
            raise FileNotFoundError(f"Dataset {dataset_id} not found in tenant {tenant_id}")

        def _blob_updated(blob):
            return getattr(blob, "updated", datetime.min)

        metadata_blob = max(metadata_blobs, key=_blob_updated)
        metadata = json.loads(metadata_blob.download_as_bytes().decode())

        extension_map = {
            "parquet": "parquet",
            "csv": "csv",
            "json": "json",
            "feather": "feather",
        }

        extension = extension_map.get(metadata.get("format"))
        if not extension:
            raise ValueError(f"Unsupported dataset format: {metadata.get('format')}")

        dataset_blob_name = f"{tenant_id}/{metadata['dataset_id']}_{metadata['hash']}.{extension}"
        dataset_blob = bucket.blob(dataset_blob_name)
        data_bytes = dataset_blob.download_as_bytes()
        buffer = BytesIO(data_bytes)

        if metadata["format"] == "parquet":
            data = pd.read_parquet(buffer)
        elif metadata["format"] == "csv":
            data = pd.read_csv(buffer)
        elif metadata["format"] == "json":
            data = pd.read_json(buffer)
        elif metadata["format"] == "feather":
            data = pd.read_feather(buffer)
        else:  # pragma: no cover - already validated
            raise ValueError(f"Unsupported dataset format: {metadata['format']}")

        logger.info(f"Dataset loaded from GCS: {dataset_id}")
        return data

    def list_models(self, tenant_id: str = None) -> List[Dict]:
        bucket = self._get_bucket(self.models_bucket)
        if bucket is None:
            return []

        prefix = f"{tenant_id}/" if tenant_id else ""
        models = []
        for blob in bucket.list_blobs(prefix=prefix):
            if blob.name.endswith("metadata.json"):
                try:
                    metadata = json.loads(blob.download_as_bytes().decode())
                    models.append(metadata)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(f"Failed to parse metadata from {blob.name}: {exc}")
        return models

    def delete_model(self, model_id: str, version: str = None, tenant_id: str = "default") -> bool:
        bucket = self._get_bucket(self.models_bucket)
        if bucket is None:
            return False

        if version:
            prefix = f"{tenant_id}/{model_id}/v{version}/"
        else:
            prefix = f"{tenant_id}/{model_id}/"

        deleted = False
        for blob in list(bucket.list_blobs(prefix=prefix)):
            try:
                if hasattr(blob, "delete"):
                    blob.delete()
                elif hasattr(bucket, "delete_blob"):
                    bucket.delete_blob(blob.name)
                deleted = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Failed to delete blob {blob.name}: {exc}")
        return deleted


class StorageManager:
    """Main storage manager that handles backend selection and connectors.

    Note:
        The connector implementations currently live under
        :mod:`automl_platform.api.connectors`. Importing them here introduces a
        dependency from the storage layer to the API layer. This mirrors the
        existing deployment layout and unblocks environments where the storage
        service needs connector access. Longer-term, the connector classes
        should be promoted to a shared module that both layers can import
        without depending on each other transitively.
    """

    def __init__(self, backend: str = "local", **kwargs):
        """
        Initialize storage manager

        Args:
            backend: Storage backend type ('local', 'minio', 's3', 'gcs', 'none')
            **kwargs: Backend-specific configuration
        """
        self.backend_type = backend

        # Get encryption key from infrastructure if available
        encryption_key = kwargs.pop('encryption_key', None)

        if backend == "none":
            # Null backend to explicitly disable persistence while keeping a
            # consistent StorageManager interface for downstream services.
            self.backend = NullStorage()
        elif backend in {"minio", "s3"}:
            self.backend = MinIOStorage(encryption_key=encryption_key, **kwargs)
        elif backend == "gcs":
            self.backend = GCSStorage(encryption_key=encryption_key, **kwargs)
        else:
            self.backend = LocalStorage(encryption_key=encryption_key, **kwargs)

        # Initialize connectors
        self.connectors = {}
        self._connector_config_type = None
        self._init_connectors()

    def _init_connectors(self):
        """Initialize data source connectors."""
        from importlib import import_module

        connectors_module = import_module("automl_platform.api.connectors")

        # Register available connectors
        self.connectors = {
            'snowflake': connectors_module.SnowflakeConnector,
            'bigquery': connectors_module.BigQueryConnector,
            'postgresql': connectors_module.PostgreSQLConnector,
            'mongodb': connectors_module.MongoDBConnector
        }

        # Store ConnectionConfig for runtime instantiation.
        # ``_connector_config_cls`` is kept for backward compatibility with
        # callers that accessed the attribute directly in tests.
        self._connector_config_type = connectors_module.ConnectionConfig
        self._connector_config_cls = self._connector_config_type

    def load_from_connector(self, connector_type: str, config: Dict[str, Any],
                          query: str = None, table: str = None) -> pd.DataFrame:
        """Load data from external connector."""
        if connector_type not in self.connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")

        connector_class = self.connectors[connector_type]

        normalized_config = dict(config)
        alias_map = {
            'user': 'username',
            'dbname': 'database',
        }
        used_aliases: List[str] = []
        for alias, target in alias_map.items():
            if alias in normalized_config and target not in normalized_config:
                normalized_config[target] = normalized_config.pop(alias)
                used_aliases.append(alias)

        if used_aliases:
            warnings.warn(
                "The connection configuration keys {} are deprecated; use {} "
                "instead.".format(
                    ', '.join(sorted(used_aliases)),
                    ', '.join(alias_map[alias] for alias in sorted(used_aliases)),
                ),
                DeprecationWarning,
                stacklevel=2,
            )

        normalized_config.setdefault('connection_type', connector_type)

        if self._connector_config_type is None:
            raise RuntimeError("Connector configuration type is not initialised")

        connector_config = self._connector_config_type(**normalized_config)
        connector = connector_class(connector_config)
        
        if query:
            return connector.query(query)
        elif table:
            return connector.read_table(table)
        else:
            raise ValueError("Either query or table must be specified")
    
    def save_model(self, model: Any, metadata: Dict, version: str = None,
                  encryption: bool = False, compression: str = "none") -> str:
        """Save a model with metadata."""
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
            tenant_id=metadata.get('tenant_id', 'default'),
            encrypted=encryption,
            compression=compression
        )
        
        return self.backend.save_model(model, model_metadata)
    
    def load_model(self, model_id: str, version: str = None, tenant_id: str = "default"):
        """Load a model."""
        return self.backend.load_model(model_id, version, tenant_id)
    
    def save_dataset(self, data: pd.DataFrame, dataset_id: str, tenant_id: str = "default",
                    format: str = "parquet", compression: str = "snappy"):
        """Save a dataset."""
        return self.backend.save_dataset(data, dataset_id, tenant_id, format, compression)
    
    def load_dataset(self, dataset_id: str, tenant_id: str = "default"):
        """Load a dataset."""
        return self.backend.load_dataset(dataset_id, tenant_id)
    
    def list_models(self, tenant_id: str = None):
        """List all models."""
        return self.backend.list_models(tenant_id)
    
    def delete_model(self, model_id: str, version: str = None, tenant_id: str = "default"):
        """Delete a model."""
        if hasattr(self.backend, 'delete_model'):
            return self.backend.delete_model(model_id, version, tenant_id)
        return False
    
    def get_model_history(self, model_id: str, tenant_id: str = "default") -> List[Dict]:
        """Get version history of a model."""
        all_models = self.list_models(tenant_id)
        return [m for m in all_models if m.get('model_id') == model_id]
    
    # Export methods
    def export_model_to_docker(self, model_id: str, version: str = None, 
                              tenant_id: str = "default", output_dir: str = "./docker_export") -> str:
        """Export model as Docker container with FastAPI serving."""
        model, metadata = self.load_model(model_id, version, tenant_id)
        
        output_path = Path(output_dir) / f"{model_id}_v{version or 'latest'}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = output_path / "model.pkl"
        joblib.dump(model, model_file)
        
        # Create FastAPI app with enhanced features
        app_content = f'''from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

app = FastAPI(title="Model {model_id}", version="{version or 'latest'}")
security = HTTPBearer()
model = joblib.load("model.pkl")
metadata = {json.dumps(metadata, indent=2)}

class PredictRequest(BaseModel):
    data: Dict[str, Any]
    return_probabilities: bool = False

class BatchPredictRequest(BaseModel):
    data: List[Dict[str, Any]]
    return_probabilities: bool = False

class HealthResponse(BaseModel):
    status: str
    model_id: str
    version: str
    uptime: float
    
startup_time = time.time()

@app.get("/")
def info():
    return metadata

@app.get("/health", response_model=HealthResponse)
def health():
    return {{
        "status": "healthy",
        "model_id": "{model_id}",
        "version": "{version or 'latest'}",
        "uptime": time.time() - startup_time
    }}

@app.post("/predict")
def predict(request: PredictRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        df = pd.DataFrame([request.data])
        prediction = model.predict(df)
        
        response = {{
            "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            "timestamp": datetime.now().isoformat()
        }}
        
        if request.return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)
            response["probabilities"] = probabilities.tolist()
        
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_batch")
def predict_batch(request: BatchPredictRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        df = pd.DataFrame(request.data)
        predictions = model.predict(df)
        
        response = {{
            "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            "timestamp": datetime.now().isoformat(),
            "batch_size": len(request.data)
        }}
        
        if request.return_probabilities and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)
            response["probabilities"] = probabilities.tolist()
        
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
def metrics():
    """Prometheus-compatible metrics endpoint"""
    return {{
        "predictions_total": 0,  # Would track in production
        "prediction_duration_seconds": 0,
        "model_version": "{version or 'latest'}"
    }}
'''
        
        (output_path / "app.py").write_text(app_content)
        
        # Create requirements.txt
        requirements = """fastapi==0.104.1
uvicorn==0.24.0
pandas==2.0.3
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.1.0
catboost==1.2.2
joblib==1.3.2
numpy==1.24.3
python-multipart==0.0.6"""
        
        (output_path / "requirements.txt").write_text(requirements)
        
        # Create Dockerfile with multi-stage build
        dockerfile = f"""# Multi-stage build for smaller image
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY app.py .
COPY model.pkl .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        (output_path / "Dockerfile").write_text(dockerfile)
        
        # Create docker-compose.yml with monitoring
        compose = f"""version: '3.8'

services:
  model-api:
    build: .
    container_name: {model_id}-api
    ports:
      - "8000:8000"
    environment:
      - MODEL_ID={model_id}
      - MODEL_VERSION={version or 'latest'}
      - LOG_LEVEL=info
    restart: unless-stopped
    networks:
      - model-network
    labels:
      - "prometheus.io/scrape=true"
      - "prometheus.io/port=8000"
      - "prometheus.io/path=/metrics"

networks:
  model-network:
    driver: bridge
"""
        (output_path / "docker-compose.yml").write_text(compose)
        
        # Create Kubernetes deployment
        k8s_yaml = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {model_id}-deployment
  labels:
    app: {model_id}
spec:
  replicas: 3
  selector:
    matchLabels:
      app: {model_id}
  template:
    metadata:
      labels:
        app: {model_id}
    spec:
      containers:
      - name: model
        image: {model_id}:{version or 'latest'}
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {model_id}-service
spec:
  selector:
    app: {model_id}
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
"""
        (output_path / "k8s-deployment.yaml").write_text(k8s_yaml)
        
        # Create deployment script
        deploy_script = f"""#!/bin/bash
# Build and deploy model container

echo "Building Docker image for {model_id}..."
docker build -t {model_id}:{version or 'latest'} .

echo "Starting container..."
docker-compose up -d

echo "Model API available at http://localhost:8000"
echo "View logs: docker-compose logs -f"
echo "Stop: docker-compose down"

# Optional: Deploy to Kubernetes
# kubectl apply -f k8s-deployment.yaml
"""
        deploy_path = output_path / "deploy.sh"
        deploy_path.write_text(deploy_script)
        deploy_path.chmod(0o755)
        
        logger.info(f"Docker export completed at {output_path}")
        return str(output_path)
    
    def export_model_to_onnx(self, model_id: str, version: str = None,
                            tenant_id: str = "default", output_path: str = None) -> str:
        """Export model to ONNX format."""
        if not ONNX_AVAILABLE:
            logger.error("skl2onnx not installed. Install with: pip install skl2onnx")
            raise ImportError("skl2onnx is required for ONNX export")

        try:
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError as exc:  # pragma: no cover - optional dependency
            if ONNX_AVAILABLE and convert_sklearn is not None:
                warnings.warn(
                    "skl2onnx.common.data_types is unavailable; using a stub "
                    "FloatTensorType for testing purposes.",
                    RuntimeWarning,
                    stacklevel=2,
                )

                class FloatTensorType:  # type: ignore[no-redef]
                    def __init__(self, *args, **kwargs):
                        self.args = args
                        self.kwargs = kwargs
            else:
                logger.error("skl2onnx.common is unavailable: %s", exc)
                raise

        try:
            model, metadata = self.load_model(model_id, version, tenant_id)

            if output_path is None:
                output_path = f"./{model_id}_v{version or 'latest'}.onnx"

            # Define input type based on feature count
            n_features = len(metadata.get('feature_names', []))
            if n_features == 0:
                raise ValueError("Feature names not found in metadata")

            initial_type = [('float_input', FloatTensorType([None, n_features]))]

            # Convert to ONNX
            if convert_sklearn is None:  # pragma: no cover - defensive
                raise RuntimeError("convert_sklearn is unavailable")

            onx = convert_sklearn(model, initial_types=initial_type, target_opset=12)

            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onx.SerializeToString())

            logger.info(f"ONNX export completed at {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
            raise
    
    def export_model_to_pmml(self, model_id: str, version: str = None,
                            tenant_id: str = "default", output_path: str = None) -> str:
        """Export model to PMML format."""
        if not PMML_AVAILABLE:
            logger.error(
                "sklearn2pmml not installed. Install with: pip install sklearn2pmml"
            )
            raise ImportError("sklearn2pmml is required for PMML export")

        try:
            model, metadata = self.load_model(model_id, version, tenant_id)

            if output_path is None:
                output_path = f"./{model_id}_v{version or 'latest'}.pmml"

            # Wrap model in PMML pipeline if needed
            pipeline_cls = PMMLPipeline
            if pipeline_cls is None:  # pragma: no cover - fallback for patched tests
                class _FallbackPMMLPipeline:
                    def __init__(self, steps):
                        self.steps = steps

                pipeline_cls = _FallbackPMMLPipeline

            if not isinstance(model, pipeline_cls):
                pmml_pipeline = pipeline_cls([("model", model)])
            else:
                pmml_pipeline = model

            # Export to PMML
            if sklearn2pmml is None:  # pragma: no cover - defensive
                raise RuntimeError("sklearn2pmml callable is unavailable")

            sklearn2pmml(pmml_pipeline, output_path, with_repr=True)

            logger.info(f"PMML export completed at {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to export to PMML: {e}")
            raise
    
    def export_model_to_tensorflow(self, model_id: str, version: str = None,
                                  tenant_id: str = "default", output_path: str = None) -> str:
        """Export model to TensorFlow SavedModel format for TF Serving."""
        try:
            import tensorflow as tf
            
            model, metadata = self.load_model(model_id, version, tenant_id)
            
            if output_path is None:
                output_path = f"./{model_id}_tf_v{version or 'latest'}"
            
            # Create a simple TF model that wraps sklearn predictions
            n_features = len(metadata.get('feature_names', []))
            
            @tf.function
            def predict_fn(inputs):
                # Convert TF tensor to numpy for sklearn
                numpy_inputs = inputs.numpy()
                predictions = model.predict(numpy_inputs)
                return tf.constant(predictions, dtype=tf.float32)
            
            # Create TF SavedModel
            module = tf.Module()
            module.predict = tf.function(
                func=predict_fn,
                input_signature=[tf.TensorSpec(shape=[None, n_features], dtype=tf.float32)]
            )
            
            tf.saved_model.save(module, output_path)
            
            logger.info(f"TensorFlow SavedModel export completed at {output_path}")
            return output_path
            
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise
        except Exception as e:
            logger.error(f"Failed to export to TensorFlow: {e}")
            raise


class FeatureStore:
    """Feature store for feature reusability and versioning.

    The in-memory cache now enforces conservative defaults (100 entries, 500 MB, 1 hour
    TTL) and can be tuned per deployment.  Evictions use an LRU policy guarded by a
    re-entrant lock so concurrent workers see a consistent view.  Cache metrics such as
    hits, misses, evictions, and memory consumption are tracked for observability and
    logged whenever pruning occurs.
    """

    @dataclass
    class _FeatureCacheEntry:
        data: pd.DataFrame
        timestamp: float
        size_bytes: int

    def __init__(
        self,
        storage_manager: StorageManager,
        cache_max_entries: int = 100,
        cache_max_memory_mb: Optional[float] = 500,
        cache_ttl_seconds: Optional[float] = 3600,
        time_provider: Callable[[], float] = time.time,
    ):
        self.storage = storage_manager
        self._cache_max_entries = max(0, cache_max_entries or 0)
        self._cache_max_memory_bytes: Optional[int] = (
            int(cache_max_memory_mb * 1024 * 1024)
            if cache_max_memory_mb and cache_max_memory_mb > 0
            else None
        )
        self._cache_ttl_seconds = (
            cache_ttl_seconds if cache_ttl_seconds and cache_ttl_seconds > 0 else None
        )
        self._time_provider = time_provider
        self.feature_cache: "OrderedDict[str, FeatureStore._FeatureCacheEntry]" = (
            OrderedDict()
        )
        self._cache_current_memory = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._cache_lock = threading.RLock()
        self.feature_registry = {}

    def _cache_enabled(self) -> bool:
        return self._cache_max_entries > 0 or self._cache_max_memory_bytes is not None

    def _current_time(self) -> float:
        return self._time_provider()

    def _make_cache_key(self, feature_set_name: str, version: Optional[str]) -> str:
        return f"{feature_set_name}:{version}" if version else feature_set_name

    def _estimate_dataframe_size(self, features: pd.DataFrame) -> int:
        try:
            return int(features.memory_usage(deep=True).sum())
        except ValueError:
            return int(features.memory_usage().sum())

    def _log_cache_summary_locked(self, level: int = logging.DEBUG) -> None:
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests) if total_requests else 0.0
        logger.log(
            level,
            (
                "FeatureStore cache stats: entries=%d/%s memory=%d/%s "
                "hits=%d misses=%d evictions=%d hit_rate=%.2f"
            ),
            len(self.feature_cache),
            self._cache_max_entries if self._cache_max_entries else "unbounded",
            self._cache_current_memory,
            self._cache_max_memory_bytes
            if self._cache_max_memory_bytes is not None
            else "unbounded",
            self._cache_hits,
            self._cache_misses,
            self._cache_evictions,
            hit_rate,
        )

    def _prune_expired_locked(self) -> None:
        if not self._cache_enabled():
            if self.feature_cache:
                self.feature_cache.clear()
                self._cache_current_memory = 0
            return

        if self._cache_ttl_seconds is None:
            return

        now = self._current_time()
        expired_keys: List[str] = []
        for key, entry in list(self.feature_cache.items()):
            if now - entry.timestamp >= self._cache_ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            entry = self.feature_cache.pop(key, None)
            if entry:
                self._cache_current_memory -= entry.size_bytes
                self._cache_evictions += 1
                logger.info(
                    "Evicting feature cache entry %s due to TTL expiry (ttl=%ss)",
                    key,
                    self._cache_ttl_seconds,
                )
                self._log_cache_summary_locked(level=logging.INFO)

    def _evict_excess_entries_locked(self) -> None:
        if self._cache_max_entries <= 0:
            return

        while len(self.feature_cache) > self._cache_max_entries:
            key, entry = self.feature_cache.popitem(last=False)
            self._cache_current_memory -= entry.size_bytes
            self._cache_evictions += 1
            logger.info(
                "Evicting feature cache entry %s to enforce max entries %d",
                key,
                self._cache_max_entries,
            )
            self._log_cache_summary_locked(level=logging.INFO)

    def _evict_for_memory_pressure_locked(self) -> None:
        if self._cache_max_memory_bytes is None:
            return

        while (
            self.feature_cache
            and self._cache_current_memory > self._cache_max_memory_bytes
        ):
            key, entry = self.feature_cache.popitem(last=False)
            self._cache_current_memory -= entry.size_bytes
            self._cache_evictions += 1
            logger.info(
                "Evicting feature cache entry %s to enforce max memory %d bytes",
                key,
                self._cache_max_memory_bytes,
            )
            self._log_cache_summary_locked(level=logging.INFO)

    def _store_in_cache(self, cache_key: str, features: pd.DataFrame) -> None:
        if not self._cache_enabled():
            return

        with self._cache_lock:
            self._prune_expired_locked()
            size_bytes = self._estimate_dataframe_size(features)
            if (
                self._cache_max_memory_bytes is not None
                and size_bytes > self._cache_max_memory_bytes
            ):
                logger.warning(
                    "Skipping cache for %s because entry size %d bytes exceeds max %d bytes",
                    cache_key,
                    size_bytes,
                    self._cache_max_memory_bytes,
                )
                return

            existing = self.feature_cache.pop(cache_key, None)
            if existing:
                self._cache_current_memory -= existing.size_bytes

            entry = FeatureStore._FeatureCacheEntry(
                data=features,
                timestamp=self._current_time(),
                size_bytes=size_bytes,
            )
            self.feature_cache[cache_key] = entry
            self.feature_cache.move_to_end(cache_key)
            self._cache_current_memory += size_bytes

            self._evict_excess_entries_locked()
            self._evict_for_memory_pressure_locked()
            self._log_cache_summary_locked()

    def _get_cached_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        if not self._cache_enabled():
            return None

        with self._cache_lock:
            self._prune_expired_locked()
            entry = self.feature_cache.get(cache_key)
            if entry is None:
                self._cache_misses += 1
                self._log_cache_summary_locked()
                return None

            self.feature_cache.move_to_end(cache_key)
            self._cache_hits += 1
            self._log_cache_summary_locked()
            return entry.data

    def get_cache_stats(self) -> Dict[str, Union[int, Optional[int]]]:
        with self._cache_lock:
            self._prune_expired_locked()
            return {
                "entries": len(self.feature_cache),
                "max_entries": self._cache_max_entries if self._cache_max_entries else None,
                "current_memory_bytes": self._cache_current_memory,
                "max_memory_bytes": self._cache_max_memory_bytes,
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "evictions": self._cache_evictions,
            }

    def get_cached_keys(self) -> List[str]:
        with self._cache_lock:
            self._prune_expired_locked()
            return list(self.feature_cache.keys())

    @property
    def cache_max_entries(self) -> int:
        return self._cache_max_entries

    @property
    def cache_max_memory_bytes(self) -> Optional[int]:
        return self._cache_max_memory_bytes

    @property
    def cache_ttl_seconds(self) -> Optional[float]:
        return self._cache_ttl_seconds

    def register_feature(self, name: str, description: str,
                        computation_func: callable, dependencies: List[str] = None):
        """Register a feature computation."""
        self.feature_registry[name] = {
            "description": description,
            "computation": computation_func,
            "dependencies": dependencies or [],
            "created_at": datetime.now().isoformat()
        }
    
    def compute_features(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """Compute requested features."""
        result = df.copy()
        
        for feature_name in feature_names:
            if feature_name in self.feature_registry:
                feature_def = self.feature_registry[feature_name]
                
                # Check dependencies
                for dep in feature_def["dependencies"]:
                    if dep not in result.columns:
                        # Recursively compute dependency
                        result = self.compute_features(result, [dep])
                
                # Compute feature
                result[feature_name] = feature_def["computation"](result)
        
        return result
    
    def save_features(self, 
                     features: pd.DataFrame,
                     feature_set_name: str,
                     version: str = "1.0.0",
                     metadata: Dict = None) -> str:
        """Save computed features for reuse."""
        
        feature_metadata = {
            "feature_set_name": feature_set_name,
            "version": version,
            "columns": list(features.columns),
            "shape": list(features.shape),
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "registry": self.feature_registry
        }
        
        # Save as dataset with special naming
        dataset_id = f"features_{feature_set_name}_v{version}"
        path = self.storage.save_dataset(features, dataset_id)

        # Cache for quick access
        cache_key = self._make_cache_key(feature_set_name, version)
        self._store_in_cache(cache_key, features)

        logger.info(f"Features saved: {feature_set_name} v{version}")
        return path

    def load_features(self,
                     feature_set_name: str,
                     version: str = None) -> pd.DataFrame:
        """Load features from store."""

        # Check cache first
        cache_key = self._make_cache_key(feature_set_name, version)
        cached = self._get_cached_features(cache_key)
        if cached is not None:
            return cached

        # Load from storage
        dataset_id = f"features_{feature_set_name}"
        if version:
            dataset_id += f"_v{version}"

        features = self.storage.load_dataset(dataset_id)

        # Update cache
        self._store_in_cache(cache_key, features)

        return features

    def list_feature_sets(self) -> List[str]:
        """List available feature sets."""
        with self._cache_lock:
            self._prune_expired_locked()
            return list({k.split(':')[0] for k in self.feature_cache.keys()})


# Backward compatibility alias for modules expecting StorageService
StorageService = StorageManager
