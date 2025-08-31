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
from typing import Any, Dict, List, Optional, Union, BinaryIO, Tuple
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from io import BytesIO

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


class MinIOStorage(StorageBackend):
    """MinIO/S3 compatible storage backend with encryption support."""
    
    def __init__(self, 
                 endpoint: str = "localhost:9000",
                 access_key: str = "minioadmin",
                 secret_key: str = "minioadmin",
                 secure: bool = False,
                 region: str = "us-east-1",
                 encryption_key: bytes = None):
        
        self.encryption_key = encryption_key
        
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
        """Create required buckets if they don't exist."""
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
        """Load model from MinIO with decryption support."""
        try:
            # If no version specified, get latest
            if version is None:
                version = self._get_latest_version(model_id, tenant_id)
            
            object_name = f"{tenant_id}/{model_id}/v{version}/model.pkl"
            metadata_name = f"{tenant_id}/{model_id}/v{version}/metadata.json"
            
            if MINIO_AVAILABLE:
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
            
            if MINIO_AVAILABLE:
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
            
            if MINIO_AVAILABLE:
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
            
            if MINIO_AVAILABLE:
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


class StorageManager:
    """Main storage manager that handles backend selection and connectors."""
    
    def __init__(self, backend: str = "local", **kwargs):
        """
        Initialize storage manager
        
        Args:
            backend: Storage backend type ('local', 'minio', 's3')
            **kwargs: Backend-specific configuration
        """
        self.backend_type = backend
        
        # Get encryption key from infrastructure if available
        encryption_key = kwargs.pop('encryption_key', None)
        
        if backend == "minio" or backend == "s3":
            self.backend = MinIOStorage(encryption_key=encryption_key, **kwargs)
        else:
            self.backend = LocalStorage(encryption_key=encryption_key, **kwargs)
        
        # Initialize connectors
        self.connectors = {}
        self._init_connectors()
    
    def _init_connectors(self):
        """Initialize data source connectors."""
        from .connectors import (
            SnowflakeConnector, BigQueryConnector, 
            PostgreSQLConnector, MongoDBConnector
        )
        
        # Register available connectors
        self.connectors = {
            'snowflake': SnowflakeConnector,
            'bigquery': BigQueryConnector,
            'postgresql': PostgreSQLConnector,
            'mongodb': MongoDBConnector
        }
    
    def load_from_connector(self, connector_type: str, config: Dict[str, Any], 
                          query: str = None, table: str = None) -> pd.DataFrame:
        """Load data from external connector."""
        if connector_type not in self.connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")
        
        connector_class = self.connectors[connector_type]
        connector = connector_class(**config)
        
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
        try:
            import skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            model, metadata = self.load_model(model_id, version, tenant_id)
            
            if output_path is None:
                output_path = f"./{model_id}_v{version or 'latest'}.onnx"
            
            # Define input type based on feature count
            n_features = len(metadata.get('feature_names', []))
            if n_features == 0:
                raise ValueError("Feature names not found in metadata")
            
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            
            # Convert to ONNX
            onx = convert_sklearn(model, initial_types=initial_type, target_opset=12)
            
            # Save ONNX model
            with open(output_path, "wb") as f:
                f.write(onx.SerializeToString())
            
            logger.info(f"ONNX export completed at {output_path}")
            return output_path
            
        except ImportError:
            logger.error("skl2onnx not installed. Install with: pip install skl2onnx")
            raise
        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
            raise
    
    def export_model_to_pmml(self, model_id: str, version: str = None,
                            tenant_id: str = "default", output_path: str = None) -> str:
        """Export model to PMML format."""
        try:
            from sklearn2pmml import sklearn2pmml
            from sklearn2pmml.pipeline import PMMLPipeline
            
            model, metadata = self.load_model(model_id, version, tenant_id)
            
            if output_path is None:
                output_path = f"./{model_id}_v{version or 'latest'}.pmml"
            
            # Wrap model in PMML pipeline if needed
            if not isinstance(model, PMMLPipeline):
                pmml_pipeline = PMMLPipeline([("model", model)])
            else:
                pmml_pipeline = model
            
            # Export to PMML
            sklearn2pmml(pmml_pipeline, output_path, with_repr=True)
            
            logger.info(f"PMML export completed at {output_path}")
            return output_path
            
        except ImportError:
            logger.error("sklearn2pmml not installed. Install with: pip install sklearn2pmml")
            raise
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
    """Feature store for feature reusability and versioning."""
    
    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
        self.feature_cache = {}
        self.feature_registry = {}
    
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
        cache_key = f"{feature_set_name}:{version}"
        self.feature_cache[cache_key] = features
        
        logger.info(f"Features saved: {feature_set_name} v{version}")
        return path
    
    def load_features(self, 
                     feature_set_name: str,
                     version: str = None) -> pd.DataFrame:
        """Load features from store."""
        
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
        """List available feature sets."""
        return list(set(k.split(':')[0] for k in self.feature_cache.keys()))
