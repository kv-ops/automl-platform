"""
Feature Store for AutoML Platform
Persistent storage with versioning, caching, and time-travel capabilities
WITH PROMETHEUS METRICS INTEGRATION
Place in: automl_platform/api/feature_store.py
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json
import hashlib
import pickle
import logging
from pathlib import Path
import threading
from collections import defaultdict
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq
import time

from automl_platform.config import InsecureEnvironmentVariableError, validate_secret_value

# Métriques Prometheus
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Storage backends
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

logger = logging.getLogger(__name__)

# Créer un registre local pour les métriques du feature store
feature_store_registry = CollectorRegistry()

# Déclaration des métriques Prometheus avec le registre local
ml_feature_store_operations_total = Counter(
    'ml_feature_store_operations_total',
    'Total number of feature store operations',
    ['tenant_id', 'operation', 'feature_set'],  # operation: read, write, compute
    registry=feature_store_registry
)

ml_feature_store_latency_seconds = Histogram(
    'ml_feature_store_latency_seconds',
    'Feature store operation latency in seconds',
    ['tenant_id', 'operation', 'backend'],  # backend: local, redis, s3, minio
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
    registry=feature_store_registry
)

ml_feature_store_size_bytes = Gauge(
    'ml_feature_store_size_bytes',
    'Total size of feature store in bytes',
    ['tenant_id', 'feature_set'],
    registry=feature_store_registry
)

ml_feature_store_cache_hits = Counter(
    'ml_feature_store_cache_hits',
    'Number of cache hits',
    ['tenant_id', 'feature_set'],
    registry=feature_store_registry
)

ml_feature_store_cache_misses = Counter(
    'ml_feature_store_cache_misses',
    'Number of cache misses',
    ['tenant_id', 'feature_set'],
    registry=feature_store_registry
)

ml_feature_store_errors_total = Counter(
    'ml_feature_store_errors_total',
    'Total number of feature store errors',
    ['tenant_id', 'operation', 'error_type'],
    registry=feature_store_registry
)

ml_feature_store_features_count = Gauge(
    'ml_feature_store_features_count',
    'Number of features per feature set',
    ['tenant_id', 'feature_set'],
    registry=feature_store_registry
)


@dataclass
class FeatureDefinition:
    """Definition of a feature in the store."""
    name: str
    dtype: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Computation
    source_table: str = ""
    computation: str = ""  # SQL or Python expression
    dependencies: List[str] = field(default_factory=list)
    
    # Versioning
    version: int = 1
    is_deprecated: bool = False
    
    # Statistics
    null_ratio: float = 0.0
    unique_count: int = 0
    min_value: Any = None
    max_value: Any = None
    mean_value: float = None
    std_value: float = None


@dataclass
class FeatureSet:
    """Collection of features."""
    name: str
    description: str = ""
    features: List[FeatureDefinition] = field(default_factory=list)
    entity_key: str = "entity_id"  # Primary key for joining
    timestamp_key: str = "timestamp"
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    
    # Storage
    storage_format: str = "parquet"  # parquet, csv, delta
    partitioning: List[str] = field(default_factory=list)  # Partition columns
    
    # Tenant info for metrics
    tenant_id: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "features": [f.name for f in self.features],
            "entity_key": self.entity_key,
            "timestamp_key": self.timestamp_key,
            "tags": self.tags,
            "owner": self.owner,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "tenant_id": self.tenant_id
        }


class FeatureStore:
    """
    Centralized feature store with versioning and caching.
    Supports offline and online serving.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature store.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backend = config.get("backend", "local")  # local, s3, minio, redis
        self.cache_enabled = config.get("cache_enabled", True)
        self.cache_ttl = config.get("cache_ttl", 3600)
        self.default_tenant_id = config.get("default_tenant_id", "default")
        
        # Storage paths
        self.base_path = Path(config.get("base_path", "./feature_store"))
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Metadata store (SQLite)
        self.metadata_db = self.base_path / "metadata.db"
        self._init_metadata_store()
        
        # Feature registry
        self.feature_sets = {}
        self.features = {}
        
        # Cache
        self.cache = {}
        self.cache_timestamps = {}
        
        # Redis for online serving
        self.redis_client = None
        if REDIS_AVAILABLE and config.get("redis_enabled", False):
            self._init_redis(config.get("redis_config", {}))
        
        # Object storage
        self.object_client = None
        if self.backend in ["s3", "minio"]:
            self._init_object_storage(config)
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"Feature store initialized with backend: {self.backend}")
    
    def _init_metadata_store(self):
        """Initialize SQLite metadata store."""
        conn = sqlite3.connect(str(self.metadata_db))
        cursor = conn.cursor()
        
        # Feature sets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_sets (
                name TEXT PRIMARY KEY,
                description TEXT,
                entity_key TEXT,
                timestamp_key TEXT,
                owner TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                version INTEGER,
                metadata JSON
            )
        """)
        
        # Features table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features (
                name TEXT PRIMARY KEY,
                feature_set TEXT,
                dtype TEXT,
                description TEXT,
                computation TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                version INTEGER,
                statistics JSON,
                FOREIGN KEY (feature_set) REFERENCES feature_sets(name)
            )
        """)
        
        # Feature values table (for versioning)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_values (
                feature_name TEXT,
                entity_id TEXT,
                timestamp TIMESTAMP,
                value BLOB,
                version INTEGER,
                PRIMARY KEY (feature_name, entity_id, timestamp, version)
            )
        """)
        
        # Materialization jobs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS materialization_jobs (
                job_id TEXT PRIMARY KEY,
                feature_set TEXT,
                status TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error TEXT,
                rows_processed INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _init_redis(self, redis_config: Dict[str, Any]):
        """Initialize Redis for online serving."""
        try:
            self.redis_client = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                db=redis_config.get("db", 0),
                decode_responses=False  # Store binary data
            )
            self.redis_client.ping()
            logger.info("Connected to Redis for online feature serving")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _init_object_storage(self, config: Dict[str, Any]):
        """Initialize object storage (S3/MinIO)."""
        if self.backend == "minio" and MINIO_AVAILABLE:
            access_key = config.get("access_key") or os.getenv("MINIO_ACCESS_KEY")
            secret_key = config.get("secret_key") or os.getenv("MINIO_SECRET_KEY")
            if not access_key or not secret_key:
                raise RuntimeError("MinIO credentials must be configured for the feature store.")
            try:
                validate_secret_value("MINIO_ACCESS_KEY", access_key)
                validate_secret_value("MINIO_SECRET_KEY", secret_key)
            except InsecureEnvironmentVariableError as exc:
                raise RuntimeError(
                    "Feature store MinIO credentials use an insecure default. "
                    "Rotate the access and secret keys."
                ) from exc
            self.object_client = Minio(
                config.get("endpoint", "localhost:9000"),
                access_key=access_key,
                secret_key=secret_key,
                secure=config.get("secure", False)
            )
            self.bucket_name = config.get("bucket", "feature-store")

            # Create bucket if doesn't exist
            if not self.object_client.bucket_exists(self.bucket_name):
                self.object_client.make_bucket(self.bucket_name)
                
        elif self.backend == "s3" and S3_AVAILABLE:
            access_key = config.get("access_key")
            secret_key = config.get("secret_key")
            if not access_key or not secret_key:
                raise RuntimeError("S3 credentials must be configured for the feature store.")
            validate_secret_value("MINIO_ACCESS_KEY", access_key)
            validate_secret_value("MINIO_SECRET_KEY", secret_key)
            self.object_client = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=config.get("region", "us-east-1")
            )
            self.bucket_name = config.get("bucket", "feature-store")
    
    def register_feature_set(self, feature_set: FeatureSet) -> bool:
        """
        Register a new feature set with metrics.
        
        Args:
            feature_set: FeatureSet to register
            
        Returns:
            Success status
        """
        start_time = time.time()
        
        with self.lock:
            try:
                # Increment operation counter
                ml_feature_store_operations_total.labels(
                    tenant_id=feature_set.tenant_id,
                    operation='register',
                    feature_set=feature_set.name
                ).inc()
                
                # Store in registry
                self.feature_sets[feature_set.name] = feature_set
                
                # Store in metadata DB
                conn = sqlite3.connect(str(self.metadata_db))
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO feature_sets 
                    (name, description, entity_key, timestamp_key, owner, 
                     created_at, updated_at, version, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feature_set.name,
                    feature_set.description,
                    feature_set.entity_key,
                    feature_set.timestamp_key,
                    feature_set.owner,
                    feature_set.created_at,
                    feature_set.updated_at,
                    feature_set.version,
                    json.dumps(feature_set.to_dict())
                ))
                
                # Register individual features
                for feature in feature_set.features:
                    cursor.execute("""
                        INSERT OR REPLACE INTO features
                        (name, feature_set, dtype, description, computation,
                         created_at, updated_at, version, statistics)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        feature.name,
                        feature_set.name,
                        feature.dtype,
                        feature.description,
                        feature.computation,
                        feature.created_at,
                        feature.updated_at,
                        feature.version,
                        json.dumps({
                            "null_ratio": feature.null_ratio,
                            "unique_count": feature.unique_count,
                            "min_value": feature.min_value,
                            "max_value": feature.max_value,
                            "mean_value": feature.mean_value,
                            "std_value": feature.std_value
                        })
                    ))
                    
                    self.features[feature.name] = feature
                
                conn.commit()
                conn.close()
                
                # Update feature count gauge
                ml_feature_store_features_count.labels(
                    tenant_id=feature_set.tenant_id,
                    feature_set=feature_set.name
                ).set(len(feature_set.features))
                
                logger.info(f"Registered feature set: {feature_set.name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register feature set: {e}")
                ml_feature_store_errors_total.labels(
                    tenant_id=feature_set.tenant_id,
                    operation='register',
                    error_type=type(e).__name__
                ).inc()
                return False
                
            finally:
                # Record latency
                ml_feature_store_latency_seconds.labels(
                    tenant_id=feature_set.tenant_id,
                    operation='register',
                    backend=self.backend
                ).observe(time.time() - start_time)
    
    def write_features(self, 
                      feature_set_name: str,
                      data: pd.DataFrame,
                      timestamp: Optional[datetime] = None,
                      tenant_id: Optional[str] = None) -> bool:
        """
        Write features to store with metrics.
        
        Args:
            feature_set_name: Name of feature set
            data: DataFrame with features
            timestamp: Timestamp for versioning
            tenant_id: Tenant identifier
            
        Returns:
            Success status
        """
        start_time = time.time()
        tenant_id = tenant_id or self.default_tenant_id
        
        if feature_set_name not in self.feature_sets:
            logger.error(f"Feature set {feature_set_name} not found")
            return False
        
        feature_set = self.feature_sets[feature_set_name]
        timestamp = timestamp or datetime.now()
        
        try:
            # Increment operation counter
            ml_feature_store_operations_total.labels(
                tenant_id=tenant_id,
                operation='write',
                feature_set=feature_set_name
            ).inc()
            
            # Add timestamp if not present
            if feature_set.timestamp_key not in data.columns:
                data[feature_set.timestamp_key] = timestamp
            
            # Calculate data size
            data_size = data.memory_usage(deep=True).sum()
            
            # Store based on backend
            if self.backend == "local":
                # Store as parquet file
                file_path = self.base_path / feature_set_name / f"{timestamp.isoformat()}.parquet"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                table = pa.Table.from_pandas(data)
                pq.write_table(table, file_path)
                
            elif self.backend in ["s3", "minio"] and self.object_client:
                # Store in object storage
                buffer = data.to_parquet()
                object_name = f"{feature_set_name}/{timestamp.isoformat()}.parquet"
                
                if self.backend == "minio":
                    from io import BytesIO
                    self.object_client.put_object(
                        self.bucket_name,
                        object_name,
                        BytesIO(buffer),
                        len(buffer)
                    )
                else:  # S3
                    self.object_client.put_object(
                        Bucket=self.bucket_name,
                        Key=object_name,
                        Body=buffer
                    )
            
            # Update online store if available
            if self.redis_client:
                self._update_online_store(feature_set_name, data, tenant_id)
            
            # Update size gauge
            current_size = ml_feature_store_size_bytes._metrics.get(
                tuple(sorted([('tenant_id', tenant_id), ('feature_set', feature_set_name)]))
            )
            if current_size:
                current_value = current_size._value.get()
            else:
                current_value = 0
            
            ml_feature_store_size_bytes.labels(
                tenant_id=tenant_id,
                feature_set=feature_set_name
            ).set(current_value + data_size)
            
            # Invalidate cache
            self._invalidate_cache(feature_set_name)
            
            logger.info(f"Wrote {len(data)} rows to feature set {feature_set_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to write features: {e}")
            ml_feature_store_errors_total.labels(
                tenant_id=tenant_id,
                operation='write',
                error_type=type(e).__name__
            ).inc()
            return False
            
        finally:
            # Record latency
            ml_feature_store_latency_seconds.labels(
                tenant_id=tenant_id,
                operation='write',
                backend=self.backend
            ).observe(time.time() - start_time)
    
    def read_features(self,
                     feature_names: List[str],
                     entity_ids: List[str],
                     timestamp: Optional[datetime] = None,
                     tenant_id: Optional[str] = None) -> pd.DataFrame:
        """
        Read features from store with metrics.
        
        Args:
            feature_names: List of feature names
            entity_ids: List of entity IDs
            timestamp: Point-in-time for time travel
            tenant_id: Tenant identifier
            
        Returns:
            DataFrame with requested features
        """
        start_time = time.time()
        tenant_id = tenant_id or self.default_tenant_id
        
        # Determine feature sets
        feature_sets_involved = set()
        for fname in feature_names:
            for fs_name, fs in self.feature_sets.items():
                if any(f.name == fname for f in fs.features):
                    feature_sets_involved.add(fs_name)
        
        # Use first feature set for metrics (simplified)
        feature_set_name = list(feature_sets_involved)[0] if feature_sets_involved else "unknown"
        
        try:
            # Increment operation counter
            ml_feature_store_operations_total.labels(
                tenant_id=tenant_id,
                operation='read',
                feature_set=feature_set_name
            ).inc()
            
            # Check cache first
            cache_key = self._get_cache_key(feature_names, entity_ids, timestamp)
            if self.cache_enabled:
                cached_data = self._get_from_cache(cache_key, feature_set_name, tenant_id)
                if cached_data is not None:
                    return cached_data
            
            # Try online store first for latest features
            if self.redis_client and not timestamp:
                data = self._read_from_online_store(feature_names, entity_ids, tenant_id)
                if data is not None:
                    self._save_to_cache(cache_key, data, feature_set_name, tenant_id)
                    return data
            
            # Read from offline store
            data = self._read_from_offline_store(feature_names, entity_ids, timestamp)
            
            if data is not None:
                self._save_to_cache(cache_key, data, feature_set_name, tenant_id)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to read features: {e}")
            ml_feature_store_errors_total.labels(
                tenant_id=tenant_id,
                operation='read',
                error_type=type(e).__name__
            ).inc()
            return pd.DataFrame()
            
        finally:
            # Record latency
            ml_feature_store_latency_seconds.labels(
                tenant_id=tenant_id,
                operation='read',
                backend=self.backend
            ).observe(time.time() - start_time)
    
    def compute_statistics(self, feature_set_name: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Compute statistics for a feature set with metrics."""
        start_time = time.time()
        tenant_id = tenant_id or self.default_tenant_id
        
        try:
            # Increment operation counter
            ml_feature_store_operations_total.labels(
                tenant_id=tenant_id,
                operation='compute',
                feature_set=feature_set_name
            ).inc()
            
            if feature_set_name not in self.feature_sets:
                return {}
            
            # Read latest data
            if self.backend == "local":
                files = list((self.base_path / feature_set_name).glob("*.parquet"))
                if not files:
                    return {}
                
                df = pd.read_parquet(max(files))
            else:
                return {}
            
            stats = {}
            
            for col in df.columns:
                col_stats = {
                    "count": len(df[col]),
                    "null_count": df[col].isna().sum(),
                    "null_ratio": df[col].isna().mean(),
                    "unique_count": df[col].nunique(),
                    "dtype": str(df[col].dtype)
                }
                
                # Numeric statistics
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_stats.update({
                        "mean": df[col].mean(),
                        "std": df[col].std(),
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "median": df[col].median(),
                        "q25": df[col].quantile(0.25),
                        "q75": df[col].quantile(0.75)
                    })
                
                stats[col] = col_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute statistics: {e}")
            ml_feature_store_errors_total.labels(
                tenant_id=tenant_id,
                operation='compute',
                error_type=type(e).__name__
            ).inc()
            return {}
            
        finally:
            # Record latency
            ml_feature_store_latency_seconds.labels(
                tenant_id=tenant_id,
                operation='compute',
                backend=self.backend
            ).observe(time.time() - start_time)
    
    def _update_online_store(self, feature_set_name: str, data: pd.DataFrame, tenant_id: str):
        """Update online store with latest features."""
        if not self.redis_client:
            return
        
        feature_set = self.feature_sets[feature_set_name]
        entity_key = feature_set.entity_key
        
        try:
            pipe = self.redis_client.pipeline()
            
            for _, row in data.iterrows():
                entity_id = row[entity_key]
                
                # Store each feature
                for col in data.columns:
                    if col not in [entity_key, feature_set.timestamp_key]:
                        key = f"{tenant_id}:{feature_set_name}:{entity_id}:{col}"
                        value = pickle.dumps(row[col])
                        pipe.set(key, value, ex=self.cache_ttl)
                
                # Store metadata
                meta_key = f"{tenant_id}:{feature_set_name}:{entity_id}:_meta"
                meta_value = {
                    "timestamp": row.get(feature_set.timestamp_key, datetime.now()).isoformat(),
                    "features": list(data.columns)
                }
                pipe.set(meta_key, pickle.dumps(meta_value), ex=self.cache_ttl)
            
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Failed to update online store: {e}")
    
    def _read_from_online_store(self, feature_names: List[str], entity_ids: List[str], tenant_id: str) -> Optional[pd.DataFrame]:
        """Read features from online store."""
        if not self.redis_client:
            return None
        
        try:
            data = []
            
            for entity_id in entity_ids:
                row = {"entity_id": entity_id}
                
                for feature_name in feature_names:
                    # Find feature set for this feature
                    feature_set_name = None
                    for fs_name, fs in self.feature_sets.items():
                        if any(f.name == feature_name for f in fs.features):
                            feature_set_name = fs_name
                            break
                    
                    if feature_set_name:
                        key = f"{tenant_id}:{feature_set_name}:{entity_id}:{feature_name}"
                        value = self.redis_client.get(key)
                        
                        if value:
                            row[feature_name] = pickle.loads(value)
                        else:
                            row[feature_name] = None
                
                data.append(row)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to read from online store: {e}")
            return None
    
    def _read_from_offline_store(self, 
                                feature_names: List[str],
                                entity_ids: List[str],
                                timestamp: Optional[datetime]) -> pd.DataFrame:
        """Read features from offline store."""
        # Determine which feature sets are needed
        required_feature_sets = {}
        for feature_name in feature_names:
            for fs_name, fs in self.feature_sets.items():
                if any(f.name == feature_name for f in fs.features):
                    if fs_name not in required_feature_sets:
                        required_feature_sets[fs_name] = []
                    required_feature_sets[fs_name].append(feature_name)
        
        # Read data from each feature set
        dfs = []
        
        for fs_name, features in required_feature_sets.items():
            feature_set = self.feature_sets[fs_name]
            
            # Find appropriate file based on timestamp
            if self.backend == "local":
                files = list((self.base_path / fs_name).glob("*.parquet"))
                
                if timestamp:
                    # Find file closest to but before timestamp
                    valid_files = []
                    for f in files:
                        file_time = datetime.fromisoformat(f.stem)
                        if file_time <= timestamp:
                            valid_files.append((f, file_time))
                    
                    if valid_files:
                        files = [max(valid_files, key=lambda x: x[1])[0]]
                    else:
                        files = []
                
                if files:
                    # Read most recent file
                    df = pd.read_parquet(files[-1])
                    
                    # Filter by entity IDs and features
                    if feature_set.entity_key in df.columns:
                        df = df[df[feature_set.entity_key].isin(entity_ids)]
                    
                    cols_to_keep = [feature_set.entity_key] + features
                    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
                    df = df[cols_to_keep]
                    
                    dfs.append(df)
        
        # Merge all dataframes
        if dfs:
            result = dfs[0]
            for df in dfs[1:]:
                # Determine common key
                common_cols = list(set(result.columns) & set(df.columns))
                if common_cols:
                    result = result.merge(df, on=common_cols[0], how="outer")
            
            return result
        
        return pd.DataFrame()
    
    def _get_cache_key(self, feature_names: List[str], entity_ids: List[str], timestamp: Optional[datetime]) -> str:
        """Generate cache key."""
        key_str = f"{sorted(feature_names)}_{sorted(entity_ids)}_{timestamp}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str, feature_set_name: str, tenant_id: str) -> Optional[pd.DataFrame]:
        """Get data from cache with metrics."""
        if cache_key in self.cache:
            cached_time = self.cache_timestamps.get(cache_key)
            if cached_time and (datetime.now() - cached_time).seconds < self.cache_ttl:
                # Increment cache hit counter
                ml_feature_store_cache_hits.labels(
                    tenant_id=tenant_id,
                    feature_set=feature_set_name
                ).inc()
                return self.cache[cache_key]
        
        # Increment cache miss counter
        ml_feature_store_cache_misses.labels(
            tenant_id=tenant_id,
            feature_set=feature_set_name
        ).inc()
        return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame, feature_set_name: str, tenant_id: str):
        """Save data to cache."""
        self.cache[cache_key] = data
        self.cache_timestamps[cache_key] = datetime.now()
    
    def _invalidate_cache(self, feature_set_name: str):
        """Invalidate cache for a feature set."""
        keys_to_remove = []
        for key in self.cache:
            # Simple check - in production use proper indexing
            if feature_set_name in str(key):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
    
    def list_feature_sets(self) -> List[Dict[str, Any]]:
        """List all feature sets."""
        result = []
        
        conn = sqlite3.connect(str(self.metadata_db))
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM feature_sets")
        rows = cursor.fetchall()
        
        for row in rows:
            result.append({
                "name": row[0],
                "description": row[1],
                "entity_key": row[2],
                "timestamp_key": row[3],
                "owner": row[4],
                "created_at": row[5],
                "updated_at": row[6],
                "version": row[7]
            })
        
        conn.close()
        return result
    
    def get_feature_history(self, feature_name: str, entity_id: str, 
                          start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get historical values of a feature."""
        conn = sqlite3.connect(str(self.metadata_db))
        
        query = """
            SELECT timestamp, value, version
            FROM feature_values
            WHERE feature_name = ? AND entity_id = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(
            query, 
            conn,
            params=(feature_name, entity_id, start_time, end_time)
        )
        
        conn.close()
        
        # Deserialize values
        if not df.empty:
            df['value'] = df['value'].apply(lambda x: pickle.loads(x))
        
        return df


# FastAPI router for feature store
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

feature_store_router = APIRouter(prefix="/feature-store", tags=["feature-store"])

# Global feature store instance (initialized in api.py)
_feature_store = None

def get_feature_store():
    """Get feature store instance."""
    global _feature_store
    if _feature_store is None:
        config = {
            "backend": "local",
            "base_path": "./feature_store",
            "cache_enabled": True,
            "cache_ttl": 3600,
            "redis_enabled": False,
            "default_tenant_id": "default"
        }
        _feature_store = FeatureStore(config)
    return _feature_store

class FeatureSetRequest(BaseModel):
    name: str
    description: str
    features: List[Dict[str, Any]]
    entity_key: str = "entity_id"
    timestamp_key: str = "timestamp"
    tenant_id: str = "default"

@feature_store_router.post("/register")
async def register_feature_set(request: FeatureSetRequest):
    """Register a new feature set."""
    store = get_feature_store()
    
    # Create feature definitions
    features = []
    for f in request.features:
        features.append(FeatureDefinition(
            name=f["name"],
            dtype=f.get("dtype", "float"),
            description=f.get("description", "")
        ))
    
    # Create feature set
    feature_set = FeatureSet(
        name=request.name,
        description=request.description,
        features=features,
        entity_key=request.entity_key,
        timestamp_key=request.timestamp_key,
        tenant_id=request.tenant_id
    )
    
    success = store.register_feature_set(feature_set)
    
    if success:
        return {"status": "success", "feature_set": request.name}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register feature set"
        )

@feature_store_router.get("/list")
async def list_feature_sets():
    """List all feature sets."""
    store = get_feature_store()
    return {"feature_sets": store.list_feature_sets()}


# Export the registry and router so they can be imported by api.py
__all__ = ['feature_store_registry', 'FeatureStore', 'FeatureSet', 'FeatureDefinition', 'feature_store_router']
