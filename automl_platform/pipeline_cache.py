"""
Pipeline Cache with Intelligent Invalidation
============================================
Place in: automl_platform/pipeline_cache.py

Implements ML pipeline caching with:
- Redis/Memcached backends
- Intelligent cache invalidation
- Pipeline fingerprinting
- Memory-mapped model storage
- Distributed cache support
"""

import hashlib
import json
import pickle
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import joblib
import os
import threading
from collections import OrderedDict, defaultdict
from contextlib import contextmanager

# Cache backends
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import pymemcache
    from pymemcache.client import base
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False

# Memory mapping
try:
    import mmap
    MMAP_AVAILABLE = True
except ImportError:
    MMAP_AVAILABLE = False

# Compression
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Cache Configuration
# ============================================================================

@dataclass
class CacheConfig:
    """Cache configuration"""
    backend: str = "redis"  # redis, memcached, disk, memory
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    memcached_host: str = "localhost"
    memcached_port: int = 11211
    
    disk_cache_dir: str = "/tmp/pipeline_cache"
    max_cache_size_gb: float = 10.0
    
    ttl_seconds: int = 3600 * 24  # 24 hours default
    compression: bool = True
    compression_level: int = 3
    
    # Intelligent invalidation
    invalidate_on_drift: bool = True
    drift_threshold: float = 0.1
    invalidate_on_performance_drop: bool = True
    performance_threshold: float = 0.05
    
    # Memory mapping
    use_mmap: bool = True
    mmap_mode: str = "r"  # r, w+, c
    
    # Distributed cache
    distributed: bool = False
    cache_nodes: List[str] = None


# ============================================================================
# Pipeline Fingerprinting
# ============================================================================

class PipelineFingerprint:
    """Generate unique fingerprints for pipelines"""
    
    @staticmethod
    def generate(pipeline: Union[Pipeline, BaseEstimator], 
                 data_shape: Optional[Tuple] = None,
                 feature_names: Optional[List[str]] = None) -> str:
        """Generate unique fingerprint for pipeline"""
        
        components = []
        
        # Pipeline structure
        if isinstance(pipeline, Pipeline):
            for name, estimator in pipeline.steps:
                components.append(f"{name}:{type(estimator).__name__}")
                
                # Add key parameters
                params = estimator.get_params(deep=False)
                param_str = json.dumps(
                    {k: str(v) for k, v in sorted(params.items())},
                    sort_keys=True
                )
                components.append(param_str)
        else:
            # Single estimator
            components.append(type(pipeline).__name__)
            params = pipeline.get_params(deep=False)
            param_str = json.dumps(
                {k: str(v) for k, v in sorted(params.items())},
                sort_keys=True
            )
            components.append(param_str)
        
        # Data characteristics
        if data_shape:
            components.append(f"shape:{data_shape}")
        
        if feature_names:
            components.append(f"features:{','.join(sorted(feature_names))}")
        
        # Generate hash
        fingerprint_str = "|".join(components)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()
    
    @staticmethod
    def generate_data_fingerprint(X: Union[pd.DataFrame, np.ndarray],
                                  y: Optional[Union[pd.Series, np.ndarray]] = None) -> str:
        """Generate fingerprint for data"""
        
        components = []
        
        # Data shape and type
        components.append(f"shape:{X.shape}")
        components.append(f"dtype:{X.dtype if hasattr(X, 'dtype') else 'mixed'}")
        
        # Statistical summary
        if isinstance(X, pd.DataFrame):
            # Column names and types
            components.append(f"cols:{','.join(X.columns)}")
            components.append(f"types:{','.join(X.dtypes.astype(str))}")
            
            # Basic statistics
            stats = X.describe().to_json()
            components.append(f"stats:{stats}")
        else:
            # Array statistics
            components.append(f"mean:{np.mean(X):.6f}")
            components.append(f"std:{np.std(X):.6f}")
            components.append(f"min:{np.min(X):.6f}")
            components.append(f"max:{np.max(X):.6f}")
        
        # Target info
        if y is not None:
            components.append(f"y_shape:{y.shape}")
            if hasattr(y, 'unique'):
                components.append(f"y_unique:{len(np.unique(y))}")
        
        fingerprint_str = "|".join(components)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()


# ============================================================================
# Cache Backends
# ============================================================================

class CacheBackend:
    """Base cache backend interface"""
    
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        raise NotImplementedError
    
    def clear(self) -> bool:
        raise NotImplementedError
    
    def get_size(self) -> int:
        raise NotImplementedError


class RedisCache(CacheBackend):
    """Redis cache backend"""
    
    def __init__(self, config: CacheConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available")
        
        self.config = config
        self.client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            decode_responses=False
        )
        
    def get(self, key: str) -> Optional[Any]:
        try:
            data = self.client.get(key)
            if data:
                if self.config.compression and LZ4_AVAILABLE:
                    data = lz4.frame.decompress(data)
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            if self.config.compression and LZ4_AVAILABLE:
                data = lz4.frame.compress(data, compression_level=self.config.compression_level)
            
            ttl = ttl or self.config.ttl_seconds
            return self.client.setex(key, ttl, data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        return self.client.delete(key) > 0
    
    def exists(self, key: str) -> bool:
        return self.client.exists(key) > 0
    
    def clear(self) -> bool:
        self.client.flushdb()
        return True
    
    def get_size(self) -> int:
        info = self.client.info('memory')
        return info.get('used_memory', 0)


class MemcachedCache(CacheBackend):
    """Memcached cache backend"""
    
    def __init__(self, config: CacheConfig):
        if not MEMCACHED_AVAILABLE:
            raise ImportError("Memcached not available")
        
        self.config = config
        self.client = base.Client(
            (config.memcached_host, config.memcached_port)
        )
    
    def get(self, key: str) -> Optional[Any]:
        try:
            data = self.client.get(key)
            if data:
                if self.config.compression and LZ4_AVAILABLE:
                    data = lz4.frame.decompress(data)
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Memcached get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            if self.config.compression and LZ4_AVAILABLE:
                data = lz4.frame.compress(data, compression_level=self.config.compression_level)
            
            ttl = ttl or self.config.ttl_seconds
            return self.client.set(key, data, expire=ttl)
        except Exception as e:
            logger.error(f"Memcached set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        return self.client.delete(key)
    
    def exists(self, key: str) -> bool:
        return self.get(key) is not None
    
    def clear(self) -> bool:
        self.client.flush_all()
        return True
    
    def get_size(self) -> int:
        stats = self.client.stats()
        return stats.get(b'bytes', 0)


class DiskCache(CacheBackend):
    """Disk-based cache with memory mapping"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.disk_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.json"
        self.index = self._load_index()
        self.lock = threading.Lock()
    
    def _load_index(self) -> Dict:
        """Load cache index"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save cache index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f)
    
    def _get_path(self, key: str) -> Path:
        """Get file path for key"""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.index:
                return None
            
            # Check TTL
            entry = self.index[key]
            if time.time() > entry['expires']:
                self.delete(key)
                return None
            
            path = self._get_path(key)
            if not path.exists():
                return None
            
            try:
                if self.config.use_mmap and MMAP_AVAILABLE:
                    # Memory-mapped read
                    return joblib.load(path, mmap_mode=self.config.mmap_mode)
                else:
                    with open(path, 'rb') as f:
                        data = f.read()
                        if self.config.compression and LZ4_AVAILABLE:
                            data = lz4.frame.decompress(data)
                        return pickle.loads(data)
            except Exception as e:
                logger.error(f"Disk cache get error: {e}")
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self.lock:
            try:
                path = self._get_path(key)
                
                if self.config.use_mmap and MMAP_AVAILABLE:
                    # Memory-mapped write
                    joblib.dump(value, path, compress=self.config.compression_level if self.config.compression else 0)
                else:
                    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                    if self.config.compression and LZ4_AVAILABLE:
                        data = lz4.frame.compress(data, compression_level=self.config.compression_level)
                    
                    with open(path, 'wb') as f:
                        f.write(data)
                
                # Update index
                ttl = ttl or self.config.ttl_seconds
                self.index[key] = {
                    'path': str(path),
                    'size': path.stat().st_size,
                    'created': time.time(),
                    'expires': time.time() + ttl
                }
                self._save_index()
                
                # Check cache size
                self._enforce_size_limit()
                
                return True
                
            except Exception as e:
                logger.error(f"Disk cache set error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.index:
                path = self._get_path(key)
                if path.exists():
                    path.unlink()
                del self.index[key]
                self._save_index()
                return True
            return False
    
    def exists(self, key: str) -> bool:
        with self.lock:
            if key in self.index:
                if time.time() > self.index[key]['expires']:
                    self.delete(key)
                    return False
                return self._get_path(key).exists()
            return False
    
    def clear(self) -> bool:
        with self.lock:
            for key in list(self.index.keys()):
                self.delete(key)
            return True
    
    def get_size(self) -> int:
        return sum(entry['size'] for entry in self.index.values())
    
    def _enforce_size_limit(self):
        """Enforce cache size limit using LRU eviction"""
        max_size = self.config.max_cache_size_gb * 1024**3
        
        while self.get_size() > max_size:
            # Find oldest entry
            oldest_key = min(self.index.keys(), 
                           key=lambda k: self.index[k]['created'])
            self.delete(oldest_key)
            logger.info(f"Evicted {oldest_key} to maintain cache size")


class MemoryCache(CacheBackend):
    """In-memory cache with LRU eviction"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = OrderedDict()
        self.ttl = {}
        self.lock = threading.Lock()
        self.max_size = int(config.max_cache_size_gb * 1024**3)
        self.current_size = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if key in self.ttl and time.time() > self.ttl[key]:
                self.delete(key)
                return None
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self.lock:
            try:
                # Calculate size
                size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
                
                # Evict if necessary
                while self.current_size + size > self.max_size and self.cache:
                    self._evict_lru()
                
                self.cache[key] = value
                self.current_size += size
                
                if ttl or self.config.ttl_seconds:
                    self.ttl[key] = time.time() + (ttl or self.config.ttl_seconds)
                
                return True
                
            except Exception as e:
                logger.error(f"Memory cache set error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                value = self.cache.pop(key)
                size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
                self.current_size -= size
                self.ttl.pop(key, None)
                return True
            return False
    
    def exists(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                if key in self.ttl and time.time() > self.ttl[key]:
                    self.delete(key)
                    return False
                return True
            return False
    
    def clear(self) -> bool:
        with self.lock:
            self.cache.clear()
            self.ttl.clear()
            self.current_size = 0
            return True
    
    def get_size(self) -> int:
        return self.current_size
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self.cache:
            key, value = self.cache.popitem(last=False)
            size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            self.current_size -= size
            self.ttl.pop(key, None)
            logger.debug(f"Evicted {key} (LRU)")


# ============================================================================
# Pipeline Cache Manager
# ============================================================================

class PipelineCache:
    """Main pipeline cache manager with intelligent invalidation"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Initialize backend
        if self.config.backend == "redis":
            self.backend = RedisCache(self.config)
        elif self.config.backend == "memcached":
            self.backend = MemcachedCache(self.config)
        elif self.config.backend == "disk":
            self.backend = DiskCache(self.config)
        else:
            self.backend = MemoryCache(self.config)
        
        # Cache statistics
        self.stats = defaultdict(int)
        self.performance_history = defaultdict(list)
        
        # Invalidation tracking
        self.data_fingerprints = {}
        self.model_versions = {}
        
        logger.info(f"Pipeline cache initialized with {self.config.backend} backend")
    
    def get_pipeline(self, 
                     pipeline_id: str,
                     X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                     check_validity: bool = True) -> Optional[Pipeline]:
        """Get cached pipeline with optional validity check"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(pipeline_id, X)
        
        # Check cache
        cached = self.backend.get(cache_key)
        if cached is None:
            self.stats['misses'] += 1
            return None
        
        self.stats['hits'] += 1
        
        # Validate if requested
        if check_validity and X is not None:
            if not self._validate_cache(pipeline_id, cached, X):
                logger.info(f"Cache invalidated for {pipeline_id}")
                self.backend.delete(cache_key)
                self.stats['invalidations'] += 1
                return None
        
        logger.debug(f"Cache hit for {pipeline_id}")
        return cached['pipeline']
    
    def set_pipeline(self,
                     pipeline_id: str,
                     pipeline: Pipeline,
                     X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                     metrics: Optional[Dict[str, float]] = None,
                     ttl: Optional[int] = None) -> bool:
        """Cache pipeline with metadata"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(pipeline_id, X)
        
        # Prepare cache entry
        entry = {
            'pipeline': pipeline,
            'pipeline_id': pipeline_id,
            'fingerprint': PipelineFingerprint.generate(pipeline),
            'cached_at': time.time(),
            'metrics': metrics or {},
            'version': self._get_next_version(pipeline_id)
        }
        
        # Store data fingerprint for validation
        if X is not None:
            data_fingerprint = PipelineFingerprint.generate_data_fingerprint(X)
            entry['data_fingerprint'] = data_fingerprint
            self.data_fingerprints[pipeline_id] = data_fingerprint
        
        # Store performance metrics
        if metrics:
            self.performance_history[pipeline_id].append({
                'metrics': metrics,
                'timestamp': time.time()
            })
        
        # Cache the pipeline
        success = self.backend.set(cache_key, entry, ttl)
        
        if success:
            self.stats['sets'] += 1
            logger.debug(f"Cached pipeline {pipeline_id}")
        
        return success
    
    def invalidate(self, pipeline_id: str, reason: str = "manual"):
        """Invalidate cached pipeline"""
        
        # Find all related cache keys
        pattern = f"{pipeline_id}*"
        
        # This is simplified - real implementation would scan keys
        deleted = 0
        if hasattr(self.backend, 'client') and hasattr(self.backend.client, 'scan_iter'):
            # Redis backend
            for key in self.backend.client.scan_iter(match=pattern):
                if self.backend.delete(key.decode() if isinstance(key, bytes) else key):
                    deleted += 1
        else:
            # Simple deletion
            if self.backend.delete(pipeline_id):
                deleted = 1
        
        self.stats['invalidations'] += deleted
        logger.info(f"Invalidated {deleted} cache entries for {pipeline_id} (reason: {reason})")
        
        # Clear tracking data
        self.data_fingerprints.pop(pipeline_id, None)
        self.model_versions.pop(pipeline_id, None)
        self.performance_history.pop(pipeline_id, None)
    
    def clear_all(self) -> bool:
        """Clear entire cache"""
        success = self.backend.clear()
        if success:
            self.stats.clear()
            self.data_fingerprints.clear()
            self.model_versions.clear()
            self.performance_history.clear()
            logger.info("Cache cleared")
        return success
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'invalidations': self.stats['invalidations'],
            'hit_rate': hit_rate,
            'size_bytes': self.backend.get_size(),
            'backend': self.config.backend
        }
    
    @contextmanager
    def cached_operation(self, operation_id: str):
        """Context manager for cached operations"""
        start_time = time.time()
        
        try:
            yield self
        finally:
            duration = time.time() - start_time
            self.stats['operation_times'].append(duration)
            logger.debug(f"Cached operation {operation_id} took {duration:.3f}s")
    
    def _generate_cache_key(self, 
                           pipeline_id: str,
                           X: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> str:
        """Generate cache key"""
        
        components = [pipeline_id]
        
        if X is not None:
            # Add data characteristics to key
            if isinstance(X, pd.DataFrame):
                components.append(f"cols_{len(X.columns)}")
                components.append(f"rows_{len(X)}")
            else:
                components.append(f"shape_{X.shape}")
        
        return "_".join(components)
    
    def _validate_cache(self,
                       pipeline_id: str,
                       cached_entry: Dict,
                       X: Union[pd.DataFrame, np.ndarray]) -> bool:
        """Validate cached pipeline"""
        
        # Check data drift if enabled
        if self.config.invalidate_on_drift:
            current_fingerprint = PipelineFingerprint.generate_data_fingerprint(X)
            cached_fingerprint = cached_entry.get('data_fingerprint')
            
            if cached_fingerprint and current_fingerprint != cached_fingerprint:
                # Simplified drift detection
                logger.debug(f"Data fingerprint changed for {pipeline_id}")
                
                # Could add more sophisticated drift detection here
                if self.config.drift_threshold < 1.0:
                    return False
        
        # Check performance degradation if enabled
        if self.config.invalidate_on_performance_drop:
            history = self.performance_history.get(pipeline_id, [])
            if len(history) >= 2:
                recent_metrics = history[-1]['metrics']
                cached_metrics = cached_entry.get('metrics', {})
                
                # Compare key metric (e.g., accuracy)
                for metric in ['accuracy', 'f1_score', 'rmse']:
                    if metric in recent_metrics and metric in cached_metrics:
                        recent = recent_metrics[metric]
                        cached = cached_metrics[metric]
                        
                        # Check degradation
                        if metric == 'rmse':
                            degradation = (recent - cached) / abs(cached) if cached != 0 else 0
                        else:
                            degradation = (cached - recent) / cached if cached != 0 else 0
                        
                        if degradation > self.config.performance_threshold:
                            logger.debug(f"Performance degradation detected for {pipeline_id}: {degradation:.2%}")
                            return False
        
        # Check age
        age = time.time() - cached_entry.get('cached_at', 0)
        if age > self.config.ttl_seconds:
            logger.debug(f"Cache expired for {pipeline_id}")
            return False
        
        return True
    
    def _get_next_version(self, pipeline_id: str) -> int:
        """Get next version number for pipeline"""
        current = self.model_versions.get(pipeline_id, 0)
        next_version = current + 1
        self.model_versions[pipeline_id] = next_version
        return next_version


# ============================================================================
# Distributed Cache
# ============================================================================

class DistributedPipelineCache:
    """Distributed pipeline cache across multiple nodes"""
    
    def __init__(self, config: CacheConfig):
        if not config.distributed:
            raise ValueError("Distributed mode not enabled in config")
        
        self.config = config
        self.nodes = config.cache_nodes or []
        self.local_cache = PipelineCache(config)
        self.node_caches = {}
        
        # Initialize connections to other nodes
        for node in self.nodes:
            host, port = node.split(':')
            node_config = CacheConfig(
                backend=config.backend,
                redis_host=host,
                redis_port=int(port)
            )
            self.node_caches[node] = PipelineCache(node_config)
    
    def get_pipeline(self, pipeline_id: str, **kwargs) -> Optional[Pipeline]:
        """Get pipeline from distributed cache"""
        
        # Try local first
        pipeline = self.local_cache.get_pipeline(pipeline_id, **kwargs)
        if pipeline:
            return pipeline
        
        # Try other nodes
        for node, cache in self.node_caches.items():
            try:
                pipeline = cache.get_pipeline(pipeline_id, **kwargs)
                if pipeline:
                    # Replicate to local
                    self.local_cache.set_pipeline(pipeline_id, pipeline)
                    logger.debug(f"Retrieved pipeline from node {node}")
                    return pipeline
            except Exception as e:
                logger.warning(f"Failed to get from node {node}: {e}")
        
        return None
    
    def set_pipeline(self, pipeline_id: str, pipeline: Pipeline, **kwargs) -> bool:
        """Set pipeline in distributed cache"""
        
        # Set locally
        success = self.local_cache.set_pipeline(pipeline_id, pipeline, **kwargs)
        
        # Replicate to other nodes asynchronously
        for node, cache in self.node_caches.items():
            try:
                # This could be done asynchronously
                cache.set_pipeline(pipeline_id, pipeline, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to replicate to node {node}: {e}")
        
        return success
    
    def invalidate(self, pipeline_id: str, reason: str = "manual"):
        """Invalidate across all nodes"""
        
        # Invalidate locally
        self.local_cache.invalidate(pipeline_id, reason)
        
        # Invalidate on all nodes
        for node, cache in self.node_caches.items():
            try:
                cache.invalidate(pipeline_id, reason)
            except Exception as e:
                logger.warning(f"Failed to invalidate on node {node}: {e}")


# ============================================================================
# Cache Utilities
# ============================================================================

def warm_cache(pipelines: List[Tuple[str, Pipeline]], 
               cache: PipelineCache,
               sample_data: Optional[pd.DataFrame] = None):
    """Warm up cache with pre-trained pipelines"""
    
    logger.info(f"Warming cache with {len(pipelines)} pipelines")
    
    for pipeline_id, pipeline in pipelines:
        success = cache.set_pipeline(pipeline_id, pipeline, sample_data)
        if success:
            logger.debug(f"Warmed cache with {pipeline_id}")
    
    stats = cache.get_stats()
    logger.info(f"Cache warmed: {stats['sets']} pipelines cached")


def monitor_cache_health(cache: PipelineCache) -> Dict:
    """Monitor cache health metrics"""
    
    stats = cache.get_stats()
    
    health = {
        'status': 'healthy',
        'hit_rate': stats['hit_rate'],
        'size_mb': stats['size_bytes'] / 1024**2,
        'issues': []
    }
    
    # Check hit rate
    if stats['hit_rate'] < 0.5 and stats['hits'] + stats['misses'] > 100:
        health['issues'].append("Low hit rate")
        health['status'] = 'degraded'
    
    # Check size
    if cache.config.backend == 'memory':
        if stats['size_bytes'] > cache.config.max_cache_size_gb * 1024**3 * 0.9:
            health['issues'].append("Cache near capacity")
            health['status'] = 'warning'
    
    # Check invalidation rate
    total = stats['sets']
    if total > 0 and stats['invalidations'] / total > 0.3:
        health['issues'].append("High invalidation rate")
        health['status'] = 'degraded'
    
    return health
