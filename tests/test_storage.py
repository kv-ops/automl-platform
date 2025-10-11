"""
Test Storage Module
===================
Tests for storage backends, model versioning, and export functionality.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, ANY
from datetime import datetime
import pickle
import os
from typing import Dict

# Import storage components
from automl_platform.storage import (
    StorageManager,
    ModelMetadata,
    LocalStorage,
    MinIOStorage,
    FeatureStore,
    StorageDisabledError
)


class TestModelMetadata:
    """Test ModelMetadata dataclass."""
    
    def test_model_metadata_creation(self):
        """Test creating model metadata."""
        metadata = ModelMetadata(
            model_id="test_model",
            version="1.0.0",
            model_type="classification",
            algorithm="RandomForest",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            metrics={"accuracy": 0.95},
            parameters={"n_estimators": 100},
            feature_names=["feat1", "feat2"],
            target_name="target",
            dataset_hash="abc123",
            pipeline_hash="def456"
        )
        
        assert metadata.model_id == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.metrics["accuracy"] == 0.95
    
    def test_model_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ModelMetadata(
            model_id="test_model",
            version="1.0.0",
            model_type="classification",
            algorithm="RandomForest",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            metrics={},
            parameters={},
            feature_names=[],
            target_name="target",
            dataset_hash="",
            pipeline_hash=""
        )
        
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["model_id"] == "test_model"


class TestLocalStorage:
    """Test LocalStorage backend."""
    
    @pytest.fixture
    def local_storage(self):
        """Create LocalStorage instance with temp directory."""
        temp_dir = tempfile.mkdtemp()
        storage = LocalStorage(base_path=temp_dir)
        yield storage
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model."""
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=10, random_state=42)
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return ModelMetadata(
            model_id="test_model",
            version="1.0.0",
            model_type="classification",
            algorithm="RandomForest",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metrics={"accuracy": 0.95},
            parameters={"n_estimators": 10},
            feature_names=["feat1", "feat2"],
            target_name="target",
            dataset_hash="abc123",
            pipeline_hash="def456",
            tenant_id="test_tenant"
        )
    
    def test_save_and_load_model(self, local_storage, sample_model, sample_metadata):
        """Test saving and loading a model."""
        # Save model
        path = local_storage.save_model(sample_model, sample_metadata)
        assert path is not None
        assert Path(path).exists()
        
        # Load model
        loaded_model, loaded_metadata = local_storage.load_model(
            "test_model", "1.0.0", "test_tenant"
        )
        
        assert loaded_model is not None
        assert loaded_metadata["model_id"] == "test_model"
        assert loaded_metadata["version"] == "1.0.0"
    
    def test_save_model_with_encryption(self, sample_model, sample_metadata):
        """Test saving model with encryption."""
        from cryptography.fernet import Fernet
        encryption_key = Fernet.generate_key()
        
        temp_dir = tempfile.mkdtemp()
        storage = LocalStorage(base_path=temp_dir, encryption_key=encryption_key)
        
        sample_metadata.encrypted = True
        
        # Save model
        path = storage.save_model(sample_model, sample_metadata)
        assert path is not None
        
        # Verify file is encrypted
        with open(path, 'rb') as f:
            encrypted_data = f.read()
        
        # Should not be able to unpickle directly
        with pytest.raises(Exception):
            pickle.loads(encrypted_data)
        
        # Load with proper decryption
        loaded_model, _ = storage.load_model(
            "test_model", "1.0.0", "test_tenant"
        )
        assert loaded_model is not None
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_save_and_load_dataset(self, local_storage):
        """Test saving and loading a dataset."""
        # Create sample dataset
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # Save dataset
        path = local_storage.save_dataset(
            df, "test_dataset", "test_tenant", format="parquet"
        )
        assert path is not None
        
        # Load dataset
        loaded_df = local_storage.load_dataset("test_dataset", "test_tenant")
        pd.testing.assert_frame_equal(df, loaded_df)
    
    def test_save_dataset_formats(self, local_storage):
        """Test saving dataset in different formats."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        formats = ['parquet', 'csv', 'json']
        
        for fmt in formats:
            path = local_storage.save_dataset(
                df, f"test_dataset_{fmt}", "test_tenant", format=fmt
            )
            assert path is not None
            
            loaded_df = local_storage.load_dataset(f"test_dataset_{fmt}", "test_tenant")
            assert len(loaded_df) == len(df)
    
    def test_list_models(self, local_storage, sample_model, sample_metadata):
        """Test listing models."""
        # Save multiple models
        local_storage.save_model(sample_model, sample_metadata)
        
        sample_metadata.version = "2.0.0"
        local_storage.save_model(sample_model, sample_metadata)
        
        # List models
        models = local_storage.list_models("test_tenant")
        assert len(models) == 2
        assert all(m["model_id"] == "test_model" for m in models)
    
    def test_delete_model(self, local_storage, sample_model, sample_metadata):
        """Test deleting a model."""
        # Save model
        local_storage.save_model(sample_model, sample_metadata)

        # Delete specific version
        success = local_storage.delete_model("test_model", "1.0.0", "test_tenant")
        assert success

        # Verify deletion
        with pytest.raises(Exception):
            local_storage.load_model("test_model", "1.0.0", "test_tenant")

    def test_get_latest_version(self, local_storage, sample_model, sample_metadata):
        """Test getting latest version."""
        # Save multiple versions
        versions = ["1.0.0", "1.0.1", "2.0.0", "1.1.0"]
        
        for v in versions:
            sample_metadata.version = v
            local_storage.save_model(sample_model, sample_metadata)
        
        # Get latest version
        latest = local_storage._get_latest_version("test_model", "test_tenant")
        assert latest == "2.0.0"


class TestMinIOStorage:
    """Test MinIO/S3 storage backend."""
    
    @pytest.fixture
    def mock_minio_client(self):
        """Create mock MinIO client."""
        with patch('automl_platform.storage.Minio') as mock_minio:
            client = MagicMock()
            mock_minio.return_value = client
            
            # Setup bucket operations
            client.bucket_exists.return_value = True
            client.make_bucket.return_value = None
            
            yield client
    
    @pytest.fixture
    def minio_storage(self, mock_minio_client):
        """Create MinIOStorage instance with mock client."""
        with patch('automl_platform.storage.MINIO_AVAILABLE', True):
            storage = MinIOStorage(access_key="test-access", secret_key="test-secret")
            storage.client = mock_minio_client
            return storage
    
    def test_ensure_buckets(self, mock_minio_client):
        """Test bucket creation."""
        mock_minio_client.bucket_exists.return_value = False
        
        with patch('automl_platform.storage.MINIO_AVAILABLE', True):
            storage = MinIOStorage(access_key="test-access", secret_key="test-secret")
            storage.client = mock_minio_client
            storage._ensure_buckets()
        
        # Should create 3 buckets
        assert mock_minio_client.make_bucket.call_count >= 3
    
    def test_save_model_minio(self, minio_storage, mock_minio_client):
        """Test saving model to MinIO."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        
        metadata = ModelMetadata(
            model_id="test_model",
            version="1.0.0",
            model_type="classification",
            algorithm="RandomForest",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metrics={},
            parameters={},
            feature_names=[],
            target_name="target",
            dataset_hash="",
            pipeline_hash="",
            tenant_id="test_tenant"
        )
        
        # Save model
        path = minio_storage.save_model(model, metadata)
        
        # Verify put_object was called
        assert mock_minio_client.put_object.called
        assert "test_model" in path
    
    def test_load_model_minio(self, minio_storage, mock_minio_client):
        """Test loading model from MinIO."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.read.side_effect = [
            pickle.dumps(model),  # Model data
            json.dumps({"model_id": "test_model", "version": "1.0.0"}).encode()  # Metadata
        ]
        mock_minio_client.get_object.return_value = mock_response
        
        # Load model
        loaded_model, loaded_metadata = minio_storage.load_model(
            "test_model", "1.0.0", "test_tenant"
        )
        
        assert loaded_model is not None
        assert loaded_metadata["model_id"] == "test_model"
    
    def test_encryption_decryption(self, minio_storage):
        """Test data encryption and decryption."""
        from cryptography.fernet import Fernet
        
        encryption_key = Fernet.generate_key()
        minio_storage.encryption_key = encryption_key
        
        test_data = b"test data"
        
        # Encrypt
        encrypted = minio_storage._encrypt_data(test_data)
        assert encrypted != test_data
        
        # Decrypt
        decrypted = minio_storage._decrypt_data(encrypted)
        assert decrypted == test_data


class TestStorageManager:
    """Test main StorageManager class."""
    
    @pytest.fixture
    def storage_manager(self):
        """Create StorageManager with local backend."""
        temp_dir = tempfile.mkdtemp()
        manager = StorageManager(backend="local", base_path=temp_dir)
        yield manager
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_storage_manager_initialization(self):
        """Test StorageManager initialization."""
        temp_dir = tempfile.mkdtemp()
        
        # Test local backend
        manager = StorageManager(backend="local", base_path=temp_dir)
        assert isinstance(manager.backend, LocalStorage)
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_save_and_load_model(self, storage_manager):
        """Test saving and loading model through manager."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        metadata = {
            'model_id': 'test_model',
            'model_type': 'classification',
            'algorithm': 'RandomForest',
            'metrics': {'accuracy': 0.95},
            'feature_names': ['feat1', 'feat2']
        }

        # Save model
        path = storage_manager.save_model(model, metadata, version="1.0.0")
        assert path is not None

        # Load model
        loaded_model, loaded_metadata = storage_manager.load_model(
            "test_model", "1.0.0"
        )

        assert loaded_model is not None
        assert loaded_metadata["model_id"] == "test_model"

    def test_load_from_connector_with_mock(self, storage_manager):
        """StorageManager should instantiate and use configured connectors."""

        captured = {}

        class DummyConnector:
            def __init__(self, config):
                captured['config'] = config

            def query(self, query):
                captured['query'] = query
                return pd.DataFrame({'value': [1]})

            def read_table(self, table):  # pragma: no cover - not used in this test
                captured['table'] = table
                return pd.DataFrame({'value': [1]})

        storage_manager.connectors = {'dummy': DummyConnector}

        result = storage_manager.load_from_connector(
            'dummy',
            {
                'connection_type': 'dummy',
                'tenant_id': 'tenant_123',
                'host': 'localhost'
            },
            query='SELECT 1'
        )

        assert isinstance(result, pd.DataFrame)
        assert result.equals(pd.DataFrame({'value': [1]}))
        assert 'config' in captured and 'query' in captured
        assert captured['query'] == 'SELECT 1'
        assert isinstance(captured['config'], storage_manager._connector_config_cls)

    def test_load_from_connector_warns_for_legacy_keys(self, storage_manager):
        """Legacy connection keys should emit a deprecation warning when used."""

        captured = {}

        class DummyConnector:
            def __init__(self, config):
                captured['config'] = config

            def query(self, query):
                captured['query'] = query
                return pd.DataFrame({'value': [query]})

            def read_table(self, table):  # pragma: no cover - fallback path
                return pd.DataFrame({'value': [table]})

        storage_manager.connectors = {'dummy': DummyConnector}

        with pytest.warns(DeprecationWarning):
            result = storage_manager.load_from_connector(
                'dummy',
                {
                    'connection_type': 'dummy',
                    'tenant_id': 'tenant_123',
                    'user': 'legacy_user',
                    'dbname': 'legacy_db',
                },
                query='legacy',
            )

        assert isinstance(result, pd.DataFrame)
        assert list(result['value']) == ['legacy']
        assert captured['config'].username == 'legacy_user'
        assert captured['config'].database == 'legacy_db'
        assert captured['config'].tenant_id == 'tenant_123'
        assert captured['query'] == 'legacy'
        assert captured['config'].connection_type == 'dummy'
        assert captured['config'].tenant_id == 'tenant_123'

    def test_none_backend_fails_fast(self):
        """The 'none' storage backend should raise a descriptive error."""

        manager = StorageManager(backend="none")
        metadata = {
            "model_id": "noop",
            "model_type": "noop",
            "algorithm": "noop",
            "metrics": {},
            "parameters": {},
            "feature_names": [],
            "target_name": "target",
            "dataset_hash": "noop",
            "pipeline_hash": "noop",
        }

        with pytest.raises(StorageDisabledError, match="storage.backend='none'"):
            manager.save_model(object(), metadata)

        with pytest.raises(StorageDisabledError, match="storage.backend='none'"):
            manager.load_model("noop")

    def test_get_model_history(self, storage_manager):
        """Test getting model version history."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        
        # Save multiple versions
        for v in ["1.0.0", "1.0.1", "2.0.0"]:
            metadata = {
                'model_id': 'test_model',
                'model_type': 'classification',
                'algorithm': 'RandomForest'
            }
            storage_manager.save_model(model, metadata, version=v)
        
        # Get history
        history = storage_manager.get_model_history("test_model")
        assert len(history) == 3
    
    @patch('automl_platform.storage.joblib')
    def test_export_model_to_docker(self, mock_joblib, storage_manager):
        """Test exporting model to Docker."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        
        # Save model first
        metadata = {
            'model_id': 'test_model',
            'model_type': 'classification',
            'algorithm': 'RandomForest'
        }
        storage_manager.save_model(model, metadata, version="1.0.0")
        
        # Export to Docker
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = storage_manager.export_model_to_docker(
                "test_model", "1.0.0", output_dir=temp_dir
            )
            
            # Check files were created
            output_dir = Path(output_path)
            assert output_dir.exists()
            assert (output_dir / "app.py").exists()
            assert (output_dir / "requirements.txt").exists()
            assert (output_dir / "Dockerfile").exists()
            assert (output_dir / "docker-compose.yml").exists()
            assert (output_dir / "k8s-deployment.yaml").exists()
            assert (output_dir / "deploy.sh").exists()

    def test_storage_manager_gcs_backend(self, monkeypatch):
        """Ensure StorageManager can initialize the GCS backend."""

        from automl_platform import storage as storage_module

        class FakeBlob:
            def __init__(self, name):
                self.name = name
                self.data = b""
                self.bucket = None
                self.updated = datetime.now()

            def upload_from_file(self, file_obj, **kwargs):
                file_obj.seek(0)
                self.data = file_obj.read()
                self.updated = datetime.now()

            def upload_from_string(self, data, **kwargs):
                if isinstance(data, str):
                    data = data.encode()
                self.data = data
                self.updated = datetime.now()

            def download_as_bytes(self):
                return self.data

            def delete(self):
                if self.bucket:
                    self.bucket.delete_blob(self.name)

        class FakeBucket:
            def __init__(self, name):
                self.name = name
                self._blobs: Dict[str, FakeBlob] = {}

            def blob(self, name):
                if name not in self._blobs:
                    blob = FakeBlob(name)
                    blob.bucket = self
                    self._blobs[name] = blob
                return self._blobs[name]

            def list_blobs(self, prefix=None):
                blobs = list(self._blobs.values())
                if prefix:
                    blobs = [blob for blob in blobs if blob.name.startswith(prefix)]
                return blobs

            def delete_blob(self, name):
                self._blobs.pop(name, None)

        class FakeClient:
            def __init__(self):
                self._buckets: Dict[str, FakeBucket] = {}

            def lookup_bucket(self, name):
                return self._buckets.get(name)

            def create_bucket(self, name):
                bucket = FakeBucket(name)
                self._buckets[name] = bucket
                return bucket

            def bucket(self, name):
                return self._buckets.setdefault(name, FakeBucket(name))

        monkeypatch.setattr(storage_module, "GCS_AVAILABLE", True)
        monkeypatch.setattr(StorageManager, "_init_connectors", lambda self: None)

        fake_client = FakeClient()

        manager = StorageManager(
            backend="gcs",
            client=fake_client,
            models_bucket="models",
            datasets_bucket="datasets",
            artifacts_bucket="artifacts",
        )

        assert isinstance(manager.backend, storage_module.GCSStorage)

        model = {"coef": [1, 2, 3]}
        metadata = {
            "model_id": "gcs_model",
            "model_type": "test",
            "algorithm": "dummy",
            "metrics": {},
            "parameters": {},
            "feature_names": [],
            "tenant_id": "tenant1",
        }

        manager.save_model(model, metadata, version="1.0.0")
        loaded_model, loaded_metadata = manager.load_model("gcs_model", "1.0.0", "tenant1")

        assert loaded_model == model
        assert loaded_metadata["model_id"] == "gcs_model"

        dataset = pd.DataFrame({"value": [1, 2, 3]})
        manager.save_dataset(dataset, "dataset1", tenant_id="tenant1", format="parquet")
        loaded_dataset = manager.load_dataset("dataset1", tenant_id="tenant1")

        pd.testing.assert_frame_equal(dataset, loaded_dataset)

    @patch('automl_platform.storage.convert_sklearn')
    def test_export_model_to_onnx(self, mock_convert, storage_manager):
        """Test exporting model to ONNX format."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        
        # Save model first
        metadata = {
            'model_id': 'test_model',
            'model_type': 'classification',
            'algorithm': 'RandomForest',
            'feature_names': ['feat1', 'feat2']
        }
        storage_manager.save_model(model, metadata, version="1.0.0")
        
        # Mock ONNX conversion
        mock_onnx_model = MagicMock()
        mock_onnx_model.SerializeToString.return_value = b"onnx_model_bytes"
        mock_convert.return_value = mock_onnx_model
        
        # Export to ONNX
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "model.onnx")

            with patch('automl_platform.storage.ONNX_AVAILABLE', True):
                result = storage_manager.export_model_to_onnx(
                    "test_model", "1.0.0", output_path=output_path
                )

            assert result == output_path

    @patch('automl_platform.storage.sklearn2pmml')
    def test_export_model_to_pmml(self, mock_sklearn2pmml, storage_manager):
        """Test exporting model to PMML format."""
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        
        # Save model first
        metadata = {
            'model_id': 'test_model',
            'model_type': 'classification',
            'algorithm': 'RandomForest'
        }
        storage_manager.save_model(model, metadata, version="1.0.0")
        
        # Export to PMML
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "model.pmml")
            with patch('automl_platform.storage.PMML_AVAILABLE', True):
                result = storage_manager.export_model_to_pmml(
                    "test_model", "1.0.0", output_path=output_path
                )

            # Verify sklearn2pmml was called
            mock_sklearn2pmml.assert_called_once()


class TestFeatureStore:
    """Test FeatureStore functionality."""
    
    @pytest.fixture
    def feature_store(self):
        """Create FeatureStore instance."""
        temp_dir = tempfile.mkdtemp()
        storage_manager = StorageManager(backend="local", base_path=temp_dir)
        store = FeatureStore(storage_manager)
        yield store
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def limited_feature_store(self):
        """FeatureStore with strict cache limits for eviction tests."""
        temp_dir = tempfile.mkdtemp()
        storage_manager = StorageManager(backend="local", base_path=temp_dir)
        store = FeatureStore(
            storage_manager,
            cache_max_entries=2,
            cache_max_memory_mb=0.001,
            cache_ttl_seconds=None,
        )
        yield store
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def ttl_feature_store(self):
        """FeatureStore with controllable clock for TTL expiration tests."""

        class FakeClock:
            def __init__(self, start: float = 0.0):
                self._time = start

            def time(self) -> float:
                return self._time

            def advance(self, seconds: float) -> None:
                self._time += seconds

        temp_dir = tempfile.mkdtemp()
        storage_manager = StorageManager(backend="local", base_path=temp_dir)
        clock = FakeClock()
        store = FeatureStore(
            storage_manager,
            cache_max_entries=4,
            cache_ttl_seconds=5,
            time_provider=clock.time,
        )
        yield store, clock
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_register_feature(self, feature_store):
        """Test registering a feature computation."""
        def compute_mean(df):
            return df.mean(axis=1)
        
        feature_store.register_feature(
            name="mean_feature",
            description="Mean of all columns",
            computation_func=compute_mean
        )
        
        assert "mean_feature" in feature_store.feature_registry
        assert feature_store.feature_registry["mean_feature"]["description"] == "Mean of all columns"
    
    def test_compute_features(self, feature_store):
        """Test computing features."""
        # Register features
        def compute_mean(df):
            return df[['col1', 'col2']].mean(axis=1)
        
        def compute_sum(df):
            return df[['col1', 'col2']].sum(axis=1)
        
        feature_store.register_feature(
            name="mean_feature",
            description="Mean",
            computation_func=compute_mean
        )
        
        feature_store.register_feature(
            name="sum_feature",
            description="Sum",
            computation_func=compute_sum
        )
        
        # Create test data
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        # Compute features
        result = feature_store.compute_features(df, ["mean_feature", "sum_feature"])
        
        assert "mean_feature" in result.columns
        assert "sum_feature" in result.columns
        assert result["mean_feature"].iloc[0] == 2.5
        assert result["sum_feature"].iloc[0] == 5
    
    def test_save_and_load_features(self, feature_store):
        """Test saving and loading computed features."""
        # Create test features
        features = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        # Save features
        path = feature_store.save_features(
            features,
            "test_features",
            version="1.0.0"
        )
        assert path is not None
        
        # Load features
        loaded_features = feature_store.load_features("test_features", "1.0.0")
        pd.testing.assert_frame_equal(features, loaded_features)
    
    def test_feature_caching(self, feature_store):
        """Test feature caching."""
        features = pd.DataFrame({
            'feature1': [1, 2, 3]
        })
        
        # Save features
        feature_store.save_features(features, "cached_features", "1.0.0")
        
        # First load - from storage
        loaded1 = feature_store.load_features("cached_features", "1.0.0")
        
        # Second load - should be from cache
        loaded2 = feature_store.load_features("cached_features", "1.0.0")
        
        # Both should be identical
        pd.testing.assert_frame_equal(loaded1, loaded2)
        
        # Check cache
        assert "cached_features:1.0.0" in feature_store.get_cached_keys()

    def test_feature_cache_eviction(self, limited_feature_store):
        """Ensure least recently used feature sets are evicted when capacity is exceeded."""
        df = pd.DataFrame({'value': [1, 2, 3]})

        limited_feature_store.save_features(df, "set1", "1.0.0")
        limited_feature_store.save_features(df, "set2", "1.0.0")

        # Access set1 to mark it as most recently used
        limited_feature_store.load_features("set1", "1.0.0")

        # Saving a third set should evict set2 (the least recently used)
        limited_feature_store.save_features(df, "set3", "1.0.0")

        cached_keys = limited_feature_store.get_cached_keys()
        assert len(cached_keys) == 2
        assert "set1:1.0.0" in cached_keys
        assert "set3:1.0.0" in cached_keys
        assert "set2:1.0.0" not in cached_keys

    def test_feature_cache_ttl_expiration(self, ttl_feature_store):
        """Verify cached entries expire after the configured TTL."""
        store, clock = ttl_feature_store
        df = pd.DataFrame({'value': [10, 20, 30]})

        store.save_features(df, "ttl_set", "1.0.0")
        assert "ttl_set:1.0.0" in store.get_cached_keys()

        # Advance time beyond TTL so that cache entry expires
        clock.advance(6)

        with patch.object(store.storage, 'load_dataset', wraps=store.storage.load_dataset) as mocked_load:
            reloaded = store.load_features("ttl_set", "1.0.0")

        # Entry should be reloaded from storage after expiration
        mocked_load.assert_called_once()
        pd.testing.assert_frame_equal(reloaded, df)
        cached_keys = store.get_cached_keys()
        assert "ttl_set:1.0.0" in cached_keys
        assert len(cached_keys) == 1

        stats = store.get_cache_stats()
        assert stats["evictions"] >= 1

    def test_list_feature_sets(self, feature_store):
        """Test listing available feature sets."""
        # Save multiple feature sets
        features1 = pd.DataFrame({'f1': [1, 2]})
        features2 = pd.DataFrame({'f2': [3, 4]})

        feature_store.save_features(features1, "set1", "1.0.0")
        feature_store.save_features(features2, "set2", "1.0.0")

        # List feature sets
        sets = feature_store.list_feature_sets()
        assert "set1" in sets
        assert "set2" in sets

    def test_feature_store_default_configuration(self, feature_store):
        """Defaults should reflect conservative cache sizing and TTL."""

        assert feature_store.cache_max_entries == 100
        assert feature_store.cache_ttl_seconds == 3600
        assert feature_store.cache_max_memory_bytes == 500 * 1024 * 1024

    def test_feature_cache_memory_limit_eviction(self):
        """Ensure oversized cache totals trigger LRU evictions respecting memory limit."""

        temp_dir = tempfile.mkdtemp()
        try:
            storage_manager = StorageManager(backend="local", base_path=temp_dir)
            store = FeatureStore(
                storage_manager,
                cache_max_entries=10,
                cache_max_memory_mb=0.0012,
                cache_ttl_seconds=None,
            )

            large_df = pd.DataFrame({'value': ['x' * 512]})

            store.save_features(large_df, "large1", "1.0.0")
            store.save_features(large_df, "large2", "1.0.0")

            cached_keys = store.get_cached_keys()
            assert cached_keys == ["large2:1.0.0"]

            stats = store.get_cache_stats()
            assert stats["evictions"] >= 1
            assert stats["current_memory_bytes"] <= stats["max_memory_bytes"]
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_feature_cache_skips_entries_larger_than_limit(self):
        """Entries larger than the configured max memory should not be cached."""

        temp_dir = tempfile.mkdtemp()
        try:
            storage_manager = StorageManager(backend="local", base_path=temp_dir)
            store = FeatureStore(
                storage_manager,
                cache_max_entries=10,
                cache_max_memory_mb=0.0001,
                cache_ttl_seconds=None,
            )

            huge_df = pd.DataFrame({'value': ['y' * 2048]})

            store.save_features(huge_df, "too_big", "1.0.0")

            assert "too_big:1.0.0" not in store.get_cached_keys()
            stats = store.get_cache_stats()
            assert stats["current_memory_bytes"] == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_feature_cache_metrics(self, limited_feature_store):
        """Cache hit/miss counters should reflect access patterns."""

        df = pd.DataFrame({'value': [42]})

        limited_feature_store.save_features(df, "metrics", "1.0.0")

        # Use a fresh store to ensure a miss followed by a hit.
        storage_manager = limited_feature_store.storage
        metrics_store = FeatureStore(
            storage_manager,
            cache_max_entries=2,
            cache_max_memory_mb=0.001,
            cache_ttl_seconds=None,
        )

        metrics_store.load_features("metrics", "1.0.0")
        metrics_store.load_features("metrics", "1.0.0")

        stats = metrics_store.get_cache_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["hits"] + stats["misses"] >= 2


class TestConnectorIntegration:
    """Test connector integration with storage."""
    
    def test_load_from_connector(self):
        """Test loading data from connector."""
        temp_dir = tempfile.mkdtemp()
        storage_manager = StorageManager(backend="local", base_path=temp_dir)
        
        # Mock connector
        with patch.object(storage_manager, 'connectors') as mock_connectors:
            mock_connector_class = MagicMock()
            mock_connector_instance = MagicMock()
            mock_connector_class.return_value = mock_connector_instance
            
            # Setup mock data
            mock_data = pd.DataFrame({'col1': [1, 2, 3]})
            mock_connector_instance.query.return_value = mock_data
            
            mock_connectors.__getitem__.return_value = mock_connector_class
            mock_connectors.__contains__.return_value = True
            
            # Load from connector
            config = {'host': 'localhost', 'user': 'test'}
            data = storage_manager.load_from_connector(
                'postgresql',
                config,
                query="SELECT * FROM test"
            )
            
            # Verify
            mock_connector_instance.query.assert_called_once_with("SELECT * FROM test")
            pd.testing.assert_frame_equal(data, mock_data)
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestStorageEdgeCases:
    """Test edge cases and error handling."""
    
    def test_load_nonexistent_model(self):
        """Test loading a model that doesn't exist."""
        temp_dir = tempfile.mkdtemp()
        storage = LocalStorage(base_path=temp_dir)
        
        with pytest.raises(Exception):
            storage.load_model("nonexistent", "1.0.0")
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_save_model_with_invalid_metadata(self):
        """Test saving model with invalid metadata."""
        temp_dir = tempfile.mkdtemp()
        storage = LocalStorage(base_path=temp_dir)
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        
        # Create metadata with missing required fields
        metadata = ModelMetadata(
            model_id="",  # Invalid empty ID
            version="invalid_version",  # Invalid version format
            model_type="classification",
            algorithm="RandomForest",
            created_at="",
            updated_at="",
            metrics={},
            parameters={},
            feature_names=[],
            target_name="",
            dataset_hash="",
            pipeline_hash=""
        )
        
        # Should still save but with warnings
        path = storage.save_model(model, metadata)
        assert path is not None
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_concurrent_access(self):
        """Test concurrent access to storage."""
        import threading
        
        temp_dir = tempfile.mkdtemp()
        storage = LocalStorage(base_path=temp_dir)
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        
        def save_model(version):
            metadata = ModelMetadata(
                model_id="concurrent_model",
                version=version,
                model_type="classification",
                algorithm="RandomForest",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                metrics={},
                parameters={},
                feature_names=[],
                target_name="target",
                dataset_hash="",
                pipeline_hash=""
            )
            storage.save_model(model, metadata)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=save_model, args=(f"1.0.{i}",))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check all models were saved
        models = storage.list_models()
        assert len(models) == 5
        
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
