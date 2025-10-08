"""
Tests for ConfigManager
=======================
Comprehensive tests for the configuration management system.
"""

import pytest
import os
import yaml
from dataclasses import is_dataclass
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from automl_platform.core.config_manager import ConfigManager
from automl_platform.config import (
    AutoMLConfig,
    StorageConfig,
    WorkerConfig,
    BillingConfig,
    MonitoringConfig,
    LLMConfig,
    APIConfig,
    RGPDConfig,
    StreamingConfig,
    PlanType,
    load_config
)


def test_repo_config_uses_free_plan_type_by_default():
    """Ensure the repository config YAML keeps the free billing plan type."""
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with config_path.open() as handle:
        config_dict = yaml.safe_load(handle)

    assert config_dict["billing"]["plan_type"] == "free"


def test_billing_config_accepts_default_plan_alias():
    """Legacy default_plan key should map to plan_type with a deprecation warning."""
    with pytest.deprecated_call():
        config = BillingConfig(enabled=False, default_plan="trial")

    assert config.plan_type == "trial"
    assert config.default_plan is None


class TestConfigManager:
    """Test suite for ConfigManager."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'environment': 'test',
                'storage': {
                    'backend': 's3',
                    'endpoint': 'http://localhost:9000',
                    'access_key': 'test_key',
                    'secret_key': 'test_secret'
                },
                'worker': {
                    'enabled': True,
                    'max_workers': 4,
                    'gpu_workers': 0
                },
                'billing': {
                    'enabled': True,
                    'plan_type': 'professional'
                }
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_initialization_with_config_file(self, temp_config_file):
        """Test ConfigManager initialization with a real config file."""
        manager = ConfigManager(config_path=temp_config_file, environment="test")
        
        assert manager.config is not None
        assert manager.environment == "test"
        assert manager.config.storage.backend == "s3"
        assert manager.config.worker.max_workers == 4
        assert manager.config.billing.plan_type == "professional"
        assert manager.config_hash is not None
        assert len(manager.config_history) == 1
    
    def test_initialization_with_missing_file(self):
        """Test ConfigManager initialization with missing config file (uses defaults)."""
        manager = ConfigManager(config_path="non_existent.yaml", environment="development")
        
        assert manager.config is not None
        assert manager.environment == "development"
        # Should use default configuration
        assert isinstance(manager.config, AutoMLConfig)
        assert manager.config_hash is not None
    
    def test_environment_variable_overrides(self):
        """Test that environment variables override config file values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'storage': {'backend': 'local'},
                'worker': {'max_workers': 2},
                'api': {'port': 8000}
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Set environment variables
            env_vars = {
                'STORAGE_BACKEND': 's3',
                'MINIO_ENDPOINT': 'http://minio:9000',
                'MINIO_ACCESS_KEY': 'env_access_key',
                'MINIO_SECRET_KEY': 'env_secret_key',
                'MAX_WORKERS': '8',
                'API_PORT': '9000',
                'OPENAI_API_KEY': 'env_api_key',
                'STRIPE_API_KEY': 'env_stripe_key',
                'BILLING_PLAN': 'enterprise'
            }
            
            with patch.dict(os.environ, env_vars):
                manager = ConfigManager(config_path=temp_path)
                
                # Verify environment overrides
                assert manager.config.storage.backend == 's3'
                assert manager.config.storage.endpoint == 'http://minio:9000'
                assert manager.config.storage.access_key == 'env_access_key'
                assert manager.config.storage.secret_key == 'env_secret_key'
                assert manager.config.worker.max_workers == 8
                assert manager.config.api.port == 9000
                assert manager.config.llm.api_key == 'env_api_key'
                assert manager.config.billing.stripe_api_key == 'env_stripe_key'
                assert manager.config.billing.plan_type == 'enterprise'
        
        finally:
            os.unlink(temp_path)

    def test_from_yaml_propagates_agent_first_flag(self, tmp_path):
        """Agent-First nested flag should enable top-level setting when loading YAML."""
        config_path = tmp_path / "agent_first.yaml"
        with config_path.open('w') as f:
            yaml.safe_dump({'agent_first': {'enabled': True}}, f)

        config = AutoMLConfig.from_yaml(str(config_path))

        assert config.agent_first.enabled is True
        assert config.enable_agent_first is True

    def test_from_yaml_propagates_hybrid_mode_flag(self, tmp_path):
        """Nested hybrid flag should populate top-level enable_hybrid_mode when missing."""
        config_path = tmp_path / "agent_first_hybrid.yaml"
        with config_path.open('w') as f:
            yaml.safe_dump({'agent_first': {'enabled': True, 'enable_hybrid_mode': False}}, f)

        config = AutoMLConfig.from_yaml(str(config_path))

        assert config.agent_first.enable_hybrid_mode is False
        assert config.enable_hybrid_mode is False

    def test_from_yaml_conflicting_agent_first_prefers_nested(self, tmp_path, caplog):
        """Nested Agent-First flag should override conflicting top-level flag from YAML."""
        config_path = tmp_path / "agent_first_conflict.yaml"
        with config_path.open('w') as f:
            yaml.safe_dump({'enable_agent_first': False, 'agent_first': {'enabled': True}}, f)

        with caplog.at_level("WARNING"):
            config = AutoMLConfig.from_yaml(str(config_path))

        assert config.enable_agent_first is True
        assert "Conflicting Agent-First flags in YAML" in caplog.text

    def test_from_yaml_nested_flag_respects_env_override(self, tmp_path, caplog):
        """Environment variable should take precedence over nested Agent-First flag."""
        config_path = tmp_path / "agent_first_env.yaml"
        with config_path.open('w') as f:
            yaml.safe_dump({'agent_first': {'enabled': True}}, f)

        with patch.dict(os.environ, {"AUTOML_AGENT_FIRST": "0"}):
            with caplog.at_level("WARNING"):
                config = AutoMLConfig.from_yaml(str(config_path))

        assert config.enable_agent_first is False
        assert "AUTOML_AGENT_FIRST environment variable overrides nested" in caplog.text

    def test_load_config_builds_dataclasses_for_nested_sections(self, tmp_path):
        """load_config should convert nested dictionaries into dataclass instances."""
        config_path = tmp_path / "config.yaml"
        config_payload = {
            'database': {
                'url': 'sqlite:///tmp.db',
                'audit_url': 'postgresql://user:pass@localhost/audit_test'
            },
            'security': {
                'secret_key': 'unit-test-secret'
            },
            'api': {
                'enable_auth': False,
                'port': 9001,
            },
            'llm': {
                'model_name': 'gpt-3.5-turbo',
                'enable_agent_first': False,
            },
            'rgpd': {
                'enabled': False,
            },
            'connectors': {
                'default_connector': 'bigquery',
            },
        }

        with config_path.open('w') as handle:
            yaml.safe_dump(config_payload, handle)

        config = load_config(filepath=str(config_path))

        assert is_dataclass(config.database)
        assert config.database.url == 'sqlite:///tmp.db'
        assert config.database.audit_url == 'postgresql://user:pass@localhost/audit_test'
        assert is_dataclass(config.security)
        assert config.security.secret_key == 'unit-test-secret'
        assert is_dataclass(config.api)
        assert config.api.enable_auth is False
        assert is_dataclass(config.llm)
        assert config.llm.model_name == 'gpt-3.5-turbo'
        assert is_dataclass(config.rgpd)
        assert config.rgpd.enabled is False
        assert is_dataclass(config.connectors)
        assert config.connectors.default_connector == 'bigquery'

    def test_get_hybrid_mode_flag_prefers_nested(self):
        """AutoMLConfig should expose helper resolving hybrid mode from nested config."""
        config = AutoMLConfig()
        config.enable_hybrid_mode = True
        config.agent_first.enable_hybrid_mode = False

        assert config.get_hybrid_mode_flag() is False

    def test_get_hybrid_mode_flag_defaults_to_top_level(self):
        """Hybrid helper should fall back to top-level flag when nested is undefined."""
        config = AutoMLConfig()
        config.enable_hybrid_mode = False
        config.agent_first.enable_hybrid_mode = None

        assert config.get_hybrid_mode_flag() is False

    def test_config_validation_cross_component(self):
        """Test cross-component validation with various inconsistencies."""
        manager = ConfigManager(config_path="non_existent.yaml")

        # Test 1: Worker count exceeds plan limit
        manager.config.billing.enabled = True
        manager.config.billing.plan_type = 'starter'
        starter_limit = manager.config.billing.quotas['starter']['max_workers']
        manager.config.worker.max_workers = starter_limit + 5

        manager._validate_cross_component()
        assert manager.config.worker.max_workers == starter_limit  # Should be adjusted

        # Test 2: GPU workers without plan support
        manager.config.worker.gpu_workers = 5

        manager._validate_cross_component()
        assert manager.config.worker.gpu_workers == 0  # Should be disabled

        # Test 3: Professional plan should retain GPU workers and adjust worker count
        manager.config.billing.plan_type = 'professional'
        professional_limit = manager.config.billing.quotas['professional']['max_workers']
        manager.config.worker.max_workers = professional_limit + 3
        manager.config.worker.gpu_workers = 4

        manager._validate_cross_component()
        assert manager.config.worker.max_workers == professional_limit
        assert manager.config.worker.gpu_workers == 4  # GPU allowed for professional

        # Test 4: Kafka streaming without brokers
        manager.config.streaming.enabled = True
        manager.config.streaming.platform = "kafka"
        manager.config.streaming.brokers = []

        with pytest.raises(ValueError, match="Kafka brokers not configured"):
            manager._validate_cross_component()

        # Test 5: RGPD audit requires monitoring
        manager.config.streaming.enabled = False  # Disable to avoid previous error
        manager.config.rgpd.enabled = True
        manager.config.rgpd.audit_all_data_access = True
        manager.config.monitoring.enabled = False
        
        manager._validate_cross_component()
        assert manager.config.monitoring.enabled == True  # Should be enabled
    
    def test_validate_config_raises_exception(self):
        """Test that validation raises exception for invalid configuration."""
        manager = ConfigManager(config_path="non_existent.yaml")

        # Create invalid configuration
        manager.config.streaming.enabled = True
        manager.config.streaming.platform = "kafka"
        manager.config.streaming.brokers = []
        
        with pytest.raises(ValueError):
            manager.validate_config()

    @pytest.mark.parametrize("plan_name", ["starter", "professional", "custom"])
    def test_validate_accepts_extended_billing_plan_types(self, tmp_path, plan_name):
        """Ensure AutoMLConfig validation supports extended billing plans."""
        config = AutoMLConfig()
        base_dir = tmp_path / "config"

        config.output_dir = str(base_dir / "output")
        config.monitoring.report_output_dir = str(base_dir / "monitoring")
        config.storage.local_base_path = str(base_dir / "storage")
        config.agent_first.knowledge_base_path = str(base_dir / "knowledge")
        config.agent_first.yaml_output_dir = str(base_dir / "agent_yaml")

        config.billing.enabled = True
        config.billing.plan_type = plan_name
        if plan_name not in config.billing.quotas:
            config.billing.quotas[plan_name] = {}

        assert config.validate()

    def test_validate_accepts_gcs_backend_without_credentials_path(self):
        """GCS backend validates when relying on ambient credentials."""
        config = AutoMLConfig()
        config.storage.backend = "gcs"
        config.storage.credentials_path = None

        assert config.validate()

    def test_validate_gcs_backend_raises_on_missing_credentials_file(self, tmp_path):
        """Missing credentials path is surfaced during validation for GCS."""
        config = AutoMLConfig()
        config.storage.backend = "gcs"
        config.storage.credentials_path = str(tmp_path / "missing.json")

        with pytest.raises(AssertionError, match="credentials_path"):
            config.validate()

    def test_get_service_config(self):
        """Test getting configuration for specific services."""
        manager = ConfigManager(config_path="non_existent.yaml")

        # Get storage config
        storage_config = manager.get_service_config("storage")
        assert isinstance(storage_config, StorageConfig)
        
        # Get worker config
        worker_config = manager.get_service_config("worker")
        assert isinstance(worker_config, WorkerConfig)
        
        # Get billing config
        billing_config = manager.get_service_config("billing")
        assert isinstance(billing_config, BillingConfig)
        
        # Get non-existent service
        assert manager.get_service_config("non_existent") is None
    
    def test_update_service_config(self):
        """Test updating configuration for a specific service."""
        manager = ConfigManager(config_path="non_existent.yaml")
        
        # Update storage config
        updates = {
            'backend': 'gcs',
            'max_versions_per_model': 10
        }
        success = manager.update_service_config("storage", updates)
        assert success == True
        assert manager.config.storage.backend == 'gcs'
        assert manager.config.storage.max_versions_per_model == 10
        
        # Update with invalid key (should warn but not fail)
        updates = {'invalid_key': 'value'}
        success = manager.update_service_config("storage", updates)
        assert success == True
        
        # Update non-existent service
        success = manager.update_service_config("non_existent", {})
        assert success == False
    
    def test_save_and_reload_config(self):
        """Test saving and reloading configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = ConfigManager(config_path=temp_path)
            
            # Modify config
            manager.config.storage.backend = "modified_backend"
            old_hash = manager.config_hash
            
            # Save config
            success = manager.save_config()
            assert success == True
            
            # Reload config
            changed = manager.reload_config()
            assert changed == True  # Should detect changes
            assert manager.config.storage.backend == "modified_backend"
            assert manager.config_hash != old_hash
            
            # Reload again without changes
            changed = manager.reload_config()
            assert changed == False  # No changes
        
        finally:
            os.unlink(temp_path)
    
    def test_config_history(self):
        """Test that configuration history is maintained."""
        manager = ConfigManager(config_path="non_existent.yaml")
        initial_history_length = len(manager.config_history)
        
        # Make changes to trigger new history entry
        manager.config.storage.backend = "new_backend"
        manager.load_config()  # Reload to trigger history update
        
        assert len(manager.config_history) > initial_history_length
        
        # Check history entry structure
        latest_entry = manager.config_history[-1]
        assert 'timestamp' in latest_entry
        assert 'hash' in latest_entry
        assert 'environment' in latest_entry
        assert 'config' in latest_entry
    
    def test_get_quota_and_check_quota(self):
        """Test quota retrieval and checking."""
        manager = ConfigManager(config_path="non_existent.yaml")
        
        # Set up quotas
        manager.config.billing.plan_type = 'professional'
        manager.config.billing.quotas = {
            'professional': {
                'max_datasets': 100,
                'max_models': 50
            }
        }
        
        # Get quota
        max_datasets = manager.get_quota('max_datasets')
        assert max_datasets == 100
        
        # Check quota
        within_quota = manager.check_quota('max_datasets', 50)
        assert within_quota == True
        
        exceeds_quota = manager.check_quota('max_datasets', 150)
        assert exceeds_quota == False
    
    def test_get_feature_flags(self):
        """Test feature flags based on configuration."""
        manager = ConfigManager(config_path="non_existent.yaml")
        
        # Set up configuration
        manager.config.worker.enabled = True
        manager.config.worker.max_workers = 4
        manager.config.billing.plan_type = 'enterprise'
        manager.config.billing.quotas = {
            'enterprise': {
                'gpu_enabled': True,
                'llm_enabled': True
            }
        }
        manager.config.llm.enabled = True
        manager.config.streaming.enabled = True
        
        flags = manager.get_feature_flags()
        
        assert flags['distributed_training'] == True  # max_workers > 1
        assert flags['gpu_enabled'] == True
        assert flags['llm_enabled'] == True
        assert flags['streaming_enabled'] == True
    
    def test_get_limits(self):
        """Test getting all limits for current plan."""
        manager = ConfigManager(config_path="non_existent.yaml")
        
        manager.config.billing.plan_type = 'professional'
        manager.config.billing.quotas = {
            'professional': {
                'max_datasets': 100,
                'max_dataset_size_mb': 500,
                'max_models': 50,
                'max_concurrent_jobs': 5,
                'max_predictions_per_day': 10000,
                'max_workers': 10,
                'max_api_calls_per_day': 5000,
                'api_rate_limit': 100,
                'max_gpu_hours_per_month': 100,
                'data_retention_days': 30
            }
        }
        
        limits = manager.get_limits()
        
        assert limits['max_datasets'] == 100
        assert limits['max_dataset_size_mb'] == 500
        assert limits['max_models'] == 50
        assert limits['max_concurrent_jobs'] == 5
        assert limits['max_predictions_per_day'] == 10000
        assert limits['max_workers'] == 10
        assert limits['max_api_calls_per_day'] == 5000
        assert limits['api_rate_limit'] == 100
        assert limits['max_gpu_hours_per_month'] == 100
        assert limits['data_retention_days'] == 30
    
    def test_export_config_formats(self):
        """Test exporting configuration in different formats."""
        manager = ConfigManager(config_path="non_existent.yaml")
        
        # Export as YAML
        yaml_export = manager.export_config(format="yaml", include_secrets=True)
        assert isinstance(yaml_export, str)
        parsed_yaml = yaml.safe_load(yaml_export)
        assert 'storage' in parsed_yaml
        
        # Export as JSON
        json_export = manager.export_config(format="json", include_secrets=True)
        assert isinstance(json_export, str)
        parsed_json = json.loads(json_export)
        assert 'storage' in parsed_json
        
        # Export as environment variables
        env_export = manager.export_config(format="env", include_secrets=True)
        assert isinstance(env_export, str)
        assert 'AUTOML_' in env_export
        
        # Test invalid format
        with pytest.raises(ValueError, match="Unsupported export format"):
            manager.export_config(format="invalid")
    
    def test_export_config_without_secrets(self):
        """Test that secrets are removed when exporting without include_secrets."""
        manager = ConfigManager(config_path="non_existent.yaml")
        
        # Set some secrets
        manager.config.storage.access_key = "secret_access_key"
        manager.config.storage.secret_key = "secret_secret_key"
        manager.config.api.jwt_secret = "jwt_secret_value"
        manager.config.billing.stripe_api_key = "stripe_key"
        
        # Export without secrets
        json_export = manager.export_config(format="json", include_secrets=False)
        parsed = json.loads(json_export)
        
        # Check that secrets are redacted
        assert parsed['storage']['access_key'] == "***REDACTED***"
        assert parsed['storage']['secret_key'] == "***REDACTED***"
        assert parsed['api']['jwt_secret'] == "***REDACTED***"
        assert parsed['billing']['stripe_api_key'] == "***REDACTED***"
    
    def test_get_config_diff(self):
        """Test getting differences between configurations."""
        manager = ConfigManager(config_path="non_existent.yaml")
        
        # Create another config with differences
        other_config = AutoMLConfig()
        other_config.storage.backend = "different_backend"
        other_config.worker.max_workers = 99
        
        diff = manager.get_config_diff(other_config)
        
        # Check that differences are detected
        assert 'storage.backend' in diff
        assert diff['storage.backend']['old'] == manager.config.storage.backend
        assert diff['storage.backend']['new'] == "different_backend"
        
        assert 'worker.max_workers' in diff
        assert diff['worker.max_workers']['old'] == manager.config.worker.max_workers
        assert diff['worker.max_workers']['new'] == 99
    
    def test_get_config_summary(self):
        """Test getting configuration summary."""
        manager = ConfigManager(config_path="non_existent.yaml")
        
        # Set up some configuration
        manager.config.billing.plan_type = 'professional'
        manager.config.storage.backend = 's3'
        manager.config.worker.backend = 'celery'
        manager.config.worker.max_workers = 5
        manager.config.monitoring.enabled = True
        manager.config.streaming.enabled = True
        manager.config.streaming.platform = 'kafka'
        
        summary = manager.get_config_summary()
        
        assert summary['environment'] == manager.environment
        assert summary['plan'] == 'professional'
        assert summary['hash'] is not None
        assert 'features' in summary
        assert 'limits' in summary
        assert 'services' in summary
        
        # Check services summary
        assert summary['services']['storage'] == 's3'
        assert 'celery' in summary['services']['worker']
        assert '5 workers' in summary['services']['worker']
        assert summary['services']['monitoring'] == 'enabled'
        assert summary['services']['streaming'] == 'kafka'
    
    def test_load_config_exception_handling(self):
        """Test that exceptions are properly raised during config loading."""
        manager = ConfigManager(config_path="non_existent.yaml")
        
        # Mock a validation error
        with patch.object(manager, 'validate_config', side_effect=ValueError("Validation failed")):
            with pytest.raises(ValueError, match="Validation failed"):
                manager.load_config()
    
    def test_environment_specific_config(self):
        """Test that environment-specific configurations are applied."""
        # Test development environment
        dev_manager = ConfigManager(config_path="non_existent.yaml", environment="development")
        assert dev_manager.environment == "development"
        assert dev_manager.config.environment == "development"
        
        # Test production environment
        prod_manager = ConfigManager(config_path="non_existent.yaml", environment="production")
        assert prod_manager.environment == "production"
        assert prod_manager.config.environment == "production"
    
    def test_config_hash_generation(self):
        """Test that config hash is generated and changes when config changes."""
        manager = ConfigManager(config_path="non_existent.yaml")
        
        initial_hash = manager.config_hash
        assert initial_hash is not None
        
        # Change configuration
        manager.config.storage.backend = "changed_backend"
        manager.load_config()
        
        new_hash = manager.config_hash
        assert new_hash is not None
        assert new_hash != initial_hash
