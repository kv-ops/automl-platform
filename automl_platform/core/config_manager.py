"""
Configuration Manager for AutoML Platform
=========================================
Central configuration management with validation and environment support.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import hashlib

from ..config import (
    AutoMLConfig,
    StorageConfig,
    MonitoringConfig,
    WorkerConfig,
    LLMConfig,
    APIConfig,
    BillingConfig,
    RGPDConfig,
    ConnectorConfig,
    StreamingConfig,
    FeatureStoreConfig,
    PlanType
)

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Central configuration manager with validation, versioning, and hot-reload.
    """
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
            environment: Environment (development, staging, production)
        """
        self.config_path = config_path or os.getenv("CONFIG_PATH", "config.yaml")
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        
        # Store configurations
        self.config: Optional[AutoMLConfig] = None
        self.config_history: List[Dict[str, Any]] = []
        self.config_hash: Optional[str] = None
        
        # Load initial configuration
        self.load_config()
        
        logger.info(f"ConfigManager initialized for {self.environment} environment")
    
    def load_config(self, config_path: Optional[str] = None) -> AutoMLConfig:
        """
        Load configuration from file with environment overrides.
        
        Args:
            config_path: Optional path to config file
        
        Returns:
            Loaded configuration
        """
        path = config_path or self.config_path
        
        try:
            if Path(path).exists():
                self.config = AutoMLConfig.from_yaml(path)
            else:
                logger.warning(f"Config file {path} not found, using defaults")
                self.config = AutoMLConfig()
            
            # Apply environment overrides
            self.config.environment = self.environment
            self.config.apply_environment_config()
            
            # Apply environment variables
            self._apply_env_overrides()
            
            # Validate configuration
            self.validate_config()
            
            # Calculate config hash
            config_dict = self.config.to_dict()
            config_str = json.dumps(config_dict, sort_keys=True)
            new_hash = hashlib.sha256(config_str.encode()).hexdigest()
            
            # Store in history if changed
            if new_hash != self.config_hash:
                self.config_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "hash": new_hash,
                    "environment": self.environment,
                    "config": config_dict
                })
                self.config_hash = new_hash
                logger.info(f"Configuration loaded and validated (hash: {new_hash[:8]})")
            
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        
        # Storage overrides
        if os.getenv("STORAGE_BACKEND"):
            self.config.storage.backend = os.getenv("STORAGE_BACKEND")
        if os.getenv("MINIO_ENDPOINT"):
            self.config.storage.endpoint = os.getenv("MINIO_ENDPOINT")
        if os.getenv("MINIO_ACCESS_KEY"):
            self.config.storage.access_key = os.getenv("MINIO_ACCESS_KEY")
        if os.getenv("MINIO_SECRET_KEY"):
            self.config.storage.secret_key = os.getenv("MINIO_SECRET_KEY")
        
        # Worker overrides
        if os.getenv("CELERY_BROKER_URL"):
            self.config.worker.broker_url = os.getenv("CELERY_BROKER_URL")
        if os.getenv("CELERY_RESULT_BACKEND"):
            self.config.worker.result_backend = os.getenv("CELERY_RESULT_BACKEND")
        if os.getenv("MAX_WORKERS"):
            self.config.worker.max_workers = int(os.getenv("MAX_WORKERS"))
        
        # API overrides
        if os.getenv("API_HOST"):
            self.config.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            self.config.api.port = int(os.getenv("API_PORT"))
        if os.getenv("JWT_SECRET"):
            self.config.api.jwt_secret = os.getenv("JWT_SECRET")
        
        # LLM overrides
        if os.getenv("OPENAI_API_KEY"):
            self.config.llm.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("LLM_MODEL"):
            self.config.llm.model_name = os.getenv("LLM_MODEL")
        
        # Billing overrides
        if os.getenv("STRIPE_API_KEY"):
            self.config.billing.stripe_api_key = os.getenv("STRIPE_API_KEY")
        if os.getenv("BILLING_PLAN"):
            self.config.billing.plan_type = os.getenv("BILLING_PLAN")
    
    def validate_config(self) -> bool:
        """
        Validate all configuration parameters.
        
        Returns:
            True if valid, raises exception otherwise
        """
        try:
            # Use built-in validation
            self.config.validate()
            
            # Additional cross-component validation
            self._validate_cross_component()
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _validate_cross_component(self):
        """Validate cross-component configuration consistency."""
        
        # Check worker and billing consistency
        if self.config.billing.enabled:
            plan_quotas = self.config.billing.quotas.get(self.config.billing.plan_type, {})
            max_workers = plan_quotas.get("max_workers", 1)
            
            if self.config.worker.max_workers > max_workers:
                logger.warning(
                    f"Worker count ({self.config.worker.max_workers}) exceeds "
                    f"plan limit ({max_workers}), adjusting..."
                )
                self.config.worker.max_workers = max_workers
        
        # Check GPU configuration
        if self.config.worker.gpu_workers > 0:
            plan_quotas = self.config.billing.quotas.get(self.config.billing.plan_type, {})
            if not plan_quotas.get("gpu_enabled", False):
                logger.warning("GPU workers configured but not enabled in billing plan")
                self.config.worker.gpu_workers = 0
        
        # Check storage and monitoring consistency
        if self.config.monitoring.enabled and self.config.storage.backend == "local":
            # Ensure monitoring reports directory exists
            Path(self.config.monitoring.report_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check streaming dependencies
        if self.config.streaming.enabled:
            if self.config.streaming.platform == "kafka" and not self.config.streaming.brokers:
                raise ValueError("Kafka brokers not configured for streaming")
        
        # Check RGPD and billing consistency
        if self.config.rgpd.enabled:
            # Ensure audit logs are enabled if RGPD requires them
            if self.config.rgpd.audit_all_data_access and not self.config.monitoring.enabled:
                logger.warning("RGPD audit requires monitoring, enabling...")
                self.config.monitoring.enabled = True
    
    def get_service_config(self, service: str) -> Optional[Union[Dict, Any]]:
        """
        Get configuration for a specific service.
        
        Args:
            service: Service name (storage, worker, billing, etc.)
        
        Returns:
            Service configuration or None
        """
        if not self.config:
            return None
        
        service_configs = {
            "storage": self.config.storage,
            "monitoring": self.config.monitoring,
            "worker": self.config.worker,
            "llm": self.config.llm,
            "api": self.config.api,
            "billing": self.config.billing,
            "rgpd": self.config.rgpd,
            "connectors": self.config.connectors,
            "streaming": self.config.streaming,
            "feature_store": self.config.feature_store
        }
        
        return service_configs.get(service)
    
    def update_service_config(self, service: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration for a specific service.
        
        Args:
            service: Service name
            updates: Configuration updates
        
        Returns:
            True if update successful
        """
        try:
            service_config = self.get_service_config(service)
            
            if not service_config:
                logger.error(f"Unknown service: {service}")
                return False
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(service_config, key):
                    setattr(service_config, key, value)
                else:
                    logger.warning(f"Unknown config key for {service}: {key}")
            
            # Revalidate
            self.validate_config()
            
            # Update hash
            config_str = json.dumps(self.config.to_dict(), sort_keys=True)
            self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()
            
            logger.info(f"Updated {service} configuration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update {service} config: {e}")
            return False
    
    def save_config(self, path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            path: Optional path to save config
        
        Returns:
            True if save successful
        """
        try:
            save_path = path or self.config_path
            self.config.to_yaml(save_path)
            logger.info(f"Configuration saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def reload_config(self) -> bool:
        """
        Reload configuration from file.
        
        Returns:
            True if reload successful
        """
        try:
            old_hash = self.config_hash
            self.load_config()
            
            if self.config_hash != old_hash:
                logger.info("Configuration reloaded with changes")
                return True
            else:
                logger.info("Configuration reloaded (no changes)")
                return False
                
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def get_quota(self, key: str) -> Any:
        """
        Get quota value for current plan.
        
        Args:
            key: Quota key
        
        Returns:
            Quota value
        """
        return self.config.get_quota(key)
    
    def check_quota(self, key: str, current_usage: int) -> bool:
        """
        Check if quota is exceeded.
        
        Args:
            key: Quota key
            current_usage: Current usage value
        
        Returns:
            True if within quota
        """
        return self.config.check_quota(key, current_usage)
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """
        Get feature flags based on current configuration.
        
        Returns:
            Dictionary of feature flags
        """
        plan_quotas = self.config.billing.quotas.get(self.config.billing.plan_type, {})
        
        return {
            "distributed_training": self.config.worker.enabled and self.config.worker.max_workers > 1,
            "gpu_enabled": plan_quotas.get("gpu_enabled", False),
            "llm_enabled": self.config.llm.enabled and plan_quotas.get("llm_enabled", False),
            "streaming_enabled": self.config.streaming.enabled,
            "feature_store_enabled": self.config.feature_store.enabled,
            "monitoring_enabled": self.config.monitoring.enabled,
            "rgpd_enabled": self.config.rgpd.enabled,
            "billing_enabled": self.config.billing.enabled,
            "sso_enabled": self.config.api.enable_sso,
            "rate_limiting_enabled": self.config.api.enable_rate_limit,
            "docker_export": self.config.enable_docker_export,
            "onnx_export": self.config.enable_onnx_export,
            "pmml_export": self.config.enable_pmml_export
        }
    
    def get_limits(self) -> Dict[str, Any]:
        """
        Get all limits for current plan.
        
        Returns:
            Dictionary of limits
        """
        plan_quotas = self.config.billing.quotas.get(self.config.billing.plan_type, {})
        
        return {
            "max_datasets": plan_quotas.get("max_datasets", 0),
            "max_dataset_size_mb": plan_quotas.get("max_dataset_size_mb", 0),
            "max_models": plan_quotas.get("max_models", 0),
            "max_concurrent_jobs": plan_quotas.get("max_concurrent_jobs", 1),
            "max_predictions_per_day": plan_quotas.get("max_predictions_per_day", 0),
            "max_workers": plan_quotas.get("max_workers", 1),
            "max_api_calls_per_day": plan_quotas.get("max_api_calls_per_day", 100),
            "api_rate_limit": plan_quotas.get("api_rate_limit", 10),
            "max_storage_gb": self.config.storage.max_versions_per_model * 0.1,  # Estimate
            "max_gpu_hours_per_month": plan_quotas.get("max_gpu_hours_per_month", 0),
            "data_retention_days": plan_quotas.get("data_retention_days", 7)
        }
    
    def export_config(self, format: str = "yaml", include_secrets: bool = False) -> str:
        """
        Export configuration in specified format.
        
        Args:
            format: Export format (yaml, json, env)
            include_secrets: Include sensitive values
        
        Returns:
            Exported configuration string
        """
        config_dict = self.config.to_dict()
        
        # Remove secrets if requested
        if not include_secrets:
            config_dict = self._remove_secrets(config_dict)
        
        if format == "yaml":
            return yaml.dump(config_dict, default_flow_style=False)
        elif format == "json":
            return json.dumps(config_dict, indent=2)
        elif format == "env":
            return self._to_env_format(config_dict)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _remove_secrets(self, config_dict: Dict) -> Dict:
        """Remove sensitive values from configuration."""
        secrets_keys = [
            "api_key", "secret_key", "password", "token",
            "jwt_secret", "stripe_api_key", "stripe_webhook_secret",
            "access_key", "secret", "credentials"
        ]
        
        def remove_recursive(d):
            if isinstance(d, dict):
                return {
                    k: "***REDACTED***" if any(s in k.lower() for s in secrets_keys) else remove_recursive(v)
                    for k, v in d.items()
                }
            elif isinstance(d, list):
                return [remove_recursive(item) for item in d]
            else:
                return d
        
        return remove_recursive(config_dict)
    
    def _to_env_format(self, config_dict: Dict, prefix: str = "AUTOML") -> str:
        """Convert configuration to environment variable format."""
        env_vars = []
        
        def flatten(d, parent_key=""):
            for k, v in d.items():
                new_key = f"{parent_key}_{k}".upper() if parent_key else k.upper()
                
                if isinstance(v, dict):
                    flatten(v, new_key)
                elif isinstance(v, list):
                    env_vars.append(f"{prefix}_{new_key}={json.dumps(v)}")
                elif isinstance(v, bool):
                    env_vars.append(f"{prefix}_{new_key}={'true' if v else 'false'}")
                else:
                    env_vars.append(f"{prefix}_{new_key}={v}")
        
        flatten(config_dict)
        return "\n".join(sorted(env_vars))
    
    def get_config_diff(self, other_config: AutoMLConfig) -> Dict[str, Any]:
        """
        Get differences between current and another configuration.
        
        Args:
            other_config: Configuration to compare with
        
        Returns:
            Dictionary of differences
        """
        current_dict = self.config.to_dict()
        other_dict = other_config.to_dict()
        
        def diff_recursive(d1, d2, path=""):
            diffs = {}
            
            all_keys = set(d1.keys()) | set(d2.keys())
            
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    diffs[new_path] = {"added": d2[key]}
                elif key not in d2:
                    diffs[new_path] = {"removed": d1[key]}
                elif d1[key] != d2[key]:
                    if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                        nested_diffs = diff_recursive(d1[key], d2[key], new_path)
                        if nested_diffs:
                            diffs.update(nested_diffs)
                    else:
                        diffs[new_path] = {"old": d1[key], "new": d2[key]}
            
            return diffs
        
        return diff_recursive(current_dict, other_dict)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary.
        
        Returns:
            Summary of current configuration
        """
        return {
            "environment": self.environment,
            "plan": self.config.billing.plan_type,
            "hash": self.config_hash[:8] if self.config_hash else None,
            "features": self.get_feature_flags(),
            "limits": self.get_limits(),
            "services": {
                "storage": self.config.storage.backend,
                "worker": f"{self.config.worker.backend} ({self.config.worker.max_workers} workers)",
                "monitoring": "enabled" if self.config.monitoring.enabled else "disabled",
                "billing": "enabled" if self.config.billing.enabled else "disabled",
                "streaming": self.config.streaming.platform if self.config.streaming.enabled else "disabled",
                "llm": self.config.llm.provider if self.config.llm.enabled else "disabled"
            },
            "last_updated": self.config_history[-1]["timestamp"] if self.config_history else None
        }
