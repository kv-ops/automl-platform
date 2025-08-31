"""Enhanced configuration management for AutoML platform with Storage, Monitoring, Workers, and BILLING."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from datetime import timedelta
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlanType(Enum):
    """Subscription plans"""
    FREE = "free"
    TRIAL = "trial"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class BillingConfig:
    """Billing and quotas configuration"""
    enabled: bool = True
    plan_type: str = "free"  # free, trial, pro, enterprise
    
    # Quotas by plan
    quotas: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "free": {
            "max_datasets": 3,
            "max_dataset_size_mb": 10,
            "max_models": 5,
            "max_concurrent_jobs": 1,
            "max_predictions_per_day": 100,
            "max_workers": 1,
            "llm_enabled": False,
            "llm_calls_per_month": 0,
            "gpu_enabled": False,
            "api_rate_limit": 10,
            "data_retention_days": 7
        },
        "trial": {
            "max_datasets": 10,
            "max_dataset_size_mb": 100,
            "max_models": 20,
            "max_concurrent_jobs": 2,
            "max_predictions_per_day": 1000,
            "max_workers": 2,
            "llm_enabled": True,
            "llm_calls_per_month": 100,
            "gpu_enabled": False,
            "api_rate_limit": 60,
            "data_retention_days": 14
        },
        "pro": {
            "max_datasets": 100,
            "max_dataset_size_mb": 1000,
            "max_models": 100,
            "max_concurrent_jobs": 5,
            "max_predictions_per_day": 10000,
            "max_workers": 4,
            "llm_enabled": True,
            "llm_calls_per_month": 1000,
            "gpu_enabled": False,
            "api_rate_limit": 100,
            "data_retention_days": 90
        },
        "enterprise": {
            "max_datasets": -1,  # unlimited
            "max_dataset_size_mb": 10000,
            "max_models": -1,
            "max_concurrent_jobs": 20,
            "max_predictions_per_day": -1,
            "max_workers": 10,
            "llm_enabled": True,
            "llm_calls_per_month": -1,
            "gpu_enabled": True,
            "api_rate_limit": 1000,
            "data_retention_days": 365
        }
    })
    
    # Pricing
    pricing: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "free": {"monthly": 0, "yearly": 0},
        "trial": {"monthly": 0, "yearly": 0},
        "pro": {"monthly": 99, "yearly": 990},
        "enterprise": {"monthly": 999, "yearly": 9990}
    })
    
    # Metering
    enable_metering: bool = True
    track_compute_minutes: bool = True
    track_storage_gb: bool = True
    track_api_calls: bool = True
    track_llm_tokens: bool = True
    
    # Billing backend
    stripe_api_key: Optional[str] = os.getenv("STRIPE_API_KEY")
    stripe_webhook_secret: Optional[str] = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    # Trial settings
    trial_duration_days: int = 14
    trial_requires_card: bool = False


@dataclass
class StorageConfig:
    """Storage configuration for model versioning and datasets."""
    backend: str = "local"  # "local", "minio", "s3"
    
    # Local storage
    local_base_path: str = "./ml_storage"
    
    # MinIO/S3 configuration
    endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure: bool = False
    region: str = "us-east-1"
    
    # Bucket names - ISOLATED BY TENANT
    models_bucket: str = "models"
    datasets_bucket: str = "datasets"
    artifacts_bucket: str = "artifacts"
    
    # Feature store
    enable_feature_store: bool = True
    feature_cache_size: int = 100  # Number of feature sets to cache in memory
    
    # Versioning
    auto_versioning: bool = True
    max_versions_per_model: int = 10
    cleanup_old_versions: bool = False
    
    # Multi-tenant
    enable_multi_tenant: bool = True
    default_tenant_id: str = "default"
    isolate_buckets_per_tenant: bool = True  # Create separate buckets per tenant


@dataclass
class MonitoringConfig:
    """Monitoring configuration for drift detection and metrics."""
    enabled: bool = True
    
    # Prometheus metrics
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    
    # Drift detection
    drift_detection_enabled: bool = True
    drift_check_frequency: int = 100  # Check every N predictions
    drift_sensitivity: float = 0.05  # P-value threshold
    psi_threshold: float = 0.25  # Population Stability Index threshold
    
    # Data quality
    quality_check_enabled: bool = True
    min_quality_score: float = 70.0
    max_missing_ratio: float = 0.3
    max_outlier_ratio: float = 0.1
    
    # Performance monitoring
    track_performance: bool = True
    performance_window_days: int = 7
    min_predictions_for_metrics: int = 30
    
    # Alerting
    alerting_enabled: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log", "email", "slack"])
    
    # Alert thresholds
    accuracy_alert_threshold: float = 0.8
    drift_alert_threshold: float = 0.5
    error_rate_threshold: float = 0.05
    latency_threshold: float = 1.0  # seconds
    quality_score_threshold: float = 70.0
    
    # Alert configuration
    slack_webhook_url: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")
    email_smtp_host: Optional[str] = os.getenv("SMTP_HOST")
    email_smtp_port: int = 587
    email_from: Optional[str] = os.getenv("ALERT_EMAIL_FROM")
    email_recipients: List[str] = field(default_factory=list)
    
    # Logging and reporting
    log_predictions: bool = True
    log_features: bool = False  # Can be memory intensive
    create_reports: bool = True
    report_frequency: str = "daily"  # "hourly", "daily", "weekly"
    report_output_dir: str = "./monitoring_reports"
    
    # Integration
    grafana_enabled: bool = False
    grafana_api_url: Optional[str] = os.getenv("GRAFANA_API_URL")
    evidently_enabled: bool = True  # Use Evidently for advanced drift detection


@dataclass
class WorkerConfig:
    """Worker configuration for distributed processing."""
    enabled: bool = True
    backend: str = "celery"  # "celery", "ray", "dask", "local"
    
    # Celery configuration
    broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    
    # Worker pool
    max_workers: int = 4
    max_concurrent_jobs: int = 2  # Per worker
    worker_prefetch_multiplier: int = 1
    task_time_limit: int = 3600  # seconds
    task_soft_time_limit: int = 3300  # seconds
    
    # Queue configuration - WITH GPU SUPPORT
    queues: Dict[str, Dict] = field(default_factory=lambda: {
        "default": {"priority": 0},
        "training": {"priority": 1},
        "prediction": {"priority": 2},
        "gpu": {"priority": 3, "routing_key": "gpu.*", "gpu_required": True},
        "llm": {"priority": 2, "routing_key": "llm.*"}
    })
    
    # Resource management
    enable_gpu_queue: bool = True
    gpus_per_worker: int = 1
    memory_limit_gb: float = 8.0
    cpu_limit: int = 4
    
    # GPU configuration
    gpu_workers: int = 0  # Number of workers with GPU access
    gpu_memory_fraction: float = 0.8  # Fraction of GPU memory to use
    
    # Retry policy
    task_max_retries: int = 3
    task_retry_delay: int = 60  # seconds
    task_retry_backoff: bool = True
    task_retry_jitter: bool = True
    
    # Monitoring
    enable_task_events: bool = True
    enable_worker_monitoring: bool = True
    worker_heartbeat_interval: int = 30  # seconds
    
    # Scaling
    autoscale_enabled: bool = False
    autoscale_min_workers: int = 1
    autoscale_max_workers: int = 10
    autoscale_target_cpu: float = 70.0  # percentage


@dataclass
class LLMConfig:
    """LLM configuration for intelligent features."""
    enabled: bool = True
    
    # Provider settings
    provider: str = "openai"  # "openai", "anthropic", "huggingface", "local"
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Model selection
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # Features
    enable_data_cleaning: bool = True
    enable_feature_suggestions: bool = True
    enable_model_explanations: bool = True
    enable_report_generation: bool = True
    enable_chatbot: bool = True
    
    # RAG configuration
    enable_rag: bool = True
    vector_store: str = "chromadb"  # "chromadb", "faiss", "pinecone"
    embedding_model: str = "text-embedding-ada-002"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Caching
    cache_responses: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Cost management
    max_cost_per_day: float = 100.0
    track_usage: bool = True
    
    # Prompt templates directory
    prompts_dir: str = "./prompts"


@dataclass
class APIConfig:
    """API configuration."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Security
    enable_auth: bool = True
    jwt_secret: str = os.getenv("JWT_SECRET", "change-this-secret-key")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    
    # SSO Support
    enable_sso: bool = False
    sso_provider: str = "keycloak"  # "keycloak", "auth0", "okta"
    sso_client_id: Optional[str] = os.getenv("SSO_CLIENT_ID")
    sso_client_secret: Optional[str] = os.getenv("SSO_CLIENT_SECRET")
    sso_realm_url: Optional[str] = os.getenv("SSO_REALM_URL")
    
    # Rate limiting - PER PLAN
    enable_rate_limit: bool = True
    rate_limit_requests: int = 100  # Default, overridden by plan
    rate_limit_period: int = 60  # seconds
    
    # CORS
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Documentation
    enable_docs: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    
    # WebSocket
    enable_websocket: bool = True
    websocket_ping_interval: int = 30
    
    # File upload
    max_upload_size_mb: int = 100
    allowed_extensions: List[str] = field(default_factory=lambda: [".csv", ".parquet", ".json", ".xlsx"])


@dataclass
class AutoMLConfig:
    """Enhanced AutoML configuration with all components."""
    
    # Component configurations
    storage: StorageConfig = field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    api: APIConfig = field(default_factory=APIConfig)
    billing: BillingConfig = field(default_factory=BillingConfig)  # NEW: Billing configuration
    
    # General settings
    environment: str = "development"  # "development", "staging", "production"
    debug: bool = True
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 1
    
    # Data preprocessing
    max_missing_ratio: float = 0.5
    rare_category_threshold: float = 0.01
    high_cardinality_threshold: int = 20
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    scaling_method: str = "robust"
    
    # Feature engineering
    create_polynomial: bool = False
    polynomial_degree: int = 2
    create_interactions: bool = False
    create_datetime_features: bool = True
    create_lag_features: bool = False
    lag_periods: List[int] = field(default_factory=lambda: [1, 7, 30])
    
    # Advanced feature engineering
    enable_auto_feature_engineering: bool = True
    max_features_generated: int = 50
    feature_selection_method: str = "mutual_info"  # "mutual_info", "shap", "permutation"
    feature_selection_threshold: float = 0.01
    
    # Text processing
    text_max_features: int = 100
    text_ngram_range: tuple = (1, 2)
    text_min_df: int = 2
    
    # Model selection
    task: str = "auto"
    cv_folds: int = 5
    validation_strategy: str = "auto"
    scoring: str = "auto"
    
    # Enhanced algorithms
    algorithms: List[str] = field(default_factory=lambda: ["all"])
    exclude_algorithms: List[str] = field(default_factory=list)
    include_neural_networks: bool = False  # TabNet, FT-Transformer
    include_time_series: bool = False  # Prophet, ARIMA
    
    # Hyperparameter tuning
    hpo_method: str = "optuna"
    hpo_n_iter: int = 50  # Increased for better optimization
    hpo_time_budget: int = 3600
    early_stopping_rounds: int = 50
    warm_start: bool = True  # Resume from previous trials
    
    # Ensemble methods
    ensemble_method: str = "stacking"  # Enhanced to stacking by default
    ensemble_n_layers: int = 2
    ensemble_use_probabilities: bool = True
    calibrate_probabilities: bool = True
    
    # Class imbalance
    handle_imbalance: bool = True
    imbalance_method: str = "auto"  # Auto-select best method
    smote_neighbors: int = 5
    
    # Performance thresholds
    min_accuracy: float = 0.6
    min_auc: float = 0.6
    min_r2: float = 0.0
    
    # Output settings
    output_dir: str = "./automl_output"
    save_pipeline: bool = True
    save_predictions: bool = True
    save_feature_importance: bool = True
    generate_report: bool = True
    report_format: str = "html"  # "html", "pdf", "markdown"
    
    # Export formats
    enable_docker_export: bool = True
    enable_onnx_export: bool = True
    enable_pmml_export: bool = False
    
    # Resource limits
    max_memory_gb: float = 16.0
    max_time_minutes: int = 60
    max_models_to_train: int = 50
    
    # Multi-tenant support
    tenant_id: str = "default"
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "./logs/automl.log"
    log_to_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI")
    
    def get_quota(self, key: str) -> Any:
        """Get quota value for current plan"""
        if not self.billing.enabled:
            return -1  # Unlimited if billing disabled
        
        plan_quotas = self.billing.quotas.get(self.billing.plan_type, {})
        return plan_quotas.get(key, -1)
    
    def check_quota(self, key: str, current_usage: int) -> bool:
        """Check if quota is exceeded"""
        quota = self.get_quota(key)
        if quota == -1:  # Unlimited
            return True
        return current_usage < quota
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "AutoMLConfig":
        """Load configuration from YAML file with nested configs."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict:
            # Handle nested configurations
            if 'storage' in config_dict and isinstance(config_dict['storage'], dict):
                config_dict['storage'] = StorageConfig(**config_dict['storage'])
            
            if 'monitoring' in config_dict and isinstance(config_dict['monitoring'], dict):
                config_dict['monitoring'] = MonitoringConfig(**config_dict['monitoring'])
            
            if 'worker' in config_dict and isinstance(config_dict['worker'], dict):
                config_dict['worker'] = WorkerConfig(**config_dict['worker'])
            
            if 'llm' in config_dict and isinstance(config_dict['llm'], dict):
                config_dict['llm'] = LLMConfig(**config_dict['llm'])
            
            if 'api' in config_dict and isinstance(config_dict['api'], dict):
                config_dict['api'] = APIConfig(**config_dict['api'])
            
            if 'billing' in config_dict and isinstance(config_dict['billing'], dict):
                config_dict['billing'] = BillingConfig(**config_dict['billing'])
            
            # Handle legacy configs
            if 'algorithms' in config_dict and not isinstance(config_dict['algorithms'], list):
                config_dict['algorithms'] = [config_dict['algorithms']]
        
        return cls(**config_dict) if config_dict else cls()
    
    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        config_dict = asdict(self)
        
        # Convert dataclasses to dicts for YAML serialization
        for key in ['storage', 'monitoring', 'worker', 'llm', 'api', 'billing']:
            if key in config_dict and hasattr(config_dict[key], '__dict__'):
                config_dict[key] = asdict(config_dict[key])
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate all configuration parameters."""
        try:
            # Validate main config
            assert 0 <= self.max_missing_ratio <= 1, "max_missing_ratio must be between 0 and 1"
            assert self.cv_folds > 1, "cv_folds must be greater than 1"
            assert self.hpo_n_iter > 0, "hpo_n_iter must be positive"
            
            # Validate storage config
            assert self.storage.backend in ["local", "minio", "s3"], f"Invalid storage backend: {self.storage.backend}"
            
            # Validate monitoring config
            if self.monitoring.enabled:
                assert 0 <= self.monitoring.drift_sensitivity <= 1, "drift_sensitivity must be between 0 and 1"
                assert self.monitoring.min_quality_score >= 0, "min_quality_score must be non-negative"
            
            # Validate worker config
            if self.worker.enabled:
                assert self.worker.max_workers > 0, "max_workers must be positive"
                assert self.worker.max_concurrent_jobs > 0, "max_concurrent_jobs must be positive"
            
            # Validate LLM config
            if self.llm.enabled:
                assert self.llm.provider in ["openai", "anthropic", "huggingface", "local"], f"Invalid LLM provider: {self.llm.provider}"
                if self.llm.provider in ["openai", "anthropic"] and not self.llm.api_key:
                    logger.warning(f"No API key provided for {self.llm.provider}")
            
            # Validate billing config
            if self.billing.enabled:
                assert self.billing.plan_type in ["free", "trial", "pro", "enterprise"], f"Invalid plan type: {self.billing.plan_type}"
            
            # Create necessary directories
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            if self.storage.backend == "local":
                Path(self.storage.local_base_path).mkdir(parents=True, exist_ok=True)
            if self.monitoring.create_reports:
                Path(self.monitoring.report_output_dir).mkdir(parents=True, exist_ok=True)
            if self.log_file:
                Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            
            return True
            
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            raise
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        env_configs = {
            "development": {
                "debug": True,
                "verbose": 2,
                "hpo_n_iter": 10,
                "max_models_to_train": 10,
                "worker.max_workers": 2,
                "billing.plan_type": "trial"
            },
            "staging": {
                "debug": False,
                "verbose": 1,
                "hpo_n_iter": 30,
                "max_models_to_train": 30,
                "worker.max_workers": 4,
                "billing.plan_type": "pro"
            },
            "production": {
                "debug": False,
                "verbose": 0,
                "hpo_n_iter": 50,
                "max_models_to_train": 50,
                "worker.max_workers": 8,
                "monitoring.enabled": True,
                "monitoring.alerting_enabled": True,
                "billing.enable_metering": True
            }
        }
        
        return env_configs.get(self.environment, {})
    
    def apply_environment_config(self) -> None:
        """Apply environment-specific settings."""
        env_config = self.get_environment_config()
        
        for key, value in env_config.items():
            if '.' in key:
                # Handle nested configs
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self, key, value)


# Convenience functions
def load_config(filepath: str = None, environment: str = None) -> AutoMLConfig:
    """Load configuration with environment overrides."""
    if filepath and Path(filepath).exists():
        config = AutoMLConfig.from_yaml(filepath)
    else:
        config = AutoMLConfig()
    
    # Override environment if specified
    if environment:
        config.environment = environment
        config.apply_environment_config()
    
    # Validate configuration
    config.validate()
    
    return config


def save_config(config: AutoMLConfig, filepath: str) -> None:
    """Save configuration to file."""
    config.to_yaml(filepath)
    logger.info(f"Configuration saved to {filepath}")
