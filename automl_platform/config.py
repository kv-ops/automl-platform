"""Enhanced configuration management for AutoML platform with Agent-First support."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from datetime import timedelta
from enum import Enum

try:
    from automl_platform.api.billing import PlanType as BillingPlanType
except ImportError:  # pragma: no cover - fallback when billing module unavailable
    BillingPlanType = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _coerce_bool(value: Any) -> Optional[bool]:
    """Normalize common truthy/falsey representations to booleans."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


class PlanType(Enum):
    """Subscription plans"""
    FREE = "free"
    TRIAL = "trial"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    CUSTOM = "custom"


@dataclass
class AgentFirstConfig:
    """Configuration for Agent-First approach with intelligent agents"""
    enabled: bool = True
    
    # Universal ML Agent settings
    use_universal_agent: bool = True
    auto_detect_problem_type: bool = True
    search_best_practices: bool = True
    enable_continuous_learning: bool = True
    
    # Context detection
    context_detection_confidence_threshold: float = 0.7
    alternative_contexts_to_consider: int = 3
    detect_business_sector: bool = True
    detect_temporal_aspects: bool = True
    
    # Configuration generation
    generate_config_dynamically: bool = True
    override_templates: bool = True
    template_as_hint_only: bool = True
    adapt_to_data_characteristics: bool = True
    
    # Adaptive learning
    enable_adaptive_templates: bool = True
    max_learned_patterns: int = 100
    min_success_score_to_learn: float = 0.8
    learn_from_each_execution: bool = True
    
    # Knowledge base
    knowledge_base_path: str = "./knowledge_base"
    cache_contexts: bool = True
    cache_best_practices: bool = True
    cache_successful_patterns: bool = True
    knowledge_retention_days: int = 365
    
    # Web search for ML best practices
    enable_web_search: bool = True
    search_providers: List[str] = field(default_factory=lambda: ["papers_with_code", "arxiv", "google_scholar"])
    max_search_results: int = 10
    cache_search_results: bool = True
    search_cache_ttl: int = 86400  # 24 hours
    
    # Performance tracking
    track_agent_metrics: bool = True
    track_execution_history: bool = True
    max_execution_history: int = 1000
    generate_performance_reports: bool = True
    
    # Agent orchestration
    agent_timeout_seconds: int = 300
    max_agent_retries: int = 3
    parallel_agent_execution: bool = True
    use_intelligent_cleaning: bool = True
    
    # OpenAI configuration for agents
    openai_model: str = "gpt-4-1106-preview"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 1000
    track_openai_costs: bool = True
    max_openai_cost_per_run: float = 10.0
    
    # Agent-specific settings
    enable_profiler_agent: bool = True
    enable_validator_agent: bool = True
    enable_cleaner_agent: bool = True
    enable_controller_agent: bool = True

    # Hybrid cleaning configuration (mirrors AgentConfig defaults)
    enable_hybrid_mode: bool = True
    hybrid_mode_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "missing_threshold": 0.35,
        "outlier_threshold": 0.10,
        "quality_score_threshold": 70.0,
        "complexity_threshold": 0.8,
        "cost_threshold": 1.0,
    })
    retail_rules: Dict[str, Any] = field(default_factory=lambda: {
        "sentinel_values": [-999, -1, 9999],
        "stock_zero_acceptable": True,
        "price_negative_critical": True,
        "sku_format_strict": True,
        "gs1_compliance_required": True,
        "gs1_compliance_target": 0.98,
        "category_imputation": "by_category",
        "price_imputation": "median_by_category",
    })
    hybrid_cost_limits: Dict[str, float] = field(default_factory=lambda: {
        "max_openai": 3.0,
        "max_claude": 2.0,
        "max_total": 5.0,
        "max_per_decision": 0.10,
    })
    sector_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "retail": ["SKU", "UPC", "GS1", "inventory", "merchandising"],
        "finance": ["IFRS", "Basel", "risk management"],
        "sante": ["HL7", "ICD-10", "patient data"],
        "industrie": ["ISO", "manufacturing", "supply chain"],
    })

    # YAML configuration export
    export_yaml_configs: bool = True
    yaml_output_dir: str = "./agent_outputs"
    include_reasoning_in_yaml: bool = True


@dataclass
class BillingConfig:
    """Billing and quotas configuration"""
    enabled: bool = True
    plan_type: str = "free"  # free, trial, pro, enterprise
    
    # Quotas by plan (updated with Agent-First limits)
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
            "agent_first_enabled": False,  # New
            "agent_calls_per_month": 0,    # New
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
            "agent_first_enabled": True,    # New
            "agent_calls_per_month": 50,    # New
            "gpu_enabled": False,
            "api_rate_limit": 60,
            "data_retention_days": 14
        },
        "starter": {
            "max_datasets": 25,
            "max_dataset_size_mb": 250,
            "max_models": 10,
            "max_concurrent_jobs": 3,
            "max_predictions_per_day": 5000,
            "max_workers": 3,
            "llm_enabled": True,
            "llm_calls_per_month": 250,
            "agent_first_enabled": True,
            "agent_calls_per_month": 150,
            "gpu_enabled": False,
            "api_rate_limit": 250,
            "data_retention_days": 30
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
            "agent_first_enabled": True,    # New
            "agent_calls_per_month": 500,   # New
            "gpu_enabled": False,
            "api_rate_limit": 100,
            "data_retention_days": 90
        },
        "professional": {
            "max_datasets": 250,
            "max_dataset_size_mb": 2500,
            "max_models": 50,
            "max_concurrent_jobs": 10,
            "max_predictions_per_day": 50000,
            "max_workers": 8,
            "llm_enabled": True,
            "llm_calls_per_month": 5000,
            "agent_first_enabled": True,
            "agent_calls_per_month": 2500,
            "gpu_enabled": True,
            "api_rate_limit": 500,
            "data_retention_days": 180
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
            "agent_first_enabled": True,    # New
            "agent_calls_per_month": -1,    # New
            "gpu_enabled": True,
            "api_rate_limit": 1000,
            "data_retention_days": 365
        }
    })
    
    # Pricing
    pricing: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "free": {"monthly": 0, "yearly": 0},
        "trial": {"monthly": 0, "yearly": 0},
        "starter": {"monthly": 49, "yearly": 490},
        "pro": {"monthly": 99, "yearly": 990},
        "professional": {"monthly": 299, "yearly": 2990},
        "enterprise": {"monthly": 999, "yearly": 9990}
    })
    
    # Metering
    enable_metering: bool = True
    track_compute_minutes: bool = True
    track_storage_gb: bool = True
    track_api_calls: bool = True
    track_llm_tokens: bool = True
    track_agent_calls: bool = True  # New
    
    # Billing backend
    stripe_api_key: Optional[str] = os.getenv("STRIPE_API_KEY")
    stripe_webhook_secret: Optional[str] = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    # Trial settings
    trial_duration_days: int = 14
    trial_requires_card: bool = False


@dataclass
class RGPDConfig:
    """RGPD/GDPR compliance configuration"""
    enabled: bool = True
    
    # Data retention policies (in days)
    retention_periods: Dict[str, int] = field(default_factory=lambda: {
        "basic_data": 365 * 3,  # 3 years
        "contact_data": 365 * 3,
        "financial_data": 365 * 7,  # 7 years for financial records
        "behavioral_data": 365,  # 1 year
        "technical_data": 90,  # 3 months
        "sensitive_data": 365,  # 1 year with explicit consent
        "ml_predictions": 180,  # 6 months
        "audit_logs": 365 * 2,  # 2 years
        "consent_records": 365 * 5,  # 5 years
        "agent_interactions": 365,  # New - 1 year for agent interactions
        "knowledge_base": 365 * 2  # New - 2 years for learned patterns
    })
    
    # Consent management
    consent_required_for: List[str] = field(default_factory=lambda: [
        "marketing",
        "analytics",
        "profiling",
        "automated_decisions",
        "third_party_sharing",
        "data_processing",
        "agent_learning"  # New - consent for agent learning from data
    ])
    
    consent_renewal_days: int = 365  # Renew consent annually
    consent_reminder_days: int = 30  # Remind 30 days before expiry
    
    # Data subject rights
    enable_data_access: bool = True  # Article 15
    enable_rectification: bool = True  # Article 16
    enable_erasure: bool = True  # Article 17
    enable_restriction: bool = True  # Article 18
    enable_portability: bool = True  # Article 20
    enable_objection: bool = True  # Article 21
    
    # Processing
    request_processing_days: int = 30  # Legal requirement
    request_extension_days: int = 60  # Maximum extension
    identity_verification_required: bool = True
    
    # Data protection
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256"
    pseudonymization_enabled: bool = True
    anonymization_threshold_days: int = 365 * 2  # Anonymize after 2 years
    
    # Data breach
    breach_notification_hours: int = 72  # Notify authorities within 72 hours
    breach_user_notification: bool = True
    
    # Third-party data sharing
    third_party_processors: List[Dict[str, str]] = field(default_factory=lambda: [
        {"name": "AWS", "purpose": "infrastructure", "location": "EU"},
        {"name": "Stripe", "purpose": "payments", "location": "EU"},
        {"name": "SendGrid", "purpose": "email", "location": "EU"},
        {"name": "OpenAI", "purpose": "intelligent_agents", "location": "US"}  # New
    ])
    
    # Data mapping
    data_categories: List[str] = field(default_factory=lambda: [
        "identification",
        "contact",
        "financial",
        "behavioral",
        "technical",
        "preferences",
        "ml_data",
        "agent_interactions"  # New
    ])
    
    # Legal basis
    legal_basis_for_processing: Dict[str, str] = field(default_factory=lambda: {
        "registration": "contract",
        "ml_training": "legitimate_interest",
        "marketing": "consent",
        "analytics": "legitimate_interest",
        "security": "legal_obligation",
        "payments": "contract",
        "agent_learning": "consent"  # New
    })
    
    # Privacy by design
    privacy_by_default: bool = True
    data_minimization: bool = True
    purpose_limitation: bool = True
    
    # Compliance reporting
    enable_compliance_reports: bool = True
    compliance_report_frequency: str = "monthly"  # "daily", "weekly", "monthly", "quarterly"
    
    # DPO (Data Protection Officer) contact
    dpo_email: Optional[str] = os.getenv("DPO_EMAIL", "dpo@automl-platform.com")
    dpo_name: Optional[str] = "Data Protection Officer"
    
    # Geographic restrictions
    restricted_countries: List[str] = field(default_factory=list)  # Countries where service is not available
    eu_users_only: bool = False  # Restrict to EU users only
    
    # Audit trail
    audit_all_data_access: bool = True
    audit_retention_days: int = 365 * 2
    
    # Cookie policy
    cookie_consent_required: bool = True
    essential_cookies_only: bool = False
    cookie_retention_days: Dict[str, int] = field(default_factory=lambda: {
        "session": 0,  # Session cookies
        "persistent": 30,  # 30 days
        "analytics": 365,  # 1 year
        "marketing": 90  # 3 months
    })


@dataclass
class ConnectorConfig:
    """Configuration for data connectors."""
    enabled: bool = True
    default_connector: str = "postgresql"
    
    connectors: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "snowflake": {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "username": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": "public"
        },
        "bigquery": {
            "project_id": os.getenv("GCP_PROJECT_ID"),
            "dataset_id": os.getenv("BQ_DATASET_ID"),
            "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        },
        "databricks": {
            "host": os.getenv("DATABRICKS_HOST"),
            "token": os.getenv("DATABRICKS_TOKEN"),
            "catalog": "hive_metastore",
            "schema": "default"
        },
        "postgresql": {
            "host": os.getenv("PG_HOST", "localhost"),
            "port": int(os.getenv("PG_PORT", "5432")),
            "database": os.getenv("PG_DATABASE", "automl"),
            "username": os.getenv("PG_USER", "postgres"),
            "password": os.getenv("PG_PASSWORD")
        },
        "mongodb": {
            "host": os.getenv("MONGO_HOST", "localhost"),
            "port": int(os.getenv("MONGO_PORT", "27017")),
            "database": os.getenv("MONGO_DATABASE", "automl"),
            "username": os.getenv("MONGO_USER"),
            "password": os.getenv("MONGO_PASSWORD")
        },
        "redshift": {
            "host": os.getenv("REDSHIFT_HOST"),
            "port": int(os.getenv("REDSHIFT_PORT", "5439")),
            "database": os.getenv("REDSHIFT_DATABASE"),
            "username": os.getenv("REDSHIFT_USER"),
            "password": os.getenv("REDSHIFT_PASSWORD")
        }
    })
    
    # Connection pooling
    enable_pooling: bool = True
    pool_size: int = 5
    max_overflow: int = 10
    
    # Query optimization
    enable_query_cache: bool = True
    cache_ttl: int = 3600
    max_cache_size_mb: int = 100
    
    # Security
    encrypt_credentials: bool = True
    use_ssl: bool = True
    
    # Performance
    batch_size: int = 10000
    timeout: int = 300
    max_retries: int = 3


@dataclass 
class StreamingConfig:
    """Configuration for streaming pipelines."""
    enabled: bool = True
    platform: str = "kafka"  # kafka, pulsar, redis, flink
    
    # Broker configuration
    brokers: List[str] = field(default_factory=lambda: [
        os.getenv("KAFKA_BROKER", "localhost:9092")
    ])
    
    # Topics
    input_topic: str = "ml_input"
    output_topic: str = "ml_predictions"
    error_topic: str = "ml_errors"
    
    # Consumer settings
    consumer_group: str = "automl_consumer"
    batch_size: int = 100
    max_poll_records: int = 500
    
    # Window aggregation
    window_size: int = 60  # seconds
    slide_interval: int = 10  # seconds
    
    # Processing
    enable_exactly_once: bool = False
    checkpoint_interval: int = 30  # seconds
    max_latency_ms: int = 1000
    
    # Authentication
    security_protocol: str = "PLAINTEXT"  # PLAINTEXT, SSL, SASL_SSL
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = os.getenv("KAFKA_USER")
    sasl_password: Optional[str] = os.getenv("KAFKA_PASSWORD")
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9092


@dataclass
class FeatureStoreConfig:
    """Configuration for feature store."""
    enabled: bool = True
    backend: str = "local"  # local, s3, minio, redis
    
    # Storage
    base_path: str = "./feature_store"
    
    # Object storage (S3/MinIO)
    endpoint: Optional[str] = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key: Optional[str] = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key: Optional[str] = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    bucket: str = "feature-store"
    secure: bool = False
    
    # Redis (for online serving)
    redis_enabled: bool = True
    redis_config: Dict[str, Any] = field(default_factory=lambda: {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "db": 0
    })
    
    # Caching
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_cache_size_mb: int = 500
    
    # Versioning
    enable_versioning: bool = True
    max_versions: int = 10
    
    # Statistics
    compute_statistics: bool = True
    statistics_sample_size: int = 10000
    
    # Materialization
    enable_materialization: bool = True
    materialization_interval: int = 3600  # seconds
    
    # Multi-tenancy
    enable_multi_tenant: bool = True
    tenant_isolation: str = "logical"  # logical, physical


@dataclass
class StorageConfig:
    """Storage configuration for model versioning and datasets."""
    backend: str = "local"  # "local", "minio", "s3", "gcs"
    
    # Local storage
    local_base_path: str = "./ml_storage"
    
    # MinIO/S3 configuration
    endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure: bool = False
    region: str = "us-east-1"

    # GCS configuration
    project_id: Optional[str] = os.getenv("GCP_PROJECT_ID")
    credentials_path: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Bucket names - ISOLATED BY TENANT
    models_bucket: str = "models"
    datasets_bucket: str = "datasets"
    artifacts_bucket: str = "artifacts"
    knowledge_bucket: str = "knowledge"  # New - for agent knowledge base
    
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
    
    # Agent monitoring (New)
    track_agent_performance: bool = True
    track_agent_costs: bool = True
    track_context_accuracy: bool = True
    track_config_effectiveness: bool = True
    
    # Alerting
    alerting_enabled: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["log", "email", "slack"])
    
    # Alert thresholds
    accuracy_alert_threshold: float = 0.8
    drift_alert_threshold: float = 0.5
    error_rate_threshold: float = 0.05
    latency_threshold: float = 1.0  # seconds
    quality_score_threshold: float = 70.0
    agent_cost_threshold: float = 100.0  # New - daily cost limit for agents
    
    # Alert configuration
    slack_webhook_url: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")
    email_smtp_host: Optional[str] = os.getenv("SMTP_HOST")
    email_smtp_port: int = 587
    email_from: Optional[str] = os.getenv("ALERT_EMAIL_FROM")
    email_recipients: List[str] = field(default_factory=list)
    
    # Logging and reporting
    log_predictions: bool = True
    log_features: bool = False  # Can be memory intensive
    log_agent_decisions: bool = True  # New - log agent decisions
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
    
    # Queue configuration - WITH GPU AND AGENT SUPPORT
    queues: Dict[str, Dict] = field(default_factory=lambda: {
        "default": {"priority": 0},
        "training": {"priority": 1},
        "prediction": {"priority": 2},
        "gpu": {"priority": 3, "routing_key": "gpu.*", "gpu_required": True},
        "llm": {"priority": 2, "routing_key": "llm.*"},
        "agents": {"priority": 3, "routing_key": "agents.*"}  # New - for agent tasks
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
    enable_agent_first: bool = True  # New - enable Agent-First approach
    
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
    enable_sso: bool = True  # Changed to True for full SSO support
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
    """Enhanced AutoML configuration with all components including Expert Mode and Agent-First."""
    
    # Component configurations
    storage: StorageConfig = field(default_factory=StorageConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    api: APIConfig = field(default_factory=APIConfig)
    billing: BillingConfig = field(default_factory=BillingConfig)
    rgpd: RGPDConfig = field(default_factory=RGPDConfig)
    connectors: ConnectorConfig = field(default_factory=ConnectorConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    agent_first: AgentFirstConfig = field(default_factory=AgentFirstConfig)  # NEW
    
    # General settings
    environment: str = "development"  # "development", "staging", "production"
    debug: bool = True
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 1
    
    # EXPERT MODE CONFIGURATION
    expert_mode: bool = field(default=False)
    """
    Expert mode flag that controls visibility of advanced parameters.
    
    When False (default):
    - Hides advanced algorithm selection
    - Uses automatic HPO configuration
    - Simplifies UI/CLI options
    - Applies sensible defaults for all settings
    
    When True:
    - Shows all algorithm choices
    - Allows manual HPO configuration
    - Exposes distributed computing options (Ray workers, GPU settings)
    - Enables fine-tuning of all parameters
    
    Can be overridden by environment variable: AUTOML_EXPERT_MODE=true
    Useful for SaaS deployments where different user tiers get different complexity levels.
    """
    
    # INTELLIGENT CLEANING FLAG
    enable_intelligent_cleaning: bool = field(default=False)
    """
    Flag pour activer le nettoyage intelligent via l'API Python.

    Quand False (défaut):
    - Utilise le preprocessing standard existant
    - Compatible avec les pipelines actuels

    Quand True:
    - Active le nettoyage intelligent avec le context utilisateur
    - Orchestre profiling → schema → outliers → encodage selon le secteur d'activité
    - Intègre la protection contre le leakage des données
    """
    
    # AGENT-FIRST FLAG
    enable_agent_first: bool = field(default=False)
    # Hybrid cleaning flag mirrors Agent-First configuration when not explicitly set
    enable_hybrid_mode: bool = field(default=True)
    """
    Enable Agent-First approach for template-free AutoML.
    
    When True:
    - Automatically detects ML problem type
    - Generates optimal configurations dynamically
    - Searches for ML best practices
    - Learns from each execution
    - No templates required
    """
    
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
    
    def check_rgpd_consent(self, user_id: str, consent_type: str) -> bool:
        """Check if user has given RGPD consent"""
        if not self.rgpd.enabled:
            return True
        
        # This would check actual consent records
        # Simplified for demonstration
        return consent_type not in self.rgpd.consent_required_for
    
    def get_retention_period(self, data_category: str) -> int:
        """Get retention period for data category"""
        return self.rgpd.retention_periods.get(data_category, 365)
    
    def is_erasure_allowed(self, data_category: str) -> bool:
        """Check if data can be erased"""
        if not self.rgpd.enable_erasure:
            return False
        
        # Some data must be retained for legal reasons
        if data_category == "financial_data":
            return False  # Must retain for tax purposes
        
        return True
    
    def get_simplified_algorithms(self, task: Optional[str] = None) -> List[str]:
        """Get simplified algorithm list for non-expert mode.

        Args:
            task: Optional task hint ("classification", "regression",
                "timeseries" or "auto"). When not provided or set to
                "auto", simplified algorithms for both classification and
                regression will be returned so that downstream task detection
                can still succeed.

        Returns:
            A list of algorithm names matching the keys produced by
            :func:`automl_platform.model_selection.get_available_models`.
        """
        if self.expert_mode:
            return self.algorithms

        # Import lazily to avoid circular dependencies at module import time
        from automl_platform.model_selection import get_available_models

        preferred_by_task: Dict[str, List[str]] = {
            "classification": [
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "LogisticRegression",
                "XGBClassifier",
                "LGBMClassifier",
            ],
            "regression": [
                "RandomForestRegressor",
                "GradientBoostingRegressor",
                "LinearRegression",
                "XGBRegressor",
                "LGBMRegressor",
            ],
            "timeseries": [
                "RandomForestRegressor",
                "GradientBoostingRegressor",
                "Ridge",
                "LinearRegression",
            ],
        }

        normalized_task = (task or "auto").lower()
        if normalized_task in ("", "auto"):
            tasks_to_consider = ["classification", "regression"]
        elif normalized_task in preferred_by_task:
            tasks_to_consider = [normalized_task]
        else:
            # Unknown task - default to classification for safety
            tasks_to_consider = ["classification"]

        simplified_algorithms: List[str] = []

        for task_name in tasks_to_consider:
            include_timeseries = task_name == "timeseries"
            available_models = get_available_models(
                task_name,
                include_timeseries=include_timeseries,
            )

            preferred = preferred_by_task.get(task_name, [])
            selected = [name for name in preferred if name in available_models]

            # Ensure we keep at least two algorithms per task when possible
            if len(selected) < 2:
                for model_name in available_models.keys():
                    if model_name not in selected:
                        selected.append(model_name)
                    if len(selected) >= 2:
                        break

            for model_name in selected:
                if model_name not in simplified_algorithms:
                    simplified_algorithms.append(model_name)

        if not simplified_algorithms:
            # As a final fallback, expose whatever models are available for the
            # first considered task to avoid empty selections downstream.
            fallback_task = tasks_to_consider[0]
            fallback_models = get_available_models(fallback_task)
            simplified_algorithms = list(fallback_models.keys())

        return simplified_algorithms
    
    def get_simplified_hpo_config(self) -> Dict[str, Any]:
        """Get simplified HPO configuration for non-expert mode"""
        if self.expert_mode:
            return {
                "method": self.hpo_method,
                "n_iter": self.hpo_n_iter,
                "time_budget": self.hpo_time_budget,
                "early_stopping_rounds": self.early_stopping_rounds
            }
        else:
            # Use optimized defaults for non-experts
            return {
                "method": "optuna",
                "n_iter": 20,  # Reduced iterations for faster results
                "time_budget": 600,  # 10 minutes max
                "early_stopping_rounds": 10
            }
    
    def should_use_agent_first(self) -> bool:
        """Determine if Agent-First approach should be used"""
        # Check multiple conditions
        if not self.enable_agent_first:
            return False
        
        if not self.agent_first.enabled:
            return False
        
        # Check if user has quota for agent calls
        if self.billing.enabled:
            agent_quota = self.get_quota("agent_calls_per_month")
            if agent_quota == 0:
                return False
        
        # Check if OpenAI API key is available
        if not self.llm.api_key and not os.getenv("OPENAI_API_KEY"):
            return False

        return True

    def get_hybrid_mode_flag(self) -> bool:
        """Return the resolved hybrid mode flag preferring nested Agent-First config."""
        nested = None
        if hasattr(self, 'agent_first') and self.agent_first is not None:
            nested = _coerce_bool(getattr(self.agent_first, 'enable_hybrid_mode', None))

        top_level = _coerce_bool(getattr(self, 'enable_hybrid_mode', None))

        if nested is not None:
            return nested
        if top_level is not None:
            return top_level
        return False
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "AutoMLConfig":
        """Load configuration from YAML file with nested configs."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        if config_dict:
            original_enable_agent_first = config_dict.get('enable_agent_first')
            original_enable_hybrid_mode = config_dict.get('enable_hybrid_mode')
            hybrid_flag_defined = 'enable_hybrid_mode' in config_dict
            agent_first_overridden_by_env = False
            # Check for expert mode in environment variable first
            expert_mode_env = os.getenv("AUTOML_EXPERT_MODE", "").lower()
            if expert_mode_env in ["true", "1", "yes", "on"]:
                config_dict['expert_mode'] = True
            elif expert_mode_env in ["false", "0", "no", "off"]:
                config_dict['expert_mode'] = False
            # Otherwise use value from YAML or default

            # Check for Agent-First mode in environment variable
            agent_first_env = os.getenv("AUTOML_AGENT_FIRST", "").lower()
            if agent_first_env in ["true", "1", "yes", "on"]:
                config_dict['enable_agent_first'] = True
                agent_first_overridden_by_env = True
            elif agent_first_env in ["false", "0", "no", "off"]:
                config_dict['enable_agent_first'] = False
                agent_first_overridden_by_env = True

            # Handle nested configurations
            nested_configs = [
                'storage', 'monitoring', 'worker', 'llm', 'api', 'billing',
                'rgpd', 'connectors', 'streaming', 'feature_store', 'agent_first'
            ]

            for config_name in nested_configs:
                if config_name in config_dict and isinstance(config_dict[config_name], dict):
                    config_class = globals().get(f"{config_name.replace('_', ' ').title().replace(' ', '')}Config")
                    if config_class:
                        config_dict[config_name] = config_class(**config_dict[config_name])

            # Propagate Agent-First enablement when nested flag is provided
            agent_first_cfg = config_dict.get('agent_first')
            agent_first_enabled = None
            agent_first_hybrid = None

            if isinstance(agent_first_cfg, dict):
                agent_first_enabled = agent_first_cfg.get('enabled')
                agent_first_hybrid = agent_first_cfg.get('enable_hybrid_mode')
            elif hasattr(agent_first_cfg, 'enabled'):
                agent_first_enabled = getattr(agent_first_cfg, 'enabled')
                agent_first_hybrid = getattr(agent_first_cfg, 'enable_hybrid_mode', None)

            if agent_first_enabled is not None:
                current_enable_agent_first = config_dict.get('enable_agent_first')

                if agent_first_overridden_by_env:
                    if current_enable_agent_first != agent_first_enabled:
                        logger.warning(
                            "AUTOML_AGENT_FIRST environment variable overrides nested agent_first.enabled=%s; "
                            "keeping enable_agent_first=%s",
                            agent_first_enabled,
                            current_enable_agent_first,
                        )
                elif current_enable_agent_first is None and 'enable_agent_first' not in config_dict:
                    config_dict['enable_agent_first'] = agent_first_enabled
                elif original_enable_agent_first is None:
                    config_dict['enable_agent_first'] = agent_first_enabled
                elif original_enable_agent_first != agent_first_enabled:
                    logger.warning(
                        "Conflicting Agent-First flags in YAML: enable_agent_first=%s vs agent_first.enabled=%s. "
                        "Using nested agent_first.enabled.",
                        original_enable_agent_first,
                        agent_first_enabled,
                    )
                    config_dict['enable_agent_first'] = agent_first_enabled

            if agent_first_hybrid is not None:
                normalized_hybrid = _coerce_bool(agent_first_hybrid)
                normalized_top_level = _coerce_bool(original_enable_hybrid_mode)

                if not hybrid_flag_defined or normalized_top_level is None:
                    config_dict['enable_hybrid_mode'] = normalized_hybrid
                elif normalized_top_level != normalized_hybrid:
                    logger.warning(
                        "Conflicting hybrid mode flags in YAML: enable_hybrid_mode=%s vs agent_first.enable_hybrid_mode=%s. "
                        "Using nested agent_first.enable_hybrid_mode.",
                        original_enable_hybrid_mode,
                        agent_first_hybrid,
                    )
                    config_dict['enable_hybrid_mode'] = normalized_hybrid

            # Handle legacy configs
            if 'algorithms' in config_dict and not isinstance(config_dict['algorithms'], list):
                config_dict['algorithms'] = [config_dict['algorithms']]

        return cls(**config_dict) if config_dict else cls()
    
    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        config_dict = asdict(self)
        
        # Convert dataclasses to dicts for YAML serialization
        for key in ['storage', 'monitoring', 'worker', 'llm', 'api', 'billing', 'rgpd',
                   'connectors', 'streaming', 'feature_store', 'agent_first']:
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
            assert self.storage.backend in ["local", "minio", "s3", "gcs"], f"Invalid storage backend: {self.storage.backend}"
            if self.storage.backend == "gcs" and self.storage.credentials_path:
                credentials_file = Path(self.storage.credentials_path).expanduser()
                assert credentials_file.exists(), (
                    f"GCS credentials_path does not exist: {credentials_file}"
                )
            
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
            
            # Validate Agent-First config
            if self.agent_first.enabled:
                assert self.agent_first.context_detection_confidence_threshold >= 0 and self.agent_first.context_detection_confidence_threshold <= 1
                assert self.agent_first.max_learned_patterns > 0
                assert self.agent_first.min_success_score_to_learn >= 0 and self.agent_first.min_success_score_to_learn <= 1
            
            # Validate billing config
            if self.billing.enabled:
                allowed_plan_types = set()

                if getattr(self.billing, "quotas", None):
                    allowed_plan_types.update(
                        (plan_key.value if isinstance(plan_key, Enum) else str(plan_key)).lower()
                        for plan_key in self.billing.quotas.keys()
                    )

                if BillingPlanType is not None:
                    allowed_plan_types.update(plan.value.lower() for plan in BillingPlanType)

                if not allowed_plan_types:
                    allowed_plan_types.update(plan.value.lower() for plan in PlanType)

                plan_value = self.billing.plan_type
                if isinstance(plan_value, Enum):
                    plan_value = plan_value.value
                plan_value = str(plan_value).lower()

                assert plan_value in allowed_plan_types, f"Invalid plan type: {self.billing.plan_type}"

                # Persist the normalized plan type for downstream lookups that expect
                # lower-case string keys (e.g. quota resolution).
                self.billing.plan_type = plan_value
                
            # Validate RGPD config
            if self.rgpd.enabled:
                assert self.rgpd.request_processing_days <= 30, "GDPR requires processing within 30 days"
                assert self.rgpd.breach_notification_hours <= 72, "GDPR requires breach notification within 72 hours"
                assert self.rgpd.consent_renewal_days > 0, "Consent renewal period must be positive"
                
                # Validate retention periods
                for category, days in self.rgpd.retention_periods.items():
                    assert days > 0, f"Retention period for {category} must be positive"
            
            # Create necessary directories
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            if self.storage.backend == "local":
                Path(self.storage.local_base_path).mkdir(parents=True, exist_ok=True)
            if self.monitoring.create_reports:
                Path(self.monitoring.report_output_dir).mkdir(parents=True, exist_ok=True)
            if self.log_file:
                Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            if self.agent_first.enabled:
                Path(self.agent_first.knowledge_base_path).mkdir(parents=True, exist_ok=True)
                Path(self.agent_first.yaml_output_dir).mkdir(parents=True, exist_ok=True)
            
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
                "billing.plan_type": "trial",
                "rgpd.enabled": False,  # Disabled in dev for easier testing
                "expert_mode": True,  # Enable expert mode in dev by default
                "enable_intelligent_cleaning": False,  # Disabled by default in dev
                "enable_agent_first": True  # Enable Agent-First in dev for testing
            },
            "staging": {
                "debug": False,
                "verbose": 1,
                "hpo_n_iter": 30,
                "max_models_to_train": 30,
                "worker.max_workers": 4,
                "billing.plan_type": "pro",
                "rgpd.enabled": True,
                "expert_mode": False,  # Simplified for staging tests
                "enable_intelligent_cleaning": True,  # Enabled in staging
                "enable_agent_first": True  # Enabled in staging
            },
            "production": {
                "debug": False,
                "verbose": 0,
                "hpo_n_iter": 50,
                "max_models_to_train": 50,
                "worker.max_workers": 8,
                "monitoring.enabled": True,
                "monitoring.alerting_enabled": True,
                "billing.enable_metering": True,
                "rgpd.enabled": True,
                "rgpd.identity_verification_required": True,
                "expert_mode": False,  # Default to simplified in production
                "enable_intelligent_cleaning": True,  # Enabled in production
                "enable_agent_first": False  # Disabled by default in production (opt-in)
            }
        }
        
        return env_configs.get(self.environment, {})
    
    def apply_environment_config(self) -> None:
        """Apply environment-specific settings."""
        env_config = self.get_environment_config()
        
        # Check for expert mode in environment variable (takes precedence)
        expert_mode_env = os.getenv("AUTOML_EXPERT_MODE", "").lower()
        if expert_mode_env in ["true", "1", "yes", "on"]:
            env_config['expert_mode'] = True
        elif expert_mode_env in ["false", "0", "no", "off"]:
            env_config['expert_mode'] = False
        
        # Check for Agent-First mode in environment variable
        agent_first_env = os.getenv("AUTOML_AGENT_FIRST", "").lower()
        if agent_first_env in ["true", "1", "yes", "on"]:
            env_config['enable_agent_first'] = True
        elif agent_first_env in ["false", "0", "no", "off"]:
            env_config['enable_agent_first'] = False
        
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
def load_config(filepath: str = None, environment: str = None, expert_mode: Optional[bool] = None, agent_first: Optional[bool] = None) -> AutoMLConfig:
    """
    Load configuration with environment overrides and expert mode support.
    
    Args:
        filepath: Path to YAML config file
        environment: Environment name (development, staging, production)
        expert_mode: Override expert mode setting (None = use env var or config)
        agent_first: Override Agent-First setting (None = use env var or config)
    
    Returns:
        AutoMLConfig instance with appropriate settings
    """
    if filepath and Path(filepath).exists():
        config = AutoMLConfig.from_yaml(filepath)
    else:
        config = AutoMLConfig()
    
    # Override environment if specified
    if environment:
        config.environment = environment
        config.apply_environment_config()
    
    # Override expert mode if specified
    if expert_mode is not None:
        config.expert_mode = expert_mode
    else:
        # Check environment variable
        expert_mode_env = os.getenv("AUTOML_EXPERT_MODE", "").lower()
        if expert_mode_env in ["true", "1", "yes", "on"]:
            config.expert_mode = True
        elif expert_mode_env in ["false", "0", "no", "off"]:
            config.expert_mode = False
    
    # Override Agent-First mode if specified
    if agent_first is not None:
        config.enable_agent_first = agent_first
    else:
        # Check environment variable
        agent_first_env = os.getenv("AUTOML_AGENT_FIRST", "").lower()
        if agent_first_env in ["true", "1", "yes", "on"]:
            config.enable_agent_first = True
        elif agent_first_env in ["false", "0", "no", "off"]:
            config.enable_agent_first = False
    
    # Log mode status
    if config.expert_mode:
        logger.info("Expert mode ENABLED - All advanced options available")
    else:
        logger.info("Expert mode DISABLED - Using simplified configuration")
    
    if config.enable_agent_first and config.should_use_agent_first():
        logger.info("Agent-First mode ENABLED - Template-free AutoML with intelligent agents")
    else:
        logger.info("Agent-First mode DISABLED - Using traditional AutoML approach")
    
    # Validate configuration
    config.validate()
    
    return config


def save_config(config: AutoMLConfig, filepath: str) -> None:
    """Save configuration to file."""
    config.to_yaml(filepath)
    logger.info(f"Configuration saved to {filepath}")
