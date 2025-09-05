"""
AutoML Platform - Production-ready machine learning automation with optimizations

A comprehensive platform for automated machine learning with advanced features including:
- Automated model selection and hyperparameter optimization
- Feature engineering and data preprocessing
- Model monitoring and drift detection
- Multi-tenant architecture with billing
- LLM integration for intelligent assistance
- Real-time streaming capabilities
- Enterprise-grade security and deployment
- SSO authentication (Keycloak, Auth0, Okta)
- RGPD/GDPR compliance with full data subject rights
- Immutable audit trail with hash chain
- Distributed training with Ray/Dask
- Incremental learning for large datasets
- Intelligent pipeline caching
- Batch inference processing
- WebSocket real-time updates
- Multi-source data connectors
- Feature store management
- Model versioning and promotion
- A/B testing with statistical analysis
- MLflow integration for experiment tracking
- Advanced monitoring with Prometheus/Grafana
- Job scheduling with CPU/GPU queue management

Version: 3.0.1
"""

__version__ = "3.0.1"
__author__ = "AutoML Platform Team"
__email__ = "support@automl-platform.com"


# Core imports for easy access
from .config import AutoMLConfig, load_config
from .orchestrator import AutoMLOrchestrator
from .enhanced_orchestrator import EnhancedAutoMLOrchestrator
from .data_prep import DataPreprocessor, validate_data
from .metrics import calculate_metrics, detect_task
from .model_selection import get_available_models

# Advanced modules
from .ensemble import (
    AutoMLEnsemble, 
    AutoGluonEnsemble, 
    WeightedEnsemble,
    create_diverse_ensemble,
    create_ensemble_pipeline
)
from .feature_engineering import (
    AutoFeatureEngineer,
    create_time_series_features,
    create_text_features
)
from .inference import (
    load_pipeline,
    predict,
    predict_proba,
    predict_batch,
    save_predictions,
    explain_prediction,
    validate_input
)

# LLM and Intelligence
from .llm import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    LLMCache,
    AdvancedRAGSystem,
    EnhancedDataCleaningAgent,
    AutoMLLLMAssistant,
    WebSocketChatHandler
)

# Data Quality
from .data_quality_agent import (
    DataQualityAssessment,
    AkkioStyleCleaningAgent,
    DataRobotStyleQualityMonitor,
    IntelligentDataQualityAgent
)

# Deployment and Export
from .deployment import (
    ModelExportService,
    ExportFormat,
    ServingMode,
    DeploymentTarget,
    ModelMetadata,
    DeploymentSpec
)
from .export_service import (
    ModelExporter,
    ExportConfig
)

# A/B Testing
from .ab_testing import (
    ABTestingService,
    TestStatus,
    ModelType,
    ABTestConfig,
    ABTestResult,
    MetricsComparator,
    StatisticalTester
)

# Autoscaling and Resource Management
from .autoscaling import (
    AutoscalingService,
    ResourceManager,
    GPUScheduler,
    JobScheduler,
    ResourceType,
    ScalingStrategy,
    ResourceAllocation,
    ClusterNode
)

# Monitoring and MLOps
from .monitoring import (
    ModelMonitor,
    DriftDetector,
    DataQualityMonitor,
    AlertManager,
    MonitoringService,
    MonitoringIntegration,
    ModelPerformanceMetrics,
    NotificationPriority
)

# MLOps Service
from .mlops_service import (
    MLflowRegistry,
    RetrainingService,
    ModelExporter as MLOpsModelExporter,
    ModelVersionManager,
    create_mlops_service,
    ModelStage
)

# MLflow Registry Integration
from .mlflow_registry import (
    MLflowRegistry as MLflowRegistryV2,
    ModelStage as MLflowModelStage
)

# Storage and Persistence
from .storage import (
    StorageManager,
    StorageBackend,
    MinIOStorage,
    LocalStorage,
    ModelMetadata as StorageModelMetadata,
    FeatureStore
)

# Prompts and Templates
from .prompts import (
    PromptTemplates,
    PromptOptimizer
)

# Job Scheduling
from .scheduler import (
    SchedulerFactory,
    CeleryScheduler,
    RayScheduler,
    LocalScheduler,
    JobRequest,
    JobStatus,
    QueueType,
    PLAN_LIMITS
)

# Worker Management
from .worker import (
    AutoMLTask,
    train_full_pipeline,
    train_distributed_pipeline,
    train_incremental_pipeline,
    train_neural_pipeline_gpu,
    predict_batch as worker_predict_batch,
    warm_pipeline_cache,
    check_cache_health,
    clear_pipeline_cache,
    optimize_memory_usage,
    get_system_status,
    GPUResourceManager
)

# TabNet Implementation
from .tabnet_sklearn import (
    TabNetClassifier,
    TabNetRegressor,
    TabNet,
    TabNetLayer,
    TabNetEncoder,
    AttentiveTransformer
)

# Streamlit A/B Testing Dashboard
try:
    from .streamlit_ab_testing import (
        ABTestingDashboard,
        integrate_ab_testing_to_main_app
    )
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    ABTestingDashboard = None
    integrate_ab_testing_to_main_app = None

# Optimization imports
try:
    from .distributed_training import DistributedTrainer
    from .incremental_learning import IncrementalLearner
    from .pipeline_cache import PipelineCache, CacheConfig
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False
    DistributedTrainer = None
    IncrementalLearner = None
    PipelineCache = None
    CacheConfig = None

# Authentication and Security imports
from .auth import (
    init_auth_system,
    auth_router,
    get_current_user,
    require_permission,
    require_plan,
    AuthConfig,
    PasswordService,
    TokenService,
    APIKeyService,
    OAuthService,
    RBACService,
    QuotaService,
    AuditService,
    RateLimiter,
    PlanType,
    User,
    Tenant,
    Role,
    Permission,
    Project,
    APIKey,
    AuditLog
)
from .sso_service import SSOService, SSOProvider
from .audit_service import AuditService as AuditServiceV2, AuditEventType, AuditSeverity
from .rgpd_compliance_service import (
    RGPDComplianceService,
    GDPRRequestType,
    ConsentType,
    get_rgpd_service
)
from .auth_middleware import UnifiedAuthMiddleware, RGPDMiddleware, setup_auth_middleware

# All API module imports have been removed since they're already present in another file

# Set flags for removed API modules to False
BATCH_AVAILABLE = False
BatchInferenceEngine = None

BILLING_AVAILABLE = False
billing_router = None

CONNECTORS_AVAILABLE = False
ConnectorFactory = None

FEATURE_STORE_AVAILABLE = False
APIFeatureStore = None

INFRASTRUCTURE_AVAILABLE = False
TenantManager = None

LLM_AVAILABLE = False
llm_router = None

MLOPS_AVAILABLE = False
mlops_router = None

VERSIONING_AVAILABLE = False
ModelVersionManager = None

STREAMING_AVAILABLE = False
StreamingOrchestrator = None

WEBSOCKET_AVAILABLE = False
connection_manager = None
WebSocketServer = None
ConnectionManager = None
ChatService = None
NotificationService = None
LiveMonitoringService = None
CollaborationService = None
MessageType = None
Message = None
Notification = None
TrainingMetrics = None

# Make commonly used classes available at package level
__all__ = [
    # Core AutoML
    "AutoMLConfig",
    "load_config", 
    "AutoMLOrchestrator",
    "EnhancedAutoMLOrchestrator",
    "DataPreprocessor",
    "validate_data",
    "calculate_metrics",
    "detect_task",
    "get_available_models",
    
    # Advanced features
    "AutoMLEnsemble",
    "AutoGluonEnsemble",
    "WeightedEnsemble",
    "create_diverse_ensemble",
    "create_ensemble_pipeline",
    "AutoFeatureEngineer",
    "create_time_series_features",
    "create_text_features",
    
    # Inference
    "load_pipeline",
    "predict",
    "predict_proba",
    "predict_batch",
    "save_predictions",
    "explain_prediction",
    "validate_input",
    
    # LLM and Intelligence
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "LLMCache",
    "AdvancedRAGSystem",
    "EnhancedDataCleaningAgent",
    "AutoMLLLMAssistant",
    "WebSocketChatHandler",
    
    # Data Quality
    "DataQualityAssessment",
    "AkkioStyleCleaningAgent",
    "DataRobotStyleQualityMonitor",
    "IntelligentDataQualityAgent",
    
    # Deployment
    "ModelExportService",
    "ExportFormat",
    "ServingMode",
    "DeploymentTarget",
    "ModelMetadata",
    "DeploymentSpec",
    "ModelExporter",
    "ExportConfig",
    
    # A/B Testing
    "ABTestingService",
    "TestStatus",
    "ModelType",
    "ABTestConfig",
    "ABTestResult",
    "MetricsComparator",
    "StatisticalTester",
    
    # Autoscaling
    "AutoscalingService",
    "ResourceManager",
    "GPUScheduler",
    "JobScheduler",
    "ResourceType",
    "ScalingStrategy",
    "ResourceAllocation",
    "ClusterNode",
    
    # Monitoring and MLOps
    "ModelMonitor",
    "DriftDetector",
    "DataQualityMonitor",
    "AlertManager",
    "MonitoringService",
    "MonitoringIntegration",
    "ModelPerformanceMetrics",
    "MLflowRegistry",
    "RetrainingService",
    "ModelVersionManager",
    "create_mlops_service",
    
    # Storage
    "StorageManager",
    "StorageBackend",
    "MinIOStorage",
    "LocalStorage",
    "FeatureStore",
    
    # Prompts
    "PromptTemplates",
    "PromptOptimizer",
    
    # Scheduling
    "SchedulerFactory",
    "CeleryScheduler",
    "RayScheduler",
    "JobRequest",
    "JobStatus",
    "QueueType",
    
    # Worker
    "AutoMLTask",
    "train_full_pipeline",
    "train_distributed_pipeline",
    "train_incremental_pipeline",
    "train_neural_pipeline_gpu",
    "GPUResourceManager",
    
    # TabNet
    "TabNetClassifier",
    "TabNetRegressor",
    
    # Streamlit Dashboard
    "ABTestingDashboard",
    "integrate_ab_testing_to_main_app",
    
    # Optimizations
    "DistributedTrainer",
    "IncrementalLearner",
    "PipelineCache",
    "CacheConfig",
    "OPTIMIZATIONS_AVAILABLE",
    
    # Authentication & Security
    "init_auth_system",
    "auth_router",
    "get_current_user",
    "require_permission",
    "require_plan",
    "AuthConfig",
    "PasswordService",
    "TokenService",
    "APIKeyService",
    "OAuthService",
    "RBACService",
    "QuotaService",
    "AuditService",
    "RateLimiter",
    "PlanType",
    
    # SSO
    "SSOService",
    "SSOProvider",
    
    # Audit
    "AuditEventType",
    "AuditSeverity",
    
    # RGPD/GDPR
    "RGPDComplianceService",
    "GDPRRequestType",
    "ConsentType",
    "get_rgpd_service",
    
    # Middleware
    "UnifiedAuthMiddleware",
    "RGPDMiddleware",
    "setup_auth_middleware",
]

# Package metadata
PACKAGE_INFO = {
    "name": "automl_platform",
    "version": __version__,
    "description": "Production-ready AutoML platform with advanced features and optimizations",
    "requires_python": ">=3.8",
    "license": "MIT",
    "keywords": ["machine learning", "automl", "automation", "ai", "ml", "sso", "rgpd", "gdpr", 
                  "distributed", "cache", "batch", "streaming", "websocket", "mlops", "llm",
                  "feature-engineering", "ensemble", "deployment", "ab-testing", "mlflow",
                  "monitoring", "scheduler", "worker", "tabnet"],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    "features": {
        "core": ["automl", "feature_engineering", "model_selection", "ensemble", "inference"],
        "enterprise": ["multi_tenant", "billing", "monitoring", "ab_testing", "scheduler"],
        "security": ["sso", "rbac", "audit_trail", "encryption"],
        "compliance": ["rgpd", "gdpr", "data_retention", "consent_management"],
        "advanced": ["llm_integration", "streaming", "drift_detection", "data_quality", "prompts"],
        "optimizations": ["distributed_training", "incremental_learning", "pipeline_cache", "memory_optimization"],
        "deployment": ["model_export", "onnx", "pmml", "edge_deployment", "autoscaling", "docker"],
        "api": ["batch_inference", "websocket", "connectors", "feature_store", "model_versioning"],
        "mlops": ["mlflow", "experiment_tracking", "model_registry", "retraining", "monitoring"],
        "ui": ["streamlit", "ab_testing_dashboard", "websocket_chat"],
        "ml": ["tabnet", "neural_networks", "gpu_support"]
    }
}

def get_version():
    """Get the current version of the AutoML platform."""
    return __version__

def get_package_info():
    """Get package metadata information."""
    return PACKAGE_INFO.copy()

def check_optimizations():
    """Check which optimization features are available."""
    return {
        "distributed_training": DistributedTrainer is not None,
        "incremental_learning": IncrementalLearner is not None,
        "pipeline_cache": PipelineCache is not None,
        "all_available": OPTIMIZATIONS_AVAILABLE
    }

def check_api_features():
    """Check which API features are available."""
    return {
        "batch_processing": BATCH_AVAILABLE,
        "billing": BILLING_AVAILABLE,
        "connectors": CONNECTORS_AVAILABLE,
        "feature_store": FEATURE_STORE_AVAILABLE,
        "infrastructure": INFRASTRUCTURE_AVAILABLE,
        "llm": LLM_AVAILABLE,
        "mlops": MLOPS_AVAILABLE,
        "versioning": VERSIONING_AVAILABLE,
        "streaming": STREAMING_AVAILABLE,
        "websocket": WEBSOCKET_AVAILABLE,
        "streamlit": STREAMLIT_AVAILABLE
    }

def initialize_platform(config_path: str = None, environment: str = "production", 
                       enable_optimizations: bool = True, enable_api_features: bool = True):
    """
    Initialize the AutoML platform with all services including optimizations and API features.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name (development, staging, production)
        enable_optimizations: Whether to enable optimization features
        enable_api_features: Whether to enable API features
    
    Returns:
        Dictionary with initialized services
    """
    # Load configuration
    config = load_config(config_path, environment)
    
    # Initialize services
    services = {}
    
    # Initialize authentication system
    if config.api.enable_auth:
        services["auth"] = init_auth_system()
    
    # Initialize SSO if enabled
    if config.api.enable_sso:
        services["sso"] = SSOService()
    
    # Initialize audit service
    services["audit"] = AuditServiceV2()
    
    # Initialize RGPD service if enabled
    if config.rgpd.enabled:
        services["rgpd"] = RGPDComplianceService(
            audit_service=services["audit"]
        )
    
    # Initialize LLM Assistant
    if config.llm.enabled:
        llm_config = {
            'provider': config.llm.provider,
            'api_key': config.llm.api_key,
            'model_name': config.llm.model_name,
            'enable_rag': config.llm.enable_rag,
            'cache_responses': config.llm.cache_responses
        }
        services["llm_assistant"] = AutoMLLLMAssistant(llm_config)
    
    # Initialize Data Quality Agent
    services["data_quality"] = IntelligentDataQualityAgent(
        llm_provider=services.get("llm_assistant").llm if "llm_assistant" in services else None
    )
    
    # Initialize A/B Testing Service
    services["ab_testing"] = ABTestingService()
    
    # Initialize Autoscaling Service
    services["autoscaling"] = AutoscalingService(config)
    
    # Initialize Model Export Service
    services["model_export"] = ModelExportService()
    services["model_exporter"] = ModelExporter()
    
    # Initialize Storage Manager
    services["storage"] = StorageManager(
        backend=config.storage.backend if hasattr(config, 'storage') else 'local'
    )
    
    # Initialize Monitoring Service
    if hasattr(config, 'monitoring') and config.monitoring.enabled:
        services["monitoring"] = MonitoringService(services["storage"])
    
    # Initialize MLOps Service
    services["mlops"] = create_mlops_service(config)
    
    # Initialize Scheduler
    services["scheduler"] = SchedulerFactory.create_scheduler(
        config,
        services.get("billing")
    )
    
    # Note: WebSocket Server and API features initialization removed since modules are not available
    
    # Initialize optimization services if available and enabled
    if enable_optimizations and OPTIMIZATIONS_AVAILABLE:
        # Initialize pipeline cache
        if hasattr(config, 'cache') and config.cache.enabled:
            cache_config = CacheConfig(
                backend=config.cache.backend,
                redis_host=config.cache.redis_host,
                ttl_seconds=config.cache.ttl
            )
            services["cache"] = PipelineCache(cache_config)
        
        # Initialize distributed trainer
        if hasattr(config, 'distributed') and config.distributed.enabled:
            services["distributed"] = DistributedTrainer(
                backend=config.distributed.backend,
                n_workers=config.distributed.n_workers
            )
        
        # Initialize incremental learner
        if hasattr(config, 'incremental') and config.incremental.enabled:
            services["incremental"] = IncrementalLearner(
                max_memory_mb=config.incremental.max_memory_mb
            )
    
    # Note: API features initialization removed since modules are not available
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"AutoML Platform initialized in {environment} mode")
    logger.info(f"Optimizations available: {check_optimizations()}")
    logger.info(f"API features available: {check_api_features()}")
    
    return services

def create_app(config_path: str = None, environment: str = "production", 
              enable_optimizations: bool = True, enable_api_features: bool = True):
    """
    Create FastAPI application with all services configured including optimizations and API features.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
        enable_optimizations: Whether to enable optimization features
        enable_api_features: Whether to enable API features
    
    Returns:
        FastAPI application instance
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    # Load configuration
    config = load_config(config_path, environment)
    
    # Create FastAPI app
    app = FastAPI(
        title="AutoML Platform",
        description="Enterprise AutoML Platform with SSO, RGPD compliance, multi-tenant support, optimizations, and complete API",
        version=__version__,
        docs_url="/docs" if config.api.enable_docs else None,
        redoc_url="/redoc" if config.api.enable_docs else None
    )
    
    # Add CORS middleware if enabled
    if config.api.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Setup authentication and RGPD middleware
    setup_auth_middleware(app, config)
    
    # Register routers
    # Include auth router
    app.include_router(auth_router)
    
    # Note: API routers removed since modules are not available
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        from datetime import datetime
        return {
            "status": "healthy",
            "version": __version__,
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "auth": config.api.enable_auth,
                "sso": config.api.enable_sso,
                "rgpd": config.rgpd.enabled,
                "monitoring": config.monitoring.enabled,
                "optimizations": check_optimizations(),
                "api_features": check_api_features()
            }
        }
    
    # System status endpoint
    @app.get("/api/system/status")
    async def system_status():
        """Get detailed system status including optimization and API components."""
        from datetime import datetime
        import psutil
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": __version__,
            "environment": environment,
            "resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            },
            "optimizations": check_optimizations(),
            "api_features": check_api_features()
        }
        
        # Add cache stats if available
        if OPTIMIZATIONS_AVAILABLE and hasattr(config, 'cache') and config.cache.enabled:
            try:
                cache_config = CacheConfig(backend=config.cache.backend)
                cache = PipelineCache(cache_config)
                status["cache_stats"] = cache.get_stats()
            except:
                pass
        
        return status
    
    # Optimization endpoints
    if enable_optimizations and OPTIMIZATIONS_AVAILABLE:
        from fastapi import APIRouter, HTTPException
        
        opt_router = APIRouter(prefix="/api/optimizations", tags=["optimizations"])
        
        @opt_router.get("/status")
        async def optimization_status():
            """Get status of optimization components."""
            return {
                "available": check_optimizations(),
                "config": {
                    "cache_enabled": hasattr(config, 'cache') and config.cache.enabled,
                    "distributed_enabled": hasattr(config, 'distributed') and config.distributed.enabled,
                    "incremental_enabled": hasattr(config, 'incremental') and config.incremental.enabled
                }
            }
        
        @opt_router.post("/cache/clear")
        async def clear_cache():
            """Clear pipeline cache."""
            if not (hasattr(config, 'cache') and config.cache.enabled):
                raise HTTPException(status_code=400, detail="Cache not enabled")
            
            try:
                cache_config = CacheConfig(backend=config.cache.backend)
                cache = PipelineCache(cache_config)
                success = cache.clear_all()
                return {"success": success, "message": "Cache cleared"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @opt_router.get("/cache/stats")
        async def cache_stats():
            """Get cache statistics."""
            if not (hasattr(config, 'cache') and config.cache.enabled):
                raise HTTPException(status_code=400, detail="Cache not enabled")
            
            try:
                cache_config = CacheConfig(backend=config.cache.backend)
                cache = PipelineCache(cache_config)
                return cache.get_stats()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        app.include_router(opt_router)
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Starting AutoML Platform API...")
        
        # Note: WebSocket service initialization removed since module is not available
        
        logger.info("AutoML Platform API started successfully")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Shutting down AutoML Platform API...")
        
        # Note: WebSocket service shutdown removed since module is not available
        
        logger.info("AutoML Platform API shut down successfully")
    
    return app

def create_orchestrator(config_path: str = None, 
                       environment: str = "production",
                       enhanced: bool = True,
                       enable_optimizations: bool = True,
                       enable_api_features: bool = True):
    """
    Create an AutoML orchestrator with optional enhancements, optimizations, and API features.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
        enhanced: Whether to use EnhancedAutoMLOrchestrator
        enable_optimizations: Whether to enable optimization features
        enable_api_features: Whether to enable API features
    
    Returns:
        Orchestrator instance
    """
    config = load_config(config_path, environment)
    
    # Enable optimization features in config if requested
    if enable_optimizations and OPTIMIZATIONS_AVAILABLE:
        config.distributed_training = True
        config.incremental_learning = True
        config.enable_cache = True
        config.cache_backend = "redis"
        config.distributed_backend = "ray"
    
    # Initialize API services if requested
    services = {}
    if enable_api_features:
        # Initialize version manager if available
        if VERSIONING_AVAILABLE:
            services["version_manager"] = ModelVersionManager(config)
        
        # Initialize LLM Assistant
        if hasattr(config, 'llm') and config.llm.enabled:
            llm_config = {
                'provider': config.llm.provider,
                'api_key': config.llm.api_key,
                'model_name': config.llm.model_name,
                'enable_rag': config.llm.enable_rag,
                'cache_responses': config.llm.cache_responses
            }
            services["llm_assistant"] = AutoMLLLMAssistant(llm_config)
    
    # Create orchestrator
    if enhanced:
        orchestrator = EnhancedAutoMLOrchestrator(config)
    else:
        orchestrator = AutoMLOrchestrator(config)
    
    # Attach services to orchestrator
    for service_name, service in services.items():
        setattr(orchestrator, service_name, service)
    
    return orchestrator

# Check Python version compatibility
import sys
if sys.version_info < (3, 8):
    raise RuntimeError(
        f"AutoML Platform requires Python 3.8 or later. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )

# Log initialization
import logging
logger = logging.getLogger(__name__)
logger.info(f"AutoML Platform v{__version__} initialized")
if OPTIMIZATIONS_AVAILABLE:
    logger.info("Optimization components available: distributed training, incremental learning, pipeline cache")
else:
    logger.info("Running without optimization components. Install ray, dask, river for full functionality")

api_features = check_api_features()
if any(api_features.values()):
    logger.info(f"API features loaded: {', '.join([k for k, v in api_features.items() if v])}")
else:
    logger.info("No API features loaded. Check module dependencies.")

# Log newly added modules
logger.info("Additional modules loaded: monitoring, mlops_service, mlflow_registry, storage, prompts, scheduler, worker, tabnet")
if STREAMLIT_AVAILABLE:
    logger.info("Streamlit A/B testing dashboard available")
