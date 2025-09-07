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
- Interactive UI with Streamlit dashboard

Version: 3.1.0
"""

__version__ = "3.1.0"
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
    STREAMLIT_AB_TESTING_AVAILABLE = True
except ImportError:
    STREAMLIT_AB_TESTING_AVAILABLE = False
    ABTestingDashboard = None
    integrate_ab_testing_to_main_app = None

# UI Module - Main dashboard and components
try:
    from .ui import (
        AutoMLDashboard,
        DataQualityVisualizer,
        ModelLeaderboard,
        FeatureImportanceVisualizer,
        DriftMonitor,
        ChatInterface,
        check_ui_dependencies,
        get_ui_status,
        launch_dashboard
    )
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False
    AutoMLDashboard = None
    DataQualityVisualizer = None
    ModelLeaderboard = None
    FeatureImportanceVisualizer = None
    DriftMonitor = None
    ChatInterface = None
    check_ui_dependencies = None
    get_ui_status = None
    launch_dashboard = None

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

# ============================================================================
# Dynamic detection of API module availability
# Each flag is set based on whether the corresponding module can be imported
# ============================================================================

# Batch Inference API
try:
    from .api import batch_inference
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False

# Billing API
try:
    from .api import billing
    BILLING_AVAILABLE = True
except ImportError:
    BILLING_AVAILABLE = False

# Data Connectors API
try:
    from .api import connectors
    CONNECTORS_AVAILABLE = True
except ImportError:
    CONNECTORS_AVAILABLE = False

# Feature Store API
try:
    from .api import feature_store
    FEATURE_STORE_AVAILABLE = True
except ImportError:
    FEATURE_STORE_AVAILABLE = False

# Infrastructure API
try:
    from .api import infrastructure
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False

# LLM Endpoints API
try:
    from .api import llm_endpoints
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# MLOps Endpoints API
try:
    from .api import mlops_endpoints
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False

# Model Versioning API
try:
    from .api import model_versioning
    VERSIONING_AVAILABLE = True
except ImportError:
    VERSIONING_AVAILABLE = False

# Streaming API
try:
    from .api import streaming
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

# WebSocket API
try:
    from .api import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Build __all__ list dynamically
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
    
    # Availability flags
    "UI_AVAILABLE",
    "OPTIMIZATIONS_AVAILABLE",
    "STREAMLIT_AB_TESTING_AVAILABLE",
]

# Add UI components conditionally
if UI_AVAILABLE:
    __all__.extend([
        "AutoMLDashboard",
        "DataQualityVisualizer",
        "ModelLeaderboard",
        "FeatureImportanceVisualizer",
        "DriftMonitor",
        "ChatInterface",
        "check_ui_dependencies",
        "get_ui_status",
        "launch_dashboard",
    ])

# Add Streamlit A/B Testing Dashboard conditionally
if STREAMLIT_AB_TESTING_AVAILABLE:
    __all__.extend([
        "ABTestingDashboard",
        "integrate_ab_testing_to_main_app",
    ])

# Add optimization components conditionally
if OPTIMIZATIONS_AVAILABLE:
    __all__.extend([
        "DistributedTrainer",
        "IncrementalLearner",
        "PipelineCache",
        "CacheConfig",
    ])

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
                  "monitoring", "scheduler", "worker", "tabnet", "ui", "dashboard", "streamlit"],
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
        "ui": ["streamlit", "ab_testing_dashboard", "interactive_visualizations", "chat_interface"],
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
    """
    Check which API features are available.
    All flags are dynamically determined based on module import success.
    """
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
        "streamlit": STREAMLIT_AB_TESTING_AVAILABLE
    }

def check_ui_features():
    """
    Check which UI features are available.
    
    Returns:
        dict: UI feature availability
    """
    features = {
        "dashboard": UI_AVAILABLE,
        "streamlit_ab_testing": STREAMLIT_AB_TESTING_AVAILABLE
    }
    
    # Get detailed UI status if available
    if UI_AVAILABLE and check_ui_dependencies:
        features["dependencies"] = check_ui_dependencies()
    
    if UI_AVAILABLE and get_ui_status:
        features["status"] = get_ui_status()
    
    return features

def check_all_features():
    """
    Check all platform features availability.
    
    Returns:
        dict: Complete feature availability report
    """
    return {
        "optimizations": check_optimizations(),
        "api": check_api_features(),
        "ui": check_ui_features(),
        "core": {
            "config": True,
            "orchestrator": True,
            "data_prep": True,
            "metrics": True,
            "model_selection": True
        }
    }

def initialize_platform(config_path: str = None, environment: str = "production", 
                       enable_optimizations: bool = True, enable_api_features: bool = True,
                       enable_ui: bool = True):
    """
    Initialize the AutoML platform with all services including optimizations, API features, and UI.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name (development, staging, production)
        enable_optimizations: Whether to enable optimization features
        enable_api_features: Whether to enable API features
        enable_ui: Whether to enable UI features
    
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
    
    # Initialize UI if available and enabled
    if enable_ui and UI_AVAILABLE:
        services["ui"] = {
            "dashboard": AutoMLDashboard,
            "launch_dashboard": launch_dashboard,
            "status": get_ui_status() if get_ui_status else None
        }
    
    # Initialize API services if available and enabled
    if enable_api_features:
        # Initialize Batch Processing if available
        if BATCH_AVAILABLE:
            from .api.batch_inference import BatchInferenceEngine
            services["batch_engine"] = BatchInferenceEngine(config)
        
        # Initialize Billing if available
        if BILLING_AVAILABLE:
            from .api.billing import BillingService
            services["billing"] = BillingService(config)
        
        # Initialize Connectors if available
        if CONNECTORS_AVAILABLE:
            from .api.connectors import ConnectorFactory
            services["connectors"] = ConnectorFactory(config)
        
        # Initialize Feature Store if available
        if FEATURE_STORE_AVAILABLE:
            from .api.feature_store import APIFeatureStore
            services["feature_store"] = APIFeatureStore(config)
        
        # Initialize Infrastructure if available
        if INFRASTRUCTURE_AVAILABLE:
            from .api.infrastructure import TenantManager
            services["tenant_manager"] = TenantManager(config)
        
        # Initialize Model Versioning if available
        if VERSIONING_AVAILABLE:
            from .api.model_versioning import ModelVersionManager as APIModelVersionManager
            services["version_manager"] = APIModelVersionManager(config)
        
        # Initialize Streaming if available
        if STREAMING_AVAILABLE:
            from .api.streaming import StreamingOrchestrator
            services["streaming"] = StreamingOrchestrator(config)
        
        # Initialize WebSocket if available
        if WEBSOCKET_AVAILABLE:
            from .api.websocket import ConnectionManager, WebSocketServer
            services["websocket_manager"] = ConnectionManager()
            services["websocket_server"] = WebSocketServer(config)
    
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
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"AutoML Platform initialized in {environment} mode")
    logger.info(f"Features enabled: {check_all_features()}")
    
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
    
    # Include API routers if available and enabled
    if enable_api_features:
        if BILLING_AVAILABLE:
            from .api.billing import billing_router
            app.include_router(billing_router)
        
        if LLM_AVAILABLE:
            from .api.llm_endpoints import llm_router
            app.include_router(llm_router)
        
        if MLOPS_AVAILABLE:
            from .api.mlops_endpoints import mlops_router
            app.include_router(mlops_router)
    
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
                "ui": UI_AVAILABLE,
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
            "features": check_all_features()
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
        
        # Initialize WebSocket service if available
        if WEBSOCKET_AVAILABLE and enable_api_features:
            from .api.websocket import WebSocketServer
            app.state.websocket_server = WebSocketServer(config)
            logger.info("WebSocket service initialized")
        
        logger.info("AutoML Platform API started successfully")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Shutting down AutoML Platform API...")
        
        # Shutdown WebSocket service if available
        if hasattr(app.state, 'websocket_server'):
            await app.state.websocket_server.shutdown()
            logger.info("WebSocket service shut down")
        
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
        # Check if API modules are available and import them
        if VERSIONING_AVAILABLE:
            from .api.model_versioning import ModelVersionManager as APIModelVersionManager
            services["version_manager"] = APIModelVersionManager(config)
        
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

# Log feature availability
all_features = check_all_features()
logger.info(f"Platform features: {all_features}")

# Special messages for key features
if UI_AVAILABLE:
    logger.info("UI Dashboard available - run 'launch_dashboard()' to start")

if OPTIMIZATIONS_AVAILABLE:
    logger.info("Optimization components available: distributed training, incremental learning, pipeline cache")

if any(check_api_features().values()):
    logger.info(f"API features available: {', '.join([k for k, v in check_api_features().items() if v])}")
