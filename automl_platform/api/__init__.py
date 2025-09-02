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

"""


# Core imports for easy access
from .config import AutoMLConfig, load_config
from .orchestrator import AutoMLOrchestrator
from .enhanced_orchestrator import EnhancedAutoMLOrchestrator
from .data_prep import DataPreprocessor, validate_data
from .metrics import calculate_metrics, detect_task
from .model_selection import get_available_models

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
    require_plan
)
from .sso_service import SSOService, SSOProvider
from .audit_service import AuditService, AuditEventType, AuditSeverity
from .rgpd_compliance_service import (
    RGPDComplianceService,
    GDPRRequestType,
    ConsentType,
    get_rgpd_service
)
from .auth_middleware import UnifiedAuthMiddleware, RGPDMiddleware, setup_auth_middleware

# API Module imports
try:
    # Batch processing
    from .api.batch_inference import (
        BatchInferenceEngine,
        BatchJobConfig,
        BatchJobResult,
        BatchStatus,
        BatchPriority,
        BatchScheduler
    )
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False
    BatchInferenceEngine = None

try:
    # Billing
    from .api.billing import BillingManager, PlanType, UsageTracker
    from .api.billing_routes import billing_router
    BILLING_AVAILABLE = True
except ImportError:
    BILLING_AVAILABLE = False
    billing_router = None

try:
    # Data connectors
    from .api.connectors import ConnectorFactory, ConnectionConfig
    CONNECTORS_AVAILABLE = True
except ImportError:
    CONNECTORS_AVAILABLE = False
    ConnectorFactory = None

try:
    # Feature store
    from .api.feature_store import FeatureStore, FeatureSet, FeatureDefinition
    FEATURE_STORE_AVAILABLE = True
except ImportError:
    FEATURE_STORE_AVAILABLE = False
    FeatureStore = None

try:
    # Infrastructure
    from .api.infrastructure import (
        TenantManager,
        SecurityManager,
        ResourceMonitor,
        DeploymentManager
    )
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    TenantManager = None

try:
    # LLM endpoints
    from .api.llm_endpoints import router as llm_router
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    llm_router = None

try:
    # MLOps endpoints
    from .api.mlops_endpoints import router as mlops_router
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False
    mlops_router = None

try:
    # Model versioning
    from .api.model_versioning import (
        ModelVersionManager,
        ModelVersion,
        VersionStatus,
        PromotionStrategy
    )
    VERSIONING_AVAILABLE = True
except ImportError:
    VERSIONING_AVAILABLE = False
    ModelVersionManager = None

try:
    # Streaming
    from .api.streaming import (
        StreamConfig,
        StreamingOrchestrator,
        MLStreamProcessor
    )
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    StreamingOrchestrator = None

try:
    # WebSocket
    from .api.websocket import (
        connection_manager,
        websocket_endpoint,
        initialize_websocket_service,
        shutdown_websocket_service
    )
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    connection_manager = None

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
    
    # SSO
    "SSOService",
    "SSOProvider",
    
    # Audit
    "AuditService",
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
    
    # API Components
    "BatchInferenceEngine",
    "BatchJobConfig",
    "BillingManager",
    "billing_router",
    "ConnectorFactory",
    "ConnectionConfig",
    "FeatureStore",
    "TenantManager",
    "llm_router",
    "mlops_router",
    "ModelVersionManager",
    "StreamingOrchestrator",
    "connection_manager",
    "websocket_endpoint"
]

# Package metadata
PACKAGE_INFO = {
    "name": "automl_platform",
    "version": __version__,
    "description": "Production-ready AutoML platform with advanced features and optimizations",
    "requires_python": ">=3.8",
    "license": "MIT",
    "keywords": ["machine learning", "automl", "automation", "ai", "ml", "sso", "rgpd", "gdpr", 
                  "distributed", "cache", "batch", "streaming", "websocket", "mlops"],
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
        "core": ["automl", "feature_engineering", "model_selection"],
        "enterprise": ["multi_tenant", "billing", "monitoring"],
        "security": ["sso", "rbac", "audit_trail", "encryption"],
        "compliance": ["rgpd", "gdpr", "data_retention", "consent_management"],
        "advanced": ["llm_integration", "streaming", "drift_detection"],
        "optimizations": ["distributed_training", "incremental_learning", "pipeline_cache", "memory_optimization"],
        "api": ["batch_inference", "websocket", "connectors", "feature_store", "model_versioning"]
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
        "websocket": WEBSOCKET_AVAILABLE
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
    services["audit"] = AuditService()
    
    # Initialize RGPD service if enabled
    if config.rgpd.enabled:
        services["rgpd"] = RGPDComplianceService(
            audit_service=services["audit"]
        )
    
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
    
    # Initialize API features if enabled
    if enable_api_features:
        # Initialize batch processing
        if BATCH_AVAILABLE and hasattr(config, 'batch') and config.batch.enabled:
            services["batch"] = BatchInferenceEngine(config)
            services["batch_scheduler"] = BatchScheduler(services["batch"])
        
        # Initialize billing
        if BILLING_AVAILABLE and hasattr(config, 'billing') and config.billing.enabled:
            services["billing"] = BillingManager(
                tenant_manager=services.get("tenant_manager"),
                db_url=config.billing.db_url
            )
        
        # Initialize feature store
        if FEATURE_STORE_AVAILABLE and hasattr(config, 'feature_store') and config.feature_store.enabled:
            services["feature_store"] = FeatureStore(config.feature_store.to_dict())
        
        # Initialize infrastructure
        if INFRASTRUCTURE_AVAILABLE:
            services["tenant_manager"] = TenantManager()
            services["security_manager"] = SecurityManager()
            services["resource_monitor"] = ResourceMonitor(services["tenant_manager"])
            services["deployment_manager"] = DeploymentManager(services["tenant_manager"])
        
        # Initialize model versioning
        if VERSIONING_AVAILABLE:
            services["version_manager"] = ModelVersionManager(config)
        
        # Initialize streaming
        if STREAMING_AVAILABLE and hasattr(config, 'streaming') and config.streaming.enabled:
            stream_config = StreamConfig(
                platform=config.streaming.platform,
                brokers=config.streaming.brokers,
                topic=config.streaming.topic
            )
            services["streaming"] = StreamingOrchestrator(stream_config)
    
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
    from .api import create_auth_router
    
    # Auth & RGPD routes
    auth_router = create_auth_router()
    app.include_router(auth_router, prefix="/api")
    
    # Include API routers if available and enabled
    if enable_api_features:
        # Billing routes
        if BILLING_AVAILABLE and billing_router:
            app.include_router(billing_router)
        
        # LLM routes
        if LLM_AVAILABLE and llm_router:
            app.include_router(llm_router)
        
        # MLOps routes
        if MLOPS_AVAILABLE and mlops_router:
            app.include_router(mlops_router)
    
    # WebSocket endpoint
    if WEBSOCKET_AVAILABLE and websocket_endpoint:
        @app.websocket("/ws")
        async def websocket_route(websocket, token: str):
            await websocket_endpoint(websocket, token)
    
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
        logger.info("Starting AutoML Platform API...")
        
        # Initialize WebSocket service if available
        if WEBSOCKET_AVAILABLE and hasattr(config, 'websocket') and config.websocket.enabled:
            redis_url = config.websocket.redis_url if hasattr(config.websocket, 'redis_url') else None
            await initialize_websocket_service(redis_url)
        
        logger.info("AutoML Platform API started successfully")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        logger.info("Shutting down AutoML Platform API...")
        
        # Shutdown WebSocket service if available
        if WEBSOCKET_AVAILABLE:
            await shutdown_websocket_service()
        
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
        # Initialize feature store if available
        if FEATURE_STORE_AVAILABLE and hasattr(config, 'feature_store') and config.feature_store.enabled:
            services["feature_store"] = FeatureStore(config.feature_store.to_dict())
        
        # Initialize version manager if available
        if VERSIONING_AVAILABLE:
            services["version_manager"] = ModelVersionManager(config)
    
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
