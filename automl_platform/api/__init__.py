"""
AutoML Platform API Module - API endpoints and services

This module contains all API-specific components including:
- Batch inference processing
- Billing and usage tracking
- Data connectors for multiple sources
- Feature store management
- Infrastructure and deployment management
- LLM integration endpoints
- MLOps endpoints
- Model versioning and promotion
- Real-time streaming capabilities
- WebSocket connections

Version: 2.1.0
"""

__version__ = "2.1.0"
__author__ = "AutoML Platform Team"
__email__ = "support@automl-platform.com"

# API Module imports - Only include what actually exists in api/ directory

# Batch processing
try:
    from .batch_inference import (
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
    BatchJobConfig = None
    BatchJobResult = None
    BatchStatus = None
    BatchPriority = None
    BatchScheduler = None

# Billing
try:
    from .billing import BillingManager, PlanType, UsageTracker
    from .billing_routes import billing_router
    BILLING_AVAILABLE = True
except ImportError:
    BILLING_AVAILABLE = False
    BillingManager = None
    PlanType = None
    UsageTracker = None
    billing_router = None

# Data connectors
try:
    from .connectors import ConnectorFactory, ConnectionConfig
    CONNECTORS_AVAILABLE = True
except ImportError:
    CONNECTORS_AVAILABLE = False
    ConnectorFactory = None
    ConnectionConfig = None

# Feature store
try:
    from .feature_store import FeatureStore, FeatureSet, FeatureDefinition
    FEATURE_STORE_AVAILABLE = True
except ImportError:
    FEATURE_STORE_AVAILABLE = False
    FeatureStore = None
    FeatureSet = None
    FeatureDefinition = None

# Infrastructure
try:
    from .infrastructure import (
        TenantManager,
        SecurityManager,
        ResourceMonitor,
        DeploymentManager
    )
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    TenantManager = None
    SecurityManager = None
    ResourceMonitor = None
    DeploymentManager = None

# LLM endpoints
try:
    from .llm_endpoints import router as llm_router
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    llm_router = None

# MLOps endpoints
try:
    from .mlops_endpoints import router as mlops_router
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False
    mlops_router = None

# Model versioning
try:
    from .model_versioning import (
        ModelVersionManager,
        ModelVersion,
        VersionStatus,
        PromotionStrategy
    )
    VERSIONING_AVAILABLE = True
except ImportError:
    VERSIONING_AVAILABLE = False
    ModelVersionManager = None
    ModelVersion = None
    VersionStatus = None
    PromotionStrategy = None

# Streaming
try:
    from .streaming import (
        StreamConfig,
        StreamingOrchestrator,
        MLStreamProcessor
    )
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    StreamConfig = None
    StreamingOrchestrator = None
    MLStreamProcessor = None

# WebSocket
try:
    from .websocket import (
        connection_manager,
        websocket_endpoint,
        initialize_websocket_service,
        shutdown_websocket_service
    )
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    connection_manager = None
    websocket_endpoint = None
    initialize_websocket_service = None
    shutdown_websocket_service = None

# Export only API-specific components
__all__ = [
    # Batch Processing
    "BatchInferenceEngine",
    "BatchJobConfig",
    "BatchJobResult",
    "BatchStatus",
    "BatchPriority",
    "BatchScheduler",
    "BATCH_AVAILABLE",
    
    # Billing
    "BillingManager",
    "PlanType",
    "UsageTracker",
    "billing_router",
    "BILLING_AVAILABLE",
    
    # Connectors
    "ConnectorFactory",
    "ConnectionConfig",
    "CONNECTORS_AVAILABLE",
    
    # Feature Store
    "FeatureStore",
    "FeatureSet",
    "FeatureDefinition",
    "FEATURE_STORE_AVAILABLE",
    
    # Infrastructure
    "TenantManager",
    "SecurityManager", 
    "ResourceMonitor",
    "DeploymentManager",
    "INFRASTRUCTURE_AVAILABLE",
    
    # LLM
    "llm_router",
    "LLM_AVAILABLE",
    
    # MLOps
    "mlops_router",
    "MLOPS_AVAILABLE",
    
    # Model Versioning
    "ModelVersionManager",
    "ModelVersion",
    "VersionStatus",
    "PromotionStrategy",
    "VERSIONING_AVAILABLE",
    
    # Streaming
    "StreamConfig",
    "StreamingOrchestrator",
    "MLStreamProcessor",
    "STREAMING_AVAILABLE",
    
    # WebSocket
    "connection_manager",
    "websocket_endpoint",
    "initialize_websocket_service",
    "shutdown_websocket_service",
    "WEBSOCKET_AVAILABLE"
]

# Package metadata for API module
API_MODULE_INFO = {
    "name": "automl_platform.api",
    "version": __version__,
    "description": "API endpoints and services for AutoML Platform",
    "components": {
        "batch_inference": "Batch processing and inference",
        "billing": "Multi-tenant billing and usage tracking",
        "connectors": "Data source connectors",
        "feature_store": "Feature management and storage",
        "infrastructure": "Tenant and deployment management",
        "llm": "LLM integration endpoints",
        "mlops": "MLOps and monitoring endpoints",
        "model_versioning": "Model version control",
        "streaming": "Real-time streaming processing",
        "websocket": "WebSocket real-time communication"
    }
}

def get_api_version():
    """Get the current version of the API module."""
    return __version__

def get_api_info():
    """Get API module metadata information."""
    return API_MODULE_INFO.copy()

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

def create_auth_router():
    """
    Create authentication router with RGPD endpoints.
    This is a placeholder that should be implemented based on actual auth requirements.
    """
    from fastapi import APIRouter, HTTPException, Depends
    from typing import Optional
    
    router = APIRouter(prefix="/auth", tags=["authentication"])
    
    @router.post("/login")
    async def login():
        """Login endpoint placeholder."""
        raise HTTPException(status_code=501, detail="Not implemented")
    
    @router.post("/logout")
    async def logout():
        """Logout endpoint placeholder."""
        raise HTTPException(status_code=501, detail="Not implemented")
    
    @router.get("/me")
    async def get_current_user():
        """Get current user placeholder."""
        raise HTTPException(status_code=501, detail="Not implemented")
    
    # RGPD/GDPR endpoints
    @router.post("/gdpr/consent")
    async def update_consent():
        """Update user consent placeholder."""
        raise HTTPException(status_code=501, detail="Not implemented")
    
    @router.get("/gdpr/data")
    async def export_user_data():
        """Export user data placeholder."""
        raise HTTPException(status_code=501, detail="Not implemented")
    
    @router.delete("/gdpr/data")
    async def delete_user_data():
        """Delete user data placeholder."""
        raise HTTPException(status_code=501, detail="Not implemented")
    
    return router

# Logging
import logging
logger = logging.getLogger(__name__)

# Log API module initialization
logger.info(f"AutoML Platform API Module v{__version__} initialized")

# Check and log available features
api_features = check_api_features()
available_features = [feature for feature, available in api_features.items() if available]
if available_features:
    logger.info(f"API features available: {', '.join(available_features)}")
else:
    logger.info("No API features currently loaded. Check individual module dependencies.")

# Check Python version compatibility
import sys
if sys.version_info < (3, 8):
    raise RuntimeError(
        f"AutoML Platform API requires Python 3.8 or later. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )
