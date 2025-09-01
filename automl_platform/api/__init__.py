"""
AutoML Platform - Production-ready machine learning automation

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

Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "AutoML Platform Team"
__email__ = "support@automl-platform.com"

# Core imports for easy access
from .config import AutoMLConfig, load_config
from .orchestrator import AutoMLOrchestrator
from .data_prep import DataPreprocessor, validate_data
from .metrics import calculate_metrics, detect_task
from .model_selection import get_available_models

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

# Make commonly used classes available at package level
__all__ = [
    # Core AutoML
    "AutoMLConfig",
    "load_config", 
    "AutoMLOrchestrator",
    "DataPreprocessor",
    "validate_data",
    "calculate_metrics",
    "detect_task",
    "get_available_models",
    
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
    "setup_auth_middleware"
]

# Package metadata
PACKAGE_INFO = {
    "name": "automl_platform",
    "version": __version__,
    "description": "Production-ready AutoML platform with advanced features",
    "requires_python": ">=3.8",
    "license": "MIT",
    "keywords": ["machine learning", "automl", "automation", "ai", "ml", "sso", "rgpd", "gdpr"],
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
        "advanced": ["llm_integration", "streaming", "drift_detection"]
    }
}

def get_version():
    """Get the current version of the AutoML platform."""
    return __version__

def get_package_info():
    """Get package metadata information."""
    return PACKAGE_INFO.copy()

def initialize_platform(config_path: str = None, environment: str = "production"):
    """
    Initialize the AutoML platform with all services.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name (development, staging, production)
    
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
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"AutoML Platform initialized in {environment} mode")
    
    return services

def create_app(config_path: str = None, environment: str = "production"):
    """
    Create FastAPI application with all services configured.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
    
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
        description="Enterprise AutoML Platform with SSO, RGPD compliance, and multi-tenant support",
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
                "monitoring": config.monitoring.enabled
            }
        }
    
    return app

# Check Python version compatibility
import sys
if sys.version_info < (3, 8):
    raise RuntimeError(
        f"AutoML Platform requires Python 3.8 or later. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )
