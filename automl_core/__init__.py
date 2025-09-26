"""
AutoML Core Facade Package
==========================
Re-exports core services from automl_platform for easier imports.

This package provides a simplified interface to the AutoML platform's core services.
"""

# Try to import from automl_platform.core if available
try:
    from automl_platform.core import (
        HealthMonitor,
        ServiceRegistry,
        ConfigManager
    )
    CORE_SERVICES_AVAILABLE = True
except ImportError:
    # Core services not available
    HealthMonitor = None
    ServiceRegistry = None
    ConfigManager = None
    CORE_SERVICES_AVAILABLE = False

# Import RGPD compliance service
try:
    from automl_platform.rgpd_compliance_service import (
        RGPDComplianceService,
        RGPDRequest,
        RGPDRequestType,
        RGPDRequestStatus,
        ConsentRecord,
        ConsentType,
        ConsentStatus,
        GDPRRequestType,
        DatabaseService,
        AuditService
    )
    RGPD_SERVICE_AVAILABLE = True
except ImportError:
    # RGPD service not available
    RGPDComplianceService = None
    RGPDRequest = None
    RGPDRequestType = None
    RGPDRequestStatus = None
    ConsentRecord = None
    ConsentType = None
    ConsentStatus = None
    GDPRRequestType = None
    DatabaseService = None
    AuditService = None
    RGPD_SERVICE_AVAILABLE = False

# Build exports list dynamically based on what's available
__all__ = []

if CORE_SERVICES_AVAILABLE:
    __all__.extend([
        'HealthMonitor',
        'ServiceRegistry',
        'ConfigManager'
    ])

if RGPD_SERVICE_AVAILABLE:
    __all__.extend([
        'RGPDComplianceService',
        'RGPDRequest',
        'RGPDRequestType',
        'RGPDRequestStatus',
        'ConsentRecord',
        'ConsentType',
        'ConsentStatus',
        'GDPRRequestType',
        'DatabaseService',
        'AuditService'
    ])

# Version
__version__ = '1.0.0'

# Package metadata
__author__ = 'AutoML Platform Team'
__email__ = 'contact@automl-platform.com'
__description__ = 'Core services facade for AutoML Platform'

# Module docstring for help()
__doc__ = """
AutoML Core Package
===================

This package provides a facade for the AutoML platform's core services.

Available Services:
------------------
""" + ("""
- HealthMonitor: System health monitoring
- ServiceRegistry: Service registration and discovery
- ConfigManager: Configuration management
""" if CORE_SERVICES_AVAILABLE else "- Core services not available\n") + ("""
- RGPDComplianceService: GDPR compliance service
""" if RGPD_SERVICE_AVAILABLE else "- RGPD service not available\n") + """

Usage:
------
    from automl_core import RGPDComplianceService
    
    # Or import from submodules
    from automl_core.rgpd_compliance_service import RGPDComplianceService

For detailed documentation, see the individual service modules.
"""
