"""
Core infrastructure services for AutoML Platform
================================================
Central services for health monitoring, service registry, and configuration management.
"""

from .health_monitor import HealthMonitor
from .service_registry import ServiceRegistry
from .config_manager import ConfigManager

# Also export RGPD compliance service components if needed
try:
    from ..rgpd_compliance_service import (
        RGPDComplianceService,
        RGPDRequest,
        RGPDRequestType,
        RGPDRequestStatus,
        ConsentRecord,
        ConsentType,
        ConsentStatus
    )
    
    __all__ = [
        'HealthMonitor',
        'ServiceRegistry',
        'ConfigManager',
        'RGPDComplianceService',
        'RGPDRequest',
        'RGPDRequestType',
        'RGPDRequestStatus',
        'ConsentRecord',
        'ConsentType',
        'ConsentStatus'
    ]
except ImportError:
    # If RGPD service is not available, export only core services
    __all__ = [
        'HealthMonitor',
        'ServiceRegistry',
        'ConfigManager'
    ]

# Version
__version__ = '3.2.1'
