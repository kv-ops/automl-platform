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
        # Main service
        RGPDComplianceService,
        get_rgpd_service,
        rgpd_service,
        
        # Request types and enums
        RGPDRequest,
        RGPDRequestType,
        RGPDRequestStatus,
        GDPRRequestType,  # Alias for compatibility
        
        # Consent management
        ConsentRecord,
        ConsentType,
        ConsentStatus,
        
        # Data categories
        DataCategory,
        
        # Service dependencies (important for test patching)
        DatabaseService,
        AuditService,
        
        # Audit enums
        AuditEventType,
        AuditSeverity,
        
        # Availability flags
        CRYPTO_AVAILABLE,
        REDIS_AVAILABLE,
        SQLALCHEMY_AVAILABLE,
        PANDAS_AVAILABLE
    )
    
    # Try to import optional database models
    try:
        from ..rgpd_compliance_service import (
            ConsentRecordDB,
            DataSubjectRequest,
            PersonalDataRecord,
            Base
        )
        _has_db_models = True
    except ImportError:
        _has_db_models = False
    
    # Try to import optional crypto functions
    try:
        from ..rgpd_compliance_service import Fernet
        _has_crypto = True
    except ImportError:
        _has_crypto = False
    
    # Build __all__ list with all available symbols
    __all__ = [
        # Core services
        'HealthMonitor',
        'ServiceRegistry',
        'ConfigManager',
        
        # RGPD main service
        'RGPDComplianceService',
        'get_rgpd_service',
        'rgpd_service',
        
        # RGPD request types and enums
        'RGPDRequest',
        'RGPDRequestType',
        'RGPDRequestStatus',
        'GDPRRequestType',
        
        # RGPD consent management
        'ConsentRecord',
        'ConsentType',
        'ConsentStatus',
        
        # RGPD data categories
        'DataCategory',
        
        # RGPD service dependencies
        'DatabaseService',
        'AuditService',
        
        # RGPD audit enums
        'AuditEventType',
        'AuditSeverity',
        
        # Availability flags
        'CRYPTO_AVAILABLE',
        'REDIS_AVAILABLE',
        'SQLALCHEMY_AVAILABLE',
        'PANDAS_AVAILABLE'
    ]
    
    # Add optional database models if available
    if _has_db_models:
        __all__.extend([
            'ConsentRecordDB',
            'DataSubjectRequest',
            'PersonalDataRecord',
            'Base'
        ])
    
    # Add crypto if available
    if _has_crypto:
        __all__.append('Fernet')
    
except ImportError:
    # If RGPD service is not available, export only core services
    __all__ = [
        'HealthMonitor',
        'ServiceRegistry',
        'ConfigManager'
    ]
    
    # Define placeholder classes to prevent import errors
    class RGPDComplianceService:
        """Placeholder for RGPD Compliance Service when not available"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("RGPD Compliance Service is not available. Please install required dependencies.")
    
    class DatabaseService:
        """Placeholder for Database Service when not available"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Database Service is not available.")
    
    class AuditService:
        """Placeholder for Audit Service when not available"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Audit Service is not available.")
    
    # Define placeholder function
    def get_rgpd_service():
        """Placeholder for get_rgpd_service when not available"""
        raise NotImplementedError("RGPD Compliance Service is not available.")
    
    rgpd_service = None
    
    # Set availability flags to False
    CRYPTO_AVAILABLE = False
    REDIS_AVAILABLE = False
    SQLALCHEMY_AVAILABLE = False
    PANDAS_AVAILABLE = False

# Version
__version__ = '3.2.1'
