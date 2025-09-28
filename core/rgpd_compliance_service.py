"""
RGPD Compliance Service Facade
===============================
Re-exports from automl_platform.rgpd_compliance_service to maintain backward compatibility.
This allows existing code to:
- Use 'from core.rgpd_compliance_service import ...' imports
- Patch 'core.rgpd_compliance_service.DatabaseService' in tests
- Access all RGPD service functionality through the core namespace
"""

# Re-export everything from automl_platform.rgpd_compliance_service
from automl_platform.rgpd_compliance_service import *

# Explicitly import all symbols to ensure they're available for patching
from automl_platform.rgpd_compliance_service import (
    # Main service
    RGPDComplianceService,
    get_rgpd_service,
    
    # Request types and enums
    RGPDRequest,
    RGPDRequestType,
    RGPDRequestStatus,
    GDPRRequestType,  # Alias
    
    # Consent management
    ConsentRecord,
    ConsentType,
    ConsentStatus,
    
    # Data categories
    DataCategory,
    
    # Service dependencies
    DatabaseService,
    AuditService,
    
    # Audit enums (if available)
    AuditEventType,
    AuditSeverity,
    
    # Database models (if SQLAlchemy is available)
)

# Try to import optional database models
try:
    from automl_platform.rgpd_compliance_service import (
        ConsentRecordDB,
        DataSubjectRequest,
        PersonalDataRecord,
        Base,
    )
except ImportError:
    # These models might not be available if SQLAlchemy is not installed
    pass

# Try to import optional crypto functions
try:
    from automl_platform.rgpd_compliance_service import (
        Fernet,
        CRYPTO_AVAILABLE,
    )
except ImportError:
    # Crypto might not be available
    CRYPTO_AVAILABLE = False
    Fernet = None

# Expose redis namespace for tests to patch even when optional dependency
try:
    from automl_platform.rgpd_compliance_service import redis
except ImportError:
    class _RedisNamespace:
        Redis = None

    redis = _RedisNamespace()

# Try to import Redis availability flag
try:
    from automl_platform.rgpd_compliance_service import REDIS_AVAILABLE
except ImportError:
    REDIS_AVAILABLE = False

# Try to import SQLAlchemy availability flag
try:
    from automl_platform.rgpd_compliance_service import SQLALCHEMY_AVAILABLE
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Try to import Pandas availability flag
try:
    from automl_platform.rgpd_compliance_service import PANDAS_AVAILABLE
except ImportError:
    PANDAS_AVAILABLE = False

# Ensure singleton instance is available
if 'rgpd_service' not in locals():
    rgpd_service = None

# Export all symbols (maintaining compatibility with the original module)
__all__ = [
    # Main service
    'RGPDComplianceService',
    'get_rgpd_service',
    'rgpd_service',
    
    # Request types and enums
    'RGPDRequest',
    'RGPDRequestType',
    'RGPDRequestStatus',
    'GDPRRequestType',
    
    # Consent management
    'ConsentRecord',
    'ConsentType',
    'ConsentStatus',
    
    # Data categories
    'DataCategory',
    
    # Service dependencies
    'DatabaseService',
    'AuditService',
    
    # Audit enums
    'AuditEventType',
    'AuditSeverity',
    
    # Availability flags
    'CRYPTO_AVAILABLE',
    'REDIS_AVAILABLE',
    'SQLALCHEMY_AVAILABLE',
    'PANDAS_AVAILABLE',
    'redis',
]

# Add optional exports if they're available
if 'ConsentRecordDB' in locals():
    __all__.extend([
        'ConsentRecordDB',
        'DataSubjectRequest',
        'PersonalDataRecord',
        'Base',
    ])

if 'Fernet' in locals() and Fernet is not None:
    __all__.append('Fernet')
