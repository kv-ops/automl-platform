"""
RGPD Compliance Service Facade
==============================
Re-exports the RGPD compliance service from automl_platform.
This allows imports like: from automl_core.rgpd_compliance_service import RGPDComplianceService

This facade ensures compatibility with tests that expect to import from a core-like module.
"""

# Re-export everything from the main module
from automl_platform.rgpd_compliance_service import (
    # Main service class
    RGPDComplianceService,
    
    # Request-related classes and enums
    RGPDRequest,
    RGPDRequestType,
    RGPDRequestStatus,
    
    # Consent-related classes and enums  
    ConsentRecord,
    ConsentType,
    ConsentStatus,
    
    # Aliases and additional enums
    GDPRRequestType,
    DataCategory,
    
    # Stub services for testing
    DatabaseService,
    AuditService,
    
    # Helper function
    get_rgpd_service
)

# Also try to import audit-related enums if available
try:
    from automl_platform.rgpd_compliance_service import (
        AuditEventType,
        AuditSeverity
    )
    AUDIT_ENUMS_AVAILABLE = True
except ImportError:
    # Define minimal audit enums if not available from main module
    from enum import Enum
    
    AuditEventType = Enum('AuditEventType', {
        'DATA_CREATE': 'data_create',
        'DATA_READ': 'data_read', 
        'DATA_UPDATE': 'data_update',
        'DATA_DELETE': 'data_delete',
        'CONSENT_UPDATE': 'consent_update',
        'GDPR_REQUEST': 'gdpr_request',
        'LOGIN': 'login',
        'LOGOUT': 'logout',
        'MODEL_TRAIN': 'model_train',
        'MODEL_PREDICT': 'model_predict',
        'MODEL_EXPORT': 'model_export',
        'MODEL_DELETE': 'model_delete'
    })
    
    AuditSeverity = Enum('AuditSeverity', {
        'INFO': 'info',
        'WARNING': 'warning',
        'ERROR': 'error',
        'CRITICAL': 'critical'
    })
    
    AUDIT_ENUMS_AVAILABLE = False

# SQLAlchemy models (if needed by tests)
try:
    from automl_platform.rgpd_compliance_service import (
        ConsentRecordDB,
        DataSubjectRequest,
        PersonalDataRecord,
        Base
    )
    SQLALCHEMY_MODELS_AVAILABLE = True
except ImportError:
    # These are optional, only used if SQLAlchemy is available
    ConsentRecordDB = None
    DataSubjectRequest = None
    PersonalDataRecord = None
    Base = None
    SQLALCHEMY_MODELS_AVAILABLE = False

# Ensure all symbols are available for wildcard imports
__all__ = [
    # Core service
    'RGPDComplianceService',
    
    # Request-related
    'RGPDRequest',
    'RGPDRequestType', 
    'RGPDRequestStatus',
    
    # Consent-related
    'ConsentRecord',
    'ConsentType',
    'ConsentStatus',
    
    # Aliases and additional
    'GDPRRequestType',
    'DataCategory',
    
    # Stub services
    'DatabaseService',
    'AuditService',
    
    # Audit enums
    'AuditEventType',
    'AuditSeverity',
    
    # Helper
    'get_rgpd_service'
]

# Add SQLAlchemy models to exports if available
if SQLALCHEMY_MODELS_AVAILABLE:
    __all__.extend([
        'ConsentRecordDB',
        'DataSubjectRequest', 
        'PersonalDataRecord',
        'Base'
    ])

# Version info
__version__ = '1.0.0'

# Module docstring for help()
__doc__ = """
RGPD Compliance Service Module
==============================

This module provides a comprehensive GDPR/RGPD compliance service with:

Classes:
--------
- RGPDComplianceService: Main service class for GDPR compliance
- RGPDRequest: Data structure for GDPR requests
- ConsentRecord: Data structure for consent records
- DatabaseService: Mock database service for testing
- AuditService: Mock audit service for testing

Enums:
------
- RGPDRequestType: Types of GDPR requests (ACCESS, RECTIFICATION, ERASURE, etc.)
- RGPDRequestStatus: Status of requests (PENDING, COMPLETED, REJECTED, etc.)
- ConsentType: Types of consent (MARKETING, ANALYTICS, COOKIES, etc.)
- ConsentStatus: Status of consent (GRANTED, REVOKED, PENDING, EXPIRED)
- DataCategory: Categories of personal data
- AuditEventType: Types of audit events
- AuditSeverity: Severity levels for audit events

Usage:
------
    from automl_core.rgpd_compliance_service import RGPDComplianceService
    
    # Initialize service
    config = {'rgpd': {'enabled': True}}
    service = RGPDComplianceService(config)
    
    # Create a GDPR request
    request_id = service.create_request(
        user_id='user123',
        request_data={'type': 'ACCESS', 'details': 'Request my data'}
    )
    
    # Process the request
    service.process_request(request_id)
    
    # Manage consent
    consent_id = service.create_consent(
        user_id='user123',
        consent_data={'type': 'MARKETING', 'status': 'GRANTED'}
    )

For more information, see the main module documentation at:
automl_platform.rgpd_compliance_service
"""

# Compatibility layer for different import styles
# This allows both attribute and direct access
import sys
_current_module = sys.modules[__name__]

# Add all exported items as module attributes
for item_name in __all__:
    if item_name in globals():
        setattr(_current_module, item_name, globals()[item_name])
