"""
Core facade package
===================
This package provides a simplified import path for core services.
Located at the project root to allow: from core.rgpd_compliance_service import ...
"""

# Re-export from automl_platform.core if needed
try:
    from automl_platform.core import *
except ImportError:
    pass

# Version
__version__ = '3.2.1'
