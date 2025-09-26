"""
Core Package Facade
===================
Re-exports from automl_platform.core to maintain backward compatibility.
This allows existing code to continue using 'from core import ...' imports.
"""

# Re-export everything from automl_platform.core
try:
    from automl_platform.core import *
    from automl_platform.core import __all__ as _core_all
    
    # Export the same symbols as the original package
    if _core_all:
        __all__ = _core_all
    else:
        # If no __all__ defined, export common core modules
        __all__ = [
            'config',
            'database',
            'auth',
            'cache',
            'logging',
            'metrics',
            'utils'
        ]
except ImportError:
    # Fallback if automl_platform.core doesn't exist yet
    # Define minimal exports to prevent import errors
    __all__ = []
    
    # You can add fallback implementations here if needed
    class Config:
        """Fallback configuration class"""
        def __init__(self):
            self.settings = {}
    
    class Database:
        """Fallback database class"""
        def __init__(self):
            self.connection = None
    
    # Export fallback classes if main package not available
    config = Config()
    database = Database()
