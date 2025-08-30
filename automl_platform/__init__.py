"""
AutoML Platform Package
A comprehensive AutoML solution with advanced features
"""

__version__ = "1.0.0"
__author__ = "AutoML Platform Team"

# Package metadata
__all__ = [
    'AutoMLConfig',
    'DataPreprocessor',
    'AutoMLOrchestrator',
    'detect_task',
    'calculate_metrics',
    '__version__'
]

# Lazy imports to avoid circular dependencies
# Import only when explicitly needed by using:
# from automl_platform.config import AutoMLConfig
# from automl_platform.orchestrator import AutoMLOrchestrator
# etc.

def get_config():
    """Lazy import for config module"""
    from .config import AutoMLConfig, load_config
    return AutoMLConfig, load_config

def get_orchestrator():
    """Lazy import for orchestrator module"""
    from .orchestrator import AutoMLOrchestrator
    return AutoMLOrchestrator

def get_data_prep():
    """Lazy import for data_prep module"""
    from .data_prep import DataPreprocessor, validate_data
    return DataPreprocessor, validate_data

def get_metrics():
    """Lazy import for metrics module"""
    from .metrics import calculate_metrics, detect_task
    return calculate_metrics, detect_task

def get_model_selection():
    """Lazy import for model_selection module"""
    from .model_selection import get_available_models, get_param_grid
    return get_available_models, get_param_grid

# Optional: Import the most commonly used classes only
# These are safe imports that don't cause circular dependencies
try:
    from .config import AutoMLConfig
    from .metrics import detect_task, calculate_metrics
except ImportError as e:
    # If there's still an import error, just pass
    # The modules can still be imported directly
    pass
