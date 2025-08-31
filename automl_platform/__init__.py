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

# Make commonly used classes available at package level
__all__ = [
    "AutoMLConfig",
    "load_config", 
    "AutoMLOrchestrator",
    "DataPreprocessor",
    "validate_data",
    "calculate_metrics",
    "detect_task",
    "get_available_models"
]

# Package metadata
PACKAGE_INFO = {
    "name": "automl_platform",
    "version": __version__,
    "description": "Production-ready AutoML platform with advanced features",
    "requires_python": ">=3.8",
    "license": "MIT",
    "keywords": ["machine learning", "automl", "automation", "ai", "ml"],
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
}

def get_version():
    """Get the current version of the AutoML platform."""
    return __version__

def get_package_info():
    """Get package metadata information."""
    return PACKAGE_INFO.copy()

# Check Python version compatibility
import sys
if sys.version_info < (3, 8):
    raise RuntimeError(
        f"AutoML Platform requires Python 3.8 or later. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )
