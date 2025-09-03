"""
Core infrastructure services for AutoML Platform
================================================
Central services for health monitoring, service registry, and configuration management.
"""

from .health_monitor import HealthMonitor
from .service_registry import ServiceRegistry
from .config_manager import ConfigManager

__all__ = [
    'HealthMonitor',
    'ServiceRegistry',
    'ConfigManager'
]

# Version
__version__ = '1.0.0'
