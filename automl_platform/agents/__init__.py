"""
AutoML Platform - Intelligent Data Cleaning Agents
Agent-First approach with template-free intelligence
"""

# Core agents
from .data_cleaning_orchestrator import DataCleaningOrchestrator
from .profiler_agent import ProfilerAgent
from .validator_agent import ValidatorAgent
from .cleaner_agent import CleanerAgent
from .controller_agent import ControllerAgent
from .agent_config import AgentConfig, AgentType

# Intelligent modules (Agent-First approach)
from .intelligent_context_detector import IntelligentContextDetector, MLContext
from .intelligent_config_generator import IntelligentConfigGenerator, OptimalConfig
from .adaptive_template_system import AdaptiveTemplateSystem, AdaptiveTemplate
from .universal_ml_agent import UniversalMLAgent, MLPipelineResult, KnowledgeBase

# Utilities
from .yaml_config_handler import YAMLConfigHandler
from .intelligent_data_cleaning import IntelligentDataCleaner, smart_clean_data

__all__ = [
    # Core agents
    'DataCleaningOrchestrator',
    'ProfilerAgent',
    'ValidatorAgent', 
    'CleanerAgent',
    'ControllerAgent',
    'AgentConfig',
    'AgentType',
    
    # Intelligent modules
    'IntelligentContextDetector',
    'MLContext',
    'IntelligentConfigGenerator',
    'OptimalConfig',
    'AdaptiveTemplateSystem',
    'AdaptiveTemplate',
    'UniversalMLAgent',
    'MLPipelineResult',
    'KnowledgeBase',
    
    # Utilities
    'YAMLConfigHandler',
    'IntelligentDataCleaner',
    'smart_clean_data'
]

__version__ = '2.0.0'  # Major version bump for Agent-First approach
