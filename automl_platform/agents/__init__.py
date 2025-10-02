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
from .agent_config import AgentConfig, AgentType, LLMProvider

# Intelligent modules (Agent-First approach)
from .intelligent_context_detector import IntelligentContextDetector, MLContext
from .intelligent_config_generator import IntelligentConfigGenerator, OptimalConfig
from .adaptive_template_system import AdaptiveTemplateSystem, AdaptiveTemplate

# Universal ML Agent (PRODUCTION VERSION)
from .universal_ml_agent import (
    ProductionUniversalMLAgent,
    ProductionMLPipelineResult,
    ProductionKnowledgeBase,
    MemoryMonitor,
    MemoryBudget,
    LRUMemoryCache,
    memory_safe,
    dataframe_batch_processor
)

# Utilities
from .yaml_config_handler import YAMLConfigHandler
from .intelligent_data_cleaning import IntelligentDataCleaner, smart_clean_data

# Core utilities from utils.py
from .utils import (
    # Circuit Breaker & Retry
    CircuitBreaker,
    async_retry,
    
    # JSON Parsing
    parse_llm_json,
    validate_llm_json_schema,
    
    # Cost Tracking
    track_llm_cost,
    
    # Memory Management
    BoundedList,
    
    # Logging
    sanitize_for_logging,
    safe_log_config,
    
    # Async Utilities
    run_parallel,
    AsyncInitMixin,
    
    # Performance Monitoring
    PerformanceMetrics,
    PerformanceMonitor,
    
    # Health Check
    HealthChecker
)


__all__ = [
    # ============================================================================
    # CORE AGENTS
    # ============================================================================
    'DataCleaningOrchestrator',
    'ProfilerAgent',
    'ValidatorAgent', 
    'CleanerAgent',
    'ControllerAgent',
    'AgentConfig',
    'AgentType',
    'LLMProvider',
    
    # ============================================================================
    # INTELLIGENT MODULES (Agent-First)
    # ============================================================================
    'IntelligentContextDetector',
    'MLContext',
    'IntelligentConfigGenerator',
    'OptimalConfig',
    'AdaptiveTemplateSystem',
    'AdaptiveTemplate',
    
    # ============================================================================
    # UNIVERSAL ML AGENT (Production)
    # ============================================================================
    'ProductionUniversalMLAgent',
    'ProductionMLPipelineResult',
    'ProductionKnowledgeBase',
    
    # Memory Protection Classes
    'MemoryMonitor',
    'MemoryBudget',
    'LRUMemoryCache',
    'memory_safe',
    'dataframe_batch_processor',
    
    # ============================================================================
    # UTILITIES
    # ============================================================================
    'YAMLConfigHandler',
    'IntelligentDataCleaner',
    'smart_clean_data',
    
    # ============================================================================
    # CORE UTILITIES (from utils.py)
    # ============================================================================
    # Circuit Breaker & Retry
    'CircuitBreaker',
    'async_retry',
    
    # JSON Parsing
    'parse_llm_json',
    'validate_llm_json_schema',
    
    # Cost Tracking
    'track_llm_cost',
    
    # Memory Management
    'BoundedList',
    
    # Logging
    'sanitize_for_logging',
    'safe_log_config',
    
    # Async Utilities
    'run_parallel',
    'AsyncInitMixin',
    
    # Performance Monitoring
    'PerformanceMetrics',
    'PerformanceMonitor',
    
    # Health Check
    'HealthChecker'
]

__version__ = '3.2.1'


# ============================================================================
# CONVENIENCE ALIASES (for backward compatibility)
# ============================================================================
# Si vous vouliez garder les anciens noms pour la rétrocompatibilité
UniversalMLAgent = ProductionUniversalMLAgent
MLPipelineResult = ProductionMLPipelineResult
KnowledgeBase = ProductionKnowledgeBase
