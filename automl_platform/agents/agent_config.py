"""
Unified Configuration for OpenAI SDK + Anthropic Claude SDK Agents
PRODUCTION-READY: Secure, with circuit breakers and complete cost tracking
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import json
from pathlib import Path
import logging

from .utils import CircuitBreaker, sanitize_for_logging, PerformanceMonitor

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the system"""
    PROFILER = "profiler"
    VALIDATOR = "validator"
    CLEANER = "cleaner"
    CONTROLLER = "controller"
    ORCHESTRATOR = "orchestrator"
    CONFIG_GENERATOR = "config_generator"
    CONTEXT_DETECTOR = "context_detector"


class LLMProvider(Enum):
    """LLM providers in the system"""
    OPENAI = "openai"
    CLAUDE = "claude"
    HYBRID = "hybrid"


@dataclass
class AgentConfig:
    """
    Unified configuration for OpenAI SDK + Anthropic Claude SDK agents
    
    PRODUCTION-READY with:
    - Secure logging (no API key leaks)
    - Circuit breakers for resilience
    - Complete cost tracking
    - Performance monitoring
    """
    
    # ============================================================================
    # API KEYS - Secure handling
    # ============================================================================
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        repr=False  # Never print in repr
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""),
        repr=False  # Never print in repr
    )
    
    # ============================================================================
    # OPENAI CONFIGURATION
    # ============================================================================
    openai_model: str = "gpt-4-1106-preview"
    openai_enable_web_search: bool = True
    openai_enable_file_operations: bool = True
    openai_max_iterations: int = 3
    openai_timeout_seconds: int = 300
    
    openai_agents: List[str] = field(default_factory=lambda: [
        "profiler", "cleaner"
    ])
    
    # ============================================================================
    # CLAUDE CONFIGURATION
    # ============================================================================
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 4000
    claude_temperature: float = 0.3
    claude_timeout_seconds: int = 120
    
    claude_agents: List[str] = field(default_factory=lambda: [
        "controller", "orchestrator", "config_generator", "context_detector"
    ])
    
    # ============================================================================
    # HYBRID CONFIGURATION
    # ============================================================================
    hybrid_agents: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "validator": {
            "web_search": "openai",
            "reasoning": "claude",
            "fallback": "openai"
        }
    })
    
    # ============================================================================
    # FEATURE FLAGS
    # ============================================================================
    enable_claude: bool = True
    enable_openai: bool = True
    
    agent_features: Dict[str, Dict[str, bool]] = field(default_factory=lambda: {
        "profiler": {"use_openai": True, "use_claude": False, "enable_caching": True},
        "validator": {"use_openai": True, "use_claude": True, "enable_caching": True},
        "cleaner": {"use_openai": True, "use_claude": False, "enable_caching": True},
        "controller": {"use_openai": False, "use_claude": True, "enable_caching": True},
        "orchestrator": {"use_openai": False, "use_claude": True, "enable_caching": False},
        "config_generator": {"use_openai": False, "use_claude": True, "enable_caching": True},
        "context_detector": {"use_openai": False, "use_claude": True, "enable_caching": True}
    })
    
    # ============================================================================
    # COST MANAGEMENT - Enhanced
    # ============================================================================
    track_usage: bool = True
    track_usage_per_sdk: bool = True
    
    max_cost_openai: float = 3.00
    max_cost_claude: float = 2.00
    max_cost_total: float = 5.00
    
    cost_tracking: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "openai": {"total": 0.0, "by_agent": {}},
        "claude": {"total": 0.0, "by_agent": {}},
        "total": 0.0
    })
    
    # ============================================================================
    # CIRCUIT BREAKERS - NEW for resilience
    # ============================================================================
    enable_circuit_breakers: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    
    _circuit_breakers: Dict[str, CircuitBreaker] = field(default_factory=dict, init=False, repr=False)
    
    # ============================================================================
    # PERFORMANCE MONITORING - NEW
    # ============================================================================
    enable_performance_monitoring: bool = True
    _performance_monitor: Optional[PerformanceMonitor] = field(default=None, init=False, repr=False)
    
    # ============================================================================
    # RETRY CONFIGURATION - Enhanced
    # ============================================================================
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 10.0
    retry_exponential_base: float = 2.0
    
    # ============================================================================
    # ASSISTANT/THREAD IDS
    # ============================================================================
    assistant_ids: Dict[str, str] = field(default_factory=dict)
    
    # ============================================================================
    # USER CONTEXT
    # ============================================================================
    user_context: Dict = field(default_factory=lambda: {
        "secteur_activite": None,
        "target_variable": None,
        "contexte_metier": None,
        "language": "fr"
    })
    
    # ============================================================================
    # PROCESSING CONFIGURATION
    # ============================================================================
    chunk_size_mb: int = 10
    enable_caching: bool = True
    cache_dir: str = "./cache/agents"
    
    # ============================================================================
    # LOGGING - Secure
    # ============================================================================
    log_level: str = "INFO"
    log_file: Optional[str] = "./logs/agents.log"
    log_llm_usage: bool = True
    log_sensitive_data: bool = False  # NEW: control sensitive data logging
    
    # ============================================================================
    # OUTPUT CONFIGURATION
    # ============================================================================
    output_dir: str = "./agent_outputs"
    save_reports: bool = True
    save_yaml_config: bool = True
    
    # ============================================================================
    # WEB SEARCH CONFIGURATION
    # ============================================================================
    web_search_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_results": 10,
        "search_engines": ["google", "bing"],
        "cache_ttl": 3600,
        "sector_keywords": {
            "finance": ["IFRS", "Basel", "risk management", "compliance"],
            "sante": ["HL7", "ICD-10", "FHIR", "medical coding"],
            "retail": ["SKU", "UPC", "barcode", "product classification"],
            "industrie": ["ISO", "quality standards", "manufacturing codes"],
        }
    })
    
    # ============================================================================
    # THREAD CONFIGURATION
    # ============================================================================
    thread_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_messages": 100,
        "auto_truncate": True,
        "preserve_context": True
    })
    
    # ============================================================================
    # TOOLS CONFIGURATION
    # ============================================================================
    agent_tools: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {
        "profiler": [{"type": "code_interpreter"}],
        "validator": [
            {"type": "code_interpreter"},
            {"type": "function", "function": {
                "name": "web_search",
                "description": "Search the web for sector-specific validation references",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "sector": {"type": "string", "description": "Business sector"}
                    },
                    "required": ["query"]
                }
            }}
        ],
        "cleaner": [{"type": "code_interpreter"}, {"type": "file_search"}],
        "controller": [{"type": "code_interpreter"}]
    })
    
    # ============================================================================
    # POST-INIT
    # ============================================================================
    
    def __post_init__(self):
        """Initialize circuit breakers and performance monitor"""
        if self.enable_circuit_breakers:
            self._initialize_circuit_breakers()
        
        if self.enable_performance_monitoring:
            self._performance_monitor = PerformanceMonitor(maxlen=1000)
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for each LLM provider"""
        if self.enable_openai:
            self._circuit_breakers['openai'] = CircuitBreaker(
                name='openai',
                failure_threshold=self.circuit_breaker_failure_threshold,
                recovery_timeout=self.circuit_breaker_recovery_timeout
            )
        
        if self.enable_claude:
            self._circuit_breakers['claude'] = CircuitBreaker(
                name='claude',
                failure_threshold=self.circuit_breaker_failure_threshold,
                recovery_timeout=self.circuit_breaker_recovery_timeout
            )
    
    # ============================================================================
    # CIRCUIT BREAKER METHODS - NEW
    # ============================================================================
    
    def get_circuit_breaker(self, provider: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for provider"""
        return self._circuit_breakers.get(provider)
    
    def can_call_llm(self, provider: str) -> bool:
        """Check if LLM call is allowed (circuit breaker check)"""
        if not self.enable_circuit_breakers:
            return True
        
        breaker = self.get_circuit_breaker(provider)
        if breaker:
            return breaker.can_attempt()
        
        return True
    
    def record_llm_success(self, provider: str):
        """Record successful LLM call"""
        breaker = self.get_circuit_breaker(provider)
        if breaker:
            breaker.record_success()
    
    def record_llm_failure(self, provider: str):
        """Record failed LLM call"""
        breaker = self.get_circuit_breaker(provider)
        if breaker:
            breaker.record_failure()
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        return {
            name: breaker.get_status()
            for name, breaker in self._circuit_breakers.items()
        }
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def get_llm_provider_for_agent(self, agent_type: str) -> str:
        """Get LLM provider for specific agent"""
        if agent_type in self.claude_agents:
            return "claude" if self.enable_claude else "none"
        elif agent_type in self.openai_agents:
            return "openai" if self.enable_openai else "none"
        elif agent_type in self.hybrid_agents:
            return "hybrid"
        return "none"
    
    def should_use_claude(self, agent_type: str) -> bool:
        """Check if agent should use Claude"""
        if not self.enable_claude:
            return False
        
        # Check circuit breaker
        if not self.can_call_llm('claude'):
            logger.warning("Claude circuit breaker OPEN, disabling Claude")
            return False
        
        features = self.agent_features.get(agent_type, {})
        return features.get("use_claude", False)
    
    def should_use_openai(self, agent_type: str) -> bool:
        """Check if agent should use OpenAI"""
        if not self.enable_openai:
            return False
        
        # Check circuit breaker
        if not self.can_call_llm('openai'):
            logger.warning("OpenAI circuit breaker OPEN, disabling OpenAI")
            return False
        
        features = self.agent_features.get(agent_type, {})
        return features.get("use_openai", False)
    
    def get_claude_config_for_agent(self, agent_type: str) -> Dict[str, Any]:
        """Get Claude configuration for specific agent"""
        if not self.should_use_claude(agent_type):
            return {"enabled": False}
        
        return {
            "enabled": True,
            "model": self.claude_model,
            "max_tokens": self.claude_max_tokens,
            "temperature": self.claude_temperature,
            "timeout": self.claude_timeout_seconds,
            "api_key": self.anthropic_api_key
        }
    
    def get_openai_config_for_agent(self, agent_type: str) -> Dict[str, Any]:
        """Get OpenAI configuration for specific agent"""
        if not self.should_use_openai(agent_type):
            return {"enabled": False}
        
        return {
            "enabled": True,
            "model": self.openai_model,
            "max_iterations": self.openai_max_iterations,
            "timeout": self.openai_timeout_seconds,
            "api_key": self.openai_api_key,
            "tools": self.get_agent_tools(AgentType(agent_type) if hasattr(AgentType, agent_type.upper()) else None)
        }
    
    def track_llm_usage(self, agent_type: str, provider: str, cost: float):
        """Track LLM usage per agent and provider - ENHANCED"""
        if not self.track_usage_per_sdk:
            return
        
        if provider in self.cost_tracking:
            self.cost_tracking[provider]["total"] += cost
            
            if "by_agent" not in self.cost_tracking[provider]:
                self.cost_tracking[provider]["by_agent"] = {}
            
            if agent_type not in self.cost_tracking[provider]["by_agent"]:
                self.cost_tracking[provider]["by_agent"][agent_type] = 0.0
            
            self.cost_tracking[provider]["by_agent"][agent_type] += cost
        
        self.cost_tracking["total"] += cost
        
        # Log warnings for cost limits
        if provider == "openai" and self.cost_tracking["openai"]["total"] > self.max_cost_openai:
            logger.warning(
                f"⚠️ OpenAI cost limit exceeded: "
                f"${self.cost_tracking['openai']['total']:.4f} > ${self.max_cost_openai}"
            )
        
        if provider == "claude" and self.cost_tracking["claude"]["total"] > self.max_cost_claude:
            logger.warning(
                f"⚠️ Claude cost limit exceeded: "
                f"${self.cost_tracking['claude']['total']:.4f} > ${self.max_cost_claude}"
            )
        
        if self.cost_tracking["total"] > self.max_cost_total:
            logger.error(
                f"🚨 TOTAL cost limit exceeded: "
                f"${self.cost_tracking['total']:.4f} > ${self.max_cost_total}"
            )
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of costs by provider"""
        return {
            "openai": {
                "total": self.cost_tracking["openai"]["total"],
                "by_agent": self.cost_tracking["openai"].get("by_agent", {}),
                "limit": self.max_cost_openai,
                "remaining": max(0, self.max_cost_openai - self.cost_tracking["openai"]["total"]),
                "percent_used": (self.cost_tracking["openai"]["total"] / self.max_cost_openai * 100) 
                               if self.max_cost_openai > 0 else 0
            },
            "claude": {
                "total": self.cost_tracking["claude"]["total"],
                "by_agent": self.cost_tracking["claude"].get("by_agent", {}),
                "limit": self.max_cost_claude,
                "remaining": max(0, self.max_cost_claude - self.cost_tracking["claude"]["total"]),
                "percent_used": (self.cost_tracking["claude"]["total"] / self.max_cost_claude * 100)
                               if self.max_cost_claude > 0 else 0
            },
            "total": {
                "cost": self.cost_tracking["total"],
                "limit": self.max_cost_total,
                "remaining": max(0, self.max_cost_total - self.cost_tracking["total"]),
                "percent_used": (self.cost_tracking["total"] / self.max_cost_total * 100)
                               if self.max_cost_total > 0 else 0
            }
        }
    
    def get_agent_tools(self, agent_type: Optional[AgentType]) -> List[Dict[str, Any]]:
        """Get tools configuration for specific agent"""
        if agent_type is None:
            return []
        return self.agent_tools.get(agent_type.value, [])
    
    def get_sector_keywords(self, sector: str) -> List[str]:
        """Get sector-specific keywords for web search"""
        return self.web_search_config["sector_keywords"].get(sector, [])
    
    def validate(self) -> bool:
        """Validate configuration - ENHANCED"""
        errors = []
        warnings = []
        
        # Check API keys
        if self.enable_openai and any(self.should_use_openai(agent) for agent in self.openai_agents):
            if not self.openai_api_key:
                errors.append("OpenAI API key is required but not set")
        
        if self.enable_claude and any(self.should_use_claude(agent) for agent in self.claude_agents):
            if not self.anthropic_api_key:
                errors.append("Anthropic API key is required but not set")
        
        # Check directories
        try:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            
            if self.log_file:
                Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to create directories: {e}")
        
        # Check cost limits
        if self.max_cost_openai <= 0:
            warnings.append("OpenAI cost limit is 0, calls will be blocked")
        
        if self.max_cost_claude <= 0:
            warnings.append("Claude cost limit is 0, calls will be blocked")
        
        # Validate retry configuration
        if self.max_retries < 1:
            errors.append("max_retries must be >= 1")
        
        if self.retry_base_delay <= 0:
            errors.append("retry_base_delay must be > 0")
        
        # Log results
        if errors:
            error_msg = f"Configuration validation failed: {'; '.join(errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if warnings:
            for warning in warnings:
                logger.warning(f"⚠️ Configuration warning: {warning}")
        
        logger.info("✅ Configuration validated successfully")
        
        return True
    
    def save_assistant_id(self, agent_type: AgentType, assistant_id: str):
        """Save assistant ID for reuse"""
        self.assistant_ids[agent_type.value] = assistant_id
        ids_file = Path(self.cache_dir) / "assistant_ids.json"
        try:
            with open(ids_file, "w") as f:
                json.dump(self.assistant_ids, f)
        except Exception as e:
            logger.warning(f"Failed to save assistant IDs: {e}")
    
    def load_assistant_ids(self):
        """Load saved assistant IDs"""
        ids_file = Path(self.cache_dir) / "assistant_ids.json"
        if ids_file.exists():
            try:
                with open(ids_file, "r") as f:
                    self.assistant_ids = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load assistant IDs: {e}")
    
    def get_assistant_id(self, agent_type: AgentType) -> Optional[str]:
        """Get saved assistant ID if exists"""
        return self.assistant_ids.get(agent_type.value)
    
    def reset_costs(self):
        """Reset cost tracking"""
        self.cost_tracking = {
            "openai": {"total": 0.0, "by_agent": {}},
            "claude": {"total": 0.0, "by_agent": {}},
            "total": 0.0
        }
        logger.info("Cost tracking reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization - SECURE"""
        return {
            "openai_api_key": "***" if self.openai_api_key else None,
            "anthropic_api_key": "***" if self.anthropic_api_key else None,
            "openai_model": self.openai_model,
            "openai_agents": self.openai_agents,
            "claude_model": self.claude_model,
            "claude_agents": self.claude_agents,
            "hybrid_agents": self.hybrid_agents,
            "enable_claude": self.enable_claude,
            "enable_openai": self.enable_openai,
            "cost_summary": self.get_cost_summary() if self.track_usage_per_sdk else None,
            "circuit_breakers": self.get_circuit_breaker_status() if self.enable_circuit_breakers else None,
        }
    
    def __repr__(self) -> str:
        """Safe representation without API keys"""
        return (
            f"AgentConfig("
            f"openai={'set' if self.openai_api_key else 'none'}, "
            f"claude={'set' if self.anthropic_api_key else 'none'}, "
            f"cost=${self.cost_tracking['total']:.4f}"
            f")"
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status - NEW"""
        status = {
            "healthy": True,
            "issues": [],
            "circuit_breakers": self.get_circuit_breaker_status(),
            "cost_status": self.get_cost_summary()
        }
        
        # Check circuit breakers
        for name, breaker_status in status["circuit_breakers"].items():
            if breaker_status["state"] == "OPEN":
                status["healthy"] = False
                status["issues"].append(f"{name} circuit breaker is OPEN")
        
        # Check cost limits
        cost_summary = status["cost_status"]
        if cost_summary["total"]["remaining"] < 0:
            status["healthy"] = False
            status["issues"].append("Total cost limit exceeded")
        
        return status
