"""
Unified Configuration for OpenAI SDK + Anthropic Claude SDK Agents
PRODUCTION-READY: Secure, with circuit breakers and complete cost tracking
ENHANCED: Support for hybrid retail mode and local/agent arbitration
FINALIZED: All thresholds configurable
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
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
    - Hybrid mode support for retail
    - Fully configurable thresholds
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
    claude_model: str = "claude-3-opus-20240229"
    claude_max_tokens: int = 4000
    claude_temperature: float = 0.3
    claude_timeout_seconds: int = 120
    
    claude_agents: List[str] = field(default_factory=lambda: [
        "controller", "orchestrator", "config_generator", "context_detector"
    ])
    
    # ============================================================================
    # HYBRID CONFIGURATION - ENHANCED FOR RETAIL
    # ============================================================================
    hybrid_agents: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "validator": {
            "web_search": "openai",
            "reasoning": "claude",
            "fallback": "openai"
        }
    })
    
    # Hybrid retail mode configuration
    enable_hybrid_mode: bool = True
    hybrid_mode_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "missing_threshold": 0.35,  # 35% missing triggers agent
        "outlier_threshold": 0.10,   # 10% outliers triggers agent
        "quality_score_threshold": 70,  # Below 70 triggers agent
        "complexity_threshold": 0.8,    # Complexity score > 0.8 triggers agent
        "cost_threshold": 1.0           # Max cost per decision
    })
    
    # Configurable thresholds for data quality - FINALIZED
    data_quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "missing_warning_threshold": 0.35,    # 35% missing is warning level
        "missing_critical_threshold": 0.50,   # 50% missing is critical
        "outlier_warning_threshold": 0.05,    # 5% outliers is warning
        "outlier_critical_threshold": 0.15,   # 15% outliers is critical
        "high_cardinality_threshold": 0.90,   # 90% unique values
        "correlation_high_threshold": 0.95,   # 95% correlation suggests leakage
        "imbalance_severe_threshold": 20,     # 20:1 class ratio
        # Production readiness thresholds
        "quality_score_threshold": 93,        # Minimum quality score for production
        "production_missing_threshold": 0.05, # Max 5% missing for production
        "gs1_compliance_threshold": 0.98      # 98% GS1 compliance for production
    })
    
    # Retail-specific rules (0 handled contextually)
    retail_rules: Dict[str, Any] = field(default_factory=lambda: {
        "sentinel_values": [-999, -1, 9999],  # 0 removed - handled contextually
        "stock_zero_acceptable": True,
        "price_negative_critical": True,
        "sku_format_strict": True,
        "gs1_compliance_required": True,
        "gs1_compliance_target": 0.98,  # 98% target
        "category_imputation": "by_category",
        "price_imputation": "median_by_category"
    })
    
    # ============================================================================
    # FEATURE FLAGS
    # ============================================================================
    enable_claude: bool = True
    enable_openai: bool = True
    
    # Hybrid mode activation
    prefer_local_when_possible: bool = True
    
    agent_features: Dict[str, Dict[str, bool]] = field(default_factory=lambda: {
        "profiler": {"use_openai": True, "use_claude": False, "enable_caching": True, "use_hybrid": True},
        "validator": {"use_openai": True, "use_claude": True, "enable_caching": True, "use_hybrid": True},
        "cleaner": {"use_openai": True, "use_claude": False, "enable_caching": True, "use_hybrid": True},
        "controller": {"use_openai": False, "use_claude": True, "enable_caching": True, "use_hybrid": True},
        "orchestrator": {"use_openai": False, "use_claude": True, "enable_caching": False, "use_hybrid": True},
        "config_generator": {"use_openai": False, "use_claude": True, "enable_caching": True, "use_hybrid": False},
        "context_detector": {"use_openai": False, "use_claude": True, "enable_caching": True, "use_hybrid": False}
    })
    
    # ============================================================================
    # COST MANAGEMENT - Enhanced with hybrid tracking
    # ============================================================================
    track_usage: bool = True
    track_usage_per_sdk: bool = True
    track_hybrid_decisions: bool = True
    
    max_cost_openai: float = 3.00
    max_cost_claude: float = 2.00
    max_cost_total: float = 5.00
    max_cost_per_decision: float = 0.10  # Per-decision cost limit
    
    cost_tracking: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "openai": {"total": 0.0, "by_agent": {}},
        "claude": {"total": 0.0, "by_agent": {}},
        "hybrid_local": {"total": 0.0, "count": 0},
        "hybrid_agent": {"total": 0.0, "count": 0},
        "total": 0.0
    })
    
    # Hybrid decision tracking
    hybrid_decision_log: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    
    # ============================================================================
    # CIRCUIT BREAKERS - For resilience
    # ============================================================================
    enable_circuit_breakers: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    
    _circuit_breakers: Dict[str, CircuitBreaker] = field(default_factory=dict, init=False, repr=False)
    
    # ============================================================================
    # PERFORMANCE MONITORING
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
    # USER CONTEXT - ENHANCED FOR RETAIL
    # ============================================================================
    user_context: Dict = field(default_factory=lambda: {
        "secteur_activite": None,
        "target_variable": None,
        "contexte_metier": None,
        "language": "fr",
        "business_rules": {},
        "compliance_requirements": []
    })
    
    # ============================================================================
    # PROCESSING CONFIGURATION
    # ============================================================================
    chunk_size: int = 100000
    enable_caching: bool = True
    cache_dir: str = "./cache/agents"
    
    # ============================================================================
    # LOGGING - Secure
    # ============================================================================
    log_level: str = "INFO"
    log_file: Optional[str] = "./logs/agents.log"
    log_llm_usage: bool = True
    log_sensitive_data: bool = False  # Control sensitive data logging
    log_hybrid_decisions: bool = True  # Log hybrid arbitration
    
    # ============================================================================
    # OUTPUT CONFIGURATION
    # ============================================================================
    output_dir: str = "./agent_outputs"
    save_reports: bool = True
    save_yaml_config: bool = True
    yaml_config_path: Optional[str] = "./configs/cleaning_config.yaml"
    
    # ============================================================================
    # WEB SEARCH CONFIGURATION - ENHANCED FOR RETAIL
    # ============================================================================
    web_search_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_results": 10,
        "search_engines": ["google", "bing"],
        "cache_ttl": 3600,
        "sector_keywords": {
            "finance": ["IFRS", "Basel", "risk management", "compliance"],
            "sante": ["HL7", "ICD-10", "FHIR", "medical coding"],
            "retail": ["SKU", "UPC", "barcode", "product classification", "GS1", "inventory"],
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
    # HYBRID MODE METHODS
    # ============================================================================
    
    def should_use_agent(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if agent should be used based on context and thresholds.
        
        Returns:
            Tuple of (should_use_agent, reason)
        """
        if not self.enable_hybrid_mode:
            return True, "Hybrid mode disabled"
        
        # Check cost constraints first
        if self.cost_tracking["total"] >= self.max_cost_total:
            return False, "Cost limit exceeded"
        
        # Calculate decision score
        score = 0.0
        reasons = []
        
        # Missing data check
        missing_ratio = context.get("missing_ratio", 0)
        if missing_ratio > self.hybrid_mode_thresholds["missing_threshold"]:
            score += 0.3
            reasons.append(f"High missing ratio: {missing_ratio:.1%}")
        
        # Quality score check
        quality_score = context.get("quality_score", 100)
        if quality_score < self.hybrid_mode_thresholds["quality_score_threshold"]:
            score += 0.3
            reasons.append(f"Low quality score: {quality_score:.1f}")
        
        # Complexity check
        complexity = context.get("complexity_score", 0)
        if complexity > self.hybrid_mode_thresholds["complexity_threshold"]:
            score += 0.2
            reasons.append(f"High complexity: {complexity:.2f}")
        
        # Retail-specific checks
        if self.user_context.get("secteur_activite") == "retail":
            if context.get("has_sentinel_values"):
                score += 0.1
                reasons.append("Sentinel values detected")
            if context.get("has_negative_prices"):
                score += 0.2
                reasons.append("Negative prices found")
        
        # Make decision
        use_agent = score > 0.5
        reason = "; ".join(reasons) if reasons else "Normal conditions"
        
        # Log decision
        if self.log_hybrid_decisions:
            self.hybrid_decision_log.append({
                "timestamp": os.times(),
                "use_agent": use_agent,
                "score": score,
                "reason": reason,
                "context": context
            })
        
        return use_agent, reason
    
    def get_retail_rules(self, rule_type: str) -> Any:
        """Get retail-specific rules."""
        return self.retail_rules.get(rule_type)
    
    def get_quality_threshold(self, threshold_name: str) -> float:
        """Get data quality threshold by name."""
        return self.data_quality_thresholds.get(threshold_name, 0.0)
    
    def is_sentinel_value(self, value: Any, column_name: str = "") -> bool:
        """
        Check if value is a sentinel value in retail context.
        Considers column context (e.g., stock columns where 0 is valid).
        """
        if self.user_context.get("secteur_activite") != "retail":
            return False
        
        sentinels = self.retail_rules.get("sentinel_values", [])
        
        # Check if it's a stock/quantity column where 0 is acceptable
        stock_keywords = ['stock', 'quantity', 'qty', 'inventory', 'count', 'units']
        is_stock_column = any(keyword in column_name.lower() for keyword in stock_keywords)
        
        # If it's a stock column and value is 0, it's not a sentinel
        if is_stock_column and value == 0:
            return False
        
        return value in sentinels
    
    def get_hybrid_stats(self) -> Dict[str, Any]:
        """Get statistics about hybrid decision making."""
        if not self.hybrid_decision_log:
            return {"message": "No hybrid decisions logged"}
        
        total = len(self.hybrid_decision_log)
        agent_used = sum(1 for d in self.hybrid_decision_log if d["use_agent"])
        local_used = total - agent_used
        
        return {
            "total_decisions": total,
            "agent_decisions": agent_used,
            "local_decisions": local_used,
            "agent_percentage": (agent_used / total * 100) if total > 0 else 0,
            "local_percentage": (local_used / total * 100) if total > 0 else 0,
            "cost_savings": self.cost_tracking["hybrid_local"]["total"],
            "last_10_decisions": self.hybrid_decision_log[-10:]
        }
    
    # ============================================================================
    # CIRCUIT BREAKER METHODS
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
        
        # Track hybrid decisions separately
        if self.enable_hybrid_mode and provider in ["hybrid_local", "hybrid_agent"]:
            self.cost_tracking[provider]["total"] += cost
            self.cost_tracking[provider]["count"] += 1
        elif provider in self.cost_tracking:
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
        summary = {
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
        
        # Add hybrid statistics if enabled
        if self.enable_hybrid_mode:
            summary["hybrid"] = {
                "local_decisions": self.cost_tracking.get("hybrid_local", {}).get("count", 0),
                "agent_decisions": self.cost_tracking.get("hybrid_agent", {}).get("count", 0),
                "cost_savings": self.cost_tracking.get("hybrid_local", {}).get("total", 0)
            }
        
        return summary
    
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
        
        # Validate hybrid thresholds
        if self.enable_hybrid_mode:
            for key, value in self.hybrid_mode_thresholds.items():
                if value < 0 or value > 1 and key != "cost_threshold":
                    warnings.append(f"Hybrid threshold {key} should be between 0 and 1")
        
        # Validate data quality thresholds
        thresholds_to_check = [
            "missing_warning_threshold", "missing_critical_threshold",
            "outlier_warning_threshold", "outlier_critical_threshold",
            "high_cardinality_threshold", "correlation_high_threshold",
            "production_missing_threshold", "gs1_compliance_threshold"
        ]
        
        for key in thresholds_to_check:
            value = self.data_quality_thresholds.get(key, 0)
            if key == "quality_score_threshold":
                if value < 0 or value > 100:
                    warnings.append(f"Quality score threshold should be between 0 and 100")
            elif key != "imbalance_severe_threshold":
                if value < 0 or value > 1:
                    warnings.append(f"Data quality threshold {key} should be between 0 and 1")
        
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
            "hybrid_local": {"total": 0.0, "count": 0},
            "hybrid_agent": {"total": 0.0, "count": 0},
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
            "enable_hybrid_mode": self.enable_hybrid_mode,
            "retail_rules": self.retail_rules,
            "data_quality_thresholds": self.data_quality_thresholds,
            "cost_summary": self.get_cost_summary() if self.track_usage_per_sdk else None,
            "circuit_breakers": self.get_circuit_breaker_status() if self.enable_circuit_breakers else None,
            "hybrid_stats": self.get_hybrid_stats() if self.enable_hybrid_mode else None
        }
    
    def __repr__(self) -> str:
        """Safe representation without API keys"""
        return (
            f"AgentConfig("
            f"openai={'set' if self.openai_api_key else 'none'}, "
            f"claude={'set' if self.anthropic_api_key else 'none'}, "
            f"hybrid={self.enable_hybrid_mode}, "
            f"cost=${self.cost_tracking['total']:.4f}"
            f")"
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
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
        
        # Check hybrid mode health
        if self.enable_hybrid_mode:
            hybrid_stats = self.get_hybrid_stats()
            if hybrid_stats.get("total_decisions", 0) > 100:
                agent_pct = hybrid_stats.get("agent_percentage", 0)
                if agent_pct > 80:
                    status["issues"].append(f"High agent usage in hybrid mode: {agent_pct:.1f}%")
        
        return status
