"""
Unified Configuration for OpenAI SDK + Anthropic Claude SDK Agents
Provides centralized cost control, feature flags, and dual-LLM management
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import json
from pathlib import Path
import logging

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
    
    Architecture Philosophy:
    - OpenAI: Data-intensive operations (profiling, technical cleaning)
    - Claude: Strategic reasoning (validation, orchestration, config generation)
    - Hybrid: Best of both worlds (validator uses both)
    """
    
    # ============================================================================
    # API KEYS - Both providers
    # ============================================================================
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    
    # ============================================================================
    # OPENAI CONFIGURATION
    # ============================================================================
    openai_model: str = "gpt-4-1106-preview"  # Supports Code Interpreter
    openai_enable_web_search: bool = True
    openai_enable_file_operations: bool = True
    openai_max_iterations: int = 3
    openai_timeout_seconds: int = 300
    
    # OpenAI agents (data-intensive operations)
    openai_agents: List[str] = field(default_factory=lambda: [
        "profiler",      # Stats computation
        "cleaner"        # Technical transformations
    ])
    
    # ============================================================================
    # CLAUDE CONFIGURATION
    # ============================================================================
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 4000
    claude_temperature: float = 0.3
    claude_timeout_seconds: int = 120
    
    # Claude agents (strategic reasoning)
    claude_agents: List[str] = field(default_factory=lambda: [
        "controller",           # Quality validation
        "orchestrator",         # Strategic decisions
        "config_generator",     # Algorithm selection
        "context_detector"      # ML problem detection
    ])
    
    # ============================================================================
    # HYBRID CONFIGURATION
    # ============================================================================
    hybrid_agents: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "validator": {
            "web_search": "openai",      # OpenAI for web search
            "reasoning": "claude",        # Claude for validation logic
            "fallback": "openai"          # If Claude unavailable
        }
    })
    
    # ============================================================================
    # FEATURE FLAGS - Fine-grained control
    # ============================================================================
    enable_claude: bool = True  # Global Claude enable/disable
    enable_openai: bool = True  # Global OpenAI enable/disable
    
    # Agent-specific feature flags
    agent_features: Dict[str, Dict[str, bool]] = field(default_factory=lambda: {
        "profiler": {
            "use_openai": True,
            "use_claude": False,  # Profiler = pure OpenAI
            "enable_caching": True
        },
        "validator": {
            "use_openai": True,
            "use_claude": True,   # Validator = hybrid
            "enable_caching": True
        },
        "cleaner": {
            "use_openai": True,
            "use_claude": False,  # Cleaner = pure OpenAI
            "enable_caching": True
        },
        "controller": {
            "use_openai": False,
            "use_claude": True,   # Controller = pure Claude
            "enable_caching": True
        },
        "orchestrator": {
            "use_openai": False,
            "use_claude": True,   # Orchestrator = pure Claude
            "enable_caching": False
        },
        "config_generator": {
            "use_openai": False,
            "use_claude": True,   # Config = pure Claude
            "enable_caching": True
        },
        "context_detector": {
            "use_openai": False,
            "use_claude": True,   # Context = pure Claude
            "enable_caching": True
        }
    })
    
    # ============================================================================
    # COST MANAGEMENT
    # ============================================================================
    track_usage: bool = True
    track_usage_per_sdk: bool = True  # Separate OpenAI vs Claude tracking
    
    # Cost limits per SDK
    max_cost_openai: float = 3.00    # Max $3 per dataset for OpenAI
    max_cost_claude: float = 2.00    # Max $2 per dataset for Claude
    max_cost_total: float = 5.00     # Total cap
    
    # Cost tracking
    cost_tracking: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "openai": {"total": 0.0, "by_agent": {}},
        "claude": {"total": 0.0, "by_agent": {}},
        "total": 0.0
    })
    
    # ============================================================================
    # ASSISTANT/THREAD IDS (persisted across runs)
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
    # LOGGING
    # ============================================================================
    log_level: str = "INFO"
    log_file: Optional[str] = "./logs/agents.log"
    log_llm_usage: bool = True  # Log every LLM call
    
    # ============================================================================
    # RETRY CONFIGURATION
    # ============================================================================
    max_retries: int = 3
    retry_delay: int = 2  # seconds
    exponential_backoff: bool = True
    
    # ============================================================================
    # OUTPUT CONFIGURATION
    # ============================================================================
    output_dir: str = "./agent_outputs"
    save_reports: bool = True
    save_yaml_config: bool = True
    
    # ============================================================================
    # WEB SEARCH CONFIGURATION (for validator)
    # ============================================================================
    web_search_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_results": 10,
        "search_engines": ["google", "bing"],
        "cache_ttl": 3600,  # 1 hour
        "sector_keywords": {
            "finance": ["IFRS", "Basel", "risk management", "compliance"],
            "sante": ["HL7", "ICD-10", "FHIR", "medical coding"],
            "retail": ["SKU", "UPC", "barcode", "product classification"],
            "industrie": ["ISO", "quality standards", "manufacturing codes"],
        }
    })
    
    # ============================================================================
    # THREAD CONFIGURATION (for OpenAI Assistants)
    # ============================================================================
    thread_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_messages": 100,
        "auto_truncate": True,
        "preserve_context": True
    })
    
    # ============================================================================
    # TOOLS CONFIGURATION PER AGENT
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
        features = self.agent_features.get(agent_type, {})
        return features.get("use_claude", False)
    
    def should_use_openai(self, agent_type: str) -> bool:
        """Check if agent should use OpenAI"""
        if not self.enable_openai:
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
        """Track LLM usage per agent and provider"""
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
        
        if provider == "openai" and self.cost_tracking["openai"]["total"] > self.max_cost_openai:
            logger.warning(f"⚠️ OpenAI cost limit exceeded")
        
        if provider == "claude" and self.cost_tracking["claude"]["total"] > self.max_cost_claude:
            logger.warning(f"⚠️ Claude cost limit exceeded")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of costs by provider"""
        return {
            "openai": {
                "total": self.cost_tracking["openai"]["total"],
                "by_agent": self.cost_tracking["openai"].get("by_agent", {}),
                "limit": self.max_cost_openai,
                "remaining": max(0, self.max_cost_openai - self.cost_tracking["openai"]["total"])
            },
            "claude": {
                "total": self.cost_tracking["claude"]["total"],
                "by_agent": self.cost_tracking["claude"].get("by_agent", {}),
                "limit": self.max_cost_claude,
                "remaining": max(0, self.max_cost_claude - self.cost_tracking["claude"]["total"])
            },
            "total": {
                "cost": self.cost_tracking["total"],
                "limit": self.max_cost_total,
                "remaining": max(0, self.max_cost_total - self.cost_tracking["total"])
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
        """Validate configuration"""
        errors = []
        
        if self.enable_openai and any(self.should_use_openai(agent) for agent in self.openai_agents):
            if not self.openai_api_key:
                errors.append("OpenAI API key is required")
        
        if self.enable_claude and any(self.should_use_claude(agent) for agent in self.claude_agents):
            if not self.anthropic_api_key:
                errors.append("Anthropic API key is required")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Configuration validated successfully")
        
        return True
    
    def save_assistant_id(self, agent_type: AgentType, assistant_id: str):
        """Save assistant ID for reuse"""
        self.assistant_ids[agent_type.value] = assistant_id
        ids_file = Path(self.cache_dir) / "assistant_ids.json"
        with open(ids_file, "w") as f:
            json.dump(self.assistant_ids, f)
    
    def load_assistant_ids(self):
        """Load saved assistant IDs"""
        ids_file = Path(self.cache_dir) / "assistant_ids.json"
        if ids_file.exists():
            with open(ids_file, "r") as f:
                self.assistant_ids = json.load(f)
    
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
        """Convert to dictionary for serialization"""
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
        }
