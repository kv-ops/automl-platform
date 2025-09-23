"""
Configuration for OpenAI SDK Agents
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import json
from pathlib import Path


class AgentType(Enum):
    """Types of agents in the system"""
    PROFILER = "profiler"
    VALIDATOR = "validator"
    CLEANER = "cleaner"
    CONTROLLER = "controller"


@dataclass
class AgentConfig:
    """Configuration for OpenAI SDK agents"""
    
    # OpenAI API Configuration
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-4-1106-preview"  # Supports Code Interpreter
    enable_web_search: bool = True
    enable_file_operations: bool = True
    max_iterations: int = 3
    timeout_seconds: int = 300
    
    # Assistant IDs (persisted across runs)
    assistant_ids: Dict[str, str] = field(default_factory=dict)
    
    # User context for sector validation
    user_context: Dict = field(default_factory=lambda: {
        "secteur_activite": None,
        "target_variable": None,
        "contexte_metier": None,
        "language": "fr"  # Support French prompts
    })
    
    # Processing configuration
    chunk_size_mb: int = 10  # For large dataset handling
    enable_caching: bool = True
    cache_dir: str = "./cache/agents"
    
    # Cost management
    max_cost_per_dataset: float = 5.00
    track_usage: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "./logs/agents.log"
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: int = 2  # seconds
    exponential_backoff: bool = True
    
    # Output configuration
    output_dir: str = "./agent_outputs"
    save_reports: bool = True
    save_yaml_config: bool = True
    
    # Web search configuration (for validator)
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
    
    # Thread configuration for OpenAI Assistants
    thread_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_messages": 100,
        "auto_truncate": True,
        "preserve_context": True
    })
    
    # Tools configuration per agent
    agent_tools: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {
        "profiler": [{"type": "code_interpreter"}],
        "validator": [{"type": "code_interpreter"}, {"type": "function", "function": {
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
        }}],
        "cleaner": [{"type": "code_interpreter"}, {"type": "file_search"}],
        "controller": [{"type": "code_interpreter"}]
    })
    
    def get_agent_tools(self, agent_type: AgentType) -> List[Dict[str, Any]]:
        """Get tools configuration for specific agent"""
        return self.agent_tools.get(agent_type.value, [])
    
    def get_sector_keywords(self, sector: str) -> List[str]:
        """Get sector-specific keywords for web search"""
        return self.web_search_config["sector_keywords"].get(sector, [])
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Create necessary directories
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        return True
    
    def save_assistant_id(self, agent_type: AgentType, assistant_id: str):
        """Save assistant ID for reuse"""
        self.assistant_ids[agent_type.value] = assistant_id
        
        # Persist to file
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "openai_api_key": "***",  # Masked for security
            "model": self.model,
            "enable_web_search": self.enable_web_search,
            "enable_file_operations": self.enable_file_operations,
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "user_context": self.user_context,
            "chunk_size_mb": self.chunk_size_mb,
            "max_cost_per_dataset": self.max_cost_per_dataset,
            "web_search_config": self.web_search_config
        }
