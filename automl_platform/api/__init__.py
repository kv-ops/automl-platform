"""
AutoML Platform API Module

This module contains API-specific components for the AutoML platform including:
- Infrastructure management and multi-tenant isolation
- Billing and subscription management
- Data connectors for external data sources
- Streaming processing capabilities
- LLM endpoints for AI-powered features

These modules are designed to be used with the main AutoML platform to provide
enterprise-grade features and integrations.
"""

__version__ = "2.0.0"

# API-specific imports for convenience
try:
    from .billing import BillingManager, UsageTracker, PlanType, BillingPeriod
    from .infrastructure import TenantManager, SecurityManager, DeploymentManager
    from .connectors import ConnectorFactory, ConnectionConfig
    from .streaming import StreamingOrchestrator, StreamConfig, MLStreamProcessor
    from .llm_endpoints import router as llm_router
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(
        f"Some API modules could not be imported: {e}. "
        "Please ensure all dependencies are installed.",
        ImportWarning
    )

# Available components when importing from api
__all__ = [
    # Billing
    "BillingManager",
    "UsageTracker", 
    "PlanType",
    "BillingPeriod",
    
    # Infrastructure
    "TenantManager",
    "SecurityManager",
    "DeploymentManager",
    
    # Connectors
    "ConnectorFactory",
    "ConnectionConfig",
    
    # Streaming
    "StreamingOrchestrator",
    "StreamConfig",
    "MLStreamProcessor",
    
    # LLM
    "llm_router",
]

# API module information
API_INFO = {
    "name": "automl_platform.api",
    "version": __version__,
    "description": "API components for enterprise AutoML features",
    "components": {
        "billing": "Subscription and usage management",
        "infrastructure": "Multi-tenant architecture and security",
        "connectors": "External data source integrations",
        "streaming": "Real-time data processing",
        "llm_endpoints": "AI-powered features and assistance"
    },
    "dependencies": {
        "required": [
            "fastapi>=0.104.0",
            "sqlalchemy>=2.0.0",
            "cryptography>=41.0.0",
            "jwt>=1.3.1"
        ],
        "optional": {
            "billing": ["stripe", "paypal"],
            "infrastructure": ["docker", "kubernetes"], 
            "connectors": [
                "snowflake-connector-python",
                "google-cloud-bigquery",
                "databricks-sql-connector",
                "psycopg2-binary",
                "pymongo"
            ],
            "streaming": ["kafka-python", "pulsar-client", "redis"],
            "llm": ["openai", "anthropic", "chromadb", "langchain"]
        }
    }
}

def get_api_info():
    """Get API module information and dependencies."""
    return API_INFO.copy()

def check_dependencies():
    """Check which optional dependencies are available."""
    available_deps = {}
    
    # Check billing dependencies
    billing_deps = []
    try:
        import stripe
        billing_deps.append("stripe")
    except ImportError:
        pass
    
    try:
        import paypalrestsdk
        billing_deps.append("paypal")
    except ImportError:
        pass
    
    available_deps["billing"] = billing_deps
    
    # Check infrastructure dependencies  
    infra_deps = []
    try:
        import docker
        infra_deps.append("docker")
    except ImportError:
        pass
        
    try:
        import kubernetes
        infra_deps.append("kubernetes")
    except ImportError:
        pass
    
    available_deps["infrastructure"] = infra_deps
    
    # Check connector dependencies
    connector_deps = []
    deps_to_check = [
        ("snowflake.connector", "snowflake"),
        ("google.cloud.bigquery", "bigquery"),
        ("databricks.sql", "databricks"),
        ("psycopg2", "postgresql"),
        ("pymongo", "mongodb")
    ]
    
    for module_name, dep_name in deps_to_check:
        try:
            __import__(module_name)
            connector_deps.append(dep_name)
        except ImportError:
            pass
    
    available_deps["connectors"] = connector_deps
    
    # Check streaming dependencies
    streaming_deps = []
    streaming_to_check = [
        ("kafka", "kafka"),
        ("pulsar", "pulsar"),
        ("redis", "redis")
    ]
    
    for module_name, dep_name in streaming_to_check:
        try:
            __import__(module_name)
            streaming_deps.append(dep_name)
        except ImportError:
            pass
    
    available_deps["streaming"] = streaming_deps
    
    # Check LLM dependencies
    llm_deps = []
    llm_to_check = [
        ("openai", "openai"),
        ("anthropic", "anthropic"), 
        ("chromadb", "chromadb"),
        ("langchain", "langchain")
    ]
    
    for module_name, dep_name in llm_to_check:
        try:
            __import__(module_name)
            llm_deps.append(dep_name)
        except ImportError:
            pass
    
    available_deps["llm"] = llm_deps
    
    return available_deps

# Module initialization
def _initialize_api():
    """Initialize API module and check for critical dependencies."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Check for critical dependencies
    missing_critical = []
    
    try:
        import fastapi
    except ImportError:
        missing_critical.append("fastapi")
    
    try:
        import sqlalchemy
    except ImportError:
        missing_critical.append("sqlalchemy")
        
    if missing_critical:
        logger.error(
            f"Critical dependencies missing for API module: {missing_critical}. "
            "Please install required dependencies."
        )
    else:
        logger.info("AutoML Platform API module initialized successfully")
    
    return len(missing_critical) == 0

# Initialize on import
_api_initialized = _initialize_api()
