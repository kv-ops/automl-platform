"""
UI module for AutoML Platform
=============================

This module provides a complete no-code interface and reusable UI components
for the AutoML platform, featuring:
- No-code dashboard for non-technical users
- Drag-and-drop data import
- Visual model configuration wizard
- Interactive data quality visualizations
- Real-time model training monitoring
- One-click deployment
- AI-powered chat assistant
- Automated report generation
- Multi-language support
"""

__version__ = "2.0.0"  # Upgraded for no-code capabilities
__author__ = "AutoML Platform Team"

# ============================================================================
# Metrics imports from ui_metrics.py
# ============================================================================

try:
    from .ui_metrics import (
        # Metrics
        ml_ui_sessions_total,
        ml_ui_page_views_total,
        ml_ui_api_calls_total,
        ml_ui_response_time_seconds,
        ml_ui_errors_total,
        ml_ui_feature_usage,
        
        # Helper class
        UIMetrics,
        UISession,
        
        # Helper functions
        track_streamlit_page,
        track_streamlit_action
    )
    from prometheus_client import CollectorRegistry
    import time
    
    METRICS_AVAILABLE = True
    
    # Create a registry for UI metrics
    ui_registry = CollectorRegistry()
    
except ImportError as e:
    METRICS_AVAILABLE = False
    ui_registry = None
    ml_ui_sessions_total = None
    ml_ui_page_views_total = None
    ml_ui_api_calls_total = None
    ml_ui_response_time_seconds = None
    ml_ui_errors_total = None
    ml_ui_feature_usage = None
    UIMetrics = None
    UISession = None
    track_streamlit_page = None
    track_streamlit_action = None

# ============================================================================
# Additional metrics tracking functions (wrapper functions for compatibility)
# ============================================================================

def track_ui_response_time(component, action):
    """Decorator to track UI response time (wrapper for UIMetrics)."""
    def decorator(func):
        if METRICS_AVAILABLE and UIMetrics:
            metrics = UIMetrics()
            return metrics.measure_response_time(component, action)(func)
        else:
            return func
    return decorator

def track_page_view(page, source="direct", tenant_id="default", user_role="user"):
    """Track a page view in the UI (wrapper function)."""
    if METRICS_AVAILABLE and track_streamlit_page:
        track_streamlit_page(page, tenant_id, user_role)

def track_api_call(endpoint, method, status, tenant_id="default"):
    """Track an API call from the UI (wrapper function)."""
    if METRICS_AVAILABLE and UIMetrics:
        metrics = UIMetrics(tenant_id)
        metrics.track_api_call(endpoint, method)

def track_component_render(component, status="success", tenant_id="default"):
    """Track a component render (wrapper function)."""
    if METRICS_AVAILABLE and track_streamlit_action:
        track_streamlit_action(f"{component}_render", status, tenant_id)

def start_ui_session(page="main", user_type="standard", tenant_id="default"):
    """Track the start of a UI session (wrapper function)."""
    if METRICS_AVAILABLE and UIMetrics:
        metrics = UIMetrics(tenant_id, user_type)
        metrics.increment_session()
        track_streamlit_page(page, tenant_id, user_type)

def end_ui_session(page="main", tenant_id="default", user_role="user"):
    """Track the end of a UI session (wrapper function)."""
    if METRICS_AVAILABLE and UIMetrics:
        metrics = UIMetrics(tenant_id, user_role)
        metrics.decrement_session()

# ============================================================================
# Conditional imports to handle optional dependencies
# ============================================================================

# Check if Streamlit is available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Check if Plotly is available
try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    plotly = None

# Check if Streamlit extras are available
try:
    from streamlit_option_menu import option_menu
    STREAMLIT_EXTRAS_AVAILABLE = True
except ImportError:
    STREAMLIT_EXTRAS_AVAILABLE = False
    option_menu = None

# Import components if dependencies are available
if STREAMLIT_AVAILABLE and PLOTLY_AVAILABLE:
    try:
        from .components import (
            # Data Quality Components
            DataQualityVisualizer,
            
            # Model Components
            ModelLeaderboard,
            FeatureImportanceVisualizer,
            
            # Monitoring Components
            DriftMonitor,
            TrainingProgressTracker,
            
            # Interactive Components
            ChatInterface,
            ExperimentComparator,
            
            # Alert and Report Components
            AlertsAndNotifications,
            ReportGenerator,
            
            # Advanced Visualizations
            DataFlowDiagram,
            ModelDeploymentUI,
            CustomMetrics,
            
            # Utility functions
            create_sidebar_filters,
            create_action_buttons,
            show_loading_animation
        )
        COMPONENTS_AVAILABLE = True
    except ImportError as e:
        COMPONENTS_AVAILABLE = False
        # Set all component classes to None if import fails
        DataQualityVisualizer = None
        ModelLeaderboard = None
        FeatureImportanceVisualizer = None
        DriftMonitor = None
        TrainingProgressTracker = None
        ChatInterface = None
        ExperimentComparator = None
        AlertsAndNotifications = None
        ReportGenerator = None
        DataFlowDiagram = None
        ModelDeploymentUI = None
        CustomMetrics = None
        create_sidebar_filters = None
        create_action_buttons = None
        show_loading_animation = None
else:
    COMPONENTS_AVAILABLE = False
    # Set all to None if dependencies not available
    DataQualityVisualizer = None
    ModelLeaderboard = None
    FeatureImportanceVisualizer = None
    DriftMonitor = None
    TrainingProgressTracker = None
    ChatInterface = None
    ExperimentComparator = None
    AlertsAndNotifications = None
    ReportGenerator = None
    DataFlowDiagram = None
    ModelDeploymentUI = None
    CustomMetrics = None
    create_sidebar_filters = None
    create_action_buttons = None
    show_loading_animation = None

# Import main dashboard if available
if STREAMLIT_AVAILABLE:
    try:
        # Try to import the new no-code dashboard first
        from .dashboard import main as dashboard_main, AutoMLWizard, DataConnector, SessionState
        DASHBOARD_AVAILABLE = True
        NOCODE_DASHBOARD = True
    except ImportError:
        try:
            # Fallback to old streamlit_app if dashboard.py doesn't exist
            from .streamlit_app import AutoMLDashboard
            DASHBOARD_AVAILABLE = True
            NOCODE_DASHBOARD = False
            dashboard_main = None
            AutoMLWizard = None
            DataConnector = None
            SessionState = None
        except ImportError:
            DASHBOARD_AVAILABLE = False
            NOCODE_DASHBOARD = False
            AutoMLDashboard = None
            dashboard_main = None
            AutoMLWizard = None
            DataConnector = None
            SessionState = None
else:
    DASHBOARD_AVAILABLE = False
    NOCODE_DASHBOARD = False
    AutoMLDashboard = None
    dashboard_main = None
    AutoMLWizard = None
    DataConnector = None
    SessionState = None

# ============================================================================
# No-Code specific imports and helpers
# ============================================================================

class NoCodeConfig:
    """Configuration for no-code features."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        'max_file_size_mb': 1000,
        'supported_file_types': ['csv', 'xlsx', 'xls', 'parquet', 'json'],
        'supported_databases': ['PostgreSQL', 'MySQL', 'MongoDB', 'Snowflake', 'BigQuery', 'SQL Server'],
        'auto_save_interval': 60,  # seconds
        'enable_chat_assistant': True,
        'enable_auto_ml': True,
        'default_language': 'en',
        'theme': 'light',
        'enable_telemetry': True
    }
    
    @classmethod
    def get(cls, key, default=None):
        """Get configuration value."""
        return cls.DEFAULT_CONFIG.get(key, default)
    
    @classmethod
    def update(cls, config_dict):
        """Update configuration."""
        cls.DEFAULT_CONFIG.update(config_dict)

# ============================================================================
# Dynamically build __all__ based on available components
# ============================================================================

def _build_all_list():
    """Dynamically build the __all__ list based on available components."""
    all_list = ["__version__", "NoCodeConfig"]
    
    # Always include availability flags
    all_list.extend([
        "STREAMLIT_AVAILABLE",
        "PLOTLY_AVAILABLE", 
        "STREAMLIT_EXTRAS_AVAILABLE",
        "COMPONENTS_AVAILABLE",
        "DASHBOARD_AVAILABLE",
        "NOCODE_DASHBOARD",
        "METRICS_AVAILABLE"
    ])
    
    # Include metrics if available
    if METRICS_AVAILABLE:
        all_list.extend([
            "ui_registry",
            "ml_ui_sessions_total",
            "ml_ui_page_views_total",
            "ml_ui_api_calls_total",
            "ml_ui_response_time_seconds",
            "ml_ui_errors_total",
            "ml_ui_feature_usage",
            "UIMetrics",
            "UISession",
            "track_ui_response_time",
            "track_page_view",
            "track_api_call",
            "track_component_render",
            "track_streamlit_page",
            "track_streamlit_action",
            "start_ui_session",
            "end_ui_session"
        ])
    
    # Always include helper functions (they handle unavailability gracefully)
    all_list.extend([
        "check_ui_dependencies",
        "get_ui_status",
        "launch_dashboard",
        "launch_wizard",
        "quick_start"
    ])
    
    # Add dashboard components
    if NOCODE_DASHBOARD:
        if dashboard_main is not None:
            all_list.append("dashboard_main")
        if AutoMLWizard is not None:
            all_list.append("AutoMLWizard")
        if DataConnector is not None:
            all_list.append("DataConnector")
        if SessionState is not None:
            all_list.append("SessionState")
    elif DASHBOARD_AVAILABLE and AutoMLDashboard is not None:
        all_list.append("AutoMLDashboard")
    
    # Add components only if they're available
    if COMPONENTS_AVAILABLE:
        components = [
            "DataQualityVisualizer",
            "ModelLeaderboard",
            "FeatureImportanceVisualizer",
            "DriftMonitor",
            "TrainingProgressTracker",
            "ChatInterface",
            "ExperimentComparator",
            "AlertsAndNotifications",
            "ReportGenerator",
            "DataFlowDiagram",
            "ModelDeploymentUI",
            "CustomMetrics",
            "create_sidebar_filters",
            "create_action_buttons",
            "show_loading_animation"
        ]
        
        for component in components:
            if globals().get(component) is not None:
                all_list.append(component)
    
    # Add factory functions
    all_list.extend([
        "create_data_quality_visualizer",
        "create_model_leaderboard",
        "create_chat_interface"
    ])
    
    return all_list

# ============================================================================
# Public API
# ============================================================================

__all__ = _build_all_list()

# ============================================================================
# Helper Functions
# ============================================================================

def check_ui_dependencies():
    """
    Check if all UI dependencies are installed and available.
    
    Returns:
        dict: Status of each dependency
    """
    dependencies = {
        "streamlit": STREAMLIT_AVAILABLE,
        "plotly": PLOTLY_AVAILABLE,
        "streamlit_extras": STREAMLIT_EXTRAS_AVAILABLE,
        "components": COMPONENTS_AVAILABLE,
        "dashboard": DASHBOARD_AVAILABLE,
        "nocode_dashboard": NOCODE_DASHBOARD,
        "metrics": METRICS_AVAILABLE
    }
    
    # Check for additional optional dependencies
    optional_deps = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("prometheus_client", "prometheus-client"),
        ("reportlab", "reportlab"),
        ("xlsxwriter", "xlsxwriter"),
        ("streamlit_authenticator", "streamlit-authenticator")
    ]
    
    for module_name, package_name in optional_deps:
        try:
            __import__(module_name)
            dependencies[package_name] = True
        except ImportError:
            dependencies[package_name] = False
    
    return dependencies


def get_ui_status():
    """
    Get detailed UI module status.
    
    Returns:
        dict: Detailed status information
    """
    status = {
        "version": __version__,
        "dependencies": check_ui_dependencies(),
        "components": {
            "data_quality": DataQualityVisualizer is not None,
            "model_leaderboard": ModelLeaderboard is not None,
            "feature_importance": FeatureImportanceVisualizer is not None,
            "drift_monitor": DriftMonitor is not None,
            "training_tracker": TrainingProgressTracker is not None,
            "chat_interface": ChatInterface is not None,
            "experiment_comparator": ExperimentComparator is not None,
            "alerts": AlertsAndNotifications is not None,
            "report_generator": ReportGenerator is not None,
            "data_flow": DataFlowDiagram is not None,
            "deployment_ui": ModelDeploymentUI is not None,
            "custom_metrics": CustomMetrics is not None
        },
        "dashboard_available": DASHBOARD_AVAILABLE,
        "nocode_enabled": NOCODE_DASHBOARD,
        "metrics_enabled": METRICS_AVAILABLE,
        "metrics_module": UIMetrics is not None,
        "config": NoCodeConfig.DEFAULT_CONFIG
    }
    
    # Calculate overall readiness
    total_components = len(status["components"])
    available_components = sum(status["components"].values())
    status["readiness"] = f"{available_components}/{total_components} components available"
    
    # Add feature flags
    status["features"] = {
        "wizard": AutoMLWizard is not None,
        "data_connector": DataConnector is not None,
        "session_state": SessionState is not None,
        "chat_enabled": NoCodeConfig.get('enable_chat_assistant'),
        "auto_ml_enabled": NoCodeConfig.get('enable_auto_ml')
    }
    
    return status


@track_ui_response_time("dashboard", "launch")
def launch_dashboard(config=None, mode="nocode", **kwargs):
    """
    Launch the Streamlit dashboard application.
    
    Args:
        config (dict, optional): Configuration dictionary
        mode (str): Launch mode - "nocode" (default) or "classic"
        **kwargs: Additional arguments passed to the dashboard
    
    Returns:
        None
    
    Raises:
        ImportError: If Streamlit or required dependencies are not installed
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError(
            "Streamlit is not installed. Please install it with: "
            "pip install streamlit"
        )
    
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is not installed. Please install it with: "
            "pip install plotly"
        )
    
    if not DASHBOARD_AVAILABLE:
        raise ImportError(
            "Dashboard components are not available. "
            "Please check that all required files are present."
        )
    
    # Update configuration if provided
    if config:
        NoCodeConfig.update(config)
    
    # Track dashboard launch
    tenant_id = kwargs.get('tenant_id', 'default')
    user_type = kwargs.get('user_type', 'standard')
    
    start_ui_session(page="dashboard", user_type=user_type, tenant_id=tenant_id)
    track_page_view("dashboard", "launch", tenant_id, user_type)
    
    # Import here to avoid issues if streamlit is not installed
    import sys
    import subprocess
    from pathlib import Path
    
    # Get the path to the appropriate app
    ui_module_path = Path(__file__).parent
    
    if mode == "nocode" and NOCODE_DASHBOARD:
        app_path = ui_module_path / "dashboard.py"
    else:
        app_path = ui_module_path / "streamlit_app.py"
    
    if not app_path.exists():
        # Try alternative path
        alt_path = ui_module_path / ("dashboard.py" if mode == "nocode" else "streamlit_app.py")
        if alt_path.exists():
            app_path = alt_path
        else:
            raise FileNotFoundError(f"Dashboard app not found at {app_path}")
    
    # Prepare command
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    
    # Add Streamlit configuration
    streamlit_args = [
        "--server.port", str(kwargs.get('port', 8501)),
        "--server.address", kwargs.get('address', '0.0.0.0'),
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "none" if kwargs.get('production', False) else "auto"
    ]
    
    cmd.extend(streamlit_args)
    
    # Add application arguments after "--"
    if config or kwargs:
        cmd.append("--")
        
        if config:
            import json
            import tempfile
            
            # Save config to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f)
                config_path = f.name
            
            cmd.extend(["--config", config_path])
        
        # Add any additional arguments
        for key, value in kwargs.items():
            if key not in ['tenant_id', 'user_type', 'port', 'address', 'production']:
                cmd.extend([f"--{key}", str(value)])
    
    # Launch the dashboard
    print(f"🚀 Launching AutoML Dashboard ({'No-Code' if mode == 'nocode' else 'Classic'} Mode)...")
    print(f"📍 Access at: http://localhost:{kwargs.get('port', 8501)}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching dashboard: {e}")
        if METRICS_AVAILABLE and ml_ui_errors_total:
            ml_ui_errors_total.labels(
                tenant_id=tenant_id,
                error_type="LaunchError",
                page="dashboard"
            ).inc()
        raise
    except KeyboardInterrupt:
        print("\n✋ Dashboard stopped by user")
        end_ui_session(page="dashboard", tenant_id=tenant_id)


def launch_wizard(**kwargs):
    """
    Launch the AutoML wizard directly.
    
    Args:
        **kwargs: Arguments passed to the wizard
    
    Returns:
        None
    """
    if not NOCODE_DASHBOARD:
        raise ImportError("No-code dashboard is not available. Please install required dependencies.")
    
    # Configure wizard mode
    config = {
        'start_page': 'wizard',
        'wizard_mode': True
    }
    
    return launch_dashboard(config=config, mode="nocode", **kwargs)


def quick_start(file_path=None, target_column=None, **kwargs):
    """
    Quick start AutoML training with minimal configuration.
    
    Args:
        file_path (str, optional): Path to data file
        target_column (str, optional): Target column name
        **kwargs: Additional configuration
    
    Returns:
        None
    """
    if not NOCODE_DASHBOARD:
        raise ImportError("No-code features are not available. Please install required dependencies.")
    
    config = {
        'quick_start': True,
        'initial_file': file_path,
        'target_column': target_column,
        'auto_train': kwargs.get('auto_train', True)
    }
    
    return launch_dashboard(config=config, mode="nocode", **kwargs)


# ============================================================================
# Component Factory Functions with Metrics
# ============================================================================

@track_ui_response_time("data_quality_visualizer", "create")
def create_data_quality_visualizer(tenant_id="default"):
    """
    Factory function to create a DataQualityVisualizer instance.
    
    Returns:
        DataQualityVisualizer or None: Visualizer instance if available
    """
    if not COMPONENTS_AVAILABLE:
        raise ImportError("UI components are not available. Install required dependencies.")
    
    track_component_render("data_quality_visualizer", tenant_id=tenant_id)
    return DataQualityVisualizer()


@track_ui_response_time("model_leaderboard", "create")
def create_model_leaderboard(tenant_id="default"):
    """
    Factory function to create a ModelLeaderboard instance.
    
    Returns:
        ModelLeaderboard or None: Leaderboard instance if available
    """
    if not COMPONENTS_AVAILABLE:
        raise ImportError("UI components are not available. Install required dependencies.")
    
    track_component_render("model_leaderboard", tenant_id=tenant_id)
    return ModelLeaderboard()


@track_ui_response_time("chat_interface", "create")
def create_chat_interface(tenant_id="default"):
    """
    Factory function to create a ChatInterface instance.
    
    Returns:
        ChatInterface or None: Chat interface instance if available
    """
    if not COMPONENTS_AVAILABLE:
        raise ImportError("UI components are not available. Install required dependencies.")
    
    track_component_render("chat_interface", tenant_id=tenant_id)
    return ChatInterface()


# ============================================================================
# Module initialization
# ============================================================================

# Log module status on import
import logging
logger = logging.getLogger(__name__)

if STREAMLIT_AVAILABLE and PLOTLY_AVAILABLE:
    if NOCODE_DASHBOARD:
        logger.info(f"AutoML UI module v{__version__} loaded with No-Code Dashboard")
    elif COMPONENTS_AVAILABLE:
        logger.info(f"AutoML UI module v{__version__} loaded successfully with all components")
    else:
        logger.warning("AutoML UI module loaded but some components are unavailable")
    
    if METRICS_AVAILABLE:
        logger.info("UI metrics collection enabled")
else:
    missing_deps = []
    if not STREAMLIT_AVAILABLE:
        missing_deps.append("streamlit")
    if not PLOTLY_AVAILABLE:
        missing_deps.append("plotly")
    
    logger.warning(
        f"AutoML UI module loaded with missing dependencies: {', '.join(missing_deps)}. "
        f"Install with: pip install {' '.join(missing_deps)}"
    )

# Print status if running as main module
if __name__ == "__main__":
    print("AutoML Platform UI Module - No-Code Edition")
    print("=" * 60)
    print(f"Version: {__version__}")
    print("\n📊 Dependency Status:")
    
    deps = check_ui_dependencies()
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"  {status} {dep}")
    
    print("\n🧩 Component Status:")
    ui_status = get_ui_status()
    for component, available in ui_status["components"].items():
        status = "✅" if available else "❌"
        print(f"  {status} {component}")
    
    print(f"\n📈 Overall: {ui_status['readiness']}")
    
    if NOCODE_DASHBOARD:
        print("\n✨ No-Code Features:")
        for feature, available in ui_status["features"].items():
            status = "✅" if available else "❌"
            print(f"  {status} {feature}")
    
    if METRICS_AVAILABLE:
        print("\n📊 Metrics collection: Enabled")
    
    if DASHBOARD_AVAILABLE:
        print("\n🚀 Quick Start Commands:")
        print("  Launch Dashboard:  automl-dashboard")
        print("  Launch Wizard:     automl-wizard")
        print("  Python:           from automl_platform.ui import launch_dashboard")
        print("                    launch_dashboard()")
        print("\n💡 Tips:")
        print("  - For production: launch_dashboard(production=True)")
        print("  - Custom port:    launch_dashboard(port=8080)")
        print("  - Quick train:    quick_start('data.csv', 'target')")
