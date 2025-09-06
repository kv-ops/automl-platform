"""
UI module for AutoML Platform
=============================

This module provides reusable UI components and a complete Streamlit dashboard
for the AutoML platform, featuring:
- Interactive data quality visualizations
- Model training and monitoring interfaces
- AI-powered chat assistant
- Comprehensive reporting tools
- Real-time training progress tracking
"""

__version__ = "1.0.0"
__author__ = "AutoML Platform Team"

# ============================================================================
# Metrics imports and initialization
# ============================================================================

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    import time
    METRICS_AVAILABLE = True
    
    # Create a registry for UI metrics
    ui_registry = CollectorRegistry()
    
    # Define UI metrics
    ml_ui_sessions_total = Counter(
        'ml_ui_sessions_total',
        'Total number of UI sessions started',
        ['page', 'user_type'],
        registry=ui_registry
    )
    
    ml_ui_response_time_seconds = Histogram(
        'ml_ui_response_time_seconds',
        'UI response time in seconds',
        ['component', 'action'],
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        registry=ui_registry
    )
    
    ml_ui_page_views_total = Counter(
        'ml_ui_page_views_total',
        'Total number of page views',
        ['page', 'source'],
        registry=ui_registry
    )
    
    ml_ui_api_calls_total = Counter(
        'ml_ui_api_calls_total',
        'Total number of API calls from UI',
        ['endpoint', 'method', 'status'],
        registry=ui_registry
    )
    
    ml_ui_active_users = Gauge(
        'ml_ui_active_users',
        'Number of active UI users',
        ['page'],
        registry=ui_registry
    )
    
    ml_ui_component_renders_total = Counter(
        'ml_ui_component_renders_total',
        'Total number of component renders',
        ['component', 'status'],
        registry=ui_registry
    )
    
    ml_ui_errors_total = Counter(
        'ml_ui_errors_total',
        'Total number of UI errors',
        ['component', 'error_type'],
        registry=ui_registry
    )
    
except ImportError:
    METRICS_AVAILABLE = False
    ui_registry = None
    ml_ui_sessions_total = None
    ml_ui_response_time_seconds = None
    ml_ui_page_views_total = None
    ml_ui_api_calls_total = None
    ml_ui_active_users = None
    ml_ui_component_renders_total = None
    ml_ui_errors_total = None

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

# Import main Streamlit app if available
if STREAMLIT_AVAILABLE:
    try:
        from .streamlit_app import AutoMLDashboard
        DASHBOARD_AVAILABLE = True
    except ImportError:
        DASHBOARD_AVAILABLE = False
        AutoMLDashboard = None
else:
    DASHBOARD_AVAILABLE = False
    AutoMLDashboard = None

# ============================================================================
# Metrics tracking decorators and functions
# ============================================================================

def track_ui_response_time(component, action):
    """Decorator to track UI response time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if METRICS_AVAILABLE and ml_ui_response_time_seconds:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    ml_ui_response_time_seconds.labels(
                        component=component,
                        action=action
                    ).observe(time.time() - start_time)
                    return result
                except Exception as e:
                    if ml_ui_errors_total:
                        ml_ui_errors_total.labels(
                            component=component,
                            error_type=type(e).__name__
                        ).inc()
                    raise
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def track_page_view(page, source="direct"):
    """Track a page view in the UI."""
    if METRICS_AVAILABLE and ml_ui_page_views_total:
        ml_ui_page_views_total.labels(page=page, source=source).inc()

def track_api_call(endpoint, method, status):
    """Track an API call from the UI."""
    if METRICS_AVAILABLE and ml_ui_api_calls_total:
        ml_ui_api_calls_total.labels(
            endpoint=endpoint,
            method=method,
            status=status
        ).inc()

def track_component_render(component, status="success"):
    """Track a component render."""
    if METRICS_AVAILABLE and ml_ui_component_renders_total:
        ml_ui_component_renders_total.labels(
            component=component,
            status=status
        ).inc()

def start_ui_session(page="main", user_type="standard"):
    """Track the start of a UI session."""
    if METRICS_AVAILABLE:
        if ml_ui_sessions_total:
            ml_ui_sessions_total.labels(page=page, user_type=user_type).inc()
        if ml_ui_active_users:
            ml_ui_active_users.labels(page=page).inc()

def end_ui_session(page="main"):
    """Track the end of a UI session."""
    if METRICS_AVAILABLE and ml_ui_active_users:
        ml_ui_active_users.labels(page=page).dec()

# ============================================================================
# Dynamically build __all__ based on available components
# ============================================================================

def _build_all_list():
    """Dynamically build the __all__ list based on available components."""
    all_list = ["__version__"]
    
    # Always include availability flags
    all_list.extend([
        "STREAMLIT_AVAILABLE",
        "PLOTLY_AVAILABLE", 
        "COMPONENTS_AVAILABLE",
        "DASHBOARD_AVAILABLE",
        "METRICS_AVAILABLE"
    ])
    
    # Include metrics if available
    if METRICS_AVAILABLE:
        all_list.extend([
            "ui_registry",
            "ml_ui_sessions_total",
            "ml_ui_response_time_seconds",
            "ml_ui_page_views_total",
            "ml_ui_api_calls_total",
            "ml_ui_active_users",
            "ml_ui_component_renders_total",
            "ml_ui_errors_total",
            "track_ui_response_time",
            "track_page_view",
            "track_api_call",
            "track_component_render",
            "start_ui_session",
            "end_ui_session"
        ])
    
    # Always include helper functions (they handle unavailability gracefully)
    all_list.extend([
        "check_ui_dependencies",
        "get_ui_status",
        "launch_dashboard"
    ])
    
    # Add components only if they're available
    if DASHBOARD_AVAILABLE and AutoMLDashboard is not None:
        all_list.append("AutoMLDashboard")
    
    if COMPONENTS_AVAILABLE:
        # Data Quality Components
        if DataQualityVisualizer is not None:
            all_list.append("DataQualityVisualizer")
        
        # Model Components  
        if ModelLeaderboard is not None:
            all_list.append("ModelLeaderboard")
        if FeatureImportanceVisualizer is not None:
            all_list.append("FeatureImportanceVisualizer")
        
        # Monitoring Components
        if DriftMonitor is not None:
            all_list.append("DriftMonitor")
        if TrainingProgressTracker is not None:
            all_list.append("TrainingProgressTracker")
        
        # Interactive Components
        if ChatInterface is not None:
            all_list.append("ChatInterface")
        if ExperimentComparator is not None:
            all_list.append("ExperimentComparator")
        
        # Alert and Report Components
        if AlertsAndNotifications is not None:
            all_list.append("AlertsAndNotifications")
        if ReportGenerator is not None:
            all_list.append("ReportGenerator")
        
        # Advanced Visualizations
        if DataFlowDiagram is not None:
            all_list.append("DataFlowDiagram")
        if ModelDeploymentUI is not None:
            all_list.append("ModelDeploymentUI")
        if CustomMetrics is not None:
            all_list.append("CustomMetrics")
        
        # Utility functions
        if create_sidebar_filters is not None:
            all_list.append("create_sidebar_filters")
        if create_action_buttons is not None:
            all_list.append("create_action_buttons")
        if show_loading_animation is not None:
            all_list.append("show_loading_animation")
    
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
        "components": COMPONENTS_AVAILABLE,
        "dashboard": DASHBOARD_AVAILABLE,
        "metrics": METRICS_AVAILABLE
    }
    
    # Check for additional optional dependencies
    try:
        import pandas
        dependencies["pandas"] = True
    except ImportError:
        dependencies["pandas"] = False
    
    try:
        import numpy
        dependencies["numpy"] = True
    except ImportError:
        dependencies["numpy"] = False
    
    try:
        import sklearn
        dependencies["sklearn"] = True
    except ImportError:
        dependencies["sklearn"] = False
    
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
        "metrics_enabled": METRICS_AVAILABLE
    }
    
    # Calculate overall readiness
    total_components = len(status["components"])
    available_components = sum(status["components"].values())
    status["readiness"] = f"{available_components}/{total_components} components available"
    
    return status


@track_ui_response_time("dashboard", "launch")
def launch_dashboard(config=None, **kwargs):
    """
    Launch the Streamlit dashboard application.
    
    Args:
        config (dict, optional): Configuration dictionary
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
    
    # Track dashboard launch
    start_ui_session(page="dashboard", user_type="admin")
    track_page_view("dashboard", "launch")
    
    # Import here to avoid issues if streamlit is not installed
    import sys
    import subprocess
    from pathlib import Path
    
    # Get the path to the streamlit app
    ui_module_path = Path(__file__).parent
    app_path = ui_module_path / "streamlit_app.py"
    
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app not found at {app_path}")
    
    # Prepare command
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    
    # Add configuration if provided
    if config:
        import json
        import tempfile
        
        # Save config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        cmd.extend(["--", "--config", config_path])
    
    # Add any additional arguments
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    
    # Launch the dashboard
    print(f"Launching AutoML Dashboard...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching dashboard: {e}")
        if METRICS_AVAILABLE and ml_ui_errors_total:
            ml_ui_errors_total.labels(
                component="dashboard",
                error_type="LaunchError"
            ).inc()
        raise
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        end_ui_session(page="dashboard")


# ============================================================================
# Component Factory Functions with Metrics
# ============================================================================

@track_ui_response_time("data_quality_visualizer", "create")
def create_data_quality_visualizer():
    """
    Factory function to create a DataQualityVisualizer instance.
    
    Returns:
        DataQualityVisualizer or None: Visualizer instance if available
    """
    if not COMPONENTS_AVAILABLE:
        raise ImportError("UI components are not available. Install required dependencies.")
    
    track_component_render("data_quality_visualizer")
    return DataQualityVisualizer()


@track_ui_response_time("model_leaderboard", "create")
def create_model_leaderboard():
    """
    Factory function to create a ModelLeaderboard instance.
    
    Returns:
        ModelLeaderboard or None: Leaderboard instance if available
    """
    if not COMPONENTS_AVAILABLE:
        raise ImportError("UI components are not available. Install required dependencies.")
    
    track_component_render("model_leaderboard")
    return ModelLeaderboard()


@track_ui_response_time("chat_interface", "create")
def create_chat_interface():
    """
    Factory function to create a ChatInterface instance.
    
    Returns:
        ChatInterface or None: Chat interface instance if available
    """
    if not COMPONENTS_AVAILABLE:
        raise ImportError("UI components are not available. Install required dependencies.")
    
    track_component_render("chat_interface")
    return ChatInterface()


# ============================================================================
# Module initialization
# ============================================================================

# Log module status on import
import logging
logger = logging.getLogger(__name__)

if STREAMLIT_AVAILABLE and PLOTLY_AVAILABLE:
    if COMPONENTS_AVAILABLE:
        logger.info(f"AutoML UI module v{__version__} loaded successfully with all components")
        if METRICS_AVAILABLE:
            logger.info("UI metrics collection enabled")
    else:
        logger.warning("AutoML UI module loaded but some components are unavailable")
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
    print("AutoML Platform UI Module")
    print("=" * 50)
    print(f"Version: {__version__}")
    print("\nDependency Status:")
    
    deps = check_ui_dependencies()
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    print("\nComponent Status:")
    ui_status = get_ui_status()
    for component, available in ui_status["components"].items():
        status = "✓" if available else "✗"
        print(f"  {status} {component}")
    
    print(f"\nOverall: {ui_status['readiness']}")
    
    if METRICS_AVAILABLE:
        print("\nMetrics collection: Enabled")
    
    if DASHBOARD_AVAILABLE:
        print("\nTo launch the dashboard, run:")
        print("  python -m automl_platform.ui")
        print("or in Python:")
        print("  from automl_platform.ui import launch_dashboard")
        print("  launch_dashboard()")
