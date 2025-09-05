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
# Public API
# ============================================================================

__all__ = [
    # Version info
    "__version__",
    
    # Availability flags
    "STREAMLIT_AVAILABLE",
    "PLOTLY_AVAILABLE",
    "COMPONENTS_AVAILABLE",
    "DASHBOARD_AVAILABLE",
    
    # Main Dashboard
    "AutoMLDashboard",
    
    # Data Quality Components
    "DataQualityVisualizer",
    
    # Model Components
    "ModelLeaderboard",
    "FeatureImportanceVisualizer",
    
    # Monitoring Components
    "DriftMonitor",
    "TrainingProgressTracker",
    
    # Interactive Components
    "ChatInterface",
    "ExperimentComparator",
    
    # Alert and Report Components
    "AlertsAndNotifications",
    "ReportGenerator",
    
    # Advanced Visualizations
    "DataFlowDiagram",
    "ModelDeploymentUI",
    "CustomMetrics",
    
    # Utility functions
    "create_sidebar_filters",
    "create_action_buttons",
    "show_loading_animation",
    
    # Helper functions
    "check_ui_dependencies",
    "get_ui_status",
    "launch_dashboard"
]

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
        "dashboard": DASHBOARD_AVAILABLE
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
        "dashboard_available": DASHBOARD_AVAILABLE
    }
    
    # Calculate overall readiness
    total_components = len(status["components"])
    available_components = sum(status["components"].values())
    status["readiness"] = f"{available_components}/{total_components} components available"
    
    return status


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
        raise
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")


# ============================================================================
# Component Factory Functions
# ============================================================================

def create_data_quality_visualizer():
    """
    Factory function to create a DataQualityVisualizer instance.
    
    Returns:
        DataQualityVisualizer or None: Visualizer instance if available
    """
    if not COMPONENTS_AVAILABLE:
        raise ImportError("UI components are not available. Install required dependencies.")
    
    return DataQualityVisualizer()


def create_model_leaderboard():
    """
    Factory function to create a ModelLeaderboard instance.
    
    Returns:
        ModelLeaderboard or None: Leaderboard instance if available
    """
    if not COMPONENTS_AVAILABLE:
        raise ImportError("UI components are not available. Install required dependencies.")
    
    return ModelLeaderboard()


def create_chat_interface():
    """
    Factory function to create a ChatInterface instance.
    
    Returns:
        ChatInterface or None: Chat interface instance if available
    """
    if not COMPONENTS_AVAILABLE:
        raise ImportError("UI components are not available. Install required dependencies.")
    
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
    
    if DASHBOARD_AVAILABLE:
        print("\nTo launch the dashboard, run:")
        print("  python -m automl_platform.ui")
        print("or in Python:")
        print("  from automl_platform.ui import launch_dashboard")
        print("  launch_dashboard()")
