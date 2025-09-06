"""
UI Metrics for AutoML Platform
Tracks user interactions, page views, and API calls from the UI
Place in: automl_platform/ui/ui_metrics.py
"""

import time
from functools import wraps
from typing import Dict, Any, Optional
from datetime import datetime

# Métriques Prometheus
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Créer un registre local pour les métriques UI
ui_registry = CollectorRegistry()

# Déclaration des métriques UI avec le registre local
ml_ui_sessions_total = Gauge(
    'ml_ui_sessions_total',
    'Total number of active UI sessions',
    ['tenant_id', 'user_role'],
    registry=ui_registry
)

ml_ui_page_views_total = Counter(
    'ml_ui_page_views_total',
    'Total number of page views',
    ['tenant_id', 'page', 'user_role'],
    registry=ui_registry
)

ml_ui_api_calls_total = Counter(
    'ml_ui_api_calls_total',
    'Total API calls from UI',
    ['tenant_id', 'endpoint', 'method'],
    registry=ui_registry
)

ml_ui_response_time_seconds = Histogram(
    'ml_ui_response_time_seconds',
    'UI response time in seconds',
    ['page', 'action'],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=ui_registry
)

ml_ui_errors_total = Counter(
    'ml_ui_errors_total',
    'Total UI errors',
    ['tenant_id', 'error_type', 'page'],
    registry=ui_registry
)

ml_ui_feature_usage = Counter(
    'ml_ui_feature_usage',
    'Feature usage tracking',
    ['tenant_id', 'feature_name', 'action'],
    registry=ui_registry
)

# Additional UI-specific metrics
ml_ui_component_renders_total = Counter(
    'ml_ui_component_renders_total',
    'Total component renders',
    ['component'],
    registry=ui_registry
)

ml_ui_active_users = Gauge(
    'ml_ui_active_users',
    'Number of active users',
    registry=ui_registry
)


class UIMetrics:
    """Helper class for tracking UI metrics in Streamlit and components."""
    
    def __init__(self, tenant_id: str = "default", user_role: str = "user"):
        self.tenant_id = tenant_id
        self.user_role = user_role
        self.session_id = None
        self.session_start = datetime.now()
        
    def increment_session(self):
        """Increment active session count."""
        ml_ui_sessions_total.labels(
            tenant_id=self.tenant_id,
            user_role=self.user_role
        ).inc()
        ml_ui_active_users.inc()
        
    def decrement_session(self):
        """Decrement active session count."""
        ml_ui_sessions_total.labels(
            tenant_id=self.tenant_id,
            user_role=self.user_role
        ).dec()
        ml_ui_active_users.dec()
        
    def track_page_view(self, page: str):
        """Track a page view."""
        ml_ui_page_views_total.labels(
            tenant_id=self.tenant_id,
            page=page,
            user_role=self.user_role
        ).inc()
        
    def track_api_call(self, endpoint: str, method: str = "GET"):
        """Track an API call from the UI."""
        ml_ui_api_calls_total.labels(
            tenant_id=self.tenant_id,
            endpoint=endpoint,
            method=method
        ).inc()
        
    def track_error(self, error_type: str, page: str):
        """Track UI errors."""
        ml_ui_errors_total.labels(
            tenant_id=self.tenant_id,
            error_type=error_type,
            page=page
        ).inc()
        
    def track_feature_usage(self, feature_name: str, action: str = "click"):
        """Track feature usage."""
        ml_ui_feature_usage.labels(
            tenant_id=self.tenant_id,
            feature_name=feature_name,
            action=action
        ).inc()
        
    def track_component_render(self, component_name: str):
        """Track component renders."""
        ml_ui_component_renders_total.labels(
            component=component_name
        ).inc()
        
    def measure_response_time(self, page: str, action: str):
        """Decorator to measure response time."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    ml_ui_response_time_seconds.labels(
                        page=page,
                        action=action
                    ).observe(time.time() - start_time)
            return wrapper
        return decorator
    
    def get_session_duration(self) -> float:
        """Get current session duration in seconds."""
        return (datetime.now() - self.session_start).total_seconds()


# Streamlit specific metrics helpers
def track_streamlit_page(page_name: str, tenant_id: str = "default", user_role: str = "user"):
    """Track Streamlit page view."""
    ml_ui_page_views_total.labels(
        tenant_id=tenant_id,
        page=page_name,
        user_role=user_role
    ).inc()


def track_streamlit_action(feature_name: str, action: str = "click", tenant_id: str = "default"):
    """Track Streamlit button/action."""
    ml_ui_feature_usage.labels(
        tenant_id=tenant_id,
        feature_name=feature_name,
        action=action
    ).inc()


def track_streamlit_component(component_name: str):
    """Track Streamlit component render."""
    ml_ui_component_renders_total.labels(
        component=component_name
    ).inc()


# Context manager for session tracking
class UISession:
    """Context manager for tracking UI sessions."""
    
    def __init__(self, tenant_id: str = "default", user_role: str = "user"):
        self.metrics = UIMetrics(tenant_id, user_role)
        
    def __enter__(self):
        self.metrics.increment_session()
        return self.metrics
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics.decrement_session()
        
        # Log any errors that occurred
        if exc_type is not None:
            self.metrics.track_error(
                error_type=exc_type.__name__,
                page="unknown"
            )


# Decorator for tracking page loads
def track_page(page_name: str):
    """Decorator to track page loads in Streamlit."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get tenant_id from session state if available
            import streamlit as st
            tenant_id = st.session_state.get('tenant_id', 'default')
            user_role = st.session_state.get('user_role', 'user')
            
            # Track page view
            track_streamlit_page(page_name, tenant_id, user_role)
            
            # Track component render
            track_streamlit_component(page_name)
            
            # Execute the page function
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Decorator for tracking actions
def track_action(action_name: str):
    """Decorator to track user actions in Streamlit."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get tenant_id from session state if available
            import streamlit as st
            tenant_id = st.session_state.get('tenant_id', 'default')
            
            # Track action
            track_streamlit_action(action_name, "execute", tenant_id)
            
            # Measure response time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                ml_ui_response_time_seconds.labels(
                    page="unknown",
                    action=action_name
                ).observe(time.time() - start_time)
        return wrapper
    return decorator


# Example usage in Streamlit
def example_streamlit_integration():
    """Example of how to integrate metrics in Streamlit app."""
    import streamlit as st
    
    # Initialize session metrics if not already done
    if 'ui_metrics' not in st.session_state:
        st.session_state.ui_metrics = UIMetrics(
            tenant_id=st.session_state.get('tenant_id', 'default'),
            user_role=st.session_state.get('user_role', 'user')
        )
        st.session_state.ui_metrics.increment_session()
    
    # Track page view
    track_streamlit_page("dashboard", st.session_state.get('tenant_id', 'default'))
    
    # Track component renders
    track_streamlit_component("sidebar")
    track_streamlit_component("main_content")
    
    # Track button click
    if st.button("Train Model"):
        track_streamlit_action("train_model_button", "click")
        # ... rest of the logic
    
    # Track API call
    st.session_state.ui_metrics.track_api_call("/models", "GET")
    
    # Measure response time for a function
    @st.session_state.ui_metrics.measure_response_time("models", "list")
    def list_models():
        # ... fetch models
        pass
    
    # Track errors
    try:
        # Some operation that might fail
        pass
    except Exception as e:
        st.session_state.ui_metrics.track_error(
            error_type=type(e).__name__,
            page="dashboard"
        )


# Page-specific tracking functions
@track_page("home")
def render_home_page():
    """Example home page with metrics tracking."""
    import streamlit as st
    st.title("Home")
    # Page content here


@track_page("models")
def render_models_page():
    """Example models page with metrics tracking."""
    import streamlit as st
    st.title("Models")
    # Page content here


@track_page("data")
def render_data_page():
    """Example data page with metrics tracking."""
    import streamlit as st
    st.title("Data")
    # Page content here


# Action-specific tracking functions
@track_action("upload_data")
def upload_data_action(file):
    """Example data upload action with metrics tracking."""
    # Upload logic here
    pass


@track_action("train_model")
def train_model_action(config):
    """Example model training action with metrics tracking."""
    # Training logic here
    pass


@track_action("export_results")
def export_results_action(format="csv"):
    """Example export action with metrics tracking."""
    # Export logic here
    pass


# Flag to indicate metrics are available
METRICS_AVAILABLE = True

# Export the registry and main components so they can be imported by api.py
__all__ = ['ui_registry', 'UIMetrics', 'UISession', 'track_streamlit_page', 
          'track_streamlit_action', 'track_streamlit_component', 'track_page', 
          'track_action', 'METRICS_AVAILABLE']
