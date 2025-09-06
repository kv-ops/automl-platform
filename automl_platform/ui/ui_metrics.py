"""
UI Metrics for AutoML Platform
Tracks user interactions, page views, and API calls from the UI
Place in: automl_platform/api/ui_metrics.py
"""

import time
from functools import wraps
from typing import Dict, Any, Optional
from datetime import datetime

# Métriques Prometheus
from prometheus_client import Counter, Histogram, Gauge

# Déclaration des métriques UI
ml_ui_sessions_total = Gauge(
    'ml_ui_sessions_total',
    'Total number of active UI sessions',
    ['tenant_id', 'user_role']
)

ml_ui_page_views_total = Counter(
    'ml_ui_page_views_total',
    'Total number of page views',
    ['tenant_id', 'page', 'user_role']
)

ml_ui_api_calls_total = Counter(
    'ml_ui_api_calls_total',
    'Total API calls from UI',
    ['tenant_id', 'endpoint', 'method']
)

ml_ui_response_time_seconds = Histogram(
    'ml_ui_response_time_seconds',
    'UI response time in seconds',
    ['page', 'action'],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

ml_ui_errors_total = Counter(
    'ml_ui_errors_total',
    'Total UI errors',
    ['tenant_id', 'error_type', 'page']
)

ml_ui_feature_usage = Counter(
    'ml_ui_feature_usage',
    'Feature usage tracking',
    ['tenant_id', 'feature_name', 'action']
)


class UIMetrics:
    """Helper class for tracking UI metrics in Streamlit and components."""
    
    def __init__(self, tenant_id: str = "default", user_role: str = "user"):
        self.tenant_id = tenant_id
        self.user_role = user_role
        self.session_id = None
        
    def increment_session(self):
        """Increment active session count."""
        ml_ui_sessions_total.labels(
            tenant_id=self.tenant_id,
            user_role=self.user_role
        ).inc()
        
    def decrement_session(self):
        """Decrement active session count."""
        ml_ui_sessions_total.labels(
            tenant_id=self.tenant_id,
            user_role=self.user_role
        ).dec()
        
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


# Example usage in Streamlit
def example_streamlit_integration():
    """Example of how to integrate metrics in Streamlit app."""
    import streamlit as st
    
    # Track page view
    track_streamlit_page("dashboard", st.session_state.get('tenant_id', 'default'))
    
    # Track button click
    if st.button("Train Model"):
        track_streamlit_action("train_model_button", "click")
        # ... rest of the logic
    
    # Track API call
    metrics = UIMetrics(st.session_state.get('tenant_id', 'default'))
    metrics.track_api_call("/models", "GET")
    
    # Measure response time
    @metrics.measure_response_time("models", "list")
    def list_models():
        # ... fetch models
        pass
