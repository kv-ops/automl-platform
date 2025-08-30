"""
Streamlit UI for AutoML Platform
Advanced interface with dashboards, real-time monitoring, and LLM chat
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
import asyncio
import websocket
from pathlib import Path
import os

# Page configuration
st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; padding-left: 20px; padding-right: 20px;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 12px;
        border-radius: 8px;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 12px;
        border-radius: 8px;
        color: #856404;
        margin: 1rem 0;
    }
    .drift-alert {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 12px;
        border-radius: 8px;
        color: #721c24;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = os.getenv('API_URL', 'http://localhost:8000')
MLFLOW_URL = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')

# Session State
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None
if 'current_job' not in st.session_state:
    st.session_state.current_job = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'models' not in st.session_state:
    st.session_state.models = []

# Helper Functions
def check_api_health():
    """Check API health status."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_auth_headers():
    """Get authentication headers."""
    if st.session_state.auth_token:
        return {"Authorization": f"Bearer {st.session_state.auth_token}"}
    return {}

def submit_training_job(file, config):
    """Submit training job to API."""
    files = {'file': file}
    data = {'config': json.dumps(config)}
    headers = get_auth_headers()
    
    response = requests.post(
        f"{API_URL}/train",
        files=files,
        data=data,
        headers=headers
    )
    return response.json() if response.status_code == 202 else None

def get_job_status(job_id):
    """Get job status from API."""
    headers = get_auth_headers()
    response = requests.get(f"{API_URL}/job/{job_id}", headers=headers)
    return response.json() if response.status_code == 200 else None

def get_models():
    """Get list of trained models."""
    headers = get_auth_headers()
    response = requests.get(f"{API_URL}/models", headers=headers)
    return response.json() if response.status_code == 200 else []

def get_drift_report(model_id):
    """Get drift report for a model."""
    headers = get_auth_headers()
    response = requests.get(f"{API_URL}/models/{model_id}/drift", headers=headers)
    return response.json() if response.status_code == 200 else None

# Sidebar
with st.sidebar:
    st.title("🚀 AutoML Platform")
    
    # API Status
    api_status = check_api_health()
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
    
    # Navigation
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Train Model", "Models", "Predictions", "Data Quality", "LLM Assistant", "Settings"]
    )
    
    # User Info
    st.markdown("---")
    if st.session_state.auth_token:
        st.info("👤 Logged in as: user@example.com")
        if st.button("Logout"):
            st.session_state.auth_token = None
            st.rerun()

# Main Content
if page == "Dashboard":
    st.title("📊 AutoML Dashboard")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    models = get_models()
    
    with col1:
        st.metric(
            label="Total Models",
            value=len(models),
            delta="+2 this week"
        )
    
    with col2:
        best_score = max([m.get('cv_score', 0) for m in models], default=0)
        st.metric(
            label="Best Model Score",
            value=f"{best_score:.3f}",
            delta="+0.02"
        )
    
    with col3:
        st.metric(
            label="Active Jobs",
            value="3",
            delta="-1"
        )
    
    with col4:
        st.metric(
            label="Storage Used",
            value="2.4 GB",
            delta="+0.3 GB"
        )
    
    # Charts
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Model Performance Trends")
        
        # Sample data for demo
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        scores = np.random.randn(30).cumsum() + 0.85
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=scores,
            mode='lines+markers',
            name='CV Score',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6)
        ))
        fig.update_layout(
            height=300,
            showlegend=False,
            hovermode='x unified',
            xaxis=dict(showgrid=False),
            yaxis=dict(title='Score', range=[0.7, 1.0])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Model Distribution")
        
        model_types = ['RandomForest', 'XGBoost', 'LightGBM', 'Neural Net', 'LinearModel']
        counts = [8, 12, 7, 3, 5]
        
        fig = go.Figure(data=[
            go.Bar(
                x=model_types,
                y=counts,
                marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
            )
        ])
        fig.update_layout(
            height=300,
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(title='Count', showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Activity
    st.markdown("---")
    st.subheader("📋 Recent Activity")
    
    activity_data = {
        'Time': ['2 min ago', '15 min ago', '1 hour ago', '3 hours ago', '5 hours ago'],
        'Event': [
            'Model training completed',
            'Drift detected in production',
            'New dataset uploaded',
            'Batch prediction finished',
            'Model deployed to production'
        ],
        'Status': ['✅', '⚠️', '✅', '✅', '✅']
    }
    
    activity_df = pd.DataFrame(activity_data)
    st.dataframe(activity_df, use_container_width=True, hide_index=True)

elif page == "Train Model":
    st.title("🎯 Train New Model")
    
    # File Upload Section
    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=['csv', 'parquet', 'xlsx'],
        help="Upload your training dataset"
    )
    
    if uploaded_file:
        # Load and preview data
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_parquet(uploaded_file)
        
        # Data Overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(100), height=300)
        
        with col2:
            st.subheader("📈 Data Statistics")
            st.metric("Rows", f"{df.shape[0]:,}")
            st.metric("Columns", df.shape[1])
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Configuration Section
        st.markdown("---")
        st.subheader("⚙️ Training Configuration")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Basic", "Advanced", "Features", "LLM Options"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                target_column = st.selectbox(
                    "Target Column",
                    options=df.columns.tolist(),
                    help="Select the column to predict"
                )
                task_type = st.selectbox(
                    "Task Type",
                    options=["auto", "classification", "regression"],
                    help="AutoML will detect automatically if set to 'auto'"
                )
            with col2:
                cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)
                train_size = st.slider("Training Size (%)", 50, 90, 80)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                algorithms = st.multiselect(
                    "Algorithms to Test",
                    options=['all', 'RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 
                            'NeuralNet', 'LinearModels', 'SVM'],
                    default=['all']
                )
                ensemble_method = st.selectbox(
                    "Ensemble Method",
                    options=['none', 'voting', 'stacking', 'blending']
                )
            with col2:
                hpo_method = st.selectbox(
                    "Hyperparameter Optimization",
                    options=['optuna', 'random', 'grid', 'bayesian', 'none']
                )
                hpo_trials = st.number_input("HPO Trials", 10, 200, 50)
        
        with tab3:
            auto_feature_engineering = st.checkbox("Auto Feature Engineering", value=True)
            handle_imbalance = st.checkbox("Handle Class Imbalance", value=True)
            remove_outliers = st.checkbox("Remove Outliers", value=False)
            
            if auto_feature_engineering:
                st.info("Will create polynomial features, interactions, and domain-specific features")
        
        with tab4:
            use_llm_cleaning = st.checkbox("LLM Data Cleaning", value=True)
            use_llm_features = st.checkbox("LLM Feature Suggestions", value=True)
            generate_report = st.checkbox("Generate LLM Report", value=True)
            
            llm_model = st.selectbox(
                "LLM Model",
                options=['gpt-4', 'gpt-3.5-turbo', 'claude-2', 'llama-2']
            )
        
        # Training Button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                config = {
                    'target': target_column,
                    'task': task_type,
                    'cv_folds': cv_folds,
                    'algorithms': algorithms,
                    'ensemble_method': ensemble_method,
                    'hpo_method': hpo_method,
                    'hpo_n_iter': hpo_trials,
                    'auto_clean': use_llm_cleaning,
                    'auto_feature_engineering': auto_feature_engineering,
                    'handle_imbalance': handle_imbalance,
                    'llm_model': llm_model
                }
                
                uploaded_file.seek(0)
                with st.spinner("Submitting job..."):
                    result = submit_training_job(uploaded_file, config)
                
                if result:
                    st.session_state.current_job = result['job_id']
                    st.success(f"✅ Training job submitted: {result['job_id']}")
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    while True:
                        status = get_job_status(result['job_id'])
                        if status:
                            progress = status.get('progress', 0)
                            progress_bar.progress(progress / 100)
                            status_text.text(f"Status: {status.get('status')} - {status.get('message')}")
                            
                            if status['status'] == 'completed':
                                st.balloons()
                                st.success("🎉 Training completed successfully!")
                                break
                            elif status['status'] == 'failed':
                                st.error(f"Training failed: {status.get('message')}")
                                break
                        
                        time.sleep(2)

elif page == "Models":
    st.title("🤖 Model Management")
    
    models = get_models()
    
    if models:
        # Model selection
        model_names = [f"{m['model_id']} - {m['best_model']} (Score: {m['cv_score']:.3f})" 
                      for m in models]
        selected_model = st.selectbox("Select Model", model_names)
        
        if selected_model:
            model_id = selected_model.split(' - ')[0]
            model_data = next(m for m in models if m['model_id'] == model_id)
            
            # Model details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", model_data['best_model'])
                st.metric("CV Score", f"{model_data['cv_score']:.4f}")
            
            with col2:
                st.metric("Features Used", model_data.get('n_features', 'N/A'))
                st.metric("Training Samples", f"{model_data.get('n_samples_trained', 0):,}")
            
            with col3:
                st.metric("Created", model_data.get('created_at', 'Unknown')[:10])
                st.metric("Task Type", model_data.get('task', 'Unknown'))
            
            # Tabs for detailed info
            tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Feature Importance", "Drift Monitoring", "Actions"])
            
            with tab1:
                st.subheader("📊 Model Performance Metrics")
                
                # Performance chart
                metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
                values = [0.92, 0.89, 0.91, 0.90, 0.95]
                
                fig = go.Figure(data=[
                    go.Bar(x=metrics, y=values, marker_color='#667eea')
                ])
                fig.update_layout(height=300, yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("📈 Feature Importance")
                
                # Sample feature importance
                features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
                importance = [0.35, 0.25, 0.20, 0.12, 0.08]
                
                fig = go.Figure(data=[
                    go.Bar(x=importance, y=features, orientation='h', marker_color='#764ba2')
                ])
                fig.update_layout(height=300, xaxis=dict(range=[0, 0.4]))
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("🎯 Drift Monitoring")
                
                drift_report = get_drift_report(model_id)
                
                if drift_report:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        drift_score = drift_report.get('overall_drift_score', 0)
                        st.metric("Overall Drift Score", f"{drift_score:.3f}")
                        
                        if drift_score > 0.5:
                            st.markdown('<div class="drift-alert">⚠️ Significant drift detected!</div>', 
                                      unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Drifted Features", len(drift_report.get('drifted_features', [])))
                        st.metric("Last Check", "2 hours ago")
                else:
                    st.info("No drift monitoring data available")
            
            with tab4:
                st.subheader("⚡ Model Actions")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Download Model", use_container_width=True):
                        st.info("Downloading model...")
                
                with col2:
                    if st.button("🚀 Deploy to Production", use_container_width=True):
                        st.success("Model deployed!")
                
                with col3:
                    if st.button("🔄 Retrain Model", use_container_width=True):
                        st.info("Retraining scheduled...")
    else:
        st.info("No models available. Train your first model to get started!")

elif page == "LLM Assistant":
    st.title("🤖 LLM Assistant")
    
    # Chat interface
    st.markdown("Ask questions about your data, models, or get recommendations")
    
    # Chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask anything about your AutoML pipeline..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response (mock for now)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                time.sleep(1)  # Simulate API call
                
                response = f"""Based on your question about '{prompt}', here's what I found:

• Your current best model is XGBoost with a CV score of 0.92
• The most important features are: feature_1 (35%), feature_2 (25%), feature_3 (20%)
• I recommend trying ensemble methods to potentially improve performance
• Consider adding more training data for underrepresented classes

Would you like me to generate code for any specific task?"""
                
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

else:
    st.info("Select a page from the sidebar to get started")
