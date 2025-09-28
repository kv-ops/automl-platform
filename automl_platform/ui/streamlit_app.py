"""
Streamlit UI for AutoML Platform
Interactive dashboard with LLM integration
"""

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import time
import requests
from typing import Dict, Any, List, Optional
import base64
import io

# Configure Streamlit
st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except (StreamlitSecretNotFoundError, AttributeError, KeyError):
        return default


API_BASE_URL = _get_secret("API_BASE_URL", "http://localhost:8000")
API_KEY = _get_secret("API_KEY", "")


class AutoMLDashboard:
    """Main dashboard application."""
    
    def __init__(self):
        self.init_session_state()
        self.api_headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    
    def init_session_state(self):
        """Initialize session state variables."""
        if 'current_experiment' not in st.session_state:
            st.session_state.current_experiment = None
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'training_status' not in st.session_state:
            st.session_state.training_status = 'idle'
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = []
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = None
        if 'feature_suggestions' not in st.session_state:
            st.session_state.feature_suggestions = []
    
    def run(self):
        """Run the main application."""
        # Header
        st.title("🤖 AutoML Platform")
        st.markdown("**Intelligent Machine Learning with LLM Assistance**")
        
        # Sidebar
        with st.sidebar:
            self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Upload & Quality",
            "🎯 Model Training",
            "📈 Leaderboard",
            "🔍 Model Analysis",
            "💬 AI Assistant",
            "📋 Reports"
        ])
        
        with tab1:
            self.render_data_tab()
        
        with tab2:
            self.render_training_tab()
        
        with tab3:
            self.render_leaderboard_tab()
        
        with tab4:
            self.render_analysis_tab()
        
        with tab5:
            self.render_chat_tab()
        
        with tab6:
            self.render_reports_tab()
    
    def render_sidebar(self):
        """Render sidebar with experiment info and controls."""
        st.header("🎛️ Control Panel")
        
        # Experiment status
        if st.session_state.current_experiment:
            st.success(f"📂 Experiment: {st.session_state.current_experiment}")
            
            # Training status indicator
            if st.session_state.training_status == 'training':
                st.info("🔄 Training in progress...")
                st.progress(0.5)  # Would be updated with real progress
            elif st.session_state.training_status == 'completed':
                st.success("✅ Training completed!")
            
            # Quick metrics
            if st.session_state.models_trained:
                st.metric("Models Trained", len(st.session_state.models_trained))
                best_score = max([m.get('score', 0) for m in st.session_state.models_trained])
                st.metric("Best Score", f"{best_score:.4f}")
        else:
            st.info("No active experiment")
        
        st.divider()
        
        # Configuration
        st.subheader("⚙️ Configuration")
        
        # API Settings
        with st.expander("API Settings"):
            api_url = st.text_input("API URL", value=API_BASE_URL)
            api_key = st.text_input("API Key", value=API_KEY, type="password")
            
            if st.button("Test Connection"):
                if self.test_api_connection(api_url):
                    st.success("✅ Connected!")
                else:
                    st.error("❌ Connection failed")
        
        # Model Settings
        with st.expander("Model Settings"):
            st.selectbox("Task Type", ["auto", "classification", "regression"])
            st.multiselect("Algorithms", 
                          ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "Neural Network"],
                          default=["RandomForest", "XGBoost"])
            st.slider("CV Folds", 3, 10, 5)
            st.slider("Max Training Time (min)", 1, 60, 10)
        
        # LLM Settings
        with st.expander("AI Assistant Settings"):
            st.selectbox("LLM Provider", ["OpenAI GPT-4", "Anthropic Claude", "Local Model"])
            st.slider("Temperature", 0.0, 1.0, 0.7)
            st.checkbox("Enable Auto Feature Engineering", value=True)
            st.checkbox("Enable Data Cleaning Agent", value=True)
        
        st.divider()
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        with col2:
            if st.button("📥 Export Config"):
                self.export_configuration()
    
    def render_data_tab(self):
        """Render data upload and quality assessment tab."""
        st.header("📊 Data Upload & Quality Assessment")
        
        # File upload
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'parquet', 'json'],
                help="Upload your dataset for analysis"
            )
            
            if uploaded_file is not None:
                # Load data
                df = self.load_data(uploaded_file)
                st.session_state.uploaded_data = df
                
                # Display data info
                st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Data preview
                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df.head(100), use_container_width=True)
                
                # Data quality assessment
                st.subheader("🔍 Data Quality Assessment")
                
                quality_score, issues = self.assess_data_quality(df)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Quality Score", f"{quality_score:.1f}/100",
                             delta=f"{quality_score-70:.1f}" if quality_score > 70 else None)
                with col2:
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    st.metric("Missing Data", f"{missing_pct:.1f}%")
                with col3:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    st.metric("Numeric Features", len(numeric_cols))
                with col4:
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    st.metric("Categorical Features", len(categorical_cols))
                
                # Issues and recommendations
                if issues:
                    st.warning(f"⚠️ Found {len(issues)} data quality issues")
                    with st.expander("View Issues and Recommendations"):
                        for issue in issues:
                            st.write(f"• {issue}")
                
                # Feature statistics
                st.subheader("📊 Feature Statistics")
                
                # Numeric features
                if len(numeric_cols) > 0:
                    st.write("**Numeric Features:**")
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                
                # Categorical features
                if len(categorical_cols) > 0:
                    st.write("**Categorical Features:**")
                    cat_stats = pd.DataFrame({
                        'Feature': categorical_cols,
                        'Unique Values': [df[col].nunique() for col in categorical_cols],
                        'Most Common': [df[col].mode()[0] if not df[col].mode().empty else 'N/A' 
                                      for col in categorical_cols],
                        'Missing %': [(df[col].isnull().sum() / len(df)) * 100 
                                    for col in categorical_cols]
                    })
                    st.dataframe(cat_stats, use_container_width=True)
                
                # Correlation heatmap
                if len(numeric_cols) > 1:
                    st.subheader("🔥 Correlation Heatmap")
                    fig = px.imshow(df[numeric_cols].corr(),
                                  text_auto=True,
                                  aspect="auto",
                                  color_continuous_scale='RdBu_r')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # LLM-powered data cleaning
                if st.button("🤖 AI-Powered Data Cleaning"):
                    with st.spinner("Analyzing data quality with AI..."):
                        cleaned_df, cleaning_report = self.ai_clean_data(df)
                        st.session_state.uploaded_data = cleaned_df
                        
                        st.success("✅ Data cleaning completed!")
                        st.write(cleaning_report)
                        
                        # Show before/after comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Before:**")
                            st.write(f"Shape: {df.shape}")
                            st.write(f"Missing: {df.isnull().sum().sum()}")
                        with col2:
                            st.write("**After:**")
                            st.write(f"Shape: {cleaned_df.shape}")
                            st.write(f"Missing: {cleaned_df.isnull().sum().sum()}")
        
        with col2:
            # Quick insights
            st.subheader("💡 Quick Insights")
            
            if st.session_state.uploaded_data is not None:
                df = st.session_state.uploaded_data
                
                # Dataset characteristics
                st.info(f"""
                **Dataset Characteristics:**
                - Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
                - Duplicated Rows: {df.duplicated().sum()}
                - Complete Rows: {df.dropna().shape[0]}
                """)
                
                # Target column selection
                st.subheader("🎯 Target Selection")
                target_col = st.selectbox(
                    "Select target column",
                    options=[''] + list(df.columns),
                    help="Choose the column you want to predict"
                )
                
                if target_col:
                    # Target distribution
                    st.write("**Target Distribution:**")
                    if df[target_col].dtype in ['int64', 'float64']:
                        fig = px.histogram(df, x=target_col, nbins=30)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.pie(values=df[target_col].value_counts().values,
                                    names=df[target_col].value_counts().index)
                        st.plotly_chart(fig, use_container_width=True)
    
    def render_training_tab(self):
        """Render model training tab."""
        st.header("🎯 Model Training")
        
        if st.session_state.uploaded_data is None:
            st.warning("⚠️ Please upload data first in the Data Upload tab")
            return
        
        df = st.session_state.uploaded_data
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            target_column = st.selectbox(
                "Target Column",
                options=list(df.columns),
                help="Select the column to predict"
            )
        
        with col2:
            experiment_name = st.text_input(
                "Experiment Name",
                value=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Name for this experiment"
            )
        
        with col3:
            task_type = st.selectbox(
                "Task Type",
                ["auto", "classification", "regression"]
            )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                algorithms = st.multiselect(
                    "Algorithms",
                    ["RandomForest", "XGBoost", "LightGBM", "CatBoost", 
                     "LogisticRegression", "SVM", "NeuralNetwork"],
                    default=["RandomForest", "XGBoost", "LightGBM"]
                )
                
                cv_folds = st.slider("Cross-validation Folds", 3, 10, 5)
            
            with col2:
                max_runtime = st.slider("Max Runtime (minutes)", 1, 60, 10)
                enable_hpo = st.checkbox("Enable Hyperparameter Optimization", value=True)
                hpo_iterations = st.slider("HPO Iterations", 10, 100, 30)
            
            with col3:
                enable_feature_engineering = st.checkbox("Auto Feature Engineering", value=True)
                enable_ensemble = st.checkbox("Enable Ensemble", value=True)
                handle_imbalance = st.checkbox("Handle Class Imbalance", value=True)
        
        # Feature engineering suggestions
        if enable_feature_engineering:
            st.subheader("🔧 AI-Suggested Features")
            
            if st.button("Get Feature Suggestions"):
                with st.spinner("Generating feature suggestions with AI..."):
                    suggestions = self.get_feature_suggestions(df, target_column)
                    st.session_state.feature_suggestions = suggestions
            
            if st.session_state.feature_suggestions:
                for i, suggestion in enumerate(st.session_state.feature_suggestions[:5]):
                    with st.expander(f"Feature {i+1}: {suggestion['name']}"):
                        st.write(f"**Description:** {suggestion['description']}")
                        st.write(f"**Importance:** {suggestion['importance']}")
                        st.code(suggestion['code'], language='python')
                        
                        if st.button(f"Apply Feature {i+1}", key=f"apply_feat_{i}"):
                            # Apply feature engineering
                            st.success(f"Applied: {suggestion['name']}")
        
        # Start training
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                st.session_state.current_experiment = experiment_name
                st.session_state.training_status = 'training'
                
                # Training progress
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                # Simulate training with progress updates
                with st.spinner("Training models..."):
                    for i in range(101):
                        progress_placeholder.progress(i / 100)
                        
                        if i % 20 == 0:
                            status_placeholder.info(f"Training model {i//20 + 1}/5...")
                        
                        time.sleep(0.05)  # Simulate processing
                
                st.session_state.training_status = 'completed'
                
                # Mock results
                st.session_state.models_trained = [
                    {"model": "XGBoost", "score": 0.92, "time": 45},
                    {"model": "LightGBM", "score": 0.91, "time": 38},
                    {"model": "RandomForest", "score": 0.89, "time": 52},
                    {"model": "CatBoost", "score": 0.90, "time": 48},
                    {"model": "LogisticRegression", "score": 0.85, "time": 12}
                ]
                
                st.success("✅ Training completed successfully!")
                st.balloons()
    
    def render_leaderboard_tab(self):
        """Render model leaderboard tab."""
        st.header("📈 Model Leaderboard")
        
        if not st.session_state.models_trained:
            st.info("No models trained yet. Start training in the Model Training tab.")
            return
        
        # Convert to DataFrame
        leaderboard_df = pd.DataFrame(st.session_state.models_trained)
        leaderboard_df = leaderboard_df.sort_values('score', ascending=False)
        leaderboard_df['rank'] = range(1, len(leaderboard_df) + 1)
        
        # Display leaderboard
        st.dataframe(
            leaderboard_df[['rank', 'model', 'score', 'time']],
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Score comparison
            fig = px.bar(leaderboard_df, x='model', y='score',
                        title='Model Performance Comparison',
                        color='score',
                        color_continuous_scale='viridis')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time vs Performance
            fig = px.scatter(leaderboard_df, x='time', y='score',
                           text='model',
                           title='Training Time vs Performance',
                           size='score',
                           color='score',
                           color_continuous_scale='viridis')
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        
        # Model selection
        st.subheader("🎯 Model Selection")
        
        selected_model = st.selectbox(
            "Select model for deployment",
            options=leaderboard_df['model'].tolist()
        )
        
        if st.button("Select Model"):
            st.session_state.selected_model = selected_model
            st.success(f"✅ Selected {selected_model} for deployment")
    
    def render_analysis_tab(self):
        """Render model analysis tab."""
        st.header("🔍 Model Analysis")
        
        if not st.session_state.selected_model:
            st.info("Please select a model from the Leaderboard tab first.")
            return
        
        st.subheader(f"Analysis for {st.session_state.selected_model}")
        
        # Feature importance
        st.subheader("📊 Feature Importance")
        
        # Mock feature importance data
        features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        importance = [0.35, 0.25, 0.20, 0.12, 0.08]
        
        fig = px.bar(x=importance, y=features, orientation='h',
                    title='Top 5 Important Features',
                    labels={'x': 'Importance', 'y': 'Features'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Model explanation
        with st.expander("🤖 AI Model Explanation"):
            explanation = """
            **Model Performance Summary:**
            
            The XGBoost model achieved excellent performance with 92% accuracy.
            The model primarily relies on feature_1 and feature_2, which together 
            account for 60% of the predictive power.
            
            **Key Insights:**
            - Strong correlation between feature_1 and the target variable
            - feature_2 shows non-linear relationships captured well by tree-based model
            - Model performs best on mid-range values, with slight degradation on extremes
            
            **Recommendations:**
            - Consider ensemble with linear model for better extreme value handling
            - Monitor feature_1 for drift as it's the most important predictor
            - Retrain quarterly to maintain performance
            """
            st.markdown(explanation)
        
        # Error analysis
        st.subheader("📉 Error Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix for classification
            st.write("**Confusion Matrix**")
            cm_data = [[85, 15], [10, 90]]
            fig = px.imshow(cm_data, text_auto=True,
                          labels=dict(x="Predicted", y="Actual"),
                          color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residuals plot for regression
            st.write("**Residuals Distribution**")
            residuals = np.random.normal(0, 1, 100)
            fig = px.histogram(residuals, nbins=30,
                             title="Prediction Residuals")
            st.plotly_chart(fig, use_container_width=True)
        
        # SHAP values
        st.subheader("🎯 SHAP Explanation")
        st.info("SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions")
        
        # Mock SHAP waterfall
        st.image("https://shap.readthedocs.io/en/latest/_images/waterfall_plot.png",
                caption="Example SHAP Waterfall Plot")
    
    def render_chat_tab(self):
        """Render AI assistant chat tab."""
        st.header("💬 AI Assistant")
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>You:</b> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <b>AI:</b> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Input area
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask a question about your data or models...",
                key="chat_input",
                placeholder="e.g., 'What features should I engineer for better performance?'"
            )
        
        with col2:
            send_button = st.button("Send", use_container_width=True)
        
        if send_button and user_input:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Get AI response (mock)
            with st.spinner("AI is thinking..."):
                time.sleep(1)  # Simulate API call
                
                ai_response = self.get_ai_response(user_input)
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': ai_response
                })
            
            st.rerun()
        
        # Quick prompts
        st.subheader("💡 Quick Prompts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Explain best model"):
                self.add_quick_prompt("Explain why the best model performs well")
        
        with col2:
            if st.button("Suggest improvements"):
                self.add_quick_prompt("How can I improve model performance?")
        
        with col3:
            if st.button("Feature ideas"):
                self.add_quick_prompt("Suggest new features to engineer")
    
    def render_reports_tab(self):
        """Render reports generation tab."""
        st.header("📋 Reports")
        
        if not st.session_state.current_experiment:
            st.info("No experiment to report on. Complete training first.")
            return
        
        st.subheader("📄 Generate Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Executive Summary", "Technical Report", "Model Card", "Compliance Report"]
            )
            
            format_type = st.selectbox(
                "Output Format",
                ["PDF", "HTML", "Markdown", "PowerPoint"]
            )
        
        with col2:
            include_visuals = st.checkbox("Include Visualizations", value=True)
            include_code = st.checkbox("Include Code Snippets", value=False)
            include_recommendations = st.checkbox("Include AI Recommendations", value=True)
        
        if st.button("📥 Generate Report", type="primary"):
            with st.spinner(f"Generating {report_type}..."):
                time.sleep(2)  # Simulate generation
                
                # Mock report content
                report_content = f"""
                # {report_type} - {st.session_state.current_experiment}
                
                ## Executive Summary
                The AutoML experiment successfully trained 5 models with the best achieving 92% accuracy.
                
                ## Key Findings
                - XGBoost emerged as the top performer
                - Feature engineering improved baseline by 8%
                - Model is production-ready with monitoring
                
                ## Recommendations
                1. Deploy XGBoost model to production
                2. Set up drift monitoring for top 3 features
                3. Retrain quarterly with new data
                """
                
                st.success(f"✅ {report_type} generated successfully!")
                
                # Display preview
                with st.expander("Preview Report"):
                    st.markdown(report_content)
                
                # Download button
                st.download_button(
                    label=f"Download {format_type}",
                    data=report_content,
                    file_name=f"{st.session_state.current_experiment}_report.md",
                    mime="text/markdown"
                )
        
        # Previous reports
        st.subheader("📚 Previous Reports")
        
        reports_data = [
            {"name": "exp_20240115_report.pdf", "date": "2024-01-15", "type": "Technical"},
            {"name": "exp_20240110_summary.html", "date": "2024-01-10", "type": "Executive"},
            {"name": "exp_20240105_model.md", "date": "2024-01-05", "type": "Model Card"}
        ]
        
        reports_df = pd.DataFrame(reports_data)
        st.dataframe(reports_df, use_container_width=True, hide_index=True)
    
    # Helper methods
    
    def test_api_connection(self, url: str) -> bool:
        """Test API connection."""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def load_data(self, file) -> pd.DataFrame:
        """Load data from uploaded file."""
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        elif file.name.endswith('.parquet'):
            return pd.read_parquet(file)
        elif file.name.endswith('.json'):
            return pd.read_json(file)
        else:
            st.error("Unsupported file format")
            return pd.DataFrame()
    
    def assess_data_quality(self, df: pd.DataFrame) -> tuple:
        """Assess data quality and return score and issues."""
        issues = []
        score = 100
        
        # Check for missing values
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 30:
            issues.append(f"High missing data percentage: {missing_pct:.1f}%")
            score -= 20
        elif missing_pct > 10:
            issues.append(f"Moderate missing data: {missing_pct:.1f}%")
            score -= 10
        
        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate rows")
            score -= 10
        
        # Check for high cardinality
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 0.5 * len(df):
                issues.append(f"High cardinality in {col}: {df[col].nunique()} unique values")
                score -= 5
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append(f"Constant column: {col}")
                score -= 5
        
        return max(score, 0), issues
    
    def ai_clean_data(self, df: pd.DataFrame) -> tuple:
        """Clean data using AI suggestions."""
        # Mock cleaning
        cleaned_df = df.copy()
        
        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle missing values
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['float64', 'int64']:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown', inplace=True)
        
        report = """
        **Data Cleaning Report:**
        - Removed duplicate rows
        - Imputed missing numeric values with median
        - Imputed missing categorical values with mode
        - Standardized column names
        - Removed constant columns
        """
        
        return cleaned_df, report
    
    def get_feature_suggestions(self, df: pd.DataFrame, target: str) -> List[Dict]:
        """Get AI-powered feature suggestions."""
        # Mock suggestions
        return [
            {
                "name": "age_income_ratio",
                "description": "Ratio of age to income for purchasing power analysis",
                "importance": "high",
                "code": "df['age_income_ratio'] = df['age'] / (df['income'] + 1)"
            },
            {
                "name": "income_squared",
                "description": "Polynomial feature to capture non-linear income effects",
                "importance": "medium",
                "code": "df['income_squared'] = df['income'] ** 2"
            },
            {
                "name": "age_group",
                "description": "Categorical age groups for better segmentation",
                "importance": "high",
                "code": "df['age_group'] = pd.cut(df['age'], bins=[0,25,35,50,100], labels=['Young','Middle','Senior','Elder'])"
            }
        ]
    
    def get_ai_response(self, query: str) -> str:
        """Get AI assistant response."""
        # Mock responses based on query content
        if "explain" in query.lower():
            return "The model performs well due to strong feature engineering and proper hyperparameter tuning. The XGBoost algorithm captures non-linear relationships effectively."
        elif "improve" in query.lower():
            return "To improve performance: 1) Engineer interaction features, 2) Try ensemble methods, 3) Collect more training data, 4) Implement cross-validation with stratification."
        elif "feature" in query.lower():
            return "Consider creating: ratio features, polynomial terms, time-based aggregations, and domain-specific transformations based on business logic."
        else:
            return "I can help you with model explanations, performance improvements, feature engineering, and data quality analysis. What would you like to know?"
    
    def add_quick_prompt(self, prompt: str):
        """Add a quick prompt to chat."""
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        
        ai_response = self.get_ai_response(prompt)
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': ai_response
        })
        
        st.rerun()
    
    def export_configuration(self):
        """Export current configuration."""
        config = {
            "experiment": st.session_state.current_experiment,
            "models_trained": st.session_state.models_trained,
            "timestamp": datetime.now().isoformat()
        }
        
        st.download_button(
            label="Download Config",
            data=json.dumps(config, indent=2),
            file_name="automl_config.json",
            mime="application/json"
        )


# Run the app
if __name__ == "__main__":
    app = AutoMLDashboard()
    app.run()
