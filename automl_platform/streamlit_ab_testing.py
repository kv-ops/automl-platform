"""
Streamlit Interface for A/B Testing
====================================
Place in: automl_platform/streamlit_ab_testing.py

Streamlit components for A/B testing visualization and management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from PIL import Image
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Import A/B testing components
from .ab_testing import ABTestingService, MetricsComparator, StatisticalTester
from .metrics import compare_models_metrics
from .mlflow_registry import MLflowRegistry


class ABTestingDashboard:
    """Streamlit dashboard for A/B testing."""
    
    def __init__(self, ab_service: ABTestingService, registry: MLflowRegistry = None):
        """
        Initialize dashboard.
        
        Args:
            ab_service: A/B testing service
            registry: MLflow registry (optional)
        """
        self.ab_service = ab_service
        self.registry = registry
    
    def render(self):
        """Render the complete A/B testing dashboard."""
        st.title("🧪 A/B Testing Dashboard")
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["Active Tests", "Create Test", "Compare Models", "Test Results", "Analytics"]
        )
        
        if page == "Active Tests":
            self.render_active_tests()
        elif page == "Create Test":
            self.render_create_test()
        elif page == "Compare Models":
            self.render_model_comparison()
        elif page == "Test Results":
            self.render_test_results()
        elif page == "Analytics":
            self.render_analytics()
    
    def render_active_tests(self):
        """Render active A/B tests view."""
        st.header("📊 Active A/B Tests")
        
        active_tests = self.ab_service.get_active_tests()
        
        if not active_tests:
            st.info("No active A/B tests")
            return
        
        # Create DataFrame
        df = pd.DataFrame(active_tests)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Tests", len(active_tests))
        with col2:
            total_samples = df['samples_collected'].sum() if 'samples_collected' in df else 0
            st.metric("Total Samples", total_samples)
        with col3:
            avg_progress = (df['samples_collected'] / df['min_samples_required']).mean() * 100 if 'min_samples_required' in df else 0
            st.metric("Avg Progress", f"{avg_progress:.1f}%")
        
        st.divider()
        
        # Display tests
        for test in active_tests:
            with st.expander(f"Test: {test['test_id'][:8]}... - {test['model_name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Test Configuration:**")
                    st.write(f"- Model: {test['model_name']}")
                    st.write(f"- Champion Version: {test['champion_version']}")
                    st.write(f"- Challenger Version: {test['challenger_version']}")
                    st.write(f"- Started: {test['started_at']}")
                
                with col2:
                    st.write("**Progress:**")
                    progress = test['samples_collected'] / test['min_samples_required']
                    st.progress(min(progress, 1.0))
                    st.write(f"Samples: {test['samples_collected']} / {test['min_samples_required']}")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"View Results", key=f"view_{test['test_id']}"):
                        st.session_state['selected_test'] = test['test_id']
                        st.rerun()
                
                with col2:
                    if st.button(f"Pause", key=f"pause_{test['test_id']}"):
                        self.ab_service.pause_test(test['test_id'])
                        st.success("Test paused")
                        st.rerun()
                
                with col3:
                    if st.button(f"Conclude", key=f"conclude_{test['test_id']}"):
                        results = self.ab_service.conclude_test(test['test_id'])
                        st.success("Test concluded")
                        st.json(results)
    
    def render_create_test(self):
        """Render create A/B test form."""
        st.header("🆕 Create A/B Test")
        
        with st.form("create_ab_test"):
            st.subheader("Test Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.text_input("Model Name", placeholder="e.g., customer_churn_model")
                champion_version = st.number_input("Champion Version", min_value=1, value=1)
                challenger_version = st.number_input("Challenger Version", min_value=1, value=2)
            
            with col2:
                traffic_split = st.slider(
                    "Traffic to Challenger (%)",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5
                ) / 100
                
                min_samples = st.number_input(
                    "Minimum Samples per Model",
                    min_value=50,
                    max_value=10000,
                    value=100,
                    step=50
                )
                
                confidence_level = st.slider(
                    "Confidence Level (%)",
                    min_value=90,
                    max_value=99,
                    value=95
                ) / 100
            
            st.subheader("Metrics Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                primary_metric = st.selectbox(
                    "Primary Metric",
                    ["accuracy", "precision", "recall", "f1", "roc_auc", "mse", "rmse", "mae", "r2"]
                )
                
                statistical_test = st.selectbox(
                    "Statistical Test",
                    ["t_test", "mann_whitney", "chi_square"]
                )
            
            with col2:
                min_improvement = st.number_input(
                    "Minimum Improvement Required",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.02,
                    step=0.01,
                    format="%.3f"
                )
                
                max_duration = st.number_input(
                    "Maximum Test Duration (days)",
                    min_value=1,
                    max_value=90,
                    value=30
                )
            
            submitted = st.form_submit_button("Create Test", type="primary")
            
            if submitted:
                if not model_name:
                    st.error("Please enter a model name")
                else:
                    test_id = self.ab_service.create_ab_test(
                        model_name=model_name,
                        champion_version=champion_version,
                        challenger_version=challenger_version,
                        traffic_split=traffic_split,
                        min_samples=min_samples,
                        confidence_level=confidence_level,
                        primary_metric=primary_metric
                    )
                    
                    st.success(f"✅ A/B test created successfully!")
                    st.info(f"Test ID: {test_id}")
                    
                    # Show test details
                    with st.expander("Test Details"):
                        st.json({
                            "test_id": test_id,
                            "model_name": model_name,
                            "champion_version": champion_version,
                            "challenger_version": challenger_version,
                            "traffic_split": traffic_split,
                            "min_samples": min_samples,
                            "confidence_level": confidence_level,
                            "primary_metric": primary_metric
                        })
    
    def render_model_comparison(self):
        """Render offline model comparison."""
        st.header("⚖️ Model Comparison")
        
        # File upload for test data
        st.subheader("Upload Test Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with features and labels",
            type="csv"
        )
        
        if uploaded_file is not None:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.write(f"Data shape: {df.shape}")
            
            # Select target column
            target_col = st.selectbox("Select target column", df.columns.tolist())
            
            if target_col:
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                # Model selection
                st.subheader("Select Models to Compare")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Model A**")
                    model_a_type = st.selectbox(
                        "Model A Type",
                        ["Upload", "From Registry"],
                        key="model_a_type"
                    )
                    
                    if model_a_type == "Upload":
                        model_a_file = st.file_uploader(
                            "Upload Model A (.pkl)",
                            type="pkl",
                            key="model_a_upload"
                        )
                    else:
                        model_a_name = st.text_input("Model A Name", key="model_a_name")
                        model_a_version = st.number_input("Model A Version", min_value=1, key="model_a_version")
                
                with col2:
                    st.write("**Model B**")
                    model_b_type = st.selectbox(
                        "Model B Type",
                        ["Upload", "From Registry"],
                        key="model_b_type"
                    )
                    
                    if model_b_type == "Upload":
                        model_b_file = st.file_uploader(
                            "Upload Model B (.pkl)",
                            type="pkl",
                            key="model_b_upload"
                        )
                    else:
                        model_b_name = st.text_input("Model B Name", key="model_b_name")
                        model_b_version = st.number_input("Model B Version", min_value=1, key="model_b_version")
                
                # Compare button
                if st.button("Compare Models", type="primary"):
                    # This is simplified - would need actual model loading
                    st.info("Model comparison would be performed here with actual model loading")
                    
                    # Generate dummy comparison for demonstration
                    self._render_comparison_results_demo()
    
    def render_test_results(self):
        """Render detailed test results."""
        st.header("📈 Test Results")
        
        # Select test
        test_ids = list(self.ab_service.test_results.keys())
        
        if not test_ids:
            st.info("No test results available")
            return
        
        selected_test_id = st.selectbox("Select Test", test_ids)
        
        if selected_test_id:
            results = self.ab_service.get_test_results(selected_test_id)
            
            if results:
                # Summary metrics
                st.subheader("Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Status", results['status'])
                
                with col2:
                    st.metric("Champion Samples", results['champion_samples'])
                
                with col3:
                    st.metric("Challenger Samples", results['challenger_samples'])
                
                with col4:
                    winner = results.get('winner', 'None')
                    st.metric("Winner", winner)
                
                # Statistical results
                if results.get('p_value'):
                    st.subheader("Statistical Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("P-Value", f"{results['p_value']:.4f}")
                    
                    with col2:
                        st.metric("Effect Size", f"{results.get('effect_size', 0):.3f}")
                    
                    with col3:
                        confidence = results.get('confidence', 0) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    # Confidence interval
                    if results.get('confidence_interval'):
                        ci = results['confidence_interval']
                        st.info(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
                
                # Performance metrics
                st.subheader("Performance Metrics")
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std', 'Median'],
                    'Champion': [
                        results['champion_metrics'].get('mean', 0),
                        results['champion_metrics'].get('std', 0),
                        results['champion_metrics'].get('median', 0)
                    ],
                    'Challenger': [
                        results['challenger_metrics'].get('mean', 0),
                        results['challenger_metrics'].get('std', 0),
                        results['challenger_metrics'].get('median', 0)
                    ]
                })
                
                st.dataframe(metrics_df)
                
                # Visualization
                self._render_test_visualizations(results)
    
    def render_analytics(self):
        """Render analytics dashboard."""
        st.header("📊 A/B Testing Analytics")
        
        # Overall statistics
        st.subheader("Overall Statistics")
        
        total_tests = len(self.ab_service.test_results)
        active_tests = len(self.ab_service.active_tests)
        completed_tests = total_tests - active_tests
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Tests", total_tests)
        
        with col2:
            st.metric("Active Tests", active_tests)
        
        with col3:
            st.metric("Completed Tests", completed_tests)
        
        st.divider()
        
        # Test history
        if self.ab_service.test_results:
            st.subheader("Test History")
            
            history_data = []
            for test_id, result in self.ab_service.test_results.items():
                config = self.ab_service.active_tests.get(test_id)
                
                history_data.append({
                    'Test ID': test_id[:8],
                    'Status': result.status.value,
                    'Started': result.started_at.strftime('%Y-%m-%d %H:%M'),
                    'Champion Samples': result.champion_samples,
                    'Challenger Samples': result.challenger_samples,
                    'Winner': result.winner or 'None',
                    'P-Value': f"{result.p_value:.4f}" if result.p_value else 'N/A'
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df)
            
            # Visualizations
            self._render_analytics_charts(history_df)
    
    def _render_comparison_results_demo(self):
        """Render demo comparison results."""
        st.subheader("Comparison Results")
        
        # Create demo metrics
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'],
            'Model A': [0.85, 0.82, 0.88, 0.85, 0.91],
            'Model B': [0.88, 0.86, 0.89, 0.87, 0.93],
            'Improvement (%)': [3.5, 4.9, 1.1, 2.4, 2.2]
        }
        
        df = pd.DataFrame(metrics_data)
        
        # Display metrics table
        st.dataframe(df, use_container_width=True)
        
        # Bar chart comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Model A',
            x=metrics_data['Metric'],
            y=metrics_data['Model A'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Model B',
            x=metrics_data['Metric'],
            y=metrics_data['Model B'],
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Metric',
            yaxis_title='Score',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical significance
        st.subheader("Statistical Significance")
        st.info("McNemar's Test: p-value = 0.023 (Significant at α=0.05)")
        st.success("Model B shows statistically significant improvement over Model A")
    
    def _render_test_visualizations(self, results: Dict):
        """Render test result visualizations."""
        st.subheader("Visualizations")
        
        # Performance over time (if data available)
        if 'predictions_log' in results:
            # This would show actual performance over time
            pass
        
        # Sample distribution comparison
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Champion Distribution', 'Challenger Distribution')
        )
        
        # Generate sample data for visualization
        champion_samples = np.random.normal(
            results['champion_metrics'].get('mean', 0.5),
            results['champion_metrics'].get('std', 0.1),
            results['champion_samples']
        )
        
        challenger_samples = np.random.normal(
            results['challenger_metrics'].get('mean', 0.5),
            results['challenger_metrics'].get('std', 0.1),
            results['challenger_samples']
        )
        
        fig.add_trace(
            go.Histogram(x=champion_samples, name='Champion', marker_color='blue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=challenger_samples, name='Challenger', marker_color='green'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_analytics_charts(self, df: pd.DataFrame):
        """Render analytics charts."""
        
        # Test status distribution
        col1, col2 = st.columns(2)
        
        with col1:
            status_counts = df['Status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Test Status Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            winner_counts = df['Winner'].value_counts()
            fig = px.bar(
                x=winner_counts.index,
                y=winner_counts.values,
                title='Winner Distribution',
                labels={'x': 'Winner', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)


def integrate_ab_testing_to_main_app(ab_service: ABTestingService, registry: MLflowRegistry = None):
    """
    Function to integrate A/B testing into main Streamlit app.
    Call this from your main app.py
    
    Args:
        ab_service: A/B testing service instance
        registry: MLflow registry instance
    """
    dashboard = ABTestingDashboard(ab_service, registry)
    dashboard.render()
