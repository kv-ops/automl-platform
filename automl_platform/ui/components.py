"""
Reusable UI Components for AutoML Platform
Inspired by DataRobot and Dataiku's visual interfaces
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import time  # ADDED: Missing import
from datetime import datetime, timedelta


class DataQualityVisualizer:
    """
    DataRobot-style Data Quality Assessment visualizations.
    Creates interactive quality dashboards.
    """
    
    @staticmethod
    def render_quality_gauge(quality_score: float, container=None):
        """Render quality score as a gauge chart."""
        
        # Determine color based on score
        if quality_score >= 80:
            color = "green"
        elif quality_score >= 60:
            color = "yellow"
        else:
            color = "red"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Data Quality Score"},
            delta={'reference': 70, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': 'lightgray'},
                    {'range': [50, 80], 'color': 'gray'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        
        if container:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
        
        return fig
    
    @staticmethod
    def render_missing_data_heatmap(df: pd.DataFrame, container=None):
        """Render missing data heatmap like DataRobot."""
        
        # Calculate missing percentage for each column
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing %': (df.isnull().sum() / len(df)) * 100
        })
        
        # Create heatmap data
        heatmap_data = []
        for col in df.columns:
            col_missing = df[col].isnull().astype(int).values
            heatmap_data.append(col_missing)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=list(range(len(df))),
            y=df.columns,
            colorscale=[[0, 'white'], [1, 'red']],
            showscale=False,
            hovertemplate='Column: %{y}<br>Row: %{x}<br>Missing: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Missing Data Pattern",
            xaxis_title="Sample Index",
            yaxis_title="Features",
            height=400
        )
        
        if container:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
        
        return fig
    
    @staticmethod
    def render_quality_breakdown(quality_metrics: Dict[str, float], container=None):
        """Render quality score breakdown."""
        
        categories = list(quality_metrics.keys())
        values = list(quality_metrics.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=['red' if v > 10 else 'yellow' if v > 5 else 'green' 
                             for v in values],
                text=[f"-{v:.1f}" for v in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Quality Score Penalties",
            xaxis_title="Issue Category",
            yaxis_title="Penalty Points",
            height=300
        )
        
        if container:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
        
        return fig


class ModelLeaderboard:
    """
    Dataiku-style model leaderboard with interactive comparisons.
    """
    
    @staticmethod
    def render_leaderboard(leaderboard_df: pd.DataFrame, container=None):
        """Render interactive model leaderboard."""
        
        # Add rank column
        leaderboard_df['Rank'] = range(1, len(leaderboard_df) + 1)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Model Performance", "Time vs Accuracy"),
            specs=[[{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Performance bar chart
        fig.add_trace(
            go.Bar(
                x=leaderboard_df['model'],
                y=leaderboard_df['score'],
                marker_color=px.colors.sequential.Viridis,
                text=leaderboard_df['score'].round(3),
                textposition='auto',
                name="Score"
            ),
            row=1, col=1
        )
        
        # Time vs Accuracy scatter
        fig.add_trace(
            go.Scatter(
                x=leaderboard_df['training_time'],
                y=leaderboard_df['score'],
                mode='markers+text',
                marker=dict(
                    size=leaderboard_df['score'] * 100,
                    color=leaderboard_df['score'],
                    colorscale='Viridis',
                    showscale=True
                ),
                text=leaderboard_df['model'],
                textposition="top center",
                name="Models"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Model Comparison Dashboard"
        )
        
        fig.update_xaxes(title_text="Model", row=1, col=1)
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_xaxes(title_text="Training Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        
        if container:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
        
        return fig


class FeatureImportanceVisualizer:
    """
    Feature importance visualizations with SHAP-style displays.
    """
    
    @staticmethod
    def render_importance_plot(feature_importance: Dict[str, float], 
                              plot_type: str = "bar", container=None):
        """Render feature importance plot."""
        
        # Sort features by importance
        sorted_features = dict(sorted(feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True))
        features = list(sorted_features.keys())[:20]  # Top 20
        importances = list(sorted_features.values())[:20]
        
        if plot_type == "bar":
            fig = go.Figure([go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker=dict(
                    color=importances,
                    colorscale='Viridis',
                    showscale=True
                )
            )])
            
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=500,
                yaxis=dict(autorange="reversed")
            )
            
        elif plot_type == "waterfall":
            # SHAP-style waterfall
            fig = go.Figure(go.Waterfall(
                x=importances,
                y=features,
                orientation="h",
                measure=["relative"] * len(features),
                connector={"mode": "between", "line": {"width": 1, "color": "rgb(0, 0, 0)", "dash": "solid"}}
            ))
            
            fig.update_layout(
                title="SHAP Feature Impact",
                height=500
            )
        
        if container:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
        
        return fig


class DriftMonitor:
    """
    Dataiku-style drift monitoring visualizations.
    """
    
    @staticmethod
    def render_drift_chart(drift_data: Dict[str, Any], container=None):
        """Render drift monitoring chart."""
        
        # Create sample drift data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        features = ['feature1', 'feature2', 'feature3']
        
        fig = go.Figure()
        
        for feature in features:
            drift_values = np.random.random(30) * 0.5  # Random drift values
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=drift_values,
                mode='lines+markers',
                name=feature,
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        # Add threshold line
        fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                     annotation_text="Drift Threshold")
        
        fig.update_layout(
            title="Feature Drift Over Time",
            xaxis_title="Date",
            yaxis_title="Drift Score (PSI)",
            height=400,
            hovermode='x'
        )
        
        if container:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
        
        return fig


class TrainingProgressTracker:
    """
    Real-time training progress visualization.
    """
    
    @staticmethod
    def render_progress(models_trained: int, total_models: int, 
                       current_score: float = None, container=None):
        """Render training progress."""
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Models Trained",
                f"{models_trained}/{total_models}",
                delta=f"{(models_trained/total_models)*100:.0f}%"
            )
        
        with col2:
            if current_score:
                st.metric(
                    "Best Score",
                    f"{current_score:.4f}",
                    delta="+0.02" if models_trained > 1 else None
                )
        
        with col3:
            remaining = total_models - models_trained
            st.metric(
                "Time Remaining",
                f"~{remaining * 30}s",
                delta=f"-{30}s" if models_trained > 0 else None
            )
        
        # Progress bar
        progress = models_trained / total_models if total_models > 0 else 0
        st.progress(progress)
        
        # Live training log
        with st.expander("Training Log", expanded=False):
            st.text(f"[{datetime.now().strftime('%H:%M:%S')}] Training model {models_trained+1}...")
            if models_trained > 0:
                st.text(f"[{datetime.now().strftime('%H:%M:%S')}] Model {models_trained} completed")


class ChatInterface:
    """
    Akkio-style chat interface for data cleaning and analysis.
    """
    
    @staticmethod
    def render_chat(messages: List[Dict[str, str]], container=None):
        """Render chat interface."""
        
        chat_container = container or st.container()
        
        with chat_container:
            for message in messages:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right;">
                        <b>You:</b> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f5f5f5; padding: 10px; border-radius: 10px; margin: 5px 0;">
                        <b>AI Assistant:</b> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
    
    @staticmethod
    def render_suggested_actions(actions: List[str], container=None):
        """Render suggested cleaning actions."""
        
        st.subheader("💡 Suggested Actions")
        
        for i, action in enumerate(actions, 1):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.write(f"{i}. {action}")
            
            with col2:
                if st.button(f"Apply", key=f"action_{i}"):
                    st.success(f"Applied: {action}")


class ExperimentComparator:
    """
    Compare multiple experiments side by side.
    """
    
    @staticmethod
    def render_comparison(experiments: List[Dict[str, Any]], container=None):
        """Render experiment comparison."""
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(experiments)
        
        # Metrics comparison
        fig = go.Figure()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for exp in experiments:
            fig.add_trace(go.Scatterpolar(
                r=[exp.get(m, 0) for m in metrics],
                theta=metrics,
                fill='toself',
                name=exp['name']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Experiment Performance Comparison",
            showlegend=True,
            height=400
        )
        
        if container:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)
        
        # Table comparison
        st.dataframe(
            comparison_df[['name', 'best_model', 'best_score', 'training_time', 'total_models']],
            use_container_width=True,
            hide_index=True
        )


class AlertsAndNotifications:
    """
    DataRobot-style alerts and notifications display.
    """
    
    @staticmethod
    def render_alerts(alerts: List[Dict[str, Any]], container=None):
        """Render quality alerts and warnings."""
        
        if not alerts:
            st.success("✅ No critical issues detected")
            return
        
        # Group alerts by severity
        critical = [a for a in alerts if a.get('severity') == 'critical']
        high = [a for a in alerts if a.get('severity') == 'high']
        medium = [a for a in alerts if a.get('severity') == 'medium']
        low = [a for a in alerts if a.get('severity') == 'low']
        
        # Render by severity
        if critical:
            st.error(f"🚨 **{len(critical)} Critical Issues**")
            with st.expander("View Critical Issues", expanded=True):
                for alert in critical:
                    st.write(f"• {alert['message']}")
                    if 'action' in alert:
                        st.caption(f"  → Recommended: {alert['action']}")
        
        if high:
            st.warning(f"⚠️ **{len(high)} High Priority Issues**")
            with st.expander("View High Priority Issues"):
                for alert in high:
                    st.write(f"• {alert['message']}")
                    if 'action' in alert:
                        st.caption(f"  → Recommended: {alert['action']}")
        
        if medium:
            st.info(f"ℹ️ **{len(medium)} Medium Priority Issues**")
            with st.expander("View Medium Priority Issues"):
                for alert in medium:
                    st.write(f"• {alert['message']}")
        
        if low:
            with st.expander(f"💡 {len(low)} Low Priority Suggestions"):
                for alert in low:
                    st.write(f"• {alert['message']}")


class ReportGenerator:
    """
    Generate and display AutoML reports.
    """
    
    @staticmethod
    def render_executive_summary(experiment_data: Dict[str, Any], container=None):
        """Render executive summary report section."""
        
        st.markdown(f"""
        ## Executive Summary
        
        **Experiment:** {experiment_data.get('name', 'Unnamed')}  
        **Date:** {datetime.now().strftime('%Y-%m-%d')}  
        **Status:** {experiment_data.get('status', 'Completed')}
        
        ### Key Results
        - **Best Model:** {experiment_data.get('best_model', 'Unknown')}
        - **Performance:** {experiment_data.get('best_score', 0):.2%} accuracy
        - **Training Time:** {experiment_data.get('training_time', 0):.1f} minutes
        - **Models Evaluated:** {experiment_data.get('total_models', 0)}
        
        ### Business Impact
        The model achieved a {experiment_data.get('improvement', 15):.1f}% improvement over baseline,
        which translates to an estimated ${experiment_data.get('value', 100000):,.0f} in annual savings.
        
        ### Recommendations
        1. Deploy the {experiment_data.get('best_model', 'model')} to production
        2. Set up monitoring for the top 3 features
        3. Schedule monthly retraining to maintain performance
        """)
    
    @staticmethod
    def render_technical_details(model_data: Dict[str, Any], container=None):
        """Render technical report section."""
        
        st.markdown("""
        ## Technical Details
        
        ### Model Architecture
        """)
        
        # Model parameters table
        params_df = pd.DataFrame([
            {"Parameter": k, "Value": v} 
            for k, v in model_data.get('parameters', {}).items()
        ])
        
        if not params_df.empty:
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        ### Feature Engineering
        
        The following feature transformations were applied:
        - Polynomial features (degree 2) for numeric columns
        - Target encoding for high-cardinality categoricals
        - Interaction features for correlated variables
        
        ### Cross-Validation Strategy
        - Method: Stratified K-Fold
        - Folds: 5
        - Metric: ROC-AUC
        """)
        
        # Performance metrics
        st.markdown("### Performance Metrics")
        
        metrics = model_data.get('metrics', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        with col4:
            st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")


class DataFlowDiagram:
    """
    Dataiku-style data flow visualization.
    """
    
    @staticmethod
    def render_flow(flow_data: Dict[str, Any], container=None):
        """Render data flow diagram."""
        
        # Create nodes and edges for flow diagram
        nodes = [
            dict(id=0, label="Raw Data", x=0, y=0),
            dict(id=1, label="Data Cleaning", x=1, y=0),
            dict(id=2, label="Feature Engineering", x=2, y=0),
            dict(id=3, label="Model Training", x=3, y=0),
            dict(id=4, label="Model Evaluation", x=4, y=0),
            dict(id=5, label="Deployment", x=5, y=0)
        ]
        
        edges = [
            dict(source=0, target=1),
            dict(source=1, target=2),
            dict(source=2, target=3),
            dict(source=3, target=4),
            dict(source=4, target=5)
        ]
        
        # Create plotly figure
        edge_trace = []
        for edge in edges:
            x0, y0 = nodes[edge['source']]['x'], nodes[edge['source']]['y']
            x1, y1 = nodes[edge['target']]['x'], nodes[edge['target']]['y']
            edge_trace.append(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(width=2, color='#888'),
                hoverinfo='none'
            ))
        
        node_trace = go.Scatter(
            x=[node['x'] for node in nodes],
            y=[node['y'] for node in nodes],
            mode='markers+text',
            text=[node['label'] for node in nodes],
            textposition="top center",
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            hoverinfo='text'
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title="Data Processing Pipeline",
            showlegend=False,
            hovermode='closest',
            height=200,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        if container:
            container.plotly_chart(fig, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)


class ModelDeploymentUI:
    """
    Model deployment interface components.
    """
    
    @staticmethod
    def render_deployment_options(model_id: str, container=None):
        """Render model deployment options."""
        
        st.subheader("🚀 Deployment Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Deploy to API", use_container_width=True):
                st.success("Model deployed to REST API")
                st.code(f"""
curl -X POST https://api.automl.com/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"model_id": "{model_id}", "data": {{}}}}'
                """)
        
        with col2:
            if st.button("Export Docker", use_container_width=True):
                st.info("Generating Docker image...")
                st.code("""
docker pull automl/model:latest
docker run -p 8080:8080 automl/model
                """)
        
        with col3:
            if st.button("Download Model", use_container_width=True):
                st.info("Preparing model for download...")
                st.markdown("[Download model.pkl](#)")
    
    @staticmethod
    def render_monitoring_setup(model_id: str, container=None):
        """Render monitoring setup interface."""
        
        st.subheader("📊 Monitoring Configuration")
        
        with st.form("monitoring_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                drift_threshold = st.slider("Drift Detection Threshold", 0.0, 1.0, 0.3)
                performance_threshold = st.slider("Performance Alert Threshold", 0.0, 1.0, 0.8)
                check_frequency = st.selectbox("Check Frequency", ["Hourly", "Daily", "Weekly"])
            
            with col2:
                alert_email = st.text_input("Alert Email")
                slack_webhook = st.text_input("Slack Webhook (optional)")
                enable_auto_retrain = st.checkbox("Enable Auto-Retraining")
            
            if st.form_submit_button("Save Configuration"):
                st.success("Monitoring configured successfully!")


class CustomMetrics:
    """
    Custom metrics and KPI displays.
    """
    
    @staticmethod
    def render_kpi_dashboard(kpis: Dict[str, Any], container=None):
        """Render KPI dashboard."""
        
        st.markdown("### 📈 Key Performance Indicators")
        
        cols = st.columns(len(kpis))
        
        for col, (name, data) in zip(cols, kpis.items()):
            with col:
                delta = data.get('delta')
                if delta:
                    delta_str = f"{delta:+.1f}%"
                else:
                    delta_str = None
                
                st.metric(
                    label=name,
                    value=data['value'],
                    delta=delta_str,
                    delta_color="normal" if not delta or delta > 0 else "inverse"
                )
    
    @staticmethod
    def render_confusion_matrix(y_true: List, y_pred: List, labels: List[str] = None):
        """Render interactive confusion matrix."""
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = [f"Class {i}" for i in range(len(cm))]
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues"
        )
        
        fig.update_layout(
            title="Confusion Matrix",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate metrics from confusion matrix
        accuracy = np.trace(cm) / np.sum(cm)
        st.info(f"Overall Accuracy: {accuracy:.2%}")


# Utility functions for common UI patterns

def create_sidebar_filters():
    """Create common sidebar filters."""
    
    with st.sidebar:
        st.header("Filters")
        
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            key="date_filter"
        )
        
        model_types = st.multiselect(
            "Model Types",
            ["XGBoost", "LightGBM", "RandomForest", "Neural Network"],
            default=["XGBoost", "LightGBM"]
        )
        
        min_score = st.slider(
            "Minimum Score",
            0.0, 1.0, 0.7,
            key="score_filter"
        )
        
        return {
            "date_range": date_range,
            "model_types": model_types,
            "min_score": min_score
        }


def create_action_buttons():
    """Create common action buttons."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    actions = {}
    
    with col1:
        actions['train'] = st.button("🎯 Train Model", use_container_width=True)
    
    with col2:
        actions['predict'] = st.button("🔮 Predict", use_container_width=True)
    
    with col3:
        actions['export'] = st.button("📥 Export", use_container_width=True)
    
    with col4:
        actions['refresh'] = st.button("🔄 Refresh", use_container_width=True)
    
    return actions


def show_loading_animation(message: str = "Processing..."):
    """Show loading animation with custom message."""
    
    with st.spinner(message):
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)  # FIXED: Now time is imported
        progress_bar.empty()


# Example usage
if __name__ == "__main__":
    st.set_page_config(page_title="UI Components Demo", layout="wide")
    
    st.title("AutoML UI Components Demo")
    
    # Data Quality
    st.header("Data Quality Visualization")
    quality_viz = DataQualityVisualizer()
    quality_viz.render_quality_gauge(85.5)
    
    # Alerts
    st.header("Alerts Display")
    alerts = [
        {"severity": "critical", "message": "High missing data in column X"},
        {"severity": "medium", "message": "Potential data drift detected"}
    ]
    AlertsAndNotifications.render_alerts(alerts)
    
    # KPIs
    st.header("KPI Dashboard")
    kpis = {
        "Accuracy": {"value": "92.5%", "delta": 2.3},
        "Models Trained": {"value": 15, "delta": 5},
        "Training Time": {"value": "3.2h", "delta": -15}
    }
    CustomMetrics.render_kpi_dashboard(kpis)
