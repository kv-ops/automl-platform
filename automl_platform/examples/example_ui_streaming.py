# User input
        user_question = st.text_input(
            "Ask the AI Assistant:",
            placeholder="e.g., How can I handle class imbalance?",
            key="ai_question"
        )
        
        if user_question:
            st.chat_message("user").write(user_question)
            
            # Simulate AI response
            with st.spinner("AI is thinking..."):
                time.sleep(1)
            
            response = self._generate_ai_response(user_question)
            st.chat_message("assistant").write(response)
        
        # AI capabilities
        st.subheader("🎯 AI Assistant Capabilities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Data & Feature Engineering:**
            - Quality assessment recommendations
            - Feature engineering suggestions
            - Data cleaning strategies
            - Handling missing values
            """)
        
        with col2:
            st.markdown("""
            **Model & Performance:**
            - Algorithm selection guidance
            - Hyperparameter recommendations
            - Performance improvement tips
            - Deployment considerations
            """)
        
        # Tips for using AI
        st.markdown("""
        <div class="highlight-box">
        <h4>💡 Tips for Using AI Assistant</h4>
        <ul>
        <li>Be specific in your questions for better answers</li>
        <li>Ask for code examples when needed</li>
        <li>Request explanations for complex concepts</li>
        <li>Use for debugging model issues</li>
        <li>Get recommendations based on your specific data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Mark as completed
        if 5 not in st.session_state.tutorial_completed:
            st.session_state.tutorial_completed.append(5)
    
    def step_6_reports(self):
        """Step 6: Reports and export"""
        st.header("📋 Step 6: Reports & Export")
        
        st.markdown("""
        <div class="tutorial-box">
        <h4>📝 Learning Objectives</h4>
        Generate comprehensive reports and export models for deployment.
        </div>
        """, unsafe_allow_html=True)
        
        # Report generation
        st.subheader("📄 Generate Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Executive Summary", "Technical Report", "Model Card", 
                 "Compliance Report", "Performance Report"]
            )
            
            format_type = st.selectbox(
                "Output Format",
                ["PDF", "HTML", "Markdown", "PowerPoint", "Word"]
            )
        
        with col2:
            st.markdown("**Report Options:**")
            include_viz = st.checkbox("Include Visualizations", value=True)
            include_code = st.checkbox("Include Code Snippets", value=False)
            include_data = st.checkbox("Include Data Summary", value=True)
            include_recommendations = st.checkbox("Include AI Recommendations", value=True)
        
        # Generate report button
        if st.button("📥 Generate Report", type="primary", use_container_width=True):
            with st.spinner(f"Generating {report_type}..."):
                time.sleep(2)
            
            st.success(f"✅ {report_type} generated successfully!")
            
            # Report preview
            with st.expander("📄 Report Preview"):
                st.markdown(f"""
                # {report_type}
                ## AutoML Model Training Report
                
                **Date:** {datetime.now().strftime('%Y-%m-%d')}
                **Project:** Loan Approval Prediction
                
                ### Executive Summary
                Successfully trained and evaluated 5 machine learning models for loan approval prediction.
                The ensemble model achieved the best performance with 93.1% accuracy.
                
                ### Key Findings
                - **Best Model:** Ensemble (XGBoost + LightGBM)
                - **Accuracy:** 93.1% (±1.2%)
                - **Key Features:** Credit score, income, loan amount
                - **Training Time:** 2 minutes 20 seconds
                
                ### Model Performance
                | Model | CV Score | Std Dev | Training Time |
                |-------|----------|---------|---------------|
                | Ensemble | 0.931 | 0.012 | 140s |
                | XGBoost | 0.924 | 0.015 | 45s |
                | LightGBM | 0.918 | 0.018 | 38s |
                
                ### Recommendations
                1. Deploy ensemble model for production use
                2. Monitor credit score feature for drift
                3. Retrain monthly with new data
                4. Implement A/B testing for model updates
                
                ### Next Steps
                - Set up model monitoring dashboard
                - Configure automated retraining pipeline
                - Implement prediction API endpoint
                - Create model documentation
                """)
            
            # Download button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label=f"⬇️ Download {format_type} Report",
                    data="Report content here...",
                    file_name=f"automl_report.{format_type.lower()}",
                    mime="text/plain"
                )
        
        # Model export
        st.subheader("📦 Model Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox(
                "Export Format",
                ["Pickle", "ONNX", "PMML", "TensorFlow", "Docker Container"]
            )
            
            include_preprocessor = st.checkbox("Include Preprocessor", value=True)
            include_metadata = st.checkbox("Include Metadata", value=True)
        
        with col2:
            st.markdown("**Deployment Target:**")
            deployment = st.radio(
                "",
                ["Local Server", "Cloud (AWS/GCP/Azure)", "Edge Device", "Mobile App"]
            )
            
            optimize_for = st.selectbox(
                "Optimize For",
                ["Balanced", "Speed", "Accuracy", "Size"]
            )
        
        if st.button("📦 Export Model", type="primary", use_container_width=True):
            with st.spinner(f"Exporting model as {export_format}..."):
                time.sleep(2)
            
            st.success("✅ Model exported successfully!")
            
            # Export details
            st.markdown("""
            <div class="success-message">
            <h4>Export Complete</h4>
            <ul>
            <li><b>Format:</b> ONNX</li>
            <li><b>Size:</b> 2.3 MB</li>
            <li><b>Includes:</b> Model, Preprocessor, Metadata</li>
            <li><b>Optimized for:</b> Speed</li>
            <li><b>Compatible with:</b> Python 3.8+, ONNX Runtime 1.10+</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Download exported model
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="⬇️ Download Model Package",
                    data=b"Model binary data...",
                    file_name="model_package.zip",
                    mime="application/zip"
                )
        
        # Deployment guide
        st.subheader("🚀 Deployment Guide")
        
        with st.expander("View Deployment Instructions"):
            st.markdown("""
            ### Quick Start Deployment
            
            **1. Install Dependencies:**
            ```bash
            pip install onnxruntime numpy pandas
            ```
            
            **2. Load Model:**
            ```python
            import onnxruntime as ort
            import numpy as np
            
            # Load model
            session = ort.InferenceSession("model.onnx")
            
            # Prepare input
            input_data = np.array([[...]], dtype=np.float32)
            
            # Make prediction
            output = session.run(None, {"input": input_data})
            prediction = output[0]
            ```
            
            **3. API Endpoint:**
            ```python
            from fastapi import FastAPI
            
            app = FastAPI()
            
            @app.post("/predict")
            async def predict(data: dict):
                # Process input
                # Run model
                # Return prediction
                return {"prediction": prediction}
            ```
            
            **4. Docker Deployment:**
            ```dockerfile
            FROM python:3.8-slim
            COPY . /app
            WORKDIR /app
            RUN pip install -r requirements.txt
            CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
            ```
            """)
        
        # Completion message
        if len(st.session_state.tutorial_completed) == 7:
            st.markdown("""
            <div class="success-message">
            <h3>🎉 Congratulations!</h3>
            <p>You've completed the AutoML Platform tutorial!</p>
            <p>You're now ready to:</p>
            <ul>
            <li>Upload and analyze your own datasets</li>
            <li>Train and optimize ML models</li>
            <li>Interpret and deploy models</li>
            <li>Generate professional reports</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Mark as completed
        if 6 not in st.session_state.tutorial_completed:
            st.session_state.tutorial_completed.append(6)
    
    def _generate_ai_response(self, question: str) -> str:
        """Generate AI response based on question"""
        
        responses = {
            "imbalance": """For handling class imbalance, try these approaches:
            
1. **Resampling Techniques:**
   - SMOTE (Synthetic Minority Over-sampling)
   - Random undersampling of majority class
   - Combination of over and undersampling

2. **Algorithm-level:**
   - Use class_weight='balanced' in scikit-learn
   - Adjust sample_weight in XGBoost/LightGBM
   
3. **Evaluation Metrics:**
   - Use F1-score, Precision-Recall AUC instead of accuracy
   - Focus on recall for minority class detection""",
            
            "default": """I can help you with:
            
- Data preprocessing and cleaning strategies
- Feature engineering recommendations
- Model selection and hyperparameter tuning
- Performance optimization techniques
- Deployment best practices

Please ask a specific question about your ML project!"""
        }
        
        # Simple keyword matching
        if "imbalance" in question.lower() or "balanced" in question.lower():
            return responses["imbalance"]
        
        return responses["default"]
    
    def reset_tutorial(self):
        """Reset tutorial progress"""
        st.session_state.tutorial_step = 0
        st.session_state.tutorial_data = None
        st.session_state.tutorial_model = None
        st.session_state.tutorial_completed = []
        st.success("Tutorial reset successfully!")


def main():
    """Main execution"""
    walkthrough = InteractiveWalkthrough()
    walkthrough.run()


if __name__ == "__main__":
    main()"""
Streamlit UI Walkthrough and Demo
=================================
Place in: automl_platform/examples/ui_walkthrough.py

Interactive guide demonstrating the Streamlit dashboard features,
including data upload, model training, chat AI, and visualizations.

To run: streamlit run ui_walkthrough.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="AutoML Platform - Interactive Walkthrough",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for tutorial styling
st.markdown("""
<style>
    .tutorial-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1976d2;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .step-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .highlight-box {
        background-color: #fff9c4;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-message {
        background-color: #c8e6c9;
        color: #2e7d32;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .code-example {
        background-color: #263238;
        color: #aed581;
        padding: 10px;
        border-radius: 5px;
        font-family: monospace;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class InteractiveWalkthrough:
    """Interactive walkthrough of the AutoML platform UI"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'tutorial_step' not in st.session_state:
            st.session_state.tutorial_step = 0
        if 'tutorial_data' not in st.session_state:
            st.session_state.tutorial_data = None
        if 'tutorial_model' not in st.session_state:
            st.session_state.tutorial_model = None
        if 'tutorial_completed' not in st.session_state:
            st.session_state.tutorial_completed = []
        if 'demo_mode' not in st.session_state:
            st.session_state.demo_mode = True
    
    def run(self):
        """Run the interactive walkthrough"""
        
        # Header
        st.title("🎓 AutoML Platform - Interactive Walkthrough")
        st.markdown("**Learn how to use the AutoML platform through hands-on examples**")
        
        # Progress tracker
        progress = len(st.session_state.tutorial_completed) / 7
        st.progress(progress)
        st.markdown(f"**Progress: {len(st.session_state.tutorial_completed)}/7 steps completed**")
        
        # Sidebar navigation
        with st.sidebar:
            st.header("📚 Tutorial Navigation")
            
            # Tutorial steps
            steps = [
                "🏠 Welcome & Overview",
                "📊 Data Upload & Quality",
                "🎯 Model Training",
                "📈 Leaderboard Analysis",
                "🔍 Model Interpretation",
                "💬 AI Assistant",
                "📋 Reports & Export"
            ]
            
            for i, step in enumerate(steps):
                if st.button(step, key=f"nav_{i}"):
                    st.session_state.tutorial_step = i
            
            st.divider()
            
            # Tutorial controls
            st.subheader("⚙️ Tutorial Settings")
            st.session_state.demo_mode = st.checkbox(
                "Demo Mode",
                value=st.session_state.demo_mode,
                help="Use simulated data and models"
            )
            
            if st.button("🔄 Reset Tutorial"):
                self.reset_tutorial()
                st.rerun()
            
            st.divider()
            
            # Quick help
            with st.expander("❓ Quick Help"):
                st.markdown("""
                **Navigation:**
                - Click steps in sidebar to jump
                - Use Next/Previous buttons
                - Progress saves automatically
                
                **Demo Mode:**
                - ON: Use simulated data
                - OFF: Connect to real API
                
                **Tips:**
                - 💡 Yellow boxes = Tips
                - 📝 Blue boxes = Instructions
                - ✅ Green boxes = Success
                """)
        
        # Main content area
        if st.session_state.tutorial_step == 0:
            self.step_0_welcome()
        elif st.session_state.tutorial_step == 1:
            self.step_1_data_upload()
        elif st.session_state.tutorial_step == 2:
            self.step_2_model_training()
        elif st.session_state.tutorial_step == 3:
            self.step_3_leaderboard()
        elif st.session_state.tutorial_step == 4:
            self.step_4_model_interpretation()
        elif st.session_state.tutorial_step == 5:
            self.step_5_ai_assistant()
        elif st.session_state.tutorial_step == 6:
            self.step_6_reports()
        
        # Navigation buttons
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.tutorial_step > 0:
                if st.button("⬅️ Previous", use_container_width=True):
                    st.session_state.tutorial_step -= 1
                    st.rerun()
        
        with col3:
            if st.session_state.tutorial_step < 6:
                if st.button("Next ➡️", use_container_width=True):
                    st.session_state.tutorial_step += 1
                    st.rerun()
    
    def step_0_welcome(self):
        """Step 0: Welcome and overview"""
        st.header("🏠 Welcome to AutoML Platform")
        
        st.markdown("""
        <div class="tutorial-box">
        <h4>📝 About This Tutorial</h4>
        This interactive walkthrough will guide you through all features of the AutoML platform.
        You'll learn how to upload data, train models, interpret results, and use AI assistance.
        </div>
        """, unsafe_allow_html=True)
        
        # Platform overview
        st.subheader("🎯 What You'll Learn")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Core Features:**
            - 📊 Data upload and quality assessment
            - 🎯 Automated model training
            - 📈 Model comparison and selection
            - 🔍 Feature importance analysis
            """)
        
        with col2:
            st.markdown("""
            **Advanced Features:**
            - 💬 AI-powered assistance
            - 📋 Automated report generation
            - 🔧 Feature engineering
            - 📦 Model export and deployment
            """)
        
        # Quick demo
        st.subheader("🚀 Quick Demo")
        
        if st.button("▶️ Watch Platform Demo", type="primary"):
            with st.spinner("Loading demo..."):
                # Simulate demo
                tabs = st.tabs(["Data", "Training", "Results"])
                
                with tabs[0]:
                    # Create sample data
                    df_demo = pd.DataFrame({
                        'feature_1': np.random.randn(100),
                        'feature_2': np.random.randn(100),
                        'feature_3': np.random.randn(100),
                        'target': np.random.choice([0, 1], 100)
                    })
                    
                    st.dataframe(df_demo.head(10))
                    
                    # Quality metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows", 100)
                    col2.metric("Features", 3)
                    col3.metric("Quality Score", "95/100")
                
                with tabs[1]:
                    # Training progress
                    progress_bar = st.progress(0)
                    for i in range(101):
                        progress_bar.progress(i / 100)
                        time.sleep(0.01)
                    
                    st.success("✅ Training completed!")
                
                with tabs[2]:
                    # Results
                    results_df = pd.DataFrame({
                        'Model': ['XGBoost', 'RandomForest', 'LightGBM'],
                        'Accuracy': [0.92, 0.89, 0.91],
                        'Training Time': [45, 52, 38]
                    })
                    
                    fig = px.bar(results_df, x='Model', y='Accuracy', 
                                title='Model Performance')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Platform architecture
        st.subheader("🏗️ Platform Architecture")
        
        st.markdown("""
        <div class="highlight-box">
        <h4>💡 Key Components</h4>
        <ul>
        <li><b>AutoML Engine:</b> Automated model training and optimization</li>
        <li><b>MLflow Integration:</b> Model versioning and registry</li>
        <li><b>Streaming Pipeline:</b> Real-time data processing</li>
        <li><b>AI Assistant:</b> LLM-powered guidance</li>
        <li><b>GDPR Compliance:</b> Data privacy and consent management</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Mark as completed
        if 0 not in st.session_state.tutorial_completed:
            st.session_state.tutorial_completed.append(0)
    
    def step_1_data_upload(self):
        """Step 1: Data upload and quality assessment"""
        st.header("📊 Step 1: Data Upload & Quality Assessment")
        
        st.markdown("""
        <div class="tutorial-box">
        <h4>📝 Learning Objectives</h4>
        Learn how to upload datasets, assess data quality, and prepare data for training.
        </div>
        """, unsafe_allow_html=True)
        
        # Data upload section
        st.subheader("📤 Upload Your Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a dataset for analysis"
            )
            
            if uploaded_file or st.session_state.demo_mode:
                # Use demo data if in demo mode
                if st.session_state.demo_mode:
                    # Create sample dataset
                    np.random.seed(42)
                    n_samples = 500
                    
                    df = pd.DataFrame({
                        'age': np.random.randint(18, 80, n_samples),
                        'income': np.random.exponential(50000, n_samples),
                        'credit_score': np.random.normal(700, 100, n_samples),
                        'loan_amount': np.random.uniform(1000, 50000, n_samples),
                        'employment_years': np.random.poisson(5, n_samples),
                        'approved': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
                    })
                    
                    # Add some missing values
                    df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
                    df.loc[np.random.choice(df.index, 30), 'credit_score'] = np.nan
                    
                    st.info("📌 Using demo dataset: Loan Approval Data")
                else:
                    df = pd.read_csv(uploaded_file)
                
                st.session_state.tutorial_data = df
                
                # Data preview
                st.subheader("👀 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data quality assessment
                st.subheader("🔍 Data Quality Assessment")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                
                with col2:
                    st.metric("Columns", len(df.columns))
                
                with col3:
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    st.metric("Missing %", f"{missing_pct:.1f}%")
                
                with col4:
                    quality_score = max(0, 100 - missing_pct * 2)
                    st.metric("Quality Score", f"{quality_score:.0f}/100")
                
                # Data types
                st.subheader("📋 Data Types")
                
                type_df = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Unique Values': [df[col].nunique() for col in df.columns],
                    'Missing': [df[col].isnull().sum() for col in df.columns]
                })
                
                st.dataframe(type_df, use_container_width=True)
                
                # Data cleaning options
                st.subheader("🧹 Data Cleaning")
                
                with st.expander("Cleaning Options"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        handle_missing = st.selectbox(
                            "Handle Missing Values",
                            ["Drop rows", "Fill with mean", "Fill with median", "Forward fill"]
                        )
                        
                        remove_duplicates = st.checkbox("Remove duplicate rows")
                    
                    with col2:
                        normalize = st.checkbox("Normalize numeric features")
                        
                        encode_categorical = st.checkbox("Encode categorical variables")
                    
                    if st.button("Apply Cleaning"):
                        with st.spinner("Cleaning data..."):
                            time.sleep(1)
                            st.success("✅ Data cleaning completed!")
                            
                            # Show cleaning report
                            st.markdown("""
                            **Cleaning Report:**
                            - Filled 80 missing values
                            - Removed 5 duplicate rows
                            - Normalized 4 numeric features
                            - Encoded 0 categorical variables
                            """)
        
        with col2:
            # Tips and best practices
            st.markdown("""
            <div class="highlight-box">
            <h4>💡 Tips</h4>
            <ul>
            <li>Ensure target column is included</li>
            <li>Check for class imbalance</li>
            <li>Remove ID columns</li>
            <li>Handle missing values appropriately</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Data statistics
            if st.session_state.tutorial_data is not None:
                st.subheader("📊 Quick Stats")
                
                df = st.session_state.tutorial_data
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("Select column", numeric_cols)
                    
                    # Distribution plot
                    fig = px.histogram(df, x=selected_col, nbins=30,
                                     title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.write(f"**{selected_col} Statistics:**")
                    st.write(f"- Mean: {df[selected_col].mean():.2f}")
                    st.write(f"- Std: {df[selected_col].std():.2f}")
                    st.write(f"- Min: {df[selected_col].min():.2f}")
                    st.write(f"- Max: {df[selected_col].max():.2f}")
        
        # Mark as completed
        if 1 not in st.session_state.tutorial_completed:
            st.session_state.tutorial_completed.append(1)
    
    def step_2_model_training(self):
        """Step 2: Model training configuration"""
        st.header("🎯 Step 2: Model Training")
        
        st.markdown("""
        <div class="tutorial-box">
        <h4>📝 Learning Objectives</h4>
        Configure and launch automated model training with multiple algorithms.
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.tutorial_data is None and not st.session_state.demo_mode:
            st.warning("⚠️ Please complete Step 1 (Data Upload) first")
            return
        
        # Training configuration
        st.subheader("⚙️ Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Target selection
            if st.session_state.demo_mode:
                columns = ['age', 'income', 'credit_score', 'loan_amount', 
                          'employment_years', 'approved']
                target_column = st.selectbox("Target Column", columns, index=5)
            else:
                df = st.session_state.tutorial_data
                target_column = st.selectbox("Target Column", df.columns)
            
            task_type = st.selectbox("Task Type", 
                                    ["auto", "classification", "regression"])
        
        with col2:
            # Algorithm selection
            algorithms = st.multiselect(
                "Algorithms",
                ["RandomForest", "XGBoost", "LightGBM", "CatBoost", 
                 "LogisticRegression", "NeuralNetwork"],
                default=["RandomForest", "XGBoost", "LightGBM"]
            )
            
            cv_folds = st.slider("Cross-validation Folds", 3, 10, 5)
        
        with col3:
            # Time and resources
            max_time = st.slider("Max Training Time (min)", 1, 30, 5)
            
            enable_hpo = st.checkbox("Hyperparameter Optimization", value=True)
            
            if enable_hpo:
                hpo_trials = st.slider("HPO Trials", 10, 50, 20)
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Feature Engineering")
                auto_feature = st.checkbox("Automatic Feature Engineering", value=True)
                polynomial_features = st.checkbox("Polynomial Features")
                interaction_features = st.checkbox("Interaction Features")
                
            with col2:
                st.subheader("Model Settings")
                ensemble = st.checkbox("Enable Ensemble", value=True)
                handle_imbalance = st.checkbox("Handle Class Imbalance")
                early_stopping = st.checkbox("Early Stopping", value=True)
        
        # Training execution
        st.subheader("🚀 Start Training")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🎯 Train Models", type="primary", use_container_width=True):
                # Training simulation
                st.markdown("---")
                st.subheader("Training Progress")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate training phases
                phases = [
                    ("Preparing data...", 0.1),
                    ("Training RandomForest...", 0.3),
                    ("Training XGBoost...", 0.5),
                    ("Training LightGBM...", 0.7),
                    ("Hyperparameter optimization...", 0.85),
                    ("Creating ensemble...", 0.95),
                    ("Finalizing results...", 1.0)
                ]
                
                for phase, progress in phases:
                    status_text.text(phase)
                    progress_bar.progress(progress)
                    time.sleep(0.5)
                
                # Success message
                st.success("✅ Training completed successfully!")
                
                # Results preview
                st.subheader("📊 Training Results")
                
                results_df = pd.DataFrame({
                    'Model': ['XGBoost', 'LightGBM', 'RandomForest', 'Ensemble'],
                    'CV Score': [0.924, 0.918, 0.895, 0.931],
                    'Training Time (s)': [45, 38, 52, 5],
                    'Parameters': [150, 120, 100, 370]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Best model highlight
                st.markdown("""
                <div class="success-message">
                <h4>🏆 Best Model: Ensemble</h4>
                <ul>
                <li>CV Score: 0.931</li>
                <li>Improvement over baseline: +8.5%</li>
                <li>Training completed in 2m 20s</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Save to session state
                st.session_state.tutorial_model = results_df
        
        # Training tips
        st.markdown("""
        <div class="highlight-box">
        <h4>💡 Training Tips</h4>
        <ul>
        <li><b>Algorithms:</b> Start with 3-4 diverse algorithms</li>
        <li><b>Cross-validation:</b> Use 5-fold for balanced bias-variance</li>
        <li><b>HPO:</b> More trials = better performance but longer training</li>
        <li><b>Ensemble:</b> Usually provides best performance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Mark as completed
        if 2 not in st.session_state.tutorial_completed:
            st.session_state.tutorial_completed.append(2)
    
    def step_3_leaderboard(self):
        """Step 3: Model leaderboard and comparison"""
        st.header("📈 Step 3: Model Leaderboard & Comparison")
        
        st.markdown("""
        <div class="tutorial-box">
        <h4>📝 Learning Objectives</h4>
        Analyze and compare trained models to select the best performer.
        </div>
        """, unsafe_allow_html=True)
        
        # Create or load results
        if st.session_state.tutorial_model is None:
            # Create demo results
            results_df = pd.DataFrame({
                'Rank': [1, 2, 3, 4, 5],
                'Model': ['Ensemble', 'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest'],
                'CV Score': [0.931, 0.924, 0.918, 0.912, 0.895],
                'Std Dev': [0.012, 0.015, 0.018, 0.020, 0.025],
                'Training Time (s)': [140, 45, 38, 48, 52],
                'Prediction Time (ms)': [12, 5, 4, 6, 8]
            })
        else:
            results_df = st.session_state.tutorial_model
        
        # Leaderboard display
        st.subheader("🏆 Model Leaderboard")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Models Trained", len(results_df))
        with col2:
            st.metric("Best Score", f"{results_df.iloc[0]['CV Score']:.3f}")
        with col3:
            st.metric("Total Time", f"{results_df['Training Time (s)'].sum():.0f}s")
        with col4:
            st.metric("Best Model", results_df.iloc[0]['Model'])
        
        # Leaderboard table
        st.dataframe(
            results_df.style.highlight_max(subset=['CV Score'], color='lightgreen'),
            use_container_width=True
        )
        
        # Visualizations
        st.subheader("📊 Performance Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["Performance", "Time Analysis", "Trade-offs"])
        
        with tab1:
            # Performance comparison
            fig = px.bar(results_df, x='Model', y='CV Score',
                        error_y='Std Dev',
                        title='Model Performance Comparison',
                        color='CV Score',
                        color_continuous_scale='viridis')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Time analysis
            fig = px.scatter(results_df, x='Training Time (s)', 
                           y='Prediction Time (ms)',
                           size='CV Score', text='Model',
                           title='Training vs Prediction Time',
                           color='CV Score',
                           color_continuous_scale='viridis')
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Performance vs complexity trade-off
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(go.Scatter(
                x=results_df['Training Time (s)'],
                y=results_df['CV Score'],
                mode='markers+text',
                text=results_df['Model'],
                textposition='top center',
                marker=dict(
                    size=results_df['CV Score'] * 50,
                    color=results_df['CV Score'],
                    colorscale='viridis',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title='Performance vs Training Time Trade-off',
                xaxis_title='Training Time (seconds)',
                yaxis_title='CV Score',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model selection
        st.subheader("🎯 Model Selection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_model = st.selectbox(
                "Select model for deployment",
                results_df['Model'].tolist(),
                help="Choose the model that best balances performance and requirements"
            )
            
            # Selection criteria
            criteria = st.multiselect(
                "Selection Criteria",
                ["Highest Accuracy", "Fastest Prediction", "Most Stable", "Best Trade-off"],
                default=["Highest Accuracy"]
            )
            
            if st.button("Confirm Selection", type="primary"):
                st.success(f"✅ {selected_model} selected for deployment!")
                
                # Show selection summary
                selected_row = results_df[results_df['Model'] == selected_model].iloc[0]
                
                st.markdown(f"""
                <div class="success-message">
                <h4>Selection Summary</h4>
                <ul>
                <li><b>Model:</b> {selected_model}</li>
                <li><b>Performance:</b> {selected_row['CV Score']:.3f} ± {selected_row['Std Dev']:.3f}</li>
                <li><b>Training Time:</b> {selected_row['Training Time (s)']}s</li>
                <li><b>Prediction Speed:</b> {selected_row['Prediction Time (ms)']}ms</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Recommendations
            st.markdown("""
            <div class="highlight-box">
            <h4>💡 Selection Guide</h4>
            <ul>
            <li><b>Production:</b> Balance accuracy and speed</li>
            <li><b>Research:</b> Prioritize accuracy</li>
            <li><b>Real-time:</b> Prioritize speed</li>
            <li><b>Critical:</b> Consider ensemble</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Mark as completed
        if 3 not in st.session_state.tutorial_completed:
            st.session_state.tutorial_completed.append(3)
    
    def step_4_model_interpretation(self):
        """Step 4: Model interpretation and analysis"""
        st.header("🔍 Step 4: Model Interpretation")
        
        st.markdown("""
        <div class="tutorial-box">
        <h4>📝 Learning Objectives</h4>
        Understand model decisions through feature importance and SHAP analysis.
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance
        st.subheader("📊 Feature Importance")
        
        # Create sample feature importance
        features = ['credit_score', 'income', 'loan_amount', 'employment_years', 'age']
        importance = [0.35, 0.25, 0.20, 0.12, 0.08]
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        })
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h',
                        title='Feature Importance',
                        color='Importance',
                        color_continuous_scale='blues')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart
            fig = px.pie(importance_df, values='Importance', names='Feature',
                        title='Feature Contribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        st.subheader("🔬 Feature Analysis")
        
        selected_feature = st.selectbox("Select feature for detailed analysis", features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature statistics
            st.markdown(f"**{selected_feature} Statistics:**")
            
            stats = {
                "Importance": f"{importance[features.index(selected_feature)]:.2%}",
                "Correlation with target": "0.65",
                "Missing values": "2.3%",
                "Unique values": "487",
                "Distribution": "Normal"
            }
            
            for key, value in stats.items():
                st.write(f"- {key}: {value}")
        
        with col2:
            # Partial dependence plot
            x = np.linspace(0, 100, 100)
            y = np.sin(x / 10) * 0.3 + 0.5
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
            fig.update_layout(
                title=f'Partial Dependence: {selected_feature}',
                xaxis_title=selected_feature,
                yaxis_title='Predicted Probability',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # SHAP explanation
        st.subheader("🎯 SHAP Analysis")
        
        st.info("SHAP (SHapley Additive exPlanations) shows how each feature contributes to individual predictions")
        
        # Sample prediction explanation
        with st.expander("Sample Prediction Explanation"):
            # Create sample SHAP values
            sample_data = {
                'Feature': features,
                'Value': [720, 65000, 25000, 8, 35],
                'SHAP Value': [0.15, 0.08, -0.05, 0.03, 0.01],
                'Impact': ['Positive', 'Positive', 'Negative', 'Positive', 'Positive']
            }
            
            shap_df = pd.DataFrame(sample_data)
            
            st.markdown("**Prediction: Approved (82% confidence)**")
            
            # SHAP waterfall
            fig = go.Figure(go.Waterfall(
                name="SHAP",
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "relative", "total"],
                x=shap_df['Feature'].tolist() + ['Prediction'],
                y=shap_df['SHAP Value'].tolist() + [sum(shap_df['SHAP Value'])],
                text=[f"+{v:.2f}" if v > 0 else f"{v:.2f}" 
                     for v in shap_df['SHAP Value'].tolist() + [sum(shap_df['SHAP Value'])]],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}}
            ))
            
            fig.update_layout(
                title="SHAP Waterfall Plot",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature contributions table
            st.markdown("**Feature Contributions:**")
            st.dataframe(
                shap_df.style.applymap(
                    lambda x: 'color: green' if x == 'Positive' else 'color: red',
                    subset=['Impact']
                ),
                use_container_width=True
            )
        
        # Model insights
        st.subheader("💡 Key Insights")
        
        st.markdown("""
        <div class="highlight-box">
        <h4>Model Insights Summary</h4>
        <ul>
        <li><b>Top Driver:</b> Credit score is the most influential feature (35% importance)</li>
        <li><b>Risk Factors:</b> High loan amounts relative to income decrease approval probability</li>
        <li><b>Positive Indicators:</b> Longer employment history increases approval chances</li>
        <li><b>Non-linear Relationships:</b> Age shows diminishing returns after 40 years</li>
        <li><b>Interaction Effects:</b> Income and loan amount interact strongly</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Mark as completed
        if 4 not in st.session_state.tutorial_completed:
            st.session_state.tutorial_completed.append(4)
    
    def step_5_ai_assistant(self):
        """Step 5: AI Assistant interaction"""
        st.header("💬 Step 5: AI Assistant")
        
        st.markdown("""
        <div class="tutorial-box">
        <h4>📝 Learning Objectives</h4>
        Learn to use the AI assistant for insights, recommendations, and guidance.
        </div>
        """, unsafe_allow_html=True)
        
        # AI Assistant interface
        st.subheader("🤖 AI-Powered Assistance")
        
        # Chat interface
        chat_container = st.container()
        
        # Sample conversation
        sample_messages = [
            ("user", "How can I improve my model's performance?"),
            ("assistant", """Based on your current model, here are my recommendations:

1. **Feature Engineering**: Create interaction features between income and loan_amount
2. **Handle Imbalance**: Your dataset has class imbalance (70/30). Try SMOTE or class weights
3. **Hyperparameter Tuning**: Increase XGBoost max_depth to 8 and n_estimators to 200
4. **Ensemble Method**: Combine XGBoost with LightGBM using weighted averaging

These changes could improve your accuracy by 3-5%."""),
            ("user", "What features should I engineer?"),
            ("assistant", """Here are specific feature engineering suggestions:

**Ratio Features:**
- `debt_to_income_ratio = loan_amount / income`
- `credit_utilization = loan_amount / credit_limit`

**Interaction Features:**
- `income_x_credit_score = income * credit_score / 1000`
- `age_employment_interaction = age * employment_years`

**Binned Features:**
- `age_group`: Young (18-30), Middle (31-50), Senior (50+)
- `risk_category`: Based on credit score ranges

**Domain Features:**
- `financial_stability = credit_score * employment_years / 100`
- `loan_affordability = income / (loan_amount / 12)`

These features capture non-linear relationships in your data.""")
        ]
        
        with chat_container:
            for role, message in sample_messages:
                if role == "user":
                    st.chat_message("user").write(message)
                else:
                    st.chat_message("assistant").write(message)
        
        # Interactive chat
        st.subheader("💭 Try It Yourself")
        
        # Predefined prompts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Explain Results"):
                st.chat_message("user").write("Explain why XGBoost performed best")
                st.chat_
