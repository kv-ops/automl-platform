"""
Tests for Streamlit UI Dashboard
=================================
Tests for the AutoML platform's Streamlit interface including
data upload, model training, leaderboard, and AI assistant features.
"""

import pytest
import pandas as pd
import numpy as np
import json
import io
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
import streamlit as st
from pathlib import Path

# Import Streamlit app components
try:
    from automl_platform.ui.streamlit_app import AutoMLDashboard
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False
    AutoMLDashboard = None


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components."""
    if not UI_AVAILABLE:
        pytest.skip("UI module not available")
    
    with patch.multiple('streamlit',
                       title=MagicMock(),
                       header=MagicMock(),
                       subheader=MagicMock(),
                       write=MagicMock(),
                       markdown=MagicMock(),
                       info=MagicMock(),
                       success=MagicMock(),
                       warning=MagicMock(),
                       error=MagicMock(),
                       button=MagicMock(return_value=False),
                       checkbox=MagicMock(return_value=False),
                       selectbox=MagicMock(return_value='option1'),
                       multiselect=MagicMock(return_value=['option1']),
                       slider=MagicMock(return_value=5),
                       text_input=MagicMock(return_value='test'),
                       file_uploader=MagicMock(return_value=None),
                       columns=MagicMock(return_value=[MagicMock(), MagicMock()]),
                       tabs=MagicMock(return_value=[MagicMock() for _ in range(6)]),
                       expander=MagicMock(),
                       container=MagicMock(),
                       sidebar=MagicMock(),
                       metric=MagicMock(),
                       progress=MagicMock(),
                       dataframe=MagicMock(),
                       plotly_chart=MagicMock(),
                       download_button=MagicMock(),
                       cache_data=MagicMock(),
                       session_state=MagicMock(),
                       secrets=MagicMock(),
                       balloons=MagicMock(),
                       spinner=MagicMock(),
                       empty=MagicMock(),
                       divider=MagicMock(),
                       rerun=MagicMock()) as mock_st:
        yield mock_st


@pytest.fixture
def dashboard(mock_streamlit):
    """Create AutoMLDashboard instance with mocked Streamlit."""
    if not UI_AVAILABLE:
        pytest.skip("UI module not available")
    
    # Mock session state
    st.session_state = {
        'current_experiment': None,
        'uploaded_data': None,
        'chat_history': [],
        'training_status': 'idle',
        'models_trained': [],
        'selected_model': None,
        'feature_suggestions': []
    }
    
    # Mock secrets
    st.secrets = {
        'API_BASE_URL': 'http://localhost:8000',
        'API_KEY': 'test_key'
    }
    
    return AutoMLDashboard()


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'feature4': np.random.randint(0, 100, 100),
        'target': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def sample_file(sample_dataframe):
    """Create sample file upload object."""
    buffer = io.BytesIO()
    sample_dataframe.to_csv(buffer, index=False)
    buffer.seek(0)
    
    file = MagicMock()
    file.name = "test_data.csv"
    file.read = buffer.read
    file.seek = buffer.seek
    
    return file


@pytest.fixture
def trained_models():
    """Create sample trained models data."""
    return [
        {"model": "XGBoost", "score": 0.92, "time": 45},
        {"model": "LightGBM", "score": 0.91, "time": 38},
        {"model": "RandomForest", "score": 0.89, "time": 52},
        {"model": "CatBoost", "score": 0.90, "time": 48},
        {"model": "LogisticRegression", "score": 0.85, "time": 12}
    ]


# ============================================================================
# Dashboard Initialization Tests
# ============================================================================

class TestDashboardInit:
    """Tests for dashboard initialization."""
    
    def test_dashboard_creation(self, dashboard):
        """Test dashboard object creation."""
        assert dashboard is not None
        assert hasattr(dashboard, 'api_headers')
        assert 'Authorization' in dashboard.api_headers or dashboard.api_headers == {}
    
    def test_session_state_initialization(self, dashboard):
        """Test session state initialization."""
        dashboard.init_session_state()
        
        assert 'current_experiment' in st.session_state
        assert 'uploaded_data' in st.session_state
        assert 'chat_history' in st.session_state
        assert 'training_status' in st.session_state
        assert 'models_trained' in st.session_state
        assert 'selected_model' in st.session_state
        assert 'feature_suggestions' in st.session_state
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_main_app_structure(self, mock_st, dashboard):
        """Test main app structure and tabs."""
        mock_st.tabs.return_value = [MagicMock() for _ in range(6)]
        
        dashboard.run()
        
        # Check main components were called
        mock_st.title.assert_called_with("🤖 AutoML Platform")
        mock_st.tabs.assert_called_once()
        
        # Check tabs were created
        tabs = mock_st.tabs.call_args[0][0]
        assert len(tabs) == 6
        assert "Data Upload" in tabs[0]
        assert "Model Training" in tabs[1]
        assert "Leaderboard" in tabs[2]


# ============================================================================
# Data Tab Tests
# ============================================================================

class TestDataTab:
    """Tests for data upload and quality assessment tab."""
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_file_upload(self, mock_st, dashboard, sample_file):
        """Test file upload functionality."""
        mock_st.file_uploader.return_value = sample_file
        
        dashboard.render_data_tab()
        
        mock_st.file_uploader.assert_called_once()
        assert st.session_state.uploaded_data is not None
    
    def test_load_data_csv(self, dashboard, sample_dataframe):
        """Test loading CSV data."""
        buffer = io.BytesIO()
        sample_dataframe.to_csv(buffer, index=False)
        buffer.seek(0)
        
        file = MagicMock()
        file.name = "test.csv"
        file.read = buffer.read
        file.seek = buffer.seek
        
        df = dashboard.load_data(file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)
    
    def test_load_data_excel(self, dashboard, sample_dataframe):
        """Test loading Excel data."""
        buffer = io.BytesIO()
        sample_dataframe.to_excel(buffer, index=False)
        buffer.seek(0)
        
        file = MagicMock()
        file.name = "test.xlsx"
        file.read = buffer.read
        file.seek = buffer.seek
        
        df = dashboard.load_data(file)
        
        assert isinstance(df, pd.DataFrame)
    
    def test_assess_data_quality(self, dashboard, sample_dataframe):
        """Test data quality assessment."""
        score, issues = dashboard.assess_data_quality(sample_dataframe)
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100
        assert isinstance(issues, list)
    
    def test_assess_data_quality_with_issues(self, dashboard):
        """Test data quality assessment with problematic data."""
        # Create problematic DataFrame
        df = pd.DataFrame({
            'col1': [1, 1, 1, 1, 1],  # Constant column
            'col2': [1, 2, None, 4, 5],  # Missing values
            'col3': ['a', 'b', 'c', 'd', 'e'],  # High cardinality
            'col4': [1, 2, 3, 3, 4]  # Duplicates
        })
        
        # Add duplicate rows
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        
        score, issues = dashboard.assess_data_quality(df)
        
        assert score < 100
        assert len(issues) > 0
        assert any("Constant column" in issue for issue in issues)
        assert any("duplicate" in issue.lower() for issue in issues)
    
    def test_ai_clean_data(self, dashboard, sample_dataframe):
        """Test AI-powered data cleaning."""
        # Add some issues to the data
        dirty_df = sample_dataframe.copy()
        dirty_df.iloc[0:5, 0] = np.nan  # Add missing values
        dirty_df = pd.concat([dirty_df, dirty_df.iloc[[0]]], ignore_index=True)  # Add duplicate
        
        cleaned_df, report = dashboard.ai_clean_data(dirty_df)
        
        assert isinstance(cleaned_df, pd.DataFrame)
        assert cleaned_df.isnull().sum().sum() < dirty_df.isnull().sum().sum()
        assert len(cleaned_df) <= len(dirty_df)  # Duplicates removed
        assert isinstance(report, str)
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_correlation_heatmap_display(self, mock_st, dashboard, sample_dataframe):
        """Test correlation heatmap display."""
        st.session_state.uploaded_data = sample_dataframe
        
        dashboard.render_data_tab()
        
        # Check if plotly chart was called for correlation
        assert mock_st.plotly_chart.called


# ============================================================================
# Training Tab Tests
# ============================================================================

class TestTrainingTab:
    """Tests for model training tab."""
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_training_without_data(self, mock_st, dashboard):
        """Test training tab without uploaded data."""
        st.session_state.uploaded_data = None
        
        dashboard.render_training_tab()
        
        mock_st.warning.assert_called_with("⚠️ Please upload data first in the Data Upload tab")
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_training_configuration(self, mock_st, dashboard, sample_dataframe):
        """Test training configuration options."""
        st.session_state.uploaded_data = sample_dataframe
        
        dashboard.render_training_tab()
        
        # Check configuration elements
        mock_st.selectbox.assert_called()
        mock_st.text_input.assert_called()
        mock_st.multiselect.assert_called()
        mock_st.slider.assert_called()
        mock_st.checkbox.assert_called()
    
    def test_get_feature_suggestions(self, dashboard, sample_dataframe):
        """Test feature engineering suggestions."""
        suggestions = dashboard.get_feature_suggestions(sample_dataframe, 'target')
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        for suggestion in suggestions:
            assert 'name' in suggestion
            assert 'description' in suggestion
            assert 'importance' in suggestion
            assert 'code' in suggestion
    
    @patch('automl_platform.ui.streamlit_app.st')
    @patch('time.sleep')
    def test_training_process(self, mock_sleep, mock_st, dashboard, sample_dataframe):
        """Test training process simulation."""
        st.session_state.uploaded_data = sample_dataframe
        mock_st.button.return_value = True  # Simulate button click
        
        # Mock progress placeholders
        progress_mock = MagicMock()
        status_mock = MagicMock()
        mock_st.empty.side_effect = [progress_mock, status_mock]
        
        dashboard.render_training_tab()
        
        # Check training status was updated
        assert st.session_state.training_status == 'completed'
        assert len(st.session_state.models_trained) > 0
        mock_st.success.assert_called()
        mock_st.balloons.assert_called()


# ============================================================================
# Leaderboard Tab Tests
# ============================================================================

class TestLeaderboardTab:
    """Tests for model leaderboard tab."""
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_leaderboard_without_models(self, mock_st, dashboard):
        """Test leaderboard with no trained models."""
        st.session_state.models_trained = []
        
        dashboard.render_leaderboard_tab()
        
        mock_st.info.assert_called_with("No models trained yet. Start training in the Model Training tab.")
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_leaderboard_display(self, mock_st, dashboard, trained_models):
        """Test leaderboard display with trained models."""
        st.session_state.models_trained = trained_models
        
        dashboard.render_leaderboard_tab()
        
        # Check DataFrame display
        mock_st.dataframe.assert_called()
        
        # Check visualizations
        assert mock_st.plotly_chart.call_count >= 2  # At least 2 charts
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_model_selection(self, mock_st, dashboard, trained_models):
        """Test model selection for deployment."""
        st.session_state.models_trained = trained_models
        mock_st.selectbox.return_value = "XGBoost"
        mock_st.button.return_value = True
        
        dashboard.render_leaderboard_tab()
        
        assert st.session_state.selected_model == "XGBoost"
        mock_st.success.assert_called()


# ============================================================================
# Analysis Tab Tests
# ============================================================================

class TestAnalysisTab:
    """Tests for model analysis tab."""
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_analysis_without_model(self, mock_st, dashboard):
        """Test analysis tab without selected model."""
        st.session_state.selected_model = None
        
        dashboard.render_analysis_tab()
        
        mock_st.info.assert_called_with("Please select a model from the Leaderboard tab first.")
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_analysis_with_model(self, mock_st, dashboard):
        """Test analysis tab with selected model."""
        st.session_state.selected_model = "XGBoost"
        
        dashboard.render_analysis_tab()
        
        # Check analysis components
        mock_st.subheader.assert_called()
        mock_st.plotly_chart.assert_called()  # Feature importance chart
        mock_st.expander.assert_called()  # Model explanation
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_error_analysis_display(self, mock_st, dashboard):
        """Test error analysis visualization."""
        st.session_state.selected_model = "XGBoost"
        
        dashboard.render_analysis_tab()
        
        # Check confusion matrix and residuals plots
        assert mock_st.plotly_chart.call_count >= 2


# ============================================================================
# Chat Tab Tests
# ============================================================================

class TestChatTab:
    """Tests for AI assistant chat tab."""
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_chat_interface(self, mock_st, dashboard):
        """Test chat interface rendering."""
        dashboard.render_chat_tab()
        
        mock_st.header.assert_called_with("💬 AI Assistant")
        mock_st.text_input.assert_called()
        mock_st.button.assert_called()
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_chat_history_display(self, mock_st, dashboard):
        """Test chat history display."""
        st.session_state.chat_history = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi! How can I help?'}
        ]
        
        dashboard.render_chat_tab()
        
        # Check markdown was called for messages
        assert mock_st.markdown.call_count >= 2
    
    def test_get_ai_response(self, dashboard):
        """Test AI response generation."""
        queries = [
            "explain the model",
            "how to improve performance",
            "suggest features",
            "general question"
        ]
        
        for query in queries:
            response = dashboard.get_ai_response(query)
            assert isinstance(response, str)
            assert len(response) > 0
    
    @patch('automl_platform.ui.streamlit_app.st')
    @patch('time.sleep')
    def test_send_message(self, mock_sleep, mock_st, dashboard):
        """Test sending message in chat."""
        mock_st.text_input.return_value = "How to improve model?"
        mock_st.button.return_value = True
        
        initial_history_length = len(st.session_state.chat_history)
        
        dashboard.render_chat_tab()
        
        # Check messages were added to history
        assert len(st.session_state.chat_history) > initial_history_length
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_quick_prompts(self, mock_st, dashboard):
        """Test quick prompt buttons."""
        # Mock button clicks
        mock_st.button.side_effect = [True, False, False]  # First button clicked
        
        dashboard.render_chat_tab()
        
        # Verify rerun was called after quick prompt
        mock_st.rerun.assert_called()


# ============================================================================
# Reports Tab Tests
# ============================================================================

class TestReportsTab:
    """Tests for reports generation tab."""
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_reports_without_experiment(self, mock_st, dashboard):
        """Test reports tab without experiment."""
        st.session_state.current_experiment = None
        
        dashboard.render_reports_tab()
        
        mock_st.info.assert_called_with("No experiment to report on. Complete training first.")
    
    @patch('automl_platform.ui.streamlit_app.st')
    @patch('time.sleep')
    def test_report_generation(self, mock_sleep, mock_st, dashboard):
        """Test report generation."""
        st.session_state.current_experiment = "test_experiment"
        mock_st.selectbox.side_effect = ["Executive Summary", "PDF"]
        mock_st.button.return_value = True
        
        dashboard.render_reports_tab()
        
        mock_st.success.assert_called()
        mock_st.download_button.assert_called()
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_report_options(self, mock_st, dashboard):
        """Test report configuration options."""
        st.session_state.current_experiment = "test_experiment"
        
        dashboard.render_reports_tab()
        
        # Check report options
        assert mock_st.selectbox.call_count >= 2  # Report type and format
        assert mock_st.checkbox.call_count >= 3  # Various include options
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_previous_reports_display(self, mock_st, dashboard):
        """Test display of previous reports."""
        st.session_state.current_experiment = "test_experiment"
        
        dashboard.render_reports_tab()
        
        # Check previous reports section
        mock_st.subheader.assert_any_call("📚 Previous Reports")
        mock_st.dataframe.assert_called()


# ============================================================================
# Sidebar Tests
# ============================================================================

class TestSidebar:
    """Tests for sidebar functionality."""
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_sidebar_experiment_status(self, mock_st, dashboard):
        """Test experiment status display in sidebar."""
        st.session_state.current_experiment = "test_exp"
        st.session_state.training_status = "completed"
        st.session_state.models_trained = [{"model": "XGBoost", "score": 0.92}]
        
        dashboard.render_sidebar()
        
        mock_st.success.assert_called()
        mock_st.metric.assert_called()
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_sidebar_configuration(self, mock_st, dashboard):
        """Test configuration options in sidebar."""
        dashboard.render_sidebar()
        
        # Check configuration sections
        mock_st.expander.assert_called()
        mock_st.text_input.assert_called()
        mock_st.selectbox.assert_called()
        mock_st.slider.assert_called()
    
    @patch('automl_platform.ui.streamlit_app.st')
    @patch('requests.get')
    def test_api_connection_test(self, mock_get, mock_st, dashboard):
        """Test API connection testing."""
        mock_get.return_value.status_code = 200
        mock_st.button.return_value = True
        
        result = dashboard.test_api_connection("http://localhost:8000")
        
        assert result is True
        mock_get.assert_called_with("http://localhost:8000/health", timeout=5)
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_quick_actions(self, mock_st, dashboard):
        """Test quick action buttons."""
        # Test cache clear
        mock_st.button.side_effect = [True, False]  # Clear cache clicked
        
        dashboard.render_sidebar()
        
        mock_st.cache_data.clear.assert_called()
        mock_st.success.assert_called()


# ============================================================================
# Helper Methods Tests
# ============================================================================

class TestHelperMethods:
    """Tests for helper methods."""
    
    def test_load_data_unsupported_format(self, dashboard):
        """Test loading unsupported file format."""
        file = MagicMock()
        file.name = "test.txt"
        
        with patch('automl_platform.ui.streamlit_app.st') as mock_st:
            df = dashboard.load_data(file)
            
            mock_st.error.assert_called_with("Unsupported file format")
            assert df.empty
    
    def test_add_quick_prompt(self, dashboard):
        """Test adding quick prompt to chat."""
        initial_length = len(st.session_state.chat_history)
        
        with patch('automl_platform.ui.streamlit_app.st') as mock_st:
            dashboard.add_quick_prompt("Test prompt")
            
            assert len(st.session_state.chat_history) == initial_length + 2
            mock_st.rerun.assert_called()
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_export_configuration(self, mock_st, dashboard):
        """Test configuration export."""
        st.session_state.current_experiment = "test_exp"
        st.session_state.models_trained = [{"model": "XGBoost", "score": 0.92}]
        
        dashboard.export_configuration()
        
        mock_st.download_button.assert_called()
        
        # Check download data contains expected fields
        call_args = mock_st.download_button.call_args
        data = json.loads(call_args[1]['data'])
        assert 'experiment' in data
        assert 'models_trained' in data
        assert 'timestamp' in data


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the dashboard."""
    
    @patch('automl_platform.ui.streamlit_app.st')
    @patch('time.sleep')
    def test_full_workflow(self, mock_sleep, mock_st, dashboard, sample_dataframe):
        """Test complete workflow from data upload to report generation."""
        # 1. Upload data
        st.session_state.uploaded_data = sample_dataframe
        
        # 2. Train models
        mock_st.button.return_value = True
        dashboard.render_training_tab()
        
        assert st.session_state.training_status == 'completed'
        assert len(st.session_state.models_trained) > 0
        
        # 3. Select model from leaderboard
        dashboard.render_leaderboard_tab()
        st.session_state.selected_model = "XGBoost"
        
        # 4. Analyze model
        dashboard.render_analysis_tab()
        
        # 5. Generate report
        st.session_state.current_experiment = "test_workflow"
        dashboard.render_reports_tab()
        
        # Verify all steps completed
        assert st.session_state.selected_model is not None
        assert st.session_state.current_experiment is not None
    
    @patch('automl_platform.ui.streamlit_app.st')
    def test_error_handling(self, mock_st, dashboard):
        """Test error handling in various scenarios."""
        # Test with None data
        st.session_state.uploaded_data = None
        dashboard.render_training_tab()
        mock_st.warning.assert_called()
        
        # Test with empty models list
        st.session_state.models_trained = []
        dashboard.render_leaderboard_tab()
        mock_st.info.assert_called()
        
        # Test without selected model
        st.session_state.selected_model = None
        dashboard.render_analysis_tab()
        mock_st.info.assert_called()


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for the dashboard."""
    
    def test_large_dataset_handling(self, dashboard):
        """Test handling of large datasets."""
        # Create large DataFrame
        large_df = pd.DataFrame(
            np.random.randn(10000, 100),
            columns=[f'col_{i}' for i in range(100)]
        )
        
        score, issues = dashboard.assess_data_quality(large_df)
        
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100
    
    def test_many_models_leaderboard(self, dashboard):
        """Test leaderboard with many models."""
        many_models = [
            {"model": f"Model_{i}", "score": np.random.random(), "time": np.random.randint(10, 100)}
            for i in range(100)
        ]
        
        st.session_state.models_trained = many_models
        
        with patch('automl_platform.ui.streamlit_app.st'):
            dashboard.render_leaderboard_tab()
            # Should not raise any errors
    
    def test_long_chat_history(self, dashboard):
        """Test chat with long conversation history."""
        long_history = [
            {'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'Message {i}'}
            for i in range(100)
        ]
        
        st.session_state.chat_history = long_history
        
        with patch('automl_platform.ui.streamlit_app.st'):
            dashboard.render_chat_tab()
            # Should handle long history without issues
