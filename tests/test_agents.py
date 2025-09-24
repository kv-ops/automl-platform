"""
Unit tests for intelligent data cleaning agents
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from automl_platform.agents import (
    DataCleaningOrchestrator,
    ProfilerAgent,
    ValidatorAgent,
    CleanerAgent,
    ControllerAgent,
    AgentConfig,
    AgentType
)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe with various data quality issues"""
    np.random.seed(42)
    df = pd.DataFrame({
        'id': range(1, 101),
        'amount': np.random.uniform(10, 1000, 100),
        'category': np.random.choice(['A', 'B', 'C', None], 100),
        'date': pd.date_range('2023-01-01', periods=100),
        'score': np.random.normal(50, 10, 100)
    })
    
    # Add some issues
    df.loc[5:10, 'amount'] = np.nan  # Missing values
    df.loc[20:25, 'score'] = 999  # Outliers
    df = pd.concat([df, df.iloc[30:35]])  # Duplicates
    
    return df


@pytest.fixture
def agent_config():
    """Create a test agent configuration"""
    return AgentConfig(
        openai_api_key="test-key",
        model="gpt-4-1106-preview",
        enable_web_search=True,
        enable_file_operations=True,
        max_iterations=2,
        timeout_seconds=30,
        user_context={
            "secteur_activite": "finance",
            "target_variable": "score",
            "contexte_metier": "Test context"
        }
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client"""
    client = AsyncMock()
    
    # Mock assistant creation
    client.beta.assistants.create = AsyncMock(return_value=Mock(id="test-assistant-id"))
    client.beta.assistants.retrieve = AsyncMock(return_value=Mock(id="test-assistant-id"))
    
    # Mock thread operations
    client.beta.threads.create = AsyncMock(return_value=Mock(id="test-thread-id"))
    client.beta.threads.messages.create = AsyncMock()
    client.beta.threads.messages.list = AsyncMock(return_value=Mock(data=[
        Mock(role="assistant", content=[Mock(text=Mock(value='{"result": "test"}'))])
    ]))
    
    # Mock run operations
    client.beta.threads.runs.create = AsyncMock(return_value=Mock(id="test-run-id"))
    client.beta.threads.runs.retrieve = AsyncMock(return_value=Mock(status="completed"))
    
    return client


class TestAgentConfig:
    """Test AgentConfig class"""
    
    def test_config_initialization(self):
        """Test configuration initialization"""
        config = AgentConfig()
        assert config.model == "gpt-4-1106-preview"
        assert config.enable_web_search == True
        assert config.max_iterations == 3
        
    def test_config_validation(self, tmp_path):
        """Test configuration validation"""
        config = AgentConfig()
        config.cache_dir = str(tmp_path / "cache")
        config.output_dir = str(tmp_path / "output")
        
        # Should raise without API key
        config.openai_api_key = ""
        with pytest.raises(ValueError):
            config.validate()
        
        # Should pass with API key
        config.openai_api_key = "test-key"
        assert config.validate() == True
        
    def test_agent_tools_config(self):
        """Test agent tools configuration"""
        config = AgentConfig()
        
        profiler_tools = config.get_agent_tools(AgentType.PROFILER)
        assert len(profiler_tools) == 1
        assert profiler_tools[0]["type"] == "code_interpreter"
        
        validator_tools = config.get_agent_tools(AgentType.VALIDATOR)
        assert len(validator_tools) == 2
        assert any(tool["type"] == "function" for tool in validator_tools)
        
    def test_sector_keywords(self):
        """Test sector keyword retrieval"""
        config = AgentConfig()
        
        finance_keywords = config.get_sector_keywords("finance")
        assert "IFRS" in finance_keywords
        assert "Basel" in finance_keywords
        
        health_keywords = config.get_sector_keywords("sante")
        assert "HL7" in health_keywords
        assert "ICD-10" in health_keywords


@pytest.mark.asyncio
class TestProfilerAgent:
    """Test ProfilerAgent class"""
    
    async def test_profiler_initialization(self, agent_config, mock_openai_client):
        """Test profiler agent initialization"""
        with patch('automl_platform.agents.profiler_agent.AsyncOpenAI', return_value=mock_openai_client):
            profiler = ProfilerAgent(agent_config)
            await asyncio.sleep(0.1)  # Allow async initialization
            
            assert profiler.config == agent_config
            assert profiler.assistant_id is not None
    
    async def test_profiler_analyze(self, agent_config, sample_dataframe, mock_openai_client):
        """Test profiler analysis"""
        with patch('automl_platform.agents.profiler_agent.AsyncOpenAI', return_value=mock_openai_client):
            profiler = ProfilerAgent(agent_config)
            await asyncio.sleep(0.1)
            
            # Mock response
            mock_response = {
                "summary": {"total_rows": 100, "total_columns": 5},
                "quality_issues": ["Missing values in amount"],
                "anomalies": ["Outliers in score"]
            }
            
            mock_openai_client.beta.threads.messages.list = AsyncMock(return_value=Mock(data=[
                Mock(role="assistant", content=[Mock(text=Mock(value=json.dumps(mock_response)))])
            ]))
            
            result = await profiler.analyze(sample_dataframe)
            
            assert "summary" in result
            assert "quality_issues" in result
            assert "anomalies" in result
    
    def test_prepare_data_summary(self, agent_config, sample_dataframe):
        """Test data summary preparation"""
        profiler = ProfilerAgent(agent_config)
        summary = profiler._prepare_data_summary(sample_dataframe)
        
        assert summary["shape"] == sample_dataframe.shape
        assert "columns" in summary
        assert "basic_stats" in summary
        assert "duplicates" in summary
        
        # Check statistics
        for col in sample_dataframe.columns:
            assert col in summary["basic_stats"]
            col_stats = summary["basic_stats"][col]
            assert "null_count" in col_stats
            assert "unique_count" in col_stats
    
    def test_basic_profiling_fallback(self, agent_config, sample_dataframe):
        """Test basic profiling fallback"""
        profiler = ProfilerAgent(agent_config)
        result = profiler._basic_profiling(sample_dataframe)
        
        assert "summary" in result
        assert "columns" in result
        assert "quality_issues" in result
        assert result["summary"]["total_rows"] == len(sample_dataframe)


@pytest.mark.asyncio
class TestValidatorAgent:
    """Test ValidatorAgent class"""
    
    async def test_validator_initialization(self, agent_config, mock_openai_client):
        """Test validator agent initialization"""
        with patch('automl_platform.agents.validator_agent.AsyncOpenAI', return_value=mock_openai_client):
            validator = ValidatorAgent(agent_config)
            await asyncio.sleep(0.1)
            
            assert validator.config == agent_config
            assert validator.cache_dir.exists()
    
    async def test_web_search_caching(self, agent_config, tmp_path):
        """Test web search caching functionality"""
        agent_config.cache_dir = str(tmp_path / "cache")
        validator = ValidatorAgent(agent_config)
        
        # Mock search results
        mock_results = {
            "query": "test query",
            "results": [{"title": "Test", "url": "http://test.com", "snippet": "Test snippet"}],
            "urls": ["http://test.com"]
        }
        
        with patch.object(validator, '_perform_web_search', return_value=mock_results):
            # First call should perform search
            result1 = await validator._web_search("test query")
            assert result1 == mock_results
            
            # Second call should use cache
            result2 = await validator._web_search("test query")
            assert result2 == mock_results
    
    def test_basic_validation(self, agent_config, sample_dataframe):
        """Test basic validation fallback"""
        validator = ValidatorAgent(agent_config)
        
        # Test finance sector validation
        result = validator._basic_validation(sample_dataframe, "finance")
        assert "valid" in result
        assert "issues" in result
        assert "warnings" in result
        assert "suggestions" in result
        
        # Test healthcare sector validation
        result_health = validator._basic_validation(sample_dataframe, "sante")
        assert "valid" in result_health


@pytest.mark.asyncio
class TestCleanerAgent:
    """Test CleanerAgent class"""
    
    async def test_cleaner_initialization(self, agent_config, mock_openai_client):
        """Test cleaner agent initialization"""
        with patch('automl_platform.agents.cleaner_agent.AsyncOpenAI', return_value=mock_openai_client):
            cleaner = CleanerAgent(agent_config)
            await asyncio.sleep(0.1)
            
            assert cleaner.config == agent_config
            assert cleaner.transformations_history == []
    
    def test_validate_transformation(self, agent_config):
        """Test transformation validation"""
        cleaner = CleanerAgent(agent_config)
        
        # Valid transformation
        valid_trans = {
            "column": "amount",
            "action": "fill_missing",
            "params": {"method": "median"}
        }
        assert cleaner._validate_transformation(valid_trans) == True
        
        # Invalid transformation (missing required field)
        invalid_trans = {
            "action": "fill_missing",
            "params": {"method": "median"}
        }
        assert cleaner._validate_transformation(invalid_trans) == False
        
        # Invalid action
        invalid_action = {
            "column": "amount",
            "action": "invalid_action",
            "params": {}
        }
        assert cleaner._validate_transformation(invalid_action) == False
    
    @pytest.mark.asyncio
    async def test_apply_transformations(self, agent_config, sample_dataframe):
        """Test applying transformations"""
        cleaner = CleanerAgent(agent_config)
        
        transformations = [
            {
                "column": "amount",
                "action": "fill_missing",
                "params": {"method": "median"}
            },
            {
                "column": "score",
                "action": "handle_outliers",
                "params": {"method": "clip"}
            }
        ]
        
        original_missing = sample_dataframe["amount"].isnull().sum()
        cleaned_df = await cleaner._apply_transformations(sample_dataframe, transformations)
        
        # Check missing values were filled
        assert cleaned_df["amount"].isnull().sum() < original_missing
        
        # Check outliers were clipped
        assert cleaned_df["score"].max() < 999
    
    def test_basic_cleaning_fallback(self, agent_config, sample_dataframe):
        """Test basic cleaning fallback"""
        cleaner = CleanerAgent(agent_config)
        
        cleaned_df, transformations = cleaner._basic_cleaning(
            sample_dataframe, 
            {"quality_issues": []}
        )
        
        # Check duplicates removed
        assert cleaned_df.duplicated().sum() < sample_dataframe.duplicated().sum()
        
        # Check transformations recorded
        assert len(transformations) > 0
        assert any(t["action"] == "remove_duplicates" for t in transformations)


@pytest.mark.asyncio
class TestControllerAgent:
    """Test ControllerAgent class"""
    
    async def test_controller_initialization(self, agent_config, mock_openai_client):
        """Test controller agent initialization"""
        with patch('automl_platform.agents.controller_agent.AsyncOpenAI', return_value=mock_openai_client):
            controller = ControllerAgent(agent_config)
            await asyncio.sleep(0.1)
            
            assert controller.config == agent_config
            assert controller.quality_metrics == {}
    
    def test_calculate_quality_metrics(self, agent_config, sample_dataframe):
        """Test quality metrics calculation"""
        controller = ControllerAgent(agent_config)
        
        # Create a cleaned version
        cleaned_df = sample_dataframe.drop_duplicates()
        cleaned_df["amount"].fillna(cleaned_df["amount"].median(), inplace=True)
        
        metrics = controller._calculate_quality_metrics(cleaned_df, sample_dataframe)
        
        assert "data_quality" in metrics
        assert "transformation_impact" in metrics
        assert "statistical_changes" in metrics
        assert "integrity_checks" in metrics
        assert "quality_score" in metrics
        
        # Check specific metrics
        assert metrics["data_quality"]["duplicates"]["cleaned"] < metrics["data_quality"]["duplicates"]["original"]
        assert metrics["quality_score"] >= 0 and metrics["quality_score"] <= 100
    
    def test_perform_integrity_checks(self, agent_config, sample_dataframe):
        """Test integrity checks"""
        controller = ControllerAgent(agent_config)
        
        checks = controller._perform_integrity_checks(sample_dataframe)
        
        assert "no_empty_dataframe" in checks
        assert "no_all_null_columns" in checks
        assert "no_duplicate_columns" in checks
        assert "reasonable_missing_ratio" in checks
        assert "no_constant_columns" in checks
        
        # All checks should pass for sample data
        assert checks["no_empty_dataframe"] == True
        assert checks["no_duplicate_columns"] == True


@pytest.mark.asyncio
class TestDataCleaningOrchestrator:
    """Test DataCleaningOrchestrator class"""
    
    async def test_orchestrator_initialization(self, agent_config):
        """Test orchestrator initialization"""
        with patch('automl_platform.agents.data_cleaning_orchestrator.ProfilerAgent'), \
             patch('automl_platform.agents.data_cleaning_orchestrator.ValidatorAgent'), \
             patch('automl_platform.agents.data_cleaning_orchestrator.CleanerAgent'), \
             patch('automl_platform.agents.data_cleaning_orchestrator.ControllerAgent'):
            
            orchestrator = DataCleaningOrchestrator(agent_config)
            
            assert orchestrator.config == agent_config
            assert orchestrator.profiler is not None
            assert orchestrator.validator is not None
            assert orchestrator.cleaner is not None
            assert orchestrator.controller is not None
    
    def test_needs_chunking(self, agent_config):
        """Test dataset chunking detection"""
        orchestrator = DataCleaningOrchestrator(agent_config)
        
        # Small dataset - no chunking needed
        small_df = pd.DataFrame({'a': range(100), 'b': range(100)})
        assert orchestrator._needs_chunking(small_df) == False
        
        # Large dataset simulation
        orchestrator.config.chunk_size_mb = 0.001  # Very small for testing
        assert orchestrator._needs_chunking(small_df) == True
    
    def test_chunk_dataset(self, agent_config, sample_dataframe):
        """Test dataset chunking"""
        orchestrator = DataCleaningOrchestrator(agent_config)
        orchestrator.config.chunk_size_mb = 0.001  # Force chunking
        
        chunks = orchestrator._chunk_dataset(sample_dataframe)
        
        assert len(chunks) > 1
        # Check all data is preserved
        total_rows = sum(len(chunk) for chunk in chunks)
        assert total_rows == len(sample_dataframe)
    
    def test_generate_final_report(self, agent_config, sample_dataframe):
        """Test final report generation"""
        orchestrator = DataCleaningOrchestrator(agent_config)
        
        # Create a cleaned version
        cleaned_df = sample_dataframe.drop_duplicates()
        
        orchestrator.transformations_applied = [
            {"column": "amount", "action": "fill_missing", "params": {"method": "median"}}
        ]
        orchestrator.validation_sources = ["https://example.com/standards"]
        orchestrator.start_time = time.time()
        
        report = orchestrator._generate_final_report(sample_dataframe, cleaned_df)
        
        assert "metadata" in report
        assert "transformations" in report
        assert "validation_sources" in report
        assert "quality_metrics" in report
        
        # Check metrics
        assert report["quality_metrics"]["duplicates_removed"] > 0
        assert report["metadata"]["original_shape"] == sample_dataframe.shape
        assert report["metadata"]["cleaned_shape"] == cleaned_df.shape
    
    @pytest.mark.asyncio
    async def test_fallback_cleaning(self, agent_config, sample_dataframe):
        """Test fallback to traditional cleaning"""
        orchestrator = DataCleaningOrchestrator(agent_config)
        
        user_context = {
            "secteur_activite": "finance",
            "target_variable": "score"
        }
        
        cleaned_df, report = await orchestrator._fallback_cleaning(
            sample_dataframe, 
            user_context
        )
        
        # Check basic cleaning was applied
        assert cleaned_df.duplicated().sum() == 0
        assert cleaned_df.isnull().sum().sum() < sample_dataframe.isnull().sum().sum()
        assert report["metadata"]["fallback"] == True
    
    def test_estimate_cost(self, agent_config, sample_dataframe):
        """Test cost estimation"""
        orchestrator = DataCleaningOrchestrator(agent_config)
        
        cost = orchestrator.estimate_cost(sample_dataframe)
        
        assert cost > 0
        assert cost <= orchestrator.config.max_cost_per_dataset


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    @patch('automl_platform.agents.profiler_agent.AsyncOpenAI')
    @patch('automl_platform.agents.validator_agent.AsyncOpenAI')
    @patch('automl_platform.agents.cleaner_agent.AsyncOpenAI')
    @patch('automl_platform.agents.controller_agent.AsyncOpenAI')
    async def test_full_cleaning_pipeline(
        self, 
        mock_controller_client,
        mock_cleaner_client, 
        mock_validator_client,
        mock_profiler_client,
        sample_dataframe,
        agent_config
    ):
        """Test the complete cleaning pipeline"""
        
        # Setup mocks
        mock_assistant = Mock(id="test-assistant")
        mock_thread = Mock(id="test-thread")
        mock_run = Mock(id="test-run", status="completed")
        
        for mock_client in [mock_profiler_client, mock_validator_client, 
                           mock_cleaner_client, mock_controller_client]:
            client = AsyncMock()
            client.beta.assistants.create = AsyncMock(return_value=mock_assistant)
            client.beta.threads.create = AsyncMock(return_value=mock_thread)
            client.beta.threads.runs.create = AsyncMock(return_value=mock_run)
            client.beta.threads.runs.retrieve = AsyncMock(return_value=mock_run)
            client.beta.threads.messages.list = AsyncMock(
                return_value=Mock(data=[
                    Mock(role="assistant", content=[
                        Mock(text=Mock(value='{"result": "success"}'))
                    ])
                ])
            )
            mock_client.return_value = client
        
        # Create orchestrator
        orchestrator = DataCleaningOrchestrator(agent_config)
        
        # Define user context
        user_context = {
            "secteur_activite": "finance",
            "target_variable": "score",
            "contexte_metier": "Risk assessment"
        }
        
        # Run cleaning pipeline
        cleaned_df, report = await orchestrator.clean_dataset(
            sample_dataframe, 
            user_context
        )
        
        # Verify results
        assert cleaned_df is not None
        assert report is not None
        assert "metadata" in report
        assert report["metadata"]["industry"] == "finance"
    
    @pytest.mark.asyncio
    async def test_sector_specific_cleaning(self, agent_config):
        """Test cleaning for different sectors"""
        
        # Healthcare data
        healthcare_df = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'diagnosis': ['A01', 'B02', None],
            'lab_result': [120, 999, 110]  # Include outlier
        })
        
        healthcare_context = {
            "secteur_activite": "sante",
            "target_variable": "lab_result",
            "contexte_metier": "Patient diagnosis"
        }
        
        with patch('automl_platform.agents.data_cleaning_orchestrator.ProfilerAgent'), \
             patch('automl_platform.agents.data_cleaning_orchestrator.ValidatorAgent'), \
             patch('automl_platform.agents.data_cleaning_orchestrator.CleanerAgent'), \
             patch('automl_platform.agents.data_cleaning_orchestrator.ControllerAgent'):
            
            orchestrator = DataCleaningOrchestrator(agent_config)
            
            # Test that sector context is properly set
            assert orchestrator.config.user_context.get("secteur_activite") is None
            
            # This would be called in the actual cleaning
            orchestrator.config.user_context.update(healthcare_context)
            assert orchestrator.config.user_context["secteur_activite"] == "sante"


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self, agent_config):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        
        profiler = ProfilerAgent(agent_config)
        result = profiler._basic_profiling(empty_df)
        
        assert result["summary"]["total_rows"] == 0
        assert result["summary"]["total_columns"] == 0
    
    def test_all_null_column(self, agent_config):
        """Test handling of all-null columns"""
        df = pd.DataFrame({
            'good_col': [1, 2, 3],
            'bad_col': [None, None, None]
        })
        
        cleaner = CleanerAgent(agent_config)
        cleaned_df, _ = cleaner._basic_cleaning(df, {})
        
        # All-null column should be removed
        assert 'bad_col' not in cleaned_df.columns
    
    def test_single_value_column(self, agent_config):
        """Test handling of constant columns"""
        df = pd.DataFrame({
            'constant': [1, 1, 1, 1],
            'variable': [1, 2, 3, 4]
        })
        
        controller = ControllerAgent(agent_config)
        checks = controller._perform_integrity_checks(df)
        
        assert checks["no_constant_columns"] == False
    
    @pytest.mark.asyncio
    async def test_api_timeout(self, agent_config):
        """Test handling of API timeout"""
        agent_config.timeout_seconds = 0.1  # Very short timeout
        
        with patch('automl_platform.agents.profiler_agent.AsyncOpenAI') as mock_client:
            client = AsyncMock()
            client.beta.threads.runs.retrieve = AsyncMock(
                return_value=Mock(status="in_progress")
            )
            mock_client.return_value = client
            
            profiler = ProfilerAgent(agent_config)
            profiler.assistant = Mock(id="test-assistant")
            profiler.assistant_id = "test-assistant"
            
            df = pd.DataFrame({'a': [1, 2, 3]})
            
            # Should raise TimeoutError
            with pytest.raises(TimeoutError):
                await profiler.analyze(df)
    
    def test_invalid_transformation(self, agent_config):
        """Test handling of invalid transformations"""
        cleaner = CleanerAgent(agent_config)
        
        # Invalid transformation without column
        invalid_trans = {"action": "fill_missing"}
        assert cleaner._validate_transformation(invalid_trans) == False
        
        # Invalid action
        invalid_action = {
            "column": "test",
            "action": "invalid_action"
        }
        assert cleaner._validate_transformation(invalid_action) == False
    
    @pytest.mark.asyncio
    async def test_web_search_failure(self, agent_config):
        """Test handling of web search failures"""
        validator = ValidatorAgent(agent_config)
        
        with patch.object(validator, '_perform_web_search', side_effect=Exception("Network error")):
            result = await validator._web_search("test query")
            # Should return empty results on failure
            assert result == {"query": "test query", "results": [], "urls": []}


class TestPerformance:
    """Performance and scalability tests"""
    
    def test_large_dataset_handling(self, agent_config):
        """Test handling of large datasets"""
        # Create a large dataset
        large_df = pd.DataFrame(
            np.random.randn(10000, 50),
            columns=[f'col_{i}' for i in range(50)]
        )
        
        orchestrator = DataCleaningOrchestrator(agent_config)
        orchestrator.config.chunk_size_mb = 1  # Force chunking
        
        chunks = orchestrator._chunk_dataset(large_df)
        
        assert len(chunks) > 1
        # Verify no data loss
        reconstructed = pd.concat(chunks, ignore_index=True)
        assert len(reconstructed) == len(large_df)
        assert reconstructed.shape[1] == large_df.shape[1]
    
    def test_high_cardinality_handling(self, agent_config):
        """Test handling of high cardinality columns"""
        # Create high cardinality column
        df = pd.DataFrame({
            'high_card': [f'val_{i}' for i in range(1000)],
            'low_card': ['A'] * 500 + ['B'] * 500
        })
        
        profiler = ProfilerAgent(agent_config)
        summary = profiler._prepare_data_summary(df)
        
        assert 'high_card' in summary['basic_stats']
        assert summary['basic_stats']['high_card']['unique_count'] == 1000
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self, agent_config):
        """Test concurrent execution of multiple agents"""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        profile_report = {"summary": "test"}
        
        with patch('automl_platform.agents.validator_agent.AsyncOpenAI'), \
             patch('automl_platform.agents.cleaner_agent.AsyncOpenAI'):
            
            validator = ValidatorAgent(agent_config)
            cleaner = CleanerAgent(agent_config)
            
            # Simulate concurrent execution
            tasks = [
                validator._basic_validation(df, "finance"),
                cleaner._basic_cleaning(df, profile_report)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            assert len(results) == 2
            assert not any(isinstance(r, Exception) for r in results)


# Fixtures for pytest
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
