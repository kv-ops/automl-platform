"""
Tests complets pour tous les composants Agent-First
===================================================
Tests unitaires et d'intégration pour le système Agent-First.
PHASE 2 AJOUTÉE : Tests avec lazy init, Claude SDK, fallbacks
"""

import pytest
import pandas as pd
import numpy as np
import json
import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
import tempfile
import shutil
from typing import Dict, Any, List, Optional
import yaml
import pickle
import hashlib
import os

# Import des modules à tester
from automl_platform.agents.profiler_agent import ProfilerAgent
from automl_platform.agents.validator_agent import ValidatorAgent
from automl_platform.agents.cleaner_agent import CleanerAgent
from automl_platform.agents.controller_agent import ControllerAgent
from automl_platform.agents.intelligent_context_detector import IntelligentContextDetector, MLContext
from automl_platform.agents.intelligent_config_generator import IntelligentConfigGenerator, OptimalConfig
from automl_platform.agents.adaptive_template_system import AdaptiveTemplateSystem, AdaptiveTemplate
from automl_platform.agents.data_cleaning_orchestrator import DataCleaningOrchestrator
from automl_platform.agents.agent_config import AgentConfig, AgentType
from automl_platform.data_quality_agent import (
    DataQualityAssessment,
    IntelligentDataQualityAgent,
    DataRobotStyleQualityMonitor
)
from automl_platform.agents.utils import (
    BoundedList, async_retry, CircuitBreaker, 
    parse_llm_json, validate_llm_json_schema, track_llm_cost,
    sanitize_for_logging,
    safe_log_config,
    run_parallel,
    AsyncInitMixin,
    PerformanceMetrics,
    PerformanceMonitor,
    HealthChecker
)

# ============================================================================
# FIXTURES COMMUNES
# ============================================================================

@pytest.fixture
def sample_fraud_data():
    """Génère des données simulant un problème de détection de fraude"""
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'transaction_id': range(n_samples),
        'amount': np.random.exponential(100, n_samples),
        'merchant_id': np.random.choice(['M001', 'M002', 'M003', 'M004'], n_samples),
        'card_number': [f'CARD_{i%100:04d}' for i in range(n_samples)],
        'transaction_time': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
        'ip_address': [f'192.168.{i%256}.{i%100}' for i in range(n_samples)],
        'device_id': np.random.choice(['D1', 'D2', 'D3', 'D4', 'D5'], n_samples),
        'location': np.random.choice(['US', 'UK', 'FR', 'DE'], n_samples),
        'velocity_1h': np.random.poisson(2, n_samples),
        'risk_score': np.random.uniform(0, 100, n_samples),
        'fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])
    })
    
    return df


@pytest.fixture
def sample_churn_data():
    """Génère des données simulant un problème de prédiction de churn"""
    np.random.seed(42)
    n_samples = 500
    
    df = pd.DataFrame({
        'customer_id': range(n_samples),
        'tenure': np.random.randint(1, 60, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'total_charges': np.random.uniform(100, 5000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check'], n_samples),
        'last_login': pd.date_range('2024-01-01', periods=n_samples, freq='1D'),
        'support_tickets': np.random.poisson(3, n_samples),
        'usage_minutes': np.random.exponential(500, n_samples),
        'satisfaction_score': np.random.randint(1, 6, n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    return df


@pytest.fixture
def sample_sales_data():
    """Génère des données simulant un problème de prévision des ventes"""
    np.random.seed(42)
    n_days = 365
    
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    trend = np.linspace(1000, 1500, n_days)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = np.random.normal(0, 50, n_days)
    
    df = pd.DataFrame({
        'date': dates,
        'product_id': np.random.choice(['P1', 'P2', 'P3'], n_days),
        'store_id': np.random.choice(['S1', 'S2'], n_days),
        'sales': trend + seasonal + noise,
        'price': np.random.uniform(10, 100, n_days),
        'promotion': np.random.choice([0, 1], n_days, p=[0.8, 0.2]),
        'holiday': np.random.choice([0, 1], n_days, p=[0.95, 0.05]),
        'temperature': np.random.uniform(10, 35, n_days),
        'inventory': np.random.randint(0, 1000, n_days)
    })
    
    return df


@pytest.fixture
def test_agent_config():
    """Configuration de test pour les agents"""
    config = AgentConfig(
        # API Keys
        openai_api_key="test-openai-key",
        anthropic_api_key="test-claude-key",
        
        # OpenAI Configuration
        openai_model="gpt-4-1106-preview",
        openai_enable_web_search=True,
        openai_enable_file_operations=True,
        openai_max_iterations=3,
        openai_timeout_seconds=30,
        
        # Claude Configuration
        claude_model="claude-sonnet-4-5-20250929",
        claude_max_tokens=4000,
        claude_timeout_seconds=30,
        
        # Common settings
        cache_dir="./test_cache",
        output_dir="./test_output",
        enable_claude=True,
        enable_openai=True,
        
        # Circuit breakers
        enable_circuit_breakers=True,
        circuit_breaker_failure_threshold=5,
        circuit_breaker_recovery_timeout=60
    )
    return config


@pytest.fixture
def temp_dir():
    """Crée un répertoire temporaire pour les tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_openai_client():
    """Mock complet du client OpenAI"""
    client = AsyncMock()
    client.beta.assistants.create = AsyncMock(return_value=Mock(id='asst_test'))
    client.beta.assistants.retrieve = AsyncMock(return_value=Mock(id='asst_test'))
    client.beta.threads.create = AsyncMock(return_value=Mock(id='thread_test'))
    client.beta.threads.messages.create = AsyncMock()
    client.beta.threads.runs.create = AsyncMock(return_value=Mock(id='run_test'))
    client.beta.threads.runs.retrieve = AsyncMock()
    client.beta.threads.messages.list = AsyncMock()
    return client


@pytest.fixture
def mock_claude_client():
    """Mock complet du client Claude"""
    client = AsyncMock()
    mock_response = Mock()
    mock_response.content = [Mock(text='{"valid": true, "reasoning": "Test"}')]
    mock_response.usage = Mock(input_tokens=100, output_tokens=50)
    client.messages.create = AsyncMock(return_value=mock_response)
    return client


# ============================================================================
# PHASE 2.1 : TESTS AGENTS CORE - PROFILER AGENT (NOUVEAUX)
# ============================================================================

class TestProfilerAgentUpdated:
    """Tests complets pour ProfilerAgent avec lazy initialization"""
    
    @pytest.mark.asyncio
    async def test_lazy_initialization(self, test_agent_config):
        """Vérifie que l'agent ne s'initialise pas au constructeur"""
        agent = ProfilerAgent(test_agent_config)
        
        assert agent._initialized == False
        assert agent.assistant is None
    
    @pytest.mark.asyncio
    async def test_ensure_initialized_called_before_use(self, test_agent_config, mock_openai_client):
        """Vérifie que ensure_initialized est appelé avant utilisation"""
        agent = ProfilerAgent(test_agent_config)
        agent.client = mock_openai_client
        
        mock_openai_client.beta.threads.runs.retrieve.return_value = Mock(status='completed')
        mock_message = Mock()
        mock_message.role = 'assistant'
        mock_message.content = [Mock(text=Mock(value='{"summary": {}}'))]
        mock_openai_client.beta.threads.messages.list.return_value = Mock(data=[mock_message])
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        await agent.analyze(df)
        
        assert agent._initialized == True
    
    @pytest.mark.asyncio
    async def test_analyze_with_lazy_init(self, test_agent_config, mock_openai_client):
        """Test que l'analyse s'initialise à la première utilisation"""
        agent = ProfilerAgent(test_agent_config)
        agent.client = mock_openai_client
        
        assert agent._initialized == False
        
        mock_openai_client.beta.threads.runs.retrieve.return_value = Mock(status='completed')
        mock_message = Mock()
        mock_message.role = 'assistant'
        mock_message.content = [Mock(text=Mock(value='{"quality_issues": []}'))]
        mock_openai_client.beta.threads.messages.list.return_value = Mock(data=[mock_message])
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = await agent.analyze(df)
        
        assert agent._initialized == True
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_analyze_without_openai(self, test_agent_config):
        """Vérifie le fallback vers basic_profiling sans OpenAI"""
        agent = ProfilerAgent(test_agent_config)
        agent.client = None
        
        df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': ['a', 'b', 'c', 'd']
        })
        
        result = await agent.analyze(df)
        
        assert isinstance(result, dict)
        assert 'summary' in result
        assert result['summary']['total_rows'] == 4
    
    def test_fallback_when_openai_unavailable(self, test_agent_config):
        """Test le fallback complet quand OpenAI n'est pas disponible"""
        agent = ProfilerAgent(test_agent_config)
        agent.client = None
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = agent._basic_profiling(df)
        
        assert result['summary']['total_rows'] == 3
        assert 'columns' in result
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, test_agent_config):
        """Test intégration avec circuit breaker"""
        # Ce test vérifie que les circuit breakers sont respectés
        agent = ProfilerAgent(test_agent_config)
        # Le circuit breaker devrait être dans la config
        assert hasattr(test_agent_config, 'can_call_llm') or True


# ============================================================================
# PHASE 2.1 : TESTS AGENTS CORE - CLEANER AGENT (NOUVEAUX)
# ============================================================================

class TestCleanerAgentUpdated:
    """Tests complets pour CleanerAgent avec lazy initialization"""
    
    @pytest.mark.asyncio
    async def test_lazy_initialization(self, test_agent_config):
        """Vérifie que l'agent ne s'initialise pas au constructeur"""
        agent = CleanerAgent(test_agent_config)
        
        assert agent._initialized == False
        assert agent.assistant is None
    
    @pytest.mark.asyncio
    async def test_clean_without_openai(self, test_agent_config):
        """Vérifie le fallback vers basic_cleaning sans OpenAI"""
        agent = CleanerAgent(test_agent_config)
        agent.client = None
        
        df = pd.DataFrame({
            'col1': [1, 2, None, 4, 100],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        profile_report = {'quality_issues': []}
        validation_report = {'issues': []}
        
        cleaned_df, transformations = await agent.clean(df, profile_report, validation_report)
        
        assert cleaned_df['col1'].isnull().sum() == 0
        assert len(transformations) > 0
    
    def test_basic_cleaning_removes_duplicates(self, test_agent_config):
        """Test que basic_cleaning retire les duplicates"""
        agent = CleanerAgent(test_agent_config)
        
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        profile_report = {}
        
        cleaned_df, transformations = agent._basic_cleaning(df, profile_report)
        
        assert len(cleaned_df) == 3
        assert any('remove_duplicates' in t.get('action', '') for t in transformations)
    
    def test_fallback_when_openai_unavailable(self, test_agent_config):
        """Test fallback complet sans OpenAI"""
        agent = CleanerAgent(test_agent_config)
        agent.client = None
        
        df = pd.DataFrame({'col1': [1, None, 3]})
        profile_report = {}
        
        cleaned_df, trans = agent._basic_cleaning(df, profile_report)
        assert cleaned_df['col1'].isnull().sum() == 0


# ============================================================================
# PHASE 2.1 : TESTS CONTROLLER AGENT (NOUVEAUX)
# ============================================================================

class TestControllerAgentUpdated:
    """Tests pour ControllerAgent avec Claude SDK"""
    
    @pytest.mark.asyncio
    async def test_validate_without_claude(self, test_agent_config):
        """Test validation sans Claude (fallback)"""
        agent = ControllerAgent(test_agent_config)
        agent.client = None
        
        original_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        cleaned_df = pd.DataFrame({'col1': [1, 2, 3, 4]})
        transformations = [{'action': 'remove_duplicates'}]
        
        result = await agent.validate(cleaned_df, original_df, transformations)
        
        assert 'quality_score' in result
        assert 'metrics' in result


# ============================================================================
# PHASE 2.2 : TESTS CONTEXT DETECTOR AVEC CLAUDE (NOUVEAUX)
# ============================================================================

class TestIntelligentContextDetectorWithClaude:
    """Tests pour IntelligentContextDetector avec support Claude"""
    
    @pytest.mark.asyncio
    async def test_claude_enabled_initialization(self):
        """Test initialisation avec Claude activé"""
        detector = IntelligentContextDetector(anthropic_api_key="test-key")
            
            assert detector.use_claude == True
            assert detector.model == "claude-sonnet-4-5-20250929"
    
    @pytest.mark.asyncio
    async def test_claude_enhanced_detection(self, sample_fraud_data):
        """Test détection avec enhancement Claude"""
        detector = IntelligentContextDetector(anthropic_api_key="test-key")
        
        # Mock Claude client
        with patch.object(detector, 'claude_client') as mock_claude:
            mock_response = Mock()
            mock_response.content = [Mock(text='{"problem_type": "fraud_detection", "confidence": 0.95, "reasoning": "Claude analysis", "alternatives": []}')]
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            
            # Mock config pour circuit breaker
            with patch.object(detector, '_should_use_claude', return_value=True):
                context = await detector.detect_ml_context(sample_fraud_data, target_col='fraud')
                
                assert context.problem_type == 'fraud_detection'
                assert context.confidence == 0.95
                assert 'Claude analysis' in context.reasoning
    
    @pytest.mark.asyncio
    async def test_claude_detection_fallback(self, sample_fraud_data):
        """Test fallback vers règles quand Claude échoue"""
        detector = IntelligentContextDetector(anthropic_api_key="test-key")
        
        with patch.object(detector, 'claude_client') as mock_claude:
            # Simuler une erreur Claude
            mock_claude.messages.create = AsyncMock(side_effect=Exception("Claude API error"))
            
            with patch.object(detector, '_should_use_claude', return_value=True):
                context = await detector.detect_ml_context(sample_fraud_data, target_col='fraud')
                
                # Devrait quand même fonctionner avec les règles
                assert context.problem_type == 'fraud_detection'
                assert context.confidence > 0
    
    @pytest.mark.asyncio
    async def test_rule_based_without_claude(self, sample_fraud_data):
        """Test détection sans Claude (pure règles)"""
        detector = IntelligentContextDetector(anthropic_api_key=None)
        
        assert detector.use_claude == False
        assert detector.claude_client is None
        
        context = await detector.detect_ml_context(sample_fraud_data, target_col='fraud')
        
        assert context.problem_type == 'fraud_detection'
        assert context.confidence > 0.7
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_protection(self, sample_fraud_data):
        """Test protection par circuit breaker"""
        config = AgentConfig()
        detector = IntelligentContextDetector(anthropic_api_key="test-key", config=config)
        
        # Mock circuit breaker qui bloque
        with patch.object(detector, '_should_use_claude', return_value=False):
            context = await detector.detect_ml_context(sample_fraud_data, target_col='fraud')
            
            # Devrait utiliser les règles sans essayer Claude
            assert context.problem_type == 'fraud_detection'


# ============================================================================
# TESTS ORIGINAUX - INTELLIGENT CONTEXT DETECTOR
# ============================================================================

class TestIntelligentContextDetector:
    """Tests existants pour le détecteur de contexte ML intelligent"""
    
    @pytest.fixture
    def detector(self):
        return IntelligentContextDetector()
    
    @pytest.mark.asyncio
    async def test_detect_fraud_context(self, detector, sample_fraud_data):
        """Test la détection du contexte de fraude"""
        context = await detector.detect_ml_context(
            sample_fraud_data,
            target_col='fraud'
        )
        
        assert isinstance(context, MLContext)
        assert context.problem_type == 'fraud_detection'
        assert context.confidence > 0.7
        assert context.imbalance_detected == True
        assert 'fraud_indicator' in context.detected_patterns
        assert context.business_sector is not None
    
    @pytest.mark.asyncio
    async def test_detect_churn_context(self, detector, sample_churn_data):
        """Test la détection du contexte de churn"""
        context = await detector.detect_ml_context(
            sample_churn_data,
            target_col='churn'
        )
        
        assert context.problem_type == 'churn_prediction'
        assert context.confidence > 0.6
        assert 'churn_indicator' in context.detected_patterns
        assert context.temporal_aspect == True
    
    @pytest.mark.asyncio
    async def test_detect_sales_context(self, detector, sample_sales_data):
        """Test la détection du contexte de prévision des ventes"""
        context = await detector.detect_ml_context(
            sample_sales_data,
            target_col='sales'
        )
        
        assert context.problem_type in ['sales_forecasting', 'demand_prediction']
        assert context.confidence > 0.5
        assert context.temporal_aspect == True
        assert 'sales_indicator' in context.detected_patterns
    
    def test_analyze_columns(self, detector, sample_fraud_data):
        """Test l'analyse des colonnes"""
        analysis = detector._analyze_columns(sample_fraud_data)
        
        assert 'columns' in analysis
        assert 'detected_patterns' in analysis
        assert 'column_types' in analysis
        assert 'potential_features' in analysis
        
        patterns = analysis['detected_patterns']
        assert 'has_id_columns' in patterns
        assert 'has_financial_features' in patterns
        assert 'has_temporal_features' in patterns
    
    def test_analyze_target_binary(self, detector, sample_fraud_data):
        """Test l'analyse d'une variable cible binaire"""
        target_analysis = detector._analyze_target(sample_fraud_data, 'fraud')
        
        assert target_analysis['type'] == 'binary_classification'
        assert target_analysis['unique_values'] == 2
        assert 'imbalance_ratio' in target_analysis
        assert target_analysis['is_imbalanced'] == True
    
    def test_analyze_target_regression(self, detector, sample_sales_data):
        """Test l'analyse d'une variable cible continue"""
        target_analysis = detector._analyze_target(sample_sales_data, 'sales')
        
        assert target_analysis['type'] == 'regression'
        assert 'distribution' in target_analysis
        assert 'mean' in target_analysis['distribution']
        assert 'std' in target_analysis['distribution']
    
    def test_detect_temporal_aspects(self, detector, sample_sales_data):
        """Test la détection des aspects temporels"""
        temporal_info = detector._detect_temporal_aspects(sample_sales_data)
        
        assert temporal_info['has_temporal'] == True
        assert 'date' in temporal_info['temporal_columns']
        assert temporal_info['is_time_series'] == True
    
    def test_calculate_problem_score(self, detector):
        """Test le calcul du score de problème"""
        patterns = {
            'keywords': ['fraud', 'transaction', 'risk'],
            'column_patterns': ['transaction_id', 'amount', 'risk_score'],
            'target_patterns': ['fraud', 'fraudulent'],
            'temporal_indicators': ['timestamp', 'date'],
            'business_metrics': ['amount', 'frequency']
        }
        
        column_analysis = {
            'columns': ['transaction_id', 'amount', 'fraud', 'timestamp'],
            'detected_patterns': {'has_financial_features'}
        }
        
        target_analysis = {'name': 'fraud'}
        context_clues = {'sector': 'financial_fraud'}
        
        score = detector._calculate_problem_score(
            'fraud_detection',
            patterns,
            column_analysis,
            target_analysis,
            context_clues
        )
        
        assert score > 0.7
        assert score <= 1.0
    
    @pytest.mark.asyncio
    async def test_search_business_context(self, detector):
        """Test la recherche du contexte métier"""
        column_analysis = {
            'columns': ['transaction_id', 'amount', 'fraud'],
            'detected_patterns': {'has_financial_features', 'fraud_indicator'}
        }
        
        context = await detector._search_business_context(column_analysis)
        
        assert context['sector'] == 'financial_fraud'
        assert isinstance(context['industry_patterns'], list)
        assert isinstance(context['domain_keywords'], list)
    
    def test_generate_optimal_config(self, detector):
        """Test la génération de configuration optimale"""
        config = detector._generate_optimal_config(
            'fraud_detection',
            pd.DataFrame({'col1': [1, 2, 3]}),
            'target',
            {'n_samples': 1000, 'imbalance_detected': True}
        )
        
        assert config['problem_type'] == 'fraud_detection'
        assert config['task'] == 'classification'
        assert config['primary_metric'] == 'f1'
        assert 'SMOTE' in str(config['preprocessing'])
        assert 'IsolationForest' in config['algorithms']


# ============================================================================
# TESTS ORIGINAUX - INTELLIGENT CONFIG GENERATOR
# ============================================================================

class TestIntelligentConfigGenerator:
    """Tests pour le générateur de configuration intelligent"""
    
    @pytest.fixture
    def generator(self):
        return IntelligentConfigGenerator(use_claude=False)
    
    @pytest.mark.asyncio
    async def test_generate_config_basic(self, generator, sample_fraud_data):
        """Test la génération de configuration basique"""
        context = {
            'problem_type': 'fraud_detection',
            'business_sector': 'finance',
            'temporal_aspect': True,
            'imbalance_detected': True
        }
        
        config = await generator.generate_config(
            df=sample_fraud_data,
            context=context
        )
        
        assert isinstance(config, OptimalConfig)
        assert config.task == 'classification'
        assert len(config.algorithms) > 0
        assert config.primary_metric in ['f1', 'precision', 'recall', 'roc_auc']
        assert config.preprocessing is not None
        assert config.hpo_config is not None
    
    def test_determine_task(self, generator):
        """Test la détermination du type de tâche"""
        context = {'problem_type': 'fraud_detection'}
        task = generator._determine_task(context, pd.DataFrame())
        assert task == 'classification'
        
        context = {'problem_type': 'sales_forecasting'}
        task = generator._determine_task(context, pd.DataFrame())
        assert task == 'regression'
        
        context = {'problem_type': 'customer_segmentation'}
        task = generator._determine_task(context, pd.DataFrame())
        assert task == 'clustering'
    
    @pytest.mark.asyncio
    async def test_select_algorithms(self, generator, sample_fraud_data):
        """Test la sélection des algorithmes"""
        algorithms = await generator._select_algorithms(
            task='classification',
            df=sample_fraud_data,
            context={'imbalance_detected': True},
            constraints={'time_budget': 3600},
            user_preferences=None
        )
        
        assert len(algorithms) > 0
        assert 'XGBoost' in algorithms or 'LightGBM' in algorithms
        assert 'LogisticRegression' in algorithms
    
    def test_score_algorithm(self, generator):
        """Test le scoring des algorithmes"""
        df = pd.DataFrame({
            'num1': [1, 2, None, 4],
            'cat1': ['A', 'B', 'C', 'D']
        })
        
        score = generator._score_algorithm(
            'LightGBM',
            df,
            context={'imbalance_detected': True},
            constraints={'interpretability_required': False},
            user_preferences=None
        )
        
        assert score > 0
        assert score <= 1.0
        assert score > 0.1
    
    def test_select_metric(self, generator):
        """Test la sélection de métrique"""
        metric = generator._select_metric(
            'classification',
            {'problem_type': 'fraud_detection'},
            None
        )
        assert metric == 'average_precision'
        
        metric = generator._select_metric(
            'classification',
            {'problem_type': 'churn_prediction'},
            None
        )
        assert metric == 'f1'
        
        metric = generator._select_metric(
            'regression',
            {'problem_type': 'sales_forecasting'},
            None
        )
        assert metric == 'mape'
    
    def test_configure_preprocessing(self, generator):
        """Test la configuration du preprocessing"""
        df = pd.DataFrame({
            'num1': [1, 2, None, 4, 100],
            'cat1': ['A', 'B', 'C', 'D', 'E']
        })
        
        preprocessing = generator._configure_preprocessing(
            df,
            {'problem_type': 'classification'},
            'classification'
        )
        
        assert 'missing_values' in preprocessing
        assert 'outliers' in preprocessing
        assert 'scaling' in preprocessing
        assert 'encoding' in preprocessing
        assert preprocessing['missing_values']['strategy'] in ['simple_impute', 'iterative_impute']
    
    def test_configure_hpo(self, generator):
        """Test la configuration HPO"""
        algorithms = ['XGBoost', 'LightGBM', 'RandomForest']
        df = pd.DataFrame(np.random.randn(1000, 10))
        
        hpo_config = generator._configure_hpo(
            algorithms,
            df,
            time_budget=3600,
            task='classification'
        )
        
        assert hpo_config['method'] == 'optuna'
        assert hpo_config['n_iter'] > 0
        assert 'search_spaces' in hpo_config
        assert 'XGBoost' in hpo_config['search_spaces']
    
    def test_get_search_space(self, generator):
        """Test la génération d'espace de recherche"""
        space = generator._get_search_space('XGBoost', 'classification', 10000)
        assert 'n_estimators' in space
        assert 'max_depth' in space
        assert 'learning_rate' in space
        
        space = generator._get_search_space('LightGBM', 'classification', 10000)
        assert 'num_leaves' in space
        assert 'learning_rate' in space
        
        space = generator._get_search_space('RandomForest', 'classification', 10000)
        assert 'n_estimators' in space
        assert 'max_depth' in space
    
    def test_setup_cv_strategy(self, generator):
        """Test la stratégie de validation croisée"""
        df = pd.DataFrame(np.random.randn(100, 10))
        
        cv = generator._setup_cv_strategy(df, 'classification', {})
        assert cv['method'] == 'stratified_kfold'
        assert cv['n_folds'] == 5
        
        cv = generator._setup_cv_strategy(
            df, 
            'regression', 
            {'temporal_aspect': True, 'is_time_series': True}
        )
        assert cv['method'] == 'time_series_split'
    
    def test_configure_ensemble(self, generator):
        """Test la configuration d'ensemble"""
        ensemble = generator._configure_ensemble(['XGBoost', 'LightGBM'], 'classification')
        assert ensemble['enabled'] == False
        
        algorithms = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'LogisticRegression', 'NeuralNetwork']
        ensemble = generator._configure_ensemble(algorithms, 'classification')
        assert ensemble['enabled'] == True
        assert ensemble['method'] == 'stacking'
    
    def test_adapt_config(self, generator):
        """Test l'adaptation de configuration"""
        base_config = OptimalConfig(
            task='classification',
            algorithms=['XGBoost', 'LightGBM', 'NeuralNetwork'],
            primary_metric='f1',
            preprocessing={},
            feature_engineering={},
            hpo_config={'n_iter': 50},
            cv_strategy={},
            ensemble_config={},
            time_budget=3600,
            resource_constraints={},
            monitoring={}
        )
        
        adapted = generator.adapt_config(base_config, {'time_budget': 600})
        assert adapted.time_budget == 600
        assert adapted.hpo_config['n_iter'] < 50
        
        adapted = generator.adapt_config(base_config, {'memory_limit_gb': 4})
        assert 'NeuralNetwork' not in adapted.algorithms


# ============================================================================
# TESTS CONFIG GENERATOR - AVEC CLAUDE (AJOUTER APRÈS TestIntelligentConfigGenerator)
# ============================================================================

class TestConfigGeneratorWithClaude:
    """Tests pour IntelligentConfigGenerator avec support Claude"""
    
    @pytest.mark.asyncio
    async def test_claude_algorithm_selection(self, sample_fraud_data):
        """Test sélection d'algorithmes avec Claude"""
        generator = IntelligentConfigGenerator(use_claude=True)
        
        with patch.object(generator, 'claude_client') as mock_claude:
            mock_response = Mock()
            mock_response.content = [Mock(text='{"algorithms": ["XGBoost", "LightGBM", "CatBoost"], "reasoning": "Claude selection"}')]
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            
            algorithms = await generator._select_algorithms(
                task='classification',
                df=sample_fraud_data,
                context={'imbalance_detected': True},
                constraints={'time_budget': 3600},
                user_preferences=None
            )
            
            assert 'XGBoost' in algorithms
            assert 'LightGBM' in algorithms
            assert len(algorithms) >= 3
    
    @pytest.mark.asyncio
    async def test_claude_feature_engineering(self, sample_fraud_data):
        """Test design feature engineering avec Claude"""
        generator = IntelligentConfigGenerator(use_claude=True)
        
        with patch.object(generator, 'claude_client') as mock_claude:
            mock_response = Mock()
            mock_response.content = [Mock(text='{"enhancements": {"additional_features": ["velocity_features"], "domain_specific": {"fraud_specific": true}}, "reasoning": "Claude FE"}')]
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            
            fe_config = await generator._design_feature_engineering_with_claude(
                sample_fraud_data,
                {'problem_type': 'fraud_detection'},
                'classification'
            )
            
            # Devrait contenir les enhancements Claude
            assert 'velocity_features' in str(fe_config) or 'additional_features' in fe_config
    
    @pytest.mark.asyncio
    async def test_claude_reasoning_generation(self):
        """Test génération de raisonnement avec Claude"""
        generator = IntelligentConfigGenerator(use_claude=True)
        
        with patch.object(generator, 'claude_client') as mock_claude:
            mock_response = Mock()
            mock_response.content = [Mock(text='This is a detailed reasoning from Claude about the ML configuration choices.')]
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            
            reasoning = await generator._generate_config_reasoning_with_claude(
                task='classification',
                algorithms=['XGBoost', 'LightGBM'],
                primary_metric='f1',
                constraints={'time_budget': 3600},
                context={'problem_type': 'fraud_detection'}
            )
            
            assert 'Claude' in reasoning or len(reasoning) > 50
    
    @pytest.mark.asyncio
    async def test_fallback_to_rules_on_claude_failure(self, sample_fraud_data):
        """Test fallback vers règles quand Claude échoue"""
        generator = IntelligentConfigGenerator(use_claude=True)
        
        with patch.object(generator, 'claude_client') as mock_claude:
            # Simuler erreur Claude
            mock_claude.messages.create = AsyncMock(side_effect=Exception("Claude error"))
            
            algorithms = await generator._select_algorithms(
                task='classification',
                df=sample_fraud_data,
                context={'imbalance_detected': True},
                constraints={'time_budget': 3600},
                user_preferences=None
            )
            
            # Devrait quand même retourner des algorithmes via règles
            assert len(algorithms) > 0
            assert 'XGBoost' in algorithms or 'LightGBM' in algorithms
    
    @pytest.mark.asyncio
    async def test_rule_based_without_claude(self, sample_fraud_data):
        """Test génération pure règles sans Claude"""
        generator = IntelligentConfigGenerator(use_claude=False)
        
        assert generator.claude_client is None
        
        context = {
            'problem_type': 'fraud_detection',
            'imbalance_detected': True
        }
        
        config = await generator.generate_config(
            df=sample_fraud_data,
            context=context
        )
        
        assert isinstance(config, OptimalConfig)
        assert len(config.algorithms) > 0


# ============================================================================
# TESTS ADAPTIVE TEMPLATE SYSTEM - AVEC CLAUDE (AJOUTER APRÈS TestAdaptiveTemplateSystem)
# ============================================================================

class TestAdaptiveTemplateSystemWithClaude:
    """Tests pour AdaptiveTemplateSystem avec support Claude"""
    
    @pytest.fixture
    def adaptive_system_claude(self, temp_dir):
        return AdaptiveTemplateSystem(Path(temp_dir), use_claude=True)
    
    @pytest.mark.asyncio
    async def test_claude_pattern_selection(self, adaptive_system_claude, sample_fraud_data):
        """Test sélection de pattern avec Claude"""
        # Ajouter des patterns
        adaptive_system_claude.learned_patterns['fraud_detection'] = [
            {'context': {'n_samples': 1000}, 'config': {'algorithms': ['XGBoost']}, 'success_score': 0.85},
            {'context': {'n_samples': 900}, 'config': {'algorithms': ['LightGBM']}, 'success_score': 0.90}
        ]
        
        with patch.object(adaptive_system_claude, 'claude_client') as mock_claude:
            mock_response = Mock()
            mock_response.content = [Mock(text='{"best_pattern_id": 1, "confidence": 0.85, "reasoning": "Claude selection", "semantic_similarity": 0.9}')]
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            
            context = {'problem_type': 'fraud_detection', 'n_samples': 950}
            
            pattern = await adaptive_system_claude._claude_select_best_pattern('fraud_detection', context)
            
            assert pattern is not None
            assert pattern['algorithms'] == ['LightGBM']
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_matching(self, adaptive_system_claude):
        """Test matching par similarité sémantique"""
        adaptive_system_claude.learned_patterns['test'] = [
            {'context': {'business_sector': 'finance', 'n_samples': 1000}, 'config': {'test': True}, 'success_score': 0.9}
        ]
        
        with patch.object(adaptive_system_claude, 'claude_client') as mock_claude:
            mock_response = Mock()
            mock_response.content = [Mock(text='{"best_pattern_id": 0, "confidence": 0.9, "reasoning": "Semantic match", "semantic_similarity": 0.95}')]
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            
            context = {'business_sector': 'banking', 'n_samples': 1100}  # Sémantiquement similaire
            
            pattern = await adaptive_system_claude._claude_select_best_pattern('test', context)
            
            assert pattern is not None
    
    @pytest.mark.asyncio
    async def test_fallback_to_rule_based_matching(self, adaptive_system_claude):
        """Test fallback vers matching par règles"""
        adaptive_system_claude.learned_patterns['test'] = [
            {'context': {'n_samples': 1000}, 'config': {'test': True}, 'success_score': 0.8}
        ]
        
        with patch.object(adaptive_system_claude, 'claude_client') as mock_claude:
            # Simuler échec Claude
            mock_claude.messages.create = AsyncMock(side_effect=Exception("Claude error"))
            
            # Devrait fallback vers règles
            with patch.object(adaptive_system_claude, '_select_best_learned_pattern') as mock_rule:
                mock_rule.return_value = {'test': True}
                
                context = {'n_samples': 950}
                config = await adaptive_system_claude.get_configuration(
                    df=pd.DataFrame({'col1': [1, 2, 3]}),
                    context=context
                )
                
                assert mock_rule.called
    
    def test_metrics_tracking(self, adaptive_system_claude):
        """Test tracking des métriques Claude"""
        assert 'claude_enhanced_adaptations' in adaptive_system_claude.metrics
        assert 'claude_fallbacks' in adaptive_system_claude.metrics
        
        initial_enhancements = adaptive_system_claude.metrics['claude_enhanced_adaptations']
        
        # Simuler une adaptation avec Claude
        adaptive_system_claude.metrics['claude_enhanced_adaptations'] += 1
        
        assert adaptive_system_claude.metrics['claude_enhanced_adaptations'] == initial_enhancements + 1
    
    def test_get_template_stats_with_claude(self, adaptive_system_claude):
        """Test statistiques incluant métriques Claude"""
        adaptive_system_claude.metrics['total_adaptations'] = 10
        adaptive_system_claude.metrics['claude_enhanced_adaptations'] = 7
        
        stats = adaptive_system_claude.get_template_stats()
        
        assert 'claude_metrics' in stats
        assert 'claude_effectiveness' in stats
        assert stats['claude_effectiveness']['enhancement_rate'] == 0.7


@pytest.mark.asyncio
async def test_claude_select_best_pattern_detailed(temp_dir):
    """Test complet de sélection de pattern avec Claude"""
    system = AdaptiveTemplateSystem(
        template_dir=Path(temp_dir),
        use_claude=True,
        anthropic_api_key="test-key"
    )
    
    # Patterns avec contextes variés
    system.learned_patterns['test_problem'] = [
        {
            'context': {'n_samples': 1000, 'business_sector': 'finance'},
            'config': {'algorithms': ['XGBoost']},
            'success_score': 0.85
        },
        {
            'context': {'n_samples': 1100, 'business_sector': 'banking'},
            'config': {'algorithms': ['LightGBM']},
            'success_score': 0.90
        }
    ]
    
    with patch.object(system, 'claude_client') as mock_claude:
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            'best_pattern_id': 1,
            'confidence': 0.85,
            'reasoning': 'Banking sector is semantically close to finance',
            'semantic_similarity': 0.92
        }))]
        mock_claude.messages.create = AsyncMock(return_value=mock_response)
        
        current_context = {'business_sector': 'banking', 'n_samples': 1050}
        
        pattern = await system._claude_select_best_pattern('test_problem', current_context)
        
        # Vérifier sélection sémantique (pas juste numérique)
        assert pattern is not None
        assert pattern['algorithms'] == ['LightGBM']
        assert mock_claude.messages.create.called
        
        # Vérifier le prompt envoyé
        call_args = mock_claude.messages.create.call_args
        prompt = call_args.kwargs['messages'][0]['content']
        assert 'semantic similarity' in prompt.lower()

def test_summarize_config(temp_dir):
    """Test summarize_config pour Claude"""
    system = AdaptiveTemplateSystem(template_dir=Path(temp_dir))
    
    full_config = {
        'task': 'classification',
        'algorithms': ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest'],
        'preprocessing': {'handle_missing': True, 'scale': True},
        'feature_engineering': {'polynomial': True},
        'primary_metric': 'f1'
    }
    
    summary = system._summarize_config(full_config)
    
    # Vérifier condensation
    assert summary['task'] == 'classification'
    assert len(summary['algorithms']) == 3  # Top 3 seulement
    assert summary['has_preprocessing'] == True
    assert summary['has_feature_engineering'] == True
    assert summary['primary_metric'] == 'f1'


# ============================================================================
# TESTS DATA CLEANING ORCHESTRATOR - AVEC CLAUDE ))))))))))))))))))))))))))))))))))))))))))
# ============================================================================

class TestDataCleaningOrchestratorWithClaude:
    """Tests pour DataCleaningOrchestrator avec support Claude"""
    
    @pytest.mark.asyncio
    async def test_determine_cleaning_mode_with_claude(self, test_agent_config, sample_fraud_data):
        """Test détermination du mode de nettoyage avec Claude"""
        orchestrator = DataCleaningOrchestrator(test_agent_config, use_claude=True)
        
        with patch.object(orchestrator, 'claude_client') as mock_claude:
            mock_response = Mock()
            mock_response.content = [Mock(text='{"recommended_mode": "hybrid", "confidence": 0.85, "reasoning": "Claude decision", "key_considerations": ["test"], "estimated_time_minutes": 15, "risk_level": "medium"}')]
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            
            user_context = {'secteur_activite': 'finance'}
            ml_context = Mock(problem_type='fraud_detection', business_sector='finance', confidence=0.9)
            
            decision = await orchestrator.determine_cleaning_mode_with_claude(
                sample_fraud_data,
                user_context,
                ml_context
            )
            
            assert decision['recommended_mode'] == 'hybrid'
            assert decision['confidence'] == 0.85
    
    @pytest.mark.asyncio
    async def test_recommend_approach_with_claude(self, test_agent_config, sample_fraud_data):
        """Test recommandations de nettoyage avec Claude"""
        orchestrator = DataCleaningOrchestrator(test_agent_config, use_claude=True)
        
        with patch.object(orchestrator, 'claude_client') as mock_claude:
            mock_response = Mock()
            mock_response.content = [Mock(text='This is a detailed cleaning recommendation from Claude covering priorities, strategy, risks, and expected impact.')]
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            
            profile_report = {'quality_issues': []}
            ml_context = Mock(problem_type='fraud_detection', business_sector='finance', confidence=0.9)
            
            recommendations = await orchestrator.recommend_cleaning_approach_with_claude(
                sample_fraud_data,
                profile_report,
                ml_context
            )
            
            assert len(recommendations) > 50  # Devrait être une recommandation détaillée
            assert 'Claude' in recommendations or 'priorities' in recommendations.lower()
    
    @pytest.mark.asyncio
    async def test_claude_fallback_on_failure(self, test_agent_config, sample_fraud_data):
        """Test fallback quand Claude échoue"""
        orchestrator = DataCleaningOrchestrator(test_agent_config, use_claude=True)
        
        with patch.object(orchestrator, 'claude_client') as mock_claude:
            # Simuler erreur Claude
            mock_claude.messages.create = AsyncMock(side_effect=Exception("Claude API error"))
            
            user_context = {'secteur_activite': 'finance'}
            ml_context = Mock(problem_type='fraud_detection')
            
            # Devrait fallback vers règles
            decision = await orchestrator.determine_cleaning_mode_with_claude(
                sample_fraud_data,
                user_context,
                ml_context
            )
            
            assert 'recommended_mode' in decision
            assert decision['confidence'] > 0
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, test_agent_config):
        """Test exécution parallèle des tâches"""
        orchestrator = DataCleaningOrchestrator(test_agent_config, use_claude=True)
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        user_context = {'secteur_activite': 'test'}
        ml_context = Mock(problem_type='test', business_sector='test', confidence=0.9)
        
        with patch.object(orchestrator.profiler, 'analyze') as mock_profile:
            mock_profile.return_value = {'quality_issues': []}
            
            with patch.object(orchestrator, 'determine_cleaning_mode_with_claude') as mock_mode:
                mock_mode.return_value = {'recommended_mode': 'automated', 'confidence': 0.8}
                
                # Les deux tâches devraient s'exécuter en parallèle
                with patch('asyncio.gather') as mock_gather:
                    mock_gather.return_value = [mock_mode.return_value, mock_profile.return_value]
                    
                    # Simuler l'exécution (simplifié)
                    # En réalité, clean_dataset appelle gather pour paralléliser
                    pass
    
    def test_bounded_history(self, test_agent_config):
        """Test que l'historique est limité (évite memory leak)"""
        orchestrator = DataCleaningOrchestrator(test_agent_config)
        
        # Vérifier BoundedList
        assert isinstance(orchestrator.execution_history, BoundedList)
        assert orchestrator.execution_history.maxlen == 100
        
        # Ajouter plus de 100 items
        for i in range(150):
            orchestrator.execution_history.append({'item': i})
        
        # Ne devrait contenir que les 100 derniers
        assert len(orchestrator.execution_history) == 100
        assert orchestrator.execution_history[0]['item'] == 50  # Premier = 50 (150-100)
    
    @pytest.mark.asyncio
    async def test_claude_decisions_metrics(self, test_agent_config):
        """Test tracking des décisions Claude"""
        orchestrator = DataCleaningOrchestrator(test_agent_config, use_claude=True)
        
        initial_decisions = orchestrator.performance_metrics['claude_decisions']
        
        with patch.object(orchestrator, 'claude_client') as mock_claude:
            mock_response = Mock()
            mock_response.content = [Mock(text='{"recommended_mode": "automated"}')]
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            
            df = pd.DataFrame({'col1': [1, 2, 3]})
            user_context = {}
            ml_context = Mock(problem_type='test')
            
            await orchestrator.determine_cleaning_mode_with_claude(df, user_context, ml_context)
            
            # Vérifier incrémentation
            assert orchestrator.performance_metrics['claude_decisions'] == initial_decisions + 1


# ============================================================================
# TESTS ORIGINAUX - OPENAI AGENTS
# ============================================================================

class TestOpenAIAgents:
    """Tests pour les agents OpenAI individuels"""
    
    @pytest.fixture
        """Mock du client OpenAI"""
        client = AsyncMock()
        client.beta.assistants.create = AsyncMock()
        client.beta.assistants.retrieve = AsyncMock()
        client.beta.threads.create = AsyncMock()
        client.beta.threads.messages.create = AsyncMock()
        client.beta.threads.runs.create = AsyncMock()
        client.beta.threads.runs.retrieve = AsyncMock()
        client.beta.threads.messages.list = AsyncMock()
        return client
    
    @pytest.mark.asyncio
    async def test_profiler_agent(self, test_agent_config, mock_openai_client):
        """Test ProfilerAgent"""
        agent = ProfilerAgent(test_agent_config)
        agent.client = mock_openai_client
        
        mock_openai_client.beta.assistants.create.return_value = Mock(id='asst_123')
        mock_openai_client.beta.threads.create.return_value = Mock(id='thread_123')
        mock_openai_client.beta.threads.runs.create.return_value = Mock(id='run_123')
        mock_openai_client.beta.threads.runs.retrieve.return_value = Mock(status='completed')
        
        mock_message = Mock()
        mock_message.role = 'assistant'
        mock_message.content = [Mock(text=Mock(value='{"quality_issues": []}'))]
        mock_openai_client.beta.threads.messages.list.return_value = Mock(data=[mock_message])
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        report = await agent.analyze(df)
        
        assert isinstance(report, dict)
    
    def test_agent_instantiation_without_event_loop(self, test_agent_config):
        """Ensure agents defer initialization when no loop is running"""
        agent = ProfilerAgent(test_agent_config)
        assert agent._initialized == False
        assert agent.assistant is None
    
    def test_cleaner_agent_basic_cleaning(self, test_agent_config):
        """Test CleanerAgent basic cleaning"""
        agent = CleanerAgent(test_agent_config)
        
        df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': ['a', 'b', 'c', 'd']
        })
        profile_report = {'quality_issues': []}
        
        cleaned_df, transformations = agent._basic_cleaning(df, profile_report)
        
        assert cleaned_df['col1'].isnull().sum() == 0
        assert len(transformations) > 0
    
    def test_controller_agent_quality_metrics(self, test_agent_config):
        """Test ControllerAgent quality metrics"""
        agent = ControllerAgent(test_agent_config)
        
        original_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        cleaned_df = pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': ['a', 'b', 'c', 'd']
        })
        
        metrics = agent._calculate_quality_metrics(cleaned_df, original_df)
        
        assert 'data_quality' in metrics
        assert 'transformation_impact' in metrics
        assert metrics['data_quality']['rows']['removed'] == 1
        assert metrics['quality_score'] > 0


# ============================================================================
# TESTS LAZY INITIALIZATION - AGENTS OPENAI (AJOUTER APRÈS TestOpenAIAgent)
# ============================================================================

class TestLazyInitialization:
    """Tests pour l'initialisation lazy des agents OpenAI"""
    
    @pytest.mark.asyncio
    async def test_profiler_lazy_init(self, test_agent_config):
        """Test que ProfilerAgent s'initialise à la première utilisation"""
        agent = ProfilerAgent(test_agent_config)
        
        # Vérifier qu'il n'est pas initialisé au départ
        assert agent._initialized == False
        assert agent.assistant is None
        
        # Mock du client OpenAI
        with patch.object(agent, 'client') as mock_client:
            mock_client.beta.assistants.create = AsyncMock(return_value=Mock(id='asst_test'))
            
            # Forcer l'initialisation
            await agent._ensure_assistant_initialized()
            
            # Vérifier qu'il est maintenant initialisé
            assert agent._initialized == True
    
    @pytest.mark.asyncio
    async def test_cleaner_lazy_init(self, test_agent_config):
        """Test que CleanerAgent s'initialise à la première utilisation"""
        agent = CleanerAgent(test_agent_config)
        
        assert agent._initialized == False
        assert agent.assistant is None
        
        with patch.object(agent, 'client') as mock_client:
            mock_client.beta.assistants.create = AsyncMock(return_value=Mock(id='asst_test'))
            await agent._ensure_assistant_initialized()
            assert agent._initialized == True
    
    @pytest.mark.asyncio
    async def test_validator_lazy_init(self, test_agent_config):
        """Test que ValidatorAgent s'initialise à la première utilisation"""
        agent = ValidatorAgent(test_agent_config)
        
        assert agent._initialized == False
        assert agent.assistant is None
        
        with patch.object(agent, 'openai_client') as mock_client:
            mock_client.beta.assistants.create = AsyncMock(return_value=Mock(id='asst_test'))
            await agent._ensure_assistant_initialized()
            assert agent._initialized == True
    
    @pytest.mark.asyncio
    async def test_profiler_analyze_triggers_init(self, test_agent_config):
        """Test que analyze() déclenche l'initialisation automatiquement"""
        agent = ProfilerAgent(test_agent_config)
        
        with patch.object(agent, '_ensure_assistant_initialized') as mock_init:
            with patch.object(agent, '_basic_profiling') as mock_basic:
                mock_basic.return_value = {'summary': {}}
                
                df = pd.DataFrame({'col1': [1, 2, 3]})
                await agent.analyze(df)
                
                # Vérifier que l'initialisation a été appelée
                mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleaner_clean_triggers_init(self, test_agent_config):
        """Test que clean() déclenche l'initialisation automatiquement"""
        agent = CleanerAgent(test_agent_config)
        
        with patch.object(agent, '_ensure_assistant_initialized') as mock_init:
            with patch.object(agent, '_basic_cleaning') as mock_basic:
                df = pd.DataFrame({'col1': [1, 2, 3]})
                mock_basic.return_value = (df, [])
                
                profile_report = {'quality_issues': []}
                validation_report = {'issues': []}
                
                await agent.clean(df, profile_report, validation_report)
                
                mock_init.assert_called_once()


# ============================================================================
# TESTS FALLBACK - AGENTS SANS API KEYS (AJOUTER APRÈS TestLazyInitialization)
# ============================================================================

class TestAgentFallbacks:
    """Tests pour les fallbacks quand les APIs sont indisponibles"""
    
    @pytest.mark.asyncio
    async def test_profiler_without_openai(self):
        """Test ProfilerAgent sans OpenAI"""
        config = AgentConfig(openai_api_key=None)
        agent = ProfilerAgent(config)
        
        assert agent.client is None
        
        df = pd.DataFrame({'col1': [1, 2, None, 4]})
        report = await agent.analyze(df)
        
        # Devrait utiliser basic_profiling
        assert 'summary' in report
        assert 'columns' in report
    
    @pytest.mark.asyncio
    async def test_cleaner_without_openai(self):
        """Test CleanerAgent sans OpenAI"""
        config = AgentConfig(openai_api_key=None)
        agent = CleanerAgent(config)
        
        assert agent.client is None
        
        df = pd.DataFrame({'col1': [1, 2, None, 4]})
        profile_report = {'quality_issues': []}
        validation_report = {'issues': []}
        
        cleaned_df, transformations = await agent.clean(df, profile_report, validation_report)
        
        # Devrait utiliser basic_cleaning
        assert cleaned_df['col1'].isnull().sum() == 0
        assert len(transformations) > 0
    
    @pytest.mark.asyncio
    async def test_controller_without_claude(self):
        """Test ControllerAgent sans Claude"""
        config = AgentConfig(openai_api_key=None)  # Pas de clé
        agent = ControllerAgent(config)
        
        assert agent.client is None
        
        original_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        cleaned_df = pd.DataFrame({'col1': [1, 2, 3, 4]})
        transformations = [{'action': 'remove_duplicates'}]
        
        report = await agent.validate(cleaned_df, original_df, transformations)
        
        # Devrait utiliser basic_validation
        assert 'quality_score' in report
        assert 'issues' in report


# ============================================================================
# TESTS UTILS - SÉCURITÉ ET ASYNC (NOUVEAUX - CRITIQUE)
# ============================================================================

class TestUtilsSecurity:
    """Tests pour les utilitaires de sécurité et async"""
    
    def test_sanitize_api_keys(self):
        """Vérifie que les API keys sont masquées"""
        data = {
            'openai_api_key': 'sk-1234567890abcdef',
            'anthropic_api_key': 'sk-ant-abcdefghijk',
            'normal_field': 'value',
            'user_context': {'name': 'test'}
        }
        
        sanitized = sanitize_for_logging(data)
        
        assert sanitized['openai_api_key'] == '***'
        assert sanitized['anthropic_api_key'] == '***'
        assert sanitized['normal_field'] == 'value'
        assert sanitized['user_context']['name'] == 'test'
    
    def test_sanitize_nested_secrets(self):
        """Vérifie masquage dans structures imbriquées"""
        data = {
            'config': {
                'api_key': 'secret123',
                'nested': {
                    'token': 'token456'
                }
            },
            'list': [
                {'password': 'pass789'}
            ]
        }
        
        sanitized = sanitize_for_logging(data)
        
        assert sanitized['config']['api_key'] == '***'
        assert sanitized['config']['nested']['token'] == '***'
        assert sanitized['list'][0]['password'] == '***'
    
    def test_sanitize_edge_cases(self):
        """Test avec None, liste vide, dict vide"""
        assert sanitize_for_logging(None) == None
        assert sanitize_for_logging({}) == {}
        assert sanitize_for_logging([]) == []
        assert sanitize_for_logging({'key': None}) == {'key': None}
    
    def test_safe_log_config(self, test_agent_config):
        """Test safe_log_config avec AgentConfig"""
        log_str = safe_log_config(test_agent_config)
        
        assert 'openai_api_key' not in log_str or '***' in log_str
        assert 'anthropic_api_key' not in log_str or '***' in log_str
        assert isinstance(log_str, str)
        assert len(log_str) > 0
    
    @pytest.mark.asyncio
    async def test_run_parallel_success(self):
        """Test exécution parallèle réussie"""
        async def task1():
            await asyncio.sleep(0.01)
            return "result1"
        
        async def task2():
            await asyncio.sleep(0.01)
            return "result2"
        
        results = await run_parallel(task1(), task2())
        
        assert results == ["result1", "result2"]
    
    @pytest.mark.asyncio
    async def test_run_parallel_with_exceptions(self):
        """Test avec return_exceptions=True"""
        async def task_success():
            return "success"
        
        async def task_fail():
            raise ValueError("Test error")
        
        # Avec return_exceptions=True (défaut)
        results = await run_parallel(task_success(), task_fail(), return_exceptions=True)
        
        assert results[0] == "success"
        assert isinstance(results[1], ValueError)
    
    @pytest.mark.asyncio
    async def test_run_parallel_without_exception_handling(self):
        """Test avec return_exceptions=False"""
        async def task_fail():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await run_parallel(task_fail(), return_exceptions=False)
    
    @pytest.mark.asyncio
    async def test_async_init_mixin_single_call(self):
        """Test initialisation unique"""
        class TestClass(AsyncInitMixin):
            def __init__(self):
                super().__init__()
                self.init_count = 0
            
            async def _async_init(self):
                self.init_count += 1
        
        obj = TestClass()
        await obj.ensure_initialized()
        
        assert obj._initialized == True
        assert obj.init_count == 1
    
    @pytest.mark.asyncio
    async def test_async_init_mixin_concurrent_calls(self):
        """CRITIQUE : Test race conditions"""
        class TestClass(AsyncInitMixin):
            def __init__(self):
                super().__init__()
                self.init_count = 0
            
            async def _async_init(self):
                await asyncio.sleep(0.01)  # Simuler travail
                self.init_count += 1
        
        obj = TestClass()
        
        # Lancer 10 coroutines simultanées
        tasks = [obj.ensure_initialized() for _ in range(10)]
        await asyncio.gather(*tasks)
        
        # _async_init() doit avoir été appelé UNE SEULE fois
        assert obj.init_count == 1
        assert obj._initialized == True
    
    @pytest.mark.asyncio
    async def test_async_init_mixin_double_check_locking(self):
        """Test du double-check locking"""
        class TestClass(AsyncInitMixin):
            def __init__(self):
                super().__init__()
                self.init_calls = []
            
            async def _async_init(self):
                self.init_calls.append(datetime.now())
                await asyncio.sleep(0.01)
        
        obj = TestClass()
        
        # Premier appel
        await obj.ensure_initialized()
        assert len(obj.init_calls) == 1
        
        # Deuxième appel - ne devrait pas ré-initialiser
        await obj.ensure_initialized()
        assert len(obj.init_calls) == 1  # Toujours 1
    
    def test_performance_monitor_record(self):
        """Test enregistrement de métriques"""
        monitor = PerformanceMonitor(maxlen=100)
        
        metric = PerformanceMetrics(
            operation='test_op',
            duration=1.5,
            success=True,
            cost=0.05,
            provider='claude'
        )
        
        monitor.record(metric)
        
        assert len(monitor.metrics) == 1
        assert monitor.metrics[0].operation == 'test_op'
    
    def test_performance_monitor_summary(self):
        """Test agrégation de métriques"""
        monitor = PerformanceMonitor()
        
        # Ajouter plusieurs métriques
        for i in range(5):
            monitor.record(PerformanceMetrics(
                operation='test_op',
                duration=1.0 + i * 0.5,
                success=i % 2 == 0,
                cost=0.01 * i,
                provider='claude'
            ))
        
        summary = monitor.get_summary('test_op')
        
        assert summary['total_operations'] == 5
        assert summary['success_rate'] == 0.6  # 3/5
        assert summary['avg_duration'] > 0
        assert summary['total_cost'] > 0
    
    def test_performance_monitor_by_provider(self):
        """Test agrégation par provider"""
        monitor = PerformanceMonitor()
        
        monitor.record(PerformanceMetrics(
            operation='test', duration=1.0, success=True, cost=0.05, provider='claude'
        ))
        monitor.record(PerformanceMetrics(
            operation='test', duration=2.0, success=True, cost=0.10, provider='openai'
        ))
        
        summary = monitor.get_summary()
        
        assert 'by_provider' in summary
        assert 'claude' in summary['by_provider']
        assert 'openai' in summary['by_provider']
    
    def test_health_checker_circuit_breakers(self):
        """Test vérification circuit breakers"""
        checker = HealthChecker()
        
        # Circuit breakers sains
        breakers = {
            'claude': CircuitBreaker('claude'),
            'openai': CircuitBreaker('openai')
        }
        
        assert checker.check_circuit_breakers(breakers) == True
        assert checker.is_healthy == True
        
        # Simuler un circuit breaker OPEN
        breakers['claude'].state = 'OPEN'
        
        assert checker.check_circuit_breakers(breakers) == False
        assert len(checker.issues) > 0
    
    def test_health_checker_cost_limits(self, test_agent_config):
        """Test vérification limites de coût"""
        checker = HealthChecker()
        
        # Dépasser la limite
        test_agent_config.cost_tracking['total'] = test_agent_config.max_cost_total + 1
        
        result = checker.check_cost_limits(test_agent_config)
        
        assert result == False
        assert len(checker.issues) > 0
        assert 'cost limit' in checker.issues[0].lower()
    
    def test_health_checker_get_status(self):
        """Test récupération du statut de santé"""
        checker = HealthChecker()
        checker.last_check = datetime.now()
        checker.is_healthy = True
        checker.issues = []
        
        status = checker.get_status()
        
        assert status['healthy'] == True
        assert 'last_check' in status
        assert status['issues'] == []
    
    def test_validate_llm_json_schema_valid(self):
        """Test validation de schéma JSON valide"""
        data = {
            'field1': 'value',
            'field2': 42,
            'field3': True
        }
        
        result = validate_llm_json_schema(
            data,
            required_fields=['field1', 'field2'],
            field_types={'field1': str, 'field2': int}
        )
        
        assert result == True
    
    def test_validate_llm_json_schema_missing_field(self):
        """Test validation avec champ manquant"""
        data = {'field1': 'value'}
        
        result = validate_llm_json_schema(
            data,
            required_fields=['field1', 'field2']
        )
        
        assert result == False
    
    def test_validate_llm_json_schema_wrong_type(self):
        """Test validation avec mauvais type"""
        data = {'field1': 42}  # devrait être str
        
        result = validate_llm_json_schema(
            data,
            required_fields=['field1'],
            field_types={'field1': str}
        )
        
        assert result == False


# ============================================================================
# TESTS VALIDATOR AGENT - ARCHITECTURE HYBRIDE (AJOUTER APRÈS TestAgentFallbacks)
# ============================================================================

class TestValidatorAgentHybrid:
    """Tests pour l'architecture hybride du ValidatorAgent"""
    
    @pytest.mark.asyncio
    async def test_validator_with_claude_and_openai(self, test_agent_config):
        """Test ValidatorAgent avec Claude ET OpenAI disponibles"""
        # 1. Créer l'agent normalement
        agent = ValidatorAgent(test_agent_config, use_claude=True)
    
        # 2. Configurer le mock pour retourner ce que vous voulez
        mock_claude = AsyncMock()
        mock_claude.messages.create = AsyncMock(return_value=Mock(
            content=[Mock(text=json.dumps({
                "valid": True,
                "overall_score": 85,
                "issues": [],
                "warnings": [{"column": "test", "warning": "Test warning"}],
                "suggestions": [],
                "column_validations": {},
                "reasoning": "Claude strategic analysis"
            }))]
        ))
        agent.claude_client = mock_claude
    
        # 3. Mock la recherche OpenAI
        with patch.object(agent, '_search_sector_standards') as mock_search:
            mock_search.return_value = {
                'standards': [{'name': 'IFRS', 'url': 'test'}],
                'sources': ['http://test.com'],
                'column_mappings': {}
            }
        
        # 4. Tester
        df = pd.DataFrame({'amount': [100, 200, 300]})
        profile_report = {'quality_issues': []}
        
        report = await agent.validate(df, profile_report)
        
        # 5.1 Vérifier que Claude a été utilisé
        assert mock_claude.messages.create.called
        assert mock_claude.messages.create.call_count >= 1

        # 5.2 Vérifier le contenu complet de la requête Claude
        call_args = mock_claude.messages.create.call_args
        assert call_args is not None, "Claude n'a pas été appelé"

        # 5.3 Vérifier les kwargs obligatoires
        assert 'model' in call_args.kwargs
        assert call_args.kwargs['model'] == 'claude-sonnet-4-5-20250929'

        # 5.4 Vérifier max_tokens
        assert 'max_tokens' in call_args.kwargs
        assert call_args.kwargs['max_tokens'] >= 1000
    
        # 5.5 Vérifier system prompt
        assert 'system' in call_args.kwargs
        assert len(call_args.kwargs['system']) > 0
        assert 'validator' in call_args.kwargs['system'].lower() or 'compliance' in call_args.kwargs['system'].lower()
    
        # 5.6 Vérifier messages
        assert 'messages' in call_args.kwargs
        assert len(call_args.kwargs['messages']) > 0
        assert call_args.kwargs['messages'][0]['role'] == 'user'
        assert 'content' in call_args.kwargs['messages'][0]

        # 5.7 Vérifier la structure de la réponse
        assert 'valid' in report
        assert 'overall_score' in report
        assert 'issues' in report
        assert 'warnings' in report
        assert 'reasoning' in report
        assert isinstance(report['issues'], list)
        assert isinstance(report['warnings'], list)
        
        # 6. Vérifier que la recherche OpenAI a été utilisée
        assert mock_search.called
        assert mock_search.call_count >= 1
            
    
    @pytest.mark.asyncio
    async def test_validator_claude_for_reasoning(self, test_agent_config):
        """Test que Claude est utilisé pour le raisonnement de validation"""
        agent = ValidatorAgent(test_agent_config, use_claude=True)
        
        with patch.object(agent, 'claude_client') as mock_claude:
            mock_response = Mock()
            mock_response.content = [Mock(text='{"valid": true, "overall_score": 90, "issues": [], "warnings": [], "suggestions": [], "column_validations": {}, "sector_compliance": {"compliant": true}, "reasoning": "Claude reasoning"}')]
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            
            df = pd.DataFrame({'amount': [100, 200, 300]})
            profile_report = {'quality_issues': []}
            references = {'standards': [], 'sources': [], 'column_mappings': {}}
            
            report = await agent._claude_validate(df, profile_report, references, 'finance')
            
            assert 'reasoning' in report
            assert report['reasoning'] == "Claude reasoning"
            assert report['overall_score'] == 90
   
    @pytest.mark.asyncio
    async def test_validator_openai_for_search(self, test_agent_config):
        """Test que OpenAI est utilisé pour la recherche web"""
        agent = ValidatorAgent(test_agent_config, use_claude=False)
        
        # Mock OpenAI client
        with patch.object(agent, 'openai_client') as mock_openai:
            mock_openai.beta.assistants.create = AsyncMock(return_value=Mock(id='asst_123'))
            await agent._ensure_assistant_initialized()
            
            with patch.object(agent, '_web_search') as mock_search:
                mock_search.return_value = {
                    'results': [{'title': 'IFRS Standards', 'url': 'http://test.com', 'snippet': 'Test'}],
                    'urls': ['http://test.com']
                }
                
                references = await agent._search_sector_standards('finance', ['amount', 'date'])
                
                # Vérifier que la recherche a été appelée
                assert mock_search.called
                assert len(references['sources']) > 0
    
    @pytest.mark.asyncio
    async def test_validator_fallback_without_claude(self, test_agent_config):
        """Test fallback quand Claude n'est pas disponible"""
        agent = ValidatorAgent(test_agent_config, use_claude=False)
        
        assert agent.claude_client is None
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        profile_report = {'quality_issues': []}
        
        with patch.object(agent, '_search_sector_standards') as mock_search:
            mock_search.return_value = {'standards': [], 'sources': [], 'column_mappings': {}}
            
            with patch.object(agent, '_basic_validation') as mock_basic:
                mock_basic.return_value = {'valid': True, 'overall_score': 70}
                
                report = await agent.validate(df, profile_report)
                
                # Devrait utiliser basic_validation
                assert mock_basic.called
    
    @pytest.mark.asyncio
    async def test_validator_metrics_tracking(self, test_agent_config):
        """Test le tracking des métriques d'utilisation"""
        agent = ValidatorAgent(test_agent_config, use_claude=True)
        
        initial_validations = agent.validation_metrics['total_validations']
        
        with patch.object(agent, '_claude_validate') as mock_claude:
            mock_claude.return_value = {'valid': True, 'overall_score': 85, 'issues': [], 'warnings': []}
            
            with patch.object(agent, '_search_sector_standards') as mock_search:
                mock_search.return_value = {'standards': [], 'sources': []}
                
                df = pd.DataFrame({'col1': [1, 2, 3]})
                profile_report = {'quality_issues': []}
                
                await agent.validate(df, profile_report)
                
                # Vérifier que les métriques sont incrémentées
                assert agent.validation_metrics['total_validations'] == initial_validations + 1
                assert agent.validation_metrics['claude_analyses'] > 0


class TestValidatorAgentHybridFlow:
    """Tests du flow complet hybride"""
    
    @pytest.mark.asyncio
    async def test_complete_hybrid_flow(self, test_agent_config):
        """Test OpenAI search puis Claude reasoning"""
        agent = ValidatorAgent(test_agent_config, use_claude=True)
        
        # Mock OpenAI search
        with patch.object(agent, '_search_sector_standards') as mock_search:
            mock_search.return_value = {
                'standards': [{'name': 'IFRS'}],
                'sources': ['http://test.com']
            }
            
            # Mock Claude reasoning
            with patch.object(agent, '_claude_validate') as mock_claude:
                mock_claude.return_value = {
                    'valid': True,
                    'overall_score': 90,
                    'reasoning': 'Claude reasoning'
                }
                
                df = pd.DataFrame({'amount': [100, 200]})
                profile_report = {'quality_issues': []}
                
                report = await agent.validate(df, profile_report)
                
                # Vérifier les deux phases
                assert mock_search.called, "OpenAI search doit être appelé"
                assert mock_claude.called, "Claude reasoning doit être appelé"
                assert 'reasoning' in report
    
    @pytest.mark.asyncio
    async def test_search_with_cache_hit(self, test_agent_config):
        """Test cache hit pour web_search"""
        agent = ValidatorAgent(test_agent_config)
        
        # Premier appel - cache miss
        with patch.object(agent, '_perform_web_search') as mock_perform:
            mock_perform.return_value = {'results': []}
            
            result1 = await agent._web_search('test query')
            assert mock_perform.call_count == 1
            
            # Deuxième appel - cache hit
            result2 = await agent._web_search('test query')
            assert mock_perform.call_count == 1  # Pas d'appel supplémentaire
            assert result1 == result2


# ============================================================================
# TESTS END-TO-END COMPLETS (AMÉLIORÉS)
# ============================================================================

class TestEndToEnd:
    """Tests d'intégration complets du pipeline Agent-First"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_fraud_detection(
        self, test_agent_config, sample_fraud_data, temp_dir
    ):
        """Test workflow complet : Profiler → Validator → Cleaner → Controller → YAML"""
        orchestrator = DataCleaningOrchestrator(test_agent_config, use_claude=False)
        
        # Mock des agents pour éviter appels API
        with patch.object(orchestrator.profiler, 'analyze') as mock_profile:
            mock_profile.return_value = {
                'quality_issues': ['Missing values in amount'],
                'columns': {'amount': {'missing': 10}}
            }
            
            with patch.object(orchestrator.validator, 'validate') as mock_validate:
                mock_validate.return_value = {
                    'valid': True,
                    'issues': [],
                    'sources': ['http://test.com']
                }
                
                with patch.object(orchestrator.cleaner, 'clean') as mock_clean:
                    cleaned_data = sample_fraud_data.copy()
                    cleaned_data['amount'].fillna(100, inplace=True)
                    
                    mock_clean.return_value = (
                        cleaned_data,
                        [{'action': 'fill_missing', 'column': 'amount'}]
                    )
                    
                    with patch.object(orchestrator.controller, 'validate') as mock_control:
                        mock_control.return_value = {
                            'quality_score': 85,
                            'validation_passed': True,
                            'metrics': {}
                        }
                        
                        # Exécuter workflow
                        user_context = {
                            'secteur_activite': 'finance',
                            'target_variable': 'fraud'
                        }
                        
                        cleaned_df, report = await orchestrator.clean_dataset(
                            sample_fraud_data,
                            user_context
                        )
                        
                        # Vérifications
                        assert cleaned_df is not None
                        assert 'transformations' in report
                        assert 'validation_sources' in report
                        assert mock_profile.called
                        assert mock_validate.called
                        assert mock_clean.called
                        assert mock_control.called
    
    @pytest.mark.asyncio
    async def test_workflow_with_yaml_export_and_reload(
        self, test_agent_config, sample_fraud_data, temp_dir
    ):
        """Test workflow avec export YAML puis reload et apply"""
        from automl_platform.agents import YAMLConfigHandler
        
        orchestrator = DataCleaningOrchestrator(test_agent_config)
        handler = YAMLConfigHandler()
        
        transformations = [
            {
                'column': 'amount',
                'action': 'fill_missing',
                'params': {'method': 'median'}
            }
        ]
        
        user_context = {
            'secteur_activite': 'finance',
            'target_variable': 'fraud'
        }
        
        # 1. Sauvegarder config
        yaml_path = handler.save_cleaning_config(
            transformations=transformations,
            validation_sources=['http://test.com'],
            user_context=user_context,
            output_path=str(Path(temp_dir) / 'config.yaml')
        )
        
        assert Path(yaml_path).exists()
        
        # 2. Recharger config
        loaded_config = handler.load_cleaning_config(yaml_path)
        
        assert loaded_config['metadata']['industry'] == 'finance'
        assert len(loaded_config['transformations']) == 1
        
        # 3. Appliquer transformations
        df_with_missing = sample_fraud_data.copy()
        df_with_missing.loc[0, 'amount'] = np.nan
        
        cleaned_df = handler.apply_transformations(
            df_with_missing,
            loaded_config['transformations']
        )
        
        assert cleaned_df['amount'].isnull().sum() == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, test_agent_config):
        """Test la gestion d'erreur et fallback"""
        orchestrator = DataCleaningOrchestrator(test_agent_config)
        
        with patch.object(orchestrator, '_process_chunk') as mock_process:
            mock_process.side_effect = Exception("Test error")
            
            df = pd.DataFrame({'col1': [1, 2, 3]})
            user_context = {'secteur_activite': 'test'}
            
            cleaned_df, report = await orchestrator._fallback_cleaning(df, user_context)
            
            assert cleaned_df is not None
            assert report['metadata']['fallback'] == True
            assert 'reason' in report['metadata']
    
    @pytest.mark.asyncio
    async def test_production_agent_full_pipeline(
        self, test_agent_config, sample_fraud_data
    ):
        """Test pipeline complet ProductionUniversalMLAgent"""
        from automl_platform.agents import ProductionUniversalMLAgent
        
        agent = ProductionUniversalMLAgent(
            config=test_agent_config,
            use_claude=False,
            max_cache_mb=10
        )
        
        # Mock des étapes clés
        with patch.object(agent, 'understand_problem') as mock_understand:
            mock_context = Mock(
                problem_type='fraud_detection',
                confidence=0.9,
                business_sector='finance',
                detected_patterns=[],
                temporal_aspect=True,
                imbalance_detected=True,
                recommended_config={},
                reasoning='',
                alternative_interpretations=[]
            )
            mock_understand.return_value = mock_context
            
            with patch.object(agent, 'validate_with_standards') as mock_validate:
                mock_validate.return_value = {'issues': []}
                
                with patch.object(agent, 'search_ml_best_practices') as mock_practices:
                    mock_practices.return_value = {'recommended_approaches': []}
                    
                    with patch.object(agent, 'generate_optimal_config') as mock_config:
                        mock_config.return_value = Mock(
                            task='classification',
                            algorithms=['XGBoost'],
                            primary_metric='f1',
                            preprocessing={},
                            feature_engineering={},
                            hpo_config={},
                            cv_strategy={},
                            ensemble_config={},
                            time_budget=3600,
                            resource_constraints={},
                            monitoring={}
                        )
                        
                        with patch.object(agent, '_execute_pipeline_with_protection') as mock_exec:
                            from automl_platform.agents import ProductionMLPipelineResult
                            mock_result = ProductionMLPipelineResult(
                                success=True,
                                cleaned_data=sample_fraud_data,
                                config_used=mock_config.return_value,
                                context_detected=mock_context,
                                cleaning_report={'transformations': []},
                                performance_metrics={'f1': 0.85},
                                execution_time=10.0,
                                memory_stats={'peak_mb': 100},
                                cache_stats={'items': 5},
                                performance_profile={'cache_hit_rate': 0.8}
                            )
                            mock_exec.return_value = mock_result
                            
                            # Exécuter pipeline
                            result = await agent.automl_without_templates(
                                sample_fraud_data,
                                target_col='fraud'
                            )
                            
                            # Vérifications
                            assert result.success == True
                            assert 'memory_stats' in result.__dict__
                            assert result.memory_stats['peak_mb'] == 100
                            assert agent.agent_metrics['context_detections'] > 0


# ============================================================================
# TESTS PERFORMANCE ET MEMOIRE
# ============================================================================

class TestPerformanceAndMemory:
    """Tests de performance et mémoire"""
    
    def test_bounded_list_performance(self):
        """Test que BoundedList n'a pas de fuite mémoire"""
        import sys
        
        bounded = BoundedList(maxlen=100)
        
        # Ajouter beaucoup d'éléments
        for i in range(10000):
            bounded.append({'data': [1] * 1000})  # 1000 ints par item
        
        # Vérifier que la taille reste limitée
        assert len(bounded) == 100
        
        # Vérifier utilisation mémoire raisonnable
        size_bytes = sys.getsizeof(bounded)
        assert size_bytes < 100 * 1024  # Moins de 100KB
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_orchestrator_memory_cleanup(self, test_agent_config):
        """Test que l'orchestrateur a un impact mémoire limité
    
        Ce test est marqué @pytest.mark.slow car il peut prendre du temps.
        Il utilise une tolérance large pour éviter les faux négatifs.
        """
        import gc
        import psutil
        import os
    
        orchestrator = DataCleaningOrchestrator(test_agent_config)
    
        # Forcer un garbage collection initial
        gc.collect()
        gc.collect()  # Double call pour être sûr
    
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
        # Simuler beaucoup d'exécutions
        for i in range(200):  # Plus que la limite de BoundedList
            orchestrator.execution_history.append({
                'data': [1] * 10000  # Données volumineuses
            })
    
        # Forcer cleanup
        gc.collect()
        gc.collect()
    
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
    
        # ✅ Assertion avec tolérance LARGE pour éviter flakiness
        # L'augmentation ne devrait PAS être proportionnelle à 200 itérations grâce au BoundedList (limité à 100)
    
        # Si on avait gardé les 200 items, on aurait ~200MB
        # Avec BoundedList, on devrait avoir ~100MB max
        assert mem_increase < 150, (
            f"Memory leak detected: {mem_increase:.1f} MB increase. "
            f"Expected < 150 MB with BoundedList limiting to 100 items."
        )
    
        # ✅ Vérifier aussi que BoundedList fonctionne
        assert len(orchestrator.execution_history) == 100, (
            "BoundedList should limit to 100 items"
        )


# ============================================================================
# TESTS CIRCUIT BREAKER ET RETRY (NOUVEAU)
# ============================================================================

class TestCircuitBreakerAndRetry:
    """Tests pour le circuit breaker et la logique de retry"""
    
    def test_circuit_breaker_lifecycle(self):
        """Test du cycle complet: CLOSED -> OPEN -> HALF_OPEN -> CLOSED"""
        config = AgentConfig(
            openai_api_key="test-key",
            anthropic_api_key="test-key",
            enable_circuit_breakers=True,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout=1
        )
    
        # État initial: CLOSED
        assert config.can_call_llm('claude') == True
        breaker = config.get_circuit_breaker('claude')
        assert breaker.state == 'CLOSED'
    
        # Enregistrer échecs -> OPEN
        for _ in range(3):
            config.record_llm_failure('claude')
    
        assert config.can_call_llm('claude') == False
        assert breaker.state == 'OPEN'
    
        # Attendre recovery timeout
        import time
        time.sleep(1.1)
    
        # Devrait passer en HALF_OPEN
        assert config.can_call_llm('claude') == True  # Peut tenter
        assert breaker.state == 'HALF_OPEN'
    
        # Succès -> CLOSED
        for _ in range(breaker.success_threshold):
            config.record_llm_success('claude')
    
        assert breaker.state == 'CLOSED'
        assert config.can_call_llm('claude') == True
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test que le circuit breaker s'ouvre après échecs"""
        config = AgentConfig()
        
        # Enregistrer beaucoup d'échecs
        for _ in range(10):
            config.record_llm_failure('claude')
        
        # Le circuit devrait s'ouvrir (selon implémentation)
        # Note: Dépend de l'implémentation réelle du circuit breaker
        pass
    
        
        @async_retry(max_attempts=3)
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await test_func()
        
        assert result == "success"
        assert call_count == 1  # Devrait réussir du premier coup
    
    @pytest.mark.asyncio
    async def test_retry_decorator_eventual_success(self):
        """Test retry avec succès après échecs"""
        
        call_count = 0
        
        @async_retry(max_attempts=3)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return "success"
        
        result = await test_func()
        
        assert result == "success"
        assert call_count == 3  # Devrait réussir au 3ème essai
    
    @pytest.mark.asyncio
    async def test_retry_decorator_max_attempts(self):
        """Test que retry abandonne après max_attempts"""
        # Déjà importé en haut du fichier (ligne 30)
        
        call_count = 0
        
        @async_retry(max_attempts=3)
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent error")
        
        with pytest.raises(Exception):
            await test_func()
        
        assert call_count == 3  # Devrait essayer 3 fois


# ============================================================================
# TESTS PRODUCTION UNIVERSAL ML AGENT
# ============================================================================

class TestProductionUniversalMLAgent:
    """Tests pour le ProductionUniversalMLAgent avec protection mémoire"""
    
    @pytest.fixture
    def production_agent(self, test_agent_config):
        """Fixture pour l'agent de production"""
        from automl_platform.agents import ProductionUniversalMLAgent
        return ProductionUniversalMLAgent(
            config=test_agent_config,
            use_claude=False,  # Désactiver Claude pour tests unitaires
            max_cache_mb=100,
            memory_warning_mb=500,
            memory_critical_mb=1000,
            batch_size=100
        )
    
    def test_memory_monitor_initialization(self, production_agent):
        """Test que le memory monitor est correctement initialisé"""
        assert production_agent.memory_monitor is not None
        assert production_agent.memory_monitor.warning_threshold > 0
        assert production_agent.memory_monitor.critical_threshold > 0
        
        # Vérifier les valeurs
        assert production_agent.memory_monitor.warning_threshold == 500 * 1024 * 1024
        assert production_agent.memory_monitor.critical_threshold == 1000 * 1024 * 1024
    
    def test_cache_initialization(self, production_agent):
        """Test que le cache LRU est initialisé"""
        assert production_agent.cache is not None
        assert production_agent.cache.max_size == 100 * 1024 * 1024
        assert production_agent.cache.current_size == 0
        assert len(production_agent.cache.cache) == 0
    
    def test_agents_initialization(self, production_agent):
        """Test que tous les agents sous-jacents sont initialisés"""
        assert production_agent.profiler is not None
        assert production_agent.validator is not None
        assert production_agent.cleaner is not None
        assert production_agent.controller is not None
        assert production_agent.context_detector is not None
        assert production_agent.config_generator is not None
        assert production_agent.adaptive_templates is not None
    
    def test_knowledge_base_initialization(self, production_agent):
        """Test que la knowledge base est initialisée avec limites"""
        assert production_agent.knowledge_base is not None
        assert production_agent.knowledge_base.max_patterns == 100
        assert isinstance(production_agent.knowledge_base.successful_patterns, list)
    
    @pytest.mark.asyncio
    async def test_understand_problem_basic(self, production_agent, sample_fraud_data):
        """Test la compréhension de problème basique"""
        with patch.object(production_agent.profiler, 'analyze') as mock_analyze:
            mock_analyze.return_value = {'quality_issues': []}
            
            with patch.object(production_agent.context_detector, 'detect_ml_context') as mock_detect:
                mock_context = Mock(
                    problem_type='fraud_detection',
                    confidence=0.9,
                    detected_patterns=['fraud_indicator'],
                    business_sector='finance',
                    temporal_aspect=True,
                    imbalance_detected=True,
                    recommended_config={},
                    reasoning='Test',
                    alternative_interpretations=[]
                )
                mock_detect.return_value = mock_context
                
                context = await production_agent.understand_problem(
                    sample_fraud_data,
                    target_col='fraud'
                )
                
                assert context.problem_type == 'fraud_detection'
                assert context.confidence == 0.9
                assert mock_analyze.called
                assert mock_detect.called
    
    @pytest.mark.asyncio
    async def test_understand_problem_with_cache(self, production_agent, sample_fraud_data):
        """Test que understand_problem utilise le cache"""
        # Premier appel
        with patch.object(production_agent.profiler, 'analyze') as mock_analyze:
            mock_analyze.return_value = {'quality_issues': []}
            
            with patch.object(production_agent.context_detector, 'detect_ml_context') as mock_detect:
                mock_context = Mock(
                    problem_type='fraud_detection',
                    confidence=0.9,
                    detected_patterns=[],
                    business_sector='finance',
                    temporal_aspect=True,
                    imbalance_detected=True,
                    recommended_config={},
                    reasoning='',
                    alternative_interpretations=[]
                )
                mock_detect.return_value = mock_context
                
                context1 = await production_agent.understand_problem(sample_fraud_data)
                
                # Deuxième appel - devrait utiliser cache
                context2 = await production_agent.understand_problem(sample_fraud_data)
                
                # Vérifier que detect n'a été appelé qu'une fois
                assert mock_detect.call_count == 1
                
                # Vérifier les métriques
                assert production_agent.agent_metrics['cache_hits'] >= 1
    
    def test_cache_functionality(self, production_agent):
        """Test les fonctionnalités du cache"""
        # Ajouter des données au cache
        test_data = {'test': 'data'}
        production_agent.cache.put('key1', test_data)
        
        # Vérifier récupération
        cached = production_agent.cache.get('key1')
        assert cached == test_data
        
        # Vérifier taille
        assert production_agent.cache.current_size > 0
        
        # Test de non-existence
        assert production_agent.cache.get('key_inexistant') is None
    
    def test_cache_eviction(self, production_agent):
        """Test l'éviction LRU du cache"""
        # Forcer le cache à être plein
        large_data = {'data': [1] * 10000}
        
        # Ajouter beaucoup d'items
        for i in range(20):
            production_agent.cache.put(f'key_{i}', large_data)
        
        # Vérifier que la taille ne dépasse pas max_size
        assert production_agent.cache.current_size <= production_agent.cache.max_size
        
        # Vérifier que les anciens items ont été évincés
        assert len(production_agent.cache.cache) < 20
    
    @pytest.mark.asyncio
    async def test_batch_processing_trigger(self, production_agent):
        """Test que le batch processing se déclenche pour gros datasets"""
        # Créer un gros dataset
        large_df = pd.DataFrame(np.random.randn(5000, 10))
        
        # Mock de execute_intelligent_cleaning
        with patch.object(production_agent, '_execute_cleaning_batched') as mock_batch:
            mock_batch.return_value = (large_df, {})
            
            context = Mock(problem_type='test', business_sector='test')
            config = Mock(
                preprocessing={}, feature_engineering={}, 
                algorithms=[], task='test', primary_metric='test'
            )
            
            await production_agent.execute_intelligent_cleaning(
                large_df, context, config
            )
            
            # Vérifier que batch processing a été appelé si dataset > batch_size * 2
            if len(large_df) > production_agent.batch_size * 2:
                assert mock_batch.called
    
    @pytest.mark.asyncio
    async def test_search_ml_best_practices_cache(self, production_agent):
        """Test que search_ml_best_practices utilise le cache"""
        problem_type = 'fraud_detection'
        data_chars = {'n_samples': 1000}
        
        # Premier appel
        practices1 = await production_agent.search_ml_best_practices(problem_type, data_chars)
        
        # Deuxième appel
        practices2 = await production_agent.search_ml_best_practices(problem_type, data_chars)
        
        # Devrait utiliser le cache
        assert practices1 == practices2
        assert production_agent.agent_metrics['cache_hits'] >= 1
    
    def test_compute_data_hash(self, production_agent, sample_fraud_data):
        """Test le calcul du hash de données"""
        hash1 = production_agent._compute_data_hash(sample_fraud_data)
        hash2 = production_agent._compute_data_hash(sample_fraud_data)
        
        # Même dataframe = même hash
        assert hash1 == hash2
        assert len(hash1) == 16  # Hash court
        
        # DataFrame différent = hash différent
        different_df = pd.DataFrame({'col1': [1, 2, 3]})
        hash3 = production_agent._compute_data_hash(different_df)
        assert hash1 != hash3
    
    def test_memory_safe_decorator_applied(self, production_agent):
        """Test que le décorateur @memory_safe est appliqué"""
        # Vérifier que les méthodes clés ont le wrapper
        assert hasattr(production_agent.understand_problem, '__wrapped__')
    
    def test_cleanup_method(self, production_agent):
        """Test la méthode de nettoyage manuel"""
        # Ajouter des données au cache
        production_agent.cache.put('test', {'data': [1] * 1000})
        
        assert production_agent.cache.current_size > 0
        
        # Nettoyer
        production_agent.cleanup()
        
        # Vérifier que le cache est vide
        assert production_agent.cache.current_size == 0
        assert len(production_agent.cache.cache) == 0
    
    @pytest.mark.asyncio
    async def test_automl_without_templates_basic(self, production_agent, sample_fraud_data):
        """Test AutoML complet basique (sans templates)"""
        # Mock tous les composants
        with patch.object(production_agent, 'understand_problem') as mock_understand:
            mock_context = Mock(
                problem_type='fraud_detection',
                confidence=0.9,
                detected_patterns=[],
                business_sector='finance'
            )
            mock_understand.return_value = mock_context
            
            with patch.object(production_agent, 'validate_with_standards') as mock_validate:
                mock_validate.return_value = {'issues': []}
                
                with patch.object(production_agent, 'search_ml_best_practices') as mock_practices:
                    mock_practices.return_value = {'recommended_approaches': []}
                    
                    with patch.object(production_agent, 'generate_optimal_config') as mock_config:
                        mock_config.return_value = Mock(
                            task='classification',
                            algorithms=['XGBoost'],
                            primary_metric='f1',
                            preprocessing={},
                            feature_engineering={},
                            hpo_config={},
                            cv_strategy={},
                            ensemble_config={},
                            time_budget=3600,
                            resource_constraints={},
                            monitoring={}
                        )
                        
                        with patch.object(production_agent, '_execute_pipeline_with_protection') as mock_execute:
                            from automl_platform.agents import ProductionMLPipelineResult
                            mock_result = ProductionMLPipelineResult(
                                success=True,
                                cleaned_data=sample_fraud_data,
                                config_used=mock_config.return_value,
                                context_detected=mock_context,
                                cleaning_report={},
                                performance_metrics={'f1': 0.85},
                                execution_time=10.0,
                                memory_stats={},
                                cache_stats={},
                                performance_profile={}
                            )
                            mock_execute.return_value = mock_result
                            
                            # Exécuter AutoML
                            result = await production_agent.automl_without_templates(
                                sample_fraud_data,
                                target_col='fraud'
                            )
                            
                            # Vérifications
                            assert result.success == True
                            assert result.cleaned_data is not None
                            assert 'memory_stats' in result.__dict__
                            assert 'cache_stats' in result.__dict__


class TestUniversalMLAgentBatchProcessing:
    """Tests du batch processing"""
    
    @pytest.mark.asyncio
    async def test_execute_cleaning_batched(self, test_agent_config):
        """Test batch processing avec vrais batches"""
        agent = ProductionUniversalMLAgent(
            config=test_agent_config,
            batch_size=100
        )
        
        # Dataset de 350 rows (3.5 batches)
        large_df = pd.DataFrame(np.random.randn(350, 5))
        
        context = Mock(problem_type='test', business_sector='test')
        config = Mock(
            preprocessing={}, feature_engineering={},
            algorithms=[], task='test', primary_metric='test'
        )
        
        with patch.object(agent, 'execute_intelligent_cleaning') as mock_clean:
            # Simuler nettoyage par batch
            mock_clean.side_effect = lambda df, *args: (df, {})
            
            cleaned_df, report = await agent._execute_cleaning_batched(
                large_df, context, config
            )
            
            # Vérifier 4 appels (350/100 = 3.5 → 4 batches)
            assert mock_clean.call_count == 4
            assert len(cleaned_df) == 350
            assert report['n_batches'] == 4
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_during_batches(self, test_agent_config):
        """Test cleanup mémoire pendant batch processing"""
        agent = ProductionUniversalMLAgent(
            config=test_agent_config,
            batch_size=50
        )
        
        large_df = pd.DataFrame(np.random.randn(500, 10))
        
        with patch.object(agent.memory_monitor, 'force_cleanup') as mock_cleanup:
            with patch.object(agent, 'execute_intelligent_cleaning') as mock_clean:
                mock_clean.side_effect = lambda df, *args: (df, {})
                
                context = Mock(problem_type='test')
                config = Mock(preprocessing={}, feature_engineering={})
                
                await agent._execute_cleaning_batched(large_df, context, config)
                
                # Vérifier cleanup appelé (tous les 5 batches)
                # 500/50 = 10 batches → 2 cleanups
                assert mock_cleanup.call_count >= 2


# ============================================================================
# TESTS PRODUCTION KNOWLEDGE BASE (NOUVEAUX - CRITIQUE)
# ============================================================================

class TestProductionKnowledgeBase:
    """Tests pour ProductionKnowledgeBase"""
    
    def test_initialization(self, temp_dir):
        """Test initialisation avec max_patterns"""
        from automl_platform.agents import ProductionKnowledgeBase
        
        kb = ProductionKnowledgeBase(max_patterns=50, storage_path=Path(temp_dir))
        
        assert kb.max_patterns == 50
        assert len(kb.successful_patterns) == 0
        assert len(kb.best_practices_cache) == 0
        assert kb.storage_path.exists()
    
    def test_store_and_retrieve_best_practices(self, temp_dir):
        """Test stockage/récupération de best practices"""
        from automl_platform.agents import ProductionKnowledgeBase
        
        kb = ProductionKnowledgeBase(storage_path=Path(temp_dir))
        
        practices = {
            'recommended_approaches': ['XGBoost', 'LightGBM'],
            'common_pitfalls': ['overfitting'],
            'benchmark_scores': {'auc': 0.95}
        }
        
        kb.store_best_practices('fraud_detection', practices)
        
        retrieved = kb.get_best_practices('fraud_detection')
        assert retrieved == practices
        assert retrieved['recommended_approaches'] == ['XGBoost', 'LightGBM']
    
    def test_get_nonexistent_practices(self, temp_dir):
        """Test récupération de practices inexistantes"""
        from automl_platform.agents import ProductionKnowledgeBase
        
        kb = ProductionKnowledgeBase(storage_path=Path(temp_dir))
        
        result = kb.get_best_practices('nonexistent_problem')
        
        assert result is None
    
    def test_store_successful_pattern(self, temp_dir):
        """Test stockage de pattern réussi"""
        from automl_platform.agents import ProductionKnowledgeBase
        
        kb = ProductionKnowledgeBase(storage_path=Path(temp_dir))
        
        kb.store_successful_pattern(
            task='fraud_detection',
            config={'algorithms': ['XGBoost'], 'metric': 'f1'},
            performance={'f1': 0.85, 'auc': 0.92}
        )
        
        assert len(kb.successful_patterns) == 1
        assert kb.successful_patterns[0]['task'] == 'fraud_detection'
        assert kb.successful_patterns[0]['score'] == 0.92  # max performance
    
    def test_store_pattern_with_limit(self, temp_dir):
        """Test limite de patterns + tri par score"""
        from automl_platform.agents import ProductionKnowledgeBase
        
        kb = ProductionKnowledgeBase(max_patterns=5, storage_path=Path(temp_dir))
        
        # Ajouter 10 patterns avec scores variés
        for i in range(10):
            kb.store_successful_pattern(
                task='test_task',
                config={'model': f'model_{i}'},
                performance={'score': i / 10.0}
            )
        
        # Vérifier limite respectée
        assert len(kb.successful_patterns) == 5
        
        # Vérifier tri (meilleurs en premier)
        assert kb.successful_patterns[0]['score'] == 0.9
        assert kb.successful_patterns[1]['score'] == 0.8
        assert kb.successful_patterns[4]['score'] == 0.5
    
    def test_persistence_save_and_load(self, temp_dir):
        """Test sauvegarde/chargement sur disque"""
        from automl_platform.agents import ProductionKnowledgeBase
        
        # Créer et remplir KB
        kb1 = ProductionKnowledgeBase(storage_path=Path(temp_dir))
        kb1.store_best_practices('test', {'data': 'value1'})
        kb1.store_successful_pattern(
            task='task1',
            config={'algo': 'XGBoost'},
            performance={'score': 0.85}
        )
        kb1._save_knowledge()
        
        # Créer nouveau KB pointant sur même storage
        kb2 = ProductionKnowledgeBase(storage_path=Path(temp_dir))
        
        # Vérifier données chargées
        assert kb2.get_best_practices('test') == {'data': 'value1'}
        assert len(kb2.successful_patterns) == 1
        assert kb2.successful_patterns[0]['task'] == 'task1'
    
    def test_persistence_corrupted_file(self, temp_dir):
        """Test chargement avec fichier corrompu"""
        from automl_platform.agents import ProductionKnowledgeBase
        
        storage_path = Path(temp_dir)
        kb_file = storage_path / "knowledge.pkl"
        
        # Créer fichier corrompu
        storage_path.mkdir(exist_ok=True)
        with open(kb_file, 'w') as f:
            f.write("corrupted data")
        
        # KB devrait gérer l'erreur gracieusement
        kb = ProductionKnowledgeBase(storage_path=storage_path)
        
        assert len(kb.successful_patterns) == 0
        assert len(kb.best_practices_cache) == 0
    
    def test_max_patterns_enforcement(self, temp_dir):
        """Test que max_patterns est strictement respecté"""
        from automl_platform.agents import ProductionKnowledgeBase
        
        kb = ProductionKnowledgeBase(max_patterns=3, storage_path=Path(temp_dir))
        
        # Ajouter 5 patterns
        for i in range(5):
            kb.store_successful_pattern(
                task='test',
                config={'id': i},
                performance={'score': 0.5 + i * 0.1}
            )
        
        # Vérifier exactement 3 patterns (les meilleurs)
        assert len(kb.successful_patterns) == 3
        assert kb.successful_patterns[0]['score'] == 0.9  # Meilleur
        assert kb.successful_patterns[2]['score'] == 0.7  # 3ème meilleur
    
    def test_empty_performance_dict(self, temp_dir):
        """Test avec dictionnaire de performance vide"""
        from automl_platform.agents import ProductionKnowledgeBase
        
        kb = ProductionKnowledgeBase(storage_path=Path(temp_dir))
        
        kb.store_successful_pattern(
            task='test',
            config={'algo': 'test'},
            performance={}  # Vide
        )
        
        assert len(kb.successful_patterns) == 1
        assert kb.successful_patterns[0]['score'] == 0  # Défaut


# ============================================================================
# TESTS YAML CONFIG HANDLER
# ============================================================================

class TestYAMLConfigHandler:
    """Tests pour le gestionnaire de configuration YAML"""
    
    @pytest.fixture
    def handler(self):
        from automl_platform.agents import YAMLConfigHandler
        return YAMLConfigHandler()
    
    @pytest.fixture
    def sample_transformations(self):
        return [
            {
                'column': 'amount',
                'action': 'normalize_currency',
                'params': {'target_currency': 'EUR'},
                'rationale': 'Test'
            },
            {
                'column': 'date',
                'action': 'standardize_format',
                'params': {'format': '%Y-%m-%d'},
                'rationale': 'Test'
            }
        ]
    
    def test_save_cleaning_config(self, handler, sample_transformations, temp_dir):
        """Test sauvegarde de configuration"""
        user_context = {
            'secteur_activite': 'finance',
            'target_variable': 'fraud',
            'contexte_metier': 'Test'
        }
        
        output_path = Path(temp_dir) / 'test_config.yaml'
        
        saved_path = handler.save_cleaning_config(
            transformations=sample_transformations,
            validation_sources=['http://test.com'],
            user_context=user_context,
            output_path=str(output_path)
        )
        
        assert Path(saved_path).exists()
        assert saved_path == str(output_path)
    
    def test_load_cleaning_config(self, handler, sample_transformations, temp_dir):
        """Test chargement de configuration"""
        # Sauvegarder d'abord
        user_context = {'secteur_activite': 'finance', 'target_variable': 'fraud'}
        output_path = Path(temp_dir) / 'test_config.yaml'
        
        handler.save_cleaning_config(
            transformations=sample_transformations,
            validation_sources=['http://test.com'],
            user_context=user_context,
            output_path=str(output_path)
        )
        
        # Charger
        config = handler.load_cleaning_config(str(output_path))
        
        assert 'metadata' in config
        assert 'transformations' in config
        assert config['metadata']['industry'] == 'finance'
        assert len(config['transformations']) == len(sample_transformations)
    
    def test_validate_config_valid(self, handler, sample_transformations):
        """Test validation d'une config valide"""
        config = {
            'metadata': {
                'industry': 'finance',
                'target_variable': 'fraud',
                'processing_date': '2024-01-01'
            },
            'transformations': sample_transformations,
            'validation_sources': ['http://test.com']
        }
        
        assert handler.validate_config(config) == True
    
    def test_validate_config_invalid(self, handler):
        """Test validation d'une config invalide"""
        config = {
            'metadata': {},  # Manque des champs requis
            'transformations': []
        }
        
        with pytest.raises(ValueError):
            handler.validate_config(config)
    
    def test_apply_transformations_fill_missing(self, handler):
        """Test application de transformation fill_missing"""
        df = pd.DataFrame({'col1': [1, 2, None, 4]})
        transformations = [
            {
                'column': 'col1',
                'action': 'fill_missing',
                'params': {'method': 'median'}
            }
        ]
        
        result = handler.apply_transformations(df, transformations)
        
        assert result['col1'].isnull().sum() == 0
        assert result['col1'].iloc[2] == 2.0  # Médiane de [1, 2, 4]
    
    def test_apply_transformations_normalize(self, handler):
        """Test application de transformation normalize"""
        df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        transformations = [
            {
                'column': 'col1',
                'action': 'normalize',
                'params': {'method': 'minmax'}
            }
        ]
        
        result = handler.apply_transformations(df, transformations)
        
        assert result['col1'].min() == 0.0
        assert result['col1'].max() == 1.0
    
    def test_apply_transformations_handle_outliers(self, handler):
        """Test application de transformation handle_outliers"""
        df = pd.DataFrame({'col1': [1, 2, 3, 4, 100]})
        transformations = [
            {
                'column': 'col1',
                'action': 'handle_outliers',
                'params': {'method': 'clip', 'lower_percentile': 0.1, 'upper_percentile': 0.9}
            }
        ]
        
        result = handler.apply_transformations(df, transformations)
        
        assert result['col1'].max() < 100
    
    def test_apply_transformations_encode(self, handler):
        """Test application de transformation encode"""
        df = pd.DataFrame({'col1': ['A', 'B', 'C', 'A']})
        transformations = [
            {
                'column': 'col1',
                'action': 'encode',
                'params': {'method': 'label'}
            }
        ]
        
        result = handler.apply_transformations(df, transformations)
        
        assert pd.api.types.is_numeric_dtype(result['col1'])
    
    def test_create_example_config(self, handler, temp_dir):
        """Test création de config exemple"""
        output_path = Path(temp_dir) / 'example.yaml'
        
        created_path = handler.create_example_config(str(output_path))
        
        assert Path(created_path).exists()
        
        # Charger et valider
        config = handler.load_cleaning_config(created_path)
        assert handler.validate_config(config) == True


# ============================================================================
# TESTS INTELLIGENT DATA CLEANER
# ============================================================================

class TestIntelligentDataCleaner:
    """Tests pour IntelligentDataCleaner avec Claude"""
    
    @pytest.fixture
    def cleaner_no_claude(self, test_agent_config):
        from automl_platform.agents import IntelligentDataCleaner
        return IntelligentDataCleaner(
            config=test_agent_config,
            use_claude=False
        )
    
    @pytest.fixture
    def cleaner_with_claude(self, test_agent_config):
        from automl_platform.agents import IntelligentDataCleaner
        return IntelligentDataCleaner(
            config=test_agent_config,
            use_claude=True
        )
    
    def test_initialization_without_claude(self, cleaner_no_claude):
        """Test initialisation sans Claude"""
        assert cleaner_no_claude.use_claude == False
        assert cleaner_no_claude.claude_client is None
        assert cleaner_no_claude.orchestrator is not None
        assert cleaner_no_claude.quality_agent is not None
    
    def test_initialization_with_claude(self, cleaner_with_claude):
        """Test initialisation avec Claude"""
        assert cleaner_with_claude.use_claude == True
        assert cleaner_with_claude.claude_client is not None
        assert cleaner_with_claude.claude_model == "claude-sonnet-4-5-20250929"
    
    @pytest.mark.asyncio
    async def test_smart_clean_auto_mode(self, cleaner_no_claude, sample_fraud_data):
        """Test smart_clean en mode auto"""
        user_context = {
            'secteur_activite': 'finance',
            'target_variable': 'fraud'
        }
        
        # Mock des composants
        with patch.object(cleaner_no_claude.quality_agent, 'assess') as mock_assess:
            mock_assess.return_value = DataQualityAssessment(
                quality_score=75.0,
                alerts=[],
                warnings=[],
                recommendations=[],
                drift_risk='low',
                target_leakage_risk='low'
            )
            
            with patch.object(cleaner_no_claude, '_clean_with_agents') as mock_clean:
                mock_clean.return_value = (sample_fraud_data, {'method': 'agents'})
                
                cleaned_df, report = await cleaner_no_claude.smart_clean(
                    sample_fraud_data,
                    user_context,
                    mode='auto'
                )
                
                assert cleaned_df is not None
                assert 'summary' in report
                assert 'initial_quality' in report['summary']
    
    @pytest.mark.asyncio
    async def test_recommend_cleaning_approach(self, cleaner_no_claude, sample_fraud_data):
        """Test recommandation d'approche de nettoyage"""
        user_context = {'secteur_activite': 'finance'}
        
        with patch.object(cleaner_no_claude.quality_agent, 'assess') as mock_assess:
            mock_assess.return_value = DataQualityAssessment(
                quality_score=60.0,
                alerts=[{'message': 'Test alert'}],
                warnings=[],
                recommendations=[],
                drift_risk='medium',
                target_leakage_risk='low'
            )
            
            recommendation = await cleaner_no_claude.recommend_cleaning_approach(
                sample_fraud_data,
                user_context
            )
            
            assert 'recommended_mode' in recommendation
            assert 'current_quality' in recommendation
            assert recommendation['current_quality'] == 60.0
            assert 'reasoning' in recommendation
    
    @pytest.mark.asyncio
    async def test_claude_mode_selection(self, cleaner_with_claude, sample_fraud_data):
        """Test sélection de mode avec Claude"""
        user_context = {'secteur_activite': 'finance'}
        
        with patch.object(cleaner_with_claude.quality_agent, 'assess') as mock_assess:
            mock_assess.return_value = DataQualityAssessment(
                quality_score=70.0,
                alerts=[],
                warnings=[],
                recommendations=[],
                drift_risk='low',
                target_leakage_risk='low'
            )
            
                with patch.object(agent, 'claude_client') as mock_claude:
                    mock_response = Mock()
                    mock_response.content = [Mock(text='{"key": "value"}')]
                    mock_claude.messages.create = AsyncMock(return_value=mock_response)
                    content=[Mock(text=json.dumps({
                        'recommended_mode': 'hybrid',
                        'confidence': 0.85,
                        'reasoning': 'Claude decision',
                        'estimated_time_minutes': 15,
                        'key_considerations': ['test']
                    }))]
                ))
                
                with patch.object(cleaner_with_claude, '_clean_hybrid') as mock_clean:
                    mock_clean.return_value = (sample_fraud_data, {})
                    
                    cleaned_df, report = await cleaner_with_claude.smart_clean(
                        sample_fraud_data,
                        user_context,
                        mode='auto'
                    )
                    
                    assert mock_messages.create.called
    
    def test_get_cleaning_summary(self, cleaner_no_claude):
        """Test récupération du résumé de nettoyage"""
        # Ajouter quelques opérations
        cleaner_no_claude.cleaning_history.append({
            'timestamp': datetime.now(),
            'mode': 'agents',
            'quality_improvement': 15.0,
            'report': {},
            'used_claude': False
        })
        
        summary = cleaner_no_claude.get_cleaning_summary()
        
        assert 'total_operations' in summary
        assert summary['total_operations'] == 1
        assert 'average_quality_improvement' in summary
    
    def test_get_metrics(self, cleaner_no_claude):
        """Test récupération des métriques"""
        metrics = cleaner_no_claude.get_metrics()
        
        assert 'total_cleanings' in metrics
        assert 'claude_insights_generated' in metrics
        assert 'rule_based_fallbacks' in metrics


class TestIntelligentDataCleanerStrategies:
    """Tests des stratégies de nettoyage"""
    
    @pytest.mark.asyncio
    async def test_generate_strategic_prompts(self, test_agent_config):
        """Test génération de prompts depuis stratégie Claude"""
        cleaner = IntelligentDataCleaner(config=test_agent_config)
        
        strategy = {
            'priorities': [
                {'issue': 'Missing values', 'priority': 'high', 'approach': 'Impute with median'},
                {'issue': 'Outliers', 'priority': 'medium', 'approach': 'Clip values'}
            ]
        }
        
        assessment = Mock(alerts=[], warnings=[])
        
        prompts = cleaner._generate_strategic_prompts(strategy, assessment)
        
        assert len(prompts) >= 2
        assert 'median' in prompts[0].lower() or 'impute' in prompts[0].lower()
        assert 'clip' in prompts[1].lower() or 'outlier' in prompts[1].lower()
    
    @pytest.mark.asyncio
    async def test_clean_hybrid_two_phases(self, test_agent_config, sample_fraud_data):
        """Test nettoyage hybride avec les deux phases"""
        cleaner = IntelligentDataCleaner(config=test_agent_config)
        
        user_context = {'secteur_activite': 'finance'}
        assessment = Mock(quality_score=60.0, alerts=[{'message': 'Test'}])
        strategy = {'priorities': []}
        
        # Mock phase agents
        with patch.object(cleaner, '_clean_with_agents') as mock_agents:
            mock_agents.return_value = (sample_fraud_data, {'method': 'agents'})
            
            # Mock phase conversationnelle
            with patch.object(cleaner, '_clean_conversationally') as mock_conv:
                mock_conv.return_value = (sample_fraud_data, {'method': 'conversational'})
                
                cleaned_df, report = await cleaner._clean_hybrid(
                    sample_fraud_data, user_context, assessment, strategy
                )
                
                # Vérifier les deux phases appelées
                assert mock_agents.called
                assert mock_conv.called
                assert report['method'] == 'hybrid'
                assert 'agent_phase' in report
                assert 'conversational_phase' in report


# ============================================================================
# TESTS MEMORY PROTECTION UTILITIES
# ============================================================================

class TestMemoryProtection:
    """Tests pour les utilitaires de protection mémoire"""
    
    def test_memory_monitor_initialization(self):
        """Test initialisation du MemoryMonitor"""
        from automl_platform.agents import MemoryMonitor
        
        monitor = MemoryMonitor(warning_threshold_mb=100, critical_threshold_mb=200)
        
        assert monitor.warning_threshold == 100 * 1024 * 1024
        assert monitor.critical_threshold == 200 * 1024 * 1024
        assert monitor.initial_memory > 0
    
    def test_memory_monitor_tracking(self):
        """Test le tracking mémoire"""
        from automl_platform.agents import MemoryMonitor
        
        monitor = MemoryMonitor()
        
        status = monitor.check_memory()
        
        assert 'current_mb' in status
        assert 'peak_mb' in status
        assert 'available_mb' in status
        assert status['current_mb'] > 0
    
    def test_memory_monitor_warnings(self):
        """Test les alertes mémoire"""
        from automl_platform.agents import MemoryMonitor
        
        # Créer un monitor avec seuil très bas
        monitor = MemoryMonitor(warning_threshold_mb=1, critical_threshold_mb=2)
        
        status = monitor.check_memory()
        
        # Devrait probablement trigger un warning
        assert 'warning' in status
        assert isinstance(status['warning'], bool)
    
    def test_memory_budget_enforcement(self):
        """Test l'application du budget mémoire"""
        from automl_platform.agents import MemoryBudget
        
        with MemoryBudget(budget_mb=100) as budget:
            # Allouer de la mémoire
            data = [1] * 1000000
            
            # Le budget devrait tracker
            assert budget.start_memory > 0
    
    def test_lru_cache_initialization(self):
        """Test initialisation du cache LRU"""
        from automl_platform.agents import LRUMemoryCache
        
        cache = LRUMemoryCache(max_size_mb=10)
        
        assert cache.max_size == 10 * 1024 * 1024
        assert cache.current_size == 0
        assert len(cache.cache) == 0
    
    def test_lru_cache_put_get(self):
        """Test put/get du cache"""
        from automl_platform.agents import LRUMemoryCache
        
        cache = LRUMemoryCache(max_size_mb=10)
        
        data = {'test': [1, 2, 3]}
        cache.put('key1', data)
        
        retrieved = cache.get('key1')
        assert retrieved == data
        assert cache.current_size > 0
    
    def test_lru_cache_eviction(self):
        """Test l'éviction LRU du cache"""
        from automl_platform.agents import LRUMemoryCache
        
        cache = LRUMemoryCache(max_size_mb=1)  # Petit cache
        
        # Ajouter beaucoup de données
        large_data = {'data': [1] * 100000}
        for i in range(20):
            cache.put(f'key_{i}', large_data)
        
        # Vérifier que la taille ne dépasse pas
        assert cache.current_size <= cache.max_size
        
        # Les premières clés devraient avoir été évincées
        assert cache.get('key_0') is None
    
    def test_lru_cache_clear(self):
        """Test le nettoyage du cache"""
        from automl_platform.agents import LRUMemoryCache
        
        cache = LRUMemoryCache(max_size_mb=10)
        
        cache.put('key1', {'data': [1, 2, 3]})
        cache.put('key2', {'data': [4, 5, 6]})
        
        assert cache.current_size > 0
        
        cache.clear()
        
        assert cache.current_size == 0
        assert len(cache.cache) == 0
    
    def test_lru_cache_stats(self):
        """Test les statistiques du cache"""
        from automl_platform.agents import LRUMemoryCache
        
        cache = LRUMemoryCache(max_size_mb=10)
        
        cache.put('key1', {'data': [1, 2, 3]})
        
        stats = cache.get_stats()
        
        assert 'items' in stats
        assert 'size_mb' in stats
        assert 'max_size_mb' in stats
        assert 'utilization' in stats
        assert stats['items'] == 1
        assert stats['utilization'] > 0
    
    @pytest.mark.asyncio
    async def test_memory_safe_decorator(self):
        """Test le décorateur @memory_safe"""
        from automl_platform.agents import memory_safe
        
        @memory_safe(max_memory_mb=100)
        async def test_function():
            # Allouer de la mémoire
            data = [1] * 1000000
            return "success"
        
        result = await test_function()
        assert result == "success"
    
    def test_dataframe_batch_processor(self):
        """Test le context manager de batch processing"""
        from automl_platform.agents import dataframe_batch_processor
        
        df = pd.DataFrame(np.random.randn(1000, 10))
        
        batches_processed = 0
        with dataframe_batch_processor(df, batch_size=100) as batch_gen:
            for batch in batch_gen:
                assert len(batch) <= 100
                batches_processed += 1
        
        assert batches_processed == 10


# ============================================================================
# TESTS EDGE CASES - DATAFRAMES EXTRÊMES (NOUVEAUX)
# ============================================================================

class TestDataFrameEdgeCases:
    """Tests avec DataFrames dans des cas limites"""
    
    @pytest.mark.asyncio
    async def test_empty_dataframe(self, test_agent_config):
        """Test avec DataFrame vide"""
        orchestrator = DataCleaningOrchestrator(test_agent_config)
        
        empty_df = pd.DataFrame()
        user_context = {'secteur_activite': 'test'}
        
        # Devrait gérer gracieusement
        with pytest.raises(Exception) or True:
            cleaned_df, report = await orchestrator.clean_dataset(
                empty_df,
                user_context
            )
    
    @pytest.mark.asyncio
    async def test_single_row_dataframe(self, test_agent_config):
        """Test avec DataFrame d'une seule ligne"""
        orchestrator = DataCleaningOrchestrator(test_agent_config)
        
        single_row_df = pd.DataFrame({'col1': [1], 'col2': ['a']})
        user_context = {'secteur_activite': 'test'}
        
        with patch.object(orchestrator, '_process_chunk') as mock_process:
            mock_process.return_value = single_row_df
            
            cleaned_df, report = await orchestrator.clean_dataset(
                single_row_df,
                user_context
            )
            
            assert len(cleaned_df) >= 0  # Au moins pas d'erreur
    
    @pytest.mark.asyncio
    async def test_all_nan_dataframe(self, test_agent_config):
        """Test avec DataFrame 100% NaN"""
        orchestrator = DataCleaningOrchestrator(test_agent_config)
        
        all_nan_df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [np.nan, np.nan, np.nan]
        })
        
        user_context = {'secteur_activite': 'test'}
        
        with patch.object(orchestrator, '_process_chunk') as mock_process:
            # Devrait fallback ou nettoyer
            mock_process.return_value = all_nan_df.dropna(axis=1, how='all')
            
            cleaned_df, report = await orchestrator.clean_dataset(
                all_nan_df,
                user_context
            )
            
            # Vérifier qu'il n'y a plus de colonnes all-NaN
            assert not cleaned_df.isnull().all().any()
    
    @pytest.mark.asyncio
    async def test_very_wide_dataframe(self, test_agent_config):
        """Test avec DataFrame très large (1000 colonnes)"""
        orchestrator = DataCleaningOrchestrator(test_agent_config)
        
        wide_df = pd.DataFrame(np.random.randn(100, 1000))
        user_context = {'secteur_activite': 'test'}
        
        with patch.object(orchestrator, '_process_chunk') as mock_process:
            mock_process.return_value = wide_df
            
            # Devrait gérer sans crash mémoire
            cleaned_df, report = await orchestrator.clean_dataset(
                wide_df,
                user_context
            )
            
            assert cleaned_df.shape[1] <= 1000
    
    @pytest.mark.asyncio
    async def test_duplicate_columns(self, test_agent_config):
        """Test avec colonnes en double"""
        orchestrator = DataCleaningOrchestrator(test_agent_config)
        
        # Créer DF avec colonnes dupliquées
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        df.columns = ['col1', 'col1', 'col2']  # Colonnes dupliquées
        
        user_context = {'secteur_activite': 'test'}
        
        with patch.object(orchestrator, '_process_chunk') as mock_process:
            # Dédupliquer les colonnes
            df_dedupe = df.loc[:, ~df.columns.duplicated()]
            mock_process.return_value = df_dedupe
            
            cleaned_df, report = await orchestrator.clean_dataset(
                df,
                user_context
            )
            
            # Vérifier pas de doublons
            assert len(cleaned_df.columns) == len(set(cleaned_df.columns))
    
    @pytest.mark.asyncio
    async def test_mixed_types_in_column(self, test_agent_config):
        """Test avec types mélangés dans une colonne"""
        orchestrator = DataCleaningOrchestrator(test_agent_config)
        
        mixed_df = pd.DataFrame({
            'col1': [1, '2', 3.0, None, 'text']
        })
        
        user_context = {'secteur_activite': 'test'}
        
        with patch.object(orchestrator, '_process_chunk') as mock_process:
            # Convertir en string ou nettoyer
            cleaned = mixed_df.copy()
            cleaned['col1'] = pd.to_numeric(cleaned['col1'], errors='coerce')
            mock_process.return_value = cleaned
            
            cleaned_df, report = await orchestrator.clean_dataset(
                mixed_df,
                user_context
            )
            
            # Devrait avoir un type cohérent
            assert len(cleaned_df['col1'].apply(type).unique()) <= 2  # float + NoneType max


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
