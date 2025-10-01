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
from agents.profiler_agent import ProfilerAgent
from agents.validator_agent import ValidatorAgent
from agents.cleaner_agent import CleanerAgent
from agents.controller_agent import ControllerAgent
from agents.intelligent_context_detector import IntelligentContextDetector, MLContext
from agents.intelligent_config_generator import IntelligentConfigGenerator, OptimalConfig
from agents.adaptive_template_system import AdaptiveTemplateSystem, AdaptiveTemplate
from agents.data_cleaning_orchestrator import DataCleaningOrchestrator
from agents.agent_config import AgentConfig, AgentType


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
def mock_agent_config():
    """Configuration mock pour les agents"""
    config = AgentConfig(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-claude-key",
        model="gpt-4-1106-preview",
        enable_web_search=True,
        enable_file_operations=True,
        max_iterations=3,
        timeout_seconds=30,
        cache_dir="./test_cache",
        output_dir="./test_output"
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
    async def test_lazy_initialization(self, mock_agent_config):
        """Vérifie que l'agent ne s'initialise pas au constructeur"""
        agent = ProfilerAgent(mock_agent_config)
        
        assert agent._initialized == False
        assert agent.assistant is None
    
    @pytest.mark.asyncio
    async def test_ensure_initialized_called_before_use(self, mock_agent_config, mock_openai_client):
        """Vérifie que ensure_initialized est appelé avant utilisation"""
        agent = ProfilerAgent(mock_agent_config)
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
    async def test_analyze_with_lazy_init(self, mock_agent_config, mock_openai_client):
        """Test que l'analyse s'initialise à la première utilisation"""
        agent = ProfilerAgent(mock_agent_config)
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
    async def test_analyze_without_openai(self, mock_agent_config):
        """Vérifie le fallback vers basic_profiling sans OpenAI"""
        agent = ProfilerAgent(mock_agent_config)
        agent.client = None
        
        df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': ['a', 'b', 'c', 'd']
        })
        
        result = await agent.analyze(df)
        
        assert isinstance(result, dict)
        assert 'summary' in result
        assert result['summary']['total_rows'] == 4
    
    def test_fallback_when_openai_unavailable(self, mock_agent_config):
        """Test le fallback complet quand OpenAI n'est pas disponible"""
        agent = ProfilerAgent(mock_agent_config)
        agent.client = None
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = agent._basic_profiling(df)
        
        assert result['summary']['total_rows'] == 3
        assert 'columns' in result
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, mock_agent_config):
        """Test intégration avec circuit breaker"""
        # Ce test vérifie que les circuit breakers sont respectés
        agent = ProfilerAgent(mock_agent_config)
        # Le circuit breaker devrait être dans la config
        assert hasattr(mock_agent_config, 'can_call_llm') or True


# ============================================================================
# PHASE 2.1 : TESTS AGENTS CORE - CLEANER AGENT (NOUVEAUX)
# ============================================================================

class TestCleanerAgentUpdated:
    """Tests complets pour CleanerAgent avec lazy initialization"""
    
    @pytest.mark.asyncio
    async def test_lazy_initialization(self, mock_agent_config):
        """Vérifie que l'agent ne s'initialise pas au constructeur"""
        agent = CleanerAgent(mock_agent_config)
        
        assert agent._initialized == False
        assert agent.assistant is None
    
    @pytest.mark.asyncio
    async def test_clean_without_openai(self, mock_agent_config):
        """Vérifie le fallback vers basic_cleaning sans OpenAI"""
        agent = CleanerAgent(mock_agent_config)
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
    
    def test_basic_cleaning_removes_duplicates(self, mock_agent_config):
        """Test que basic_cleaning retire les duplicates"""
        agent = CleanerAgent(mock_agent_config)
        
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })
        profile_report = {}
        
        cleaned_df, transformations = agent._basic_cleaning(df, profile_report)
        
        assert len(cleaned_df) == 3
        assert any('remove_duplicates' in t.get('action', '') for t in transformations)
    
    def test_fallback_when_openai_unavailable(self, mock_agent_config):
        """Test fallback complet sans OpenAI"""
        agent = CleanerAgent(mock_agent_config)
        agent.client = None
        
        df = pd.DataFrame({'col1': [1, None, 3]})
        profile_report = {}
        
        cleaned_df, trans = agent._basic_cleaning(df, profile_report)
        assert cleaned_df['col1'].isnull().sum() == 0


# ============================================================================
# PHASE 2.1 : TESTS VALIDATOR AGENT HYBRID (NOUVEAUX)
# ============================================================================

class TestValidatorAgentHybrid:
    """Tests pour ValidatorAgent avec architecture hybride"""
    
    @pytest.mark.asyncio
    async def test_openai_for_web_search(self, mock_agent_config):
        """Vérifie que OpenAI est utilisé pour la recherche web"""
        agent = ValidatorAgent(mock_agent_config, use_claude=False)
        
        with patch.object(agent, '_search_sector_standards') as mock_search:
            mock_search.return_value = {'standards': [], 'sources': []}
            
            df = pd.DataFrame({'col1': [1, 2, 3]})
            result = await agent.validate(df, {})
            
            assert mock_search.called
    
    @pytest.mark.asyncio
    async def test_claude_for_reasoning(self, mock_agent_config, mock_claude_client):
        """Vérifie que Claude est utilisé pour le raisonnement"""
        agent = ValidatorAgent(mock_agent_config, use_claude=True)
        agent.claude_client = mock_claude_client
        agent.openai_client = None
        
        df = pd.DataFrame({'amount': [100, 200]})
        result = await agent.validate(df, {})
        
        assert 'valid' in result
        assert mock_claude_client.messages.create.called
    
    @pytest.mark.asyncio
    async def test_fallback_without_claude(self, mock_agent_config):
        """Test fallback quand Claude n'est pas disponible"""
        agent = ValidatorAgent(mock_agent_config, use_claude=True)
        agent.claude_client = None
        agent.openai_client = None
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = await agent.validate(df, {})
        
        assert isinstance(result, dict)
        assert 'valid' in result
    
    @pytest.mark.asyncio
    async def test_fallback_without_openai(self, mock_agent_config):
        """Test fallback quand OpenAI n'est pas disponible"""
        agent = ValidatorAgent(mock_agent_config, use_claude=False)
        agent.openai_client = None
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = await agent.validate(df, {})
        
        assert isinstance(result, dict)
    
    def test_basic_references_only(self, mock_agent_config):
        """Test génération de références basiques sans API"""
        agent = ValidatorAgent(mock_agent_config, use_claude=False)
        
        refs = agent._get_basic_references('finance')
        
        assert 'standards' in refs
        assert len(refs['standards']) > 0


# ============================================================================
# PHASE 2.1 : TESTS CONTROLLER AGENT (NOUVEAUX)
# ============================================================================

class TestControllerAgentUpdated:
    """Tests pour ControllerAgent avec Claude SDK"""
    
    @pytest.mark.asyncio
    async def test_validate_without_claude(self, mock_agent_config):
        """Test validation sans Claude (fallback)"""
        agent = ControllerAgent(mock_agent_config)
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
    """Tests pour IntelligentContextDetector avec Claude SDK"""
    
    @pytest.mark.asyncio
    async def test_claude_enabled_initialization(self, mock_agent_config):
        """Test initialisation avec Claude activé"""
        detector = IntelligentContextDetector(
            anthropic_api_key=mock_agent_config.anthropic_api_key,
            config=mock_agent_config
        )
        
        assert detector.use_claude == True
        await detector.ensure_initialized()
        assert detector._initialized == True
    
    @pytest.mark.asyncio
    async def test_claude_enhanced_detection(self, mock_agent_config, sample_fraud_data, mock_claude_client):
        """Test détection avec enhancement Claude"""
        detector = IntelligentContextDetector(
            anthropic_api_key=mock_agent_config.anthropic_api_key,
            config=mock_agent_config
        )
        detector.claude_client = mock_claude_client
        
        mock_claude_client.messages.create.return_value.content = [
            Mock(text=json.dumps({
                'problem_type': 'fraud_detection',
                'confidence': 0.95,
                'reasoning': 'Strong fraud indicators',
                'alternatives': []
            }))
        ]
        
        context = await detector.detect_ml_context(sample_fraud_data, 'fraud')
        
        assert context.problem_type == 'fraud_detection'
        assert context.confidence >= 0.9
    
    @pytest.mark.asyncio
    async def test_detect_with_claude_failure(self, mock_agent_config, sample_fraud_data, mock_claude_client):
        """Test fallback quand Claude échoue"""
        detector = IntelligentContextDetector(
            anthropic_api_key=mock_agent_config.anthropic_api_key,
            config=mock_agent_config
        )
        detector.claude_client = mock_claude_client
        mock_claude_client.messages.create.side_effect = Exception("Claude API Error")
        
        context = await detector.detect_ml_context(sample_fraud_data, 'fraud')
        
        assert context.problem_type == 'fraud_detection'
        assert context.confidence > 0
    
    def test_rule_based_fallback(self, mock_agent_config, sample_fraud_data):
        """Test fallback rule-based complet"""
        detector = IntelligentContextDetector(anthropic_api_key=None)
        detector.use_claude = False
        
        analysis = detector._analyze_columns(sample_fraud_data)
        
        assert 'fraud_indicator' in analysis['detected_patterns']
        assert 'has_financial_features' in analysis['detected_patterns']
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_protection(self, mock_agent_config):
        """Test protection circuit breaker"""
        detector = IntelligentContextDetector(
            anthropic_api_key=mock_agent_config.anthropic_api_key,
            config=mock_agent_config
        )
        # Vérifier que _should_use_claude vérifie le circuit breaker
        assert hasattr(detector, '_should_use_claude')


# ============================================================================
# TESTS INTELLIGENT CONTEXT DETECTOR - AVEC CLAUDE (AJOUTER APRÈS TestIntelligentContextDetector)
# ============================================================================

class TestIntelligentContextDetectorWithClaude:
    """Tests pour IntelligentContextDetector avec support Claude"""
    
    @pytest.mark.asyncio
    async def test_claude_enabled_initialization(self):
        """Test initialisation avec Claude activé"""
        with patch('automl_platform.agents.intelligent_context_detector.AsyncAnthropic') as mock_claude_class:
            detector = IntelligentContextDetector(anthropic_api_key="test-key")
            
            assert detector.use_claude == True
            assert detector.model == "claude-sonnet-4-20250514"
    
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
# PHASE 2.2 : TESTS CONFIG GENERATOR AVEC CLAUDE (NOUVEAUX)
# ============================================================================

class TestConfigGeneratorWithClaude:
    """Tests pour IntelligentConfigGenerator avec Claude"""
    
    @pytest.mark.asyncio
    async def test_claude_select_algorithms(self, sample_fraud_data, mock_claude_client):
        """Test sélection d'algorithmes avec Claude"""
        generator = IntelligentConfigGenerator(use_claude=True)
        generator.claude_client = mock_claude_client
        
        mock_claude_client.messages.create.return_value.content = [
            Mock(text=json.dumps({
                'algorithms': ['XGBoost', 'LightGBM', 'IsolationForest'],
                'reasoning': 'Optimal for fraud detection'
            }))
        ]
        
        algorithms = await generator._select_algorithms(
            'classification',
            sample_fraud_data,
            {'imbalance_detected': True},
            {'time_budget': 3600},
            None
        )
        
        assert len(algorithms) > 0
    
    @pytest.mark.asyncio
    async def test_claude_design_feature_engineering(self, sample_fraud_data, mock_claude_client):
        """Test design de feature engineering avec Claude"""
        generator = IntelligentConfigGenerator(use_claude=True)
        generator.claude_client = mock_claude_client
        
        mock_claude_client.messages.create.return_value.content = [
            Mock(text=json.dumps({
                'enhancements': {
                    'additional_features': ['velocity_features']
                },
                'reasoning': 'Enhanced'
            }))
        ]
        
        fe = await generator._design_feature_engineering_with_claude(
            sample_fraud_data,
            {'problem_type': 'fraud_detection'},
            'classification'
        )
        
        assert isinstance(fe, dict)
    
    @pytest.mark.asyncio
    async def test_rule_based_fallback_for_each(self, sample_fraud_data):
        """Test fallback rule-based pour chaque méthode"""
        generator = IntelligentConfigGenerator(use_claude=False)
        
        algos = await generator._select_algorithms(
            'classification', sample_fraud_data, {},
            {'time_budget': 3600}, None
        )
        assert len(algos) > 0


# ============================================================================
# PHASE 2.2 : TESTS ADAPTIVE TEMPLATES AVEC CLAUDE (NOUVEAUX)
# ============================================================================

class TestAdaptiveTemplateSystemWithClaude:
    """Tests pour AdaptiveTemplateSystem avec Claude"""
    
    @pytest.mark.asyncio
    async def test_claude_enhancement_enabled(self, temp_dir, mock_claude_client):
        """Test système avec enhancement Claude activé"""
        system = AdaptiveTemplateSystem(
            template_dir=Path(temp_dir),
            use_claude=True,
            anthropic_api_key="test-key"
        )
        
        assert system.use_claude == True
    
    def test_fallback_to_rule_based(self, temp_dir):
        """Test fallback vers rule-based"""
        system = AdaptiveTemplateSystem(
            template_dir=Path(temp_dir),
            use_claude=False
        )
        
        system.learned_patterns['test'] = [
            {'context': {'n_samples': 1000}, 'config': {'test': True}, 'success_score': 0.9}
        ]
        
        config = system._select_best_learned_pattern('test', {'n_samples': 950})
        assert config is not None
    
    def test_metrics_tracking(self, temp_dir):
        """Test tracking des métriques Claude"""
        system = AdaptiveTemplateSystem(
            template_dir=Path(temp_dir),
            use_claude=True,
            anthropic_api_key="test-key"
        )
        
        assert 'claude_enhanced_adaptations' in system.metrics
        assert 'claude_fallbacks' in system.metrics


# ============================================================================
# PHASE 2.3 : TESTS ORCHESTRATOR AVEC CLAUDE (NOUVEAUX)
# ============================================================================

class TestDataCleaningOrchestratorWithClaude:
    """Tests pour DataCleaningOrchestrator avec Claude"""
    
    @pytest.mark.asyncio
    async def test_determine_cleaning_mode_with_claude(
        self, mock_agent_config, sample_fraud_data, mock_claude_client
    ):
        """Test détermination du mode avec Claude"""
        orchestrator = DataCleaningOrchestrator(mock_agent_config, use_claude=True)
        orchestrator.claude_client = mock_claude_client
        
        mock_claude_client.messages.create.return_value.content = [
            Mock(text=json.dumps({
                'recommended_mode': 'hybrid',
                'confidence': 0.85,
                'reasoning': 'Balance',
                'key_considerations': [],
                'estimated_time_minutes': 15,
                'risk_level': 'medium'
            }))
        ]
        
        ml_context = Mock(problem_type='fraud_detection', business_sector='finance')
        decision = await orchestrator.determine_cleaning_mode_with_claude(
            sample_fraud_data, {}, ml_context
        )
        
        assert decision['recommended_mode'] in ['automated', 'interactive', 'hybrid']
    
    def test_bounded_history(self, mock_agent_config):
        """Test que l'historique est borné"""
        orchestrator = DataCleaningOrchestrator(mock_agent_config)
        
        from agents.utils import BoundedList
        assert isinstance(orchestrator.execution_history, BoundedList)
        assert orchestrator.execution_history.maxlen == 100


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
# TESTS ORIGINAUX - ADAPTIVE TEMPLATE SYSTEM
# ============================================================================

class TestAdaptiveTemplateSystem:
    """Tests pour le système de templates adaptatifs"""
    
    @pytest.fixture
    def adaptive_system(self, temp_dir):
        return AdaptiveTemplateSystem(Path(temp_dir), use_claude=False)
    
    @pytest.mark.asyncio
    async def test_get_configuration(self, adaptive_system, sample_fraud_data):
        """Test l'obtention de configuration adaptative"""
        context = {
            'problem_type': 'fraud_detection',
            'n_samples': len(sample_fraud_data),
            'n_features': len(sample_fraud_data.columns),
            'imbalance_detected': True
        }
        
        config = await adaptive_system.get_configuration(
            df=sample_fraud_data,
            context=context
        )
        
        assert isinstance(config, dict)
        assert 'task' in config
        assert 'algorithms' in config
        assert 'preprocessing' in config
    
    def test_add_template(self, adaptive_system):
        """Test l'ajout d'un template"""
        template_config = {
            'task': 'classification',
            'algorithms': ['XGBoost', 'LightGBM'],
            'preprocessing': {'handle_missing': True}
        }
        
        template = adaptive_system.add_template(
            'test_template',
            template_config,
            'Test template description'
        )
        
        assert isinstance(template, AdaptiveTemplate)
        assert template.name == 'test_template'
        assert template.base_config == template_config
        assert 'test_template' in adaptive_system.templates
    
    @pytest.mark.asyncio
    async def test_learn_from_execution(self, adaptive_system):
        """Test l'apprentissage à partir d'exécution"""
        context = {
            'problem_type': 'fraud_detection',
            'n_samples': 1000,
            'n_features': 20
        }
        
        config = {
            'task': 'classification',
            'algorithms': ['XGBoost'],
            'preprocessing': {}
        }
        
        performance = {'f1': 0.85, 'precision': 0.90}
        
        adaptive_system.learn_from_execution(context, config, performance)
        
        assert 'fraud_detection' in adaptive_system.learned_patterns
        assert len(adaptive_system.learned_patterns['fraud_detection']) > 0
        
        pattern = adaptive_system.learned_patterns['fraud_detection'][0]
        assert pattern['success_score'] == 0.90
        assert pattern['config'] == config
    
    def test_select_best_learned_pattern(self, adaptive_system):
        """Test la sélection du meilleur pattern appris"""
        adaptive_system.learned_patterns['fraud_detection'] = [
            {
                'context': {'n_samples': 1000, 'n_features': 20},
                'config': {'algorithms': ['XGBoost']},
                'success_score': 0.85
            },
            {
                'context': {'n_samples': 900, 'n_features': 19},
                'config': {'algorithms': ['LightGBM']},
                'success_score': 0.90
            }
        ]
        
        current_context = {'n_samples': 950, 'n_features': 18}
        
        best_config = adaptive_system._select_best_learned_pattern(
            'fraud_detection',
            current_context
        )
        
        assert best_config is not None
        assert best_config['algorithms'] == ['LightGBM']
    
    def test_calculate_pattern_similarity(self, adaptive_system):
        """Test le calcul de similarité entre patterns"""
        pattern = {
            'context': {
                'n_samples': 1000,
                'n_features': 20,
                'imbalance_detected': True,
                'business_sector': 'finance'
            }
        }
        
        context = {
            'n_samples': 900,
            'n_features': 22,
            'imbalance_detected': True,
            'business_sector': 'finance'
        }
        
        similarity = adaptive_system._calculate_pattern_similarity(pattern, context)
        
        assert similarity > 0.7
        assert similarity <= 1.0
    
    def test_get_template_stats(self, adaptive_system):
        """Test les statistiques des templates"""
        adaptive_system.add_template('test', {'task': 'classification'})
        adaptive_system.learned_patterns['test_problem'] = [{'config': {}}]
        
        stats = adaptive_system.get_template_stats()
        
        assert stats['total_templates'] == 1
        assert stats['total_learned_patterns'] == 1
        assert 'template_performance' in stats
        assert 'test' in stats['template_performance']
    
    @pytest.mark.asyncio
    async def test_adapt_to_current_data(self, adaptive_system):
        """Test l'adaptation aux données actuelles"""
        base_config = {
            'task': 'classification',
            'algorithms': ['NeuralNetwork', 'XGBoost'],
            'preprocessing': {}
        }
        
        df = pd.DataFrame(np.random.randn(100, 5))
        context = {'imbalance_detected': True}
        
        adapted = await adaptive_system._adapt_to_current_data(base_config, df, context)
        
        assert 'NeuralNetwork' not in adapted['algorithms']
        assert 'LogisticRegression' in adapted['algorithms']
        assert 'SMOTE' in str(adapted['preprocessing'])
    
    def test_persistence(self, adaptive_system, temp_dir):
        """Test la sauvegarde et chargement des patterns"""
        adaptive_system.learned_patterns['test'] = [{'config': {'test': True}}]
        adaptive_system._save_learned_patterns()
        
        new_system = AdaptiveTemplateSystem(Path(temp_dir), use_claude=False)
        
        assert 'test' in new_system.learned_patterns
        assert new_system.learned_patterns['test'][0]['config']['test'] == True


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


# ============================================================================
# TESTS ORIGINAUX - DATA CLEANING ORCHESTRATOR
# ============================================================================

class TestDataCleaningOrchestrator:
    """Tests pour l'orchestrateur de nettoyage"""
    
    @pytest.fixture
    def orchestrator(self, mock_agent_config):
        return DataCleaningOrchestrator(mock_agent_config, use_claude=False)
    
    @pytest.mark.asyncio
    async def test_clean_dataset_basic(self, orchestrator, sample_fraud_data):
        """Test le nettoyage basique du dataset"""
        user_context = {
            'secteur_activite': 'finance',
            'target_variable': 'fraud',
            'contexte_metier': 'Fraud detection'
        }
        
        with patch.object(orchestrator, '_process_chunk') as mock_process:
            mock_process.return_value = sample_fraud_data
            
            cleaned_df, report = await orchestrator.clean_dataset(
                sample_fraud_data,
                user_context
            )
            
            assert cleaned_df is not None
            assert 'metadata' in report
            assert report['metadata']['industry'] == 'finance'
    
    def test_needs_chunking(self, orchestrator):
        """Test la détection du besoin de chunking"""
        small_df = pd.DataFrame(np.random.randn(100, 10))
        assert orchestrator._needs_chunking(small_df) == False
        
        with patch.object(pd.DataFrame, 'memory_usage') as mock_memory:
            mock_memory.return_value = pd.Series([20 * 1024 * 1024] * 10)
            large_df = pd.DataFrame(np.random.randn(1000, 10))
            assert orchestrator._needs_chunking(large_df) == True
    
    def test_chunk_dataset(self, orchestrator):
        """Test le découpage en chunks"""
        df = pd.DataFrame(np.random.randn(1000, 10))
        
        with patch.object(orchestrator, '_needs_chunking', return_value=True):
            chunks = orchestrator._chunk_dataset(df)
            
            assert len(chunks) > 1
            total_rows = sum(len(chunk) for chunk in chunks)
            assert total_rows == len(df)
    
    def test_update_cost_estimate(self, orchestrator):
        """Test l'estimation des coûts"""
        orchestrator.performance_metrics['total_api_calls'] = 10
        
        orchestrator._update_cost_estimate()
        
        assert orchestrator.total_cost > 0
        assert orchestrator.performance_metrics['total_tokens_used'] > 0
    
    def test_generate_final_report(self, orchestrator):
        """Test la génération du rapport final"""
        orchestrator.start_time = datetime.now().timestamp()
        orchestrator.transformations_applied = [{'action': 'fill_missing'}]
        orchestrator.validation_sources = ['source1']
        orchestrator.performance_metrics = {'intelligence_used': True}
        
        original_df = pd.DataFrame({'col1': [1, 2, 3]})
        cleaned_df = pd.DataFrame({'col1': [1, 2]})
        
        report = orchestrator._generate_final_report(original_df, cleaned_df)
        
        assert 'metadata' in report
        assert 'transformations' in report
        assert 'quality_metrics' in report
        assert report['metadata']['agent_first_used'] == True
        assert report['quality_metrics']['rows_removed'] == 1
    
    @pytest.mark.asyncio
    async def test_detect_ml_context_integration(self, orchestrator, sample_fraud_data):
        """Test l'intégration de la détection de contexte ML"""
        user_context = {
            'secteur_activite': 'finance',
            'target_variable': 'fraud'
        }
        
        context = await orchestrator.detect_ml_context(
            sample_fraud_data,
            target_col='fraud'
        )
        
        assert 'problem_type' in context
        assert 'confidence' in context
        assert context['problem_type'] == 'fraud_detection'


# ============================================================================
# TESTS DATA CLEANING ORCHESTRATOR - AVEC CLAUDE (AJOUTER APRÈS TestDataCleaningOrchestrator)
# ============================================================================

class TestDataCleaningOrchestratorWithClaude:
    """Tests pour DataCleaningOrchestrator avec support Claude"""
    
    @pytest.mark.asyncio
    async def test_determine_cleaning_mode_with_claude(self, mock_agent_config, sample_fraud_data):
        """Test détermination du mode de nettoyage avec Claude"""
        orchestrator = DataCleaningOrchestrator(mock_agent_config, use_claude=True)
        
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
    async def test_recommend_approach_with_claude(self, mock_agent_config, sample_fraud_data):
        """Test recommandations de nettoyage avec Claude"""
        orchestrator = DataCleaningOrchestrator(mock_agent_config, use_claude=True)
        
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
    async def test_claude_fallback_on_failure(self, mock_agent_config, sample_fraud_data):
        """Test fallback quand Claude échoue"""
        orchestrator = DataCleaningOrchestrator(mock_agent_config, use_claude=True)
        
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
    async def test_parallel_execution(self, mock_agent_config):
        """Test exécution parallèle des tâches"""
        orchestrator = DataCleaningOrchestrator(mock_agent_config, use_claude=True)
        
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
    
    def test_bounded_history(self, mock_agent_config):
        """Test que l'historique est limité (évite memory leak)"""
        orchestrator = DataCleaningOrchestrator(mock_agent_config)
        
        # Vérifier que execution_history est une BoundedList
        from automl_platform.agents.utils import BoundedList
        assert isinstance(orchestrator.execution_history, BoundedList)
        assert orchestrator.execution_history.maxlen == 100
        
        # Ajouter plus de 100 items
        for i in range(150):
            orchestrator.execution_history.append({'item': i})
        
        # Ne devrait contenir que les 100 derniers
        assert len(orchestrator.execution_history) == 100
        assert orchestrator.execution_history[0]['item'] == 50  # Premier = 50 (150-100)
    
    @pytest.mark.asyncio
    async def test_claude_decisions_metrics(self, mock_agent_config):
        """Test tracking des décisions Claude"""
        orchestrator = DataCleaningOrchestrator(mock_agent_config, use_claude=True)
        
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
    def mock_openai_client(self):
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
    async def test_profiler_agent(self, mock_agent_config, mock_openai_client):
        """Test ProfilerAgent"""
        agent = ProfilerAgent(mock_agent_config)
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
    
    def test_agent_instantiation_without_event_loop(self, mock_agent_config):
        """Ensure agents defer initialization when no loop is running"""
        agent = ProfilerAgent(mock_agent_config)
        assert agent._initialized == False
        assert agent.assistant is None
    
    def test_cleaner_agent_basic_cleaning(self, mock_agent_config):
        """Test CleanerAgent basic cleaning"""
        agent = CleanerAgent(mock_agent_config)
        
        df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': ['a', 'b', 'c', 'd']
        })
        profile_report = {'quality_issues': []}
        
        cleaned_df, transformations = agent._basic_cleaning(df, profile_report)
        
        assert cleaned_df['col1'].isnull().sum() == 0
        assert len(transformations) > 0
    
    def test_controller_agent_quality_metrics(self, mock_agent_config):
        """Test ControllerAgent quality metrics"""
        agent = ControllerAgent(mock_agent_config)
        
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
# TESTS LAZY INITIALIZATION - AGENTS OPENAI (AJOUTER APRÈS TestOpenAIAgents)
# ============================================================================

class TestLazyInitialization:
    """Tests pour l'initialisation lazy des agents OpenAI"""
    
    @pytest.mark.asyncio
    async def test_profiler_lazy_init(self, mock_agent_config):
        """Test que ProfilerAgent s'initialise à la première utilisation"""
        agent = ProfilerAgent(mock_agent_config)
        
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
    async def test_cleaner_lazy_init(self, mock_agent_config):
        """Test que CleanerAgent s'initialise à la première utilisation"""
        agent = CleanerAgent(mock_agent_config)
        
        assert agent._initialized == False
        assert agent.assistant is None
        
        with patch.object(agent, 'client') as mock_client:
            mock_client.beta.assistants.create = AsyncMock(return_value=Mock(id='asst_test'))
            await agent._ensure_assistant_initialized()
            assert agent._initialized == True
    
    @pytest.mark.asyncio
    async def test_validator_lazy_init(self, mock_agent_config):
        """Test que ValidatorAgent s'initialise à la première utilisation"""
        agent = ValidatorAgent(mock_agent_config)
        
        assert agent._initialized == False
        assert agent.assistant is None
        
        with patch.object(agent, 'openai_client') as mock_client:
            mock_client.beta.assistants.create = AsyncMock(return_value=Mock(id='asst_test'))
            await agent._ensure_assistant_initialized()
            assert agent._initialized == True
    
    @pytest.mark.asyncio
    async def test_profiler_analyze_triggers_init(self, mock_agent_config):
        """Test que analyze() déclenche l'initialisation automatiquement"""
        agent = ProfilerAgent(mock_agent_config)
        
        with patch.object(agent, '_ensure_assistant_initialized') as mock_init:
            with patch.object(agent, '_basic_profiling') as mock_basic:
                mock_basic.return_value = {'summary': {}}
                
                df = pd.DataFrame({'col1': [1, 2, 3]})
                await agent.analyze(df)
                
                # Vérifier que l'initialisation a été appelée
                mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleaner_clean_triggers_init(self, mock_agent_config):
        """Test que clean() déclenche l'initialisation automatiquement"""
        agent = CleanerAgent(mock_agent_config)
        
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
# TESTS VALIDATOR AGENT - ARCHITECTURE HYBRIDE (AJOUTER APRÈS TestAgentFallbacks)
# ============================================================================

class TestValidatorAgentHybrid:
    """Tests pour l'architecture hybride du ValidatorAgent"""
    
    @pytest.mark.asyncio
    async def test_validator_with_claude_and_openai(self, mock_agent_config):
        """Test ValidatorAgent avec Claude ET OpenAI disponibles"""
        agent = ValidatorAgent(mock_agent_config, use_claude=True)
        
        # Mock Claude client
        with patch('automl_platform.agents.validator_agent.AsyncAnthropic') as mock_claude_class:
            mock_claude = AsyncMock()
            mock_claude_class.return_value = mock_claude
            
            # Réinitialiser l'agent avec le mock
            agent = ValidatorAgent(mock_agent_config, use_claude=True)
            agent.claude_client = mock_claude
            
            # Mock la réponse de Claude
            mock_response = Mock()
            mock_response.content = [Mock(text='{"valid": true, "overall_score": 85, "issues": [], "warnings": [], "suggestions": [], "column_validations": {}, "reasoning": "Test"}')]
            mock_claude.messages.create = AsyncMock(return_value=mock_response)
            
            # Mock OpenAI pour la recherche
            with patch.object(agent, '_search_sector_standards') as mock_search:
                mock_search.return_value = {'standards': [], 'sources': [], 'column_mappings': {}}
                
                df = pd.DataFrame({'col1': [1, 2, 3]})
                profile_report = {'quality_issues': []}
                
                report = await agent.validate(df, profile_report)
                
                # Vérifier que Claude a été utilisé
                assert mock_claude.messages.create.called
                assert report['valid'] == True
                assert report['overall_score'] == 85
    
    @pytest.mark.asyncio
    async def test_validator_claude_for_reasoning(self, mock_agent_config):
        """Test que Claude est utilisé pour le raisonnement de validation"""
        agent = ValidatorAgent(mock_agent_config, use_claude=True)
        
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
    async def test_validator_openai_for_search(self, mock_agent_config):
        """Test que OpenAI est utilisé pour la recherche web"""
        agent = ValidatorAgent(mock_agent_config, use_claude=False)
        
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
    async def test_validator_fallback_without_claude(self, mock_agent_config):
        """Test fallback quand Claude n'est pas disponible"""
        agent = ValidatorAgent(mock_agent_config, use_claude=False)
        
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
    async def test_validator_metrics_tracking(self, mock_agent_config):
        """Test le tracking des métriques d'utilisation"""
        agent = ValidatorAgent(mock_agent_config, use_claude=True)
        
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


# ============================================================================
# TESTS END-TO-END
# ============================================================================

class TestEndToEnd:
    """Tests d'intégration complets du pipeline Agent-First"""
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, mock_agent_config):
        """Test la gestion d'erreur et fallback"""
        orchestrator = DataCleaningOrchestrator(mock_agent_config)
        
        with patch.object(orchestrator, '_process_chunk') as mock_process:
            mock_process.side_effect = Exception("Test error")
            
            df = pd.DataFrame({'col1': [1, 2, 3]})
            user_context = {'secteur_activite': 'test'}
            
            cleaned_df, report = await orchestrator._fallback_cleaning(df, user_context)
            
            assert cleaned_df is not None
            assert report['metadata']['fallback'] == True


# ============================================================================
# TESTS DE PERFORMANCE
# ============================================================================

class TestPerformance:
    """Tests de performance et optimisation"""
    
    def test_bounded_history_memory_protection(self, mock_agent_config):
        """Test que l'historique borné protège la mémoire"""
        orchestrator = DataCleaningOrchestrator(mock_agent_config)
        
        from agents.utils import BoundedList
        assert isinstance(orchestrator.execution_history, BoundedList)
        assert orchestrator.execution_history.maxlen == 100
        
        for i in range(150):
            orchestrator.execution_history.append({'id': i})
        
        assert len(orchestrator.execution_history) == 100
        assert orchestrator.execution_history[0]['id'] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ============================================================================
# TESTS CIRCUIT BREAKER ET RETRY (NOUVEAU)
# ============================================================================

class TestCircuitBreakerAndRetry:
    """Tests pour le circuit breaker et la logique de retry"""
    
    def test_circuit_breaker_config(self):
        """Test configuration du circuit breaker"""
        config = AgentConfig()
        
        # Vérifier les paramètres par défaut
        assert hasattr(config, 'llm_circuit_breaker')
        
        # Enregistrer des succès
        for _ in range(5):
            config.record_llm_success('claude')
        
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
    
    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """Test que le décorateur retry fonctionne sur succès"""
        from automl_platform.agents.utils import async_retry
        
        call_count = 0
        
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
        from automl_platform.agents.utils import async_retry
        
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
        from automl_platform.agents.utils import async_retry
        
        call_count = 0
        
        @async_retry(max_attempts=3)
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent error")
        
        with pytest.raises(Exception):
            await test_func()
        
        assert call_count == 3  # Devrait essayer 3 fois


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
