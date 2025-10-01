"""
Fixtures Centralisées pour Tests AutoML Platform
=================================================
Fixtures réutilisables pour tous les tests du projet.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any, List, Optional
import asyncio

# Import des modules à tester
from automl_platform.agents import (
    AgentConfig,
    AgentType,
    MLContext,
    OptimalConfig
)


# ============================================================================
# FIXTURES - DATA SAMPLES
# ============================================================================

@pytest.fixture
def sample_fraud_data():
    """Dataset simulant un problème de détection de fraude"""
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
    """Dataset simulant un problème de prédiction de churn"""
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
    """Dataset simulant un problème de prévision des ventes"""
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
def sample_clean_data():
    """Dataset propre sans problèmes de qualité"""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric1': np.random.randn(100),
        'numeric2': np.random.randn(100) * 2 + 5,
        'categorical': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })


@pytest.fixture
def sample_problematic_data():
    """Dataset avec de nombreux problèmes de qualité"""
    np.random.seed(42)
    df = pd.DataFrame({
        'numeric_clean': np.random.randn(100),
        'numeric_missing': np.concatenate([np.random.randn(40), [np.nan] * 60]),
        'numeric_outliers': np.concatenate([
            np.random.randn(90),
            [100, -100, 200, -200, 300, 400, 500, -500, 600, -600]
        ]),
        'categorical': np.random.choice(['A', 'B', 'C'], 100),
        'high_cardinality': [f'ID_{i}' for i in range(100)],
        'constant': [1] * 100,
        'target': np.random.choice([0, 1], 100, p=[0.95, 0.05])
    })
    
    # Ajouter des duplicates
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    
    return df


# ============================================================================
# FIXTURES - MOCK CLIENTS (CLAUDE)
# ============================================================================

@pytest.fixture
def mock_claude_client():
    """Mock AsyncAnthropic client avec réponses JSON valides"""
    mock = AsyncMock()
    
    # Mock standard response
    mock.messages.create.return_value = Mock(
        content=[Mock(text=json.dumps({
            "problem_type": "fraud_detection",
            "confidence": 0.9,
            "reasoning": "Strong indicators of fraud detection pattern",
            "detected_patterns": ["fraud_indicator", "financial_features"]
        }))],
        usage=Mock(
            input_tokens=1000,
            output_tokens=500
        )
    )
    
    return mock


@pytest.fixture
def mock_claude_success():
    """Mock Claude avec réponse JSON valide complexe"""
    mock = AsyncMock()
    
    mock.messages.create.return_value = Mock(
        content=[Mock(text=json.dumps({
            "problem_type": "fraud_detection",
            "confidence": 0.92,
            "reasoning": "Detected fraud patterns in transaction data",
            "detected_patterns": ["fraud_indicator", "financial_features", "temporal_features"],
            "alternatives": [
                {"type": "anomaly_detection", "confidence": 0.75, "reason": "High anomaly scores"},
                {"type": "risk_assessment", "confidence": 0.68, "reason": "Risk indicators present"}
            ],
            "recommendations": {
                "algorithms": ["XGBoost", "IsolationForest", "LightGBM"],
                "preprocessing": ["SMOTE", "StandardScaler"],
                "monitoring": ["real_time", "drift_detection"]
            }
        }))],
        usage=Mock(
            input_tokens=1500,
            output_tokens=800
        )
    )
    
    return mock


@pytest.fixture
def mock_claude_failure():
    """Mock Claude qui génère des erreurs"""
    mock = AsyncMock()
    mock.messages.create.side_effect = Exception("Claude API Error: Rate limit exceeded")
    return mock


@pytest.fixture
def mock_claude_timeout():
    """Mock Claude avec timeout"""
    mock = AsyncMock()
    mock.messages.create.side_effect = asyncio.TimeoutError("Request timeout")
    return mock


@pytest.fixture
def mock_claude_invalid_json():
    """Mock Claude avec réponse non-JSON"""
    mock = AsyncMock()
    
    mock.messages.create.return_value = Mock(
        content=[Mock(text="This is not valid JSON, just plain text response")],
        usage=Mock(input_tokens=500, output_tokens=200)
    )
    
    return mock


@pytest.fixture
def mock_claude_partial_json():
    """Mock Claude avec JSON partiel/malformé"""
    mock = AsyncMock()
    
    mock.messages.create.return_value = Mock(
        content=[Mock(text='{"problem_type": "fraud_detection", "confidence": 0.9')],  # JSON incomplet
        usage=Mock(input_tokens=500, output_tokens=200)
    )
    
    return mock


# ============================================================================
# FIXTURES - MOCK CLIENTS (OPENAI)
# ============================================================================

@pytest.fixture
def mock_openai_client():
    """Mock AsyncOpenAI client basique"""
    client = AsyncMock()
    
    # Mock assistants
    client.beta.assistants.create = AsyncMock(return_value=Mock(id='asst_test123'))
    client.beta.assistants.retrieve = AsyncMock(return_value=Mock(id='asst_test123'))
    
    # Mock threads
    client.beta.threads.create = AsyncMock(return_value=Mock(id='thread_test123'))
    client.beta.threads.messages.create = AsyncMock()
    client.beta.threads.messages.list = AsyncMock()
    
    # Mock runs
    client.beta.threads.runs.create = AsyncMock(return_value=Mock(id='run_test123'))
    client.beta.threads.runs.retrieve = AsyncMock()
    client.beta.threads.runs.submit_tool_outputs = AsyncMock()
    
    return client


@pytest.fixture
def mock_openai_with_lazy_init():
    """Mock OpenAI avec support lazy initialization"""
    client = AsyncMock()
    
    # Simuler que l'assistant n'existe pas au début
    client.beta.assistants.retrieve.side_effect = Exception("Assistant not found")
    
    # Mais peut être créé
    client.beta.assistants.create = AsyncMock(return_value=Mock(id='asst_new123'))
    
    # Mock threads et runs
    client.beta.threads.create = AsyncMock(return_value=Mock(id='thread_123'))
    client.beta.threads.runs.create = AsyncMock(return_value=Mock(id='run_123'))
    
    # Mock run status - completed
    client.beta.threads.runs.retrieve = AsyncMock(return_value=Mock(
        status='completed',
        id='run_123'
    ))
    
    # Mock messages
    mock_message = Mock()
    mock_message.role = 'assistant'
    mock_message.content = [Mock(text=Mock(value='{"quality_issues": [], "summary": "Clean data"}'))]
    
    client.beta.threads.messages.list = AsyncMock(return_value=Mock(data=[mock_message]))
    
    return client


@pytest.fixture
def mock_openai_with_function_calling():
    """Mock OpenAI avec support function calling (web search)"""
    client = AsyncMock()
    
    client.beta.assistants.create = AsyncMock(return_value=Mock(id='asst_fc123'))
    client.beta.threads.create = AsyncMock(return_value=Mock(id='thread_fc123'))
    client.beta.threads.runs.create = AsyncMock(return_value=Mock(id='run_fc123'))
    
    # Premier retrieve: requires_action pour web search
    # Deuxième retrieve: completed
    client.beta.threads.runs.retrieve = AsyncMock(side_effect=[
        Mock(
            status='requires_action',
            id='run_fc123',
            required_action=Mock(
                submit_tool_outputs=Mock(
                    tool_calls=[
                        Mock(
                            id='call_123',
                            function=Mock(
                                name='web_search',
                                arguments=json.dumps({'query': 'IFRS standards'})
                            )
                        )
                    ]
                )
            )
        ),
        Mock(status='completed', id='run_fc123')
    ])
    
    # Mock submit tool outputs
    client.beta.threads.runs.submit_tool_outputs = AsyncMock()
    
    # Mock final messages
    mock_message = Mock()
    mock_message.role = 'assistant'
    mock_message.content = [Mock(text=Mock(value='{"valid": true, "standards_found": ["IFRS"]}'))]
    client.beta.threads.messages.list = AsyncMock(return_value=Mock(data=[mock_message]))
    
    return client


# ============================================================================
# FIXTURES - AGENT CONFIG
# ============================================================================

@pytest.fixture
def base_agent_config():
    """Configuration agent basique pour tests"""
    return AgentConfig(
        openai_api_key="test-openai-key-123",
        anthropic_api_key="test-anthropic-key-456",
        openai_model="gpt-4-1106-preview",
        claude_model="claude-sonnet-4-20250514",
        enable_claude=True,
        enable_openai=True,
        max_cost_openai=1.00,
        max_cost_claude=1.00,
        max_cost_total=2.00,
        cache_dir="./test_cache",
        output_dir="./test_output",
        log_file="./test_logs/test.log"
    )


@pytest.fixture
def agent_config_with_claude():
    """AgentConfig configuré avec Claude + circuit breakers"""
    return AgentConfig(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-claude-key",
        enable_claude=True,
        enable_openai=True,
        enable_circuit_breakers=True,
        enable_performance_monitoring=True,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_recovery_timeout=30,
        max_retries=3,
        retry_base_delay=0.5,
        cache_dir="./test_cache",
        output_dir="./test_output"
    )


@pytest.fixture
def agent_config_openai_only():
    """AgentConfig avec seulement OpenAI"""
    return AgentConfig(
        openai_api_key="test-openai-key",
        anthropic_api_key="",
        enable_claude=False,
        enable_openai=True,
        cache_dir="./test_cache",
        output_dir="./test_output"
    )


@pytest.fixture
def agent_config_claude_only():
    """AgentConfig avec seulement Claude"""
    return AgentConfig(
        openai_api_key="",
        anthropic_api_key="test-claude-key",
        enable_claude=True,
        enable_openai=False,
        cache_dir="./test_cache",
        output_dir="./test_output"
    )


@pytest.fixture
def agent_config_no_llm():
    """AgentConfig sans LLM (fallback mode)"""
    return AgentConfig(
        openai_api_key="",
        anthropic_api_key="",
        enable_claude=False,
        enable_openai=False,
        cache_dir="./test_cache",
        output_dir="./test_output"
    )


# ============================================================================
# FIXTURES - ML CONTEXTS
# ============================================================================

@pytest.fixture
def ml_context_fraud():
    """MLContext pour détection de fraude"""
    return MLContext(
        problem_type='fraud_detection',
        confidence=0.92,
        detected_patterns=['fraud_indicator', 'financial_features', 'temporal_features'],
        business_sector='finance',
        temporal_aspect=True,
        imbalance_detected=True,
        recommended_config={
            'task': 'classification',
            'algorithms': ['XGBoost', 'IsolationForest', 'LightGBM'],
            'primary_metric': 'f1'
        },
        reasoning='Strong indicators of fraud detection with temporal patterns and class imbalance',
        alternative_interpretations=[
            {'type': 'anomaly_detection', 'confidence': 0.75, 'reason': 'High anomaly scores'}
        ]
    )


@pytest.fixture
def ml_context_churn():
    """MLContext pour prédiction de churn"""
    return MLContext(
        problem_type='churn_prediction',
        confidence=0.88,
        detected_patterns=['churn_indicator', 'customer_features', 'temporal_features'],
        business_sector='telecom',
        temporal_aspect=True,
        imbalance_detected=True,
        recommended_config={
            'task': 'classification',
            'algorithms': ['XGBoost', 'LightGBM', 'RandomForest'],
            'primary_metric': 'roc_auc'
        },
        reasoning='Customer churn prediction with engagement metrics'
    )


@pytest.fixture
def optimal_config_classification():
    """OptimalConfig pour classification"""
    return OptimalConfig(
        task='classification',
        algorithms=['XGBoost', 'LightGBM', 'RandomForest'],
        primary_metric='f1',
        preprocessing={
            'missing_values': {'strategy': 'iterative_impute'},
            'outliers': {'method': 'clip'},
            'scaling': {'method': 'robust'},
            'handle_imbalance': {'method': 'SMOTE'}
        },
        feature_engineering={
            'automatic': True,
            'polynomial_features': False,
            'interaction_features': True
        },
        hpo_config={
            'method': 'optuna',
            'n_iter': 50,
            'timeout': 3600
        },
        cv_strategy={
            'method': 'stratified_kfold',
            'n_folds': 5
        },
        ensemble_config={
            'enabled': True,
            'method': 'stacking'
        },
        time_budget=3600,
        resource_constraints={
            'memory_limit_gb': 16,
            'n_jobs': -1
        },
        monitoring={
            'enabled': True,
            'drift_detection': True
        },
        confidence=0.95,
        reasoning='Optimized for imbalanced classification with robust preprocessing'
    )


# ============================================================================
# FIXTURES - MEMORY & ENVIRONMENT
# ============================================================================

@pytest.fixture
def memory_constrained_env():
    """Environnement avec contraintes mémoire simulées"""
    
    class MockProcess:
        def memory_info(self):
            return Mock(rss=500 * 1024 * 1024)  # 500 MB
    
    class MockVirtualMemory:
        def __init__(self):
            self.available = 1024 * 1024 * 1024  # 1 GB available
            self.total = 2048 * 1024 * 1024     # 2 GB total
    
    with patch('psutil.Process', return_value=MockProcess()):
        with patch('psutil.virtual_memory', return_value=MockVirtualMemory()):
            yield


@pytest.fixture
def memory_critical_env():
    """Environnement avec mémoire critique (presque OOM)"""
    
    class MockProcess:
        def memory_info(self):
            return Mock(rss=1800 * 1024 * 1024)  # 1.8 GB - critique
    
    class MockVirtualMemory:
        def __init__(self):
            self.available = 200 * 1024 * 1024  # Seulement 200 MB disponible
            self.total = 2048 * 1024 * 1024
    
    with patch('psutil.Process', return_value=MockProcess()):
        with patch('psutil.virtual_memory', return_value=MockVirtualMemory()):
            yield


# ============================================================================
# FIXTURES - FILE SYSTEM
# ============================================================================

@pytest.fixture
def temp_dir():
    """Répertoire temporaire pour tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_cache_dir(temp_dir):
    """Répertoire cache temporaire"""
    cache_path = Path(temp_dir) / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


@pytest.fixture
def temp_output_dir(temp_dir):
    """Répertoire output temporaire"""
    output_path = Path(temp_dir) / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# ============================================================================
# FIXTURES - UTILITIES
# ============================================================================

@pytest.fixture
def mock_datetime():
    """Mock datetime pour tests déterministes"""
    fixed_time = datetime(2024, 1, 15, 10, 30, 0)
    
    class MockDatetime:
        @staticmethod
        def now():
            return fixed_time
        
        @staticmethod
        def fromtimestamp(ts):
            return datetime.fromtimestamp(ts)
    
    with patch('automl_platform.agents.utils.datetime', MockDatetime):
        yield fixed_time


@pytest.fixture
def event_loop():
    """Event loop pour tests async"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# FIXTURES - RESPONSE TEMPLATES
# ============================================================================

@pytest.fixture
def claude_response_templates():
    """Templates de réponses Claude pour différents cas"""
    return {
        'context_detection': {
            "problem_type": "fraud_detection",
            "confidence": 0.9,
            "reasoning": "Clear fraud detection patterns",
            "detected_patterns": ["fraud_indicator", "financial_features"],
            "alternatives": [
                {"type": "anomaly_detection", "confidence": 0.7, "reason": "Anomaly patterns"}
            ]
        },
        'algorithm_selection': {
            "algorithms": ["XGBoost", "LightGBM", "IsolationForest"],
            "reasoning": "Optimal for fraud with imbalance"
        },
        'validation': {
            "valid": True,
            "overall_score": 85,
            "issues": [],
            "warnings": [
                {"column": "amount", "warning": "Some outliers detected", "recommendation": "Consider capping"}
            ],
            "suggestions": [
                {"type": "quality", "suggestion": "Add data validation", "priority": "medium"}
            ],
            "column_validations": {
                "transaction_id": {"valid": True, "issues": [], "standard_compliance": "compliant"}
            },
            "sector_compliance": {
                "compliant": True,
                "missing_standards": [],
                "violations": []
            },
            "reasoning": "Data meets quality standards with minor warnings"
        },
        'cleaning_strategy': {
            "summary": "Apply robust preprocessing with SMOTE for imbalance",
            "priorities": [
                {
                    "issue": "Class imbalance",
                    "priority": "critical",
                    "approach": "SMOTE resampling",
                    "impact": "Improve minority class detection"
                }
            ],
            "risks": [
                {
                    "risk": "Overfitting on synthetic samples",
                    "likelihood": "medium",
                    "mitigation": "Use cross-validation"
                }
            ],
            "expected_improvement": {
                "quality_score_gain": 15,
                "key_metrics": {"f1": 0.85}
            },
            "success_criteria": ["F1 > 0.80", "No data leakage"]
        }
    }


# ============================================================================
# MARKERS CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configuration des markers pytest"""
    config.addinivalue_line("markers", "slow: tests lents (>10s)")
    config.addinivalue_line("markers", "claude: tests nécessitant Claude API")
    config.addinivalue_line("markers", "memory: tests de mémoire")
    config.addinivalue_line("markers", "integration: tests d'intégration")
    config.addinivalue_line("markers", "unit: tests unitaires")
    config.addinivalue_line("markers", "openai: tests nécessitant OpenAI API")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_mock_llm_response(content: Dict[str, Any], tokens_in: int = 1000, tokens_out: int = 500):
    """Helper pour créer des réponses LLM mock cohérentes"""
    return Mock(
        content=[Mock(text=json.dumps(content))],
        usage=Mock(
            input_tokens=tokens_in,
            output_tokens=tokens_out
        )
    )


def assert_dataframe_quality(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1):
    """Helper pour vérifier la qualité basique d'un DataFrame"""
    assert isinstance(df, pd.DataFrame), "Not a DataFrame"
    assert len(df) >= min_rows, f"Too few rows: {len(df)} < {min_rows}"
    assert len(df.columns) >= min_cols, f"Too few columns: {len(df.columns)} < {min_cols}"
    assert not df.empty, "DataFrame is empty"


def assert_ml_context_valid(context: MLContext):
    """Helper pour valider un MLContext"""
    assert isinstance(context, MLContext), "Not an MLContext"
    assert context.problem_type is not None, "problem_type is None"
    assert 0 <= context.confidence <= 1, f"Invalid confidence: {context.confidence}"
    assert isinstance(context.detected_patterns, list), "detected_patterns not a list"
    assert isinstance(context.reasoning, str), "reasoning not a string"
