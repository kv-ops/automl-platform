"""
Tests complets pour tous les composants Agent-First
===================================================
Tests unitaires et d'intégration pour le système Agent-First.
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
from automl_platform.agents import (
    IntelligentContextDetector,
    MLContext,
    IntelligentConfigGenerator,
    OptimalConfig,
    AdaptiveTemplateSystem,
    AdaptiveTemplate,
    UniversalMLAgent,
    MLPipelineResult,
    KnowledgeBase,
    DataCleaningOrchestrator,
    ProfilerAgent,
    ValidatorAgent,
    CleanerAgent,
    ControllerAgent,
    AgentConfig,
    AgentType,
    YAMLConfigHandler
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
        'fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02])  # 2% fraud rate
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
        'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])  # 20% churn rate
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
        openai_api_key="test-key-123",
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


# ============================================================================
# TESTS INTELLIGENT CONTEXT DETECTOR
# ============================================================================

class TestIntelligentContextDetector:
    """Tests pour le détecteur de contexte ML intelligent"""
    
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
        assert context.temporal_aspect == True  # Has last_login date
    
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
        
        # Vérifier les patterns détectés
        patterns = analysis['detected_patterns']
        assert 'has_id_columns' in patterns  # transaction_id
        assert 'has_financial_features' in patterns  # amount
        assert 'has_temporal_features' in patterns  # transaction_time
    
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
        assert config['primary_metric'] == 'f1'  # Due to imbalance
        assert 'SMOTE' in str(config['preprocessing'])
        assert 'IsolationForest' in config['algorithms']


# ============================================================================
# TESTS INTELLIGENT CONFIG GENERATOR
# ============================================================================

class TestIntelligentConfigGenerator:
    """Tests pour le générateur de configuration intelligent"""
    
    @pytest.fixture
    def generator(self):
        return IntelligentConfigGenerator()
    
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
        assert 'LogisticRegression' in algorithms  # Baseline
    
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
        
        # LightGBM devrait avoir un bon score car il gère les valeurs manquantes
        assert score > 0.1
    
    def test_select_metric(self, generator):
        """Test la sélection de métrique"""
        # Fraud detection
        metric = generator._select_metric(
            'classification',
            {'problem_type': 'fraud_detection'},
            None
        )
        assert metric == 'average_precision'
        
        # Churn with imbalance
        metric = generator._select_metric(
            'classification',
            {'problem_type': 'churn_prediction'},
            None
        )
        assert metric == 'f1'
        
        # Sales forecasting
        metric = generator._select_metric(
            'regression',
            {'problem_type': 'sales_forecasting'},
            None
        )
        assert metric == 'mape'
    
    def test_configure_preprocessing(self, generator):
        """Test la configuration du preprocessing"""
        df = pd.DataFrame({
            'num1': [1, 2, None, 4, 100],  # Has missing and outlier
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
        
        # Vérifier la stratégie d'imputation
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
        # XGBoost
        space = generator._get_search_space('XGBoost', 'classification', 10000)
        assert 'n_estimators' in space
        assert 'max_depth' in space
        assert 'learning_rate' in space
        
        # LightGBM
        space = generator._get_search_space('LightGBM', 'classification', 10000)
        assert 'num_leaves' in space
        assert 'learning_rate' in space
        
        # RandomForest
        space = generator._get_search_space('RandomForest', 'classification', 10000)
        assert 'n_estimators' in space
        assert 'max_depth' in space
    
    def test_setup_cv_strategy(self, generator):
        """Test la stratégie de validation croisée"""
        df = pd.DataFrame(np.random.randn(100, 10))
        
        # Classification
        cv = generator._setup_cv_strategy(df, 'classification', {})
        assert cv['method'] == 'stratified_kfold'
        assert cv['n_folds'] == 5
        
        # Time series
        cv = generator._setup_cv_strategy(
            df, 
            'regression', 
            {'temporal_aspect': True, 'is_time_series': True}
        )
        assert cv['method'] == 'time_series_split'
    
    def test_configure_ensemble(self, generator):
        """Test la configuration d'ensemble"""
        # Peu d'algorithmes
        ensemble = generator._configure_ensemble(['XGBoost', 'LightGBM'], 'classification')
        assert ensemble['enabled'] == False
        
        # Beaucoup d'algorithmes
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
        
        # Adapter pour contraintes de temps
        adapted = generator.adapt_config(base_config, {'time_budget': 600})
        assert adapted.time_budget == 600
        assert adapted.hpo_config['n_iter'] < 50
        
        # Adapter pour contraintes de mémoire
        adapted = generator.adapt_config(base_config, {'memory_limit_gb': 4})
        assert 'NeuralNetwork' not in adapted.algorithms


# ============================================================================
# TESTS ADAPTIVE TEMPLATE SYSTEM
# ============================================================================

class TestAdaptiveTemplateSystem:
    """Tests pour le système de templates adaptatifs"""
    
    @pytest.fixture
    def adaptive_system(self, temp_dir):
        return AdaptiveTemplateSystem(Path(temp_dir))
    
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
        
        # Apprendre du succès
        adaptive_system.learn_from_execution(context, config, performance)
        
        assert 'fraud_detection' in adaptive_system.learned_patterns
        assert len(adaptive_system.learned_patterns['fraud_detection']) > 0
        
        pattern = adaptive_system.learned_patterns['fraud_detection'][0]
        assert pattern['success_score'] == 0.90
        assert pattern['config'] == config
    
    def test_select_best_learned_pattern(self, adaptive_system):
        """Test la sélection du meilleur pattern appris"""
        # Ajouter des patterns
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
        
        # Contexte similaire
        current_context = {'n_samples': 950, 'n_features': 18}
        
        best_config = adaptive_system._select_best_learned_pattern(
            'fraud_detection',
            current_context
        )
        
        assert best_config is not None
        assert best_config['algorithms'] == ['LightGBM']  # Meilleur score
    
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
            'n_samples': 900,  # 90% similar
            'n_features': 22,  # 90% similar
            'imbalance_detected': True,  # Match
            'business_sector': 'finance'  # Match
        }
        
        similarity = adaptive_system._calculate_pattern_similarity(pattern, context)
        
        assert similarity > 0.7
        assert similarity <= 1.0
    
    def test_get_template_stats(self, adaptive_system):
        """Test les statistiques des templates"""
        # Ajouter un template
        adaptive_system.add_template('test', {'task': 'classification'})
        
        # Ajouter un pattern appris
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
        
        # Petites données - devrait retirer NeuralNetwork
        df = pd.DataFrame(np.random.randn(100, 5))
        context = {'imbalance_detected': True}
        
        adapted = await adaptive_system._adapt_to_current_data(base_config, df, context)
        
        assert 'NeuralNetwork' not in adapted['algorithms']
        assert 'LogisticRegression' in adapted['algorithms']  # Ajouté pour petites données
        assert 'SMOTE' in str(adapted['preprocessing'])  # Pour imbalance
    
    def test_persistence(self, adaptive_system, temp_dir):
        """Test la sauvegarde et chargement des patterns"""
        # Ajouter des données
        adaptive_system.learned_patterns['test'] = [{'config': {'test': True}}]
        
        # Sauvegarder
        adaptive_system._save_learned_patterns()
        
        # Créer une nouvelle instance et charger
        new_system = AdaptiveTemplateSystem(Path(temp_dir))
        
        assert 'test' in new_system.learned_patterns
        assert new_system.learned_patterns['test'][0]['config']['test'] == True


# ============================================================================
# TESTS UNIVERSAL ML AGENT
# ============================================================================

class TestUniversalMLAgent:
    """Tests pour l'agent ML universel"""
    
    @pytest.fixture
    def universal_agent(self, mock_agent_config):
        return UniversalMLAgent(mock_agent_config)
    
    @pytest.mark.asyncio
    async def test_understand_problem(self, universal_agent, sample_fraud_data):
        """Test la compréhension du problème"""
        with patch.object(universal_agent.profiler, 'analyze') as mock_analyze:
            mock_analyze.return_value = {'quality_issues': []}
            
            context = await universal_agent.understand_problem(
                sample_fraud_data,
                target_col='fraud'
            )
            
            assert isinstance(context, MLContext)
            assert context.problem_type == 'fraud_detection'
            assert context.confidence > 0
    
    @pytest.mark.asyncio
    async def test_search_ml_best_practices(self, universal_agent):
        """Test la recherche des meilleures pratiques ML"""
        with patch.object(universal_agent.validator, '_web_search') as mock_search:
            mock_search.return_value = {
                'results': [
                    {'snippet': 'State of the art XGBoost for fraud detection'},
                    {'snippet': 'Best practice: use SMOTE for imbalanced data'}
                ],
                'urls': ['http://example.com']
            }
            
            best_practices = await universal_agent.search_ml_best_practices(
                'fraud_detection',
                {'n_samples': 1000, 'n_features': 20}
            )
            
            assert 'recommended_approaches' in best_practices
            assert 'recent_innovations' in best_practices
            assert len(best_practices['sources']) > 0
    
    @pytest.mark.asyncio
    async def test_generate_optimal_config(self, universal_agent):
        """Test la génération de configuration optimale"""
        understanding = MLContext(
            problem_type='fraud_detection',
            confidence=0.9,
            detected_patterns=['fraud_indicator'],
            business_sector='finance',
            temporal_aspect=True,
            imbalance_detected=True,
            recommended_config={},
            reasoning='Test reasoning'
        )
        
        best_practices = {
            'recommended_approaches': ['Use XGBoost', 'Apply SMOTE']
        }
        
        config = await universal_agent.generate_optimal_config(
            understanding,
            best_practices
        )
        
        assert isinstance(config, OptimalConfig)
        assert config.task == 'classification'
        assert len(config.algorithms) > 0
    
    @pytest.mark.asyncio
    async def test_execute_intelligent_cleaning(self, universal_agent, sample_fraud_data):
        """Test le nettoyage intelligent"""
        context = MLContext(
            problem_type='fraud_detection',
            confidence=0.9,
            detected_patterns=[],
            business_sector='finance',
            temporal_aspect=False,
            imbalance_detected=True,
            recommended_config={},
            reasoning='Test'
        )
        
        config = OptimalConfig(
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
        
        with patch.object(universal_agent.cleaning_orchestrator, 'clean_dataset') as mock_clean:
            mock_clean.return_value = (sample_fraud_data, {'quality_metrics': {'improvement': 10}})
            
            cleaned_df, report = await universal_agent.execute_intelligent_cleaning(
                sample_fraud_data,
                context,
                config,
                'fraud'
            )
            
            assert cleaned_df is not None
            assert 'agent_metrics' in report
    
    @pytest.mark.asyncio
    async def test_automl_without_templates(self, universal_agent, sample_fraud_data):
        """Test AutoML complet sans templates"""
        with patch.object(universal_agent, 'understand_problem') as mock_understand:
            mock_understand.return_value = MLContext(
                problem_type='fraud_detection',
                confidence=0.9,
                detected_patterns=[],
                business_sector='finance',
                temporal_aspect=False,
                imbalance_detected=True,
                recommended_config={},
                reasoning='Test'
            )
            
            with patch.object(universal_agent, 'validate_with_standards') as mock_validate:
                mock_validate.return_value = {'issues': []}
                
                with patch.object(universal_agent, 'search_ml_best_practices') as mock_search:
                    mock_search.return_value = {'recommended_approaches': []}
                    
                    with patch.object(universal_agent, 'execute_with_continuous_learning') as mock_execute:
                        mock_execute.return_value = MLPipelineResult(
                            success=True,
                            cleaned_data=sample_fraud_data,
                            config_used=OptimalConfig(
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
                            ),
                            context_detected=mock_understand.return_value,
                            cleaning_report={},
                            performance_metrics={'f1': 0.85},
                            execution_time=10.5
                        )
                        
                        result = await universal_agent.automl_without_templates(
                            sample_fraud_data,
                            target_col='fraud'
                        )
                        
                        assert result.success == True
                        assert result.performance_metrics['f1'] == 0.85
                        assert result.execution_time == 10.5
    
    def test_compute_data_hash(self, universal_agent):
        """Test le calcul de hash des données"""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        
        hash1 = universal_agent._compute_data_hash(df)
        hash2 = universal_agent._compute_data_hash(df)
        
        # Même données = même hash
        assert hash1 == hash2
        
        # Données différentes = hash différent
        df2 = pd.DataFrame({'col1': [1, 2, 4], 'col2': ['a', 'b', 'c']})
        hash3 = universal_agent._compute_data_hash(df2)
        assert hash1 != hash3
    
    def test_get_execution_summary(self, universal_agent):
        """Test le résumé d'exécution"""
        # Ajouter des exécutions
        universal_agent.execution_history = [
            {
                'timestamp': datetime.now(),
                'problem_type': 'fraud_detection',
                'success': True,
                'performance': {'f1': 0.85},
                'execution_time': 10.5,
                'agent_metrics': {'profiler_calls': 1}
            },
            {
                'timestamp': datetime.now(),
                'problem_type': 'churn_prediction',
                'success': False,
                'performance': {},
                'execution_time': 5.2,
                'agent_metrics': {'profiler_calls': 1}
            }
        ]
        
        summary = universal_agent.get_execution_summary()
        
        assert summary['total_executions'] == 2
        assert summary['success_rate'] == 0.5
        assert summary['average_execution_time'] == 7.85
        assert 'fraud_detection' in summary['problem_types_handled']
        assert 'churn_prediction' in summary['problem_types_handled']


# ============================================================================
# TESTS KNOWLEDGE BASE
# ============================================================================

class TestKnowledgeBase:
    """Tests pour la base de connaissances"""
    
    @pytest.fixture
    def knowledge_base(self, temp_dir):
        return KnowledgeBase(Path(temp_dir))
    
    def test_store_and_get_context(self, knowledge_base):
        """Test le stockage et récupération de contexte"""
        data_hash = "test_hash_123"
        context = MLContext(
            problem_type='fraud_detection',
            confidence=0.9,
            detected_patterns=[],
            business_sector='finance',
            temporal_aspect=False,
            imbalance_detected=True,
            recommended_config={},
            reasoning='Test'
        )
        
        # Stocker
        knowledge_base.store_context(data_hash, context)
        
        # Récupérer
        retrieved = knowledge_base.get_cached_context(data_hash)
        
        assert retrieved == context
        assert retrieved.problem_type == 'fraud_detection'
    
    def test_store_and_get_best_practices(self, knowledge_base):
        """Test le stockage des meilleures pratiques"""
        practices = {
            'recommended_approaches': ['XGBoost', 'SMOTE'],
            'recent_innovations': ['AutoML'],
            'common_pitfalls': ['Overfitting'],
            'benchmark_scores': {'f1': 0.85}
        }
        
        knowledge_base.store_best_practices('fraud_detection', practices)
        retrieved = knowledge_base.get_best_practices('fraud_detection')
        
        assert retrieved == practices
    
    def test_store_successful_pattern(self, knowledge_base):
        """Test le stockage de pattern réussi"""
        knowledge_base.store_successful_pattern(
            task='classification',
            config={'algorithms': ['XGBoost']},
            performance={'f1': 0.90}
        )
        
        patterns = knowledge_base.get_similar_successful_patterns('classification')
        
        assert len(patterns) == 1
        assert patterns[0]['task'] == 'classification'
        assert patterns[0]['success_score'] == 0.90
    
    def test_get_similar_patterns_sorting(self, knowledge_base):
        """Test la récupération triée des patterns similaires"""
        # Ajouter plusieurs patterns
        for i, score in enumerate([0.70, 0.90, 0.80, 0.95, 0.85]):
            knowledge_base.store_successful_pattern(
                task='classification',
                config={'id': i},
                performance={'metric': score}
            )
        
        patterns = knowledge_base.get_similar_successful_patterns('classification', limit=3)
        
        assert len(patterns) == 3
        assert patterns[0]['success_score'] == 0.95
        assert patterns[1]['success_score'] == 0.90
        assert patterns[2]['success_score'] == 0.85
    
    def test_persistence(self, knowledge_base, temp_dir):
        """Test la persistance sur disque"""
        # Ajouter des données
        knowledge_base.store_context('hash1', MLContext(
            problem_type='test',
            confidence=0.9,
            detected_patterns=[],
            business_sector=None,
            temporal_aspect=False,
            imbalance_detected=False,
            recommended_config={},
            reasoning='Test'
        ))
        
        knowledge_base.store_best_practices('test', {'approach': 'test'})
        knowledge_base.store_successful_pattern('test', {}, {'metric': 0.9})
        
        # Sauvegarder
        knowledge_base._save_knowledge()
        
        # Créer nouvelle instance et charger
        new_kb = KnowledgeBase(Path(temp_dir))
        
        assert 'hash1' in new_kb.context_cache
        assert 'test' in new_kb.best_practices_cache
        assert len(new_kb.successful_patterns) == 1


# ============================================================================
# TESTS DATA CLEANING ORCHESTRATOR
# ============================================================================

class TestDataCleaningOrchestrator:
    """Tests pour l'orchestrateur de nettoyage"""
    
    @pytest.fixture
    def orchestrator(self, mock_agent_config):
        return DataCleaningOrchestrator(mock_agent_config)
    
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
        # Petit dataset
        small_df = pd.DataFrame(np.random.randn(100, 10))
        assert orchestrator._needs_chunking(small_df) == False
        
        # Grand dataset (simulé)
        with patch.object(pd.DataFrame, 'memory_usage') as mock_memory:
            mock_memory.return_value = pd.Series([20 * 1024 * 1024] * 10)  # 20MB par colonne
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
# TESTS OPENAI AGENTS
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
        
        # Mock assistant creation
        mock_openai_client.beta.assistants.create.return_value = Mock(id='asst_123')
        
        # Mock thread and run
        mock_openai_client.beta.threads.create.return_value = Mock(id='thread_123')
        mock_openai_client.beta.threads.runs.create.return_value = Mock(id='run_123')
        mock_openai_client.beta.threads.runs.retrieve.return_value = Mock(status='completed')
        
        # Mock messages
        mock_message = Mock()
        mock_message.role = 'assistant'
        mock_message.content = [Mock(text=Mock(value='{"quality_issues": []}'))]
        mock_openai_client.beta.threads.messages.list.return_value = Mock(data=[mock_message])
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        report = await agent.analyze(df)
        
        assert isinstance(report, dict)
    
    @pytest.mark.asyncio
    async def test_validator_agent(self, mock_agent_config, mock_openai_client):
        """Test ValidatorAgent"""
        agent = ValidatorAgent(mock_agent_config)
        agent.client = mock_openai_client
        
        mock_openai_client.beta.assistants.create.return_value = Mock(id='asst_456')
        
        df = pd.DataFrame({'col1': [1, 2, 3]})
        profile_report = {'summary': 'test'}
        
        with patch.object(agent, '_search_sector_standards') as mock_search:
            mock_search.return_value = {'standards': [], 'sources': []}
            
            with patch.object(agent, '_handle_run_with_functions') as mock_handle:
                mock_handle.return_value = {'content': '{"valid": true}'}
                
                report = await agent.validate(df, profile_report)

                
                assert isinstance(report, dict)
                assert 'valid' in report

    
    def test_agent_instantiation_without_event_loop(self, mock_agent_config, monkeypatch):
        """Ensure OpenAI-based agents defer initialization when no loop is running"""
        from automl_platform.agents import cleaner_agent, controller_agent, profiler_agent, validator_agent

        class DummyAsyncOpenAI:
            def __init__(self, *args, **kwargs):
                self.beta = MagicMock()

        agents = [
            (cleaner_agent, CleanerAgent),
            (controller_agent, ControllerAgent),
            (profiler_agent, ProfilerAgent),
            (validator_agent, ValidatorAgent),
        ]

        for module, agent_cls in agents:
            monkeypatch.setattr(module, "AsyncOpenAI", DummyAsyncOpenAI)
            instance = agent_cls(mock_agent_config)

            assert instance.client is not None
            assert getattr(instance, "_initialization_task") is None
            assert instance.assistant is None

    def test_cleaner_agent_basic_cleaning(self, mock_agent_config):
        """Test CleanerAgent basic cleaning"""
        agent = CleanerAgent(mock_agent_config)
        
        df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': ['a', 'b', 'c', 'd']
        })
        profile_report = {'quality_issues': []}
        
        cleaned_df, transformations = agent._basic_cleaning(df, profile_report)
        
        assert cleaned_df['col1'].isnull().sum() == 0  # No missing values
        assert len(transformations) > 0
    
    def test_controller_agent_quality_metrics(self, mock_agent_config):
        """Test ControllerAgent quality metrics"""
        agent = ControllerAgent(mock_agent_config)
        
        original_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        
        cleaned_df = pd.DataFrame({
            'col1': [1, 2, 3, 4],  # Une ligne supprimée
            'col2': ['a', 'b', 'c', 'd']
        })
        
        metrics = agent._calculate_quality_metrics(cleaned_df, original_df)
        
        assert 'data_quality' in metrics
        assert 'transformation_impact' in metrics
        assert metrics['data_quality']['rows']['removed'] == 1
        assert metrics['quality_score'] > 0


# ============================================================================
# TESTS YAML CONFIG HANDLER
# ============================================================================

class TestYAMLConfigHandler:
    """Tests pour le gestionnaire de configuration YAML"""
    
    def test_save_cleaning_config(self, temp_dir):
        """Test la sauvegarde de configuration"""
        transformations = [
            {
                'column': 'amount',
                'action': 'normalize_currency',
                'params': {'target_currency': 'EUR'}
            }
        ]
        
        validation_sources = ['https://example.com']
        user_context = {
            'secteur_activite': 'finance',
            'target_variable': 'fraud'
        }
        
        output_path = Path(temp_dir) / 'config.yaml'
        saved_path = YAMLConfigHandler.save_cleaning_config(
            transformations,
            validation_sources,
            user_context,
            output_path=str(output_path)
        )
        
        assert Path(saved_path).exists()
        
        # Charger et vérifier
        with open(saved_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config['metadata']['industry'] == 'finance'
        assert len(config['transformations']) == 1
        assert config['transformations'][0]['action'] == 'normalize_currency'
    
    def test_load_cleaning_config(self, temp_dir):
        """Test le chargement de configuration"""
        # Créer un fichier de config
        config = {
            'metadata': {'industry': 'test'},
            'transformations': [{'action': 'test'}],
            'validation_sources': ['test']
        }
        
        config_path = Path(temp_dir) / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Charger
        loaded = YAMLConfigHandler.load_cleaning_config(str(config_path))
        
        assert loaded == config
    
    def test_apply_transformations(self):
        """Test l'application de transformations"""
        df = pd.DataFrame({
            'col1': [1, 2, None, 4],
            'col2': ['a', 'b', 'c', 'd']
        })
        
        transformations = [
            {
                'column': 'col1',
                'action': 'fill_missing',
                'params': {'method': 'median'}
            }
        ]
        
        df_transformed = YAMLConfigHandler.apply_transformations(df, transformations)
        
        assert df_transformed['col1'].isnull().sum() == 0
        assert df_transformed['col1'].iloc[2] == 2.0  # Median of [1, 2, 4]
    
    def test_validate_config(self):
        """Test la validation de configuration"""
        # Config valide
        valid_config = {
            'metadata': {
                'industry': 'test',
                'target_variable': 'target',
                'processing_date': '2024-01-01'
            },
            'transformations': [
                {'column': 'col1', 'action': 'test'}
            ],
            'validation_sources': ['source1']
        }
        
        assert YAMLConfigHandler.validate_config(valid_config) == True
        
        # Config invalide (manque metadata)
        invalid_config = {
            'transformations': [],
            'validation_sources': []
        }
        
        with pytest.raises(ValueError):
            YAMLConfigHandler.validate_config(invalid_config)


# ============================================================================
# TESTS D'INTÉGRATION END-TO-END
# ============================================================================

class TestEndToEnd:
    """Tests d'intégration complets du pipeline Agent-First"""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_fraud(self, mock_agent_config, sample_fraud_data):
        """Test pipeline complet pour détection de fraude"""
        # Initialiser l'agent universel
        agent = UniversalMLAgent(mock_agent_config)
        
        # Mock les appels OpenAI
        with patch.object(agent.profiler, 'analyze') as mock_profile:
            mock_profile.return_value = {'quality_issues': []}
            
            with patch.object(agent.validator, 'validate') as mock_validate:
                mock_validate.return_value = {'issues': []}
                
                with patch.object(agent.validator, '_web_search') as mock_search:
                    mock_search.return_value = {'results': [], 'urls': []}
                    
                    with patch.object(agent.cleaning_orchestrator, 'clean_dataset') as mock_clean:
                        mock_clean.return_value = (sample_fraud_data, {'quality_metrics': {}})
                        
                        # Exécuter le pipeline
                        result = await agent.automl_without_templates(
                            df=sample_fraud_data,
                            target_col='fraud',
                            user_hints={'sector': 'finance'}
                        )
                        
                        # Vérifications
                        assert result.success == True
                        assert result.context_detected.problem_type == 'fraud_detection'
                        assert result.config_used.task == 'classification'
                        assert 'fraud_detection' in agent.execution_history[0]['problem_type']
    
    @pytest.mark.asyncio
    async def test_learning_and_adaptation(self, mock_agent_config, temp_dir):
        """Test l'apprentissage et l'adaptation continue"""
        # Créer agent avec knowledge base temporaire
        agent = UniversalMLAgent(mock_agent_config)
        agent.knowledge_base = KnowledgeBase(Path(temp_dir))
        
        # Simuler plusieurs exécutions
        for i in range(3):
            context = {
                'problem_type': 'test_problem',
                'n_samples': 1000 + i * 100,
                'n_features': 20
            }
            
            config = {'algorithms': [f'Algo{i}']}
            performance = {'metric': 0.80 + i * 0.05}
            
            agent.adaptive_templates.learn_from_execution(
                context, config, performance
            )
        
        # Vérifier l'apprentissage
        patterns = agent.adaptive_templates.learned_patterns.get('test_problem', [])
        assert len(patterns) == 3
        
        # Le meilleur pattern devrait avoir le score le plus élevé
        best_pattern = patterns[0]
        assert best_pattern['success_score'] == 0.90
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, mock_agent_config):
        """Test la gestion d'erreur et fallback"""
        orchestrator = DataCleaningOrchestrator(mock_agent_config)
        
        # Forcer une erreur dans le processing
        with patch.object(orchestrator, '_process_chunk') as mock_process:
            mock_process.side_effect = Exception("Test error")
            
            df = pd.DataFrame({'col1': [1, 2, 3]})
            user_context = {'secteur_activite': 'test'}
            
            # Devrait utiliser le fallback
            cleaned_df, report = await orchestrator._fallback_cleaning(df, user_context)
            
            assert cleaned_df is not None
            assert report['metadata']['fallback'] == True


# ============================================================================
# TESTS DE PERFORMANCE
# ============================================================================

class TestPerformance:
    """Tests de performance et optimisation"""
    
    def test_knowledge_base_cache_performance(self, temp_dir):
        """Test la performance du cache de la knowledge base"""
        kb = KnowledgeBase(Path(temp_dir))
        
        # Ajouter beaucoup de contextes
        for i in range(100):
            context = MLContext(
                problem_type=f'problem_{i%10}',
                confidence=0.9,
                detected_patterns=[],
                business_sector='test',
                temporal_aspect=False,
                imbalance_detected=False,
                recommended_config={},
                reasoning='Test'
            )
            kb.store_context(f'hash_{i}', context)
        
        # Tester la récupération
        import time
        start = time.time()
        for i in range(100):
            kb.get_cached_context(f'hash_{i}')
        elapsed = time.time() - start
        
        # Devrait être très rapide (< 0.1s pour 100 récupérations)
        assert elapsed < 0.1
    
    def test_adaptive_system_pattern_limit(self, temp_dir):
        """Test la limitation du nombre de patterns appris"""
        system = AdaptiveTemplateSystem(Path(temp_dir))
        
        # Ajouter plus de patterns que la limite
        for i in range(15):
            system.learned_patterns['test'] = system.learned_patterns.get('test', [])
            system.learned_patterns['test'].append({
                'success_score': 0.5 + i * 0.01,
                'config': {'id': i}
            })
        
        # Garder seulement les 10 meilleurs
        system.learned_patterns['test'] = sorted(
            system.learned_patterns['test'],
            key=lambda x: x['success_score'],
            reverse=True
        )[:10]
        
        assert len(system.learned_patterns['test']) == 10
        # Le pire score gardé devrait être >= 0.55
        assert system.learned_patterns['test'][-1]['success_score'] >= 0.55


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
