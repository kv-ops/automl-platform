"""
Intelligent Config Generator for AutoML Platform
================================================
Generates optimal configurations without templates.
Adapts to any ML problem dynamically.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OptimalConfig:
    """Optimal configuration for ML pipeline"""
    # Core settings
    task: str
    algorithms: List[str]
    primary_metric: str
    
    # Preprocessing
    preprocessing: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    
    # Training
    hpo_config: Dict[str, Any]
    cv_strategy: Dict[str, Any]
    ensemble_config: Dict[str, Any]
    
    # Runtime
    time_budget: int
    resource_constraints: Dict[str, Any]
    
    # Monitoring
    monitoring: Dict[str, Any]
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        config = asdict(self)
        config['generated_at'] = self.generated_at.isoformat()
        return config


class IntelligentConfigGenerator:
    """
    Generates optimal ML configurations dynamically.
    No templates needed - pure intelligence based on data and context.
    """
    
    # Algorithm capabilities matrix
    ALGORITHM_CAPABILITIES = {
        'XGBoost': {
            'handles_missing': True,
            'handles_categorical': False,
            'interpretability': 'medium',
            'training_speed': 'fast',
            'prediction_speed': 'fast',
            'memory_usage': 'medium',
            'good_for': ['classification', 'regression', 'ranking'],
            'scales_well': True,
            'handles_imbalance': True
        },
        'LightGBM': {
            'handles_missing': True,
            'handles_categorical': True,
            'interpretability': 'medium',
            'training_speed': 'very_fast',
            'prediction_speed': 'very_fast',
            'memory_usage': 'low',
            'good_for': ['classification', 'regression', 'ranking'],
            'scales_well': True,
            'handles_imbalance': True
        },
        'CatBoost': {
            'handles_missing': True,
            'handles_categorical': True,
            'interpretability': 'medium',
            'training_speed': 'medium',
            'prediction_speed': 'fast',
            'memory_usage': 'medium',
            'good_for': ['classification', 'regression', 'ranking'],
            'scales_well': True,
            'handles_imbalance': True
        },
        'RandomForest': {
            'handles_missing': False,
            'handles_categorical': False,
            'interpretability': 'high',
            'training_speed': 'medium',
            'prediction_speed': 'medium',
            'memory_usage': 'high',
            'good_for': ['classification', 'regression'],
            'scales_well': False,
            'handles_imbalance': False
        },
        'LogisticRegression': {
            'handles_missing': False,
            'handles_categorical': False,
            'interpretability': 'very_high',
            'training_speed': 'very_fast',
            'prediction_speed': 'very_fast',
            'memory_usage': 'low',
            'good_for': ['classification'],
            'scales_well': True,
            'handles_imbalance': False
        },
        'NeuralNetwork': {
            'handles_missing': False,
            'handles_categorical': False,
            'interpretability': 'low',
            'training_speed': 'slow',
            'prediction_speed': 'fast',
            'memory_usage': 'high',
            'good_for': ['classification', 'regression', 'ranking'],
            'scales_well': True,
            'handles_imbalance': False
        },
        'IsolationForest': {
            'handles_missing': False,
            'handles_categorical': False,
            'interpretability': 'low',
            'training_speed': 'fast',
            'prediction_speed': 'fast',
            'memory_usage': 'medium',
            'good_for': ['anomaly_detection'],
            'scales_well': True,
            'handles_imbalance': False
        },
        'Prophet': {
            'handles_missing': True,
            'handles_categorical': False,
            'interpretability': 'high',
            'training_speed': 'medium',
            'prediction_speed': 'fast',
            'memory_usage': 'low',
            'good_for': ['time_series_forecasting'],
            'scales_well': True,
            'handles_imbalance': False
        },
        'ARIMA': {
            'handles_missing': False,
            'handles_categorical': False,
            'interpretability': 'high',
            'training_speed': 'fast',
            'prediction_speed': 'fast',
            'memory_usage': 'low',
            'good_for': ['time_series_forecasting'],
            'scales_well': False,
            'handles_imbalance': False
        }
    }
    
    # Metric optimization strategies
    METRIC_STRATEGIES = {
        'accuracy': {'focus': 'overall_correctness', 'threshold_optimization': False},
        'precision': {'focus': 'minimize_false_positives', 'threshold_optimization': True},
        'recall': {'focus': 'minimize_false_negatives', 'threshold_optimization': True},
        'f1': {'focus': 'balance', 'threshold_optimization': True},
        'roc_auc': {'focus': 'ranking_quality', 'threshold_optimization': False},
        'mape': {'focus': 'percentage_error', 'threshold_optimization': False},
        'rmse': {'focus': 'squared_error', 'threshold_optimization': False},
        'ndcg': {'focus': 'ranking_quality', 'threshold_optimization': False},
        'map': {'focus': 'ranking_precision', 'threshold_optimization': False}
    }
    
    def __init__(self):
        """Initialize the config generator"""
        self.generated_configs = []
        self.performance_history = []
        
    async def generate_config(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> OptimalConfig:
        """
        Generate optimal configuration based on data and context.
        
        This is the core intelligence - creates perfect configs from scratch!
        """
        logger.info("🔧 Generating optimal configuration...")
        
        # Default constraints
        if constraints is None:
            constraints = {
                'time_budget': 3600,
                'memory_limit_gb': 16,
                'interpretability_required': False,
                'real_time_scoring': False
            }
        
        # Step 1: Determine task type
        task = self._determine_task(context, df)
        
        # Step 2: Select optimal algorithms
        algorithms = await self._select_algorithms(
            task, df, context, constraints, user_preferences
        )
        
        # Step 3: Choose primary metric
        primary_metric = self._select_metric(task, context, user_preferences)
        
        # Step 4: Configure preprocessing
        preprocessing = self._configure_preprocessing(df, context, task)
        
        # Step 5: Design feature engineering
        feature_engineering = self._design_feature_engineering(df, context, task)
        
        # Step 6: Configure HPO
        hpo_config = self._configure_hpo(
            algorithms, df, constraints['time_budget'], task
        )
        
        # Step 7: Setup CV strategy
        cv_strategy = self._setup_cv_strategy(df, task, context)
        
        # Step 8: Configure ensemble
        ensemble_config = self._configure_ensemble(algorithms, task)
        
        # Step 9: Setup monitoring
        monitoring = self._setup_monitoring(task, context, constraints)
        
        # Step 10: Generate reasoning
        reasoning = self._generate_config_reasoning(
            task, algorithms, primary_metric, constraints
        )
        
        config = OptimalConfig(
            task=task,
            algorithms=algorithms,
            primary_metric=primary_metric,
            preprocessing=preprocessing,
            feature_engineering=feature_engineering,
            hpo_config=hpo_config,
            cv_strategy=cv_strategy,
            ensemble_config=ensemble_config,
            time_budget=constraints['time_budget'],
            resource_constraints={
                'memory_limit_gb': constraints.get('memory_limit_gb', 16),
                'n_jobs': constraints.get('n_jobs', -1),
                'gpu_enabled': constraints.get('gpu_enabled', False)
            },
            monitoring=monitoring,
            confidence=0.95,
            reasoning=reasoning
        )
        
        # Store for learning
        self.generated_configs.append(config)
        
        return config
    
    def _determine_task(self, context: Dict[str, Any], df: pd.DataFrame) -> str:
        """Determine the ML task type"""
        problem_type = context.get('problem_type', 'unknown')
        
        task_mapping = {
            'churn_prediction': 'classification',
            'fraud_detection': 'classification',
            'credit_scoring': 'classification',
            'customer_segmentation': 'clustering',
            'sales_forecasting': 'regression',
            'demand_prediction': 'regression',
            'recommendation_system': 'ranking',
            'anomaly_detection': 'anomaly_detection',
            'predictive_maintenance': 'classification',
            'time_series_forecasting': 'time_series'
        }
        
        task = task_mapping.get(problem_type, 'auto')
        
        # If auto, infer from target
        if task == 'auto' and context.get('target_variable'):
            target_col = context['target_variable']
            if target_col in df.columns:
                if df[target_col].nunique() == 2:
                    task = 'classification'
                elif df[target_col].nunique() < 10:
                    task = 'classification'
                else:
                    task = 'regression'
        
        return task
    
    async def _select_algorithms(
        self,
        task: str,
        df: pd.DataFrame,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Select optimal algorithms based on multiple factors"""
        
        suitable_algorithms = []
        
        # Filter by task compatibility
        for algo, capabilities in self.ALGORITHM_CAPABILITIES.items():
            if task in capabilities['good_for']:
                suitable_algorithms.append(algo)
        
        # Score algorithms based on data characteristics
        algo_scores = {}
        for algo in suitable_algorithms:
            score = self._score_algorithm(
                algo, df, context, constraints, user_preferences
            )
            algo_scores[algo] = score
        
        # Sort by score and select top algorithms
        sorted_algos = sorted(algo_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N algorithms based on time budget
        if constraints['time_budget'] < 600:  # < 10 minutes
            n_algorithms = 3
        elif constraints['time_budget'] < 3600:  # < 1 hour
            n_algorithms = 5
        else:
            n_algorithms = 7
        
        selected = [algo for algo, _ in sorted_algos[:n_algorithms]]
        
        # Always include a baseline
        if task == 'classification' and 'LogisticRegression' not in selected:
            selected.append('LogisticRegression')
        elif task == 'regression' and 'Ridge' not in selected:
            selected.append('Ridge')
        
        return selected
    
    def _score_algorithm(
        self,
        algorithm: str,
        df: pd.DataFrame,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]]
    ) -> float:
        """Score an algorithm based on multiple criteria"""
        
        capabilities = self.ALGORITHM_CAPABILITIES[algorithm]
        score = 0.0
        
        # Data size considerations
        n_samples = len(df)
        n_features = len(df.columns)
        
        if n_samples > 100000 and capabilities['scales_well']:
            score += 0.2
        elif n_samples < 1000 and capabilities['training_speed'] == 'very_fast':
            score += 0.15
        
        # Missing data handling
        if df.isnull().any().any() and capabilities['handles_missing']:
            score += 0.15
        
        # Categorical data handling
        has_categorical = len(df.select_dtypes(include=['object']).columns) > 0
        if has_categorical and capabilities['handles_categorical']:
            score += 0.15
        
        # Interpretability requirements
        if constraints.get('interpretability_required'):
            if capabilities['interpretability'] in ['very_high', 'high']:
                score += 0.25
            elif capabilities['interpretability'] == 'medium':
                score += 0.1
        
        # Speed requirements
        if constraints.get('real_time_scoring'):
            if capabilities['prediction_speed'] in ['very_fast', 'fast']:
                score += 0.15
        
        # Memory constraints
        memory_limit = constraints.get('memory_limit_gb', 16)
        if memory_limit < 8:
            if capabilities['memory_usage'] == 'low':
                score += 0.15
            elif capabilities['memory_usage'] == 'medium':
                score += 0.05
        
        # Imbalance handling
        if context.get('imbalance_detected') and capabilities['handles_imbalance']:
            score += 0.2
        
        # User preferences
        if user_preferences:
            if algorithm in user_preferences.get('preferred_algorithms', []):
                score += 0.3
            if algorithm in user_preferences.get('excluded_algorithms', []):
                score = 0  # Exclude completely
        
        return score
    
    def _select_metric(
        self,
        task: str,
        context: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]]
    ) -> str:
        """Select the primary optimization metric"""
        
        # User preference takes priority
        if user_preferences and 'primary_metric' in user_preferences:
            return user_preferences['primary_metric']
        
        problem_type = context.get('problem_type', 'unknown')
        
        # Problem-specific defaults
        metric_defaults = {
            'fraud_detection': 'roc_auc',
            'churn_prediction': 'f1',
            'credit_scoring': 'roc_auc',
            'sales_forecasting': 'mape',
            'recommendation_system': 'ndcg',
            'anomaly_detection': 'f1',
            'predictive_maintenance': 'recall'
        }
        
        if problem_type in metric_defaults:
            return metric_defaults[problem_type]
        
        # Task-based defaults
        if task == 'classification':
            if context.get('imbalance_detected'):
                return 'f1'
            return 'roc_auc'
        elif task == 'regression':
            return 'rmse'
        elif task == 'ranking':
            return 'ndcg'
        elif task == 'clustering':
            return 'silhouette'
        
        return 'auto'
    
    def _configure_preprocessing(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        task: str
    ) -> Dict[str, Any]:
        """Configure intelligent preprocessing"""
        
        preprocessing = {
            'missing_values': {},
            'outliers': {},
            'scaling': {},
            'encoding': {},
            'feature_selection': {}
        }
        
        # Missing values strategy
        total_cells = max(df.shape[0] * df.shape[1], 1)
        missing_ratio = df.isnull().sum().sum() / total_cells
        if missing_ratio > 0:
            if missing_ratio < 0.05:
                preprocessing['missing_values'] = {
                    'strategy': 'simple_impute',
                    'numeric_strategy': 'median',
                    'categorical_strategy': 'mode'
                }
            elif missing_ratio < 0.2:
                preprocessing['missing_values'] = {
                    'strategy': 'iterative_impute',
                    'max_iter': 10
                }
            else:
                preprocessing['missing_values'] = {
                    'strategy': 'advanced',
                    'method': 'missforest',
                    'drop_threshold': 0.5
                }
        
        # Outlier handling
        problem_type = context.get('problem_type')
        if problem_type != 'fraud_detection':  # Keep outliers for fraud
            preprocessing['outliers'] = {
                'method': 'iqr',
                'factor': 1.5,
                'strategy': 'clip'  # clip, remove, or transform
            }
        else:
            preprocessing['outliers'] = {'method': 'none'}
        
        # Scaling strategy
        if task in ['classification', 'regression']:
            # Check for outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            has_outliers = False
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                if outliers > len(df) * 0.05:
                    has_outliers = True
                    break
            
            if has_outliers:
                preprocessing['scaling'] = {'method': 'robust'}
            else:
                preprocessing['scaling'] = {'method': 'standard'}
        
        # Encoding strategy
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_cols) > 0:
            high_cardinality = []
            low_cardinality = []
            
            for col in categorical_cols:
                if df[col].nunique() > 20:
                    high_cardinality.append(col)
                else:
                    low_cardinality.append(col)
            
            preprocessing['encoding'] = {
                'low_cardinality': {
                    'columns': low_cardinality,
                    'method': 'onehot'
                },
                'high_cardinality': {
                    'columns': high_cardinality,
                    'method': 'target_encoding' if task == 'classification' else 'frequency_encoding'
                }
            }
        
        # Feature selection
        if len(df.columns) > 100:
            preprocessing['feature_selection'] = {
                'method': 'mutual_info',
                'max_features': 50,
                'threshold': 0.01
            }
        
        return preprocessing
    
    def _design_feature_engineering(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        task: str
    ) -> Dict[str, Any]:
        """Design feature engineering strategy"""
        
        feature_engineering = {
            'automatic': True,
            'polynomial_features': False,
            'interaction_features': False,
            'datetime_features': False,
            'text_features': False,
            'domain_specific': {}
        }
        
        problem_type = context.get('problem_type')
        
        # Polynomial features for small datasets
        if len(df) < 10000 and len(df.columns) < 20:
            feature_engineering['polynomial_features'] = {
                'degree': 2,
                'include_bias': False,
                'interaction_only': False
            }
        
        # Interaction features for medium datasets
        if len(df) < 100000 and len(df.columns) < 50:
            feature_engineering['interaction_features'] = {
                'max_features': 10,
                'method': 'multiplication'
            }
        
        # Datetime features if temporal columns exist
        if context.get('temporal_aspect'):
            feature_engineering['datetime_features'] = {
                'extract': ['year', 'month', 'day', 'dayofweek', 'hour'],
                'cyclical_encoding': True,
                'holiday_features': problem_type in ['sales_forecasting', 'demand_prediction']
            }
        
        # Problem-specific features
        if problem_type == 'churn_prediction':
            feature_engineering['domain_specific'] = {
                'recency_features': True,
                'frequency_features': True,
                'monetary_features': True,
                'trend_features': True
            }
        elif problem_type == 'fraud_detection':
            feature_engineering['domain_specific'] = {
                'velocity_features': True,
                'frequency_encoding': True,
                'time_since_last': True,
                'unusual_pattern_flags': True
            }
        elif problem_type == 'sales_forecasting':
            feature_engineering['domain_specific'] = {
                'lag_features': [1, 7, 30, 365],
                'rolling_features': [7, 30, 90],
                'seasonal_features': True,
                'trend_features': True
            }
        
        return feature_engineering
    
    def _configure_hpo(
        self,
        algorithms: List[str],
        df: pd.DataFrame,
        time_budget: int,
        task: str
    ) -> Dict[str, Any]:
        """Configure hyperparameter optimization"""
        
        n_samples = len(df)
        n_algorithms = len(algorithms)
        
        # Determine HPO method based on time and data
        if time_budget < 300:  # < 5 minutes
            method = 'random'
            n_iter = 10
        elif n_samples < 1000:
            method = 'grid'
            n_iter = 20
        elif time_budget < 1800:  # < 30 minutes
            method = 'optuna'
            n_iter = 30
        else:
            method = 'optuna'
            n_iter = 50
        
        # Adjust iterations based on number of algorithms
        n_iter_per_algo = n_iter // max(n_algorithms, 1)
        
        hpo_config = {
            'method': method,
            'n_iter': n_iter,
            'n_iter_per_algo': n_iter_per_algo,
            'timeout': time_budget * 0.7,  # 70% of budget for HPO
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 0.001
            },
            'pruning': method == 'optuna',
            'warm_start': True,
            'n_jobs': -1
        }
        
        # Algorithm-specific search spaces
        search_spaces = {}
        for algo in algorithms:
            search_spaces[algo] = self._get_search_space(algo, task, n_samples)
        
        hpo_config['search_spaces'] = search_spaces
        
        return hpo_config
    
    def _get_search_space(
        self,
        algorithm: str,
        task: str,
        n_samples: int
    ) -> Dict[str, Any]:
        """Get algorithm-specific search space"""
        
        # Adaptive search spaces based on data size
        if algorithm == 'XGBoost':
            return {
                'n_estimators': [100, 500] if n_samples < 10000 else [100, 1000],
                'max_depth': [3, 10],
                'learning_rate': [0.01, 0.3],
                'subsample': [0.6, 1.0],
                'colsample_bytree': [0.6, 1.0],
                'gamma': [0, 0.3],
                'reg_alpha': [0, 1],
                'reg_lambda': [0, 1]
            }
        elif algorithm == 'LightGBM':
            return {
                'n_estimators': [100, 500] if n_samples < 10000 else [100, 1000],
                'num_leaves': [31, 255],
                'learning_rate': [0.01, 0.3],
                'feature_fraction': [0.6, 1.0],
                'bagging_fraction': [0.6, 1.0],
                'min_child_samples': [5, 100],
                'lambda_l1': [0, 1],
                'lambda_l2': [0, 1]
            }
        elif algorithm == 'RandomForest':
            return {
                'n_estimators': [50, 300],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 10, 20],
                'min_samples_leaf': [1, 5, 10],
                'max_features': ['sqrt', 'log2', 0.5, 0.8]
            }
        elif algorithm == 'LogisticRegression':
            return {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None]
            }
        elif algorithm == 'NeuralNetwork':
            return {
                'hidden_layers': [[64, 32], [128, 64], [256, 128, 64]],
                'activation': ['relu', 'tanh'],
                'learning_rate': [0.001, 0.01, 0.1],
                'batch_size': [32, 64, 128],
                'dropout': [0, 0.2, 0.5],
                'epochs': [50, 100, 200]
            }
        else:
            # Default search space
            return {'default_params': True}
    
    def _setup_cv_strategy(
        self,
        df: pd.DataFrame,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Setup cross-validation strategy"""
        
        n_samples = len(df)
        
        # Determine number of folds based on data size
        if n_samples < 500:
            n_folds = 5
        elif n_samples < 5000:
            n_folds = 5
        else:
            n_folds = 10
        
        cv_strategy = {
            'n_folds': n_folds,
            'shuffle': True,
            'random_state': 42
        }
        
        # Time series special handling
        if context.get('temporal_aspect') and context.get('is_time_series'):
            cv_strategy = {
                'method': 'time_series_split',
                'n_splits': min(5, n_samples // 100),
                'gap': 0,
                'test_size': None
            }
        # Stratified for classification
        elif task == 'classification':
            cv_strategy['method'] = 'stratified_kfold'
        # Group-based if needed
        elif context.get('group_column'):
            cv_strategy['method'] = 'group_kfold'
            cv_strategy['groups'] = context['group_column']
        else:
            cv_strategy['method'] = 'kfold'
        
        return cv_strategy
    
    def _configure_ensemble(
        self,
        algorithms: List[str],
        task: str
    ) -> Dict[str, Any]:
        """Configure ensemble strategy"""
        
        n_algorithms = len(algorithms)
        
        if n_algorithms < 3:
            # No ensemble for few algorithms
            return {'enabled': False}
        
        ensemble_config = {
            'enabled': True,
            'method': 'auto',
            'n_layers': 1
        }
        
        if task == 'classification':
            ensemble_config['method'] = 'voting'
            ensemble_config['voting'] = 'soft'
        elif task == 'regression':
            ensemble_config['method'] = 'averaging'
            ensemble_config['weights'] = 'optimized'
        elif task == 'ranking':
            ensemble_config['method'] = 'rank_averaging'
        
        # Stacking for many diverse algorithms
        if n_algorithms > 5:
            ensemble_config['method'] = 'stacking'
            ensemble_config['meta_learner'] = 'LogisticRegression' if task == 'classification' else 'Ridge'
            ensemble_config['use_probabilities'] = task == 'classification'
        
        return ensemble_config
    
    def _setup_monitoring(
        self,
        task: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Setup monitoring configuration"""
        
        monitoring = {
            'enabled': True,
            'drift_detection': True,
            'performance_tracking': True,
            'feature_importance': True
        }
        
        problem_type = context.get('problem_type')
        
        # Real-time monitoring for critical problems
        if problem_type in ['fraud_detection', 'anomaly_detection']:
            monitoring['real_time'] = True
            monitoring['alert_thresholds'] = {
                'performance_drop': 0.05,
                'drift_score': 0.3,
                'prediction_latency_ms': 100
            }
        
        # Batch monitoring for others
        else:
            monitoring['batch'] = True
            monitoring['frequency'] = 'daily'
            monitoring['alert_thresholds'] = {
                'performance_drop': 0.1,
                'drift_score': 0.5
            }
        
        # Add specific metrics to track
        if task == 'classification':
            monitoring['metrics'] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        elif task == 'regression':
            monitoring['metrics'] = ['rmse', 'mae', 'mape', 'r2']
        elif task == 'ranking':
            monitoring['metrics'] = ['ndcg', 'map', 'mrr']
        
        return monitoring
    
    def _generate_config_reasoning(
        self,
        task: str,
        algorithms: List[str],
        primary_metric: str,
        constraints: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for the configuration"""
        
        reasoning = f"📊 **Configuration Strategy**\n\n"
        reasoning += f"**Task:** {task}\n"
        reasoning += f"**Optimization Metric:** {primary_metric}\n"
        reasoning += f"**Time Budget:** {constraints['time_budget']/60:.1f} minutes\n\n"
        
        reasoning += f"**Selected Algorithms ({len(algorithms)}):**\n"
        for algo in algorithms[:5]:  # Show top 5
            capabilities = self.ALGORITHM_CAPABILITIES.get(algo, {})
            reasoning += f"- {algo}: {capabilities.get('interpretability', 'unknown')} interpretability, "
            reasoning += f"{capabilities.get('training_speed', 'unknown')} training\n"
        
        reasoning += f"\n**Key Decisions:**\n"
        
        if constraints.get('interpretability_required'):
            reasoning += "- Prioritized interpretable models due to requirements\n"
        
        if constraints.get('real_time_scoring'):
            reasoning += "- Selected fast inference algorithms for real-time scoring\n"
        
        if constraints.get('memory_limit_gb', 16) < 8:
            reasoning += "- Chose memory-efficient algorithms due to resource constraints\n"
        
        return reasoning
    
    def adapt_config(
        self,
        base_config: OptimalConfig,
        new_constraints: Dict[str, Any]
    ) -> OptimalConfig:
        """Adapt an existing configuration to new constraints"""
        
        adapted = base_config
        
        # Adapt to time constraints
        if new_constraints.get('time_budget'):
            ratio = new_constraints['time_budget'] / base_config.time_budget
            adapted.hpo_config['n_iter'] = int(base_config.hpo_config['n_iter'] * ratio)
            adapted.time_budget = new_constraints['time_budget']
        
        # Adapt to memory constraints
        if new_constraints.get('memory_limit_gb'):
            if new_constraints['memory_limit_gb'] < 8:
                # Remove memory-intensive algorithms
                adapted.algorithms = [
                    algo for algo in adapted.algorithms
                    if self.ALGORITHM_CAPABILITIES[algo]['memory_usage'] != 'high'
                ]
        
        # Adapt to interpretability requirements
        if new_constraints.get('interpretability_required'):
            # Prioritize interpretable algorithms
            interpretable = []
            for algo in adapted.algorithms:
                if self.ALGORITHM_CAPABILITIES[algo]['interpretability'] in ['very_high', 'high']:
                    interpretable.append(algo)
            if interpretable:
                adapted.algorithms = interpretable
        
        return adapted
    
    def learn_from_results(
        self,
        config: OptimalConfig,
        performance: Dict[str, float],
        execution_time: float
    ):
        """Learn from execution results to improve future configs"""
        
        duration = execution_time if execution_time > 0 else 1e-6

        self.performance_history.append({
            'timestamp': datetime.now(),
            'config': config.to_dict(),
            'performance': performance,
            'execution_time': execution_time,
            'efficiency': performance.get(config.primary_metric, 0) / duration
        })
        
        # Analyze what worked well
        if performance.get(config.primary_metric, 0) > 0.9:  # Good performance
            logger.info(f"✅ Successful config: {config.primary_metric}={performance.get(config.primary_metric):.3f}")
            # Could store successful patterns for future use
