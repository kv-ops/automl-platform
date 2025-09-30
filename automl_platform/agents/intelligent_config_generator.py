"""
Intelligent Config Generator for AutoML Platform
================================================
Generates optimal configurations without templates.
Adapts to any ML problem dynamically.
NOW WITH CLAUDE SDK FOR STRATEGIC DECISIONS
"""

import importlib.util

_anthropic_spec = importlib.util.find_spec("anthropic")
if _anthropic_spec is not None:
    from anthropic import AsyncAnthropic
else:
    AsyncAnthropic = None

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
    NOW ENHANCED WITH CLAUDE SDK FOR STRATEGIC REASONING
    """
    
    # Algorithm capabilities matrix (unchanged)
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
    
    # Metric optimization strategies (unchanged)
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
    
    def __init__(self, use_claude: bool = True):
        """
        Initialize the config generator
        
        Args:
            use_claude: Whether to use Claude SDK for strategic decisions
        """
        self.generated_configs = []
        self.performance_history = []
        self.use_claude = use_claude and AsyncAnthropic is not None
        
        # Initialize Claude client if available
        if self.use_claude:
            self.claude_client = AsyncAnthropic()
            self.model = "claude-sonnet-4-20250514"
            logger.info("💎 Claude SDK enabled for strategic config generation")
        else:
            self.claude_client = None
            if use_claude:
                logger.warning("⚠️ Claude SDK requested but not available, falling back to rule-based")
            else:
                logger.info("📋 Using rule-based config generation")
        
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
        NOW ENHANCED WITH CLAUDE FOR STRATEGIC DECISIONS
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
        
        # Step 2: Select optimal algorithms (WITH CLAUDE IF AVAILABLE)
        algorithms = await self._select_algorithms(
            task, df, context, constraints, user_preferences
        )
        
        # Step 3: Choose primary metric
        primary_metric = self._select_metric(task, context, user_preferences)
        
        # Step 4: Configure preprocessing
        preprocessing = self._configure_preprocessing(df, context, task)
        
        # Step 5: Design feature engineering (WITH CLAUDE IF AVAILABLE)
        feature_engineering = await self._design_feature_engineering_with_claude(
            df, context, task
        ) if self.use_claude else self._design_feature_engineering(df, context, task)
        
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
        
        # Step 10: Generate reasoning (WITH CLAUDE IF AVAILABLE)
        reasoning = await self._generate_config_reasoning_with_claude(
            task, algorithms, primary_metric, constraints, context
        ) if self.use_claude else self._generate_config_reasoning(
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
        """
        Select optimal algorithms for the task.
        ENHANCED WITH CLAUDE FOR INTELLIGENT SELECTION
        """
        # First, get rule-based scores for all algorithms
        algo_scores = {}
        for algorithm in self.ALGORITHM_CAPABILITIES.keys():
            score = self._score_algorithm(
                algorithm, df, context, constraints, user_preferences
            )
            if score > 0:
                algo_scores[algorithm] = score
        
        # If Claude is available, use it for final strategic selection
        if self.use_claude and len(algo_scores) > 0:
            logger.info("💎 Using Claude for strategic algorithm selection...")
            try:
                selected = await self._claude_select_algorithms(
                    task, df, context, constraints, algo_scores
                )
                if selected:
                    logger.info(f"✅ Claude selected: {', '.join(selected)}")
                    return selected
            except Exception as e:
                logger.warning(f"⚠️ Claude selection failed: {e}, falling back to rule-based")
        
        # Fallback: rule-based selection
        sorted_algos = sorted(algo_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top 3-7 algorithms
        n_algos = min(7, max(3, len(sorted_algos) // 2))
        selected_algorithms = [algo for algo, _ in sorted_algos[:n_algos]]
        
        logger.info(f"📋 Rule-based selected: {', '.join(selected_algorithms)}")
        return selected_algorithms
    
    async def _claude_select_algorithms(
        self,
        task: str,
        df: pd.DataFrame,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
        algo_scores: Dict[str, float]
    ) -> List[str]:
        """Use Claude for intelligent algorithm selection"""
        
        # Prepare data characteristics
        data_chars = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'has_categorical': len(df.select_dtypes(include=['object']).columns) > 0,
            'imbalance_detected': context.get('imbalance_detected', False),
            'temporal_aspect': context.get('temporal_aspect', False)
        }
        
        prompt = f"""Analyze this ML task and select optimal algorithms.

Task: {task}
Problem Type: {context.get('problem_type', 'unknown')}
Business Sector: {context.get('business_sector', 'unknown')}

Data Characteristics:
{json.dumps(data_chars, indent=2)}

Constraints:
{json.dumps(constraints, indent=2)}

Rule-based Algorithm Scores (0-1 scale):
{json.dumps(algo_scores, indent=2)}

Algorithm Capabilities Available:
{json.dumps({k: v for k, v in self.ALGORITHM_CAPABILITIES.items() if k in algo_scores}, indent=2)}

Select 3-7 best algorithms considering:
1. Data size and characteristics (missing values, categorical features)
2. Task requirements (classification/regression/etc)
3. Time and resource constraints
4. Interpretability needs
5. Business context and problem type
6. Rule-based scores as baseline

IMPORTANT: Respond ONLY with valid JSON, no other text:
{{"algorithms": ["algo1", "algo2", ...], "reasoning": "why these algorithms"}}"""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.model,
                max_tokens=2000,
                system="You are an expert ML algorithm selector. Respond only with JSON.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Parse JSON response
            result = json.loads(response_text)
            selected_algorithms = result.get('algorithms', [])
            reasoning = result.get('reasoning', '')
            
            # Validate selected algorithms
            valid_algorithms = [
                algo for algo in selected_algorithms
                if algo in self.ALGORITHM_CAPABILITIES
            ]
            
            if valid_algorithms:
                logger.info(f"💎 Claude reasoning: {reasoning[:200]}...")
                return valid_algorithms
            
            return []
            
        except json.JSONDecodeError:
            logger.warning("⚠️ Claude response was not valid JSON")
            return []
        except Exception as e:
            logger.warning(f"⚠️ Claude algorithm selection failed: {e}")
            return []
    
    async def _design_feature_engineering_with_claude(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        task: str
    ) -> Dict[str, Any]:
        """Design feature engineering strategy with Claude's help"""
        
        # Get baseline from rules
        baseline_fe = self._design_feature_engineering(df, context, task)
        
        # Enhance with Claude
        prompt = f"""Design optimal feature engineering strategy for this ML problem.

Task: {task}
Problem Type: {context.get('problem_type', 'unknown')}
Dataset: {len(df)} rows, {len(df.columns)} columns

Baseline Feature Engineering:
{json.dumps(baseline_fe, indent=2)}

Data Context:
- Has temporal aspect: {context.get('temporal_aspect', False)}
- Business sector: {context.get('business_sector', 'unknown')}
- Imbalance detected: {context.get('imbalance_detected', False)}

Enhance this strategy by suggesting:
1. Additional domain-specific features for this problem type
2. Advanced feature interactions that could help
3. Feature selection strategies
4. Any problem-specific transformations

Respond ONLY with valid JSON:
{{
  "enhancements": {{
    "additional_features": ["feature1", "feature2"],
    "domain_specific": {{"key": "value"}},
    "advanced_techniques": ["technique1"]
  }},
  "reasoning": "why these enhancements"
}}"""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.model,
                max_tokens=1500,
                system="You are an expert in feature engineering. Respond only with JSON.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            result = json.loads(response_text)
            
            # Merge Claude's enhancements with baseline
            enhancements = result.get('enhancements', {})
            for key, value in enhancements.items():
                if key in baseline_fe:
                    if isinstance(baseline_fe[key], dict) and isinstance(value, dict):
                        baseline_fe[key].update(value)
                    elif isinstance(baseline_fe[key], list) and isinstance(value, list):
                        baseline_fe[key].extend(value)
                else:
                    baseline_fe[key] = value
            
            logger.info(f"💎 Enhanced feature engineering with Claude insights")
            return baseline_fe
            
        except Exception as e:
            logger.warning(f"⚠️ Claude feature engineering enhancement failed: {e}")
            return baseline_fe
    
    async def _generate_config_reasoning_with_claude(
        self,
        task: str,
        algorithms: List[str],
        primary_metric: str,
        constraints: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning with Claude"""
        
        prompt = f"""Generate a clear, concise explanation for this ML configuration.

Task: {task}
Problem: {context.get('problem_type', 'unknown')}
Algorithms Selected: {', '.join(algorithms)}
Primary Metric: {primary_metric}
Time Budget: {constraints['time_budget']/60:.1f} minutes

Key Constraints:
- Interpretability required: {constraints.get('interpretability_required', False)}
- Real-time scoring: {constraints.get('real_time_scoring', False)}
- Memory limit: {constraints.get('memory_limit_gb', 16)} GB

Create a 3-4 paragraph explanation covering:
1. Why these algorithms were chosen for this specific problem
2. Key decisions made and trade-offs
3. Expected performance and considerations

Be specific, technical, and actionable. Focus on the "why" behind decisions."""
        
        try:
            response = await self.claude_client.messages.create(
                model=self.model,
                max_tokens=1000,
                system="You are an expert ML engineer explaining configuration decisions.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            reasoning = response.content[0].text.strip()
            logger.info("💎 Generated reasoning with Claude")
            return reasoning
            
        except Exception as e:
            logger.warning(f"⚠️ Claude reasoning generation failed: {e}")
            return self._generate_config_reasoning(task, algorithms, primary_metric, constraints)
    
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
            'fraud_detection': 'average_precision',
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
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
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
                'strategy': 'clip'
            }
        else:
            preprocessing['outliers'] = {'method': 'none'}
        
        # Scaling strategy
        if task in ['classification', 'regression']:
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
        if time_budget < 300:
            method = 'random'
            n_iter = 10
        elif n_samples < 1000:
            method = 'grid'
            n_iter = 20
        elif time_budget < 1800:
            method = 'optuna'
            n_iter = 30
        else:
            method = 'optuna'
            n_iter = 50
        
        n_iter_per_algo = n_iter // max(n_algorithms, 1)
        
        hpo_config = {
            'method': method,
            'n_iter': n_iter,
            'n_iter_per_algo': n_iter_per_algo,
            'timeout': time_budget * 0.7,
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 0.001
            },
            'pruning': method == 'optuna',
            'warm_start': True,
            'n_jobs': -1
        }
        
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
            return {'default_params': True}
    
    def _setup_cv_strategy(
        self,
        df: pd.DataFrame,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Setup cross-validation strategy"""
        
        n_samples = len(df)
        
        if n_samples < 500:
            n_folds = 3
        elif n_samples < 5000:
            n_folds = 5
        else:
            n_folds = 10
        
        cv_strategy = {
            'n_folds': n_folds,
            'shuffle': True,
            'random_state': 42
        }
        
        if context.get('temporal_aspect') and context.get('is_time_series'):
            cv_strategy = {
                'method': 'time_series_split',
                'n_splits': min(5, n_samples // 100),
                'gap': 0,
                'test_size': None
            }
        elif task == 'classification':
            cv_strategy['method'] = 'stratified_kfold'
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
        
        if problem_type in ['fraud_detection', 'anomaly_detection']:
            monitoring['real_time'] = True
            monitoring['alert_thresholds'] = {
                'performance_drop': 0.05,
                'drift_score': 0.3,
                'prediction_latency_ms': 100
            }
        else:
            monitoring['batch'] = True
            monitoring['frequency'] = 'daily'
            monitoring['alert_thresholds'] = {
                'performance_drop': 0.1,
                'drift_score': 0.5
            }
        
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
        for algo in algorithms[:5]:
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
        
        if new_constraints.get('time_budget'):
            ratio = new_constraints['time_budget'] / base_config.time_budget
            adapted.hpo_config['n_iter'] = int(base_config.hpo_config['n_iter'] * ratio)
            adapted.time_budget = new_constraints['time_budget']
        
        if new_constraints.get('memory_limit_gb'):
            if new_constraints['memory_limit_gb'] < 8:
                adapted.algorithms = [
                    algo for algo in adapted.algorithms
                    if self.ALGORITHM_CAPABILITIES[algo]['memory_usage'] != 'high'
                ]
        
        if new_constraints.get('interpretability_required'):
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
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'config': config.to_dict(),
            'performance': performance,
            'execution_time': execution_time,
            'efficiency': performance.get(config.primary_metric, 0) / execution_time
        })
        
        if performance.get(config.primary_metric, 0) > 0.9:
            logger.info(f"✅ Successful config: {config.primary_metric}={performance.get(config.primary_metric):.3f}")
