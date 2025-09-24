"""
Intelligent Context Detector for AutoML Platform
=================================================
Automatically detects ML problem type and business context from data.
No templates needed - pure intelligence.
"""

import pandas as pd
import numpy as np
import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class MLContext:
    """Detected ML context with confidence scores"""
    problem_type: str  # churn, fraud, forecast, recommendation, etc.
    confidence: float
    detected_patterns: List[str]
    business_sector: Optional[str]
    temporal_aspect: bool
    imbalance_detected: bool
    recommended_config: Dict[str, Any]
    reasoning: str
    alternative_interpretations: List[Dict[str, Any]] = field(default_factory=list)


class IntelligentContextDetector:
    """
    The brain of the Agent-First approach.
    Automatically understands what ML problem we're dealing with.
    """
    
    # Pattern signatures for different ML problems
    PROBLEM_PATTERNS = {
        'churn_prediction': {
            'keywords': ['customer', 'subscriber', 'user', 'account', 'retention', 'cancel', 
                        'terminate', 'leave', 'stop', 'unsubscribe', 'churn'],
            'column_patterns': ['customer_id', 'user_id', 'subscription', 'last_login', 
                              'last_activity', 'tenure', 'contract', 'renewal'],
            'target_patterns': ['churn', 'churned', 'cancelled', 'active', 'status', 'retained'],
            'temporal_indicators': ['date', 'time', 'period', 'month', 'year'],
            'business_metrics': ['ltv', 'revenue', 'usage', 'engagement']
        },
        
        'fraud_detection': {
            'keywords': ['transaction', 'payment', 'fraud', 'anomaly', 'suspicious', 
                        'unauthorized', 'fake', 'scam', 'illegal'],
            'column_patterns': ['transaction_id', 'amount', 'merchant', 'card', 'ip_address',
                              'device', 'location', 'velocity', 'risk_score'],
            'target_patterns': ['fraud', 'fraudulent', 'legitimate', 'suspicious', 'anomaly'],
            'temporal_indicators': ['timestamp', 'transaction_time', 'date'],
            'business_metrics': ['amount', 'frequency', 'velocity']
        },
        
        'sales_forecasting': {
            'keywords': ['sales', 'revenue', 'forecast', 'prediction', 'demand', 'inventory',
                        'supply', 'order', 'quantity', 'volume'],
            'column_patterns': ['date', 'time', 'period', 'product', 'sku', 'store', 
                              'location', 'quantity', 'price', 'revenue'],
            'target_patterns': ['sales', 'revenue', 'quantity', 'demand', 'units_sold'],
            'temporal_indicators': ['date', 'week', 'month', 'quarter', 'year', 'season'],
            'business_metrics': ['revenue', 'margin', 'profit', 'cost']
        },
        
        'recommendation_system': {
            'keywords': ['user', 'item', 'product', 'rating', 'preference', 'recommendation',
                        'suggest', 'personalize', 'content', 'match'],
            'column_patterns': ['user_id', 'item_id', 'product_id', 'rating', 'score',
                              'interaction', 'click', 'view', 'purchase'],
            'target_patterns': ['rating', 'score', 'preference', 'liked', 'purchased'],
            'temporal_indicators': ['timestamp', 'date', 'session'],
            'business_metrics': ['conversion', 'ctr', 'engagement']
        },
        
        'credit_scoring': {
            'keywords': ['credit', 'loan', 'risk', 'default', 'payment', 'debt', 'income',
                        'financial', 'banking', 'mortgage'],
            'column_patterns': ['income', 'debt', 'credit_score', 'employment', 'loan_amount',
                              'interest_rate', 'payment_history', 'delinquency'],
            'target_patterns': ['default', 'risk', 'approved', 'creditworthy', 'score'],
            'temporal_indicators': ['loan_date', 'payment_date', 'maturity'],
            'business_metrics': ['amount', 'rate', 'ltv', 'dti']
        },
        
        'customer_segmentation': {
            'keywords': ['segment', 'cluster', 'group', 'category', 'type', 'persona',
                        'behavior', 'pattern', 'profile'],
            'column_patterns': ['customer_id', 'demographics', 'behavior', 'purchase',
                              'frequency', 'recency', 'monetary'],
            'target_patterns': ['segment', 'cluster', 'group', 'category'],
            'temporal_indicators': ['last_purchase', 'first_purchase', 'tenure'],
            'business_metrics': ['clv', 'frequency', 'monetary', 'recency']
        },
        
        'anomaly_detection': {
            'keywords': ['anomaly', 'outlier', 'unusual', 'abnormal', 'irregular',
                        'deviation', 'suspicious', 'alert'],
            'column_patterns': ['sensor', 'metric', 'value', 'measurement', 'reading',
                              'threshold', 'baseline', 'normal_range'],
            'target_patterns': ['anomaly', 'outlier', 'normal', 'abnormal', 'alert'],
            'temporal_indicators': ['timestamp', 'time', 'datetime'],
            'business_metrics': ['deviation', 'threshold', 'confidence']
        },
        
        'predictive_maintenance': {
            'keywords': ['maintenance', 'failure', 'equipment', 'sensor', 'machine',
                        'breakdown', 'repair', 'condition', 'monitoring'],
            'column_patterns': ['sensor_id', 'equipment_id', 'temperature', 'pressure',
                              'vibration', 'runtime', 'cycles', 'age'],
            'target_patterns': ['failure', 'maintenance', 'breakdown', 'rul', 'ttf'],
            'temporal_indicators': ['timestamp', 'runtime', 'cycles'],
            'business_metrics': ['downtime', 'mtbf', 'mttr', 'cost']
        }
    }
    
    def __init__(self):
        """Initialize the context detector"""
        self.detected_contexts = []
        self.column_analysis = {}
        self.web_search_cache = {}
        
    async def detect_ml_context(
        self, 
        df: pd.DataFrame, 
        target_col: Optional[str] = None,
        user_hints: Optional[Dict[str, Any]] = None
    ) -> MLContext:
        """
        Intelligently detect the ML problem type and context.
        
        This is the core intelligence - no templates needed!
        """
        logger.info("🧠 Starting intelligent context detection...")
        
        # Step 1: Analyze column names and data patterns
        column_analysis = self._analyze_columns(df)
        self.column_analysis = column_analysis
        
        # Step 2: Analyze target variable if provided
        target_analysis = self._analyze_target(df, target_col) if target_col else {}
        
        # Step 3: Detect temporal aspects
        temporal_info = self._detect_temporal_aspects(df)
        
        # Step 4: Detect data characteristics
        data_characteristics = self._analyze_data_characteristics(df, target_col)
        
        # Step 5: Search for business context clues (async)
        context_clues = await self._search_business_context(column_analysis, user_hints)
        
        # Step 6: Score each problem type
        problem_scores = {}
        for problem_type, patterns in self.PROBLEM_PATTERNS.items():
            score = self._calculate_problem_score(
                problem_type, patterns, column_analysis, 
                target_analysis, context_clues
            )
            problem_scores[problem_type] = score
        
        # Step 7: Determine the most likely problem type
        best_problem = max(problem_scores, key=problem_scores.get)
        confidence = problem_scores[best_problem]
        
        # Step 8: Get alternative interpretations
        alternatives = [
            {'type': prob, 'confidence': score}
            for prob, score in sorted(problem_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
            if score > 0.3
        ]
        
        # Step 9: Generate optimal configuration for detected problem
        recommended_config = self._generate_optimal_config(
            best_problem, df, target_col, data_characteristics
        )
        
        # Step 10: Generate reasoning
        reasoning = self._generate_reasoning(
            best_problem, confidence, column_analysis, 
            target_analysis, context_clues
        )
        
        return MLContext(
            problem_type=best_problem,
            confidence=confidence,
            detected_patterns=list(column_analysis.get('detected_patterns', [])),
            business_sector=context_clues.get('sector'),
            temporal_aspect=temporal_info.get('has_temporal', False),
            imbalance_detected=data_characteristics.get('imbalance_detected', False),
            recommended_config=recommended_config,
            reasoning=reasoning,
            alternative_interpretations=alternatives
        )
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze column names and types for patterns"""
        analysis = {
            'columns': list(df.columns),
            'detected_patterns': set(),
            'column_types': {},
            'potential_features': {},
            'potential_target': None
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Detect ID columns
            if any(id_pattern in col_lower for id_pattern in ['_id', 'id_', 'identifier']):
                analysis['detected_patterns'].add('has_id_columns')
                analysis['potential_features'][col] = 'identifier'
            
            # Detect temporal columns
            if any(time_pattern in col_lower for time_pattern in 
                   ['date', 'time', 'timestamp', 'period', 'month', 'year', 'day']):
                analysis['detected_patterns'].add('has_temporal_features')
                analysis['potential_features'][col] = 'temporal'
            
            # Detect money/financial columns
            if any(money_pattern in col_lower for money_pattern in 
                   ['amount', 'price', 'cost', 'revenue', 'payment', 'balance', 'income']):
                analysis['detected_patterns'].add('has_financial_features')
                analysis['potential_features'][col] = 'financial'
            
            # Detect potential target columns
            if any(target_pattern in col_lower for target_pattern in 
                   ['target', 'label', 'class', 'outcome', 'result', 'y_']):
                analysis['potential_target'] = col
                analysis['detected_patterns'].add('has_labeled_target')
            
            # Check for specific problem indicators
            if 'churn' in col_lower:
                analysis['detected_patterns'].add('churn_indicator')
            if 'fraud' in col_lower:
                analysis['detected_patterns'].add('fraud_indicator')
            if 'sales' in col_lower or 'revenue' in col_lower:
                analysis['detected_patterns'].add('sales_indicator')
            if 'rating' in col_lower or 'score' in col_lower:
                analysis['detected_patterns'].add('rating_indicator')
            
            # Analyze data types
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() == 2:
                    analysis['column_types'][col] = 'binary'
                elif df[col].nunique() < 10:
                    analysis['column_types'][col] = 'categorical_numeric'
                else:
                    analysis['column_types'][col] = 'continuous'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                analysis['column_types'][col] = 'datetime'
            else:
                analysis['column_types'][col] = 'categorical'
        
        return analysis
    
    def _analyze_target(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze target variable characteristics"""
        if target_col not in df.columns:
            return {}
        
        target_analysis = {
            'name': target_col,
            'dtype': str(df[target_col].dtype),
            'unique_values': df[target_col].nunique(),
            'missing_ratio': df[target_col].isnull().mean()
        }
        
        if pd.api.types.is_numeric_dtype(df[target_col]):
            if df[target_col].nunique() == 2:
                target_analysis['type'] = 'binary_classification'
                # Check for imbalance
                value_counts = df[target_col].value_counts()
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                target_analysis['imbalance_ratio'] = imbalance_ratio
                target_analysis['is_imbalanced'] = imbalance_ratio > 3
            elif df[target_col].nunique() < 10:
                target_analysis['type'] = 'multiclass_classification'
            else:
                target_analysis['type'] = 'regression'
                target_analysis['distribution'] = {
                    'mean': df[target_col].mean(),
                    'std': df[target_col].std(),
                    'skew': df[target_col].skew()
                }
        else:
            target_analysis['type'] = 'classification'
            value_counts = df[target_col].value_counts()
            target_analysis['class_distribution'] = value_counts.to_dict()
        
        return target_analysis
    
    def _detect_temporal_aspects(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if the problem has temporal/time-series aspects"""
        temporal_info = {
            'has_temporal': False,
            'temporal_columns': [],
            'suggested_frequency': None,
            'is_time_series': False
        }
        
        for col in df.columns:
            # Check for datetime columns
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                temporal_info['has_temporal'] = True
                temporal_info['temporal_columns'].append(col)
                
                # Check if it's regular time series
                if len(df) > 10:
                    time_diffs = df[col].diff().dropna()
                    if time_diffs.std() / time_diffs.mean() < 0.1:  # Regular intervals
                        temporal_info['is_time_series'] = True
                        temporal_info['suggested_frequency'] = pd.infer_freq(df[col])
            
            # Check for date-like strings
            elif df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].iloc[:100], errors='raise')
                    temporal_info['has_temporal'] = True
                    temporal_info['temporal_columns'].append(col)
                except:
                    pass
        
        return temporal_info
    
    def _analyze_data_characteristics(
        self, 
        df: pd.DataFrame, 
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze general data characteristics"""
        characteristics = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'duplicate_ratio': df.duplicated().mean(),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(exclude=[np.number]).columns),
            'imbalance_detected': False,
            'high_cardinality_features': []
        }
        
        # Check for high cardinality
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 0.5 * len(df):
                characteristics['high_cardinality_features'].append(col)
        
        # Check target imbalance if provided
        if target_col and target_col in df.columns:
            if df[target_col].nunique() < 10:  # Classification
                value_counts = df[target_col].value_counts()
                if len(value_counts) > 1:
                    imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                    characteristics['imbalance_detected'] = imbalance_ratio > 3
                    characteristics['imbalance_ratio'] = imbalance_ratio
        
        return characteristics
    
    async def _search_business_context(
        self, 
        column_analysis: Dict[str, Any],
        user_hints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for business context clues.
        In production, this would make actual web searches or API calls.
        """
        context = {
            'sector': None,
            'industry_patterns': [],
            'domain_keywords': []
        }
        
        # If user provided hints, use them
        if user_hints:
            context['sector'] = user_hints.get('sector', None)
            context['domain_keywords'] = user_hints.get('keywords', [])
            return context
        
        # Infer sector from column patterns
        detected_patterns = column_analysis.get('detected_patterns', set())
        
        if 'has_financial_features' in detected_patterns:
            if 'fraud_indicator' in detected_patterns:
                context['sector'] = 'financial_fraud'
            elif any('loan' in col.lower() or 'credit' in col.lower() 
                    for col in column_analysis['columns']):
                context['sector'] = 'credit_risk'
            else:
                context['sector'] = 'finance'
        
        elif 'churn_indicator' in detected_patterns:
            if any('telecom' in col.lower() or 'mobile' in col.lower() 
                  for col in column_analysis['columns']):
                context['sector'] = 'telecom'
            else:
                context['sector'] = 'customer_analytics'
        
        elif 'sales_indicator' in detected_patterns:
            context['sector'] = 'retail'
        
        elif any('sensor' in col.lower() or 'equipment' in col.lower() 
                for col in column_analysis['columns']):
            context['sector'] = 'industrial'
        
        # Simulate web search for domain patterns
        # In production, this would be actual API calls
        await asyncio.sleep(0.01)  # Simulate async operation
        
        return context
    
    def _calculate_problem_score(
        self,
        problem_type: str,
        patterns: Dict[str, Any],
        column_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any],
        context_clues: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for a specific problem type"""
        score = 0.0
        weights = {
            'keywords': 0.2,
            'column_patterns': 0.3,
            'target_patterns': 0.25,
            'temporal_indicators': 0.15,
            'business_metrics': 0.1
        }
        
        # Check keywords in column names
        keyword_matches = sum(
            1 for col in column_analysis['columns']
            for keyword in patterns['keywords']
            if keyword in col.lower()
        )
        if keyword_matches > 0:
            score += weights['keywords'] * min(1.0, keyword_matches / 3)
        
        # Check column patterns
        pattern_matches = sum(
            1 for col in column_analysis['columns']
            for pattern in patterns['column_patterns']
            if pattern in col.lower()
        )
        if pattern_matches > 0:
            score += weights['column_patterns'] * min(1.0, pattern_matches / 3)
        
        # Check target patterns
        if target_analysis:
            target_name = target_analysis.get('name', '').lower()
            for pattern in patterns['target_patterns']:
                if pattern in target_name:
                    score += weights['target_patterns']
                    break
        
        # Check temporal aspects
        if 'temporal_columns' in column_analysis.get('detected_patterns', set()):
            has_temporal_pattern = any(
                indicator in col.lower()
                for col in column_analysis['columns']
                for indicator in patterns['temporal_indicators']
            )
            if has_temporal_pattern:
                score += weights['temporal_indicators']
        
        # Check business metrics
        metric_matches = sum(
            1 for col in column_analysis['columns']
            for metric in patterns['business_metrics']
            if metric in col.lower()
        )
        if metric_matches > 0:
            score += weights['business_metrics'] * min(1.0, metric_matches / 2)
        
        # Boost score based on context clues
        if context_clues.get('sector'):
            if problem_type in context_clues['sector']:
                score *= 1.5
        
        return min(1.0, score)
    
    def _generate_optimal_config(
        self,
        problem_type: str,
        df: pd.DataFrame,
        target_col: Optional[str],
        data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimal configuration for the detected problem type"""
        
        config = {
            'problem_type': problem_type,
            'task': 'auto',  # Will be determined
            'algorithms': [],
            'preprocessing': {},
            'hpo': {},
            'monitoring': {},
            'business_rules': {}
        }
        
        # Determine task type
        if problem_type in ['churn_prediction', 'fraud_detection', 'credit_scoring']:
            config['task'] = 'classification'
            config['primary_metric'] = 'roc_auc'
            
            # Handle imbalance if detected
            if data_characteristics.get('imbalance_detected'):
                config['preprocessing']['handle_imbalance'] = {
                    'method': 'SMOTE',
                    'sampling_strategy': 'auto'
                }
                config['primary_metric'] = 'f1'
        
        elif problem_type in ['sales_forecasting', 'demand_prediction']:
            config['task'] = 'regression'
            config['primary_metric'] = 'mape'
            config['preprocessing']['create_lag_features'] = True
            config['preprocessing']['handle_seasonality'] = True
        
        elif problem_type == 'recommendation_system':
            config['task'] = 'ranking'
            config['primary_metric'] = 'ndcg'
        
        elif problem_type == 'customer_segmentation':
            config['task'] = 'clustering'
            config['primary_metric'] = 'silhouette'
        
        elif problem_type in ['anomaly_detection', 'predictive_maintenance']:
            config['task'] = 'anomaly_detection'
            config['primary_metric'] = 'f1'
        
        # Select optimal algorithms based on problem type and data
        config['algorithms'] = self._select_optimal_algorithms(
            problem_type, data_characteristics
        )
        
        # Configure preprocessing
        config['preprocessing'] = self._configure_preprocessing(
            problem_type, data_characteristics
        )
        
        # Configure HPO
        config['hpo'] = self._configure_hpo(
            problem_type, data_characteristics
        )
        
        # Add monitoring based on problem type
        if problem_type in ['fraud_detection', 'anomaly_detection']:
            config['monitoring']['real_time'] = True
            config['monitoring']['alert_threshold'] = 0.01
        
        return config
    
    def _select_optimal_algorithms(
        self,
        problem_type: str,
        data_characteristics: Dict[str, Any]
    ) -> List[str]:
        """Select optimal algorithms based on problem and data"""
        
        n_samples = data_characteristics['n_samples']
        n_features = data_characteristics['n_features']
        
        algorithms = []
        
        # Tree-based are generally good
        algorithms.extend(['XGBoost', 'LightGBM', 'RandomForest'])
        
        # Problem-specific additions
        if problem_type == 'fraud_detection':
            algorithms.append('IsolationForest')
            if n_samples > 10000:
                algorithms.append('NeuralNetwork')
        
        elif problem_type == 'churn_prediction':
            algorithms.append('LogisticRegression')
            if data_characteristics.get('imbalance_detected'):
                algorithms.append('BalancedRandomForest')
        
        elif problem_type == 'sales_forecasting':
            algorithms.extend(['ARIMA', 'Prophet', 'LSTM'])
        
        elif problem_type == 'recommendation_system':
            algorithms.extend(['MatrixFactorization', 'NeuralCollaborativeFiltering'])
        
        elif problem_type == 'credit_scoring':
            algorithms.extend(['LogisticRegression', 'ScoreCard'])
        
        elif problem_type == 'anomaly_detection':
            algorithms.extend(['IsolationForest', 'OneClassSVM', 'Autoencoder'])
        
        # Filter based on data size
        if n_samples < 1000:
            # Remove complex models for small data
            algorithms = [a for a in algorithms if a not in ['NeuralNetwork', 'LSTM']]
        
        return algorithms[:5]  # Top 5 algorithms
    
    def _configure_preprocessing(
        self,
        problem_type: str,
        data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure preprocessing based on problem and data"""
        
        preprocessing = {
            'handle_missing': {
                'strategy': 'smart',
                'threshold': 0.3
            },
            'handle_outliers': {
                'method': 'iqr' if problem_type != 'fraud_detection' else 'none'
            },
            'scaling': {
                'method': 'robust'
            }
        }
        
        # Problem-specific preprocessing
        if problem_type in ['sales_forecasting', 'demand_prediction']:
            preprocessing['create_time_features'] = True
            preprocessing['lag_features'] = [1, 7, 30]
            preprocessing['rolling_features'] = [7, 30]
        
        elif problem_type == 'fraud_detection':
            preprocessing['create_velocity_features'] = True
            preprocessing['create_frequency_encoding'] = True
        
        elif problem_type == 'churn_prediction':
            preprocessing['create_recency_features'] = True
            preprocessing['create_frequency_features'] = True
            preprocessing['create_monetary_features'] = True
        
        elif problem_type == 'recommendation_system':
            preprocessing['create_interaction_features'] = True
            preprocessing['normalize_ratings'] = True
        
        # Handle high cardinality
        if data_characteristics.get('high_cardinality_features'):
            preprocessing['encode_high_cardinality'] = {
                'method': 'target_encoding',
                'smoothing': True
            }
        
        return preprocessing
    
    def _configure_hpo(
        self,
        problem_type: str,
        data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure hyperparameter optimization"""
        
        n_samples = data_characteristics['n_samples']
        
        hpo = {
            'method': 'optuna',  # Default to best method
            'n_iter': 50,
            'timeout': 3600,
            'early_stopping': True
        }
        
        # Adjust based on data size
        if n_samples < 1000:
            hpo['n_iter'] = 20
            hpo['method'] = 'random'
        elif n_samples < 10000:
            hpo['n_iter'] = 30
        else:
            hpo['n_iter'] = 50
        
        # Problem-specific optimization metrics
        if problem_type == 'fraud_detection':
            hpo['optimize_metric'] = 'precision_at_recall_50'
        elif problem_type == 'churn_prediction':
            hpo['optimize_metric'] = 'f1'
        elif problem_type == 'sales_forecasting':
            hpo['optimize_metric'] = 'mape'
        elif problem_type == 'recommendation_system':
            hpo['optimize_metric'] = 'ndcg@10'
        
        return hpo
    
    def _generate_reasoning(
        self,
        problem_type: str,
        confidence: float,
        column_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any],
        context_clues: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for the detection"""
        
        reasoning = f"🎯 Detected ML Problem: **{problem_type.replace('_', ' ').title()}** (Confidence: {confidence:.1%})\n\n"
        reasoning += "**Analysis Results:**\n"
        
        # Column patterns found
        patterns = list(column_analysis.get('detected_patterns', []))
        if patterns:
            reasoning += f"- Found patterns: {', '.join(patterns)}\n"
        
        # Target analysis
        if target_analysis:
            reasoning += f"- Target variable '{target_analysis['name']}' suggests {target_analysis.get('type', 'unknown')} problem\n"
            if target_analysis.get('is_imbalanced'):
                reasoning += f"  ⚠️ Imbalanced data detected (ratio: {target_analysis['imbalance_ratio']:.1f}:1)\n"
        
        # Context
        if context_clues.get('sector'):
            reasoning += f"- Business sector identified: {context_clues['sector']}\n"
        
        # Key indicators
        reasoning += "\n**Key Indicators:**\n"
        for col in column_analysis['columns'][:5]:
            if col in column_analysis.get('potential_features', {}):
                feature_type = column_analysis['potential_features'][col]
                reasoning += f"- '{col}' → {feature_type}\n"
        
        return reasoning
    
    def learn_from_feedback(
        self,
        predicted_type: str,
        actual_type: str,
        confidence: float,
        was_correct: bool
    ):
        """Learn from user feedback to improve future predictions"""
        # In production, this would update a learning model
        logger.info(f"Learning: Predicted {predicted_type} (confidence: {confidence:.1%}), "
                   f"Actual: {actual_type}, Correct: {was_correct}")
        
        # Store learning data
        learning_data = {
            'timestamp': datetime.now(),
            'predicted': predicted_type,
            'actual': actual_type,
            'confidence': confidence,
            'correct': was_correct
        }
        
        # This could be stored in a database for continuous improvement
        self.detected_contexts.append(learning_data)
