"""
Intelligent Context Detector for AutoML Platform
=================================================
Automatically detects ML problem type and business context from data.
Enhanced with Claude SDK for sophisticated reasoning.
"""

import pandas as pd
import numpy as np
import asyncio
import json
import logging
import importlib.util
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re

_anthropic_spec = importlib.util.find_spec("anthropic")
if _anthropic_spec is not None:
    from anthropic import AsyncAnthropic
else:
    AsyncAnthropic = None

logger = logging.getLogger(__name__)


@dataclass
class MLContext:
    """Detected ML context with confidence scores"""
    problem_type: str
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
    Enhanced with Claude for sophisticated ML problem understanding.
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
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        """Initialize the context detector"""
        self.detected_contexts = []
        self.column_analysis = {}
        
        # Initialize Claude client for enhanced reasoning
        if AsyncAnthropic is not None and anthropic_api_key:
            self.claude_client = AsyncAnthropic(api_key=anthropic_api_key)
            self.use_claude = True
            logger.info("IntelligentContextDetector initialized with Claude SDK")
        else:
            self.claude_client = None
            self.use_claude = False
            logger.info("IntelligentContextDetector using rule-based detection only")
        
        self.model = "claude-sonnet-4-20250514"
        self.max_tokens = 3000
        
    async def detect_ml_context(
        self, 
        df: pd.DataFrame, 
        target_col: Optional[str] = None,
        user_hints: Optional[Dict[str, Any]] = None
    ) -> MLContext:
        """
        Intelligently detect the ML problem type and context.
        Enhanced with Claude for sophisticated reasoning.
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
        
        # Step 5: Search for business context clues
        context_clues = await self._search_business_context(column_analysis, user_hints)
        
        # Step 6: Score each problem type (rule-based)
        problem_scores = {}
        for problem_type, patterns in self.PROBLEM_PATTERNS.items():
            score = self._calculate_problem_score(
                problem_type, patterns, column_analysis, 
                target_analysis, context_clues
            )
            problem_scores[problem_type] = score
        
        # Step 7: Use Claude for enhanced reasoning if available
        if self.use_claude and self.claude_client:
            best_problem, confidence, reasoning, alternatives = await self._claude_enhanced_detection(
                df, column_analysis, target_analysis, temporal_info, 
                data_characteristics, context_clues, problem_scores
            )
        else:
            # Fallback to rule-based
            best_problem = max(problem_scores, key=problem_scores.get)
            confidence = problem_scores[best_problem]
            alternatives = [
                {'type': prob, 'confidence': score}
                for prob, score in sorted(problem_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
                if score > 0.3
            ]
            reasoning = self._generate_reasoning(
                best_problem, confidence, column_analysis, 
                target_analysis, context_clues
            )
        
        # Step 8: Generate optimal configuration
        recommended_config = self._generate_optimal_config(
            best_problem, df, target_col, data_characteristics
        )
        
        logger.info(f"✅ Problem understood: {best_problem} (confidence: {confidence:.1%})")
        
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
    
    async def _claude_enhanced_detection(
        self,
        df: pd.DataFrame,
        column_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any],
        temporal_info: Dict[str, Any],
        data_characteristics: Dict[str, Any],
        context_clues: Dict[str, Any],
        rule_based_scores: Dict[str, float]
    ) -> Tuple[str, float, str, List[Dict[str, Any]]]:
        """Use Claude for sophisticated ML problem detection"""
        
        # Prepare context for Claude
        detection_context = {
            "columns": column_analysis['columns'][:20],  # Limit for token efficiency
            "detected_patterns": list(column_analysis.get('detected_patterns', [])),
            "target_info": target_analysis,
            "temporal_aspect": temporal_info.get('has_temporal', False),
            "data_size": {"rows": len(df), "columns": len(df.columns)},
            "data_characteristics": data_characteristics,
            "business_context": context_clues,
            "rule_based_scores": {k: round(v, 2) for k, v in sorted(rule_based_scores.items(), key=lambda x: x[1], reverse=True)[:5]}
        }
        
        prompt = f"""You are an expert ML problem analyst. Analyze this dataset and determine the ML problem type.

Dataset Context:
{json.dumps(detection_context, indent=2)}

Available Problem Types:
- churn_prediction: Predicting customer attrition
- fraud_detection: Detecting fraudulent transactions/activities
- sales_forecasting: Forecasting future sales/demand
- credit_scoring: Assessing credit risk
- customer_segmentation: Clustering customers
- anomaly_detection: Detecting unusual patterns
- predictive_maintenance: Predicting equipment failures

Rule-based analysis suggests: {max(rule_based_scores, key=rule_based_scores.get)} ({max(rule_based_scores.values()):.1%} confidence)

Your task:
1. Analyze the column names, patterns, and data characteristics
2. Identify the most likely ML problem type
3. Provide confidence score (0-1)
4. Explain your reasoning
5. Suggest 2 alternative interpretations

Respond in this JSON format:
{{
  "problem_type": "...",
  "confidence": 0.85,
  "reasoning": "Detailed explanation of why...",
  "alternatives": [
    {{"type": "...", "confidence": 0.60, "reason": "..."}},
    {{"type": "...", "confidence": 0.45, "reason": "..."}}
  ]
}}"""

        try:
            response = await self.claude_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.3,  # Lower temperature for more consistent reasoning
                system="You are an expert ML problem type detector. Analyze datasets and identify the optimal ML approach.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            # Parse Claude's response
            result = self._parse_claude_detection_response(response_text, rule_based_scores)
            
            return (
                result['problem_type'],
                result['confidence'],
                result['reasoning'],
                result['alternatives']
            )
            
        except Exception as e:
            logger.warning(f"Claude detection failed, falling back to rule-based: {e}")
            best_problem = max(rule_based_scores, key=rule_based_scores.get)
            return (
                best_problem,
                rule_based_scores[best_problem],
                f"Rule-based detection (Claude unavailable): {best_problem}",
                []
            )
    
    def _parse_claude_detection_response(
        self, 
        response_text: str,
        rule_based_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Parse Claude's detection response"""
        try:
            # Try to extract JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate problem type
                if result['problem_type'] not in self.PROBLEM_PATTERNS:
                    logger.warning(f"Invalid problem type from Claude: {result['problem_type']}")
                    result['problem_type'] = max(rule_based_scores, key=rule_based_scores.get)
                
                # Ensure confidence is reasonable
                result['confidence'] = min(1.0, max(0.0, result['confidence']))
                
                return result
        except Exception as e:
            logger.warning(f"Failed to parse Claude response: {e}")
        
        # Fallback
        best_problem = max(rule_based_scores, key=rule_based_scores.get)
        return {
            'problem_type': best_problem,
            'confidence': rule_based_scores[best_problem],
            'reasoning': 'Fallback to rule-based detection',
            'alternatives': []
        }
    
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
                value_counts = df[target_col].value_counts()
                imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                target_analysis['imbalance_ratio'] = imbalance_ratio
                target_analysis['is_imbalanced'] = imbalance_ratio > 3
            elif df[target_col].nunique() < 10:
                target_analysis['type'] = 'multiclass_classification'
            else:
                target_analysis['type'] = 'regression'
                target_analysis['distribution'] = {
                    'mean': float(df[target_col].mean()),
                    'std': float(df[target_col].std()),
                    'skew': float(df[target_col].skew())
                }
        else:
            target_analysis['type'] = 'classification'
            value_counts = df[target_col].value_counts()
            target_analysis['class_distribution'] = {str(k): int(v) for k, v in value_counts.to_dict().items()}
        
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
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                temporal_info['has_temporal'] = True
                temporal_info['temporal_columns'].append(col)
                
                if len(df) > 10:
                    time_diffs = df[col].diff().dropna()
                    if len(time_diffs) > 0 and time_diffs.std() / time_diffs.mean() < 0.1:
                        temporal_info['is_time_series'] = True
                        try:
                            temporal_info['suggested_frequency'] = pd.infer_freq(df[col])
                        except:
                            pass
            
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
            'missing_ratio': float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
            'duplicate_ratio': float(df.duplicated().mean()),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(exclude=[np.number]).columns),
            'imbalance_detected': False,
            'high_cardinality_features': []
        }
        
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > 0.5 * len(df):
                characteristics['high_cardinality_features'].append(col)
        
        if target_col and target_col in df.columns:
            if df[target_col].nunique() < 10:
                value_counts = df[target_col].value_counts()
                if len(value_counts) > 1:
                    imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                    characteristics['imbalance_detected'] = imbalance_ratio > 3
                    characteristics['imbalance_ratio'] = float(imbalance_ratio)
        
        return characteristics
    
    async def _search_business_context(
        self, 
        column_analysis: Dict[str, Any],
        user_hints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for business context clues"""
        context = {
            'sector': None,
            'industry_patterns': [],
            'domain_keywords': []
        }
        
        if user_hints:
            context['sector'] = user_hints.get('sector', None)
            context['domain_keywords'] = user_hints.get('keywords', [])
            return context
        
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
        
        await asyncio.sleep(0.01)
        
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
        
        keyword_matches = sum(
            1 for col in column_analysis['columns']
            for keyword in patterns['keywords']
            if keyword in col.lower()
        )
        if keyword_matches > 0:
            score += weights['keywords'] * min(1.0, keyword_matches / 3)
        
        pattern_matches = sum(
            1 for col in column_analysis['columns']
            for pattern in patterns['column_patterns']
            if pattern in col.lower()
        )
        if pattern_matches > 0:
            score += weights['column_patterns'] * min(1.0, pattern_matches / 3)
        
        if target_analysis:
            target_name = target_analysis.get('name', '').lower()
            for pattern in patterns['target_patterns']:
                if pattern in target_name:
                    score += weights['target_patterns']
                    break
        
        if 'temporal_columns' in column_analysis.get('detected_patterns', set()):
            has_temporal_pattern = any(
                indicator in col.lower()
                for col in column_analysis['columns']
                for indicator in patterns['temporal_indicators']
            )
            if has_temporal_pattern:
                score += weights['temporal_indicators']
        
        metric_matches = sum(
            1 for col in column_analysis['columns']
            for metric in patterns['business_metrics']
            if metric in col.lower()
        )
        if metric_matches > 0:
            score += weights['business_metrics'] * min(1.0, metric_matches / 2)
        
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
            'task': 'auto',
            'algorithms': [],
            'preprocessing': {},
            'hpo': {},
            'monitoring': {}
        }
        
        if problem_type in ['churn_prediction', 'fraud_detection', 'credit_scoring']:
            config['task'] = 'classification'
            config['primary_metric'] = 'roc_auc'
            
            if data_characteristics.get('imbalance_detected'):
                config['preprocessing']['handle_imbalance'] = {
                    'method': 'SMOTE',
                    'sampling_strategy': 'auto'
                }
                config['primary_metric'] = 'f1'
        
        elif problem_type in ['sales_forecasting']:
            config['task'] = 'regression'
            config['primary_metric'] = 'mape'
            config['preprocessing']['create_lag_features'] = True
            config['preprocessing']['handle_seasonality'] = True
        
        elif problem_type == 'customer_segmentation':
            config['task'] = 'clustering'
            config['primary_metric'] = 'silhouette'
        
        elif problem_type in ['anomaly_detection', 'predictive_maintenance']:
            config['task'] = 'anomaly_detection'
            config['primary_metric'] = 'f1'
        
        config['algorithms'] = self._select_optimal_algorithms(problem_type, data_characteristics)
        
        if problem_type in ['fraud_detection', 'anomaly_detection']:
            config['monitoring']['real_time'] = True
            config['monitoring']['alert_threshold'] = 0.01
        
        return config
    
    def _select_optimal_algorithms(
        self,
        problem_type: str,
        data_characteristics: Dict[str, Any]
    ) -> List[str]:
        """Select optimal algorithms"""
        algorithms = ['XGBoost', 'LightGBM', 'RandomForest']
        
        n_samples = data_characteristics['n_samples']
        
        if problem_type == 'fraud_detection':
            algorithms.append('IsolationForest')
            if n_samples > 10000:
                algorithms.append('NeuralNetwork')
        elif problem_type == 'churn_prediction':
            algorithms.append('LogisticRegression')
        elif problem_type == 'sales_forecasting':
            algorithms.extend(['Prophet', 'ARIMA'])
        elif problem_type == 'anomaly_detection':
            algorithms.extend(['IsolationForest', 'OneClassSVM'])
        
        if n_samples < 1000:
            algorithms = [a for a in algorithms if a not in ['NeuralNetwork', 'LSTM']]
        
        return algorithms[:5]
    
    def _generate_reasoning(
        self,
        problem_type: str,
        confidence: float,
        column_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any],
        context_clues: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning"""
        
        reasoning = f"🎯 Detected: **{problem_type.replace('_', ' ').title()}** ({confidence:.1%})\n\n"
        reasoning += "**Analysis:**\n"
        
        patterns = list(column_analysis.get('detected_patterns', []))
        if patterns:
            reasoning += f"- Patterns: {', '.join(patterns)}\n"
        
        if target_analysis:
            reasoning += f"- Target '{target_analysis['name']}' → {target_analysis.get('type', 'unknown')}\n"
            if target_analysis.get('is_imbalanced'):
                reasoning += f"  ⚠️ Imbalanced ({target_analysis['imbalance_ratio']:.1f}:1)\n"
        
        if context_clues.get('sector'):
            reasoning += f"- Sector: {context_clues['sector']}\n"
        
        return reasoning
    
    def learn_from_feedback(
        self,
        predicted_type: str,
        actual_type: str,
        confidence: float,
        was_correct: bool
    ):
        """Learn from feedback"""
        learning_data = {
            'timestamp': datetime.now(),
            'predicted': predicted_type,
            'actual': actual_type,
            'confidence': confidence,
            'correct': was_correct
        }
        self.detected_contexts.append(learning_data)
        logger.info(f"Learning: {predicted_type} → {actual_type} (correct: {was_correct})")
