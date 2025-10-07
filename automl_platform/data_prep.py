"""
Enhanced Data Preparation Module with Full Agent-First Integration
===================================================================
Includes data quality checks, drift detection, advanced preprocessing,
integration with connectors, feature store, OpenAI agents, and Universal ML Agent.
Complete implementation with all Agent-First components.
ENHANCED: Hybrid mode support with local statistics calculation for efficient processing
"""

import pandas as pd
import numpy as np
import asyncio
import os
import yaml
from typing import Optional, List, Tuple, Union, Dict, Any, TYPE_CHECKING
from dataclasses import asdict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder,
    LabelEncoder, OrdinalEncoder, PowerTransformer, QuantileTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import logging
import warnings
warnings.filterwarnings('ignore')

# Import AutoMLConfig uniquement pour le type checking
if TYPE_CHECKING:
    from .config import AutoMLConfig

logger = logging.getLogger(__name__)


class EnhancedDataPreprocessor:
    """
    Advanced data preprocessing with quality checks and drift detection.
    No data leakage guaranteed through proper pipeline usage.
    Integrated with data connectors, feature store, OpenAI agents, and Universal ML Agent.
    Full Agent-First support for template-free AutoML.
    ENHANCED: Hybrid local/agent mode for optimal performance vs quality trade-off
    """
    
    def __init__(self, config: Union[Dict[str, Any], 'AutoMLConfig']):
        """
        Initialize preprocessor with support for AutoMLConfig and Agent-First.
        
        Args:
            config: Configuration dictionary or AutoMLConfig instance
        """
        # Handle both dict and AutoMLConfig instances
        if isinstance(config, dict):
            self.config = config
            self.enable_intelligent_cleaning = config.get('enable_intelligent_cleaning', False)
            self.enable_agent_first = config.get('enable_agent_first', False)
            self.enable_hybrid_mode = config.get('enable_hybrid_mode', False)  # NEW: Hybrid mode support
        else:  # AutoMLConfig instance
            self.config = config.to_dict() if hasattr(config, 'to_dict') else asdict(config)
            self.enable_intelligent_cleaning = getattr(config, 'enable_intelligent_cleaning', False)
            self.enable_agent_first = getattr(config, 'enable_agent_first', False)
            self.enable_hybrid_mode = getattr(config, 'enable_hybrid_mode', False)  # NEW: Hybrid mode support
            # Get Agent-First config if available
            if hasattr(config, 'agent_first'):
                self.agent_first_config = config.agent_first
            else:
                self.agent_first_config = None
        
        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.text_features = []
        self.binary_features = []
        self.ordinal_features = []
        
        self.pipeline = None
        self.feature_types = {}
        self.reference_stats = {}
        self.quality_report = {}
        self.drift_report = {}
        self.cleaning_report = {}  # For intelligent cleaning results
        self.ml_context = None  # For Agent-First ML context
        self.local_stats = {}  # NEW: Store local statistics for hybrid mode
        
        # Advanced options
        self.handle_outliers = self.config.get('handle_outliers', True)
        self.outlier_method = self.config.get('outlier_method', 'iqr')
        self.outlier_threshold = self.config.get('outlier_threshold', 1.5)
        self.imputation_method = self.config.get('imputation_method', 'median')
        self.scaling_method = self.config.get('scaling_method', 'robust')
        self.encoding_method = self.config.get('encoding_method', 'onehot')
        self.max_cardinality = self.config.get('high_cardinality_threshold', 20)
        self.rare_threshold = self.config.get('rare_category_threshold', 0.01)
        self.enable_quality_checks = self.config.get('enable_quality_checks', True)
        self.enable_drift_detection = self.config.get('enable_drift_detection', False)
        
        # Initialize Agent-First components if enabled
        self.universal_agent = None
        self.context_detector = None
        self.config_generator = None
        self.adaptive_templates = None
        self.orchestrator = None
        
        if self.enable_agent_first:
            self._init_agent_first_components()
        
        # Connector integration
        self.connector = None
        if self.config.get('connector_config'):
            self._init_connector(self.config['connector_config'])
        
        # Feature store integration
        self.feature_store = None
        if self.config.get('feature_store_config'):
            self._init_feature_store(self.config['feature_store_config'])
    
    def _init_agent_first_components(self):
        """Initialize Agent-First components for template-free AutoML."""
        try:
            from .agents import (
                UniversalMLAgent, 
                IntelligentContextDetector,
                IntelligentConfigGenerator,
                AdaptiveTemplateSystem,
                DataCleaningOrchestrator,
                AgentConfig
            )
            
            # Create agent configuration with hybrid mode support
            agent_config = AgentConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY") or self.config.get('openai_api_key'),
                model=self.config.get('openai_model', 'gpt-4-1106-preview'),
                openai_model=self.config.get('openai_model', 'gpt-4-1106-preview'),
                openai_enable_web_search=True,
                openai_enable_file_operations=True,
            )
            
            # Initialize all Agent-First components
            self.universal_agent = UniversalMLAgent(agent_config)
            self.context_detector = IntelligentContextDetector()
            self.config_generator = IntelligentConfigGenerator()
            self.adaptive_templates = AdaptiveTemplateSystem()
            self.orchestrator = DataCleaningOrchestrator(agent_config, self.config)
            
            logger.info("✅ Agent-First components initialized successfully")
            if self.enable_hybrid_mode:
                logger.info("🔄 Hybrid mode enabled for intelligent decision making")
            
        except ImportError as e:
            logger.warning(f"Agent-First components not available: {e}")
            self.enable_agent_first = False
        except Exception as e:
            logger.error(f"Failed to initialize Agent-First components: {e}")
            self.enable_agent_first = False
    
    def _init_connector(self, connector_config: Dict[str, Any]):
        """Initialize data connector."""
        try:
            from .api.connectors import ConnectorFactory, ConnectionConfig
            
            conn_config = ConnectionConfig(
                connection_type=connector_config.get('type', 'postgresql'),
                **connector_config.get('params', {})
            )
            
            self.connector = ConnectorFactory.create_connector(conn_config)
            logger.info(f"Initialized connector: {conn_config.connection_type}")
        except Exception as e:
            logger.warning(f"Failed to initialize connector: {e}")
            self.connector = None
    
    def _init_feature_store(self, feature_store_config: Dict[str, Any]):
        """Initialize feature store."""
        try:
            from .api.feature_store import FeatureStore
            
            self.feature_store = FeatureStore(feature_store_config)
            logger.info("Initialized feature store")
        except Exception as e:
            logger.warning(f"Failed to initialize feature store: {e}")
            self.feature_store = None
    
    def _calculate_local_stats(self, df: pd.DataFrame, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate local statistics for hybrid mode decision making.
        These stats are used to decide whether to use agents or local cleaning.
        
        Args:
            df: Input dataframe
            user_context: User context with sector information
            
        Returns:
            Dictionary of local statistics
        """
        logger.info("📊 Calculating local statistics for hybrid mode")
        
        stats = {
            'missing_ratio': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
            'duplicate_ratio': df.duplicated().mean(),
            'quality_score': 100.0,
            'complexity_score': 0.0,
            'has_sentinel_values': False,
            'has_negative_prices': False,
            'outlier_ratio': 0.0,
            'high_cardinality_count': 0,
            'constant_columns': [],
            'shape': df.shape,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }
        
        # Check for sentinel values in retail context
        if user_context.get('secteur_activite') == 'retail':
            sentinel_values = self.config.get('sentinel_values', [-999, -1, 0, 9999])
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].isin(sentinel_values).any():
                    stats['has_sentinel_values'] = True
                    break
        
        # Check for negative prices
        price_columns = [col for col in df.columns if 'price' in col.lower() or 'prix' in col.lower()]
        for col in price_columns:
            if pd.api.types.is_numeric_dtype(df[col]) and (df[col] < 0).any():
                stats['has_negative_prices'] = True
                break
        
        # Calculate quality score based on various factors
        missing_penalty = min(30, stats['missing_ratio'] * 100)
        duplicate_penalty = min(20, stats['duplicate_ratio'] * 100)
        stats['quality_score'] -= (missing_penalty + duplicate_penalty)
        
        # Calculate complexity score
        if df.shape[1] > 50:
            stats['complexity_score'] += 0.3
        if stats['missing_ratio'] > 0.3:
            stats['complexity_score'] += 0.3
        if stats['has_sentinel_values']:
            stats['complexity_score'] += 0.2
        if stats['has_negative_prices']:
            stats['complexity_score'] += 0.2
        
        # Check for high cardinality
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > self.max_cardinality:
                stats['high_cardinality_count'] += 1
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                stats['constant_columns'].append(col)
        
        # Calculate outlier ratio for numeric columns
        outlier_counts = []
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
            outlier_counts.append(outliers)
        
        if outlier_counts:
            stats['outlier_ratio'] = sum(outlier_counts) / (len(df) * len(outlier_counts))
        
        # Store for later use
        self.local_stats = stats
        
        logger.info(f"📊 Local stats: Quality={stats['quality_score']:.1f}, "
                   f"Missing={stats['missing_ratio']:.1%}, "
                   f"Complexity={stats['complexity_score']:.2f}")
        
        return stats
    
    async def agent_first_automl(
        self, 
        df: pd.DataFrame, 
        target_col: Optional[str] = None,
        user_hints: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete Agent-First AutoML without templates.
        This is the main entry point for template-free ML.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            user_hints: Optional user hints (sector, keywords, etc.)
            constraints: Optional constraints (time_budget, memory_limit, etc.)
            
        Returns:
            Tuple of (processed_dataframe, ml_pipeline_result)
        """
        if not self.enable_agent_first or not self.universal_agent:
            raise ValueError("Agent-First mode not enabled or not initialized")
        
        logger.info("🚀 Starting Agent-First AutoML without templates")
        
        # Execute complete AutoML pipeline
        result = await self.universal_agent.automl_without_templates(
            df=df,
            target_col=target_col,
            user_hints=user_hints,
            constraints=constraints
        )
        
        # Store ML context
        self.ml_context = {
            "problem_type": result.context_detected.problem_type,
            "confidence": result.context_detected.confidence,
            "business_sector": result.context_detected.business_sector,
            "config_used": result.config_used.to_dict(),
            "performance": result.performance_metrics
        }
        
        # Store cleaning report
        self.cleaning_report = result.cleaning_report
        
        logger.info(f"✅ Agent-First AutoML completed: {result.context_detected.problem_type}")
        
        return result.cleaned_data, result
    
    async def detect_ml_context(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect ML context using Agent-First intelligence.
        
        Returns:
            Dictionary with ML context (problem_type, confidence, etc.)
        """
        if not self.enable_agent_first or not self.context_detector:
            raise ValueError("Agent-First mode not enabled")
        
        context = await self.context_detector.detect_ml_context(df, target_col)
        
        self.ml_context = {
            "problem_type": context.problem_type,
            "confidence": context.confidence,
            "detected_patterns": context.detected_patterns,
            "business_sector": context.business_sector,
            "temporal_aspect": context.temporal_aspect,
            "imbalance_detected": context.imbalance_detected,
            "recommended_config": context.recommended_config,
            "reasoning": context.reasoning
        }
        
        return self.ml_context
    
    async def generate_optimal_config(
        self, 
        df: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate optimal configuration using Agent-First intelligence.
        
        Returns:
            Optimal configuration dictionary
        """
        if not self.enable_agent_first or not self.config_generator:
            raise ValueError("Agent-First mode not enabled")
        
        # Use stored context if not provided
        if context is None:
            context = self.ml_context or {}
        
        config = await self.config_generator.generate_config(
            df=df,
            context=context,
            constraints=self.config.get('constraints', {}),
            user_preferences=self.config.get('user_preferences', {})
        )
        
        return config.to_dict()
    
    async def intelligent_clean(self, df: pd.DataFrame, user_context: Dict[str, Any]) -> pd.DataFrame:
        """
        API finale pour le nettoyage intelligent avec agents OpenAI.
        ENHANCED: Support for hybrid mode with local statistics calculation
        
        Args:
            df: DataFrame à nettoyer
            user_context: Contexte utilisateur avec:
                - secteur_activite: secteur d'activité (ex: "finance", "retail")
                - target_variable: nom de la variable cible (ex: "churn")  
                - contexte_metier: description métier (ex: "Prédiction attrition clients B2B")
                
        Returns:
            DataFrame nettoyé
        """
        if not self.enable_intelligent_cleaning:
            logger.info("Intelligent cleaning disabled, using standard preprocessing")
            return self.fit_transform(df)
        
        # Calculate local statistics for hybrid mode if enabled
        if self.enable_hybrid_mode:
            logger.info("🔄 Hybrid mode enabled - calculating local statistics")
            local_stats = self._calculate_local_stats(df, user_context)
            
            # Add local stats to user context for orchestrator
            user_context['local_stats'] = local_stats
            user_context['enable_hybrid_mode'] = True
            
            # Log hybrid decision factors
            logger.info(f"📊 Hybrid decision factors:")
            logger.info(f"   - Quality Score: {local_stats['quality_score']:.1f}/100")
            logger.info(f"   - Complexity: {local_stats['complexity_score']:.2f}")
            logger.info(f"   - Missing Ratio: {local_stats['missing_ratio']:.1%}")
            logger.info(f"   - Has Sentinel Values: {local_stats['has_sentinel_values']}")
            logger.info(f"   - Has Negative Prices: {local_stats['has_negative_prices']}")
        
        # If Agent-First is enabled, use Universal Agent
        if self.enable_agent_first and self.universal_agent:
            logger.info("Using Agent-First for intelligent cleaning")
            
            # Convert user_context format for Agent-First
            user_hints = {
                'sector': user_context.get('secteur_activite', 'general'),
                'keywords': [user_context.get('contexte_metier', '')],
                'target': user_context.get('target_variable'),
                'local_stats': user_context.get('local_stats', {})  # Include local stats
            }
            
            # Use Universal Agent for complete pipeline
            cleaned_df, result = await self.agent_first_automl(
                df=df,
                target_col=user_context.get('target_variable'),
                user_hints=user_hints
            )
            
            return cleaned_df if cleaned_df is not None else df
        
        # Otherwise use traditional intelligent cleaning with orchestrator
        if self.orchestrator:
            try:
                logger.info(f"Starting intelligent cleaning for sector: {user_context.get('secteur_activite')}")
                
                # The orchestrator will use local_stats if present in user_context
                # to make hybrid decisions about using agents vs local cleaning
                cleaned_df, report = await self.orchestrator.clean_dataset(
                    df=df,
                    user_context=user_context,
                    use_intelligence=True  # Enable Agent-First features in orchestrator
                )
                
                self.cleaning_report = report
                
                # Log results based on mode used
                if report.get('mode') == 'local_enhanced':
                    logger.info(f"✅ Cleaning completed using LOCAL mode (quality sufficient)")
                    logger.info(f"   Quality: {report.get('quality_before', 0):.1f} → {report.get('quality_after', 0):.1f}")
                else:
                    quality_improvement = report.get('summary', {}).get('quality_improvement', 0)
                    logger.info(f"✅ Intelligent cleaning completed with AGENTS")
                    logger.info(f"   Quality improved by: {quality_improvement:.1f} points")
                
                # Log hybrid statistics if available
                if self.enable_hybrid_mode and 'performance' in report:
                    hybrid_stats = report.get('performance', {}).get('hybrid_decisions', {})
                    if hybrid_stats:
                        logger.info(f"🔄 Hybrid decisions made:")
                        logger.info(f"   - Agent calls: {hybrid_stats.get('agent', 0)}")
                        logger.info(f"   - Local calls: {hybrid_stats.get('local', 0)}")
                
                return cleaned_df
                
            except Exception as e:
                logger.error(f"Intelligent cleaning failed: {e}")
                logger.info("Falling back to standard preprocessing")
                return self.fit_transform(df)
        else:
            logger.warning("No orchestrator available, using standard preprocessing")
            return self.fit_transform(df)
    
    def load_data_from_connector(self, query: str = None, table_name: str = None) -> pd.DataFrame:
        """Load data using configured connector."""
        if not self.connector:
            raise ValueError("No connector configured")
        
        try:
            if query:
                df = self.connector.query(query)
            elif table_name:
                df = self.connector.read_table(table_name)
            else:
                raise ValueError("Either query or table_name must be provided")
            
            logger.info(f"Loaded {len(df)} rows from connector")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from connector: {e}")
            raise
    
    def save_to_feature_store(self, df: pd.DataFrame, feature_set_name: str) -> bool:
        """Save processed features to feature store."""
        if not self.feature_store:
            logger.warning("No feature store configured")
            return False
        
        try:
            # Register feature set if needed
            from .feature_store import FeatureSet, FeatureDefinition
            
            features = []
            for col in df.columns:
                features.append(FeatureDefinition(
                    name=col,
                    dtype=str(df[col].dtype),
                    description=f"Feature {col}"
                ))
            
            feature_set = FeatureSet(
                name=feature_set_name,
                features=features,
                entity_key="entity_id" if "entity_id" in df.columns else df.columns[0]
            )
            
            self.feature_store.register_feature_set(feature_set)
            
            # Write features
            return self.feature_store.write_features(feature_set_name, df)
            
        except Exception as e:
            logger.error(f"Failed to save to feature store: {e}")
            return False
    
    def detect_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically detect and categorize feature types.
        Enhanced detection with binary, ordinal, and high-cardinality handling.
        """
        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.text_features = []
        self.binary_features = []
        self.ordinal_features = []
        
        for col in df.columns:
            dtype = df[col].dtype
            nunique = df[col].nunique()
            non_null_count = df[col].notna().sum()
            
            # Skip if all null
            if non_null_count == 0:
                logger.warning(f"Column {col} is all null, skipping")
                continue
            
            # Detect datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.datetime_features.append(col)
            elif df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].dropna().iloc[:100], errors='raise')
                    self.datetime_features.append(col)
                except:
                    # Check if text or categorical
                    sample_values = df[col].dropna().astype(str)
                    if len(sample_values) > 0:
                        avg_len = sample_values.str.len().mean()
                        avg_words = sample_values.str.split().str.len().mean()
                        
                        # Heuristics for text detection
                        col_lower = col.lower()
                        is_text_column = ('text' in col_lower or 'description' in col_lower or 
                                        'comment' in col_lower or 'review' in col_lower)
                        
                        if is_text_column or (avg_len > 50 or avg_words > 5):
                            self.text_features.append(col)
                        elif nunique == 2:
                            self.binary_features.append(col)
                        elif nunique < self.max_cardinality:
                            self.categorical_features.append(col)
                        else:
                            # High cardinality - treat as text
                            self.text_features.append(col)
            
            # Numeric features
            elif pd.api.types.is_numeric_dtype(df[col]):
                if nunique == 2:
                    self.binary_features.append(col)
                elif nunique < 10 and df[col].dtype in ['int64', 'int32', 'int16']:
                    # Check if ordinal (sequential integers)
                    unique_sorted = sorted(df[col].dropna().unique())
                    if len(unique_sorted) > 1:
                        diffs = np.diff(unique_sorted)
                        if np.all(diffs == 1):
                            self.ordinal_features.append(col)
                        else:
                            self.categorical_features.append(col)
                    else:
                        self.categorical_features.append(col)
                else:
                    self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
        
        self.feature_types = {
            'numeric': self.numeric_features,
            'categorical': self.categorical_features,
            'datetime': self.datetime_features,
            'text': self.text_features,
            'binary': self.binary_features,
            'ordinal': self.ordinal_features
        }
        
        logger.info(f"Feature types detected: "
                   f"numeric={len(self.numeric_features)}, "
                   f"categorical={len(self.categorical_features)}, "
                   f"datetime={len(self.datetime_features)}, "
                   f"text={len(self.text_features)}, "
                   f"binary={len(self.binary_features)}, "
                   f"ordinal={len(self.ordinal_features)}")
        
        return self.feature_types
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment.
        Enhanced with Agent-First context if available.
        """
        quality_report = {
            'valid': True,
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'issues': [],
            'warnings': [],
            'statistics': {},
            'quality_score': 100.0
        }
        
        # Add ML context if available (from Agent-First)
        if self.ml_context:
            quality_report['ml_context'] = self.ml_context
        
        # Add local stats if available (from hybrid mode)
        if self.local_stats:
            quality_report['local_stats'] = self.local_stats
        
        # Check for empty dataframe
        if df.empty:
            quality_report['valid'] = False
            quality_report['issues'].append("DataFrame is empty")
            quality_report['quality_score'] = 0
            return quality_report
        
        # Check for duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            dup_ratio = n_duplicates / len(df)
            quality_report['warnings'].append(f"Found {n_duplicates} duplicate rows ({dup_ratio:.2%})")
            quality_report['quality_score'] -= min(20, dup_ratio * 100)
        
        # Check for duplicate columns
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        if dup_cols:
            quality_report['issues'].append(f"Duplicate columns: {dup_cols}")
            quality_report['quality_score'] -= 10
        
        # Check missing values
        missing_report = {}
        for col in df.columns:
            missing_ratio = df[col].isnull().mean()
            if missing_ratio > 0:
                missing_report[col] = missing_ratio
                
                if missing_ratio > self.config.get('max_missing_ratio', 0.5):
                    quality_report['issues'].append(f"Column {col} has {missing_ratio:.2%} missing values")
                    quality_report['quality_score'] -= 5
                elif missing_ratio > 0.2:
                    quality_report['warnings'].append(f"Column {col} has {missing_ratio:.2%} missing values")
        
        quality_report['statistics']['missing_values'] = missing_report
        
        # Check for constant columns
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_cols.append(col)
        
        if constant_cols:
            quality_report['warnings'].append(f"Constant columns: {constant_cols}")
            quality_report['quality_score'] -= len(constant_cols) * 2
        
        # Check for outliers in numeric columns
        outlier_report = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            outliers = self._detect_outliers_column(df[col])
            outlier_ratio = outliers.sum() / len(df)
            if outlier_ratio > 0.1:
                outlier_report[col] = outlier_ratio
                quality_report['warnings'].append(f"Column {col} has {outlier_ratio:.2%} outliers")
        
        quality_report['statistics']['outliers'] = outlier_report
        
        # Check for high cardinality categorical
        high_cardinality = []
        for col in df.select_dtypes(include=['object']).columns:
            cardinality = df[col].nunique()
            if cardinality > self.max_cardinality:
                high_cardinality.append((col, cardinality))
        
        if high_cardinality:
            quality_report['warnings'].append(f"High cardinality columns: {high_cardinality}")
        
        # Check for data type issues
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    # Try to convert to numeric
                    pd.to_numeric(df[col], errors='raise')
                    quality_report['warnings'].append(f"Column {col} stored as object but contains numeric data")
            except:
                pass
        
        # Calculate final quality score
        quality_report['quality_score'] = max(0, quality_report['quality_score'])
        
        # Determine validity
        if quality_report['quality_score'] < 50 or len(quality_report['issues']) > 5:
            quality_report['valid'] = False
        
        self.quality_report = quality_report
        return quality_report
    
    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame,
                    sensitivity: float = 0.05) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.
        """
        drift_report = {
            'drift_detected': False,
            'drifted_features': [],
            'drift_scores': {},
            'drift_tests': {},
            'summary': {}
        }
        
        # Ensure same columns
        common_cols = list(set(reference_df.columns) & set(current_df.columns))
        
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                # Kolmogorov-Smirnov test for numerical
                ref_data = reference_df[col].dropna()
                curr_data = current_df[col].dropna()
                
                if len(ref_data) > 0 and len(curr_data) > 0:
                    statistic, p_value = ks_2samp(ref_data, curr_data)
                    
                    drift_report['drift_scores'][col] = 1 - p_value
                    drift_report['drift_tests'][col] = {
                        'test': 'ks_2samp',
                        'statistic': statistic,
                        'p_value': p_value
                    }
                    
                    if p_value < sensitivity:
                        drift_report['drifted_features'].append(col)
                        drift_report['drift_detected'] = True
                        
                    # Also check distribution statistics
                    ref_mean = ref_data.mean()
                    curr_mean = curr_data.mean()
                    mean_shift = abs(curr_mean - ref_mean) / (ref_mean + 1e-10)
                    
                    if mean_shift > 0.2:  # 20% shift in mean
                        drift_report['summary'][col] = f"Mean shifted by {mean_shift:.2%}"
            
            else:
                # Chi-square test for categorical
                ref_counts = reference_df[col].value_counts()
                curr_counts = current_df[col].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                
                # Build contingency table
                contingency = pd.DataFrame({
                    'reference': [ref_counts.get(cat, 0) for cat in all_categories],
                    'current': [curr_counts.get(cat, 0) for cat in all_categories]
                })
                
                if contingency.shape[0] > 1:
                    chi2, p_value, dof, expected = chi2_contingency(contingency.values.T)
                    
                    drift_report['drift_scores'][col] = 1 - p_value
                    drift_report['drift_tests'][col] = {
                        'test': 'chi2',
                        'statistic': chi2,
                        'p_value': p_value
                    }
                    
                    if p_value < sensitivity:
                        drift_report['drifted_features'].append(col)
                        drift_report['drift_detected'] = True
        
        # Calculate PSI (Population Stability Index) for numeric features
        psi_scores = {}
        for col in self.numeric_features:
            if col in common_cols:
                psi = self._calculate_psi(reference_df[col], current_df[col])
                psi_scores[col] = psi
                
                if psi > 0.25:  # Significant population shift
                    if col not in drift_report['drifted_features']:
                        drift_report['drifted_features'].append(col)
                        drift_report['drift_detected'] = True
        
        drift_report['psi_scores'] = psi_scores
        
        self.drift_report = drift_report
        return drift_report
    
    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Handle missing values
        expected = expected.dropna()
        actual = actual.dropna()
        
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        
        # Create bins
        breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Categorize
        expected_bins = pd.cut(expected, breakpoints).value_counts().sort_index()
        actual_bins = pd.cut(actual, breakpoints).value_counts().sort_index()
        
        # Calculate percentages
        expected_percents = (expected_bins / len(expected)).replace(0, 0.0001)
        actual_percents = (actual_bins / len(actual)).replace(0, 0.0001)
        
        # PSI calculation
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return float(psi)
    
    def _detect_outliers_column(self, series: pd.Series) -> pd.Series:
        """Detect outliers in a single column."""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return pd.Series([False] * len(series), index=series.index)
        
        if self.outlier_method == 'iqr':
            Q1 = series_clean.quantile(0.25)
            Q3 = series_clean.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.outlier_threshold * IQR
            upper = Q3 + self.outlier_threshold * IQR
            return (series < lower) | (series > upper)
        
        elif self.outlier_method == 'zscore':
            z_scores = np.abs(stats.zscore(series_clean))
            threshold = 3
            outliers_clean = z_scores > threshold
            
            # Map back to original series
            outliers = pd.Series([False] * len(series), index=series.index)
            outliers[series.notna()] = outliers_clean
            return outliers
        
        elif self.outlier_method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=0.1, random_state=42)
            
            # Reshape for sklearn
            X = series_clean.values.reshape(-1, 1)
            outliers_clean = iso.fit_predict(X) == -1
            
            # Map back
            outliers = pd.Series([False] * len(series), index=series.index)
            outliers[series.notna()] = outliers_clean
            return outliers
        
        else:
            return pd.Series([False] * len(series), index=series.index)
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'clip') -> pd.DataFrame:
        """
        Handle outliers in numeric columns.
        
        Args:
            df: Input dataframe
            method: 'clip', 'remove', or 'transform'
        """
        df_cleaned = df.copy()
        
        for col in self.numeric_features:
            if col not in df_cleaned.columns:
                continue
            
            outliers = self._detect_outliers_column(df_cleaned[col])
            
            if outliers.sum() > 0:
                if method == 'clip':
                    # Clip to percentiles
                    lower = df_cleaned[col].quantile(0.01)
                    upper = df_cleaned[col].quantile(0.99)
                    df_cleaned[col] = df_cleaned[col].clip(lower, upper)
                
                elif method == 'remove':
                    # Remove outlier rows
                    df_cleaned = df_cleaned[~outliers]
                
                elif method == 'transform':
                    # Apply log transformation
                    if (df_cleaned[col] > 0).all():
                        df_cleaned[col] = np.log1p(df_cleaned[col])
        
        return df_cleaned
    
    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced datetime feature creation."""
        df_new = df.copy()
        
        for col in self.datetime_features:
            if col not in df.columns:
                continue
            
            # Convert to datetime
            dt_col = pd.to_datetime(df_new[col], errors='coerce')
            
            # Basic components
            df_new[f'{col}_year'] = dt_col.dt.year
            df_new[f'{col}_month'] = dt_col.dt.month
            df_new[f'{col}_day'] = dt_col.dt.day
            df_new[f'{col}_dayofweek'] = dt_col.dt.dayofweek
            df_new[f'{col}_quarter'] = dt_col.dt.quarter
            df_new[f'{col}_dayofyear'] = dt_col.dt.dayofyear
            df_new[f'{col}_weekofyear'] = dt_col.dt.isocalendar().week
            df_new[f'{col}_hour'] = dt_col.dt.hour
            
            # Flags
            df_new[f'{col}_is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
            df_new[f'{col}_is_month_start'] = dt_col.dt.is_month_start.astype(int)
            df_new[f'{col}_is_month_end'] = dt_col.dt.is_month_end.astype(int)
            df_new[f'{col}_is_quarter_start'] = dt_col.dt.is_quarter_start.astype(int)
            df_new[f'{col}_is_quarter_end'] = dt_col.dt.is_quarter_end.astype(int)
            df_new[f'{col}_is_year_start'] = dt_col.dt.is_year_start.astype(int)
            df_new[f'{col}_is_year_end'] = dt_col.dt.is_year_end.astype(int)
            
            # Cyclical encoding
            df_new[f'{col}_month_sin'] = np.sin(2 * np.pi * dt_col.dt.month / 12)
            df_new[f'{col}_month_cos'] = np.cos(2 * np.pi * dt_col.dt.month / 12)
            df_new[f'{col}_day_sin'] = np.sin(2 * np.pi * dt_col.dt.day / 31)
            df_new[f'{col}_day_cos'] = np.cos(2 * np.pi * dt_col.dt.day / 31)
            df_new[f'{col}_hour_sin'] = np.sin(2 * np.pi * dt_col.dt.hour / 24)
            df_new[f'{col}_hour_cos'] = np.cos(2 * np.pi * dt_col.dt.hour / 24)
            
            # Add all new columns to numeric features
            new_cols = [c for c in df_new.columns if c.startswith(f'{col}_')]
            self.numeric_features.extend(new_cols)
            
            # Drop original datetime column
            df_new = df_new.drop(columns=[col])
        
        return df_new
    
    def handle_rare_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle rare categories in categorical columns."""
        df_new = df.copy()
        
        for col in self.categorical_features:
            if col not in df.columns:
                continue
            
            # Get value counts
            value_counts = df[col].value_counts()
            n_samples = len(df)
            
            # Find rare categories
            rare_categories = value_counts[value_counts / n_samples < self.rare_threshold].index
            
            if len(rare_categories) > 0:
                # Replace rare categories with 'Other'
                df_new.loc[df_new[col].isin(rare_categories), col] = 'Other'
                
                logger.info(f"Replaced {len(rare_categories)} rare categories in {col} with 'Other'")
        
        return df_new
    
    def create_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Create enhanced preprocessing pipeline."""
        
        # Detect feature types if not done
        if not self.feature_types:
            self.detect_feature_types(X)
        
        # Quality check if enabled
        if self.enable_quality_checks:
            quality_report = self.check_data_quality(X)
            if not quality_report['valid']:
                logger.warning(f"Data quality issues detected: {quality_report['issues']}")
        
        # Handle datetime features first
        if self.datetime_features:
            X = self.create_datetime_features(X)
        
        # Handle rare categories
        if self.categorical_features:
            X = self.handle_rare_categories(X)
        
        transformers = []
        
        # Numeric pipeline with advanced options
        if self.numeric_features:
            numeric_steps = []
            
            # Imputation
            if self.imputation_method == 'median':
                numeric_steps.append(('imputer', SimpleImputer(strategy='median')))
            elif self.imputation_method == 'mean':
                numeric_steps.append(('imputer', SimpleImputer(strategy='mean')))
            elif self.imputation_method == 'knn':
                numeric_steps.append(('imputer', KNNImputer(n_neighbors=5)))
            
            # Scaling
            if self.scaling_method == 'standard':
                numeric_steps.append(('scaler', StandardScaler()))
            elif self.scaling_method == 'robust':
                numeric_steps.append(('scaler', RobustScaler()))
            elif self.scaling_method == 'minmax':
                numeric_steps.append(('scaler', MinMaxScaler()))
            elif self.scaling_method == 'quantile':
                numeric_steps.append(('scaler', QuantileTransformer(output_distribution='normal')))
            elif self.scaling_method == 'power':
                numeric_steps.append(('scaler', PowerTransformer(method='yeo-johnson')))
            
            numeric_pipeline = Pipeline(numeric_steps)
            transformers.append(('numeric', numeric_pipeline, self.numeric_features))
        
        # Categorical pipeline
        if self.categorical_features:
            categorical_steps = [
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
            ]
            
            if self.encoding_method == 'onehot':
                categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            elif self.encoding_method == 'ordinal':
                categorical_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            
            categorical_pipeline = Pipeline(categorical_steps)
            transformers.append(('categorical', categorical_pipeline, self.categorical_features))
        
        # Binary features
        if self.binary_features:
            binary_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', LabelEncoder() if len(self.binary_features) == 1 else OneHotEncoder(drop='first', sparse_output=False))
            ])
            transformers.append(('binary', binary_pipeline, self.binary_features))
        
        # Ordinal features
        if self.ordinal_features:
            ordinal_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler())
            ])
            transformers.append(('ordinal', ordinal_pipeline, self.ordinal_features))
        
        # Text features with improved handling
        if self.text_features:
            for text_col in self.text_features:
                # Better n_components calculation
                n_samples = len(X)
                n_components = min(
                    50,  # Maximum components
                    max(1, n_samples // 10),  # Based on sample size
                    self.config.get('text_max_features', 100) - 1
                )
                
                text_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(
                        max_features=self.config.get('text_max_features', 100),
                        ngram_range=self.config.get('text_ngram_range', (1, 2)),
                        min_df=self.config.get('text_min_df', 2),
                        max_df=0.95
                    )),
                    ('svd', TruncatedSVD(n_components=n_components, random_state=42))
                ])
                transformers.append((f'text_{text_col}', text_pipeline, text_col))
        
        # Create and store pipeline
        self.pipeline = ColumnTransformer(transformers, remainder='passthrough')
        
        # Store reference statistics for drift detection
        if self.enable_drift_detection:
            self._calculate_reference_stats(X)
        
        return self.pipeline
    
    def _calculate_reference_stats(self, X: pd.DataFrame):
        """Calculate and store reference statistics for drift detection."""
        self.reference_stats = {}
        
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self.reference_stats[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'min': X[col].min(),
                    'max': X[col].max(),
                    'median': X[col].median(),
                    'q1': X[col].quantile(0.25),
                    'q3': X[col].quantile(0.75)
                }
            else:
                self.reference_stats[col] = {
                    'unique_values': X[col].nunique(),
                    'value_counts': X[col].value_counts().to_dict(),
                    'mode': X[col].mode()[0] if len(X[col].mode()) > 0 else None
                }
    
    def _get_scaler(self):
        """Get scaler based on configuration."""
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler(),
            'quantile': QuantileTransformer(output_distribution='normal'),
            'power': PowerTransformer(method='yeo-johnson')
        }
        return scalers.get(self.scaling_method, RobustScaler())
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform data with quality checks."""
        # Create pipeline if not exists
        if self.pipeline is None:
            self.create_pipeline(X)
        
        # Handle datetime features
        if self.datetime_features:
            X = self.create_datetime_features(X)
        
        # Handle outliers if configured
        if self.handle_outliers and self.numeric_features:
            X = self.handle_outliers(X, method='clip')
        
        # Fit and transform
        X_transformed = self.pipeline.fit_transform(X, y)
        
        logger.info(f"Data transformed: shape {X.shape} -> {X_transformed.shape}")
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted yet")
        
        # Quality check if enabled
        if self.enable_quality_checks:
            quality_report = self.check_data_quality(X)
            if not quality_report['valid']:
                logger.warning(f"Quality issues in transform data: {quality_report['issues']}")
        
        # Drift detection if enabled
        if self.enable_drift_detection and hasattr(self, 'reference_stats'):
            # Use stored reference stats for comparison
            self._check_drift_from_reference(X)
        
        # Handle datetime features
        if self.datetime_features:
            X = self.create_datetime_features(X)
        
        # Handle outliers if configured
        if self.handle_outliers and self.numeric_features:
            X = self.handle_outliers(X, method='clip')
        
        return self.pipeline.transform(X)
    
    def _check_drift_from_reference(self, X: pd.DataFrame):
        """Check for drift from reference statistics."""
        drift_warnings = []
        
        for col in X.columns:
            if col in self.reference_stats:
                ref_stats = self.reference_stats[col]
                
                if pd.api.types.is_numeric_dtype(X[col]):
                    current_mean = X[col].mean()
                    current_std = X[col].std()
                    
                    # Check for mean shift
                    if ref_stats['std'] > 0:
                        z_score = abs(current_mean - ref_stats['mean']) / ref_stats['std']
                        if z_score > 3:
                            drift_warnings.append(f"Significant mean shift in {col}: z-score={z_score:.2f}")
                    
                    # Check for variance change
                    if ref_stats['std'] > 0:
                        std_ratio = current_std / ref_stats['std']
                        if std_ratio < 0.5 or std_ratio > 2:
                            drift_warnings.append(f"Significant variance change in {col}: ratio={std_ratio:.2f}")
        
        if drift_warnings:
            logger.warning(f"Drift detected: {drift_warnings}")
    
    def get_ml_context_summary(self) -> str:
        """Get summary of detected ML context (Agent-First)."""
        if not self.ml_context:
            return "No ML context detected yet. Run detect_ml_context() or agent_first_automl() first."
        
        summary = f"""
📊 ML Context Summary
====================
Problem Type: {self.ml_context.get('problem_type', 'Unknown')}
Confidence: {self.ml_context.get('confidence', 0):.1%}
Business Sector: {self.ml_context.get('business_sector', 'General')}
Temporal Data: {'Yes' if self.ml_context.get('temporal_aspect') else 'No'}
Imbalance: {'Detected' if self.ml_context.get('imbalance_detected') else 'Not detected'}

Detected Patterns:
{chr(10).join('- ' + p for p in self.ml_context.get('detected_patterns', [])[:5])}

Reasoning:
{self.ml_context.get('reasoning', 'N/A')}
"""
        return summary
    
    def get_hybrid_mode_summary(self) -> str:
        """Get summary of hybrid mode statistics and decisions."""
        if not self.enable_hybrid_mode:
            return "Hybrid mode is not enabled."
        
        if not self.local_stats:
            return "No local statistics calculated yet. Run intelligent_clean() first."
        
        summary = f"""
🔄 Hybrid Mode Summary
======================
Status: {'Enabled' if self.enable_hybrid_mode else 'Disabled'}

Local Statistics:
- Quality Score: {self.local_stats.get('quality_score', 0):.1f}/100
- Complexity Score: {self.local_stats.get('complexity_score', 0):.2f}
- Missing Ratio: {self.local_stats.get('missing_ratio', 0):.1%}
- Duplicate Ratio: {self.local_stats.get('duplicate_ratio', 0):.1%}
- Has Sentinel Values: {self.local_stats.get('has_sentinel_values', False)}
- Has Negative Prices: {self.local_stats.get('has_negative_prices', False)}
- Outlier Ratio: {self.local_stats.get('outlier_ratio', 0):.1%}
- High Cardinality Count: {self.local_stats.get('high_cardinality_count', 0)}
- Constant Columns: {len(self.local_stats.get('constant_columns', []))}

Cleaning Report:
- Mode Used: {self.cleaning_report.get('mode', 'N/A')}
"""
        
        if 'performance' in self.cleaning_report:
            hybrid_stats = self.cleaning_report.get('performance', {}).get('hybrid_decisions', {})
            if hybrid_stats:
                summary += f"""- Agent Decisions: {hybrid_stats.get('agent', 0)}
- Local Decisions: {hybrid_stats.get('local', 0)}
"""
        
        return summary


# Convenience functions
def handle_imbalance(X: np.ndarray, y: np.ndarray, 
                    method: str = 'auto',
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced imbalance handling with automatic method selection.
    """
    # Check imbalance ratio
    from collections import Counter
    class_counts = Counter(y)
    min_class = min(class_counts.values())
    max_class = max(class_counts.values())
    imbalance_ratio = max_class / min_class
    
    # Auto-select method based on imbalance ratio
    if method == 'auto':
        if imbalance_ratio < 3:
            method = 'none'
        elif imbalance_ratio < 10:
            method = 'smote'
        else:
            method = 'smoteenn'
    
    logger.info(f"Handling imbalance with method: {method} (ratio: {imbalance_ratio:.2f})")
    
    if method == 'none' or method == 'class_weight':
        return X, y
    
    try:
        if method == 'smote':
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=random_state)
        elif method == 'adasyn':
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(random_state=random_state)
        elif method == 'borderline':
            from imblearn.over_sampling import BorderlineSMOTE
            sampler = BorderlineSMOTE(random_state=random_state)
        elif method == 'smoteenn':
            from imblearn.combine import SMOTEENN
            sampler = SMOTEENN(random_state=random_state)
        elif method == 'smotetomek':
            from imblearn.combine import SMOTETomek
            sampler = SMOTETomek(random_state=random_state)
        else:
            logger.warning(f"Unknown method {method}, returning original data")
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info(f"Resampled data shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled
        
    except ImportError as e:
        logger.warning(f"Imbalance method {method} not available: {e}")
        return X, y
    except Exception as e:
        logger.error(f"Failed to handle imbalance: {e}")
        return X, y


def create_lag_features(df: pd.DataFrame, 
                       target_col: str,
                       lag_periods: List[int] = None,
                       rolling_windows: List[int] = None) -> pd.DataFrame:
    """
    Enhanced lag feature creation for time series.
    """
    df_lagged = df.copy()
    
    if lag_periods is None:
        lag_periods = [1, 2, 3, 7, 14, 30]
    
    if rolling_windows is None:
        rolling_windows = [3, 7, 14, 30]
    
    # Create lag features
    for lag in lag_periods:
        df_lagged[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in rolling_windows:
        df_lagged[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
        df_lagged[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
        df_lagged[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
        df_lagged[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
        df_lagged[f'{target_col}_rolling_median_{window}'] = df[target_col].rolling(window).median()
    
    # Exponential weighted statistics
    for span in [7, 30]:
        df_lagged[f'{target_col}_ewm_mean_{span}'] = df[target_col].ewm(span=span).mean()
        df_lagged[f'{target_col}_ewm_std_{span}'] = df[target_col].ewm(span=span).std()
    
    # Difference features
    for diff in [1, 7]:
        df_lagged[f'{target_col}_diff_{diff}'] = df[target_col].diff(diff)
    
    # Percentage change
    for period in [1, 7]:
        df_lagged[f'{target_col}_pct_change_{period}'] = df[target_col].pct_change(period)
    
    return df_lagged


def validate_data(df: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Enhanced data validation with detailed reporting.
    """
    if config is None:
        config = {}
    
    preprocessor = EnhancedDataPreprocessor(config)
    quality_report = preprocessor.check_data_quality(df)
    
    # Add feature type detection
    feature_types = preprocessor.detect_feature_types(df)
    quality_report['feature_types'] = feature_types
    
    # Add memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB
    quality_report['memory_usage_mb'] = memory_usage
    
    # Add data shape
    quality_report['shape'] = df.shape
    
    return quality_report


# Alias for backward compatibility
DataPreprocessor = EnhancedDataPreprocessor


# Agent-First convenience functions
async def automl_without_templates(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    config: Optional[Union[Dict, 'AutoMLConfig']] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Quick Agent-First AutoML without any templates.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        config: Optional configuration
        
    Returns:
        Tuple of (processed_dataframe, ml_pipeline_result)
    """
    if config is None:
        config = {'enable_agent_first': True}
    
    preprocessor = EnhancedDataPreprocessor(config)
    
    if not preprocessor.enable_agent_first:
        raise ValueError("Agent-First mode must be enabled in config")
    
    return await preprocessor.agent_first_automl(df, target_col)


async def detect_and_clean(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    enable_hybrid: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect ML context and clean data automatically using Agent-First.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        enable_hybrid: Enable hybrid local/agent mode
    
    Returns:
        Tuple of (cleaned_dataframe, ml_context)
    """
    config = {
        'enable_agent_first': True,
        'enable_intelligent_cleaning': True,
        'enable_hybrid_mode': enable_hybrid
    }
    
    preprocessor = EnhancedDataPreprocessor(config)
    
    # Detect context
    ml_context = await preprocessor.detect_ml_context(df, target_col)
    
    # Clean based on context
    user_context = {
        'secteur_activite': ml_context.get('business_sector', 'general'),
        'target_variable': target_col,
        'contexte_metier': ml_context.get('problem_type', 'unknown')
    }
    
    cleaned_df = await preprocessor.intelligent_clean(df, user_context)
    
    return cleaned_df, ml_context


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_classes=2, random_state=42)
    
    # Add some categorical and datetime features
    df = pd.DataFrame(X, columns=[f'num_{i}' for i in range(20)])
    df['cat_1'] = np.random.choice(['A', 'B', 'C', 'D'], size=1000)
    df['cat_2'] = np.random.choice(['X', 'Y', 'Z'], size=1000)
    df['date_1'] = pd.date_range('2023-01-01', periods=1000, freq='D')
    df['text_1'] = ['sample text ' * np.random.randint(1, 5) for _ in range(1000)]
    df['target'] = y
    
    # Create preprocessor with Agent-First and Hybrid mode enabled
    config = {
        'handle_outliers': True,
        'outlier_method': 'iqr',
        'scaling_method': 'robust',
        'enable_quality_checks': True,
        'enable_drift_detection': True,
        'enable_intelligent_cleaning': True,
        'enable_agent_first': True,  # Enable Agent-First
        'enable_hybrid_mode': True   # Enable Hybrid mode
    }
    
    preprocessor = EnhancedDataPreprocessor(config)
    
    # Check data quality
    quality_report = preprocessor.check_data_quality(df)
    print(f"Data quality score: {quality_report['quality_score']:.1f}")
    
    # Example of Agent-First AutoML (requires async)
    async def test_agent_first():
        # Complete AutoML without templates
        cleaned_df, result = await preprocessor.agent_first_automl(
            df=df,
            target_col='target',
            user_hints={'sector': 'finance', 'keywords': ['fraud', 'risk']}
        )
        
        print(f"Agent-First AutoML completed!")
        print(f"Detected problem: {result.context_detected.problem_type}")
        print(f"Confidence: {result.context_detected.confidence:.1%}")
        print(f"Best algorithm: {result.config_used.algorithms[0]}")
        
        # Get ML context summary
        print(preprocessor.get_ml_context_summary())
    
    # Example of intelligent cleaning with hybrid mode
    async def test_hybrid_cleaning():
        user_context = {
            "secteur_activite": "retail",  # Retail sector for testing hybrid mode
            "target_variable": "target",
            "contexte_metier": "Customer churn prediction for retail"
        }
        
        cleaned_df = await preprocessor.intelligent_clean(df, user_context)
        print(f"Intelligent cleaning completed: {cleaned_df.shape}")
        print(f"Cleaning report: {preprocessor.cleaning_report}")
        
        # Get hybrid mode summary
        print(preprocessor.get_hybrid_mode_summary())
    
    # Run if OpenAI API key is available
    if os.getenv('OPENAI_API_KEY'):
        import asyncio
        
        # Test Agent-First AutoML
        # asyncio.run(test_agent_first())
        
        # Test hybrid intelligent cleaning
        asyncio.run(test_hybrid_cleaning())
    else:
        # Standard cleaning without agents
        X_transformed = preprocessor.fit_transform(df.drop('target', axis=1), df['target'])
        print(f"Transformed shape: {X_transformed.shape}")
    
    # Detect drift (simulate with modified data)
    df_drift = df.copy()
    df_drift['num_0'] = df_drift['num_0'] * 2 + 10  # Introduce drift
    
    drift_report = preprocessor.detect_drift(df, df_drift)
    print(f"Drift detected: {drift_report['drift_detected']}")
    if drift_report['drifted_features']:
        print(f"Drifted features: {drift_report['drifted_features']}")
