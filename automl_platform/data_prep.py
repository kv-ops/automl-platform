"""
Enhanced Data Preparation Module
Includes data quality checks, drift detection, advanced preprocessing,
integration with connectors, feature store, and intelligent cleaning with OpenAI agents
"""

import pandas as pd
import numpy as np
import asyncio
import os
import yaml
from typing import Optional, List, Tuple, Union, Dict, Any
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

logger = logging.getLogger(__name__)


class EnhancedDataPreprocessor:
    """
    Advanced data preprocessing with quality checks and drift detection.
    No data leakage guaranteed through proper pipeline usage.
    Integrated with data connectors, feature store, and OpenAI agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
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
        
        # Advanced options
        self.handle_outliers = config.get('handle_outliers', True)
        self.outlier_method = config.get('outlier_method', 'iqr')
        self.outlier_threshold = config.get('outlier_threshold', 1.5)
        self.imputation_method = config.get('imputation_method', 'median')
        self.scaling_method = config.get('scaling_method', 'robust')
        self.encoding_method = config.get('encoding_method', 'onehot')
        self.max_cardinality = config.get('high_cardinality_threshold', 20)
        self.rare_threshold = config.get('rare_category_threshold', 0.01)
        self.enable_quality_checks = config.get('enable_quality_checks', True)
        self.enable_drift_detection = config.get('enable_drift_detection', False)
        
        # Connector integration
        self.connector = None
        if config.get('connector_config'):
            self._init_connector(config['connector_config'])
        
        # Feature store integration
        self.feature_store = None
        if config.get('feature_store_config'):
            self._init_feature_store(config['feature_store_config'])
    
    def _init_connector(self, connector_config: Dict[str, Any]):
        """Initialize data connector."""
        try:
            from automl_platform.api.connectors import ConnectorFactory, ConnectionConfig
            
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
            from automl_platform.api.feature_store import FeatureStore
            
            self.feature_store = FeatureStore(feature_store_config)
            logger.info("Initialized feature store")
        except Exception as e:
            logger.warning(f"Failed to initialize feature store: {e}")
            self.feature_store = None
    
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
            from automl_platform.feature_store import FeatureSet, FeatureDefinition
            
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
    
    async def intelligent_clean(self, df: pd.DataFrame, user_context: Dict[str, Any]) -> pd.DataFrame:
        """
        Use OpenAI agents for intelligent data cleaning
        
        Args:
            df: Input dataframe to clean
            user_context: User context including sector, target variable, etc.
            
        Returns:
            Cleaned dataframe
        """
        try:
            # Check if intelligent cleaning is enabled
            if not self.config.get('enable_intelligent_cleaning', False):
                logger.info("Intelligent cleaning not enabled, using standard cleaning")
                return self.fit_transform(df)
            
            # Check for OpenAI API key
            openai_api_key = self.config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                logger.warning("No OpenAI API key found, falling back to standard cleaning")
                return self.fit_transform(df)
            
            # Import agent components
            from automl_platform.agents import DataCleaningOrchestrator, AgentConfig
            
            # Create agent configuration
            agent_config = AgentConfig(
                openai_api_key=openai_api_key,
                model=self.config.get('openai_cleaning_model', 'gpt-4-1106-preview'),
                user_context=user_context,
                max_cost_per_dataset=self.config.get('max_cleaning_cost_per_dataset', 5.00),
                enable_web_search=self.config.get('enable_web_search', True),
                enable_file_operations=self.config.get('enable_file_operations', True)
            )
            
            # Create orchestrator
            orchestrator = DataCleaningOrchestrator(agent_config, self.config)
            
            # Run intelligent cleaning
            logger.info(f"Starting intelligent cleaning with OpenAI agents for sector: {user_context.get('secteur_activite', 'general')}")
            
            cleaned_df, report = await orchestrator.clean_dataset(df, user_context)
            
            # Log results
            logger.info(f"Intelligent cleaning completed. Quality score: {report.get('quality_metrics', {}).get('quality_score', 'N/A')}")
            
            # Store the cleaning report
            self.cleaning_report = report
            
            # Save to feature store if configured
            if self.feature_store and user_context.get('save_to_feature_store', False):
                feature_set_name = f"{user_context.get('secteur_activite', 'general')}_{user_context.get('target_variable', 'features')}"
                self.save_to_feature_store(cleaned_df, feature_set_name)
            
            return cleaned_df
            
        except ImportError as e:
            logger.error(f"Failed to import agent components: {e}")
            logger.info("Falling back to standard cleaning")
            return self.fit_transform(df)
            
        except Exception as e:
            logger.error(f"Error in intelligent cleaning: {e}")
            logger.info("Falling back to standard cleaning")
            return self.fit_transform(df)
        
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


class IntelligentDataCleaner:
    """
    Intelligent Data Cleaner using OpenAI agents
    Wrapper class for easy integration with existing architecture
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Intelligent Data Cleaner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'enable_intelligent_cleaning': True,
            'openai_cleaning_model': 'gpt-4-1106-preview',
            'max_cleaning_cost_per_dataset': 5.00
        }
        
        # Check for OpenAI API key
        self.openai_api_key = self.config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required for IntelligentDataCleaner")
        
        self.orchestrator = None
        self.preprocessor = EnhancedDataPreprocessor(self.config)
        
    async def clean(
        self, 
        df: pd.DataFrame, 
        user_context: Dict[str, Any],
        use_traditional_fallback: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean dataset using OpenAI agents
        
        Args:
            df: Input dataframe
            user_context: Context with sector, target variable, etc.
            use_traditional_fallback: Whether to fallback to traditional cleaning on error
            
        Returns:
            Tuple of (cleaned_dataframe, cleaning_report)
        """
        try:
            # Import agents
            from automl_platform.agents import DataCleaningOrchestrator, AgentConfig
            
            # Create agent configuration
            agent_config = AgentConfig(
                openai_api_key=self.openai_api_key,
                model=self.config.get('openai_cleaning_model', 'gpt-4-1106-preview'),
                user_context=user_context,
                max_cost_per_dataset=self.config.get('max_cleaning_cost_per_dataset', 5.00)
            )
            
            # Create orchestrator if not exists
            if not self.orchestrator:
                self.orchestrator = DataCleaningOrchestrator(agent_config, self.config)
            
            # Run intelligent cleaning
            logger.info(f"Starting intelligent cleaning for sector: {user_context.get('secteur_activite')}")
            cleaned_df, report = await self.orchestrator.clean_dataset(df, user_context)
            
            return cleaned_df, report
            
        except Exception as e:
            logger.error(f"Intelligent cleaning failed: {e}")
            
            if use_traditional_fallback:
                logger.info("Falling back to traditional cleaning")
                cleaned_df = self.preprocessor.fit_transform(df)
                report = {
                    "method": "traditional_fallback",
                    "error": str(e),
                    "quality_report": self.preprocessor.quality_report
                }
                return cleaned_df, report
            else:
                raise
    
    def assess_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data quality without cleaning
        
        Args:
            df: Input dataframe
            
        Returns:
            Quality assessment report
        """
        return self.preprocessor.check_data_quality(df)
    
    async def validate_against_standards(
        self, 
        df: pd.DataFrame, 
        sector: str
    ) -> Dict[str, Any]:
        """
        Validate data against sector standards
        
        Args:
            df: Input dataframe
            sector: Business sector
            
        Returns:
            Validation report
        """
        try:
            from automl_platform.agents import ValidatorAgent, AgentConfig
            
            agent_config = AgentConfig(
                openai_api_key=self.openai_api_key,
                user_context={"secteur_activite": sector}
            )
            
            validator = ValidatorAgent(agent_config)
            
            # Get basic profile first
            profile = self.preprocessor.check_data_quality(df)
            
            # Validate against standards
            validation_report = await validator.validate(df, profile)
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"error": str(e), "valid": False}
    
    @staticmethod
    def create_from_config(config_path: str) -> 'IntelligentDataCleaner':
        """
        Create IntelligentDataCleaner from YAML config file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            IntelligentDataCleaner instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return IntelligentDataCleaner(config)


# Convenience functions
def create_intelligent_cleaner(
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4-1106-preview"
) -> IntelligentDataCleaner:
    """
    Create an intelligent data cleaner instance
    
    Args:
        openai_api_key: OpenAI API key (uses env var if not provided)
        model: OpenAI model to use
        
    Returns:
        IntelligentDataCleaner instance
    """
    config = {
        'enable_intelligent_cleaning': True,
        'openai_cleaning_model': model,
        'openai_api_key': openai_api_key or os.getenv('OPENAI_API_KEY')
    }
    
    return IntelligentDataCleaner(config)


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
    
    # Create preprocessor
    config = {
        'handle_outliers': True,
        'outlier_method': 'iqr',
        'scaling_method': 'robust',
        'enable_quality_checks': True,
        'enable_drift_detection': True,
        'enable_intelligent_cleaning': True  # Enable OpenAI agents
    }
    
    preprocessor = EnhancedDataPreprocessor(config)
    
    # Check data quality
    quality_report = preprocessor.check_data_quality(df)
    print(f"Data quality score: {quality_report['quality_score']:.1f}")
    
    # Example of intelligent cleaning (requires async)
    async def test_intelligent_cleaning():
        user_context = {
            "secteur_activite": "finance",
            "target_variable": "target",
            "contexte_metier": "Risk prediction"
        }
        
        cleaned_df = await preprocessor.intelligent_clean(df, user_context)
        print(f"Intelligent cleaning completed: {cleaned_df.shape}")
    
    # Run if OpenAI API key is available
    if os.getenv('OPENAI_API_KEY'):
        import asyncio
        asyncio.run(test_intelligent_cleaning())
    else:
        # Standard cleaning
        X_transformed = preprocessor.fit_transform(df, y)
        print(f"Transformed shape: {X_transformed.shape}")
    
    # Detect drift (simulate with modified data)
    df_drift = df.copy()
    df_drift['num_0'] = df_drift['num_0'] * 2 + 10  # Introduce drift
    
    drift_report = preprocessor.detect_drift(df, df_drift)
    print(f"Drift detected: {drift_report['drift_detected']}")
    if drift_report['drifted_features']:
        print(f"Drifted features: {drift_report['drifted_features']}")
