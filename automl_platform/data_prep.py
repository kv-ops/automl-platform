"""Data preparation with sklearn pipelines - no data leakage."""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Union, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Complete data preprocessing pipeline with no leakage."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.text_features = []
        self.pipeline = None
        self.feature_types = {}
        
    def detect_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Automatically detect feature types."""
        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.text_features = []
        
        for col in df.columns:
            dtype = df[col].dtype
            nunique = df[col].nunique()
            
            # Detect datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.datetime_features.append(col)
            # Try to parse as datetime
            elif df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col], errors='raise')
                    self.datetime_features.append(col)
                except:
                    # Check if text (long strings)
                    avg_len = df[col].dropna().astype(str).str.len().mean()
                    if avg_len > 50:
                        self.text_features.append(col)
                    else:
                        self.categorical_features.append(col)
            # Numeric
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Check if actually categorical (low cardinality integers)
                if nunique < 10 and df[col].dtype in ['int64', 'int32']:
                    self.categorical_features.append(col)
                else:
                    self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
        
        self.feature_types = {
            'numeric': self.numeric_features,
            'categorical': self.categorical_features,
            'datetime': self.datetime_features,
            'text': self.text_features
        }
        
        logger.info(f"Detected feature types: {self.feature_types}")
        return self.feature_types
    
    def create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create datetime features (done per-fold to avoid leakage)."""
        df_new = df.copy()
        
        for col in self.datetime_features:
            # Convert to datetime
            dt_col = pd.to_datetime(df_new[col], errors='coerce')
            
            # Extract components
            df_new[f'{col}_year'] = dt_col.dt.year
            df_new[f'{col}_month'] = dt_col.dt.month
            df_new[f'{col}_day'] = dt_col.dt.day
            df_new[f'{col}_dayofweek'] = dt_col.dt.dayofweek
            df_new[f'{col}_quarter'] = dt_col.dt.quarter
            df_new[f'{col}_is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
            
            # Add to numeric features
            self.numeric_features.extend([
                f'{col}_year', f'{col}_month', f'{col}_day',
                f'{col}_dayofweek', f'{col}_quarter', f'{col}_is_weekend'
            ])
            
            # Drop original datetime column
            df_new = df_new.drop(columns=[col])
        
        return df_new
    
    def detect_outliers(self, X: np.ndarray, method: str = 'iqr') -> np.ndarray:
        """Detect outliers in numeric data."""
        if method == 'iqr':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return (X < lower) | (X > upper)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(X, axis=0))
            return z_scores > 3
        else:
            return np.zeros_like(X, dtype=bool)
    
    def create_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Create preprocessing pipeline (fit per fold)."""
        
        # Detect feature types if not done
        if not self.feature_types:
            self.detect_feature_types(X)
        
        # Handle datetime features first
        if self.datetime_features:
            X = self.create_datetime_features(X)
        
        transformers = []
        
        # Numeric pipeline
        if self.numeric_features:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', self._get_scaler())
            ])
            transformers.append(('numeric', numeric_pipeline, self.numeric_features))
        
        # Categorical pipeline
        if self.categorical_features:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('categorical', categorical_pipeline, self.categorical_features))
        
        # Text pipeline
        if self.text_features:
            for text_col in self.text_features:
                text_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=self.config.get('text_max_features', 100),
                                             ngram_range=(1, 2))),
                    ('svd', TruncatedSVD(n_components=min(10, self.config.get('text_max_features', 100) - 1)))
                ])
                transformers.append((f'text_{text_col}', text_pipeline, text_col))
        
        # Create and store pipeline
        self.pipeline = ColumnTransformer(transformers, remainder='passthrough')
        return self.pipeline
    
    def _get_scaler(self):
        """Get scaler based on config."""
        method = self.config.get('scaling_method', 'robust')
        if method == 'standard':
            return StandardScaler()
        elif method == 'robust':
            return RobustScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        else:
            return StandardScaler()
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform data."""
        if self.pipeline is None:
            self.create_pipeline(X)
        
        # Handle datetime features
        if self.datetime_features:
            X = self.create_datetime_features(X)
        
        return self.pipeline.fit_transform(X, y)
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted yet")
        
        # Handle datetime features
        if self.datetime_features:
            X = self.create_datetime_features(X)
        
        return self.pipeline.transform(X)


def handle_imbalance(X: np.ndarray, y: np.ndarray, method: str = 'class_weight') -> Tuple[np.ndarray, np.ndarray]:
    """Handle class imbalance (done per fold)."""
    if method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            return smote.fit_resample(X, y)
        except ImportError:
            logger.warning("SMOTE not available, returning original data")
            return X, y
    elif method == 'adasyn':
        try:
            from imblearn.over_sampling import ADASYN
            adasyn = ADASYN(random_state=42)
            return adasyn.fit_resample(X, y)
        except ImportError:
            logger.warning("ADASYN not available, returning original data")
            return X, y
    else:
        return X, y


def create_lag_features(df: pd.DataFrame, target_col: str, lag_periods: List[int]) -> pd.DataFrame:
    """Create lag features for time series (done per fold)."""
    df_lagged = df.copy()
    
    for lag in lag_periods:
        df_lagged[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [7, 30]:
        df_lagged[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
        df_lagged[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
    
    return df_lagged


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality."""
    issues = []
    
    # Check for empty dataframe
    if df.empty:
        issues.append("DataFrame is empty")
    
    # Check for duplicate columns - FIX HERE
    duplicate_mask = df.columns.duplicated()
    if duplicate_mask.any():  # Use .any() to get a single boolean value
        duplicate_cols = df.columns[duplicate_mask].tolist()
        issues.append(f"Duplicate columns: {duplicate_cols}")
    
    # Check for all null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        issues.append(f"All null columns: {null_cols}")
    
    # Check for single value columns
    single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
    if single_value_cols:
        issues.append(f"Single value columns: {single_value_cols}")
    
    # Check for high missing ratio
    high_missing = {}
    for col in df.columns:
        missing_ratio = df[col].isnull().mean()
        if missing_ratio > 0.5:
            high_missing[col] = missing_ratio
    if high_missing:
        issues.append(f"High missing ratio: {high_missing}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_ratio': df.isnull().mean().mean()
    }
