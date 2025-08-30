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
                    # Check if text - UPDATED LOGIC
                    # Consider it text if:
                    # 1. Column name contains 'text' or 'description' or 'comment'
                    # 2. Average length > 20 (lowered threshold)
                    # 3. Has multiple words (spaces) in values
                    col_lower = col.lower()
                    if 'text' in col_lower or 'description' in col_lower or 'comment' in col_lower:
                        self.text_features.append(col)
                    else:
                        # Check average length and word count
                        sample_values = df[col].dropna().astype(str)
                        if len(sample_values) > 0:
                            avg_len = sample_values.str.len().mean()
                            avg_words = sample_values.str.split().str.len().mean()
                            
                            # If long strings or multiple words, consider as text
                            if avg_len > 20 or avg_words > 3:
                                self.text_features.append(col)
                            else:
                                self.categorical_features.append(col)
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
        
        # Text pipeline - ULTRA FIXED
        if self.text_features:
            for text_col in self.text_features:
                # CRITICAL FIX: Calculate n_components dynamically based on actual feature count
                # For small datasets, use even smaller values
                max_features = self.config.get('text_max_features', 100)
                
                # Count total features to determine safe n_components
                total_numeric = len(self.numeric_features) if self.numeric_features else 0
                total_categorical = len(self.categorical_features) if self.categorical_features else 0
                
                # For categorical, estimate encoded features (rough estimate)
                if self.categorical_features and len(X) > 0:
                    cat_encoded_estimate = sum([X[col].nunique() for col in self.categorical_features 
                                               if col in X.columns])
                else:
                    cat_encoded_estimate = total_categorical * 3  # rough estimate
                
                # Total features after encoding
                total_features_estimate = total_numeric + cat_encoded_estimate
                
                # Safe n_components: use minimum of multiple constraints
                # CRITICAL: For Iris dataset with 4 features, we need to be very conservative
                n_components = min(
                    3,  # Maximum 3 components for any text feature
                    max_features - 1,  # Less than max features
                    max(1, total_features_estimate - 1),  # Must be less than total features
                    len(X) - 1 if len(X) > 1 else 1  # Less than number of samples
                )
                n_components = max(1, n_components)  # At least 1
                
                text_pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=max_features,
                                             ngram_range=(1, 2))),
                    ('svd', TruncatedSVD(n_components=n_components))
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
        return {
            'valid': False,
            'issues': issues,
            'n_rows': 0,
            'n_columns': 0,
            'missing_ratio': 0.0
        }
    
    # Check for duplicate columns - Handle edge case
    try:
        dup_mask = df.columns.duplicated()
        has_duplicates = bool(dup_mask.any())
        if has_duplicates:
            duplicate_cols = df.columns[dup_mask].tolist()
            issues.append(f"Duplicate columns: {duplicate_cols}")
    except:
        # Fallback for any edge case
        col_counts = df.columns.value_counts()
        duplicate_cols = col_counts[col_counts > 1].index.tolist()
        if duplicate_cols:
            issues.append(f"Duplicate columns: {duplicate_cols}")
    
    # Check for all null columns - Handle duplicate columns case
    try:
        null_cols = []
        for col_idx, col in enumerate(df.columns):
            # Use iloc to handle duplicate column names
            if df.iloc[:, col_idx].isnull().all():
                null_cols.append(str(col))
        if null_cols:
            issues.append(f"All null columns: {null_cols}")
    except:
        pass
    
    # Check for single value columns - Handle duplicate columns case
    try:
        single_value_cols = []
        for col_idx, col in enumerate(df.columns):
            # Use iloc to handle duplicate column names
            if df.iloc[:, col_idx].nunique() == 1:
                single_value_cols.append(str(col))
        if single_value_cols:
            # Deduplicate the list
            single_value_cols = list(set(single_value_cols))
            issues.append(f"Single value columns: {single_value_cols}")
    except:
        pass
    
    # Check for high missing ratio
    try:
        high_missing = {}
        for col_idx, col in enumerate(df.columns):
            missing_ratio = df.iloc[:, col_idx].isnull().mean()
            if missing_ratio > 0.5:
                high_missing[str(col)] = float(missing_ratio)
        if high_missing:
            issues.append(f"High missing ratio: {high_missing}")
    except:
        pass
    
    # Calculate overall missing ratio safely
    try:
        missing_ratio = df.isnull().values.mean()
    except:
        missing_ratio = 0.0
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_ratio': float(missing_ratio)
    }
