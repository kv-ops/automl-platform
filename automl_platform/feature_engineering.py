"""
Advanced Feature Engineering Module
Automatic feature generation, selection, and transformation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
import logging
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Automatic feature engineering with multiple strategies.
    Generates polynomial, interaction, ratio, and aggregate features.
    """
    
    def __init__(self,
                 max_features: int = 100,
                 feature_types: str = 'all',
                 polynomial_degree: int = 2,
                 interaction_only: bool = False,
                 include_ratios: bool = True,
                 include_logs: bool = True,
                 include_aggregates: bool = True,
                 binning: bool = True,
                 n_bins: int = 10,
                 encoding_method: str = 'target',
                 selection_method: str = 'mutual_info',
                 selection_threshold: float = 0.01,
                 task: str = 'classification',
                 random_state: int = 42):
        """
        Initialize feature engineer.
        
        Args:
            max_features: Maximum number of features to generate
            feature_types: Types of features to generate ('all', 'polynomial', 'interaction', etc.)
            polynomial_degree: Degree for polynomial features
            interaction_only: Generate only interaction features
            include_ratios: Generate ratio features
            include_logs: Generate log transformations
            include_aggregates: Generate aggregate features
            binning: Create binned features
            n_bins: Number of bins
            encoding_method: Method for encoding categorical features
            selection_method: Method for feature selection
            selection_threshold: Threshold for feature selection
            task: 'classification' or 'regression'
            random_state: Random seed
        """
        self.max_features = max_features
        self.feature_types = feature_types
        self.polynomial_degree = polynomial_degree
        self.interaction_only = interaction_only
        self.include_ratios = include_ratios
        self.include_logs = include_logs
        self.include_aggregates = include_aggregates
        self.binning = binning
        self.n_bins = n_bins
        self.encoding_method = encoding_method
        self.selection_method = selection_method
        self.selection_threshold = selection_threshold
        self.task = task
        self.random_state = random_state
        
        self.generated_features_ = []
        self.selected_features_ = []
        self.feature_names_ = []
        
    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> 'AutoFeatureEngineer':
        """Fit feature engineer."""
        
        logger.info(f"Starting automatic feature engineering on {X.shape} data")
        
        # Identify column types
        self._identify_column_types(X)
        
        # Generate features
        X_engineered = self._generate_features(X, y)
        
        # Select best features
        if y is not None and X_engineered.shape[1] > self.max_features:
            X_selected = self._select_features(X_engineered, y)
            self.selected_features_ = X_selected.columns.tolist()
        else:
            self.selected_features_ = X_engineered.columns.tolist()
        
        logger.info(f"Generated {len(self.generated_features_)} features, selected {len(self.selected_features_)}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with engineered features."""
        
        # Generate same features
        X_engineered = self._generate_features(X, fit=False)
        
        # Select same features
        if self.selected_features_:
            # Only keep features that exist
            available_features = [f for f in self.selected_features_ if f in X_engineered.columns]
            X_engineered = X_engineered[available_features]
        
        return X_engineered
    
    def _identify_column_types(self, X: pd.DataFrame):
        """Identify numeric and categorical columns."""
        
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Also identify binary and ordinal features
        self.binary_cols_ = []
        self.ordinal_cols_ = []
        
        for col in self.numeric_cols_:
            if X[col].nunique() == 2:
                self.binary_cols_.append(col)
            elif X[col].nunique() < 10:
                self.ordinal_cols_.append(col)
    
    def _generate_features(self, X: pd.DataFrame, y: np.ndarray = None, fit: bool = True) -> pd.DataFrame:
        """Generate all types of features."""
        
        features = [X.copy()]
        
        # Polynomial features
        if self.feature_types in ['all', 'polynomial'] and self.numeric_cols_:
            poly_features = self._generate_polynomial_features(X)
            if poly_features is not None:
                features.append(poly_features)
        
        # Interaction features
        if self.feature_types in ['all', 'interaction'] and len(self.numeric_cols_) > 1:
            interaction_features = self._generate_interaction_features(X)
            if interaction_features is not None:
                features.append(interaction_features)
        
        # Ratio features
        if self.include_ratios and len(self.numeric_cols_) > 1:
            ratio_features = self._generate_ratio_features(X)
            if ratio_features is not None:
                features.append(ratio_features)
        
        # Log features
        if self.include_logs and self.numeric_cols_:
            log_features = self._generate_log_features(X)
            if log_features is not None:
                features.append(log_features)
        
        # Aggregate features
        if self.include_aggregates and len(self.numeric_cols_) > 2:
            agg_features = self._generate_aggregate_features(X)
            if agg_features is not None:
                features.append(agg_features)
        
        # Binning features
        if self.binning and self.numeric_cols_:
            bin_features = self._generate_binned_features(X, fit=fit)
            if bin_features is not None:
                features.append(bin_features)
        
        # Target encoding for categorical
        if y is not None and self.categorical_cols_ and self.encoding_method == 'target':
            encoded_features = self._generate_target_encoded_features(X, y, fit=fit)
            if encoded_features is not None:
                features.append(encoded_features)
        
        # Combine all features
        X_combined = pd.concat(features, axis=1)
        
        # Remove duplicate columns
        X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
        
        if fit:
            self.generated_features_ = X_combined.columns.tolist()
        
        return X_combined
    
    def _generate_polynomial_features(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate polynomial features."""
        
        try:
            X_numeric = X[self.numeric_cols_]
            
            # Limit features to avoid explosion
            if len(self.numeric_cols_) > 10:
                X_numeric = X_numeric[self.numeric_cols_[:10]]
            
            poly = PolynomialFeatures(
                degree=self.polynomial_degree,
                interaction_only=self.interaction_only,
                include_bias=False
            )
            
            X_poly = poly.fit_transform(X_numeric)
            
            # Get feature names
            feature_names = poly.get_feature_names_out(X_numeric.columns)
            
            # Remove original features (already in main dataframe)
            mask = [name not in X_numeric.columns for name in feature_names]
            X_poly = X_poly[:, mask]
            feature_names = [name for name, keep in zip(feature_names, mask) if keep]
            
            # Create dataframe
            poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
            
            # Limit number of polynomial features
            if poly_df.shape[1] > self.max_features // 4:
                poly_df = poly_df.iloc[:, :self.max_features // 4]
            
            return poly_df
            
        except Exception as e:
            logger.warning(f"Failed to generate polynomial features: {e}")
            return None
    
    def _generate_interaction_features(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate interaction features between numeric columns."""
        
        try:
            interaction_features = {}
            
            # Limit to avoid explosion
            cols_to_use = self.numeric_cols_[:min(10, len(self.numeric_cols_))]
            
            for col1, col2 in combinations(cols_to_use, 2):
                # Multiplication
                interaction_features[f'{col1}_times_{col2}'] = X[col1] * X[col2]
                
                # Addition
                interaction_features[f'{col1}_plus_{col2}'] = X[col1] + X[col2]
                
                # Difference
                interaction_features[f'{col1}_minus_{col2}'] = X[col1] - X[col2]
                
                # Limit features
                if len(interaction_features) >= self.max_features // 4:
                    break
            
            return pd.DataFrame(interaction_features, index=X.index)
            
        except Exception as e:
            logger.warning(f"Failed to generate interaction features: {e}")
            return None
    
    def _generate_ratio_features(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate ratio features."""
        
        try:
            ratio_features = {}
            
            # Select non-zero columns
            safe_cols = []
            for col in self.numeric_cols_:
                if (X[col] != 0).any():
                    safe_cols.append(col)
            
            # Limit columns
            safe_cols = safe_cols[:min(10, len(safe_cols))]
            
            for col1, col2 in combinations(safe_cols, 2):
                # Avoid division by zero
                denominator = X[col2].replace(0, np.nan)
                
                ratio_features[f'{col1}_div_{col2}'] = X[col1] / denominator
                
                # Inverse ratio
                denominator_inv = X[col1].replace(0, np.nan)
                ratio_features[f'{col2}_div_{col1}'] = X[col2] / denominator_inv
                
                # Limit features
                if len(ratio_features) >= self.max_features // 6:
                    break
            
            ratio_df = pd.DataFrame(ratio_features, index=X.index)
            
            # Fill NaN with 0
            ratio_df = ratio_df.fillna(0)
            
            # Replace inf with large value
            ratio_df = ratio_df.replace([np.inf, -np.inf], [1e10, -1e10])
            
            return ratio_df
            
        except Exception as e:
            logger.warning(f"Failed to generate ratio features: {e}")
            return None
    
    def _generate_log_features(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate log-transformed features."""
        
        try:
            log_features = {}
            
            for col in self.numeric_cols_:
                # Only for positive values
                if (X[col] > 0).all():
                    log_features[f'{col}_log'] = np.log1p(X[col])
                elif (X[col] >= 0).all():
                    # Add small constant for zero values
                    log_features[f'{col}_log'] = np.log1p(X[col] + 1e-10)
                
                # Square root for non-negative
                if (X[col] >= 0).all():
                    log_features[f'{col}_sqrt'] = np.sqrt(X[col])
                
                # Square
                log_features[f'{col}_squared'] = X[col] ** 2
                
                # Limit features
                if len(log_features) >= self.max_features // 4:
                    break
            
            return pd.DataFrame(log_features, index=X.index)
            
        except Exception as e:
            logger.warning(f"Failed to generate log features: {e}")
            return None
    
    def _generate_aggregate_features(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate aggregate features."""
        
        try:
            agg_features = {}
            
            X_numeric = X[self.numeric_cols_]
            
            # Row-wise statistics
            agg_features['row_mean'] = X_numeric.mean(axis=1)
            agg_features['row_std'] = X_numeric.std(axis=1)
            agg_features['row_max'] = X_numeric.max(axis=1)
            agg_features['row_min'] = X_numeric.min(axis=1)
            agg_features['row_median'] = X_numeric.median(axis=1)
            agg_features['row_sum'] = X_numeric.sum(axis=1)
            agg_features['row_skew'] = X_numeric.skew(axis=1)
            agg_features['row_kurt'] = X_numeric.kurtosis(axis=1)
            
            # Count features
            agg_features['row_zeros'] = (X_numeric == 0).sum(axis=1)
            agg_features['row_positives'] = (X_numeric > 0).sum(axis=1)
            agg_features['row_negatives'] = (X_numeric < 0).sum(axis=1)
            
            return pd.DataFrame(agg_features, index=X.index)
            
        except Exception as e:
            logger.warning(f"Failed to generate aggregate features: {e}")
            return None
    
    def _generate_binned_features(self, X: pd.DataFrame, fit: bool = True) -> Optional[pd.DataFrame]:
        """Generate binned features."""
        
        try:
            binned_features = {}
            
            for col in self.numeric_cols_[:min(10, len(self.numeric_cols_))]:
                if fit:
                    # Fit discretizer
                    discretizer = KBinsDiscretizer(
                        n_bins=min(self.n_bins, X[col].nunique()),
                        encode='ordinal',
                        strategy='quantile'
                    )
                    
                    binned = discretizer.fit_transform(X[[col]])
                    
                    # Store for transform
                    if not hasattr(self, 'discretizers_'):
                        self.discretizers_ = {}
                    self.discretizers_[col] = discretizer
                else:
                    # Use fitted discretizer
                    if hasattr(self, 'discretizers_') and col in self.discretizers_:
                        binned = self.discretizers_[col].transform(X[[col]])
                    else:
                        continue
                
                binned_features[f'{col}_binned'] = binned.ravel()
            
            return pd.DataFrame(binned_features, index=X.index)
            
        except Exception as e:
            logger.warning(f"Failed to generate binned features: {e}")
            return None
    
    def _generate_target_encoded_features(self, X: pd.DataFrame, y: np.ndarray, fit: bool = True) -> Optional[pd.DataFrame]:
        """Generate target-encoded features for categorical columns."""
        
        try:
            encoded_features = {}
            
            for col in self.categorical_cols_:
                if fit:
                    # Calculate target encoding
                    if self.task == 'classification':
                        # Use class probabilities
                        encoding_dict = {}
                        for value in X[col].unique():
                            mask = X[col] == value
                            if mask.sum() > 0:
                                encoding_dict[value] = y[mask].mean()
                            else:
                                encoding_dict[value] = y.mean()
                    else:
                        # Use mean target value
                        encoding_dict = X.groupby(col)[col].size().to_dict()
                        mean_y = y.mean()
                        for value in encoding_dict:
                            mask = X[col] == value
                            if mask.sum() > 0:
                                encoding_dict[value] = y[mask].mean()
                            else:
                                encoding_dict[value] = mean_y
                    
                    # Store for transform
                    if not hasattr(self, 'target_encodings_'):
                        self.target_encodings_ = {}
                    self.target_encodings_[col] = encoding_dict
                else:
                    # Use fitted encoding
                    if hasattr(self, 'target_encodings_') and col in self.target_encodings_:
                        encoding_dict = self.target_encodings_[col]
                    else:
                        continue
                
                # Apply encoding
                default_value = np.mean(list(encoding_dict.values()))
                encoded_features[f'{col}_target_encoded'] = X[col].map(encoding_dict).fillna(default_value)
            
            return pd.DataFrame(encoded_features, index=X.index)
            
        except Exception as e:
            logger.warning(f"Failed to generate target-encoded features: {e}")
            return None
    
    def _select_features(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Select best features."""
        
        logger.info(f"Selecting best features from {X.shape[1]} candidates")
        
        try:
            if self.selection_method == 'mutual_info':
                return self._select_mutual_info(X, y)
            elif self.selection_method == 'rfe':
                return self._select_rfe(X, y)
            elif self.selection_method == 'model':
                return self._select_from_model(X, y)
            elif self.selection_method == 'variance':
                return self._select_variance(X)
            else:
                # Default: select first max_features
                return X.iloc[:, :self.max_features]
                
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}, using first {self.max_features} features")
            return X.iloc[:, :self.max_features]
    
    def _select_mutual_info(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Select features using mutual information."""
        
        # Handle NaN and inf
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if self.task == 'classification':
            selector = SelectKBest(mutual_info_classif, k=min(self.max_features, X.shape[1]))
        else:
            selector = SelectKBest(mutual_info_regression, k=min(self.max_features, X.shape[1]))
        
        X_selected = selector.fit_transform(X_clean, y)
        
        # Get selected column names
        selected_mask = selector.get_support()
        selected_columns = X.columns[selected_mask].tolist()
        
        return X[selected_columns]
    
    def _select_rfe(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Select features using RFE."""
        
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Handle NaN and inf
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if self.task == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=self.random_state)
        else:
            estimator = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=self.random_state)
        
        selector = RFE(estimator, n_features_to_select=min(self.max_features, X.shape[1]))
        selector.fit(X_clean, y)
        
        # Get selected column names
        selected_mask = selector.support_
        selected_columns = X.columns[selected_mask].tolist()
        
        return X[selected_columns]
    
    def _select_from_model(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Select features using model importance."""
        
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Handle NaN and inf
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if self.task == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=self.random_state)
        else:
            estimator = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=self.random_state)
        
        selector = SelectFromModel(estimator, max_features=self.max_features, threshold=-np.inf)
        selector.fit(X_clean, y)
        
        # Get selected column names
        selected_mask = selector.get_support()
        selected_columns = X.columns[selected_mask].tolist()
        
        return X[selected_columns]
    
    def _select_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features using variance threshold."""
        
        # Handle NaN and inf
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Remove constant features
        selector = VarianceThreshold(threshold=self.selection_threshold)
        selector.fit(X_clean)
        
        # Get selected column names
        selected_mask = selector.get_support()
        selected_columns = X.columns[selected_mask].tolist()
        
        # Limit to max_features
        if len(selected_columns) > self.max_features:
            # Calculate variance for each feature
            variances = X[selected_columns].var()
            top_features = variances.nlargest(self.max_features).index.tolist()
            return X[top_features]
        
        return X[selected_columns]


# Additional feature engineering functions
def create_time_series_features(df: pd.DataFrame,
                               date_column: str,
                               target_column: str = None) -> pd.DataFrame:
    """
    Create time series features from datetime column.
    
    Args:
        df: Input dataframe
        date_column: Name of datetime column
        target_column: Target column for lag features
        
    Returns:
        DataFrame with time series features
    """
    df = df.copy()
    
    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract datetime components
    df[f'{date_column}_year'] = df[date_column].dt.year
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_day'] = df[date_column].dt.day
    df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
    df[f'{date_column}_quarter'] = df[date_column].dt.quarter
    df[f'{date_column}_dayofyear'] = df[date_column].dt.dayofyear
    df[f'{date_column}_weekofyear'] = df[date_column].dt.isocalendar().week
    df[f'{date_column}_is_weekend'] = (df[date_column].dt.dayofweek >= 5).astype(int)
    df[f'{date_column}_is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df[f'{date_column}_is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    
    # Cyclical encoding
    df[f'{date_column}_month_sin'] = np.sin(2 * np.pi * df[f'{date_column}_month'] / 12)
    df[f'{date_column}_month_cos'] = np.cos(2 * np.pi * df[f'{date_column}_month'] / 12)
    df[f'{date_column}_day_sin'] = np.sin(2 * np.pi * df[f'{date_column}_day'] / 31)
    df[f'{date_column}_day_cos'] = np.cos(2 * np.pi * df[f'{date_column}_day'] / 31)
    
    # Lag features if target is provided
    if target_column and target_column in df.columns:
        for lag in [1, 7, 14, 30]:
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window).mean()
            df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window).std()
            df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window).min()
            df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window).max()
    
    return df


def create_text_features(df: pd.DataFrame,
                        text_columns: List[str]) -> pd.DataFrame:
    """
    Create features from text columns.
    
    Args:
        df: Input dataframe
        text_columns: List of text column names
        
    Returns:
        DataFrame with text features
    """
    df = df.copy()
    
    for col in text_columns:
        if col not in df.columns:
            continue
        
        # Basic text statistics
        df[f'{col}_length'] = df[col].str.len()
        df[f'{col}_word_count'] = df[col].str.split().str.len()
        df[f'{col}_unique_word_count'] = df[col].apply(lambda x: len(set(str(x).split())))
        
        # Character statistics
        df[f'{col}_digit_count'] = df[col].str.count(r'\d')
        df[f'{col}_upper_count'] = df[col].str.count(r'[A-Z]')
        df[f'{col}_lower_count'] = df[col].str.count(r'[a-z]')
        df[f'{col}_space_count'] = df[col].str.count(r'\s')
        df[f'{col}_punct_count'] = df[col].str.count(r'[^\w\s]')
        
        # Ratios
        df[f'{col}_digit_ratio'] = df[f'{col}_digit_count'] / (df[f'{col}_length'] + 1)
        df[f'{col}_upper_ratio'] = df[f'{col}_upper_count'] / (df[f'{col}_length'] + 1)
        df[f'{col}_space_ratio'] = df[f'{col}_space_count'] / (df[f'{col}_length'] + 1)
    
    return df


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_classes=2, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    
    # Create feature engineer
    engineer = AutoFeatureEngineer(
        max_features=50,
        feature_types='all',
        polynomial_degree=2,
        include_ratios=True,
        include_logs=True,
        task='classification'
    )
    
    # Fit and transform
    engineer.fit(X_df, y)
    X_engineered = engineer.transform(X_df)
    
    print(f"Original features: {X_df.shape}")
    print(f"Engineered features: {X_engineered.shape}")
    print(f"Selected features: {engineer.selected_features_[:10]}")
