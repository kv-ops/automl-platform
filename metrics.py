"""Metrics calculation and task detection for AutoML platform."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    median_absolute_error, max_error
)
import logging

logger = logging.getLogger(__name__)


def detect_task(y: Union[pd.Series, np.ndarray]) -> str:
    """
    Automatically detect the machine learning task type.
    
    Args:
        y: Target variable (Series or array)
        
    Returns:
        str: Task type ('classification' or 'regression')
    """
    # Convert to numpy if pandas
    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y
    
    # Check for non-numeric types
    try:
        y_numeric = pd.to_numeric(pd.Series(y_values), errors='coerce')
        if y_numeric.isna().all():
            # All non-numeric - classification
            return 'classification'
    except:
        return 'classification'
    
    # Check unique values
    unique_values = np.unique(y_values[~pd.isna(y_values)])
    n_unique = len(unique_values)
    n_samples = len(y_values)
    
    # Heuristics for classification vs regression
    if n_unique == 2:
        return 'classification'
    elif n_unique < 20 and n_unique < n_samples * 0.05:
        # Few unique values relative to samples
        return 'classification'
    elif str(y_values.dtype).startswith('object') or str(y_values.dtype).startswith('str'):
        return 'classification'
    else:
        # Check if values are all integers (potential classes)
        if np.all(y_values == y_values.astype(int)):
            if n_unique < 20:
                return 'classification'
            elif n_unique < np.sqrt(n_samples):
                # Moderate number of unique integers
                return 'classification'
        
        # Default to regression for continuous values
        return 'regression'


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray,
                     y_proba: Optional[np.ndarray] = None,
                     task: str = 'classification') -> Dict[str, float]:
    """
    Calculate comprehensive metrics for the given task.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for classification)
        task: Task type ('classification' or 'regression')
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    if task == 'classification':
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Determine if binary or multiclass
        n_classes = len(np.unique(y_true))
        is_binary = n_classes == 2
        
        # Set averaging method
        average = 'binary' if is_binary else 'weighted'
        
        # Calculate precision, recall, F1
        try:
            metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        except Exception as e:
            logger.warning(f"Error calculating classification metrics: {e}")
        
        # Add macro averages for multiclass
        if not is_binary:
            try:
                metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
            except:
                pass
        
        # Probability-based metrics
        if y_proba is not None:
            try:
                if is_binary:
                    # For binary classification
                    if len(y_proba.shape) > 1 and y_proba.shape[1] > 1:
                        # If probabilities for both classes
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                        metrics['log_loss'] = log_loss(y_true, y_proba)
                    else:
                        # If only positive class probabilities
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                        # Create proper probability matrix for log_loss
                        proba_matrix = np.column_stack([1 - y_proba, y_proba])
                        metrics['log_loss'] = log_loss(y_true, proba_matrix)
                else:
                    # Multiclass
                    try:
                        metrics['log_loss'] = log_loss(y_true, y_proba)
                        # Try multiclass ROC AUC
                        if n_classes <= 10:  # Limit for computational efficiency
                            from sklearn.preprocessing import label_binarize
                            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
                            metrics['roc_auc_ovr'] = roc_auc_score(y_true_bin, y_proba, multi_class='ovr')
                    except Exception as e:
                        logger.debug(f"Could not calculate multiclass probability metrics: {e}")
            except Exception as e:
                logger.debug(f"Could not calculate probability metrics: {e}")
        
        # Confusion matrix derived metrics
        try:
            cm = confusion_matrix(y_true, y_pred)
            if is_binary:
                tn, fp, fn, tp = cm.ravel()
                metrics['true_positives'] = int(tp)
                metrics['true_negatives'] = int(tn)
                metrics['false_positives'] = int(fp)
                metrics['false_negatives'] = int(fn)
                
                # Additional binary metrics
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
                metrics['balanced_accuracy'] = (metrics['specificity'] + metrics['sensitivity']) / 2
        except:
            pass
    
    else:  # Regression
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Additional regression metrics
        try:
            metrics['median_ae'] = median_absolute_error(y_true, y_pred)
            metrics['max_error'] = max_error(y_true, y_pred)
            metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
            
            # MAPE only if no zeros in y_true
            if not np.any(y_true == 0):
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
            
            # Custom metrics
            residuals = y_true - y_pred
            metrics['mean_residual'] = np.mean(residuals)
            metrics['std_residual'] = np.std(residuals)
            
            # Percentage of predictions within certain error bounds
            for threshold in [0.05, 0.10, 0.20]:
                within = np.mean(np.abs(residuals) <= np.abs(y_true) * threshold) if np.any(y_true != 0) else 0
                metrics[f'within_{int(threshold*100)}pct'] = within
                
        except Exception as e:
            logger.debug(f"Could not calculate additional regression metrics: {e}")
    
    # Round metrics for better display
    metrics = {k: round(float(v), 6) if isinstance(v, (int, float, np.number)) else v 
               for k, v in metrics.items()}
    
    return metrics


def get_scorer(task: str, scoring: Optional[str] = None):
    """
    Get the appropriate scorer for the task.
    
    Args:
        task: Task type ('classification' or 'regression')
        scoring: Specific scoring metric name (optional)
        
    Returns:
        Scorer name for sklearn
    """
    if scoring and scoring != 'auto':
        return scoring
    
    if task == 'classification':
        # Default to ROC AUC for binary, F1 for multiclass
        return 'roc_auc'  # Will be adjusted based on actual data
    else:
        # Default to negative MSE for regression
        return 'neg_mean_squared_error'


def calculate_feature_importance_scores(feature_names: list,
                                       importance_values: np.ndarray,
                                       top_k: int = 20) -> Dict[str, float]:
    """
    Calculate and format feature importance scores.
    
    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        top_k: Number of top features to return
        
    Returns:
        Dictionary of feature names and importance scores
    """
    # Ensure arrays are same length
    if len(feature_names) != len(importance_values):
        logger.warning(f"Feature names ({len(feature_names)}) and importance values "
                      f"({len(importance_values)}) have different lengths")
        min_len = min(len(feature_names), len(importance_values))
        feature_names = feature_names[:min_len]
        importance_values = importance_values[:min_len]
    
    # Sort by importance
    indices = np.argsort(importance_values)[::-1]
    
    # Get top k features
    top_indices = indices[:min(top_k, len(indices))]
    
    # Create dictionary
    importance_dict = {
        feature_names[i]: float(importance_values[i])
        for i in top_indices
    }
    
    return importance_dict
