"""Enhanced metrics calculation with A/B testing comparison capabilities."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    log_loss, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    median_absolute_error, max_error
)
from scipy import stats
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional dependency
    sns = None
from io import BytesIO
import base64
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


def compare_models_metrics(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_proba_a: Optional[np.ndarray] = None,
    y_proba_b: Optional[np.ndarray] = None,
    task: str = 'classification',
    model_names: Tuple[str, str] = ('Model A', 'Model B')
) -> Dict[str, Any]:
    """
    Compare metrics between two models with statistical testing.
    
    Args:
        y_true: True labels
        y_pred_a: Predictions from model A
        y_pred_b: Predictions from model B
        y_proba_a: Probabilities from model A (optional)
        y_proba_b: Probabilities from model B (optional)
        task: Task type
        model_names: Names of the models
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        'model_a': {'name': model_names[0]},
        'model_b': {'name': model_names[1]},
        'comparison': {},
        'statistical_tests': {},
        'visualizations': {}
    }
    
    # Calculate metrics for both models
    comparison['model_a']['metrics'] = calculate_metrics(y_true, y_pred_a, y_proba_a, task)
    comparison['model_b']['metrics'] = calculate_metrics(y_true, y_pred_b, y_proba_b, task)
    
    # Calculate differences
    for metric in comparison['model_a']['metrics']:
        if metric in comparison['model_b']['metrics']:
            val_a = comparison['model_a']['metrics'][metric]
            val_b = comparison['model_b']['metrics'][metric]
            
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                diff = val_b - val_a
                comparison['comparison'][f'{metric}_diff'] = diff
                
                # Percentage improvement
                if val_a != 0:
                    pct_improvement = (diff / abs(val_a)) * 100
                    comparison['comparison'][f'{metric}_improvement_pct'] = pct_improvement
    
    # Statistical significance testing
    if task == 'classification':
        # McNemar's test for paired binary outcomes
        comparison['statistical_tests'] = perform_mcnemar_test(y_true, y_pred_a, y_pred_b)
        
        # Generate ROC curves if probabilities available
        if y_proba_a is not None and y_proba_b is not None:
            comparison['visualizations']['roc_curves'] = plot_roc_curves_comparison(
                y_true, y_proba_a, y_proba_b, model_names
            )
            comparison['visualizations']['pr_curves'] = plot_pr_curves_comparison(
                y_true, y_proba_a, y_proba_b, model_names
            )
    else:
        # Paired t-test for regression residuals
        residuals_a = y_true - y_pred_a
        residuals_b = y_true - y_pred_b
        comparison['statistical_tests'] = perform_paired_t_test(residuals_a, residuals_b)
        
        # Residual plots
        comparison['visualizations']['residual_plots'] = plot_residuals_comparison(
            y_true, y_pred_a, y_pred_b, model_names
        )
    
    # Confusion matrices for classification
    if task == 'classification':
        cm_a = confusion_matrix(y_true, y_pred_a)
        cm_b = confusion_matrix(y_true, y_pred_b)
        comparison['visualizations']['confusion_matrices'] = plot_confusion_matrices_comparison(
            cm_a, cm_b, model_names
        )
    
    return comparison


def perform_mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Dict[str, float]:
    """
    Perform McNemar's test for paired binary classification outcomes.
    
    Returns:
        Dictionary with test statistics
    """
    # Create contingency table
    correct_a = (y_true == y_pred_a)
    correct_b = (y_true == y_pred_b)
    
    # Count discordant pairs
    n_01 = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    n_10 = np.sum(~correct_a & correct_b)  # A wrong, B correct
    
    # McNemar's test statistic
    if n_01 + n_10 > 0:
        statistic = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
    else:
        statistic = 0
        p_value = 1.0
    
    return {
        'test': 'McNemar',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'n_01': int(n_01),
        'n_10': int(n_10),
        'significant': p_value < 0.05
    }


def perform_paired_t_test(residuals_a: np.ndarray, residuals_b: np.ndarray) -> Dict[str, float]:
    """
    Perform paired t-test on regression residuals.
    
    Returns:
        Dictionary with test statistics
    """
    differences = residuals_a - residuals_b
    
    # Paired t-test
    statistic, p_value = stats.ttest_rel(residuals_a, residuals_b)
    
    # Effect size (Cohen's d for paired samples)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    effect_size = mean_diff / std_diff if std_diff > 0 else 0
    
    # 95% confidence interval
    se = std_diff / np.sqrt(len(differences))
    ci_lower = mean_diff - 1.96 * se
    ci_upper = mean_diff + 1.96 * se
    
    return {
        'test': 'Paired t-test',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'effect_size': float(effect_size),
        'mean_difference': float(mean_diff),
        'confidence_interval': (float(ci_lower), float(ci_upper)),
        'significant': p_value < 0.05
    }


def plot_roc_curves_comparison(
    y_true: np.ndarray,
    y_proba_a: np.ndarray,
    y_proba_b: np.ndarray,
    model_names: Tuple[str, str]
) -> str:
    """
    Plot ROC curves comparison.
    
    Returns:
        Base64 encoded plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Handle binary classification
    if len(np.unique(y_true)) == 2:
        # Get positive class probabilities
        if len(y_proba_a.shape) > 1:
            y_proba_a_pos = y_proba_a[:, 1]
        else:
            y_proba_a_pos = y_proba_a
            
        if len(y_proba_b.shape) > 1:
            y_proba_b_pos = y_proba_b[:, 1]
        else:
            y_proba_b_pos = y_proba_b
        
        # Calculate ROC curves
        fpr_a, tpr_a, _ = roc_curve(y_true, y_proba_a_pos)
        fpr_b, tpr_b, _ = roc_curve(y_true, y_proba_b_pos)
        
        auc_a = auc(fpr_a, tpr_a)
        auc_b = auc(fpr_b, tpr_b)
        
        # Plot
        ax.plot(fpr_a, tpr_a, 'b-', lw=2, label=f'{model_names[0]} (AUC = {auc_a:.3f})')
        ax.plot(fpr_b, tpr_b, 'r-', lw=2, label=f'{model_names[1]} (AUC = {auc_b:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_base64


def plot_pr_curves_comparison(
    y_true: np.ndarray,
    y_proba_a: np.ndarray,
    y_proba_b: np.ndarray,
    model_names: Tuple[str, str]
) -> str:
    """
    Plot Precision-Recall curves comparison.
    
    Returns:
        Base64 encoded plot
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Handle binary classification
    if len(np.unique(y_true)) == 2:
        # Get positive class probabilities
        if len(y_proba_a.shape) > 1:
            y_proba_a_pos = y_proba_a[:, 1]
        else:
            y_proba_a_pos = y_proba_a
            
        if len(y_proba_b.shape) > 1:
            y_proba_b_pos = y_proba_b[:, 1]
        else:
            y_proba_b_pos = y_proba_b
        
        # Calculate PR curves
        precision_a, recall_a, _ = precision_recall_curve(y_true, y_proba_a_pos)
        precision_b, recall_b, _ = precision_recall_curve(y_true, y_proba_b_pos)
        
        auc_a = auc(recall_a, precision_a)
        auc_b = auc(recall_b, precision_b)
        
        # Plot
        ax.plot(recall_a, precision_a, 'b-', lw=2, label=f'{model_names[0]} (AUC = {auc_a:.3f})')
        ax.plot(recall_b, precision_b, 'r-', lw=2, label=f'{model_names[1]} (AUC = {auc_b:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves Comparison')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_base64


def plot_confusion_matrices_comparison(
    cm_a: np.ndarray,
    cm_b: np.ndarray,
    model_names: Tuple[str, str]
) -> str:
    """
    Plot confusion matrices comparison.
    
    Returns:
        Base64 encoded plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if sns is None:
        logger.warning(
            "Seaborn is not installed; using matplotlib heatmaps for confusion matrices."
        )

        im_a = ax1.imshow(cm_a, cmap='Blues')
        ax1.set_title(f'{model_names[0]} - Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')

        im_b = ax2.imshow(cm_b, cmap='Greens')
        ax2.set_title(f'{model_names[1]} - Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')

        for ax, cm in ((ax1, cm_a), (ax2, cm_b)):
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f"{cm[i, j]}", ha='center', va='center', color='black')

        fig.colorbar(im_a, ax=ax1, fraction=0.046, pad=0.04)
        fig.colorbar(im_b, ax=ax2, fraction=0.046, pad=0.04)
    else:
        # Model A
        sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title(f'{model_names[0]} - Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')

        # Model B
        sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=ax2, cbar_kws={'label': 'Count'})
        ax2.set_title(f'{model_names[1]} - Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_base64


def plot_residuals_comparison(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_names: Tuple[str, str]
) -> str:
    """
    Plot residuals comparison for regression.
    
    Returns:
        Base64 encoded plot
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    residuals_a = y_true - y_pred_a
    residuals_b = y_true - y_pred_b
    
    # Model A - Actual vs Predicted
    ax1.scatter(y_pred_a, y_true, alpha=0.5, color='blue', s=20)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title(f'{model_names[0]} - Actual vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Model B - Actual vs Predicted
    ax2.scatter(y_pred_b, y_true, alpha=0.5, color='green', s=20)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title(f'{model_names[1]} - Actual vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    # Model A - Residuals
    ax3.scatter(y_pred_a, residuals_a, alpha=0.5, color='blue', s=20)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Residuals')
    ax3.set_title(f'{model_names[0]} - Residual Plot')
    ax3.grid(True, alpha=0.3)
    
    # Model B - Residuals
    ax4.scatter(y_pred_b, residuals_b, alpha=0.5, color='green', s=20)
    ax4.axhline(y=0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Residuals')
    ax4.set_title(f'{model_names[1]} - Residual Plot')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return plot_base64


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


def calculate_model_confidence(
    y_proba: np.ndarray,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Calculate model confidence metrics.
    
    Args:
        y_proba: Predicted probabilities
        threshold: Confidence threshold
        
    Returns:
        Dictionary with confidence metrics
    """
    if len(y_proba.shape) > 1:
        # Multi-class: use max probability
        max_proba = np.max(y_proba, axis=1)
    else:
        # Binary: use as is
        max_proba = y_proba
    
    confidence_metrics = {
        'mean_confidence': float(np.mean(max_proba)),
        'std_confidence': float(np.std(max_proba)),
        'min_confidence': float(np.min(max_proba)),
        'max_confidence': float(np.max(max_proba)),
        'high_confidence_ratio': float(np.mean(max_proba >= threshold)),
        'low_confidence_ratio': float(np.mean(max_proba < threshold))
    }
    
    return confidence_metrics
