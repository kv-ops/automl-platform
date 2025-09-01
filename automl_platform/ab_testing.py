"""
A/B Testing Service for Model Comparison
=========================================
Place in: automl_platform/ab_testing.py

Complete A/B testing implementation with statistical analysis,
traffic routing, and performance comparison.
"""

import uuid
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status."""
    ACTIVE = "active"
    PAUSED = "paused"
    CONCLUDED = "concluded"
    FAILED = "failed"


class ModelType(Enum):
    """Model type in A/B test."""
    CHAMPION = "champion"
    CHALLENGER = "challenger"


@dataclass
class ABTestConfig:
    """Configuration for A/B test."""
    test_id: str
    model_name: str
    champion_version: int
    challenger_version: int
    traffic_split: float = 0.1  # Percentage to challenger
    min_samples: int = 100
    max_duration_days: int = 30
    confidence_level: float = 0.95
    
    # Metrics to track
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = field(default_factory=lambda: ["precision", "recall", "f1"])
    
    # Statistical test
    statistical_test: str = "t_test"  # t_test, chi_square, mann_whitney
    min_improvement: float = 0.02  # Minimum improvement to declare winner
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ABTestResult:
    """Results from A/B test."""
    test_id: str
    status: TestStatus
    
    # Sample counts
    champion_samples: int = 0
    challenger_samples: int = 0
    
    # Performance metrics
    champion_metrics: Dict[str, float] = field(default_factory=dict)
    challenger_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Statistical analysis
    p_value: float = None
    confidence_interval: Tuple[float, float] = None
    effect_size: float = None
    statistical_power: float = None
    
    # Winner determination
    winner: Optional[str] = None
    improvement: float = None
    confidence: float = None
    
    # Predictions log
    predictions_log: List[Dict] = field(default_factory=list)
    
    # Timeline
    started_at: datetime = field(default_factory=datetime.utcnow)
    concluded_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = asdict(self)
        result['status'] = self.status.value
        result['started_at'] = self.started_at.isoformat()
        if self.concluded_at:
            result['concluded_at'] = self.concluded_at.isoformat()
        return result


class MetricsComparator:
    """Compare metrics between models with visualization."""
    
    @staticmethod
    def compare_classification_metrics(
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        y_proba_a: Optional[np.ndarray] = None,
        y_proba_b: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compare classification metrics between two models.
        
        Returns:
            Dictionary with metrics comparison and visualizations
        """
        metrics = {
            'model_a': {},
            'model_b': {},
            'comparison': {},
            'visualizations': {}
        }
        
        # Basic metrics for Model A
        metrics['model_a']['accuracy'] = accuracy_score(y_true, y_pred_a)
        metrics['model_a']['precision'] = precision_score(y_true, y_pred_a, average='weighted')
        metrics['model_a']['recall'] = recall_score(y_true, y_pred_a, average='weighted')
        metrics['model_a']['f1'] = f1_score(y_true, y_pred_a, average='weighted')
        
        # Basic metrics for Model B
        metrics['model_b']['accuracy'] = accuracy_score(y_true, y_pred_b)
        metrics['model_b']['precision'] = precision_score(y_true, y_pred_b, average='weighted')
        metrics['model_b']['recall'] = recall_score(y_true, y_pred_b, average='weighted')
        metrics['model_b']['f1'] = f1_score(y_true, y_pred_b, average='weighted')
        
        # Calculate improvements
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            diff = metrics['model_b'][metric] - metrics['model_a'][metric]
            metrics['comparison'][f'{metric}_diff'] = diff
            metrics['comparison'][f'{metric}_improvement_pct'] = (diff / metrics['model_a'][metric] * 100) if metrics['model_a'][metric] > 0 else 0
        
        # ROC/AUC if probabilities available
        if y_proba_a is not None and y_proba_b is not None:
            # Handle binary and multiclass
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if len(y_proba_a.shape) > 1:
                    y_proba_a_pos = y_proba_a[:, 1]
                    y_proba_b_pos = y_proba_b[:, 1]
                else:
                    y_proba_a_pos = y_proba_a
                    y_proba_b_pos = y_proba_b
                
                metrics['model_a']['roc_auc'] = roc_auc_score(y_true, y_proba_a_pos)
                metrics['model_b']['roc_auc'] = roc_auc_score(y_true, y_proba_b_pos)
                
                # Generate ROC curves
                fpr_a, tpr_a, _ = roc_curve(y_true, y_proba_a_pos)
                fpr_b, tpr_b, _ = roc_curve(y_true, y_proba_b_pos)
                
                # Create ROC curve plot
                metrics['visualizations']['roc_curves'] = MetricsComparator._plot_roc_curves(
                    fpr_a, tpr_a, metrics['model_a']['roc_auc'],
                    fpr_b, tpr_b, metrics['model_b']['roc_auc']
                )
                
                # Precision-Recall curves
                precision_a, recall_a, _ = precision_recall_curve(y_true, y_proba_a_pos)
                precision_b, recall_b, _ = precision_recall_curve(y_true, y_proba_b_pos)
                
                metrics['model_a']['pr_auc'] = auc(recall_a, precision_a)
                metrics['model_b']['pr_auc'] = auc(recall_b, precision_b)
                
                metrics['visualizations']['pr_curves'] = MetricsComparator._plot_pr_curves(
                    recall_a, precision_a, metrics['model_a']['pr_auc'],
                    recall_b, precision_b, metrics['model_b']['pr_auc']
                )
        
        # Confusion matrices
        cm_a = confusion_matrix(y_true, y_pred_a)
        cm_b = confusion_matrix(y_true, y_pred_b)
        
        metrics['model_a']['confusion_matrix'] = cm_a.tolist()
        metrics['model_b']['confusion_matrix'] = cm_b.tolist()
        
        metrics['visualizations']['confusion_matrices'] = MetricsComparator._plot_confusion_matrices(cm_a, cm_b)
        
        return metrics
    
    @staticmethod
    def compare_regression_metrics(
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare regression metrics between two models.
        
        Returns:
            Dictionary with metrics comparison and visualizations
        """
        metrics = {
            'model_a': {},
            'model_b': {},
            'comparison': {},
            'visualizations': {}
        }
        
        # Model A metrics
        metrics['model_a']['mse'] = mean_squared_error(y_true, y_pred_a)
        metrics['model_a']['rmse'] = np.sqrt(metrics['model_a']['mse'])
        metrics['model_a']['mae'] = mean_absolute_error(y_true, y_pred_a)
        metrics['model_a']['r2'] = r2_score(y_true, y_pred_a)
        
        # Model B metrics
        metrics['model_b']['mse'] = mean_squared_error(y_true, y_pred_b)
        metrics['model_b']['rmse'] = np.sqrt(metrics['model_b']['mse'])
        metrics['model_b']['mae'] = mean_absolute_error(y_true, y_pred_b)
        metrics['model_b']['r2'] = r2_score(y_true, y_pred_b)
        
        # Calculate improvements (lower is better for MSE, MAE, RMSE)
        for metric in ['mse', 'rmse', 'mae']:
            diff = metrics['model_a'][metric] - metrics['model_b'][metric]  # Reversed
            metrics['comparison'][f'{metric}_reduction'] = diff
            metrics['comparison'][f'{metric}_improvement_pct'] = (diff / metrics['model_a'][metric] * 100) if metrics['model_a'][metric] > 0 else 0
        
        # R2 improvement (higher is better)
        r2_diff = metrics['model_b']['r2'] - metrics['model_a']['r2']
        metrics['comparison']['r2_improvement'] = r2_diff
        
        # Residual plots
        residuals_a = y_true - y_pred_a
        residuals_b = y_true - y_pred_b
        
        metrics['visualizations']['residual_plots'] = MetricsComparator._plot_residuals(
            y_true, y_pred_a, residuals_a,
            y_true, y_pred_b, residuals_b
        )
        
        return metrics
    
    @staticmethod
    def _plot_roc_curves(fpr_a, tpr_a, auc_a, fpr_b, tpr_b, auc_b) -> str:
        """Create ROC curve comparison plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr_a, tpr_a, color='blue', lw=2, 
                label=f'Model A (AUC = {auc_a:.3f})')
        ax.plot(fpr_b, tpr_b, color='red', lw=2,
                label=f'Model B (AUC = {auc_b:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_base64
    
    @staticmethod
    def _plot_pr_curves(recall_a, precision_a, auc_a, recall_b, precision_b, auc_b) -> str:
        """Create Precision-Recall curve comparison plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall_a, precision_a, color='blue', lw=2,
                label=f'Model A (AUC = {auc_a:.3f})')
        ax.plot(recall_b, precision_b, color='red', lw=2,
                label=f'Model B (AUC = {auc_b:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve Comparison')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_base64
    
    @staticmethod
    def _plot_confusion_matrices(cm_a, cm_b) -> str:
        """Create confusion matrix comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Model A
        sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Model A - Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        # Model B
        sns.heatmap(cm_b, annot=True, fmt='d', cmap='Reds', ax=ax2)
        ax2.set_title('Model B - Confusion Matrix')
        ax2.set_ylabel('True Label')
        ax2.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_base64
    
    @staticmethod
    def _plot_residuals(y_true_a, y_pred_a, residuals_a,
                       y_true_b, y_pred_b, residuals_b) -> str:
        """Create residual plots comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Model A - Predicted vs Actual
        ax1.scatter(y_pred_a, y_true_a, alpha=0.5, color='blue')
        ax1.plot([y_true_a.min(), y_true_a.max()], 
                [y_true_a.min(), y_true_a.max()], 'r--', lw=2)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_title('Model A - Predicted vs Actual')
        
        # Model B - Predicted vs Actual
        ax2.scatter(y_pred_b, y_true_b, alpha=0.5, color='red')
        ax2.plot([y_true_b.min(), y_true_b.max()],
                [y_true_b.min(), y_true_b.max()], 'r--', lw=2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title('Model B - Predicted vs Actual')
        
        # Model A - Residual plot
        ax3.scatter(y_pred_a, residuals_a, alpha=0.5, color='blue')
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Model A - Residual Plot')
        
        # Model B - Residual plot
        ax4.scatter(y_pred_b, residuals_b, alpha=0.5, color='red')
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Model B - Residual Plot')
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return plot_base64


class StatisticalTester:
    """Statistical testing for A/B tests."""
    
    @staticmethod
    def perform_t_test(
        samples_a: np.ndarray,
        samples_b: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, float]:
        """
        Perform t-test between two samples.
        
        Args:
            samples_a: Samples from model A
            samples_b: Samples from model B
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test results
        """
        # Perform t-test
        statistic, p_value = stats.ttest_ind(samples_a, samples_b, alternative=alternative)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(samples_a) + np.var(samples_b)) / 2)
        effect_size = (np.mean(samples_b) - np.mean(samples_a)) / pooled_std if pooled_std > 0 else 0
        
        # Calculate confidence interval
        mean_diff = np.mean(samples_b) - np.mean(samples_a)
        se_diff = np.sqrt(np.var(samples_a)/len(samples_a) + np.var(samples_b)/len(samples_b))
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'mean_a': float(np.mean(samples_a)),
            'mean_b': float(np.mean(samples_b)),
            'std_a': float(np.std(samples_a)),
            'std_b': float(np.std(samples_b)),
            'confidence_interval': (float(ci_lower), float(ci_upper))
        }
    
    @staticmethod
    def perform_mann_whitney(
        samples_a: np.ndarray,
        samples_b: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, float]:
        """
        Perform Mann-Whitney U test (non-parametric).
        
        Returns:
            Dictionary with test results
        """
        statistic, p_value = stats.mannwhitneyu(samples_a, samples_b, alternative=alternative)
        
        # Calculate effect size (rank biserial correlation)
        n1, n2 = len(samples_a), len(samples_b)
        effect_size = 1 - (2*statistic) / (n1 * n2)
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'median_a': float(np.median(samples_a)),
            'median_b': float(np.median(samples_b))
        }
    
    @staticmethod
    def perform_chi_square(
        contingency_table: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform chi-square test for categorical outcomes.
        
        Returns:
            Dictionary with test results
        """
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Calculate Cramér's V for effect size
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        return {
            'statistic': float(chi2),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'effect_size': float(cramers_v)
        }
    
    @staticmethod
    def calculate_sample_size(
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> int:
        """
        Calculate required sample size for given effect size and power.
        
        Returns:
            Required sample size per group
        """
        from statsmodels.stats.power import TTestPower
        
        analysis = TTestPower()
        sample_size = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative='two-sided'
        )
        
        return int(np.ceil(sample_size))


class ABTestingService:
    """Main A/B testing service."""
    
    def __init__(self, registry=None):
        self.registry = registry
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, ABTestResult] = {}
        self.model_cache: Dict[str, Any] = {}
        
    def create_ab_test(self,
                      model_name: str,
                      champion_version: int,
                      challenger_version: int,
                      traffic_split: float = 0.1,
                      min_samples: int = 100,
                      confidence_level: float = 0.95,
                      primary_metric: str = "accuracy") -> str:
        """
        Create new A/B test.
        
        Returns:
            Test ID
        """
        test_id = str(uuid.uuid4())
        
        config = ABTestConfig(
            test_id=test_id,
            model_name=model_name,
            champion_version=champion_version,
            challenger_version=challenger_version,
            traffic_split=traffic_split,
            min_samples=min_samples,
            confidence_level=confidence_level,
            primary_metric=primary_metric
        )
        
        result = ABTestResult(
            test_id=test_id,
            status=TestStatus.ACTIVE
        )
        
        self.active_tests[test_id] = config
        self.test_results[test_id] = result
        
        logger.info(f"Created A/B test {test_id} for model {model_name}")
        
        return test_id
    
    def route_prediction(self, test_id: str) -> Tuple[str, int]:
        """
        Route prediction to champion or challenger based on traffic split.
        
        Returns:
            Tuple of (model_type, version)
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        config = self.active_tests[test_id]
        
        # Use hash for consistent routing
        random_value = np.random.random()
        
        if random_value < config.traffic_split:
            return (ModelType.CHALLENGER.value, config.challenger_version)
        else:
            return (ModelType.CHAMPION.value, config.champion_version)
    
    def record_result(self,
                     test_id: str,
                     model_type: str,
                     success: bool,
                     metric_value: Optional[float] = None,
                     response_time: Optional[float] = None):
        """Record prediction result for A/B test."""
        if test_id not in self.test_results:
            return
        
        result = self.test_results[test_id]
        
        # Update sample counts
        if model_type == ModelType.CHAMPION.value:
            result.champion_samples += 1
        else:
            result.challenger_samples += 1
        
        # Log prediction
        result.predictions_log.append({
            'timestamp': datetime.utcnow().isoformat(),
            'model_type': model_type,
            'success': success,
            'metric_value': metric_value,
            'response_time': response_time
        })
        
        # Check if we have enough samples
        config = self.active_tests[test_id]
        if (result.champion_samples >= config.min_samples and 
            result.challenger_samples >= config.min_samples):
            
            # Auto-conclude if we have enough data
            self._analyze_results(test_id)
    
    def _analyze_results(self, test_id: str):
        """Analyze A/B test results."""
        config = self.active_tests[test_id]
        result = self.test_results[test_id]
        
        # Extract metrics from logs
        champion_metrics = [
            log['metric_value'] for log in result.predictions_log
            if log['model_type'] == ModelType.CHAMPION.value and log['metric_value'] is not None
        ]
        
        challenger_metrics = [
            log['metric_value'] for log in result.predictions_log
            if log['model_type'] == ModelType.CHALLENGER.value and log['metric_value'] is not None
        ]
        
        if len(champion_metrics) > 0 and len(challenger_metrics) > 0:
            # Perform statistical test
            if config.statistical_test == 't_test':
                test_results = StatisticalTester.perform_t_test(
                    np.array(champion_metrics),
                    np.array(challenger_metrics)
                )
            elif config.statistical_test == 'mann_whitney':
                test_results = StatisticalTester.perform_mann_whitney(
                    np.array(champion_metrics),
                    np.array(challenger_metrics)
                )
            else:
                test_results = StatisticalTester.perform_t_test(
                    np.array(champion_metrics),
                    np.array(challenger_metrics)
                )
            
            # Update result
            result.p_value = test_results['p_value']
            result.effect_size = test_results['effect_size']
            result.confidence_interval = test_results.get('confidence_interval')
            
            # Calculate metrics
            result.champion_metrics = {
                'mean': float(np.mean(champion_metrics)),
                'std': float(np.std(champion_metrics)),
                'median': float(np.median(champion_metrics))
            }
            
            result.challenger_metrics = {
                'mean': float(np.mean(challenger_metrics)),
                'std': float(np.std(challenger_metrics)),
                'median': float(np.median(challenger_metrics))
            }
            
            # Determine winner
            improvement = result.challenger_metrics['mean'] - result.champion_metrics['mean']
            result.improvement = improvement
            
            if result.p_value < (1 - config.confidence_level):
                if improvement > config.min_improvement:
                    result.winner = ModelType.CHALLENGER.value
                    result.confidence = 1 - result.p_value
                elif improvement < -config.min_improvement:
                    result.winner = ModelType.CHAMPION.value
                    result.confidence = 1 - result.p_value
    
    def get_test_results(self, test_id: str) -> Optional[Dict]:
        """Get current test results."""
        if test_id not in self.test_results:
            return None
        
        result = self.test_results[test_id]
        config = self.active_tests[test_id]
        
        return {
            'test_id': test_id,
            'model_name': config.model_name,
            'status': result.status.value,
            'champion_version': config.champion_version,
            'challenger_version': config.challenger_version,
            'champion_samples': result.champion_samples,
            'challenger_samples': result.challenger_samples,
            'champion_metrics': result.champion_metrics,
            'challenger_metrics': result.challenger_metrics,
            'p_value': result.p_value,
            'effect_size': result.effect_size,
            'confidence_interval': result.confidence_interval,
            'winner': result.winner,
            'improvement': result.improvement,
            'confidence': result.confidence,
            'started_at': result.started_at.isoformat(),
            'traffic_split': config.traffic_split
        }
    
    def conclude_test(self, test_id: str, promote_winner: bool = False) -> Dict:
        """
        Conclude A/B test and optionally promote winner.
        
        Returns:
            Final test results
        """
        if test_id not in self.test_results:
            return {"error": "Test not found"}
        
        result = self.test_results[test_id]
        config = self.active_tests[test_id]
        
        # Final analysis
        self._analyze_results(test_id)
        
        # Mark as concluded
        result.status = TestStatus.CONCLUDED
        result.concluded_at = datetime.utcnow()
        
        # Promote winner if requested and registry available
        if promote_winner and result.winner and self.registry:
            if result.winner == ModelType.CHALLENGER.value:
                # Promote challenger to production
                self.registry.promote_model(
                    config.model_name,
                    config.challenger_version,
                    "Production"
                )
                logger.info(f"Promoted challenger version {config.challenger_version} to production")
        
        # Remove from active tests
        del self.active_tests[test_id]
        
        return self.get_test_results(test_id)
    
    def get_active_tests(self) -> List[Dict]:
        """Get list of active A/B tests."""
        active = []
        for test_id, config in self.active_tests.items():
            result = self.test_results.get(test_id)
            if result and result.status == TestStatus.ACTIVE:
                active.append({
                    'test_id': test_id,
                    'model_name': config.model_name,
                    'champion_version': config.champion_version,
                    'challenger_version': config.challenger_version,
                    'samples_collected': result.champion_samples + result.challenger_samples,
                    'min_samples_required': config.min_samples * 2,
                    'started_at': result.started_at.isoformat()
                })
        return active
    
    def pause_test(self, test_id: str):
        """Pause an A/B test."""
        if test_id in self.test_results:
            self.test_results[test_id].status = TestStatus.PAUSED
            logger.info(f"Paused A/B test {test_id}")
    
    def resume_test(self, test_id: str):
        """Resume a paused A/B test."""
        if test_id in self.test_results:
            self.test_results[test_id].status = TestStatus.ACTIVE
            logger.info(f"Resumed A/B test {test_id}")
    
    def compare_models_offline(self,
                              model_a: Any,
                              model_b: Any,
                              X_test: pd.DataFrame,
                              y_test: pd.Series,
                              task: str = "classification") -> Dict:
        """
        Perform offline comparison of two models.
        
        Args:
            model_a: First model
            model_b: Second model
            X_test: Test features
            y_test: Test labels
            task: Task type
            
        Returns:
            Comparison results with metrics and visualizations
        """
        # Make predictions
        y_pred_a = model_a.predict(X_test)
        y_pred_b = model_b.predict(X_test)
        
        # Get probabilities if available
        y_proba_a = None
        y_proba_b = None
        
        if task == "classification":
            if hasattr(model_a, 'predict_proba'):
                y_proba_a = model_a.predict_proba(X_test)
            if hasattr(model_b, 'predict_proba'):
                y_proba_b = model_b.predict_proba(X_test)
            
            return MetricsComparator.compare_classification_metrics(
                y_test.values, y_pred_a, y_pred_b, y_proba_a, y_proba_b
            )
        else:
            return MetricsComparator.compare_regression_metrics(
                y_test.values, y_pred_a, y_pred_b
            )
