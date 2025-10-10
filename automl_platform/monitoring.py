"""
Monitoring module for ML Platform
Handles model performance tracking, data drift detection, and system metrics
Integrates with Prometheus for metrics export
WITH COMPLETE SLACK/EMAIL INTEGRATION AND BILLING TRACKING
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, log_loss, confusion_matrix, classification_report
)

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    warnings.warn("prometheus_client not installed. Metrics export will be disabled.")

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    warnings.warn("evidently not installed. Advanced drift detection will be limited.")

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Container for model performance metrics with billing info"""
    model_id: str
    timestamp: str
    prediction_count: int
    tenant_id: str = "default"
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    auc_roc: Optional[float] = None
    log_loss_value: Optional[float] = None
    
    # Regression metrics
    mse: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    
    # Additional metrics
    confusion_matrix_data: Optional[List] = None
    feature_importance: Optional[Dict] = None
    prediction_distribution: Optional[Dict] = None
    
    # Drift metrics
    data_drift_detected: bool = False
    concept_drift_detected: bool = False
    drift_metrics: Optional[Dict] = None
    
    # Billing metrics
    compute_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    api_calls_count: int = 0
    
    def to_dict(self):
        return asdict(self)


class DriftDetector:
    """Detects data and concept drift in production with alerting"""
    
    def __init__(self, 
                 reference_data: pd.DataFrame = None,
                 sensitivity: float = 0.05,
                 alert_callback: callable = None):
        """
        Initialize drift detector
        
        Args:
            reference_data: Reference dataset for comparison
            sensitivity: P-value threshold for drift detection
            alert_callback: Function to call when drift detected
        """
        self.reference_data = reference_data
        self.sensitivity = sensitivity
        self.drift_history = []
        self.alert_callback = alert_callback
        
        if reference_data is not None:
            self._calculate_reference_stats()
    
    def _calculate_reference_stats(self):
        """Calculate statistics for reference data"""
        self.reference_stats = {}
        
        for col in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                self.reference_stats[col] = {
                    'mean': self.reference_data[col].mean(),
                    'std': self.reference_data[col].std(),
                    'min': self.reference_data[col].min(),
                    'max': self.reference_data[col].max(),
                    'quantiles': self.reference_data[col].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            else:
                self.reference_stats[col] = {
                    'unique_values': self.reference_data[col].nunique(),
                    'value_counts': self.reference_data[col].value_counts().to_dict()
                }
    
    def detect_data_drift(self, current_data: pd.DataFrame, tenant_id: str = "default") -> Dict:
        """
        Detect data drift using statistical tests with tenant tracking
        
        Args:
            current_data: Current production data
            tenant_id: Tenant identifier for billing
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            logger.warning("No reference data available for drift detection")
            return {"drift_detected": False, "message": "No reference data"}
        
        drift_results = {
            "drift_detected": False,
            "drifted_features": [],
            "drift_scores": {},
            "timestamp": datetime.now().isoformat(),
            "tenant_id": tenant_id
        }
        
        # Use Evidently if available for comprehensive drift detection
        if EVIDENTLY_AVAILABLE:
            try:
                drift_report = Report(metrics=[DataDriftPreset()])
                drift_report.run(
                    reference_data=self.reference_data,
                    current_data=current_data
                )
                
                result = drift_report.as_dict()
                if result['metrics'][0]['result']['dataset_drift']:
                    drift_results["drift_detected"] = True
                    drift_results["evidently_report"] = result
                    
            except Exception as e:
                logger.error(f"Evidently drift detection failed: {e}")
        
        # Manual drift detection for each feature
        for col in current_data.columns:
            if col not in self.reference_data.columns:
                continue
            
            drift_score = 0.0
            is_drifted = False
            
            if pd.api.types.is_numeric_dtype(current_data[col]):
                # Kolmogorov-Smirnov test for numerical features
                statistic, p_value = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                drift_score = 1 - p_value
                is_drifted = p_value < self.sensitivity
                
            else:
                # Chi-square test for categorical features
                ref_counts = self.reference_data[col].value_counts()
                curr_counts = current_data[col].value_counts()
                
                # Align categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_aligned = pd.Series([ref_counts.get(cat, 0) for cat in all_categories])
                curr_aligned = pd.Series([curr_counts.get(cat, 0) for cat in all_categories])
                
                if len(all_categories) > 1:
                    chi2, p_value = stats.chisquare(
                        curr_aligned + 1,  # Add 1 to avoid division by zero
                        ref_aligned + 1
                    )
                    drift_score = 1 - p_value
                    is_drifted = p_value < self.sensitivity
            
            drift_results["drift_scores"][col] = drift_score
            
            if is_drifted:
                drift_results["drifted_features"].append(col)
                drift_results["drift_detected"] = True
        
        # Store in history
        self.drift_history.append(drift_results)
        
        # Trigger alert if drift detected
        if drift_results["drift_detected"] and self.alert_callback:
            self.alert_callback(drift_results)
        
        return drift_results
    
    def detect_concept_drift(self, 
                            predictions: np.ndarray,
                            actuals: np.ndarray,
                            window_size: int = 100) -> Dict:
        """
        Detect concept drift using prediction error analysis
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            window_size: Size of sliding window
            
        Returns:
            Dictionary with concept drift detection results
        """
        if len(predictions) < window_size * 2:
            return {"concept_drift_detected": False, "message": "Insufficient data"}
        
        errors = np.abs(predictions - actuals)
        
        # Page-Hinkley test for concept drift
        threshold = 0.05
        drift_detected = False
        drift_points = []
        
        mean_error = np.mean(errors[:window_size])
        sum_error = 0
        min_sum = 0
        
        for i in range(window_size, len(errors)):
            sum_error += errors[i] - mean_error - threshold
            if sum_error < min_sum:
                min_sum = sum_error
            
            ph_statistic = sum_error - min_sum
            
            if ph_statistic > threshold * window_size:
                drift_detected = True
                drift_points.append(i)
        
        result = {
            "concept_drift_detected": drift_detected,
            "drift_points": drift_points,
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "timestamp": datetime.now().isoformat()
        }
        
        # Trigger alert if concept drift detected
        if drift_detected and self.alert_callback:
            self.alert_callback(result)
        
        return result
    
    def calculate_psi(self, 
                     expected: pd.Series,
                     actual: pd.Series,
                     bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Args:
            expected: Expected distribution
            actual: Actual distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI value (< 0.1: no shift, 0.1-0.25: small shift, > 0.25: large shift)
        """
        # Create bins based on expected distribution
        if pd.api.types.is_numeric_dtype(expected):
            breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
            breakpoints[0] = -np.inf
            breakpoints[-1] = np.inf
            
            expected_bins = pd.cut(expected, breakpoints).value_counts().sort_index()
            actual_bins = pd.cut(actual, breakpoints).value_counts().sort_index()
        else:
            expected_bins = expected.value_counts()
            actual_bins = actual.value_counts()
            
            # Align categories
            all_categories = set(expected_bins.index) | set(actual_bins.index)
            expected_bins = pd.Series([expected_bins.get(cat, 0) for cat in all_categories])
            actual_bins = pd.Series([actual_bins.get(cat, 0) for cat in all_categories])
        
        # Calculate percentages
        expected_percents = expected_bins / len(expected)
        actual_percents = actual_bins / len(actual)
        
        # Avoid division by zero
        expected_percents = expected_percents.replace(0, 0.0001)
        actual_percents = actual_percents.replace(0, 0.0001)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return float(psi)


class ModelMonitor:
    """Main monitoring class for ML models in production with billing integration"""
    
    def __init__(self, 
                 model_id: str,
                 model_type: str = "classification",
                 reference_data: pd.DataFrame = None,
                 prometheus_port: int = 8000,
                 tenant_id: str = "default",
                 billing_tracker: Optional[Any] = None):
        """
        Initialize model monitor
        
        Args:
            model_id: Unique model identifier
            model_type: Type of model ('classification' or 'regression')
            reference_data: Reference dataset for drift detection
            prometheus_port: Port for Prometheus metrics endpoint
            tenant_id: Tenant identifier for multi-tenancy
            billing_tracker: Billing tracker instance
        """
        self.model_id = model_id
        self.model_type = model_type
        self.reference_data = reference_data
        self.prometheus_port = prometheus_port
        self.tenant_id = tenant_id
        self.billing_tracker = billing_tracker
        
        # Initialize drift detector with alert callback
        self.drift_detector = DriftDetector(
            reference_data, 
            alert_callback=self._drift_alert_callback
        )
        
        # Performance history
        self.performance_history = []
        self.prediction_history = []
        
        # Billing metrics
        self.total_predictions = 0
        self.total_compute_time = 0.0
        self.total_api_calls = 0
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics with billing metrics"""
        self.registry = CollectorRegistry()
        
        # Counters
        self.prediction_counter = Counter(
            'ml_predictions_total',
            'Total number of predictions',
            ['model_id', 'tenant_id'],
            registry=self.registry
        )
        
        self.error_counter = Counter(
            'ml_prediction_errors_total',
            'Total number of prediction errors',
            ['model_id', 'error_type'],
            registry=self.registry
        )
        
        # Billing metrics
        self.billing_counter = Counter(
            'ml_billing_api_calls_total',
            'Total API calls for billing',
            ['model_id', 'tenant_id'],
            registry=self.registry
        )
        
        self.compute_time_counter = Counter(
            'ml_billing_compute_seconds_total',
            'Total compute time in seconds',
            ['model_id', 'tenant_id'],
            registry=self.registry
        )
        
        # Gauges
        self.accuracy_gauge = Gauge(
            'ml_model_accuracy',
            'Current model accuracy',
            ['model_id'],
            registry=self.registry
        )
        
        self.drift_gauge = Gauge(
            'ml_data_drift_score',
            'Data drift score',
            ['model_id', 'feature'],
            registry=self.registry
        )
        
        # Histograms
        self.prediction_latency = Histogram(
            'ml_prediction_duration_seconds',
            'Prediction latency in seconds',
            ['model_id', 'tenant_id'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry
        )
        
        self.prediction_value = Histogram(
            'ml_prediction_values',
            'Distribution of prediction values',
            ['model_id'],
            buckets=np.linspace(0, 1, 11).tolist(),
            registry=self.registry
        )
        
        logger.info(f"Prometheus metrics initialized for model {self.model_id}")
    
    def _drift_alert_callback(self, drift_info: Dict):
        """Callback for drift detection alerts"""
        alert = {
            'type': 'drift_detected',
            'model_id': self.model_id,
            'tenant_id': self.tenant_id,
            'drift_info': drift_info,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to monitoring integration
        from .monitoring import MonitoringIntegration
        
        # Send Slack alert
        if os.getenv('SLACK_WEBHOOK_URL'):
            MonitoringIntegration.send_to_slack(alert, os.getenv('SLACK_WEBHOOK_URL'))
        
        # Send email alert
        if os.getenv('ALERT_EMAIL_RECIPIENTS'):
            smtp_config = {
                'host': os.getenv('SMTP_HOST', 'localhost'),
                'port': int(os.getenv('SMTP_PORT', 587)),
                'from_email': os.getenv('SMTP_FROM_EMAIL', 'alerts@automl.com'),
                'username': os.getenv('SMTP_USERNAME'),
                'password': os.getenv('SMTP_PASSWORD')
            }
            recipients = os.getenv('ALERT_EMAIL_RECIPIENTS').split(',')
            MonitoringIntegration.send_to_email(alert, smtp_config, recipients)
    
    def log_prediction(self,
                      features: pd.DataFrame,
                      predictions: np.ndarray,
                      actuals: np.ndarray = None,
                      prediction_time: float = None):
        """
        Log predictions for monitoring with billing tracking
        
        Args:
            features: Input features
            predictions: Model predictions
            actuals: Actual values (if available)
            prediction_time: Time taken for prediction
        """
        # Track for billing
        self.total_predictions += len(predictions)
        self.total_api_calls += 1
        
        if prediction_time:
            self.total_compute_time += prediction_time
        
        # Track in billing system if available
        if self.billing_tracker:
            self.billing_tracker.track_predictions(self.tenant_id, len(predictions))
            self.billing_tracker.track_api_call(self.tenant_id, 'prediction')
            if prediction_time:
                self.billing_tracker.track_compute_time(self.tenant_id, prediction_time)
        
        # Store prediction
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'features': features.to_dict('records') if len(features) < 100 else None,
            'predictions': predictions.tolist() if len(predictions) < 100 else predictions.mean(),
            'actuals': actuals.tolist() if actuals is not None and len(actuals) < 100 else None,
            'prediction_time': prediction_time,
            'tenant_id': self.tenant_id
        }
        
        self.prediction_history.append(prediction_record)
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self.prediction_counter.labels(
                model_id=self.model_id,
                tenant_id=self.tenant_id
            ).inc(len(predictions))
            
            self.billing_counter.labels(
                model_id=self.model_id,
                tenant_id=self.tenant_id
            ).inc()
            
            if prediction_time:
                self.prediction_latency.labels(
                    model_id=self.model_id,
                    tenant_id=self.tenant_id
                ).observe(prediction_time)
                
                self.compute_time_counter.labels(
                    model_id=self.model_id,
                    tenant_id=self.tenant_id
                ).inc(prediction_time)
            
            for pred in predictions:
                self.prediction_value.labels(model_id=self.model_id).observe(float(pred))
        
        # Check for drift periodically
        if len(self.prediction_history) % 100 == 0:
            self.check_drift(features)
        
        # Calculate performance if actuals are available
        if actuals is not None:
            self.calculate_performance(predictions, actuals)
    
    def calculate_performance(self,
                            predictions: np.ndarray,
                            actuals: np.ndarray,
                            sample_weight: np.ndarray = None) -> ModelPerformanceMetrics:
        """
        Calculate model performance metrics with billing info
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            sample_weight: Sample weights
            
        Returns:
            ModelPerformanceMetrics object
        """
        import time
        start_time = time.time()
        
        metrics = ModelPerformanceMetrics(
            model_id=self.model_id,
            timestamp=datetime.now().isoformat(),
            prediction_count=len(predictions),
            tenant_id=self.tenant_id
        )
        
        try:
            if self.model_type == "classification":
                # Binary or multiclass classification metrics
                metrics.accuracy = accuracy_score(actuals, predictions, sample_weight=sample_weight)
                
                # Handle binary vs multiclass
                if len(np.unique(actuals)) == 2:
                    metrics.precision = precision_score(actuals, predictions, sample_weight=sample_weight)
                    metrics.recall = recall_score(actuals, predictions, sample_weight=sample_weight)
                    metrics.f1 = f1_score(actuals, predictions, sample_weight=sample_weight)
                    
                    # AUC-ROC if probabilities are available
                    if predictions.dtype == float and predictions.min() >= 0 and predictions.max() <= 1:
                        metrics.auc_roc = roc_auc_score(actuals, predictions, sample_weight=sample_weight)
                else:
                    metrics.precision = precision_score(actuals, predictions, average='weighted', sample_weight=sample_weight)
                    metrics.recall = recall_score(actuals, predictions, average='weighted', sample_weight=sample_weight)
                    metrics.f1 = f1_score(actuals, predictions, average='weighted', sample_weight=sample_weight)
                
                # Confusion matrix
                cm = confusion_matrix(actuals, predictions)
                metrics.confusion_matrix_data = cm.tolist()
                
            elif self.model_type == "regression":
                # Regression metrics
                metrics.mse = mean_squared_error(actuals, predictions, sample_weight=sample_weight)
                metrics.mae = mean_absolute_error(actuals, predictions, sample_weight=sample_weight)
                metrics.rmse = np.sqrt(metrics.mse)
                metrics.r2 = r2_score(actuals, predictions, sample_weight=sample_weight)
                
                # MAPE (Mean Absolute Percentage Error)
                mask = actuals != 0
                if np.any(mask):
                    metrics.mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
            
            # Calculate compute time
            metrics.compute_time_seconds = time.time() - start_time
            
            # Track API call
            metrics.api_calls_count = 1
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and metrics.accuracy is not None:
                self.accuracy_gauge.labels(model_id=self.model_id).set(metrics.accuracy)
            
            # Store in history
            self.performance_history.append(metrics)
            
            # Track in billing
            if self.billing_tracker:
                self.billing_tracker.track_compute_time(self.tenant_id, metrics.compute_time_seconds)
            
            logger.info(f"Performance calculated for model {self.model_id}: "
                       f"Accuracy={metrics.accuracy:.3f}" if metrics.accuracy else f"R2={metrics.r2:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            if PROMETHEUS_AVAILABLE:
                self.error_counter.labels(model_id=self.model_id, error_type='performance_calculation').inc()
        
        return metrics
    
    def check_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Check for data drift

        Args:
            current_data: Current production data

        Returns:
            Drift detection results
        """
        drift_results = self.drift_detector.detect_data_drift(current_data, self.tenant_id)
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            for feature, score in drift_results.get("drift_scores", {}).items():
                self.drift_gauge.labels(model_id=self.model_id, feature=feature).set(score)
        
        # Log significant drift
        if drift_results["drift_detected"]:
            logger.warning(f"Data drift detected for model {self.model_id}: "
                          f"{drift_results['drifted_features']}")
        
        return drift_results
    
    def get_performance_summary(self, 
                               last_n_days: int = 7) -> Dict:
        """
        Get performance summary for the last N days with billing info
        
        Args:
            last_n_days: Number of days to look back
            
        Returns:
            Performance summary dictionary
        """
        cutoff_date = datetime.now() - timedelta(days=last_n_days)
        
        recent_metrics = [
            m for m in self.performance_history
            if datetime.fromisoformat(m.timestamp) > cutoff_date
        ]
        
        if not recent_metrics:
            return {"message": "No recent performance data available"}
        
        summary = {
            "model_id": self.model_id,
            "tenant_id": self.tenant_id,
            "period": f"Last {last_n_days} days",
            "total_predictions": sum(m.prediction_count for m in recent_metrics),
            "metrics": {},
            "billing": {
                "total_api_calls": sum(m.api_calls_count for m in recent_metrics),
                "total_compute_seconds": sum(m.compute_time_seconds for m in recent_metrics),
                "avg_latency_seconds": np.mean([m.compute_time_seconds for m in recent_metrics])
            }
        }
        
        # Aggregate metrics
        if self.model_type == "classification":
            if recent_metrics[0].accuracy is not None:
                summary["metrics"]["avg_accuracy"] = np.mean([m.accuracy for m in recent_metrics if m.accuracy])
                summary["metrics"]["min_accuracy"] = np.min([m.accuracy for m in recent_metrics if m.accuracy])
                summary["metrics"]["max_accuracy"] = np.max([m.accuracy for m in recent_metrics if m.accuracy])
            
            if recent_metrics[0].f1 is not None:
                summary["metrics"]["avg_f1"] = np.mean([m.f1 for m in recent_metrics if m.f1])
        
        elif self.model_type == "regression":
            if recent_metrics[0].rmse is not None:
                summary["metrics"]["avg_rmse"] = np.mean([m.rmse for m in recent_metrics if m.rmse])
                summary["metrics"]["avg_mae"] = np.mean([m.mae for m in recent_metrics if m.mae])
                summary["metrics"]["avg_r2"] = np.mean([m.r2 for m in recent_metrics if m.r2])
        
        # Drift summary
        drift_events = [m for m in recent_metrics if m.data_drift_detected]
        summary["drift_events"] = len(drift_events)
        summary["drift_rate"] = len(drift_events) / len(recent_metrics) if recent_metrics else 0

        return summary

    def _filter_metrics_dict(self, metrics: ModelPerformanceMetrics) -> Dict[str, Any]:
        """Convert ModelPerformanceMetrics to a clean dictionary."""
        metrics_dict = metrics.to_dict()
        # Remove fields with None values for clarity
        metrics_dict = {k: v for k, v in metrics_dict.items() if v is not None}

        # Provide friendly aliases expected by other services
        if "auc_roc" in metrics_dict and "auc" not in metrics_dict:
            metrics_dict["auc"] = metrics_dict["auc_roc"]

        return metrics_dict

    def get_current_performance(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Return the most recent performance metrics for this monitor."""
        if model_name and model_name != self.model_id:
            return {}

        if not self.performance_history:
            return {}

        latest_metrics = self.performance_history[-1]
        return self._filter_metrics_dict(latest_metrics)

    def get_baseline_performance(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Return the earliest recorded performance metrics for this monitor."""
        if model_name and model_name != self.model_id:
            return {}

        if not self.performance_history:
            return {}

        baseline_metrics = self.performance_history[0]
        return self._filter_metrics_dict(baseline_metrics)

    def get_drift_score(self, model_name: Optional[str] = None) -> float:
        """Return the latest drift score aggregated across monitored features."""
        if model_name and model_name != self.model_id:
            return 0.0

        if not self.drift_detector.drift_history:
            return 0.0

        last_check = self.drift_detector.drift_history[-1]
        drift_scores = last_check.get("drift_scores", {}) or {}

        if drift_scores:
            return float(max(drift_scores.values()))

        # Fallback when Evidently reports drift without feature scores
        return 1.0 if last_check.get("drift_detected") else 0.0

    def get_new_data_count(self, model_name: Optional[str] = None) -> int:
        """Return the number of new data points logged since monitoring started."""
        if model_name and model_name != self.model_id:
            return 0

        total_records = 0
        for record in self.prediction_history:
            if isinstance(record.get("predictions"), list):
                total_records += len(record["predictions"])
            elif isinstance(record.get("features"), list):
                total_records += len(record["features"])
            elif isinstance(record.get("actuals"), list):
                total_records += len(record["actuals"])
            else:
                total_records += int(record.get("prediction_count", 1))

        if total_records == 0:
            total_records = int(self.total_predictions)

        return total_records

    def get_performance_metrics(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Return baseline and current performance metrics with degradations."""
        if model_name and model_name != self.model_id:
            return {}

        baseline = self.get_baseline_performance()
        current = self.get_current_performance()

        metrics: Dict[str, Any] = {
            "model_id": self.model_id,
            "baseline_metrics": baseline,
            "current_metrics": current,
        }

        key_pairs = [
            ("accuracy", "baseline_accuracy", "current_accuracy"),
            ("auc", "baseline_auc", "current_auc"),
            ("f1", "baseline_f1", "current_f1"),
            ("precision", "baseline_precision", "current_precision"),
            ("recall", "baseline_recall", "current_recall"),
            ("r2", "baseline_r2", "current_r2"),
            ("rmse", "baseline_rmse", "current_rmse"),
            ("mae", "baseline_mae", "current_mae"),
        ]

        for key, baseline_key, current_key in key_pairs:
            baseline_value = baseline.get(key) if baseline else None
            current_value = current.get(key) if current else None

            if baseline_value is not None:
                metrics[baseline_key] = baseline_value
            if current_value is not None:
                metrics[current_key] = current_value

            if baseline_value is not None and current_value is not None:
                metrics[f"{key}_degradation"] = baseline_value - current_value

        return metrics
    
    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        if not PROMETHEUS_AVAILABLE:
            return b"Prometheus client not installed"
        
        return generate_latest(self.registry)
    
    def create_monitoring_report(self, 
                                output_path: str = None) -> Dict:
        """
        Create comprehensive monitoring report with billing info
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Report dictionary
        """
        report = {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "tenant_id": self.tenant_id,
            "report_timestamp": datetime.now().isoformat(),
            "performance_summary": self.get_performance_summary(),
            "drift_analysis": {
                "total_drift_checks": len(self.drift_detector.drift_history),
                "drift_detected_count": sum(1 for d in self.drift_detector.drift_history if d["drift_detected"]),
                "last_drift_check": self.drift_detector.drift_history[-1] if self.drift_detector.drift_history else None
            },
            "prediction_statistics": {
                "total_predictions": len(self.prediction_history),
                "avg_prediction_time": np.mean([p["prediction_time"] for p in self.prediction_history if p.get("prediction_time")])
                    if self.prediction_history else None
            },
            "billing_summary": {
                "total_predictions": self.total_predictions,
                "total_api_calls": self.total_api_calls,
                "total_compute_time_seconds": self.total_compute_time,
                "estimated_cost": self._estimate_cost()
            }
        }
        
        # Add detailed performance history if available
        if self.performance_history:
            recent_performance = self.performance_history[-10:]  # Last 10 measurements
            report["recent_performance"] = [m.to_dict() for m in recent_performance]
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Monitoring report saved to {output_path}")
        
        return report
    
    def _estimate_cost(self) -> float:
        """Estimate cost based on usage"""
        # Simple cost model (customize based on your pricing)
        cost_per_1k_predictions = 0.01
        cost_per_compute_hour = 0.10
        cost_per_1k_api_calls = 0.001
        
        total_cost = (
            (self.total_predictions / 1000) * cost_per_1k_predictions +
            (self.total_compute_time / 3600) * cost_per_compute_hour +
            (self.total_api_calls / 1000) * cost_per_1k_api_calls
        )
        
        return round(total_cost, 4)


class DataQualityMonitor:
    """Monitor data quality in production with advanced checks"""
    
    def __init__(self, expected_schema: Dict = None):
        """
        Initialize data quality monitor
        
        Args:
            expected_schema: Expected data schema
        """
        self.expected_schema = expected_schema
        self.quality_checks_history = []
    
    def check_data_quality(self, data: pd.DataFrame, tenant_id: str = "default") -> Dict:
        """
        Perform comprehensive data quality checks
        
        Args:
            data: Input data to check
            tenant_id: Tenant identifier
            
        Returns:
            Data quality report
        """
        import psutil
        memory_before = psutil.Process().memory_info().rss / 1024**2  # MB
        
        quality_report = {
            "timestamp": datetime.now().isoformat(),
            "tenant_id": tenant_id,
            "rows": len(data),
            "columns": len(data.columns),
            "issues": [],
            "warnings": [],
            "quality_score": 100.0,
            "memory_usage_mb": memory_before
        }
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.any():
            missing_features = missing_counts[missing_counts > 0].to_dict()
            quality_report["issues"].append({
                "type": "missing_values",
                "details": missing_features,
                "severity": "medium"
            })
            quality_report["quality_score"] -= 10
        
        # Check for duplicates
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            quality_report["issues"].append({
                "type": "duplicate_rows",
                "count": int(duplicate_count),
                "severity": "low"
            })
            quality_report["quality_score"] -= 5
        
        # Check for outliers (numerical columns)
        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)).sum()
            
            if outliers > len(data) * 0.01:  # More than 1% outliers
                quality_report["warnings"].append({
                    "type": "outliers",
                    "column": col,
                    "count": int(outliers),
                    "percentage": float(outliers / len(data) * 100)
                })
        
        # Check data types if schema is provided
        if self.expected_schema:
            for col, expected_type in self.expected_schema.items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if not self._compatible_types(actual_type, expected_type):
                        quality_report["issues"].append({
                            "type": "schema_mismatch",
                            "column": col,
                            "expected": expected_type,
                            "actual": actual_type,
                            "severity": "high"
                        })
                        quality_report["quality_score"] -= 15
                else:
                    quality_report["issues"].append({
                        "type": "missing_column",
                        "column": col,
                        "severity": "high"
                    })
                    quality_report["quality_score"] -= 20
        
        # Check for constant columns
        for col in data.columns:
            if data[col].nunique() == 1:
                quality_report["warnings"].append({
                    "type": "constant_column",
                    "column": col,
                    "value": str(data[col].iloc[0])
                })
        
        # Check for high cardinality in categorical columns
        for col in data.select_dtypes(include=['object']).columns:
            cardinality = data[col].nunique()
            if cardinality > len(data) * 0.5:
                quality_report["warnings"].append({
                    "type": "high_cardinality",
                    "column": col,
                    "unique_values": cardinality,
                    "ratio": float(cardinality / len(data))
                })
        
        # Check for data leakage indicators
        for col in data.columns:
            # Check if column is perfectly correlated with index
            if pd.api.types.is_numeric_dtype(data[col]):
                correlation_with_index = data[col].corr(pd.Series(range(len(data))))
                if abs(correlation_with_index) > 0.99:
                    quality_report["warnings"].append({
                        "type": "potential_data_leakage",
                        "column": col,
                        "correlation_with_index": float(correlation_with_index),
                        "severity": "high"
                    })
        
        # Use Evidently for advanced quality checks if available
        if EVIDENTLY_AVAILABLE:
            try:
                quality_preset = Report(metrics=[DataQualityPreset()])
                quality_preset.run(current_data=data, reference_data=None)
                quality_report["evidently_report"] = quality_preset.as_dict()
            except Exception as e:
                logger.error(f"Evidently quality check failed: {e}")
        
        # Calculate memory after processing
        memory_after = psutil.Process().memory_info().rss / 1024**2
        quality_report["memory_usage_mb"] = memory_after - memory_before
        
        # Ensure quality score doesn't go below 0
        quality_report["quality_score"] = max(0, quality_report["quality_score"])
        
        # Store in history
        self.quality_checks_history.append(quality_report)
        
        return quality_report
    
    def _compatible_types(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible"""
        compatible_pairs = [
            ("int64", "float64"),
            ("int32", "int64"),
            ("float32", "float64"),
            ("object", "string")
        ]
        
        if actual == expected:
            return True
        
        for pair in compatible_pairs:
            if (actual in pair and expected in pair):
                return True
        
        return False
    
    def get_quality_trend(self, last_n_checks: int = 10) -> Dict:
        """Get data quality trend over recent checks"""
        if not self.quality_checks_history:
            return {"message": "No quality check history available"}
        
        recent_checks = self.quality_checks_history[-last_n_checks:]
        
        trend = {
            "checks_analyzed": len(recent_checks),
            "avg_quality_score": np.mean([c["quality_score"] for c in recent_checks]),
            "min_quality_score": np.min([c["quality_score"] for c in recent_checks]),
            "max_quality_score": np.max([c["quality_score"] for c in recent_checks]),
            "avg_memory_usage_mb": np.mean([c.get("memory_usage_mb", 0) for c in recent_checks]),
            "common_issues": defaultdict(int)
        }
        
        # Count common issues
        for check in recent_checks:
            for issue in check.get("issues", []):
                trend["common_issues"][issue["type"]] += 1
        
        trend["common_issues"] = dict(trend["common_issues"])
        
        return trend


# Alert Manager for automated alerting
class AlertManager:
    """Manage alerts based on monitoring metrics with multi-channel support"""
    
    def __init__(self, 
                 alert_config: Dict = None,
                 notification_handlers: Dict = None):
        """
        Initialize alert manager
        
        Args:
            alert_config: Alert configuration with thresholds
            notification_handlers: Dictionary of notification handlers
        """
        self.alert_config = alert_config or self._default_config()
        self.notification_handlers = notification_handlers or {}
        self.active_alerts = []
        self.alert_history = []
        
        # Set default handlers if not provided
        if 'log' not in self.notification_handlers:
            self.notification_handlers['log'] = self._log_notification
        if 'slack' not in self.notification_handlers:
            self.notification_handlers['slack'] = self._slack_notification
        if 'email' not in self.notification_handlers:
            self.notification_handlers['email'] = self._email_notification
    
    def _default_config(self) -> Dict:
        """Default alert configuration"""
        return {
            "accuracy_threshold": 0.8,
            "drift_threshold": 0.5,
            "error_rate_threshold": 0.05,
            "latency_threshold": 1.0,  # seconds
            "quality_score_threshold": 70,
            "billing_threshold": 1000.0,  # dollars
            "notification_channels": ["log", "slack", "email"]
        }
    
    def _log_notification(self, alert: Dict):
        """Log notification handler"""
        logger.warning(f"ALERT: {alert}")
    
    def _slack_notification(self, alert: Dict):
        """Slack notification handler"""
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if webhook_url:
            from .monitoring import MonitoringIntegration
            MonitoringIntegration.send_to_slack(alert, webhook_url)
    
    def _email_notification(self, alert: Dict):
        """Email notification handler"""
        recipients = os.getenv('ALERT_EMAIL_RECIPIENTS')
        if recipients:
            from .monitoring import MonitoringIntegration
            smtp_config = {
                'host': os.getenv('SMTP_HOST', 'localhost'),
                'port': int(os.getenv('SMTP_PORT', 587)),
                'from_email': os.getenv('SMTP_FROM_EMAIL', 'alerts@automl.com'),
                'username': os.getenv('SMTP_USERNAME'),
                'password': os.getenv('SMTP_PASSWORD')
            }
            MonitoringIntegration.send_to_email(
                alert, smtp_config, recipients.split(',')
            )
    
    def check_alerts(self, metrics: Dict) -> List[Dict]:
        """
        Check if any alerts should be triggered
        
        Args:
            metrics: Current metrics dictionary
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        # Check accuracy alert
        if "accuracy" in metrics and metrics["accuracy"] < self.alert_config["accuracy_threshold"]:
            alert = {
                "type": "low_accuracy",
                "severity": "high",
                "message": f"Model accuracy ({metrics['accuracy']:.3f}) below threshold ({self.alert_config['accuracy_threshold']})",
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
            triggered_alerts.append(alert)
        
        # Check drift alert
        if "drift_score" in metrics and metrics["drift_score"] > self.alert_config["drift_threshold"]:
            alert = {
                "type": "data_drift",
                "severity": "medium",
                "message": f"Data drift detected (score: {metrics['drift_score']:.3f})",
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
            triggered_alerts.append(alert)
        
        # Check latency alert
        if "latency" in metrics and metrics["latency"] > self.alert_config["latency_threshold"]:
            alert = {
                "type": "high_latency",
                "severity": "medium",
                "message": f"High prediction latency ({metrics['latency']:.2f}s)",
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
            triggered_alerts.append(alert)
        
        # Check data quality alert
        if "quality_score" in metrics and metrics["quality_score"] < self.alert_config["quality_score_threshold"]:
            alert = {
                "type": "poor_data_quality",
                "severity": "high",
                "message": f"Data quality score ({metrics['quality_score']:.1f}) below threshold ({self.alert_config['quality_score_threshold']})",
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
            triggered_alerts.append(alert)
        
        # Check billing alert
        if "billing_amount" in metrics and metrics["billing_amount"] > self.alert_config["billing_threshold"]:
            alert = {
                "type": "high_billing",
                "severity": "high",
                "message": f"Billing amount (${metrics['billing_amount']:.2f}) exceeds threshold (${self.alert_config['billing_threshold']:.2f})",
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
            triggered_alerts.append(alert)
        
        # Process triggered alerts
        for alert in triggered_alerts:
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            # Send to configured channels
            for channel in self.alert_config.get("notification_channels", ["log"]):
                if channel in self.notification_handlers:
                    self.notification_handlers[channel](alert)
        
        return triggered_alerts
    
    def resolve_alert(self, alert_type: str):
        """Resolve an active alert"""
        self.active_alerts = [a for a in self.active_alerts if a["type"] != alert_type]
        logger.info(f"Alert resolved: {alert_type}")
    
    def get_active_alerts(self) -> List[Dict]:
        """Get list of active alerts"""
        return self.active_alerts
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts"""
        summary = {
            "active_alerts_count": len(self.active_alerts),
            "total_alerts_triggered": len(self.alert_history),
            "alerts_by_type": defaultdict(int),
            "alerts_by_severity": defaultdict(int)
        }
        
        for alert in self.alert_history:
            summary["alerts_by_type"][alert["type"]] += 1
            summary["alerts_by_severity"][alert["severity"]] += 1
        
        summary["alerts_by_type"] = dict(summary["alerts_by_type"])
        summary["alerts_by_severity"] = dict(summary["alerts_by_severity"])
        
        return summary


# Integration with external monitoring systems
class MonitoringIntegration:
    """Integration with external monitoring systems"""
    
    @staticmethod
    def export_to_prometheus(monitor: ModelMonitor) -> bytes:
        """Export metrics to Prometheus format"""
        return monitor.export_metrics()
    
    @staticmethod
    def export_to_grafana_json(monitor: ModelMonitor) -> Dict:
        """Export metrics in Grafana-compatible JSON format"""
        metrics = monitor.get_performance_summary()
        
        grafana_data = {
            "dashboardId": monitor.model_id,
            "title": f"Model {monitor.model_id} Dashboard",
            "tenant": monitor.tenant_id,
            "panels": [
                {
                    "id": 1,
                    "type": "graph",
                    "title": "Model Accuracy",
                    "targets": [
                        {
                            "target": "accuracy",
                            "datapoints": [
                                [m.accuracy, datetime.fromisoformat(m.timestamp).timestamp() * 1000]
                                for m in monitor.performance_history
                                if m.accuracy is not None
                            ]
                        }
                    ]
                },
                {
                    "id": 2,
                    "type": "stat",
                    "title": "Total Predictions",
                    "value": metrics.get("total_predictions", 0)
                },
                {
                    "id": 3,
                    "type": "gauge",
                    "title": "Data Quality Score",
                    "value": metrics.get("metrics", {}).get("quality_score", 100)
                },
                {
                    "id": 4,
                    "type": "stat",
                    "title": "Total API Calls",
                    "value": metrics.get("billing", {}).get("total_api_calls", 0)
                },
                {
                    "id": 5,
                    "type": "stat",
                    "title": "Compute Time (hours)",
                    "value": round(metrics.get("billing", {}).get("total_compute_seconds", 0) / 3600, 2)
                }
            ]
        }
        
        return grafana_data
    
    @staticmethod
    def send_to_slack(alert: Dict, webhook_url: str):
        """Send alert to Slack with proper formatting"""
        import requests
        
        # Color mapping based on severity
        color_map = {
            "high": "danger",
            "medium": "warning",
            "low": "good"
        }
        
        slack_message = {
            "text": f"🚨 ML Model Alert: {alert['type'].replace('_', ' ').title()}",
            "attachments": [
                {
                    "color": color_map.get(alert.get('severity', 'medium'), "warning"),
                    "fields": [
                        {"title": "Alert Type", "value": alert['type'], "short": True},
                        {"title": "Severity", "value": alert.get('severity', 'unknown'), "short": True},
                        {"title": "Timestamp", "value": alert.get('timestamp', 'N/A'), "short": True},
                        {"title": "Model ID", "value": alert.get('model_id', 'N/A'), "short": True},
                        {"title": "Tenant ID", "value": alert.get('tenant_id', 'N/A'), "short": True},
                        {"title": "Message", "value": alert.get('message', 'No message'), "short": False}
                    ],
                    "footer": "AutoML Platform",
                    "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png"
                }
            ]
        }
        
        # Add metrics if available
        if 'metrics' in alert and alert['metrics']:
            metrics_text = "\n".join([f"• {k}: {v}" for k, v in list(alert['metrics'].items())[:5]])
            slack_message["attachments"][0]["fields"].append({
                "title": "Metrics",
                "value": metrics_text,
                "short": False
            })
        
        try:
            response = requests.post(webhook_url, json=slack_message, timeout=5)
            response.raise_for_status()
            logger.info(f"Alert sent to Slack successfully: {alert['type']}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send alert to Slack: {e}")
            return False
    
    @staticmethod
    def send_to_email(alert: Dict, smtp_config: Dict, recipients: List[str]):
        """Send alert via email with HTML formatting"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['From'] = smtp_config.get('from_email', 'automl@platform.com')
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"[{alert.get('severity', 'INFO').upper()}] ML Model Alert: {alert['type']}"
        
        # Plain text version
        text_body = f"""
ML Model Alert
==============

Alert Type: {alert['type']}
Severity: {alert.get('severity', 'unknown')}
Message: {alert.get('message', 'No message')}
Timestamp: {alert.get('timestamp', 'N/A')}
Model ID: {alert.get('model_id', 'N/A')}
Tenant ID: {alert.get('tenant_id', 'N/A')}

Metrics:
{json.dumps(alert.get('metrics', {}), indent=2)}

---
This is an automated alert from the AutoML Platform.
Please do not reply to this email.
"""
        
        # HTML version
        severity_colors = {
            "high": "#dc3545",
            "medium": "#ffc107",
            "low": "#28a745"
        }
        
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        .alert-container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .alert-header {{ 
            background: {severity_colors.get(alert.get('severity', 'medium'), '#17a2b8')};
            color: white;
            padding: 15px;
            border-radius: 5px 5px 0 0;
        }}
        .alert-body {{ 
            background: #f8f9fa;
            padding: 20px;
            border: 1px solid #dee2e6;
            border-radius: 0 0 5px 5px;
        }}
        .metric-item {{ 
            background: white;
            padding: 10px;
            margin: 5px 0;
            border-left: 3px solid {severity_colors.get(alert.get('severity', 'medium'), '#17a2b8')};
        }}
        .footer {{ 
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            color: #6c757d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="alert-container">
        <div class="alert-header">
            <h2>🚨 ML Model Alert</h2>
            <h3>{alert['type'].replace('_', ' ').title()}</h3>
        </div>
        <div class="alert-body">
            <p><strong>Severity:</strong> {alert.get('severity', 'unknown').upper()}</p>
            <p><strong>Message:</strong> {alert.get('message', 'No message')}</p>
            <p><strong>Timestamp:</strong> {alert.get('timestamp', 'N/A')}</p>
            <p><strong>Model ID:</strong> {alert.get('model_id', 'N/A')}</p>
            <p><strong>Tenant ID:</strong> {alert.get('tenant_id', 'N/A')}</p>
            
            <h4>Metrics:</h4>
            <div>
                {MonitoringIntegration._format_metrics_html(alert.get('metrics', {}))}
            </div>
            
            <div class="footer">
                <p>This is an automated alert from the AutoML Platform.</p>
                <p>Please do not reply to this email.</p>
                <p>To update alert settings, visit your dashboard.</p>
            </div>
        </div>
    </div>
</body>
</html>
"""
        
        # Attach parts
        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
        try:
            # Connect to server
            if smtp_config.get('use_tls', True):
                server = smtplib.SMTP(smtp_config['host'], smtp_config.get('port', 587))
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(smtp_config['host'], smtp_config.get('port', 465))
            
            # Login if credentials provided
            if smtp_config.get('username') and smtp_config.get('password'):
                server.login(smtp_config['username'], smtp_config['password'])
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Alert sent via email successfully to {recipients}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    @staticmethod
    def _format_metrics_html(metrics: Dict) -> str:
        """Format metrics as HTML"""
        if not metrics:
            return "<p>No metrics available</p>"
        
        html = ""
        for key, value in list(metrics.items())[:10]:  # Limit to 10 metrics
            if isinstance(value, float):
                value = f"{value:.4f}"
            elif isinstance(value, dict):
                value = json.dumps(value, indent=2)[:200] + "..."
            html += f'<div class="metric-item"><strong>{key}:</strong> {value}</div>'
        
        return html
    
    @staticmethod
    def send_to_webhook(alert: Dict, webhook_url: str, headers: Dict = None):
        """Send alert to generic webhook"""
        import requests
        
        payload = {
            "alert": alert,
            "platform": "AutoML",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers or {},
                timeout=5
            )
            response.raise_for_status()
            logger.info(f"Alert sent to webhook successfully: {webhook_url}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send alert to webhook: {e}")
            return False


# Main monitoring service
class MonitoringService:
    """Central monitoring service for all models with billing integration"""
    
    def __init__(self, storage_manager = None, billing_tracker = None):
        """
        Initialize monitoring service
        
        Args:
            storage_manager: Storage manager for persisting monitoring data
            billing_tracker: Billing tracker for cost monitoring
        """
        self.monitors = {}
        self.quality_monitor = DataQualityMonitor()
        self.alert_manager = AlertManager()
        self.storage_manager = storage_manager
        self.billing_tracker = billing_tracker
    
    def register_model(self, 
                       model_id: str,
                       model_type: str,
                       reference_data: pd.DataFrame = None,
                       tenant_id: str = "default") -> ModelMonitor:
        """
        Register a model for monitoring
        
        Args:
            model_id: Model identifier
            model_type: Type of model
            reference_data: Reference dataset
            tenant_id: Tenant identifier
            
        Returns:
            ModelMonitor instance
        """
        monitor = ModelMonitor(
            model_id, model_type, reference_data,
            tenant_id=tenant_id,
            billing_tracker=self.billing_tracker
        )
        self.monitors[model_id] = monitor
        logger.info(f"Model {model_id} registered for monitoring")
        return monitor
    
    def get_monitor(self, model_id: str) -> ModelMonitor:
        """Get monitor for a specific model"""
        return self.monitors.get(model_id)
    
    def log_prediction(self,
                      model_id: str,
                      features: pd.DataFrame,
                      predictions: np.ndarray,
                      actuals: np.ndarray = None,
                      prediction_time: float = None):
        """Log prediction for a model"""
        monitor = self.monitors.get(model_id)
        if monitor:
            monitor.log_prediction(features, predictions, actuals, prediction_time)
        else:
            logger.warning(f"Model {model_id} not registered for monitoring")
    
    def check_all_models_health(self) -> Dict:
        """Check health status of all monitored models"""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "billing_summary": {}
        }
        
        total_cost = 0.0
        
        for model_id, monitor in self.monitors.items():
            summary = monitor.get_performance_summary()
            
            # Determine health status
            if not summary.get("metrics"):
                status = "no_data"
            elif summary.get("metrics", {}).get("avg_accuracy", 1.0) < 0.8:
                status = "degraded"
            elif summary.get("drift_rate", 0) > 0.2:
                status = "drift_detected"
            else:
                status = "healthy"
            
            health_report["models"][model_id] = {
                "status": status,
                "summary": summary,
                "tenant_id": monitor.tenant_id
            }
            
            # Aggregate billing
            model_cost = monitor._estimate_cost()
            total_cost += model_cost
        
        health_report["billing_summary"] = {
            "total_estimated_cost": total_cost,
            "models_count": len(self.monitors),
            "timestamp": datetime.now().isoformat()
        }
        
        return health_report
    
    def create_global_dashboard(self) -> Dict:
        """Create dashboard with all models' metrics"""
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(self.monitors),
            "models": [],
            "alerts": self.alert_manager.get_alert_summary(),
            "data_quality_trend": self.quality_monitor.get_quality_trend(),
            "billing": {
                "total_predictions": 0,
                "total_api_calls": 0,
                "total_compute_hours": 0,
                "estimated_total_cost": 0
            }
        }
        
        for model_id, monitor in self.monitors.items():
            model_info = {
                "model_id": model_id,
                "model_type": monitor.model_type,
                "tenant_id": monitor.tenant_id,
                "performance": monitor.get_performance_summary(last_n_days=1),
                "total_predictions": len(monitor.prediction_history),
                "last_drift_check": monitor.drift_detector.drift_history[-1] if monitor.drift_detector.drift_history else None,
                "estimated_cost": monitor._estimate_cost()
            }
            dashboard["models"].append(model_info)
            
            # Aggregate billing
            dashboard["billing"]["total_predictions"] += monitor.total_predictions
            dashboard["billing"]["total_api_calls"] += monitor.total_api_calls
            dashboard["billing"]["total_compute_hours"] += monitor.total_compute_time / 3600
            dashboard["billing"]["estimated_total_cost"] += model_info["estimated_cost"]
        
        return dashboard
    
    def save_monitoring_data(self):
        """Save monitoring data to storage"""
        if not self.storage_manager:
            logger.warning("No storage manager configured")
            return
        
        for model_id, monitor in self.monitors.items():
            # Save performance history
            if monitor.performance_history:
                data = pd.DataFrame([m.to_dict() for m in monitor.performance_history])
                self.storage_manager.save_dataset(
                    data,
                    f"monitoring_{model_id}_performance",
                    tenant_id=monitor.tenant_id
                )
            
            # Save drift history
            if monitor.drift_detector.drift_history:
                data = pd.DataFrame(monitor.drift_detector.drift_history)
                self.storage_manager.save_dataset(
                    data,
                    f"monitoring_{model_id}_drift",
                    tenant_id=monitor.tenant_id
                )
        
        logger.info("Monitoring data saved to storage")
