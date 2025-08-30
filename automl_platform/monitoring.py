"""
Monitoring module for ML Platform
Handles model performance tracking, data drift detection, and system metrics
Integrates with Prometheus for metrics export
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
    """Container for model performance metrics"""
    model_id: str
    timestamp: str
    prediction_count: int
    
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
    
    def to_dict(self):
        return asdict(self)


class DriftDetector:
    """Detects data and concept drift in production"""
    
    def __init__(self, 
                 reference_data: pd.DataFrame = None,
                 sensitivity: float = 0.05):
        """
        Initialize drift detector
        
        Args:
            reference_data: Reference dataset for comparison
            sensitivity: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.sensitivity = sensitivity
        self.drift_history = []
        
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
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Detect data drift using statistical tests
        
        Args:
            current_data: Current production data
            
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
            "timestamp": datetime.now().isoformat()
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
        
        return {
            "concept_drift_detected": drift_detected,
            "drift_points": drift_points,
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "timestamp": datetime.now().isoformat()
        }
    
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
    """Main monitoring class for ML models in production"""
    
    def __init__(self, 
                 model_id: str,
                 model_type: str = "classification",
                 reference_data: pd.DataFrame = None,
                 prometheus_port: int = 8000):
        """
        Initialize model monitor
        
        Args:
            model_id: Unique model identifier
            model_type: Type of model ('classification' or 'regression')
            reference_data: Reference dataset for drift detection
            prometheus_port: Port for Prometheus metrics endpoint
        """
        self.model_id = model_id
        self.model_type = model_type
        self.reference_data = reference_data
        self.prometheus_port = prometheus_port
        
        # Initialize drift detector
        self.drift_detector = DriftDetector(reference_data)
        
        # Performance history
        self.performance_history = []
        self.prediction_history = []
        
        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.registry = CollectorRegistry()
        
        # Counters
        self.prediction_counter = Counter(
            'ml_predictions_total',
            'Total number of predictions',
            ['model_id'],
            registry=self.registry
        )
        
        self.error_counter = Counter(
            'ml_prediction_errors_total',
            'Total number of prediction errors',
            ['model_id', 'error_type'],
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
            ['model_id'],
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
    
    def log_prediction(self,
                      features: pd.DataFrame,
                      predictions: np.ndarray,
                      actuals: np.ndarray = None,
                      prediction_time: float = None):
        """
        Log predictions for monitoring
        
        Args:
            features: Input features
            predictions: Model predictions
            actuals: Actual values (if available)
            prediction_time: Time taken for prediction
        """
        # Store prediction
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'features': features.to_dict('records') if len(features) < 100 else None,
            'predictions': predictions.tolist() if len(predictions) < 100 else predictions.mean(),
            'actuals': actuals.tolist() if actuals is not None and len(actuals) < 100 else None,
            'prediction_time': prediction_time
        }
        
        self.prediction_history.append(prediction_record)
        
        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self.prediction_counter.labels(model_id=self.model_id).inc(len(predictions))
            
            if prediction_time:
                self.prediction_latency.labels(model_id=self.model_id).observe(prediction_time)
            
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
        Calculate model performance metrics
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            sample_weight: Sample weights
            
        Returns:
            ModelPerformanceMetrics object
        """
        metrics = ModelPerformanceMetrics(
            model_id=self.model_id,
            timestamp=datetime.now().isoformat(),
            prediction_count=len(predictions)
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
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE and metrics.accuracy is not None:
                self.accuracy_gauge.labels(model_id=self.model_id).set(metrics.accuracy)
            
            # Store in history
            self.performance_history.append(metrics)
            
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
        drift_results = self.drift_detector.detect_data_drift(current_data)
        
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
        Get performance summary for the last N days
        
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
            "period": f"Last {last_n_days} days",
            "total_predictions": sum(m.prediction_count for m in recent_metrics),
            "metrics": {}
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
    
    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        if not PROMETHEUS_AVAILABLE:
            return b"Prometheus client not installed"
        
        return generate_latest(self.registry)
    
    def create_monitoring_report(self, 
                                output_path: str = None) -> Dict:
        """
        Create comprehensive monitoring report
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Report dictionary
        """
        report = {
            "model_id": self.model_id,
            "model_type": self.model_type,
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


class DataQualityMonitor:
    """Monitor data quality in production"""
    
    def __init__(self, expected_schema: Dict = None):
        """
        Initialize data quality monitor
        
        Args:
            expected_schema: Expected data schema
        """
        self.expected_schema = expected_schema
        self.quality_checks_history = []
    
    def check_data_quality(self, data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data quality checks
        
        Args:
            data: Input data to check
            
        Returns:
            Data quality report
        """
        quality_report = {
            "timestamp": datetime.now().isoformat(),
            "rows": len(data),
            "columns": len(data.columns),
            "issues": [],
            "warnings": [],
            "quality_score": 100.0
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
        
        # Use Evidently for advanced quality checks if available
        if EVIDENTLY_AVAILABLE:
            try:
                quality_preset = Report(metrics=[DataQualityPreset()])
                quality_preset.run(current_data=data, reference_data=None)
                quality_report["evidently_report"] = quality_preset.as_dict()
            except Exception as e:
                logger.error(f"Evidently quality check failed: {e}")
        
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
    """Manage alerts based on monitoring metrics"""
    
    def __init__(self, 
                 alert_config: Dict = None,
                 notification_handler = None):
        """
        Initialize alert manager
        
        Args:
            alert_config: Alert configuration with thresholds
            notification_handler: Function to handle notifications
        """
        self.alert_config = alert_config or self._default_config()
        self.notification_handler = notification_handler or self._default_notification
        self.active_alerts = []
        self.alert_history = []
    
    def _default_config(self) -> Dict:
        """Default alert configuration"""
        return {
            "accuracy_threshold": 0.8,
            "drift_threshold": 0.5,
            "error_rate_threshold": 0.05,
            "latency_threshold": 1.0,  # seconds
            "quality_score_threshold": 70
        }
    
    def _default_notification(self, alert: Dict):
        """Default notification handler (just logs)"""
        logger.warning(f"ALERT: {alert}")
    
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
                "type": "poor_data_
