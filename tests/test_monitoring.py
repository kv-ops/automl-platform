"""
Test Suite for Monitoring Module
=================================
Tests for model performance tracking, drift detection, alerts, and metrics export.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
import json
from collections import defaultdict
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.monitoring import (
    ModelPerformanceMetrics,
    DriftDetector,
    ModelMonitor,
    DataQualityMonitor,
    AlertManager,
    MonitoringIntegration,
    MonitoringService
)


class TestModelPerformanceMetrics(unittest.TestCase):
    """Test model performance metrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = ModelPerformanceMetrics(
            model_id="model_123",
            timestamp="2024-01-01T12:00:00",
            prediction_count=1000,
            tenant_id="tenant_123",
            accuracy=0.95,
            precision=0.92,
            recall=0.94,
            f1=0.93
        )
        
        self.assertEqual(metrics.model_id, "model_123")
        self.assertEqual(metrics.prediction_count, 1000)
        self.assertEqual(metrics.accuracy, 0.95)
        self.assertEqual(metrics.tenant_id, "tenant_123")
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ModelPerformanceMetrics(
            model_id="model_123",
            timestamp="2024-01-01T12:00:00",
            prediction_count=500,
            mse=0.05,
            mae=0.03
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict["model_id"], "model_123")
        self.assertEqual(metrics_dict["mse"], 0.05)
        self.assertEqual(metrics_dict["mae"], 0.03)


class TestDriftDetector(unittest.TestCase):
    """Test drift detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create reference data
        np.random.seed(42)
        self.reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
            'category': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        self.drift_detector = DriftDetector(
            reference_data=self.reference_data,
            sensitivity=0.05
        )
    
    def test_initialization(self):
        """Test drift detector initialization."""
        self.assertIsNotNone(self.drift_detector.reference_data)
        self.assertEqual(self.drift_detector.sensitivity, 0.05)
        self.assertIsNotNone(self.drift_detector.reference_stats)
    
    def test_calculate_reference_stats(self):
        """Test reference statistics calculation."""
        # Check numeric features
        self.assertIn('feature1', self.drift_detector.reference_stats)
        self.assertIn('mean', self.drift_detector.reference_stats['feature1'])
        self.assertIn('std', self.drift_detector.reference_stats['feature1'])
        
        # Check categorical features
        self.assertIn('category', self.drift_detector.reference_stats)
        self.assertIn('unique_values', self.drift_detector.reference_stats['category'])
        self.assertIn('value_counts', self.drift_detector.reference_stats['category'])
    
    def test_detect_data_drift_no_drift(self):
        """Test drift detection when no drift present."""
        # Create similar distribution
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        result = self.drift_detector.detect_data_drift(current_data)
        
        self.assertFalse(result["drift_detected"])
        self.assertEqual(len(result["drifted_features"]), 0)
        self.assertIn("drift_scores", result)
    
    def test_detect_data_drift_with_drift(self):
        """Test drift detection when drift is present."""
        # Create shifted distribution
        current_data = pd.DataFrame({
            'feature1': np.random.normal(3, 1, 100),  # Shifted mean
            'feature2': np.random.normal(10, 2, 100),  # Shifted mean
            'category': np.random.choice(['A', 'D'], 100)  # Different categories
        })
        
        result = self.drift_detector.detect_data_drift(current_data)
        
        self.assertTrue(result["drift_detected"])
        self.assertGreater(len(result["drifted_features"]), 0)
        self.assertIn("feature1", result["drifted_features"])
    
    @patch('automl_platform.monitoring.EVIDENTLY_AVAILABLE', True)
    @patch('automl_platform.monitoring.Report')
    def test_detect_drift_with_evidently(self, mock_report):
        """Test drift detection using Evidently."""
        # Mock Evidently report
        mock_report_instance = Mock()
        mock_report_instance.as_dict.return_value = {
            'metrics': [{'result': {'dataset_drift': True}}]
        }
        mock_report.return_value = mock_report_instance
        
        current_data = self.reference_data.copy()
        result = self.drift_detector.detect_data_drift(current_data)
        
        self.assertTrue(result["drift_detected"])
        self.assertIn("evidently_report", result)
    
    def test_detect_concept_drift(self):
        """Test concept drift detection."""
        # Create predictions and actuals
        predictions = np.array([0.5] * 100 + [0.8] * 100)
        actuals = np.array([0.6] * 100 + [0.4] * 100)  # Increasing error
        
        result = self.drift_detector.detect_concept_drift(
            predictions, actuals, window_size=50
        )
        
        self.assertIn("concept_drift_detected", result)
        self.assertIn("mean_error", result)
        self.assertIn("std_error", result)
    
    def test_calculate_psi(self):
        """Test Population Stability Index calculation."""
        # Create distributions
        expected = pd.Series(np.random.normal(0, 1, 1000))
        actual = pd.Series(np.random.normal(0.5, 1, 1000))  # Shifted
        
        psi = self.drift_detector.calculate_psi(expected, actual, bins=10)
        
        self.assertIsInstance(psi, float)
        self.assertGreater(psi, 0)  # Should detect some shift
    
    def test_drift_alert_callback(self):
        """Test drift alert callback triggering."""
        alert_callback = Mock()
        detector = DriftDetector(
            reference_data=self.reference_data,
            alert_callback=alert_callback
        )
        
        # Create drifted data
        current_data = pd.DataFrame({
            'feature1': np.random.normal(5, 1, 100),  # Large drift
            'feature2': np.random.normal(5, 2, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        result = detector.detect_data_drift(current_data)
        
        if result["drift_detected"]:
            alert_callback.assert_called_once()


class TestModelMonitor(unittest.TestCase):
    """Test model monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create reference data
        np.random.seed(42)
        self.reference_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        
        self.billing_tracker = Mock()
        
        self.monitor = ModelMonitor(
            model_id="model_123",
            model_type="classification",
            reference_data=self.reference_data,
            tenant_id="tenant_123",
            billing_tracker=self.billing_tracker
        )
    
    def test_initialization(self):
        """Test model monitor initialization."""
        self.assertEqual(self.monitor.model_id, "model_123")
        self.assertEqual(self.monitor.model_type, "classification")
        self.assertEqual(self.monitor.tenant_id, "tenant_123")
        self.assertIsNotNone(self.monitor.drift_detector)
    
    @patch('automl_platform.monitoring.PROMETHEUS_AVAILABLE', True)
    def test_prometheus_metrics_initialization(self):
        """Test Prometheus metrics initialization."""
        monitor = ModelMonitor(
            model_id="model_123",
            model_type="classification"
        )
        
        self.assertIsNotNone(monitor.registry)
        self.assertIsNotNone(monitor.prediction_counter)
        self.assertIsNotNone(monitor.accuracy_gauge)
    
    def test_log_prediction(self):
        """Test logging predictions."""
        features = pd.DataFrame({
            'feature1': [1.0, 2.0],
            'feature2': [3.0, 4.0]
        })
        predictions = np.array([0, 1])
        actuals = np.array([0, 1])
        
        self.monitor.log_prediction(
            features=features,
            predictions=predictions,
            actuals=actuals,
            prediction_time=0.5
        )
        
        self.assertEqual(self.monitor.total_predictions, 2)
        self.assertEqual(self.monitor.total_api_calls, 1)
        self.assertEqual(self.monitor.total_compute_time, 0.5)
        
        # Check billing tracking
        self.billing_tracker.track_predictions.assert_called_once_with("tenant_123", 2)
        self.billing_tracker.track_api_call.assert_called_once_with("tenant_123", "prediction")
    
    def test_calculate_performance_classification(self):
        """Test performance calculation for classification."""
        predictions = np.array([0, 1, 0, 1, 1, 0])
        actuals = np.array([0, 1, 0, 0, 1, 0])
        
        metrics = self.monitor.calculate_performance(predictions, actuals)
        
        self.assertIsNotNone(metrics.accuracy)
        self.assertIsNotNone(metrics.precision)
        self.assertIsNotNone(metrics.recall)
        self.assertIsNotNone(metrics.f1)
        self.assertIsNotNone(metrics.confusion_matrix_data)
    
    def test_calculate_performance_regression(self):
        """Test performance calculation for regression."""
        monitor = ModelMonitor(
            model_id="model_456",
            model_type="regression",
            billing_tracker=self.billing_tracker
        )
        
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        actuals = np.array([1.1, 1.9, 3.2, 3.8])
        
        metrics = monitor.calculate_performance(predictions, actuals)
        
        self.assertIsNotNone(metrics.mse)
        self.assertIsNotNone(metrics.mae)
        self.assertIsNotNone(metrics.rmse)
        self.assertIsNotNone(metrics.r2)
    
    def test_check_drift(self):
        """Test drift checking."""
        current_data = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50)
        })
        
        result = self.monitor.check_drift(current_data)
        
        self.assertIn("drift_detected", result)
        self.assertIn("drift_scores", result)
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        # Add some performance history
        for i in range(5):
            metrics = ModelPerformanceMetrics(
                model_id="model_123",
                timestamp=(datetime.now() - timedelta(days=i)).isoformat(),
                prediction_count=100,
                tenant_id="tenant_123",
                accuracy=0.9 + i * 0.01,
                api_calls_count=1,
                compute_time_seconds=0.5
            )
            self.monitor.performance_history.append(metrics)
        
        summary = self.monitor.get_performance_summary(last_n_days=7)
        
        self.assertIn("total_predictions", summary)
        self.assertEqual(summary["total_predictions"], 500)
        self.assertIn("metrics", summary)
        self.assertIn("billing", summary)
        self.assertEqual(summary["billing"]["total_api_calls"], 5)
    
    def test_create_monitoring_report(self):
        """Test creating monitoring report."""
        # Add some data
        self.monitor.total_predictions = 1000
        self.monitor.total_api_calls = 50
        self.monitor.total_compute_time = 25.5
        
        report = self.monitor.create_monitoring_report()
        
        self.assertEqual(report["model_id"], "model_123")
        self.assertEqual(report["tenant_id"], "tenant_123")
        self.assertIn("performance_summary", report)
        self.assertIn("drift_analysis", report)
        self.assertIn("billing_summary", report)
        self.assertEqual(report["billing_summary"]["total_predictions"], 1000)
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        self.monitor.total_predictions = 10000
        self.monitor.total_compute_time = 3600  # 1 hour
        self.monitor.total_api_calls = 1000
        
        cost = self.monitor._estimate_cost()
        
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0)


class TestDataQualityMonitor(unittest.TestCase):
    """Test data quality monitoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.expected_schema = {
            'age': 'int64',
            'income': 'float64',
            'category': 'object'
        }
        
        self.quality_monitor = DataQualityMonitor(self.expected_schema)
    
    def test_initialization(self):
        """Test data quality monitor initialization."""
        self.assertEqual(self.quality_monitor.expected_schema, self.expected_schema)
        self.assertEqual(len(self.quality_monitor.quality_checks_history), 0)
    
    @patch('psutil.Process')
    def test_check_data_quality_clean_data(self, mock_process):
        """Test quality check on clean data."""
        # Mock memory info
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024**2
        
        # Create clean data
        data = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'income': [50000.0, 60000.0, 70000.0, 80000.0],
            'category': ['A', 'B', 'A', 'B']
        })
        
        report = self.quality_monitor.check_data_quality(data, tenant_id="tenant_123")
        
        self.assertEqual(report["quality_score"], 100.0)
        self.assertEqual(len(report["issues"]), 0)
    
    @patch('psutil.Process')
    def test_check_missing_values(self, mock_process):
        """Test detection of missing values."""
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024**2
        
        # Data with missing values
        data = pd.DataFrame({
            'age': [25, None, 35, 40],
            'income': [50000.0, 60000.0, None, 80000.0],
            'category': ['A', 'B', 'A', 'B']
        })
        
        report = self.quality_monitor.check_data_quality(data)
        
        self.assertLess(report["quality_score"], 100.0)
        self.assertTrue(any(issue["type"] == "missing_values" for issue in report["issues"]))
    
    @patch('psutil.Process')
    def test_check_duplicates(self, mock_process):
        """Test detection of duplicate rows."""
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024**2
        
        # Data with duplicates
        data = pd.DataFrame({
            'age': [25, 25, 35, 40],
            'income': [50000.0, 50000.0, 70000.0, 80000.0],
            'category': ['A', 'A', 'A', 'B']
        })
        
        report = self.quality_monitor.check_data_quality(data)
        
        self.assertLess(report["quality_score"], 100.0)
        self.assertTrue(any(issue["type"] == "duplicate_rows" for issue in report["issues"]))
    
    @patch('psutil.Process')
    def test_check_outliers(self, mock_process):
        """Test outlier detection."""
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024**2
        
        # Data with outliers
        data = pd.DataFrame({
            'age': [25, 30, 35, 400],  # 400 is an outlier
            'income': [50000.0, 60000.0, 70000.0, 80000.0],
            'category': ['A', 'B', 'A', 'B']
        })
        
        report = self.quality_monitor.check_data_quality(data)
        
        self.assertTrue(any(warning["type"] == "outliers" for warning in report["warnings"]))
    
    @patch('psutil.Process')
    def test_check_schema_mismatch(self, mock_process):
        """Test schema mismatch detection."""
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024**2
        
        # Data with wrong types
        data = pd.DataFrame({
            'age': ['25', '30', '35', '40'],  # String instead of int
            'income': [50000.0, 60000.0, 70000.0, 80000.0],
            'category': ['A', 'B', 'A', 'B']
        })
        
        report = self.quality_monitor.check_data_quality(data)
        
        self.assertLess(report["quality_score"], 100.0)
        self.assertTrue(any(issue["type"] == "schema_mismatch" for issue in report["issues"]))
    
    @patch('psutil.Process')
    def test_check_constant_columns(self, mock_process):
        """Test detection of constant columns."""
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024**2
        
        # Data with constant column
        data = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'income': [50000.0, 50000.0, 50000.0, 50000.0],  # All same value
            'category': ['A', 'B', 'A', 'B']
        })
        
        report = self.quality_monitor.check_data_quality(data)
        
        self.assertTrue(any(warning["type"] == "constant_column" for warning in report["warnings"]))
    
    @patch('psutil.Process')
    def test_check_high_cardinality(self, mock_process):
        """Test high cardinality detection."""
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024**2
        
        # Data with high cardinality
        data = pd.DataFrame({
            'age': range(100),
            'income': range(50000, 60000, 100),
            'category': [f"cat_{i}" for i in range(100)]  # Too many unique values
        })
        
        report = self.quality_monitor.check_data_quality(data)
        
        self.assertTrue(any(warning["type"] == "high_cardinality" for warning in report["warnings"]))
    
    def test_compatible_types(self):
        """Test type compatibility checking."""
        monitor = DataQualityMonitor()
        
        # Compatible types
        self.assertTrue(monitor._compatible_types("int64", "int64"))
        self.assertTrue(monitor._compatible_types("int64", "float64"))
        self.assertTrue(monitor._compatible_types("int32", "int64"))
        self.assertTrue(monitor._compatible_types("object", "string"))
        
        # Incompatible types
        self.assertFalse(monitor._compatible_types("int64", "object"))
        self.assertFalse(monitor._compatible_types("float64", "bool"))
    
    def test_get_quality_trend(self):
        """Test getting quality trend."""
        # Add some quality checks to history
        for i in range(5):
            check = {
                "quality_score": 90 + i,
                "memory_usage_mb": 10 + i,
                "issues": [{"type": "missing_values"}] if i % 2 == 0 else []
            }
            self.quality_monitor.quality_checks_history.append(check)
        
        trend = self.quality_monitor.get_quality_trend(last_n_checks=5)
        
        self.assertEqual(trend["checks_analyzed"], 5)
        self.assertIn("avg_quality_score", trend)
        self.assertIn("min_quality_score", trend)
        self.assertIn("max_quality_score", trend)
        self.assertIn("common_issues", trend)


class TestAlertManager(unittest.TestCase):
    """Test alert management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.alert_config = {
            "accuracy_threshold": 0.85,
            "drift_threshold": 0.5,
            "latency_threshold": 1.0,
            "quality_score_threshold": 80,
            "notification_channels": ["log"]
        }
        
        self.alert_manager = AlertManager(alert_config=self.alert_config)
    
    def test_initialization(self):
        """Test alert manager initialization."""
        self.assertEqual(self.alert_manager.alert_config["accuracy_threshold"], 0.85)
        self.assertIn("log", self.alert_manager.notification_handlers)
    
    def test_check_accuracy_alert(self):
        """Test accuracy threshold alert."""
        metrics = {"accuracy": 0.80}  # Below threshold
        
        alerts = self.alert_manager.check_alerts(metrics)
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["type"], "low_accuracy")
        self.assertEqual(alerts[0]["severity"], "high")
    
    def test_check_drift_alert(self):
        """Test drift threshold alert."""
        metrics = {"drift_score": 0.6}  # Above threshold
        
        alerts = self.alert_manager.check_alerts(metrics)
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["type"], "data_drift")
        self.assertEqual(alerts[0]["severity"], "medium")
    
    def test_check_latency_alert(self):
        """Test latency threshold alert."""
        metrics = {"latency": 1.5}  # Above threshold
        
        alerts = self.alert_manager.check_alerts(metrics)
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["type"], "high_latency")
    
    def test_check_quality_alert(self):
        """Test data quality alert."""
        metrics = {"quality_score": 75}  # Below threshold
        
        alerts = self.alert_manager.check_alerts(metrics)
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["type"], "poor_data_quality")
    
    def test_check_billing_alert(self):
        """Test billing threshold alert."""
        self.alert_manager.alert_config["billing_threshold"] = 100.0
        metrics = {"billing_amount": 150.0}  # Above threshold
        
        alerts = self.alert_manager.check_alerts(metrics)
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["type"], "high_billing")
        self.assertEqual(alerts[0]["severity"], "high")
    
    def test_multiple_alerts(self):
        """Test multiple alerts triggered simultaneously."""
        metrics = {
            "accuracy": 0.80,  # Below threshold
            "drift_score": 0.6,  # Above threshold
            "latency": 1.5  # Above threshold
        }
        
        alerts = self.alert_manager.check_alerts(metrics)
        
        self.assertEqual(len(alerts), 3)
        alert_types = [alert["type"] for alert in alerts]
        self.assertIn("low_accuracy", alert_types)
        self.assertIn("data_drift", alert_types)
        self.assertIn("high_latency", alert_types)
    
    def test_resolve_alert(self):
        """Test resolving active alerts."""
        # Add some active alerts
        self.alert_manager.active_alerts = [
            {"type": "low_accuracy"},
            {"type": "data_drift"}
        ]
        
        self.alert_manager.resolve_alert("low_accuracy")
        
        self.assertEqual(len(self.alert_manager.active_alerts), 1)
        self.assertEqual(self.alert_manager.active_alerts[0]["type"], "data_drift")
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        self.alert_manager.active_alerts = [
            {"type": "low_accuracy"},
            {"type": "high_latency"}
        ]
        
        active = self.alert_manager.get_active_alerts()
        
        self.assertEqual(len(active), 2)
    
    def test_get_alert_summary(self):
        """Test getting alert summary."""
        # Add alerts to history
        self.alert_manager.alert_history = [
            {"type": "low_accuracy", "severity": "high"},
            {"type": "low_accuracy", "severity": "high"},
            {"type": "data_drift", "severity": "medium"},
            {"type": "high_latency", "severity": "medium"}
        ]
        
        summary = self.alert_manager.get_alert_summary()
        
        self.assertEqual(summary["total_alerts_triggered"], 4)
        self.assertEqual(summary["alerts_by_type"]["low_accuracy"], 2)
        self.assertEqual(summary["alerts_by_type"]["data_drift"], 1)
        self.assertEqual(summary["alerts_by_severity"]["high"], 2)
        self.assertEqual(summary["alerts_by_severity"]["medium"], 2)
    
    @patch.dict(os.environ, {'SLACK_WEBHOOK_URL': 'https://hooks.slack.com/test'})
    @patch('automl_platform.monitoring.MonitoringIntegration.send_to_slack')
    def test_slack_notification(self, mock_slack):
        """Test Slack notification handler."""
        alert_manager = AlertManager()
        alert = {"type": "test_alert", "message": "Test message"}
        
        alert_manager._slack_notification(alert)
        
        mock_slack.assert_called_once()
    
    @patch.dict(os.environ, {'ALERT_EMAIL_RECIPIENTS': 'test@example.com'})
    @patch('automl_platform.monitoring.MonitoringIntegration.send_to_email')
    def test_email_notification(self, mock_email):
        """Test email notification handler."""
        alert_manager = AlertManager()
        alert = {"type": "test_alert", "message": "Test message"}
        
        alert_manager._email_notification(alert)
        
        mock_email.assert_called_once()


class TestMonitoringIntegration(unittest.TestCase):
    """Test monitoring system integrations."""
    
    @patch('automl_platform.monitoring.PROMETHEUS_AVAILABLE', True)
    def test_export_to_prometheus(self):
        """Test Prometheus metrics export."""
        monitor = ModelMonitor(
            model_id="test_model",
            model_type="classification"
        )
        
        metrics_bytes = MonitoringIntegration.export_to_prometheus(monitor)
        
        self.assertIsInstance(metrics_bytes, bytes)
    
    def test_export_to_grafana_json(self):
        """Test Grafana JSON export."""
        monitor = ModelMonitor(
            model_id="test_model",
            model_type="classification"
        )
        
        # Add some performance history
        for i in range(3):
            metrics = ModelPerformanceMetrics(
                model_id="test_model",
                timestamp=datetime.now().isoformat(),
                prediction_count=100,
                accuracy=0.9
            )
            monitor.performance_history.append(metrics)
        
        grafana_data = MonitoringIntegration.export_to_grafana_json(monitor)
        
        self.assertIn("dashboardId", grafana_data)
        self.assertIn("panels", grafana_data)
        self.assertEqual(grafana_data["dashboardId"], "test_model")
    
    @patch('requests.post')
    def test_send_to_slack(self, mock_post):
        """Test sending alert to Slack."""
        mock_post.return_value.status_code = 200
        
        alert = {
            "type": "low_accuracy",
            "severity": "high",
            "message": "Model accuracy dropped",
            "timestamp": datetime.now().isoformat(),
            "model_id": "model_123"
        }
        
        result = MonitoringIntegration.send_to_slack(alert, "https://webhook.url")
        
        self.assertTrue(result)
        mock_post.assert_called_once()
        
        # Check message format
        call_args = mock_post.call_args
        json_data = call_args[1]['json']
        self.assertIn("attachments", json_data)
    
    @patch('smtplib.SMTP')
    def test_send_to_email(self, mock_smtp_class):
        """Test sending alert via email."""
        mock_smtp = Mock()
        mock_smtp_class.return_value = mock_smtp
        
        alert = {
            "type": "data_drift",
            "severity": "medium",
            "message": "Data drift detected",
            "timestamp": datetime.now().isoformat()
        }
        
        smtp_config = {
            'host': 'smtp.gmail.com',
            'port': 587,
            'from_email': 'alerts@automl.com'
        }
        
        recipients = ['admin@example.com']
        
        result = MonitoringIntegration.send_to_email(alert, smtp_config, recipients)
        
        self.assertTrue(result)
        mock_smtp.starttls.assert_called_once()
        mock_smtp.send_message.assert_called_once()
    
    @patch('requests.post')
    def test_send_to_webhook(self, mock_post):
        """Test sending alert to generic webhook."""
        mock_post.return_value.status_code = 200
        
        alert = {"type": "test_alert", "message": "Test"}
        
        result = MonitoringIntegration.send_to_webhook(
            alert,
            "https://webhook.url",
            headers={"Authorization": "Bearer token"}
        )
        
        self.assertTrue(result)
        mock_post.assert_called_once()


class TestMonitoringService(unittest.TestCase):
    """Test central monitoring service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.billing_tracker = Mock()
        self.storage_manager = Mock()
        
        self.service = MonitoringService(
            storage_manager=self.storage_manager,
            billing_tracker=self.billing_tracker
        )
    
    def test_initialization(self):
        """Test monitoring service initialization."""
        self.assertIsNotNone(self.service.monitors)
        self.assertIsNotNone(self.service.quality_monitor)
        self.assertIsNotNone(self.service.alert_manager)
    
    def test_register_model(self):
        """Test registering a model for monitoring."""
        reference_data = pd.DataFrame({'feature': [1, 2, 3]})
        
        monitor = self.service.register_model(
            model_id="model_789",
            model_type="regression",
            reference_data=reference_data,
            tenant_id="tenant_456"
        )
        
        self.assertIsInstance(monitor, ModelMonitor)
        self.assertEqual(monitor.model_id, "model_789")
        self.assertIn("model_789", self.service.monitors)
    
    def test_get_monitor(self):
        """Test getting monitor for specific model."""
        self.service.register_model(
            model_id="model_123",
            model_type="classification"
        )
        
        monitor = self.service.get_monitor("model_123")
        
        self.assertIsNotNone(monitor)
        self.assertEqual(monitor.model_id, "model_123")
    
    def test_log_prediction_for_model(self):
        """Test logging prediction for specific model."""
        # Register model
        monitor = self.service.register_model(
            model_id="model_123",
            model_type="classification"
        )
        
        # Mock log_prediction
        monitor.log_prediction = Mock()
        
        # Log prediction through service
        features = pd.DataFrame({'f1': [1, 2]})
        predictions = np.array([0, 1])
        
        self.service.log_prediction(
            model_id="model_123",
            features=features,
            predictions=predictions
        )
        
        monitor.log_prediction.assert_called_once()
    
    def test_check_all_models_health(self):
        """Test checking health of all models."""
        # Register models
        for i in range(3):
            monitor = self.service.register_model(
                model_id=f"model_{i}",
                model_type="classification",
                tenant_id=f"tenant_{i}"
            )
            # Add mock summary
            monitor.get_performance_summary = Mock(return_value={
                "metrics": {"avg_accuracy": 0.85 if i == 0 else 0.95},
                "drift_rate": 0.1
            })
            monitor._estimate_cost = Mock(return_value=10.0 * (i + 1))
        
        health_report = self.service.check_all_models_health()
        
        self.assertIn("models", health_report)
        self.assertEqual(len(health_report["models"]), 3)
        self.assertIn("billing_summary", health_report)
        self.assertEqual(health_report["billing_summary"]["total_estimated_cost"], 60.0)
    
    def test_create_global_dashboard(self):
        """Test creating global dashboard."""
        # Register models
        for i in range(2):
            monitor = self.service.register_model(
                model_id=f"model_{i}",
                model_type="classification"
            )
            monitor.total_predictions = 100 * (i + 1)
            monitor.total_api_calls = 10 * (i + 1)
            monitor.total_compute_time = 3600 * (i + 1)
            monitor._estimate_cost = Mock(return_value=50.0 * (i + 1))
        
        dashboard = self.service.create_global_dashboard()
        
        self.assertEqual(dashboard["total_models"], 2)
        self.assertEqual(len(dashboard["models"]), 2)
        self.assertIn("alerts", dashboard)
        self.assertIn("billing", dashboard)
        self.assertEqual(dashboard["billing"]["total_predictions"], 300)
        self.assertEqual(dashboard["billing"]["total_api_calls"], 30)
    
    def test_save_monitoring_data(self):
        """Test saving monitoring data to storage."""
        # Register model with data
        monitor = self.service.register_model(
            model_id="model_123",
            model_type="classification",
            tenant_id="tenant_123"
        )
        
        # Add performance history
        for i in range(3):
            metrics = ModelPerformanceMetrics(
                model_id="model_123",
                timestamp=datetime.now().isoformat(),
                prediction_count=100
            )
            monitor.performance_history.append(metrics)
        
        # Add drift history
        monitor.drift_detector.drift_history = [
            {"drift_detected": False, "timestamp": datetime.now().isoformat()}
        ]
        
        # Save data
        self.service.save_monitoring_data()
        
        # Check storage calls
        self.assertEqual(self.storage_manager.save_dataset.call_count, 2)


if __name__ == "__main__":
    unittest.main()
