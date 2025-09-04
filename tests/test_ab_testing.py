"""
Unit tests for A/B Testing Framework
=====================================
Tests for statistical testing, traffic routing, and model comparison
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import base64

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.ab_testing import (
    ABTestingService,
    ABTestConfig,
    ABTestResult,
    TestStatus,
    ModelType,
    MetricsComparator,
    StatisticalTester
)


class TestABTestConfig(unittest.TestCase):
    """Test cases for A/B test configuration."""
    
    def test_config_creation(self):
        """Test A/B test configuration creation."""
        config = ABTestConfig(
            test_id="test_123",
            model_name="test_model",
            champion_version=1,
            challenger_version=2,
            traffic_split=0.2,
            min_samples=100,
            confidence_level=0.95,
            primary_metric="accuracy"
        )
        
        self.assertEqual(config.test_id, "test_123")
        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(config.champion_version, 1)
        self.assertEqual(config.challenger_version, 2)
        self.assertEqual(config.traffic_split, 0.2)
        self.assertEqual(config.min_samples, 100)
        self.assertEqual(config.confidence_level, 0.95)
        self.assertEqual(config.primary_metric, "accuracy")
        self.assertEqual(config.statistical_test, "t_test")
        self.assertEqual(config.min_improvement, 0.02)
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = ABTestConfig(
            test_id="test_123",
            model_name="test_model",
            champion_version=1,
            challenger_version=2
        )
        
        self.assertEqual(config.traffic_split, 0.1)
        self.assertEqual(config.min_samples, 100)
        self.assertEqual(config.max_duration_days, 30)
        self.assertEqual(config.confidence_level, 0.95)
        self.assertEqual(config.primary_metric, "accuracy")
        self.assertIn("precision", config.secondary_metrics)
        self.assertIn("recall", config.secondary_metrics)
        self.assertIn("f1", config.secondary_metrics)


class TestABTestResult(unittest.TestCase):
    """Test cases for A/B test results."""
    
    def test_result_creation(self):
        """Test A/B test result creation."""
        result = ABTestResult(
            test_id="test_123",
            status=TestStatus.ACTIVE
        )
        
        self.assertEqual(result.test_id, "test_123")
        self.assertEqual(result.status, TestStatus.ACTIVE)
        self.assertEqual(result.champion_samples, 0)
        self.assertEqual(result.challenger_samples, 0)
        self.assertIsNone(result.p_value)
        self.assertIsNone(result.winner)
    
    def test_result_to_dict(self):
        """Test result conversion to dictionary."""
        result = ABTestResult(
            test_id="test_123",
            status=TestStatus.ACTIVE,
            champion_samples=100,
            challenger_samples=100,
            p_value=0.03
        )
        
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict['test_id'], "test_123")
        self.assertEqual(result_dict['status'], "active")
        self.assertEqual(result_dict['champion_samples'], 100)
        self.assertEqual(result_dict['p_value'], 0.03)
        self.assertIn('started_at', result_dict)


class TestStatisticalTester(unittest.TestCase):
    """Test cases for statistical testing functions."""
    
    def test_t_test(self):
        """Test t-test calculation."""
        np.random.seed(42)
        samples_a = np.random.normal(0.5, 0.1, 100)
        samples_b = np.random.normal(0.55, 0.1, 100)  # Slightly better
        
        results = StatisticalTester.perform_t_test(samples_a, samples_b)
        
        self.assertIn('statistic', results)
        self.assertIn('p_value', results)
        self.assertIn('effect_size', results)
        self.assertIn('mean_a', results)
        self.assertIn('mean_b', results)
        self.assertIn('confidence_interval', results)
        
        # Check that means are approximately correct
        self.assertAlmostEqual(results['mean_a'], 0.5, places=1)
        self.assertAlmostEqual(results['mean_b'], 0.55, places=1)
        
        # Effect size should be positive (B is better)
        self.assertGreater(results['effect_size'], 0)
    
    def test_mann_whitney(self):
        """Test Mann-Whitney U test."""
        np.random.seed(42)
        samples_a = np.random.exponential(1.0, 100)
        samples_b = np.random.exponential(1.2, 100)
        
        results = StatisticalTester.perform_mann_whitney(samples_a, samples_b)
        
        self.assertIn('statistic', results)
        self.assertIn('p_value', results)
        self.assertIn('effect_size', results)
        self.assertIn('median_a', results)
        self.assertIn('median_b', results)
    
    def test_chi_square(self):
        """Test chi-square test for categorical data."""
        # Create contingency table
        # [[correct_a, incorrect_a], [correct_b, incorrect_b]]
        contingency_table = np.array([[80, 20], [90, 10]])
        
        results = StatisticalTester.perform_chi_square(contingency_table)
        
        self.assertIn('statistic', results)
        self.assertIn('p_value', results)
        self.assertIn('degrees_of_freedom', results)
        self.assertIn('effect_size', results)  # Cramér's V
        
        self.assertEqual(results['degrees_of_freedom'], 1)
    
    @patch('automl_platform.ab_testing.TTestPower')
    def test_sample_size_calculation(self, mock_power_class):
        """Test required sample size calculation."""
        mock_power = Mock()
        mock_power.solve_power.return_value = 63.76
        mock_power_class.return_value = mock_power
        
        sample_size = StatisticalTester.calculate_sample_size(
            effect_size=0.5,
            alpha=0.05,
            power=0.80
        )
        
        self.assertEqual(sample_size, 64)  # Rounded up
        mock_power.solve_power.assert_called_once_with(
            effect_size=0.5,
            alpha=0.05,
            power=0.80,
            alternative='two-sided'
        )


class TestMetricsComparator(unittest.TestCase):
    """Test cases for metrics comparison."""
    
    @patch('automl_platform.ab_testing.plt')
    def test_compare_classification_metrics(self, mock_plt):
        """Test classification metrics comparison."""
        np.random.seed(42)
        
        # Create sample data
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 1])
        y_pred_a = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1])  # 8/10 correct
        y_pred_b = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 1])  # 10/10 correct
        
        # Mock probabilities
        y_proba_a = np.array([[0.3, 0.7], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3],
                              [0.6, 0.4], [0.3, 0.7], [0.8, 0.2], [0.4, 0.6],
                              [0.2, 0.8], [0.3, 0.7]])
        y_proba_b = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.8, 0.2],
                              [0.2, 0.8], [0.1, 0.9], [0.9, 0.1], [0.8, 0.2],
                              [0.1, 0.9], [0.2, 0.8]])
        
        # Compare metrics
        results = MetricsComparator.compare_classification_metrics(
            y_true, y_pred_a, y_pred_b, y_proba_a, y_proba_b
        )
        
        # Check structure
        self.assertIn('model_a', results)
        self.assertIn('model_b', results)
        self.assertIn('comparison', results)
        self.assertIn('visualizations', results)
        
        # Check metrics
        self.assertEqual(results['model_a']['accuracy'], 0.8)
        self.assertEqual(results['model_b']['accuracy'], 1.0)
        self.assertAlmostEqual(results['comparison']['accuracy_diff'], 0.2)
        self.assertAlmostEqual(results['comparison']['accuracy_improvement_pct'], 25.0)
        
        # Check visualizations were created
        self.assertIn('roc_curves', results['visualizations'])
        self.assertIn('pr_curves', results['visualizations'])
        self.assertIn('confusion_matrices', results['visualizations'])
    
    @patch('automl_platform.ab_testing.plt')
    def test_compare_regression_metrics(self, mock_plt):
        """Test regression metrics comparison."""
        np.random.seed(42)
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_a = np.array([1.1, 2.3, 2.8, 4.2, 4.9])
        y_pred_b = np.array([1.0, 2.1, 3.0, 3.9, 5.0])
        
        results = MetricsComparator.compare_regression_metrics(
            y_true, y_pred_a, y_pred_b
        )
        
        # Check structure
        self.assertIn('model_a', results)
        self.assertIn('model_b', results)
        self.assertIn('comparison', results)
        self.assertIn('visualizations', results)
        
        # Model B should have lower error metrics
        self.assertGreater(results['model_a']['mse'], results['model_b']['mse'])
        self.assertGreater(results['model_a']['mae'], results['model_b']['mae'])
        self.assertLess(results['model_a']['r2'], results['model_b']['r2'])
        
        # Check improvements
        self.assertGreater(results['comparison']['mse_reduction'], 0)
        self.assertGreater(results['comparison']['mae_reduction'], 0)
        self.assertGreater(results['comparison']['r2_improvement'], 0)


class TestABTestingService(unittest.TestCase):
    """Test cases for main A/B testing service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = Mock()
        self.service = ABTestingService(self.registry)
    
    def test_create_ab_test(self):
        """Test A/B test creation."""
        test_id = self.service.create_ab_test(
            model_name="test_model",
            champion_version=1,
            challenger_version=2,
            traffic_split=0.2,
            min_samples=200,
            confidence_level=0.99,
            primary_metric="precision"
        )
        
        # Check test was created
        self.assertIsNotNone(test_id)
        self.assertIn(test_id, self.service.active_tests)
        self.assertIn(test_id, self.service.test_results)
        
        # Check configuration
        config = self.service.active_tests[test_id]
        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(config.champion_version, 1)
        self.assertEqual(config.challenger_version, 2)
        self.assertEqual(config.traffic_split, 0.2)
        self.assertEqual(config.min_samples, 200)
        self.assertEqual(config.confidence_level, 0.99)
        self.assertEqual(config.primary_metric, "precision")
        
        # Check result initialization
        result = self.service.test_results[test_id]
        self.assertEqual(result.status, TestStatus.ACTIVE)
    
    def test_route_prediction(self):
        """Test traffic routing for A/B test."""
        # Create test
        test_id = self.service.create_ab_test(
            model_name="test_model",
            champion_version=1,
            challenger_version=2,
            traffic_split=0.3  # 30% to challenger
        )
        
        # Test routing multiple times
        champion_count = 0
        challenger_count = 0
        
        np.random.seed(42)  # For reproducibility
        for _ in range(1000):
            model_type, version = self.service.route_prediction(test_id)
            if model_type == ModelType.CHAMPION.value:
                champion_count += 1
                self.assertEqual(version, 1)
            else:
                challenger_count += 1
                self.assertEqual(version, 2)
        
        # Check distribution is approximately correct (30% to challenger)
        challenger_ratio = challenger_count / 1000
        self.assertAlmostEqual(challenger_ratio, 0.3, places=1)
    
    def test_record_result(self):
        """Test recording prediction results."""
        # Create test
        test_id = self.service.create_ab_test(
            model_name="test_model",
            champion_version=1,
            challenger_version=2,
            min_samples=2
        )
        
        # Record results
        self.service.record_result(test_id, ModelType.CHAMPION.value, True, 0.95, 0.1)
        self.service.record_result(test_id, ModelType.CHALLENGER.value, True, 0.97, 0.09)
        
        result = self.service.test_results[test_id]
        self.assertEqual(result.champion_samples, 1)
        self.assertEqual(result.challenger_samples, 1)
        self.assertEqual(len(result.predictions_log), 2)
        
        # Check log entries
        self.assertEqual(result.predictions_log[0]['model_type'], ModelType.CHAMPION.value)
        self.assertEqual(result.predictions_log[0]['metric_value'], 0.95)
        self.assertEqual(result.predictions_log[1]['model_type'], ModelType.CHALLENGER.value)
        self.assertEqual(result.predictions_log[1]['metric_value'], 0.97)
    
    @patch.object(StatisticalTester, 'perform_t_test')
    def test_analyze_results(self, mock_t_test):
        """Test results analysis with statistical testing."""
        # Setup mock statistical test
        mock_t_test.return_value = {
            'p_value': 0.02,
            'effect_size': 0.5,
            'confidence_interval': (0.01, 0.05),
            'mean_a': 0.85,
            'mean_b': 0.88
        }
        
        # Create test
        test_id = self.service.create_ab_test(
            model_name="test_model",
            champion_version=1,
            challenger_version=2,
            min_samples=2,
            confidence_level=0.95,
            min_improvement=0.02
        )
        
        # Add prediction logs
        result = self.service.test_results[test_id]
        result.predictions_log = [
            {'model_type': ModelType.CHAMPION.value, 'metric_value': 0.85},
            {'model_type': ModelType.CHAMPION.value, 'metric_value': 0.84},
            {'model_type': ModelType.CHALLENGER.value, 'metric_value': 0.88},
            {'model_type': ModelType.CHALLENGER.value, 'metric_value': 0.87}
        ]
        
        # Analyze results
        self.service._analyze_results(test_id)
        
        # Check analysis results
        self.assertEqual(result.p_value, 0.02)
        self.assertEqual(result.effect_size, 0.5)
        self.assertEqual(result.confidence_interval, (0.01, 0.05))
        self.assertEqual(result.winner, ModelType.CHALLENGER.value)
        self.assertAlmostEqual(result.improvement, 0.03, places=2)
        self.assertAlmostEqual(result.confidence, 0.98, places=2)
    
    def test_get_test_results(self):
        """Test getting test results."""
        # Create and setup test
        test_id = self.service.create_ab_test(
            model_name="test_model",
            champion_version=1,
            challenger_version=2
        )
        
        result = self.service.test_results[test_id]
        result.champion_samples = 100
        result.challenger_samples = 100
        result.p_value = 0.03
        result.winner = ModelType.CHALLENGER.value
        
        # Get results
        results_dict = self.service.get_test_results(test_id)
        
        self.assertEqual(results_dict['test_id'], test_id)
        self.assertEqual(results_dict['model_name'], "test_model")
        self.assertEqual(results_dict['status'], "active")
        self.assertEqual(results_dict['champion_samples'], 100)
        self.assertEqual(results_dict['challenger_samples'], 100)
        self.assertEqual(results_dict['p_value'], 0.03)
        self.assertEqual(results_dict['winner'], ModelType.CHALLENGER.value)
    
    def test_conclude_test(self):
        """Test concluding an A/B test."""
        # Create test
        test_id = self.service.create_ab_test(
            model_name="test_model",
            champion_version=1,
            challenger_version=2
        )
        
        result = self.service.test_results[test_id]
        result.winner = ModelType.CHALLENGER.value
        
        # Conclude without promotion
        final_results = self.service.conclude_test(test_id, promote_winner=False)
        
        self.assertEqual(result.status, TestStatus.CONCLUDED)
        self.assertIsNotNone(result.concluded_at)
        self.assertNotIn(test_id, self.service.active_tests)
        self.registry.promote_model.assert_not_called()
    
    def test_conclude_test_with_promotion(self):
        """Test concluding test with winner promotion."""
        # Create test
        test_id = self.service.create_ab_test(
            model_name="test_model",
            champion_version=1,
            challenger_version=2
        )
        
        result = self.service.test_results[test_id]
        result.winner = ModelType.CHALLENGER.value
        
        # Conclude with promotion
        final_results = self.service.conclude_test(test_id, promote_winner=True)
        
        self.assertEqual(result.status, TestStatus.CONCLUDED)
        self.registry.promote_model.assert_called_once_with(
            "test_model", 2, "Production"
        )
    
    def test_pause_resume_test(self):
        """Test pausing and resuming a test."""
        # Create test
        test_id = self.service.create_ab_test(
            model_name="test_model",
            champion_version=1,
            challenger_version=2
        )
        
        # Pause test
        self.service.pause_test(test_id)
        result = self.service.test_results[test_id]
        self.assertEqual(result.status, TestStatus.PAUSED)
        
        # Resume test
        self.service.resume_test(test_id)
        self.assertEqual(result.status, TestStatus.ACTIVE)
    
    def test_get_active_tests(self):
        """Test getting list of active tests."""
        # Create multiple tests
        test_id1 = self.service.create_ab_test(
            model_name="model1",
            champion_version=1,
            challenger_version=2
        )
        
        test_id2 = self.service.create_ab_test(
            model_name="model2",
            champion_version=1,
            challenger_version=2
        )
        
        # Pause one test
        self.service.pause_test(test_id2)
        
        # Get active tests
        active = self.service.get_active_tests()
        
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]['test_id'], test_id1)
        self.assertEqual(active[0]['model_name'], "model1")
    
    @patch('automl_platform.ab_testing.MetricsComparator.compare_classification_metrics')
    def test_compare_models_offline(self, mock_compare):
        """Test offline model comparison."""
        # Setup mock comparison
        mock_compare.return_value = {
            'model_a': {'accuracy': 0.85},
            'model_b': {'accuracy': 0.88},
            'comparison': {'accuracy_diff': 0.03},
            'visualizations': {}
        }
        
        # Create mock models
        model_a = Mock()
        model_a.predict.return_value = np.array([1, 0, 1])
        model_a.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.2, 0.8]])
        
        model_b = Mock()
        model_b.predict.return_value = np.array([1, 0, 1])
        model_b.predict_proba.return_value = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        
        # Test data
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = pd.Series([1, 0, 1])
        
        # Compare models
        comparison = self.service.compare_models_offline(
            model_a, model_b, X_test, y_test, task="classification"
        )
        
        mock_compare.assert_called_once()
        self.assertEqual(comparison['model_a']['accuracy'], 0.85)
        self.assertEqual(comparison['model_b']['accuracy'], 0.88)


if __name__ == "__main__":
    unittest.main()
