"""
Test Suite for Incremental Learning Module
==========================================
Tests for online learning, drift detection, and streaming ensemble functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.incremental_learning import (
    IncrementalConfig,
    IncrementalModel,
    SGDIncrementalModel,
    RiverIncrementalModel,
    NeuralIncrementalModel,
    StreamingEnsemble,
    IncrementalPipeline
)


class TestIncrementalConfig(unittest.TestCase):
    """Test incremental learning configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = IncrementalConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 0.01)
        self.assertTrue(config.enable_replay)
        self.assertTrue(config.detect_drift)
        self.assertEqual(config.drift_detector, "adwin")
        self.assertEqual(config.buffer_size, 1000)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = IncrementalConfig(
            batch_size=64,
            learning_rate=0.001,
            enable_replay=False,
            detect_drift=False,
            checkpoint_frequency=500
        )
        
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertFalse(config.enable_replay)
        self.assertFalse(config.detect_drift)
        self.assertEqual(config.checkpoint_frequency, 500)
    
    def test_to_dict(self):
        """Test configuration serialization."""
        config = IncrementalConfig(batch_size=100)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['batch_size'], 100)
        self.assertIn('learning_rate', config_dict)


class TestSGDIncrementalModel(unittest.TestCase):
    """Test SGD-based incremental learning model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IncrementalConfig(
            batch_size=10,
            enable_replay=True,
            checkpoint_frequency=50
        )
        self.model = SGDIncrementalModel(self.config, task="classification")
        
        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 10)
        self.y = np.random.randint(0, 2, 100)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.task, "classification")
        self.assertEqual(self.model.n_samples_seen, 0)
        self.assertIsNone(self.model.n_features)
    
    def test_partial_fit(self):
        """Test incremental training."""
        # First batch
        X_batch = self.X[:10]
        y_batch = self.y[:10]
        classes = np.array([0, 1])
        
        self.model.partial_fit(X_batch, y_batch, classes=classes)
        
        self.assertEqual(self.model.n_samples_seen, 10)
        self.assertEqual(self.model.n_features, 10)
        
        # Check replay buffer was updated
        if self.config.enable_replay:
            self.assertEqual(len(self.model.replay_buffer_X), 10)
            self.assertEqual(len(self.model.replay_buffer_y), 10)
    
    def test_predict(self):
        """Test prediction."""
        # Train first
        self.model.partial_fit(self.X[:50], self.y[:50], classes=np.array([0, 1]))
        
        # Predict
        predictions = self.model.predict(self.X[50:60])
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_predict_proba(self):
        """Test probability predictions."""
        # Train first
        self.model.partial_fit(self.X[:50], self.y[:50], classes=np.array([0, 1]))
        
        # Get probabilities
        probas = self.model.predict_proba(self.X[50:60])
        
        self.assertEqual(probas.shape, (10, 2))
        # Check probabilities sum to 1
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(10))
    
    def test_regression_model(self):
        """Test regression variant."""
        model = SGDIncrementalModel(self.config, task="regression")
        
        # Use continuous targets
        y_reg = np.random.randn(100)
        
        # Train
        model.partial_fit(self.X[:50], y_reg[:50])
        
        # Predict
        predictions = model.predict(self.X[50:60])
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(all(isinstance(p, (float, np.floating)) for p in predictions))
    
    @patch('automl_platform.incremental_learning.Path.mkdir')
    @patch('builtins.open', create=True)
    @patch('pickle.dump')
    def test_checkpoint_saving(self, mock_pickle_dump, mock_open, mock_mkdir):
        """Test checkpoint saving."""
        # Train to trigger checkpoint
        for i in range(0, 60, 10):
            self.model.partial_fit(
                self.X[i:i+10], 
                self.y[i:i+10], 
                classes=np.array([0, 1])
            )
        
        # Check that checkpoint was called (at 50 samples)
        self.assertTrue(mock_open.called or mock_pickle_dump.called)


class TestRiverIncrementalModel(unittest.TestCase):
    """Test River-based incremental learning model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IncrementalConfig(
            detect_drift=True,
            drift_detector="adwin"
        )
        
        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randint(0, 2, 100)
    
    @patch('automl_platform.incremental_learning.RIVER_AVAILABLE', True)
    def test_initialization(self):
        """Test model initialization with different model types."""
        # Test Hoeffding Tree
        model = RiverIncrementalModel(self.config, model_type="hoeffding_tree")
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.metric)
        
        # Test Logistic Regression
        model = RiverIncrementalModel(self.config, model_type="logistic_regression")
        self.assertIsNotNone(model.model)
    
    @patch('automl_platform.incremental_learning.RIVER_AVAILABLE', False)
    def test_river_not_available(self):
        """Test error when River is not installed."""
        with self.assertRaises(ImportError):
            RiverIncrementalModel(self.config)
    
    @patch('automl_platform.incremental_learning.RIVER_AVAILABLE', True)
    @patch('automl_platform.incremental_learning.tree.HoeffdingTreeClassifier')
    def test_partial_fit(self, mock_tree):
        """Test incremental training with River."""
        mock_model = Mock()
        mock_tree.return_value = mock_model
        
        model = RiverIncrementalModel(self.config, model_type="hoeffding_tree")
        model.model = mock_model
        
        # Train on batch
        model.partial_fit(self.X[:10], self.y[:10])
        
        # Check that learn_one was called for each sample
        self.assertEqual(mock_model.learn_one.call_count, 10)
        self.assertEqual(model.n_samples_seen, 10)
    
    @patch('automl_platform.incremental_learning.RIVER_AVAILABLE', True)
    def test_predict(self):
        """Test prediction with River model."""
        model = RiverIncrementalModel(self.config, model_type="hoeffding_tree")
        
        # Mock the model's predict_one method
        model.model.predict_one = Mock(return_value=1)
        
        # Predict
        predictions = model.predict(self.X[:5])
        
        self.assertEqual(len(predictions), 5)
        self.assertEqual(model.model.predict_one.call_count, 5)
    
    @patch('automl_platform.incremental_learning.RIVER_AVAILABLE', True)
    def test_drift_detection(self):
        """Test drift detection integration."""
        model = RiverIncrementalModel(self.config, model_type="hoeffding_tree")
        
        # Mock drift detector
        model.drift_detector = Mock()
        model.drift_detector.drift_detected = False
        
        # Train with drift check
        model.partial_fit(self.X[:10], self.y[:10])
        
        # Simulate drift detection
        model.drift_detector.drift_detected = True
        model.partial_fit(self.X[10:20], self.y[10:20])
        
        # Model should be reinitialized on drift
        self.assertTrue(model.drift_detector.drift_detected)


class TestNeuralIncrementalModel(unittest.TestCase):
    """Test neural network-based incremental learning."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IncrementalConfig(
            learning_rate=0.001,
            enable_replay=True,
            replay_frequency=20
        )
        self.model = NeuralIncrementalModel(self.config, task="classification")
        
        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 10)
        self.y = np.random.randint(0, 3, 100)
    
    def test_initialization(self):
        """Test neural model initialization."""
        self.assertIsNotNone(self.model.model)
        self.assertEqual(self.model.task, "classification")
        self.assertIsNone(self.model.classes_seen)
    
    def test_partial_fit_classification(self):
        """Test incremental training for classification."""
        classes = np.array([0, 1, 2])
        
        # First batch
        self.model.partial_fit(self.X[:10], self.y[:10], classes=classes)
        
        self.assertEqual(self.model.n_samples_seen, 10)
        np.testing.assert_array_equal(self.model.classes_seen, classes)
    
    def test_partial_fit_regression(self):
        """Test incremental training for regression."""
        model = NeuralIncrementalModel(self.config, task="regression")
        
        # Use continuous targets
        y_reg = np.random.randn(100)
        
        # Train
        model.partial_fit(self.X[:20], y_reg[:20])
        
        self.assertEqual(model.n_samples_seen, 20)
    
    @patch.object(NeuralIncrementalModel, 'replay_training')
    def test_experience_replay(self, mock_replay):
        """Test experience replay triggering."""
        # Train enough to trigger replay
        classes = np.array([0, 1, 2])
        
        for i in range(0, 30, 10):
            self.model.partial_fit(
                self.X[i:i+10], 
                self.y[i:i+10], 
                classes=classes
            )
        
        # Replay should be triggered at 20 samples (replay_frequency=20)
        self.assertTrue(mock_replay.called)


class TestStreamingEnsemble(unittest.TestCase):
    """Test streaming ensemble of incremental models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IncrementalConfig()
        
        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(50, 10)
        self.y = np.random.randint(0, 2, 50)
    
    @patch('automl_platform.incremental_learning.RIVER_AVAILABLE', False)
    def test_initialization_without_river(self):
        """Test ensemble initialization when River is not available."""
        ensemble = StreamingEnsemble(self.config, base_models=["sgd", "neural"])
        
        self.assertIn("sgd", ensemble.models)
        self.assertIn("neural", ensemble.models)
        self.assertNotIn("river_tree", ensemble.models)  # River not available
    
    def test_partial_fit(self):
        """Test ensemble training."""
        ensemble = StreamingEnsemble(self.config, base_models=["sgd"])
        
        # Train ensemble
        ensemble.partial_fit(self.X[:20], self.y[:20], classes=np.array([0, 1]))
        
        # Check that model was trained
        self.assertEqual(ensemble.models["sgd"].n_samples_seen, 20)
    
    def test_weight_update(self):
        """Test model weight updates based on performance."""
        ensemble = StreamingEnsemble(self.config, base_models=["sgd"])
        
        # Add mock performance
        ensemble.model_performance["sgd"].extend([0.8, 0.85, 0.9])
        
        # Update weights
        ensemble._update_weights()
        
        # Check weights were updated and normalized
        self.assertAlmostEqual(sum(ensemble.model_weights.values()), 1.0)
        self.assertGreater(ensemble.model_weights["sgd"], 0)
    
    def test_predict_weighted_voting(self):
        """Test weighted ensemble prediction."""
        ensemble = StreamingEnsemble(self.config, base_models=["sgd"])
        
        # Train first
        ensemble.partial_fit(self.X[:30], self.y[:30], classes=np.array([0, 1]))
        
        # Mock predictions
        ensemble.models["sgd"].predict = Mock(return_value=np.array([0, 1, 0, 1, 1]))
        
        # Predict
        predictions = ensemble.predict(self.X[:5])
        
        self.assertEqual(len(predictions), 5)


class TestIncrementalPipeline(unittest.TestCase):
    """Test incremental learning pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IncrementalConfig(max_samples=100)
        self.pipeline = IncrementalPipeline(self.config)
        
        # Create mock model
        self.model = Mock()
        self.model.n_samples_seen = 0
        self.model.partial_fit = Mock()
        self.model.predict = Mock(return_value=np.array([0, 1, 0]))
        self.model.predict_proba = Mock(return_value=np.array([[0.6, 0.4]]))
    
    def test_set_model(self):
        """Test setting model in pipeline."""
        self.pipeline.set_model(self.model)
        self.assertEqual(self.pipeline.model, self.model)
    
    def test_process_stream(self):
        """Test stream processing."""
        self.pipeline.set_model(self.model)
        
        # Create data generator
        def data_generator():
            for i in range(3):
                X = np.random.randn(10, 5)
                y = np.random.randint(0, 2, 10)
                yield X, y
        
        # Process stream
        stats = self.pipeline.process_stream(data_generator(), max_samples=30)
        
        # Check statistics
        self.assertEqual(stats['n_samples'], 30)
        self.assertEqual(stats['n_batches'], 3)
        self.assertEqual(self.model.partial_fit.call_count, 3)

    def test_evaluate_prequential(self):
        """Test prequential evaluation (test-then-train)."""
        self.pipeline.set_model(self.model)
        self.model.n_samples_seen = 1  # Pretend model has been trained
        
        # Create data generator
        def data_generator():
            for i in range(3):
                X = np.random.randn(3, 5)
                y = np.array([0, 1, 0])
                yield X, y
        
        # Evaluate
        scores = self.pipeline.evaluate_prequential(data_generator(), metric="accuracy")
        
        # Check evaluation
        self.assertEqual(len(scores), 3)
        self.assertEqual(self.model.predict.call_count, 3)
        self.assertEqual(self.model.partial_fit.call_count, 3)
    
    def test_max_samples_limit(self):
        """Test that pipeline respects max_samples limit."""
        self.pipeline.set_model(self.model)
        
        # Create infinite generator
        def infinite_generator():
            while True:
                X = np.random.randn(10, 5)
                y = np.random.randint(0, 2, 10)
                yield X, y
        
        # Process with limit
        stats = self.pipeline.process_stream(infinite_generator(), max_samples=50)
        
        # Should stop at 50 samples
        self.assertEqual(stats['n_samples'], 50)

    def test_predict_incremental_with_external_pipeline(self):
        """Pipeline should delegate incremental predictions to provided pipeline."""
        self.pipeline.set_model(self.model)
        X = pd.DataFrame(np.random.randn(10, 3))

        external_pipeline = Mock()
        external_pipeline.predict = Mock(side_effect=[
            np.zeros(4),
            np.ones(4),
            np.full(2, 2)
        ])

        predictions = self.pipeline.predict_incremental(external_pipeline, X, batch_size=4)

        self.assertEqual(external_pipeline.predict.call_count, 3)
        self.model.predict.assert_not_called()
        np.testing.assert_array_equal(predictions, np.concatenate([
            np.zeros(4),
            np.ones(4),
            np.full(2, 2)
        ]))

    def test_predict_incremental_prefers_pipeline_incremental_method(self):
        """When available the pipeline's incremental method should be used first."""
        self.pipeline.set_model(self.model)
        X = pd.DataFrame(np.random.randn(8, 3))

        outputs = [
            np.zeros(3),
            np.ones(3),
            np.full(2, 2)
        ]

        external_pipeline = Mock()
        external_pipeline.predict_incremental = Mock(side_effect=outputs)
        external_pipeline.predict = Mock(side_effect=AssertionError("Should not call standard predict"))

        predictions = self.pipeline.predict_incremental(external_pipeline, X, batch_size=3)

        self.assertEqual(external_pipeline.predict_incremental.call_count, 3)
        external_pipeline.predict.assert_not_called()
        np.testing.assert_array_equal(predictions, np.concatenate(outputs))

    def test_predict_incremental_defaults_to_internal_model(self):
        """If no external pipeline is provided, use the internal incremental model."""
        outputs = [
            np.arange(4),
            np.arange(4, 8),
            np.arange(8, 10)
        ]
        self.model.predict = Mock(side_effect=outputs)
        self.pipeline.set_model(self.model)

        X = np.random.randn(10, 2)

        predictions = self.pipeline.predict_incremental(None, X, batch_size=4)

        self.assertEqual(self.model.predict.call_count, 3)
        np.testing.assert_array_equal(predictions, np.concatenate(outputs))

    def test_predict_proba_incremental_with_external_pipeline(self):
        """Probability predictions should be batched with external pipelines."""
        self.pipeline.set_model(self.model)
        X = pd.DataFrame(np.random.randn(9, 3))

        external_pipeline = Mock()
        external_pipeline.predict_proba = Mock(side_effect=[
            np.full((4, 2), 0.5),
            np.full((4, 2), 0.25),
            np.full((1, 2), 0.75)
        ])

        probas = self.pipeline.predict_proba_incremental(external_pipeline, X, batch_size=4)

        self.assertEqual(external_pipeline.predict_proba.call_count, 3)
        self.assertEqual(probas.shape, (9, 2))
        self.model.predict_proba.assert_not_called()

    def test_predict_proba_incremental_prefers_pipeline_incremental_method(self):
        """Probability flow should prioritise dedicated incremental API when present."""
        self.pipeline.set_model(self.model)
        X = pd.DataFrame(np.random.randn(7, 3))

        outputs = [
            np.full((3, 2), 0.4),
            np.full((3, 2), 0.6),
            np.full((1, 2), 0.8)
        ]

        external_pipeline = Mock()
        external_pipeline.predict_proba_incremental = Mock(side_effect=outputs)
        external_pipeline.predict_proba = Mock(side_effect=AssertionError("Should not call standard predict_proba"))

        probas = self.pipeline.predict_proba_incremental(external_pipeline, X, batch_size=3)

        self.assertEqual(external_pipeline.predict_proba_incremental.call_count, 3)
        external_pipeline.predict_proba.assert_not_called()
        np.testing.assert_array_equal(probas, np.vstack(outputs))

    def test_predict_proba_incremental_requires_predictor(self):
        """An error should be raised when no probability predictor is available."""
        X = np.random.randn(5, 2)

        with self.assertRaises(ValueError):
            self.pipeline.predict_proba_incremental(None, X)


class TestDriftDetection(unittest.TestCase):
    """Test drift detection functionality."""
    
    @patch('automl_platform.incremental_learning.RIVER_AVAILABLE', True)
    @patch('automl_platform.incremental_learning.river.drift.ADWIN')
    def test_adwin_drift_detector(self, mock_adwin):
        """Test ADWIN drift detector initialization."""
        config = IncrementalConfig(detect_drift=True, drift_detector="adwin")
        model = SGDIncrementalModel(config)
        
        mock_adwin.assert_called_once()
        self.assertIsNotNone(model.drift_detector)
    
    @patch('automl_platform.incremental_learning.RIVER_AVAILABLE', True)
    @patch('automl_platform.incremental_learning.river.drift.DDM')
    def test_ddm_drift_detector(self, mock_ddm):
        """Test DDM drift detector initialization."""
        config = IncrementalConfig(detect_drift=True, drift_detector="ddm")
        model = SGDIncrementalModel(config)
        
        mock_ddm.assert_called_once()
        self.assertIsNotNone(model.drift_detector)
    
    def test_check_drift(self):
        """Test drift checking logic."""
        config = IncrementalConfig(detect_drift=True)
        model = SGDIncrementalModel(config)
        
        # Mock drift detector
        model.drift_detector = Mock()
        model.drift_detector.drift_detected = False
        
        # Check drift - no drift
        result = model.check_drift(0.1)
        self.assertFalse(result)
        
        # Simulate drift detection
        model.drift_detector.drift_detected = True
        result = model.check_drift(0.5)
        self.assertTrue(result)


class TestReplayBuffer(unittest.TestCase):
    """Test experience replay buffer functionality."""
    
    def test_replay_buffer_update(self):
        """Test updating replay buffer."""
        config = IncrementalConfig(enable_replay=True, buffer_size=100)
        model = SGDIncrementalModel(config)
        
        # Add samples to buffer
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model.update_replay_buffer(X, y)
        
        self.assertEqual(len(model.replay_buffer_X), 50)
        self.assertEqual(len(model.replay_buffer_y), 50)
    
    def test_replay_buffer_max_size(self):
        """Test replay buffer respects max size."""
        config = IncrementalConfig(enable_replay=True, buffer_size=20)
        model = SGDIncrementalModel(config)
        
        # Add more samples than buffer size
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        
        model.update_replay_buffer(X, y)
        
        # Buffer should only keep last 20 samples
        self.assertEqual(len(model.replay_buffer_X), 20)
        self.assertEqual(len(model.replay_buffer_y), 20)
    
    @patch.object(SGDIncrementalModel, 'partial_fit')
    def test_replay_training(self, mock_partial_fit):
        """Test training on replay buffer."""
        config = IncrementalConfig(
            enable_replay=True, 
            buffer_size=100,
            replay_sample_size=10
        )
        model = SGDIncrementalModel(config)
        
        # Fill replay buffer
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        model.update_replay_buffer(X, y)
        
        # Trigger replay training
        model.replay_training()
        
        # Check that partial_fit was called with replay samples
        mock_partial_fit.assert_called_once()
        call_args = mock_partial_fit.call_args[0]
        self.assertEqual(len(call_args[0]), 10)  # replay_sample_size


if __name__ == "__main__":
    unittest.main()
