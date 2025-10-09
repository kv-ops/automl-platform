"""
Incremental Learning Module for AutoML Platform
Supports online learning with River, SGD, and streaming algorithms
Place in: automl_platform/incremental_learning.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
import pickle
import json
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import time

# Scikit-learn incremental models
from sklearn.linear_model import SGDClassifier, SGDRegressor, PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.base import BaseEstimator, clone

river: SimpleNamespace = SimpleNamespace()
linear_model: SimpleNamespace = SimpleNamespace()
naive_bayes: SimpleNamespace = SimpleNamespace()
tree: SimpleNamespace = SimpleNamespace()
ensemble: SimpleNamespace = SimpleNamespace()
river_metrics: SimpleNamespace = SimpleNamespace()
river_preprocessing: SimpleNamespace = SimpleNamespace()
river_feature: SimpleNamespace = SimpleNamespace()

try:
    import river as _river_module  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    _river_module = None

RIVER_AVAILABLE = _river_module is not None


def _install_river_fallbacks() -> None:
    """Populate minimal stand-ins when River cannot be imported."""

    class _FallbackRiverClassifier:
        """Very small classifier that mimics River's ``learn_one`` API."""

        def __init__(self):
            self._class_counts: Dict[Any, int] = {}
            self._total = 0

        def learn_one(self, x, y):  # pragma: no cover - simple fallback behaviour
            self._class_counts[y] = self._class_counts.get(y, 0) + 1
            self._total += 1

        def predict_one(self, x):  # pragma: no cover - simple fallback behaviour
            if not self._class_counts:
                return 0
            return max(self._class_counts, key=self._class_counts.get)

        def predict_proba_one(self, x):  # pragma: no cover - simple fallback behaviour
            if not self._class_counts or self._total == 0:
                return {}
            return {
                cls: count / self._total
                for cls, count in self._class_counts.items()
            }

    class _FallbackDriftDetector:
        """Fallback drift detector that never signals drift."""

        def __init__(self):
            self.drift_detected = False

        def update(self, *args, **kwargs):  # pragma: no cover - minimal behaviour
            return self

        def reset(self):  # pragma: no cover - minimal behaviour
            self.drift_detected = False

    class _FallbackAccuracy:
        """Minimal accuracy tracker with River-like ``update``/``get`` API."""

        def __init__(self):
            self._correct = 0
            self._total = 0

        def update(self, y_true, y_pred):  # pragma: no cover - minimal behaviour
            if y_true == y_pred:
                self._correct += 1
            self._total += 1

        def get(self):  # pragma: no cover - minimal behaviour
            if self._total == 0:
                return 0.0
            return self._correct / self._total

    global river, linear_model, naive_bayes, tree, ensemble, river_metrics, river_preprocessing, river_feature
    river = SimpleNamespace(
        drift=SimpleNamespace(
            ADWIN=_FallbackDriftDetector,
            DDM=_FallbackDriftDetector,
            EDDM=_FallbackDriftDetector,
            PageHinkley=_FallbackDriftDetector,
        )
    )
    linear_model = SimpleNamespace(
        LogisticRegression=_FallbackRiverClassifier,
        Perceptron=_FallbackRiverClassifier,
    )
    naive_bayes = SimpleNamespace(
        GaussianNB=_FallbackRiverClassifier,
    )
    tree = SimpleNamespace(
        HoeffdingTreeClassifier=_FallbackRiverClassifier,
        HoeffdingAdaptiveTreeClassifier=_FallbackRiverClassifier,
    )
    ensemble = SimpleNamespace(
        AdaptiveRandomForestClassifier=_FallbackRiverClassifier,
    )
    river_metrics = SimpleNamespace(
        Accuracy=_FallbackAccuracy,
    )
    river_preprocessing = SimpleNamespace()
    river_feature = SimpleNamespace()


def _use_real_river(module) -> None:
    """Bind the actual River module to the global placeholders."""

    global river, linear_model, naive_bayes, tree, ensemble, river_metrics, river_preprocessing, river_feature
    river = module  # type: ignore[assignment]
    linear_model = module.linear_model  # type: ignore[assignment]
    naive_bayes = module.naive_bayes  # type: ignore[assignment]
    tree = module.tree  # type: ignore[assignment]
    ensemble = module.ensemble  # type: ignore[assignment]
    river_metrics = module.metrics  # type: ignore[assignment]
    river_preprocessing = module.preprocessing  # type: ignore[assignment]
    river_feature = module.feature_extraction  # type: ignore[assignment]


if RIVER_AVAILABLE and _river_module is not None:
    _use_real_river(_river_module)
else:
    _install_river_fallbacks()

# Vowpal Wabbit (optional)
try:
    from vowpalwabbit import pyvw
    VW_AVAILABLE = True
except ImportError:
    VW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class IncrementalConfig:
    """Configuration for incremental learning."""
    batch_size: int = 32
    learning_rate: float = 0.01
    max_samples: Optional[int] = None
    
    # Memory management
    buffer_size: int = 1000
    enable_replay: bool = True
    replay_frequency: int = 100
    replay_sample_size: int = 32
    
    # Concept drift handling
    detect_drift: bool = True
    drift_detector: str = "adwin"  # adwin, ddm, eddm, page_hinkley
    drift_threshold: float = 0.05
    
    # Model update strategy
    update_strategy: str = "always"  # always, drift, periodic
    update_frequency: int = 100  # For periodic updates
    
    # Performance tracking
    track_performance: bool = True
    window_size: int = 1000
    metrics_to_track: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    
    # Checkpointing
    checkpoint_frequency: int = 1000
    checkpoint_path: str = "./incremental_checkpoints"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class IncrementalModel:
    """Base class for incremental learning models."""
    
    def __init__(self, config: IncrementalConfig):
        """Initialize incremental model."""
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.n_samples_seen = 0
        self.n_features = None
        
        # Performance tracking
        self.performance_history = {
            'accuracy': deque(maxlen=config.window_size),
            'loss': deque(maxlen=config.window_size),
            'timestamps': deque(maxlen=config.window_size)
        }
        
        # Replay buffer
        if config.enable_replay:
            self.replay_buffer_X = deque(maxlen=config.buffer_size)
            self.replay_buffer_y = deque(maxlen=config.buffer_size)
        
        # Drift detection
        self.drift_detector = None
        if config.detect_drift:
            self._init_drift_detector()
    
    def _init_drift_detector(self):
        """Initialize drift detector."""
        if not RIVER_AVAILABLE:
            logger.warning("River not available for drift detection")
            return
        
        if self.config.drift_detector == "adwin":
            self.drift_detector = river.drift.ADWIN()
        elif self.config.drift_detector == "ddm":
            self.drift_detector = river.drift.DDM()
        elif self.config.drift_detector == "eddm":
            self.drift_detector = river.drift.EDDM()
        elif self.config.drift_detector == "page_hinkley":
            self.drift_detector = river.drift.PageHinkley()
        else:
            logger.warning(f"Unknown drift detector: {self.config.drift_detector}")
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None):
        """
        Incrementally train the model on a batch of data.
        
        Args:
            X: Feature batch
            y: Target batch
            classes: All possible classes (for classification)
        """
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError
    
    def update_replay_buffer(self, X: np.ndarray, y: np.ndarray):
        """Update replay buffer with new samples."""
        if self.config.enable_replay:
            for x_i, y_i in zip(X, y):
                self.replay_buffer_X.append(x_i)
                self.replay_buffer_y.append(y_i)
    
    def replay_training(self):
        """Train on samples from replay buffer."""
        if not self.config.enable_replay or len(self.replay_buffer_X) == 0:
            return
        
        # Sample from replay buffer
        n_samples = min(self.config.replay_sample_size, len(self.replay_buffer_X))
        indices = np.random.choice(len(self.replay_buffer_X), n_samples, replace=False)
        
        X_replay = np.array([self.replay_buffer_X[i] for i in indices])
        y_replay = np.array([self.replay_buffer_y[i] for i in indices])
        
        # Train on replay samples
        self.partial_fit(X_replay, y_replay)
    
    def check_drift(self, error_rate: float) -> bool:
        """
        Check for concept drift.
        
        Args:
            error_rate: Current error rate
            
        Returns:
            True if drift detected
        """
        if self.drift_detector is None:
            return False
        
        # Update drift detector
        self.drift_detector.update(error_rate)
        
        # Check for drift
        if self.drift_detector.drift_detected:
            logger.warning(f"Concept drift detected at sample {self.n_samples_seen}")
            return True
        
        return False
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = Path(self.config.checkpoint_path) / f"checkpoint_{self.n_samples_seen}.pkl"
        else:
            path = Path(path)
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model': self.model,
            'scaler': self.scaler,
            'n_samples_seen': self.n_samples_seen,
            'n_features': self.n_features,
            'performance_history': dict(self.performance_history),
            'config': self.config.to_dict()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.model = checkpoint['model']
        self.scaler = checkpoint['scaler']
        self.n_samples_seen = checkpoint['n_samples_seen']
        self.n_features = checkpoint['n_features']
        self.performance_history = checkpoint['performance_history']
        
        logger.info(f"Loaded checkpoint from {path}")


class SGDIncrementalModel(IncrementalModel):
    """SGD-based incremental learning model."""
    
    def __init__(self, config: IncrementalConfig, task: str = "classification"):
        """Initialize SGD incremental model."""
        super().__init__(config)
        self.task = task
        
        if task == "classification":
            self.model = SGDClassifier(
                loss='log_loss',
                learning_rate='adaptive',
                eta0=config.learning_rate,
                random_state=42,
                warm_start=True
            )
        else:
            self.model = SGDRegressor(
                loss='squared_error',
                learning_rate='adaptive',
                eta0=config.learning_rate,
                random_state=42,
                warm_start=True
            )
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None):
        """Incrementally train SGD model."""
        # Update scaler
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Set number of features
        if self.n_features is None:
            self.n_features = X.shape[1]
        
        # Partial fit
        if self.task == "classification" and classes is not None:
            self.model.partial_fit(X_scaled, y, classes=classes)
        else:
            self.model.partial_fit(X_scaled, y)
        
        self.n_samples_seen += len(X)
        
        # Update replay buffer
        self.update_replay_buffer(X, y)
        
        # Periodic replay
        if self.config.enable_replay and self.n_samples_seen % self.config.replay_frequency == 0:
            self.replay_training()
        
        # Checkpoint
        if self.n_samples_seen % self.config.checkpoint_frequency == 0:
            self.save_checkpoint()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class RiverIncrementalModel(IncrementalModel):
    """River-based incremental learning model."""
    
    def __init__(self, config: IncrementalConfig, model_type: str = "hoeffding_tree"):
        """Initialize River incremental model."""
        super().__init__(config)
        
        if not RIVER_AVAILABLE:
            raise ImportError(
                "river is required for RiverIncrementalModel. "
                "Install it with 'pip install river'."
            )
        
        self.model_type = model_type
        self._init_model()
        
        # Metrics
        self.metric = river_metrics.Accuracy()
    
    def _init_model(self):
        """Initialize River model."""
        if self.model_type == "hoeffding_tree":
            self.model = tree.HoeffdingTreeClassifier()
        elif self.model_type == "hoeffding_adaptive_tree":
            self.model = tree.HoeffdingAdaptiveTreeClassifier()
        elif self.model_type == "adaptive_random_forest":
            self.model = ensemble.AdaptiveRandomForestClassifier(n_models=10)
        elif self.model_type == "logistic_regression":
            self.model = linear_model.LogisticRegression()
        elif self.model_type == "perceptron":
            self.model = linear_model.Perceptron()
        elif self.model_type == "naive_bayes":
            self.model = naive_bayes.GaussianNB()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None):
        """Incrementally train River model."""
        for x_i, y_i in zip(X, y):
            # Convert to dict for River
            x_dict = {f"feature_{i}": val for i, val in enumerate(x_i)}
            
            # Learn one sample
            self.model.learn_one(x_dict, y_i)
            
            # Update metric
            y_pred = self.model.predict_one(x_dict)
            self.metric.update(y_i, y_pred)
            
            self.n_samples_seen += 1
            
            # Track performance
            if self.config.track_performance:
                self.performance_history['accuracy'].append(self.metric.get())
                self.performance_history['timestamps'].append(datetime.now())
            
            # Check for drift
            if self.config.detect_drift:
                error = 1 - self.metric.get()
                if self.check_drift(error):
                    # Reset model on drift
                    self._init_model()
                    logger.info("Model reset due to drift")
        
        # Checkpoint
        if self.n_samples_seen % self.config.checkpoint_frequency == 0:
            self.save_checkpoint()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = []
        
        for x_i in X:
            x_dict = {f"feature_{i}": val for i, val in enumerate(x_i)}
            y_pred = self.model.predict_one(x_dict)
            predictions.append(y_pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        probas = []
        
        for x_i in X:
            x_dict = {f"feature_{i}": val for i, val in enumerate(x_i)}
            proba = self.model.predict_proba_one(x_dict)
            probas.append(proba)
        
        return np.array(probas)


class NeuralIncrementalModel(IncrementalModel):
    """Neural network-based incremental learning."""
    
    def __init__(self, config: IncrementalConfig, task: str = "classification"):
        """Initialize neural incremental model."""
        super().__init__(config)
        self.task = task
        
        if task == "classification":
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                learning_rate_init=config.learning_rate,
                warm_start=True,
                max_iter=1,  # Single iteration per partial_fit
                random_state=42
            )
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                learning_rate_init=config.learning_rate,
                warm_start=True,
                max_iter=1,
                random_state=42
            )
        
        self.classes_seen = None
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None):
        """Incrementally train neural network."""
        # Update scaler
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Track classes for classification
        if self.task == "classification":
            if classes is not None:
                self.classes_seen = classes
            elif self.classes_seen is None:
                self.classes_seen = np.unique(y)
            
            self.model.partial_fit(X_scaled, y, classes=self.classes_seen)
        else:
            self.model.partial_fit(X_scaled, y)
        
        self.n_samples_seen += len(X)
        
        # Update replay buffer
        self.update_replay_buffer(X, y)
        
        # Experience replay
        if self.config.enable_replay and self.n_samples_seen % self.config.replay_frequency == 0:
            self.replay_training()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class StreamingEnsemble:
    """Ensemble of incremental models for streaming data."""
    
    def __init__(self, config: IncrementalConfig, base_models: Optional[List[str]] = None):
        """Initialize streaming ensemble."""
        self.config = config
        
        # Default models
        if base_models is None:
            base_models = ["sgd", "river_tree", "neural"]
        
        # Initialize models
        self.models = {}
        for model_name in base_models:
            if model_name == "sgd":
                self.models[model_name] = SGDIncrementalModel(config)
            elif model_name == "river_tree" and RIVER_AVAILABLE:
                self.models[model_name] = RiverIncrementalModel(config, "hoeffding_tree")
            elif model_name == "neural":
                self.models[model_name] = NeuralIncrementalModel(config)
        
        # Model weights (for weighted ensemble)
        self.model_weights = {name: 1.0 for name in self.models}
        
        # Performance tracking
        self.model_performance = {name: deque(maxlen=100) for name in self.models}
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None):
        """Train all models in ensemble."""
        for name, model in self.models.items():
            try:
                # Make prediction before training (for weight update)
                if hasattr(model, 'predict') and model.n_samples_seen > 0:
                    y_pred = model.predict(X)
                    accuracy = np.mean(y_pred == y)
                    self.model_performance[name].append(accuracy)
                
                # Train model
                model.partial_fit(X, y, classes)
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        # Update weights based on performance
        self._update_weights()
    
    def _update_weights(self):
        """Update model weights based on recent performance."""
        for name in self.models:
            if len(self.model_performance[name]) > 0:
                # Use exponential moving average of accuracy
                recent_performance = np.mean(self.model_performance[name])
                self.model_weights[name] = max(0.1, recent_performance)
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for name in self.model_weights:
                self.model_weights[name] /= total_weight
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted ensemble predictions."""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model.n_samples_seen > 0:
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                    weights.append(self.model_weights[name])
                except Exception as e:
                    logger.error(f"Error predicting with {name}: {e}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted voting for classification
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # For each sample, use weighted majority vote
        ensemble_predictions = []
        for i in range(X.shape[0]):
            sample_preds = predictions[:, i]
            unique_preds = np.unique(sample_preds)
            
            # Calculate weighted votes
            votes = {}
            for pred in unique_preds:
                mask = sample_preds == pred
                votes[pred] = np.sum(weights[mask])
            
            # Select prediction with highest weight
            ensemble_predictions.append(max(votes, key=votes.get))
        
        return np.array(ensemble_predictions)


class IncrementalPipeline:
    """Complete incremental learning pipeline."""

    def __init__(self, config: IncrementalConfig):
        """Initialize incremental pipeline."""
        self.config = config
        self.model = None
        self.preprocessor = None
        self.feature_extractor = None
        
        # Stream statistics
        self.stream_stats = {
            'n_samples': 0,
            'n_batches': 0,
            'total_time': 0,
            'avg_batch_time': 0
        }

    @staticmethod
    def _prepare_batches(X: Union[pd.DataFrame, np.ndarray, List[Any]],
                         batch_size: int):
        """Yield batches from the provided dataset preserving the original type."""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            total = len(X)
            for start in range(0, total, batch_size):
                yield X.iloc[start:start + batch_size]
        else:
            X_array = np.asarray(X)
            total = X_array.shape[0]
            for start in range(0, total, batch_size):
                yield X_array[start:start + batch_size]

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray:
        """Convert prediction output to numpy array without losing structure."""
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        if isinstance(data, pd.Series):
            return data.to_numpy()
        if isinstance(data, np.ndarray):
            return data
        return np.asarray(data)

    @staticmethod
    def _stack_batches(batches: List[np.ndarray]) -> np.ndarray:
        """Combine prediction batches into a single numpy array."""
        if not batches:
            return np.array([])

        first = batches[0]
        if first.ndim <= 1:
            return np.concatenate([batch.ravel() for batch in batches], axis=0)

        return np.vstack(batches)

    def _resolve_predict_function(self,
                                  pipeline: Optional[Any],
                                  proba: bool = False):
        """Resolve the prediction function to use for incremental inference."""
        methods_to_try: List[Tuple[Any, str]] = []

        if pipeline is not None:
            if proba:
                methods_to_try.append((pipeline, 'predict_proba_incremental'))
                methods_to_try.append((pipeline, 'predict_proba'))
            else:
                methods_to_try.append((pipeline, 'predict_incremental'))
                methods_to_try.append((pipeline, 'predict'))

        if self.model is not None:
            methods_to_try.append((self.model, 'predict_proba' if proba else 'predict'))

        for candidate, method_name in methods_to_try:
            if hasattr(candidate, method_name):
                method = getattr(candidate, method_name)
                return method, candidate is self.model

        raise ValueError("No suitable model available for prediction")

    def _prepare_batch_for_model(self, batch: Any, use_internal_model: bool) -> Any:
        """Prepare batch input depending on whether the internal model is used."""
        if not use_internal_model:
            return batch

        if isinstance(batch, pd.DataFrame):
            return batch.to_numpy()
        if isinstance(batch, pd.Series):
            return batch.to_numpy()
        return np.asarray(batch)
    
    def set_model(self, model: IncrementalModel):
        """Set the incremental model."""
        self.model = model
    
    def process_stream(self, 
                       data_generator,
                       max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Process streaming data.
        
        Args:
            data_generator: Generator yielding (X, y) batches
            max_samples: Maximum samples to process
            
        Returns:
            Processing statistics
        """
        if self.model is None:
            raise ValueError("No model set")
        
        max_samples = max_samples or self.config.max_samples
        
        for batch_idx, (X_batch, y_batch) in enumerate(data_generator):
            start_time = time.time()
            
            # Preprocess if needed
            if self.preprocessor:
                X_batch = self.preprocessor.transform(X_batch)
            
            # Extract features if needed
            if self.feature_extractor:
                X_batch = self.feature_extractor.transform(X_batch)
            
            # Train model
            self.model.partial_fit(X_batch, y_batch)
            
            # Update statistics
            batch_time = time.time() - start_time
            self.stream_stats['n_samples'] += len(X_batch)
            self.stream_stats['n_batches'] += 1
            self.stream_stats['total_time'] += batch_time
            self.stream_stats['avg_batch_time'] = \
                self.stream_stats['total_time'] / self.stream_stats['n_batches']
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Processed {self.stream_stats['n_samples']} samples, "
                          f"Avg batch time: {self.stream_stats['avg_batch_time']:.3f}s")
            
            # Check stopping condition
            if max_samples and self.stream_stats['n_samples'] >= max_samples:
                break
        
        return self.stream_stats

    def predict_incremental(self,
                             pipeline: Optional[Any],
                             X: Union[pd.DataFrame, np.ndarray, List[Any]],
                             batch_size: Optional[int] = None) -> np.ndarray:
        """Generate predictions using incremental batching."""
        if X is None:
            raise ValueError("Input features X must be provided")

        batch_size = batch_size or self.config.batch_size or len(X)
        batch_size = max(int(batch_size), 1)
        method, using_internal = self._resolve_predict_function(pipeline, proba=False)

        predictions: List[np.ndarray] = []
        for X_batch in self._prepare_batches(X, batch_size):
            batch_input = self._prepare_batch_for_model(X_batch, using_internal)
            batch_pred = method(batch_input)
            predictions.append(self._to_numpy(batch_pred))

        return self._stack_batches(predictions)

    def predict_proba_incremental(self,
                                  pipeline: Optional[Any],
                                  X: Union[pd.DataFrame, np.ndarray, List[Any]],
                                  batch_size: Optional[int] = None) -> np.ndarray:
        """Generate probability predictions using incremental batching."""
        if X is None:
            raise ValueError("Input features X must be provided")

        batch_size = batch_size or self.config.batch_size or len(X)
        batch_size = max(int(batch_size), 1)
        method, using_internal = self._resolve_predict_function(pipeline, proba=True)

        predictions: List[np.ndarray] = []
        for X_batch in self._prepare_batches(X, batch_size):
            batch_input = self._prepare_batch_for_model(X_batch, using_internal)
            batch_pred = method(batch_input)
            predictions.append(self._to_numpy(batch_pred))

        return self._stack_batches(predictions)
    
    def evaluate_prequential(self,
                            data_generator,
                            metric: str = "accuracy") -> List[float]:
        """
        Prequential evaluation (test-then-train).
        
        Args:
            data_generator: Generator yielding (X, y) batches
            metric: Evaluation metric
            
        Returns:
            List of metric values over time
        """
        if self.model is None:
            raise ValueError("No model set")
        
        metric_values = []
        
        for X_batch, y_batch in data_generator:
            # Test on current batch
            if self.model.n_samples_seen > 0:
                y_pred = self.model.predict(X_batch)
                
                if metric == "accuracy":
                    score = np.mean(y_pred == y_batch)
                elif metric == "mse":
                    score = np.mean((y_pred - y_batch) ** 2)
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                metric_values.append(score)
            
            # Train on current batch
            self.model.partial_fit(X_batch, y_batch)
        
        return metric_values


# Example usage
def main():
    """Example of incremental learning."""
    from sklearn.datasets import make_classification
    
    # Configuration
    config = IncrementalConfig(
        batch_size=32,
        learning_rate=0.01,
        enable_replay=True,
        detect_drift=True,
        checkpoint_frequency=500
    )
    
    # Create model
    model = SGDIncrementalModel(config, task="classification")
    
    # Generate streaming data
    def data_generator(n_batches=100, batch_size=32):
        """Generate streaming data batches."""
        for _ in range(n_batches):
            X, y = make_classification(
                n_samples=batch_size,
                n_features=20,
                n_classes=2,
                random_state=None  # Random data each time
            )
            yield X, y
    
    # Create pipeline
    pipeline = IncrementalPipeline(config)
    pipeline.set_model(model)
    
    # Process stream
    stats = pipeline.process_stream(
        data_generator(),
        max_samples=3200
    )
    
    print(f"Processed {stats['n_samples']} samples")
    print(f"Average batch time: {stats['avg_batch_time']:.3f}s")
    
    # Evaluate
    scores = pipeline.evaluate_prequential(
        data_generator(n_batches=10),
        metric="accuracy"
    )
    
    print(f"Average accuracy: {np.mean(scores):.3f}")


if __name__ == "__main__":
    main()


# Backward compatibility export expected by orchestrator modules
IncrementalLearner = IncrementalPipeline
