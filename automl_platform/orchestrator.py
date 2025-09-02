"""
Enhanced AutoML Orchestrator with Optimizations
===============================================
Place in: automl_platform/orchestrator.py (MODIFIED VERSION)

Integrates distributed training, incremental learning, and pipeline caching.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import time
import json
from pathlib import Path
import joblib
import logging
import uuid
from datetime import datetime

from .data_prep import DataPreprocessor, handle_imbalance, validate_data
from .model_selection import (
    get_available_models, get_param_grid, get_cv_splitter,
    tune_model, try_optuna
)
from .metrics import calculate_metrics, detect_task
from .config import AutoMLConfig

# Import optimization components
from .distributed_training import DistributedTrainer
from .incremental_learning import IncrementalLearner
from .pipeline_cache import PipelineCache, CacheConfig

logger = logging.getLogger(__name__)


class AutoMLOrchestrator:
    """Enhanced AutoML orchestrator with distributed training and caching"""
    
    def __init__(self, config: AutoMLConfig):
        """
        Initialize orchestrator with optimization components
        
        Args:
            config: AutoML configuration
        """
        self.config = config
        self.preprocessor = DataPreprocessor(config.to_dict())
        self.leaderboard = []
        self.best_pipeline = None
        self.task = None
        self.feature_importance = {}
        
        # Initialize optimization components
        self.distributed_trainer = None
        self.incremental_learner = None
        self.pipeline_cache = None
        
        # Setup distributed training if enabled
        if config.distributed_training:
            self.distributed_trainer = DistributedTrainer(
                backend=config.distributed_backend,
                n_workers=config.n_workers
            )
            logger.info(f"Distributed training enabled with {config.distributed_backend}")
        
        # Setup incremental learning if enabled
        if config.incremental_learning:
            self.incremental_learner = IncrementalLearner(
                max_memory_mb=config.max_memory_mb
            )
            logger.info("Incremental learning enabled")
        
        # Setup pipeline cache if enabled
        if config.enable_cache:
            cache_config = CacheConfig(
                backend=config.cache_backend,
                redis_host=config.redis_host,
                ttl_seconds=config.cache_ttl,
                compression=config.cache_compression,
                invalidate_on_drift=config.cache_invalidate_on_drift,
                invalidate_on_performance_drop=config.cache_invalidate_on_perf_drop
            )
            self.pipeline_cache = PipelineCache(cache_config)
            logger.info(f"Pipeline cache enabled with {config.cache_backend} backend")
        
        # Training metadata
        self.training_id = str(uuid.uuid4())
        self.training_metadata = {}
        
        logger.info(f"Orchestrator initialized with training ID: {self.training_id}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            task: Optional[str] = None,
            use_cache: bool = True,
            use_distributed: bool = None,
            use_incremental: bool = None) -> 'AutoMLOrchestrator':
        """
        Run complete AutoML pipeline with optimizations
        
        Args:
            X: Training features
            y: Training labels
            task: Task type (classification/regression/auto)
            use_cache: Whether to use pipeline cache
            use_distributed: Whether to use distributed training
            use_incremental: Whether to use incremental learning
        """
        
        # Override with config if not specified
        if use_distributed is None:
            use_distributed = self.config.distributed_training and self.distributed_trainer
        if use_incremental is None:
            use_incremental = self.config.incremental_learning and self.incremental_learner
        
        # Check cache first
        if use_cache and self.pipeline_cache:
            cache_key = f"automl_{self.config.get_hash()}_{X.shape}"
            cached_pipeline = self.pipeline_cache.get_pipeline(cache_key, X)
            if cached_pipeline:
                logger.info("Using cached pipeline")
                self.best_pipeline = cached_pipeline
                return self
        
        # Validate data
        validation = validate_data(X)
        if not validation['valid']:
            logger.warning(f"Data quality issues: {validation['issues']}")
        
        # Detect task
        if task is None or task == 'auto':
            self.task = detect_task(y)
        else:
            self.task = task
        
        logger.info(f"Task detected: {self.task}")
        
        # Store training metadata
        self.training_metadata = {
            "training_id": self.training_id,
            "task": self.task,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "start_time": datetime.utcnow().isoformat(),
            "distributed": use_distributed,
            "incremental": use_incremental
        }
        
        # Get available models
        if self.config.algorithms == ['all']:
            models = get_available_models(self.task)
        else:
            all_models = get_available_models(self.task)
            models = {k: v for k, v in all_models.items() 
                     if k in self.config.algorithms}
        
        # Filter excluded models
        for excluded in self.config.exclude_algorithms:
            models.pop(excluded, None)
        
        logger.info(f"Testing {len(models)} models")
        
        # Get CV splitter
        cv = get_cv_splitter(self.task, self.config.cv_folds, 
                           self.config.random_state)
        
        # Determine scoring
        if self.config.scoring == 'auto':
            if self.task == 'classification':
                if len(np.unique(y)) == 2:
                    scoring = 'roc_auc'
                else:
                    scoring = 'f1_weighted'
            else:
                scoring = 'neg_mean_squared_error'
        else:
            scoring = self.config.scoring
        
        # Use incremental learning for large datasets
        if use_incremental and len(X) > 10000:
            logger.info("Using incremental learning for large dataset")
            
            # Train with incremental learner
            incremental_models = self.incremental_learner.train_incremental(X, y, self.task)
            
            for model_name, model in incremental_models.items():
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('model', model)
                ])
                
                # Evaluate
                scores = cross_val_score(pipeline, X, y, cv=cv, 
                                        scoring=scoring, n_jobs=-1)
                
                result = {
                    'model': f"Incremental_{model_name}",
                    'cv_score': scores.mean(),
                    'cv_std': scores.std(),
                    'metrics': {},
                    'params': {},
                    'training_time': 0,
                    'pipeline': pipeline,
                    'incremental': True
                }
                
                self.leaderboard.append(result)
                logger.info(f"Incremental {model_name}: CV Score = {scores.mean():.4f}")
        
        # Use distributed training if enabled
        if use_distributed and self.distributed_trainer:
            logger.info("Starting distributed training")
            
            # Prepare parameter grids for all models
            param_grids = {}
            for model_name in models.keys():
                grid = get_param_grid(model_name)
                if grid:
                    param_grids[model_name] = {f'model__{k}': v for k, v in grid.items()}
            
            # Train models in parallel
            distributed_results = self.distributed_trainer.train_distributed(
                X, y, models, param_grids=param_grids
            )
            
            # Add results to leaderboard
            for result in distributed_results:
                result['distributed'] = True
                self.leaderboard.append(result)
                logger.info(f"Distributed {result['model']}: CV Score = {result['cv_score']:.4f}")
        else:
            # Standard sequential training
            for model_name_str, base_model in models.items():
                logger.info(f"Testing {model_name_str}")
                start_time = time.time()
                
                try:
                    # Create pipeline with preprocessing
                    pipeline = Pipeline([
                        ('preprocessor', DataPreprocessor(self.config.to_dict())),
                        ('model', base_model)
                    ])
                    
                    # Handle imbalance if needed
                    if self.task == 'classification' and self.config.handle_imbalance:
                        if hasattr(base_model, 'class_weight'):
                            base_model.set_params(class_weight='balanced')
                    
                    # Hyperparameter tuning
                    param_grid = get_param_grid(model_name_str)
                    if param_grid:
                        param_grid = {f'model__{k}': v for k, v in param_grid.items()}
                        tuned_model, params = tune_model(
                            pipeline, X, y, param_grid, cv, scoring,
                            self.config.hpo_n_iter
                        )
                        if params:
                            pipeline = tuned_model
                            params = {k.replace('model__', ''): v for k, v in params.items()}
                    else:
                        params = {}
                    
                    # Cross-validate
                    scores = cross_val_score(pipeline, X, y, cv=cv, 
                                            scoring=scoring, n_jobs=-1)
                    
                    # Fit final model on all data
                    pipeline.fit(X, y)
                    y_pred = pipeline.predict(X)
                    
                    if self.task == 'classification' and hasattr(pipeline, 'predict_proba'):
                        y_proba = pipeline.predict_proba(X)
                    else:
                        y_proba = None
                    
                    metrics = calculate_metrics(y, y_pred, y_proba, self.task)
                    
                    # Add to leaderboard
                    result = {
                        'model': model_name_str,
                        'cv_score': scores.mean(),
                        'cv_std': scores.std(),
                        'metrics': metrics,
                        'params': params,
                        'training_time': time.time() - start_time,
                        'pipeline': pipeline,
                        'distributed': False,
                        'incremental': False
                    }
                    
                    self.leaderboard.append(result)
                    
                    logger.info(f"{model_name_str}: CV Score = {scores.mean():.4f} (+/- {scores.std():.4f})")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {model_name_str}: {e}")
                    continue
        
        # Sort leaderboard
        self.leaderboard.sort(key=lambda x: x['cv_score'], reverse=True)
        
        # Select best pipeline
        if self.leaderboard:
            self.best_pipeline = self.leaderboard[0]['pipeline']
            
            # Cache the best pipeline
            if use_cache and self.pipeline_cache:
                cache_key = f"automl_{self.config.get_hash()}_{X.shape}"
                self.pipeline_cache.set_pipeline(
                    cache_key,
                    self.best_pipeline,
                    X,
                    metrics=self.leaderboard[0]['metrics'],
                    ttl=self.config.cache_ttl
                )
                logger.info("Best pipeline cached")
            
            # Calculate feature importance
            try:
                self._calculate_feature_importance(X, y)
            except:
                pass
        
        self.training_metadata["end_time"] = datetime.utcnow().isoformat()
        
        # Clean up distributed resources
        if self.distributed_trainer:
            self.distributed_trainer.shutdown()
        
        return self
    
    def predict(self, X: pd.DataFrame, use_incremental: bool = False) -> np.ndarray:
        """
        Make predictions with optional incremental processing
        
        Args:
            X: Features to predict
            use_incremental: Whether to use incremental prediction for large data
        """
        
        if self.best_pipeline is None:
            raise ValueError("No model trained yet")
        
        # Use incremental prediction for large datasets
        if use_incremental and self.incremental_learner and len(X) > 10000:
            return self.incremental_learner.predict_incremental(self.best_pipeline, X)
        
        return self.best_pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame, use_incremental: bool = False) -> np.ndarray:
        """Get probability predictions with optional incremental processing"""
        
        if self.best_pipeline is None:
            raise ValueError("No model trained yet")
        
        if not hasattr(self.best_pipeline, 'predict_proba'):
            raise ValueError("Model doesn't support probability predictions")
        
        # Use incremental prediction for large datasets
        if use_incremental and self.incremental_learner and len(X) > 10000:
            return self.incremental_learner.predict_proba_incremental(self.best_pipeline, X)
        
        return self.best_pipeline.predict_proba(X)
    
    def get_leaderboard(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get leaderboard as DataFrame"""
        
        if not self.leaderboard:
            return pd.DataFrame()
        
        data = []
        for result in self.leaderboard[:top_n]:
            row = {
                'model': result['model'],
                'cv_score': result['cv_score'],
                'cv_std': result['cv_std'],
                'training_time': result['training_time'],
                'distributed': result.get('distributed', False),
                'incremental': result.get('incremental', False)
            }
            
            # Add metrics
            for metric_name, metric_value in result['metrics'].items():
                row[metric_name] = metric_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_pipeline(self, filepath: str) -> None:
        """Save best pipeline"""
        
        if self.best_pipeline is None:
            raise ValueError("No pipeline to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        joblib.dump(self.best_pipeline, filepath)
        
        # Save metadata
        metadata = {
            'task': self.task,
            'best_model': self.leaderboard[0]['model'] if self.leaderboard else None,
            'cv_score': self.leaderboard[0]['cv_score'] if self.leaderboard else None,
            'metrics': self.leaderboard[0]['metrics'] if self.leaderboard else None,
            'feature_importance': self.feature_importance,
            'training_metadata': self.training_metadata,
            'config': self.config.to_dict(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        metadata_path = filepath.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """Load pipeline"""
        
        filepath = Path(filepath)
        
        # Load pipeline
        self.best_pipeline = joblib.load(filepath)
        
        # Load metadata
        metadata_path = filepath.with_suffix('.meta.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.task = metadata.get('task')
                self.feature_importance = metadata.get('feature_importance', {})
                self.training_metadata = metadata.get('training_metadata', {})
        
        logger.info(f"Pipeline loaded from {filepath}")
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calculate feature importance using permutation"""
        
        from sklearn.inspection import permutation_importance
        
        try:
            X_transformed = self.best_pipeline.named_steps['preprocessor'].transform(X)
            
            # Calculate permutation importance
            result = permutation_importance(
                self.best_pipeline.named_steps['model'],
                X_transformed, y,
                n_repeats=5,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            # Store importance
            self.feature_importance = {
                'importances_mean': result.importances_mean.tolist(),
                'importances_std': result.importances_std.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get pipeline cache statistics"""
        if self.pipeline_cache:
            return self.pipeline_cache.get_stats()
        return {}
    
    def clear_cache(self) -> bool:
        """Clear pipeline cache"""
        if self.pipeline_cache:
            return self.pipeline_cache.clear_all()
        return False
