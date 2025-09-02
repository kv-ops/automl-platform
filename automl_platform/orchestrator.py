"""
Enhanced AutoML Orchestrator with Optimizations
===============================================
Place in: automl_platform/orchestrator.py (REPLACE EXISTING)

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

# Import MLOps components
from .mlflow_registry import MLflowRegistry, ABTestingService, ModelStage
from .retraining_service import RetrainingService
from .export_service import ModelExporter

logger = logging.getLogger(__name__)


class AutoMLOrchestrator:
    """Enhanced AutoML orchestrator with distributed training, caching and MLOps"""
    
    def __init__(self, config: AutoMLConfig):
        """
        Initialize orchestrator with optimization and MLOps components
        
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
        if hasattr(config, 'distributed_training') and config.distributed_training:
            self.distributed_trainer = DistributedTrainer(
                backend=getattr(config, 'distributed_backend', 'ray'),
                n_workers=getattr(config, 'n_workers', 4)
            )
            logger.info(f"Distributed training enabled with {config.distributed_backend}")
        
        # Setup incremental learning if enabled
        if hasattr(config, 'incremental_learning') and config.incremental_learning:
            self.incremental_learner = IncrementalLearner(
                max_memory_mb=getattr(config, 'max_memory_mb', 1000)
            )
            logger.info("Incremental learning enabled")
        
        # Setup pipeline cache if enabled
        if hasattr(config, 'enable_cache') and config.enable_cache:
            cache_config = CacheConfig(
                backend=getattr(config, 'cache_backend', 'redis'),
                redis_host=getattr(config, 'redis_host', 'localhost'),
                ttl_seconds=getattr(config, 'cache_ttl', 3600),
                compression=getattr(config, 'cache_compression', True),
                invalidate_on_drift=getattr(config, 'cache_invalidate_on_drift', True),
                invalidate_on_performance_drop=getattr(config, 'cache_invalidate_on_perf_drop', True)
            )
            self.pipeline_cache = PipelineCache(cache_config)
            logger.info(f"Pipeline cache enabled with {cache_config.backend} backend")
        
        # Initialize MLOps components
        self.registry = MLflowRegistry(config)
        self.exporter = ModelExporter()
        self.ab_testing = ABTestingService(self.registry)
        
        # Training metadata
        self.training_id = str(uuid.uuid4())
        self.training_metadata = {}
        
        logger.info(f"Orchestrator initialized with training ID: {self.training_id}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            task: Optional[str] = None,
            use_cache: bool = True,
            use_distributed: bool = None,
            use_incremental: bool = None,
            register_best_model: bool = True,
            model_name: Optional[str] = None) -> 'AutoMLOrchestrator':
        """
        Run complete AutoML pipeline with optimizations and MLOps
        
        Args:
            X: Training features
            y: Training labels
            task: Task type (classification/regression/auto)
            use_cache: Whether to use pipeline cache
            use_distributed: Whether to use distributed training
            use_incremental: Whether to use incremental learning
            register_best_model: Whether to register best model in MLflow
            model_name: Custom name for the model in registry
        """
        
        # Override with config if not specified
        if use_distributed is None:
            use_distributed = hasattr(self.config, 'distributed_training') and self.config.distributed_training and self.distributed_trainer
        if use_incremental is None:
            use_incremental = hasattr(self.config, 'incremental_learning') and self.config.incremental_learning and self.incremental_learner
        
        # Generate cache key based on data and config
        cache_key = None
        if use_cache and self.pipeline_cache:
            # Create a hash of the config and data shape
            config_str = json.dumps(self.config.to_dict(), sort_keys=True)
            cache_key = f"automl_{hash(config_str)}_{X.shape}_{task}"
            
            # Try to get from cache
            cached_pipeline = self.pipeline_cache.get_pipeline(cache_key, X)
            if cached_pipeline:
                logger.info("Using cached pipeline")
                self.best_pipeline = cached_pipeline
                self.task = task or detect_task(y)
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
            "incremental": use_incremental,
            "cached": False
        }
        
        # Get available models
        if self.config.algorithms == ['all']:
            models = get_available_models(
                self.task,
                include_incremental=use_incremental
            )
        else:
            all_models = get_available_models(
                self.task,
                include_incremental=use_incremental
            )
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
            
            for model_name_str, model in incremental_models.items():
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('model', model)
                ])
                
                # Evaluate
                scores = cross_val_score(pipeline, X, y, cv=cv, 
                                        scoring=scoring, n_jobs=-1)
                
                # Calculate metrics
                pipeline.fit(X, y)
                y_pred = pipeline.predict(X)
                metrics = calculate_metrics(y, y_pred, None, self.task)
                
                result = {
                    'model': f"Incremental_{model_name_str}",
                    'cv_score': scores.mean(),
                    'cv_std': scores.std(),
                    'metrics': metrics,
                    'params': {},
                    'training_time': 0,
                    'pipeline': pipeline,
                    'incremental': True
                }
                
                self.leaderboard.append(result)
                logger.info(f"Incremental {model_name_str}: CV Score = {scores.mean():.4f}")
        
        # Use distributed training if enabled
        if use_distributed and self.distributed_trainer:
            logger.info("Starting distributed training")
            
            # Prepare parameter grids for all models
            param_grids = {}
            for model_name_str in models.keys():
                grid = get_param_grid(model_name_str)
                if grid:
                    param_grids[model_name_str] = {f'model__{k}': v for k, v in grid.items()}
            
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
                    if param_grid and self.config.hpo_n_iter > 0:
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
            if use_cache and self.pipeline_cache and cache_key:
                self.pipeline_cache.set_pipeline(
                    cache_key,
                    self.best_pipeline,
                    X,
                    metrics=self.leaderboard[0]['metrics'],
                    ttl=getattr(self.config, 'cache_ttl', 3600)
                )
                logger.info("Best pipeline cached")
            
            # Calculate feature importance
            try:
                self._calculate_feature_importance(X, y)
            except:
                pass
            
            # Register best model in MLflow
            if register_best_model:
                if model_name is None:
                    model_name = f"automl_{self.task}_{self.training_id[:8]}"
                
                best_result = self.leaderboard[0]
                
                model_version = self.registry.register_model(
                    model=self.best_pipeline,
                    model_name=model_name,
                    metrics=best_result['metrics'],
                    params=best_result['params'],
                    X_sample=X.head(100),
                    y_sample=y.head(100),
                    description=f"Best model from AutoML run {self.training_id}",
                    tags={
                        "training_id": self.training_id,
                        "task": self.task,
                        "algorithm": best_result['model'],
                        "cv_score": str(best_result['cv_score']),
                        "distributed": str(best_result.get('distributed', False)),
                        "incremental": str(best_result.get('incremental', False))
                    }
                )
                
                logger.info(f"Registered model {model_name} version {model_version.version}")
                
                self.training_metadata["registered_model"] = {
                    "name": model_name,
                    "version": model_version.version,
                    "run_id": model_version.run_id
                }
        
        self.training_metadata["end_time"] = datetime.utcnow().isoformat()
        
        # Clean up distributed resources
        if self.distributed_trainer:
            self.distributed_trainer.shutdown()
        
        return self
    
    def predict(self, X: pd.DataFrame, 
                use_incremental: bool = False,
                use_ab_test: bool = False,
                test_id: Optional[str] = None) -> np.ndarray:
        """
        Make predictions with optional incremental processing and A/B testing
        
        Args:
            X: Features to predict
            use_incremental: Whether to use incremental prediction for large data
            use_ab_test: Whether to use A/B testing
            test_id: Specific A/B test ID
        """
        
        if self.best_pipeline is None:
            raise ValueError("No model trained yet")
        
        # Use A/B testing if enabled
        if use_ab_test and test_id:
            model_type, version = self.ab_testing.route_prediction(test_id)
            # Here we would load the specific model version
            # For simplicity, using the best pipeline
            predictions = self.best_pipeline.predict(X)
            self.ab_testing.record_result(test_id, model_type, True)
            return predictions
        
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
    
    def export_best_model(self, 
                         format: str = "onnx",
                         output_dir: Optional[str] = None,
                         sample_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Export best model to specified format
        
        Args:
            format: Export format ('onnx', 'pmml', 'edge')
            output_dir: Output directory for exported model
            sample_data: Sample data for shape inference
        """
        
        if self.best_pipeline is None:
            raise ValueError("No model trained yet")
        
        if sample_data is None:
            sample_data = pd.DataFrame(
                np.random.randn(10, self.training_metadata.get("n_features", 10))
            )
        
        model_name = self.training_metadata.get("registered_model", {}).get("name", "model")
        
        if format == "onnx":
            result = self.exporter.export_to_onnx(
                self.best_pipeline,
                sample_data,
                model_name=model_name,
                output_path=output_dir
            )
        elif format == "pmml":
            sample_output = pd.Series(np.random.randint(0, 2, 10))
            result = self.exporter.export_to_pmml(
                self.best_pipeline,
                sample_data,
                sample_output,
                model_name=model_name,
                output_path=output_dir
            )
        elif format == "edge":
            result = self.exporter.export_for_edge(
                self.best_pipeline,
                sample_data,
                model_name=model_name,
                output_dir=output_dir
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Model exported to {format}: {result}")
        return result
    
    def create_ab_test(self,
                       challenger_pipeline: Any,
                       model_name: str,
                       traffic_split: float = 0.1) -> str:
        """
        Create A/B test between current production model and challenger
        """
        
        # Register challenger model
        challenger_version = self.registry.register_model(
            model=challenger_pipeline,
            model_name=model_name,
            metrics={"source": "ab_test_challenger"},
            params={},
            description="Challenger model for A/B test"
        )
        
        # Get current production version
        prod_versions = self.registry.client.get_latest_versions(
            model_name,
            stages=[ModelStage.PRODUCTION.value]
        ) if self.registry.client else []
        
        if not prod_versions:
            self.registry.promote_model(
                model_name,
                challenger_version.version,
                ModelStage.PRODUCTION
            )
            return None
        
        # Create A/B test
        test_id = self.ab_testing.create_ab_test(
            model_name=model_name,
            champion_version=prod_versions[0].version,
            challenger_version=challenger_version.version,
            traffic_split=traffic_split
        )
        
        logger.info(f"Created A/B test {test_id}")
        return test_id
    
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
            
            for metric_name, metric_value in result['metrics'].items():
                row[metric_name] = metric_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_pipeline(self, filepath: str) -> str:
        """Save best pipeline"""
        
        if self.best_pipeline is None:
            raise ValueError("No pipeline to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.best_pipeline, filepath)
        
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
        return str(filepath)
    
    def load_pipeline(self, filepath: str) -> None:
        """Load pipeline"""
        
        filepath = Path(filepath)
        self.best_pipeline = joblib.load(filepath)
        
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
            
            result = permutation_importance(
                self.best_pipeline.named_steps['model'],
                X_transformed, y,
                n_repeats=5,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
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
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get detailed comparison of all trained models"""
        
        if not self.leaderboard:
            return pd.DataFrame()
        
        comparison = []
        for i, result in enumerate(self.leaderboard):
            row = {
                'rank': i + 1,
                'model': result['model'],
                'cv_score': result['cv_score'],
                'cv_std': result['cv_std'],
                'training_time_seconds': result['training_time'],
                'distributed': result.get('distributed', False),
                'incremental': result.get('incremental', False)
            }
            
            for metric, value in result['metrics'].items():
                row[f'metric_{metric}'] = value
            
            for param, value in list(result['params'].items())[:5]:
                row[f'param_{param}'] = value
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        if len(df) > 0:
            best_score = df['cv_score'].iloc[0]
            df['relative_performance'] = (df['cv_score'] / best_score * 100).round(2)
        
        return df
