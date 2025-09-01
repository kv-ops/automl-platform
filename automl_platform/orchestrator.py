"""
Enhanced AutoML Orchestrator with MLflow and Export Integration
===============================================================
Place in: automl_platform/orchestrator.py (UPDATED VERSION)

Integrates MLflow registry, automated retraining, and model export capabilities.
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

# Import new MLOps components
from .mlflow_registry import MLflowRegistry, ABTestingService, ModelStage
from .retraining_service import RetrainingService
from .export_service import ModelExporter

logger = logging.getLogger(__name__)


class AutoMLOrchestrator:
    """Enhanced AutoML orchestrator with MLOps capabilities"""
    
    def __init__(self, config: AutoMLConfig):
        """
        Initialize orchestrator with MLOps components
        
        Args:
            config: AutoML configuration
        """
        self.config = config
        self.preprocessor = DataPreprocessor(config.to_dict())
        self.leaderboard = []
        self.best_pipeline = None
        self.task = None
        self.feature_importance = {}
        
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
            register_best_model: bool = True,
            model_name: Optional[str] = None) -> 'AutoMLOrchestrator':
        """
        Run complete AutoML pipeline with MLflow tracking
        
        Args:
            X: Training features
            y: Training labels
            task: Task type (classification/regression/auto)
            register_best_model: Whether to register best model in MLflow
            model_name: Custom name for the model in registry
        """
        
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
            "start_time": datetime.utcnow().isoformat()
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
        
        # Test each model
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
                    'pipeline': pipeline
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
                        "cv_score": str(best_result['cv_score'])
                    }
                )
                
                logger.info(f"Registered model {model_name} version {model_version.version}")
                
                self.training_metadata["registered_model"] = {
                    "name": model_name,
                    "version": model_version.version,
                    "run_id": model_version.run_id
                }
        
        self.training_metadata["end_time"] = datetime.utcnow().isoformat()
        
        return self
    
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
            
        Returns:
            Export result dictionary
        """
        
        if self.best_pipeline is None:
            raise ValueError("No model trained yet")
        
        if sample_data is None:
            # Create synthetic sample data
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
            # Need sample output for PMML
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
        
        Args:
            challenger_pipeline: New model to test
            model_name: Name of the model in registry
            traffic_split: Percentage of traffic to route to challenger
            
        Returns:
            Test ID
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
            # No production model, promote challenger directly
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
    
    def predict(self, X: pd.DataFrame, 
                use_ab_test: bool = False,
                test_id: Optional[str] = None) -> np.ndarray:
        """
        Make predictions with optional A/B testing
        
        Args:
            X: Features to predict
            use_ab_test: Whether to use A/B testing
            test_id: Specific A/B test ID
            
        Returns:
            Predictions
        """
        
        if self.best_pipeline is None:
            raise ValueError("No model trained yet")
        
        if use_ab_test and test_id:
            # Route through A/B test
            model_type, version = self.ab_testing.route_prediction(test_id)
            
            # Load appropriate model version
            # This is simplified - real implementation would load from registry
            predictions = self.best_pipeline.predict(X)
            
            # Record result (simplified - would need actual success metric)
            self.ab_testing.record_result(test_id, model_type, True)
        else:
            predictions = self.best_pipeline.predict(X)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions"""
        
        if self.best_pipeline is None:
            raise ValueError("No model trained yet")
        
        if not hasattr(self.best_pipeline, 'predict_proba'):
            raise ValueError("Model doesn't support probability predictions")
        
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
                'training_time': result['training_time']
            }
            
            # Add metrics
            for metric_name, metric_value in result['metrics'].items():
                row[metric_name] = metric_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_pipeline(self, filepath: str) -> None:
        """Save best pipeline with MLflow tracking"""
        
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
                'training_time_seconds': result['training_time']
            }
            
            # Add all metrics
            for metric, value in result['metrics'].items():
                row[f'metric_{metric}'] = value
            
            # Add key hyperparameters
            for param, value in list(result['params'].items())[:5]:  # Top 5 params
                row[f'param_{param}'] = value
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # Add relative performance
        if len(df) > 0:
            best_score = df['cv_score'].iloc[0]
            df['relative_performance'] = (df['cv_score'] / best_score * 100).round(2)
        
        return df
