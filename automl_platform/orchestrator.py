"""AutoML Orchestrator with Scheduler Integration."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer
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

# Import scheduler components
from .scheduler import (
    SchedulerFactory, JobRequest, JobStatus, QueueType
)
from .api.billing import BillingManager, PlanType

logger = logging.getLogger(__name__)


class AutoMLOrchestrator:
    """Main AutoML orchestrator with scheduler integration."""
    
    def __init__(self, config: AutoMLConfig, 
                 scheduler=None, 
                 billing_manager=None,
                 async_mode: bool = False):
        """
        Initialize orchestrator with optional scheduler for distributed execution.
        
        Args:
            config: AutoML configuration
            scheduler: Optional scheduler instance for distributed execution
            billing_manager: Optional billing manager for quota management
            async_mode: If True, submit jobs to scheduler instead of running locally
        """
        self.config = config
        self.preprocessor = DataPreprocessor(config.to_dict())
        self.leaderboard = []
        self.best_pipeline = None
        self.task = None
        self.feature_importance = {}
        
        # Scheduler and billing integration
        self.async_mode = async_mode
        self.scheduler = scheduler
        self.billing_manager = billing_manager
        
        # Initialize scheduler if async mode and not provided
        if self.async_mode and not self.scheduler:
            self.scheduler = SchedulerFactory.create_scheduler(
                config, 
                billing_manager
            )
        
        # Job tracking
        self.current_job_id = None
        self.job_history = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            task: Optional[str] = None,
            use_gpu: bool = False,
            priority: str = "default") -> 'AutoMLOrchestrator':
        """
        Run complete AutoML pipeline with optional distributed execution.
        
        Args:
            X: Training features
            y: Training labels
            task: Task type (classification/regression/auto)
            use_gpu: Whether to use GPU for training
            priority: Job priority (default/high/low)
        """
        
        # Check if we should run async through scheduler
        if self.async_mode and self.scheduler:
            return self._fit_async(X, y, task, use_gpu, priority)
        else:
            return self._fit_local(X, y, task)
    
    def _fit_async(self, X: pd.DataFrame, y: pd.Series, 
                   task: Optional[str], 
                   use_gpu: bool,
                   priority: str) -> 'AutoMLOrchestrator':
        """Submit training job to scheduler for distributed execution."""
        
        # Determine queue type based on GPU and priority
        if use_gpu:
            queue_type = QueueType.GPU_TRAINING
        elif priority == "high":
            queue_type = QueueType.CPU_PRIORITY
        else:
            queue_type = QueueType.CPU_DEFAULT
        
        # Get plan type from billing manager or config
        plan_type = PlanType.FREE.value
        if self.billing_manager:
            subscription = self.billing_manager.get_subscription(self.config.tenant_id)
            if subscription:
                plan_type = subscription['plan']
        
        # Create job request
        job_request = JobRequest(
            tenant_id=self.config.tenant_id,
            user_id=self.config.user_id or "anonymous",
            plan_type=plan_type,
            task_type="train",
            queue_type=queue_type,
            payload={
                "X": X.to_dict('records'),
                "y": y.tolist(),
                "task": task,
                "config": self.config.to_dict()
            },
            estimated_memory_gb=self._estimate_memory(X),
            estimated_time_minutes=self._estimate_time(X, y),
            requires_gpu=use_gpu,
            num_gpus=1 if use_gpu else 0
        )
        
        # Submit job to scheduler
        self.current_job_id = self.scheduler.submit_job(job_request)
        self.job_history.append(self.current_job_id)
        
        logger.info(f"Training job submitted: {self.current_job_id}")
        logger.info(f"Queue: {queue_type.queue_name}, GPU: {use_gpu}")
        
        return self
    
    def _fit_local(self, X: pd.DataFrame, y: pd.Series, 
                   task: Optional[str]) -> 'AutoMLOrchestrator':
        """Original local training implementation."""
        
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
        
        # Check quotas if billing manager available
        if self.billing_manager:
            if not self.billing_manager.check_limits(
                self.config.tenant_id, 
                'models', 
                1
            ):
                raise ValueError("Model quota exceeded. Please upgrade your plan.")
        
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
        for model_name, base_model in models.items():
            logger.info(f"Testing {model_name}")
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
                
                # Try Optuna first for important models
                if model_name in ['RandomForestClassifier', 'RandomForestRegressor',
                                 'XGBClassifier', 'XGBRegressor', 
                                 'LGBMClassifier', 'LGBMRegressor',
                                 'GradientBoostingClassifier', 'GradientBoostingRegressor']:
                    
                    temp_preprocessor = DataPreprocessor(self.config.to_dict())
                    
                    from sklearn.model_selection import train_test_split
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=self.config.random_state, 
                        stratify=y if self.task == 'classification' else None
                    )
                    
                    X_train_preprocessed = temp_preprocessor.fit_transform(X_train, y_train)
                    X_val_preprocessed = temp_preprocessor.transform(X_val)
                    
                    X_preprocessed = np.vstack([X_train_preprocessed, X_val_preprocessed])
                    y_combined = pd.concat([y_train, y_val])
                    
                    tuned_model, params = try_optuna(
                        model_name, X_preprocessed, y_combined, self.task,
                        cv, scoring, n_trials=self.config.hpo_n_iter
                    )
                    
                    if tuned_model is not None:
                        base_model.set_params(**params)
                        pipeline.set_params(model=base_model)
                    else:
                        # Fall back to grid search
                        param_grid = get_param_grid(model_name)
                        if param_grid:
                            param_grid = {f'model__{k}': v for k, v in param_grid.items()}
                            tuned_model, params = tune_model(
                                pipeline, X, y, param_grid, cv, scoring,
                                self.config.hpo_n_iter
                            )
                            if params:
                                pipeline = tuned_model
                else:
                    # Simple grid search for other models
                    param_grid = get_param_grid(model_name)
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
                
                # Cross-validate with fresh pipeline to avoid data leakage
                scores = cross_val_score(pipeline, X, y, cv=cv, 
                                        scoring=scoring, n_jobs=-1)
                
                # Fit final model on all data for metrics calculation
                pipeline.fit(X, y)
                y_pred = pipeline.predict(X)
                
                if self.task == 'classification' and hasattr(pipeline, 'predict_proba'):
                    y_proba = pipeline.predict_proba(X)
                else:
                    y_proba = None
                
                metrics = calculate_metrics(y, y_pred, y_proba, self.task)
                
                # Add to leaderboard
                result = {
                    'model': model_name,
                    'cv_score': scores.mean(),
                    'cv_std': scores.std(),
                    'metrics': metrics,
                    'params': params if 'params' in locals() else {},
                    'training_time': time.time() - start_time,
                    'pipeline': pipeline
                }
                
                self.leaderboard.append(result)
                
                logger.info(f"{model_name}: CV Score = {scores.mean():.4f} (+/- {scores.std():.4f})")
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {e}")
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
            
            # Track model count if billing manager available
            if self.billing_manager:
                self.billing_manager.increment_model_count(
                    self.config.tenant_id,
                    'gpu' if self._is_gpu_model(self.best_pipeline) else 'standard'
                )
        
        return self
    
    def get_job_status(self) -> Optional[Dict]:
        """Get status of current async job."""
        
        if not self.current_job_id or not self.scheduler:
            return None
        
        job = self.scheduler.get_job_status(self.current_job_id)
        
        if job:
            return {
                "job_id": job.job_id,
                "status": job.status.value,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error": job.error_message,
                "queue": job.queue_type.queue_name
            }
        
        return None
    
    def wait_for_completion(self, timeout: int = 3600) -> bool:
        """Wait for async job to complete."""
        
        if not self.current_job_id or not self.scheduler:
            return True
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job = self.scheduler.get_job_status(self.current_job_id)
            
            if not job:
                return False
            
            if job.status == JobStatus.COMPLETED:
                # Load results
                self._load_async_results(job)
                return True
            
            elif job.status == JobStatus.FAILED:
                logger.error(f"Job failed: {job.error_message}")
                return False
            
            elif job.status == JobStatus.CANCELLED:
                logger.warning("Job was cancelled")
                return False
            
            time.sleep(5)  # Check every 5 seconds
        
        logger.error("Timeout waiting for job completion")
        return False
    
    def _load_async_results(self, job: JobRequest):
        """Load results from completed async job."""
        
        if job.result:
            self.leaderboard = job.result.get('leaderboard', [])
            self.task = job.result.get('task')
            self.feature_importance = job.result.get('feature_importance', {})
            
            # Load best pipeline if available
            if job.result.get('best_pipeline_path'):
                self.load_pipeline(job.result['best_pipeline_path'])
    
    def _estimate_memory(self, X: pd.DataFrame) -> float:
        """Estimate memory requirements in GB."""
        
        # Rough estimation based on data size
        memory_bytes = X.memory_usage(deep=True).sum()
        
        # Account for model training overhead (10x data size)
        estimated_gb = (memory_bytes * 10) / (1024 ** 3)
        
        # Add base memory requirement
        return max(1.0, estimated_gb + 0.5)
    
    def _estimate_time(self, X: pd.DataFrame, y: pd.Series) -> int:
        """Estimate training time in minutes."""
        
        n_samples = len(X)
        n_features = X.shape[1]
        n_models = len(self.config.algorithms) if self.config.algorithms != ['all'] else 10
        
        # Rough estimation formula
        base_time = 1  # minute
        sample_factor = n_samples / 1000  # 1 minute per 1000 samples
        feature_factor = n_features / 50  # 1 minute per 50 features
        model_factor = n_models * 2  # 2 minutes per model
        
        estimated_minutes = base_time + sample_factor + feature_factor + model_factor
        
        # Account for HPO
        if self.config.hpo_n_iter > 10:
            estimated_minutes *= (self.config.hpo_n_iter / 10)
        
        return max(5, int(estimated_minutes))
    
    def _is_gpu_model(self, model) -> bool:
        """Check if model requires GPU."""
        
        gpu_models = [
            'XGBClassifier', 'XGBRegressor',
            'LGBMClassifier', 'LGBMRegressor',
            'CatBoostClassifier', 'CatBoostRegressor'
        ]
        
        model_name = type(model).__name__
        if hasattr(model, 'named_steps'):
            # It's a pipeline
            model_name = type(model.named_steps.get('model', model)).__name__
        
        return model_name in gpu_models
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.best_pipeline is None:
            raise ValueError("No model trained yet")
        return self.best_pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions."""
        if self.best_pipeline is None:
            raise ValueError("No model trained yet")
        if not hasattr(self.best_pipeline, 'predict_proba'):
            raise ValueError("Model doesn't support probability predictions")
        return self.best_pipeline.predict_proba(X)
    
    def get_leaderboard(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """Get leaderboard as DataFrame."""
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
        """Save best pipeline."""
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
            'config': self.config.to_dict(),
            'job_id': self.current_job_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        metadata_path = filepath.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """Load pipeline."""
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
        
        logger.info(f"Pipeline loaded from {filepath}")
    
    def _calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calculate feature importance using permutation."""
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
    
    def explain_predictions(self, X: pd.DataFrame, indices: Optional[List[int]] = None) -> Dict:
        """Explain predictions using SHAP or LIME."""
        try:
            import shap
            
            # Use SHAP
            X_transformed = self.best_pipeline.named_steps['preprocessor'].transform(X)
            model = self.best_pipeline.named_steps['model']
            
            # Tree explainer for tree-based models
            if hasattr(model, 'tree_') or 'Tree' in type(model).__name__ or 'Forest' in type(model).__name__:
                explainer = shap.TreeExplainer(model)
            else:
                # Kernel explainer for others
                explainer = shap.KernelExplainer(model.predict, X_transformed[:100])
            
            if indices is None:
                shap_values = explainer.shap_values(X_transformed)
            else:
                shap_values = explainer.shap_values(X_transformed[indices])
            
            return {
                'method': 'shap',
                'values': shap_values,
                'expected_value': explainer.expected_value
            }
            
        except ImportError:
            logger.warning("SHAP not available")
            
            try:
                import lime
                import lime.lime_tabular
                
                # Use LIME
                X_transformed = self.best_pipeline.named_steps['preprocessor'].transform(X)
                model = self.best_pipeline.named_steps['model']
                
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_transformed,
                    mode='classification' if self.task == 'classification' else 'regression'
                )
                
                explanations = []
                indices = indices or list(range(min(5, len(X))))
                
                for idx in indices:
                    if self.task == 'classification' and hasattr(model, 'predict_proba'):
                        exp = explainer.explain_instance(
                            X_transformed[idx], model.predict_proba
                        )
                    else:
                        exp = explainer.explain_instance(
                            X_transformed[idx], model.predict
                        )
                    explanations.append(exp.as_list())
                
                return {
                    'method': 'lime',
                    'explanations': explanations
                }
                
            except ImportError:
                logger.warning("LIME not available")
                return {
                    'method': 'feature_importance',
                    'importance': self.feature_importance
                }
