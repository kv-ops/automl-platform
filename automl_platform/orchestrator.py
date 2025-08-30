"""AutoML Orchestrator - main engine."""

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

from .data_prep import DataPreprocessor, handle_imbalance, validate_data
from .model_selection import (
    get_available_models, get_param_grid, get_cv_splitter,
    tune_model, try_optuna
)
from .metrics import calculate_metrics, detect_task
from .config import AutoMLConfig

logger = logging.getLogger(__name__)


class AutoMLOrchestrator:
    """Main AutoML orchestrator that runs the complete pipeline."""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.preprocessor = DataPreprocessor(config.to_dict())
        self.leaderboard = []
        self.best_pipeline = None
        self.task = None
        self.feature_importance = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            task: Optional[str] = None) -> 'AutoMLOrchestrator':
        """Run complete AutoML pipeline."""
        
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
                # IMPORTANT: Create fresh preprocessor for each model to avoid data leakage
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
                    
                    # Create a temporary preprocessor for Optuna optimization
                    # This will be refitted properly during cross-validation
                    temp_preprocessor = DataPreprocessor(self.config.to_dict())
                    
                    # Use a single train/test split for Optuna to avoid leakage
                    from sklearn.model_selection import train_test_split
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=0.2, random_state=self.config.random_state, 
                        stratify=y if self.task == 'classification' else None
                    )
                    
                    # Fit preprocessor only on training data
                    X_train_preprocessed = temp_preprocessor.fit_transform(X_train, y_train)
                    X_val_preprocessed = temp_preprocessor.transform(X_val)
                    
                    # Combine for Optuna (it will do its own CV)
                    X_preprocessed = np.vstack([X_train_preprocessed, X_val_preprocessed])
                    y_combined = pd.concat([y_train, y_val])
                    
                    tuned_model, params = try_optuna(
                        model_name, X_preprocessed, y_combined, self.task,
                        cv, scoring, n_trials=self.config.hpo_n_iter
                    )
                    
                    if tuned_model is not None:
                        # Update the model in the pipeline with tuned parameters
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
        
        return self
    
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
            'config': self.config.to_dict()
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
            # CRITICAL FIX: Use transform instead of fit_transform to avoid data leakage
            # The pipeline is already fitted, we just need to transform the data
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
