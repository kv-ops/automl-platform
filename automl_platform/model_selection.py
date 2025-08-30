"""Enhanced model selection with advanced algorithms and better HPO."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import (
    StratifiedKFold, KFold, TimeSeriesSplit,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def get_available_models(task: str = 'classification', 
                        include_neural: bool = False,
                        include_ensemble: bool = True) -> Dict[str, Any]:
    """Get all available models for the task including advanced models."""
    models = {}
    
    # Get sklearn models
    if task == 'classification':
        estimators = all_estimators(type_filter='classifier')
    else:
        estimators = all_estimators(type_filter='regressor')
    
    # Expanded list of safe models
    safe_models = [
        'LogisticRegression', 'RidgeClassifier', 'SGDClassifier',
        'Perceptron', 'PassiveAggressiveClassifier', 'LinearSVC',
        'SVC', 'NuSVC', 'KNeighborsClassifier', 'GaussianNB',
        'MultinomialNB', 'BernoulliNB', 'ComplementNB',
        'DecisionTreeClassifier', 'ExtraTreeClassifier',
        'RandomForestClassifier', 'ExtraTreesClassifier',
        'GradientBoostingClassifier', 'HistGradientBoostingClassifier',
        'AdaBoostClassifier', 'BaggingClassifier',
        'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
        'Lars', 'LassoLars', 'OrthogonalMatchingPursuit',
        'BayesianRidge', 'ARDRegression', 'HuberRegressor',
        'TheilSenRegressor', 'RANSACRegressor', 'SGDRegressor',
        'PassiveAggressiveRegressor', 'KNeighborsRegressor',
        'SVR', 'LinearSVR', 'DecisionTreeRegressor',
        'ExtraTreeRegressor', 'RandomForestRegressor',
        'ExtraTreesRegressor', 'GradientBoostingRegressor',
        'HistGradientBoostingRegressor', 'AdaBoostRegressor',
        'BaggingRegressor', 'DummyClassifier', 'DummyRegressor'
    ]
    
    # Instantiate sklearn models
    for name, EstimatorClass in estimators:
        if name in safe_models:
            try:
                model = _instantiate_sklearn_model(name, EstimatorClass)
                if model is not None:
                    models[name] = model
            except Exception as e:
                logger.debug(f"Could not instantiate {name}: {e}")
    
    # Add boosting libraries
    models.update(_get_boosting_models(task))
    
    # Add imbalanced-learn models for classification
    if task == 'classification' and include_ensemble:
        models.update(_get_imbalanced_models())
    
    # Add neural network models if requested
    if include_neural:
        models.update(_get_neural_models(task))
    
    # Add AutoML models
    models.update(_get_automl_models(task))
    
    logger.info(f"Found {len(models)} available models for {task}")
    return models


def _instantiate_sklearn_model(name: str, EstimatorClass):
    """Instantiate sklearn model with optimized parameters."""
    try:
        if 'SVC' in name or 'SVR' in name or 'NuSVC' in name:
            model = EstimatorClass(max_iter=1000, cache_size=500)
        elif 'SGD' in name:
            model = EstimatorClass(max_iter=1000, tol=1e-3, early_stopping=True)
        elif 'Logistic' in name:
            model = EstimatorClass(max_iter=1000, solver='lbfgs')
        elif 'RandomForest' in name or 'ExtraTrees' in name:
            model = EstimatorClass(n_estimators=100, n_jobs=-1, max_features='sqrt')
        elif 'GradientBoosting' in name:
            model = EstimatorClass(n_estimators=100, validation_fraction=0.2, n_iter_no_change=10)
        elif 'HistGradientBoosting' in name:
            model = EstimatorClass(max_iter=100, validation_fraction=0.2, n_iter_no_change=10)
        elif 'AdaBoost' in name:
            model = EstimatorClass(n_estimators=100, algorithm='SAMME' if 'Classifier' in name else 'SAMME.R')
        elif 'Bagging' in name:
            model = EstimatorClass(n_estimators=100, n_jobs=-1)
        else:
            model = EstimatorClass()
        return model
    except:
        return None


def _get_boosting_models(task: str) -> Dict[str, Any]:
    """Get boosting models (XGBoost, LightGBM, CatBoost)."""
    models = {}
    
    if task == 'classification':
        # XGBoost
        try:
            from xgboost import XGBClassifier
            models['XGBClassifier'] = XGBClassifier(
                n_estimators=100, 
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
                tree_method='auto',
                enable_categorical=True
            )
        except ImportError:
            pass
        
        # LightGBM
        try:
            from lightgbm import LGBMClassifier
            models['LGBMClassifier'] = LGBMClassifier(
                n_estimators=100,
                verbosity=-1,
                force_col_wise=True,
                categorical_feature='auto'
            )
        except ImportError:
            pass
        
        # CatBoost - ENHANCED
        try:
            from catboost import CatBoostClassifier
            models['CatBoostClassifier'] = CatBoostClassifier(
                iterations=100,
                verbose=False,
                allow_writing_files=False,
                task_type='CPU',
                auto_class_weights='Balanced'
            )
            # CatBoost with GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    models['CatBoostClassifier_GPU'] = CatBoostClassifier(
                        iterations=100,
                        verbose=False,
                        task_type='GPU',
                        devices='0'
                    )
            except:
                pass
        except ImportError:
            pass
            
    else:  # Regression
        # XGBoost
        try:
            from xgboost import XGBRegressor
            models['XGBRegressor'] = XGBRegressor(
                n_estimators=100,
                verbosity=0,
                tree_method='auto',
                enable_categorical=True
            )
        except ImportError:
            pass
        
        # LightGBM
        try:
            from lightgbm import LGBMRegressor
            models['LGBMRegressor'] = LGBMRegressor(
                n_estimators=100,
                verbosity=-1,
                force_col_wise=True,
                categorical_feature='auto'
            )
        except ImportError:
            pass
        
        # CatBoost - ENHANCED
        try:
            from catboost import CatBoostRegressor
            models['CatBoostRegressor'] = CatBoostRegressor(
                iterations=100,
                verbose=False,
                allow_writing_files=False,
                task_type='CPU'
            )
        except ImportError:
            pass
    
    return models


def _get_imbalanced_models() -> Dict[str, Any]:
    """Get models for imbalanced classification."""
    models = {}
    
    try:
        from imblearn.ensemble import (
            BalancedRandomForestClassifier,
            BalancedBaggingClassifier,
            RUSBoostClassifier,
            EasyEnsembleClassifier
        )
        
        models['BalancedRandomForestClassifier'] = BalancedRandomForestClassifier(
            n_estimators=100, n_jobs=-1, sampling_strategy='auto'
        )
        models['BalancedBaggingClassifier'] = BalancedBaggingClassifier(
            n_estimators=10, n_jobs=-1, sampling_strategy='auto'
        )
        models['RUSBoostClassifier'] = RUSBoostClassifier(
            n_estimators=100, algorithm='SAMME.R'
        )
        models['EasyEnsembleClassifier'] = EasyEnsembleClassifier(
            n_estimators=10, n_jobs=-1
        )
    except ImportError:
        pass
    
    return models


def _get_neural_models(task: str) -> Dict[str, Any]:
    """Get neural network models (TabNet, FT-Transformer, etc.)."""
    models = {}
    
    # TabNet - ENHANCED
    try:
        if task == 'classification':
            from pytorch_tabnet.tab_model import TabNetClassifier
            models['TabNetClassifier'] = TabNetClassifier(
                n_d=8, n_a=8,
                n_steps=3,
                gamma=1.3,
                n_independent=2,
                n_shared=2,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                mask_type='sparsemax',
                scheduler_params=dict(
                    mode="min",
                    patience=5,
                    min_lr=1e-5,
                    factor=0.9
                ),
                scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                seed=42,
                verbose=0
            )
        else:
            from pytorch_tabnet.tab_model import TabNetRegressor
            models['TabNetRegressor'] = TabNetRegressor(
                n_d=8, n_a=8,
                n_steps=3,
                gamma=1.3,
                n_independent=2,
                n_shared=2,
                lambda_sparse=1e-3,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                mask_type='sparsemax',
                seed=42,
                verbose=0
            )
    except ImportError:
        pass
    
    # FT-Transformer
    try:
        from tab_transformer_pytorch import FTTransformer
        # This would need wrapper class
        pass
    except ImportError:
        pass
    
    # SAINT (Self-Attention and Intersample Attention Transformer)
    try:
        # Would need implementation or wrapper
        pass
    except ImportError:
        pass
    
    # Neural Oblivious Decision Trees
    try:
        from nodepy import NODE
        # Would need wrapper
        pass
    except ImportError:
        pass
    
    return models


def _get_automl_models(task: str) -> Dict[str, Any]:
    """Get AutoML library models."""
    models = {}
    
    # AutoGluon models
    try:
        from autogluon.tabular import TabularPredictor
        # Would need wrapper
        pass
    except ImportError:
        pass
    
    # H2O AutoML
    try:
        import h2o
        from h2o.automl import H2OAutoML
        # Would need wrapper
        pass
    except ImportError:
        pass
    
    # FLAML
    try:
        from flaml import AutoML
        # Would need wrapper
        pass
    except ImportError:
        pass
    
    return models


def get_param_grid(model_name: str, search_type: str = 'random') -> Dict[str, List]:
    """Get hyperparameter grid for model with expanded options."""
    
    # Comprehensive parameter grids
    param_grids = {
        # Linear models
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'max_iter': [100, 500, 1000]
        },
        'Ridge': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        },
        
        # Tree-based models
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        'RandomForestRegressor': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        },
        
        # Gradient Boosting
        'GradientBoostingClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        },
        
        # XGBoost
        'XGBClassifier': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0, 0.01, 0.1, 1]
        },
        'XGBRegressor': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [0, 0.01, 0.1, 1]
        },
        
        # LightGBM
        'LGBMClassifier': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [20, 31, 50, 100],
            'max_depth': [-1, 5, 10, 20],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0, 0.01, 0.1]
        },
        'LGBMRegressor': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [20, 31, 50, 100],
            'max_depth': [-1, 5, 10, 20],
            'min_child_samples': [10, 20, 30],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [0, 0.01, 0.1]
        },
        
        # CatBoost - NEW
        'CatBoostClassifier': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7],
            'border_count': [32, 64, 128],
            'bagging_temperature': [0, 0.5, 1],
            'random_strength': [0, 0.5, 1]
        },
        'CatBoostRegressor': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7],
            'border_count': [32, 64, 128],
            'bagging_temperature': [0, 0.5, 1],
            'random_strength': [0, 0.5, 1]
        },
        
        # TabNet - NEW
        'TabNetClassifier': {
            'n_d': [8, 16, 32],
            'n_a': [8, 16, 32],
            'n_steps': [3, 4, 5],
            'gamma': [1.0, 1.3, 1.5],
            'n_independent': [1, 2, 3],
            'n_shared': [1, 2, 3],
            'lambda_sparse': [1e-4, 1e-3, 1e-2]
        },
        'TabNetRegressor': {
            'n_d': [8, 16, 32],
            'n_a': [8, 16, 32],
            'n_steps': [3, 4, 5],
            'gamma': [1.0, 1.3, 1.5],
            'n_independent': [1, 2, 3],
            'n_shared': [1, 2, 3],
            'lambda_sparse': [1e-4, 1e-3, 1e-2]
        },
        
        # SVM
        'SVC': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'degree': [2, 3, 4]
        },
        'SVR': {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.2]
        },
        
        # KNN
        'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        },
        'KNeighborsRegressor': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }
    }
    
    # Return reduced grid for grid search
    if search_type == 'grid' and model_name in param_grids:
        # Reduce parameter options for grid search
        grid = param_grids[model_name].copy()
        for param, values in grid.items():
            if len(values) > 3:
                # Take subset of values
                if isinstance(values[0], (int, float)):
                    grid[param] = [values[0], values[len(values)//2], values[-1]]
                else:
                    grid[param] = values[:3]
        return grid
    
    return param_grids.get(model_name, {})


def get_cv_splitter(task: str, n_splits: int = 5, random_state: int = 42):
    """Get appropriate CV splitter for task."""
    if task == 'classification':
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif task == 'regression':
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif task == 'timeseries':
        return TimeSeriesSplit(n_splits=n_splits)
    elif task == 'multilabel':
        from sklearn.model_selection import StratifiedKFold
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown task: {task}")


def tune_model(model: Any, X: np.ndarray, y: np.ndarray,
               param_grid: Dict[str, List], cv: Any,
               scoring: str, n_iter: int = 20,
               search_type: str = 'random') -> Tuple[Any, Dict]:
    """Enhanced hyperparameter tuning."""
    
    if not param_grid:
        return model, {}
    
    try:
        if search_type == 'random' or len(param_grid) > 100:
            # Use RandomizedSearchCV for large parameter spaces
            search = RandomizedSearchCV(
                model, param_grid,
                n_iter=min(n_iter, 50),
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=42,
                error_score='raise',
                return_train_score=True
            )
        else:
            # Use GridSearchCV for small parameter spaces
            from sklearn.model_selection import GridSearchCV
            search = GridSearchCV(
                model, param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score='raise',
                return_train_score=True
            )
        
        search.fit(X, y)
        
        # Log best parameters
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_
        
    except Exception as e:
        logger.warning(f"Tuning failed: {e}")
        return model, {}


def try_optuna(model_name: str, X: np.ndarray, y: np.ndarray,
               task: str, cv: Any, scoring: str,
               n_trials: int = 50,
               timeout: int = 300) -> Tuple[Any, Dict]:
    """Enhanced Optuna HPO with more models and better search spaces."""
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            params = _get_optuna_params(trial, model_name)
            
            if params is None:
                return 0
            
            # Create model
            model = _create_model_from_params(model_name, params, task)
            
            if model is None:
                return 0
            
            # Cross-validation with pruning
            from sklearn.model_selection import cross_val_score
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Evaluate
                if hasattr(model, 'predict_proba') and scoring == 'roc_auc':
                    from sklearn.metrics import roc_auc_score
                    y_pred = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred)
                else:
                    score = model.score(X_val, y_val)
                
                scores.append(score)
                
                # Report intermediate value for pruning
                trial.report(np.mean(scores), fold)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        # Create study with pruning
        sampler = TPESampler(seed=42, n_startup_trials=10)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1,
            show_progress_bar=False
        )
        
        # Get best params and recreate model
        best_params = study.best_params
        best_model = _create_model_from_params(model_name, best_params, task)
        
        logger.info(f"Optuna best score: {study.best_value:.4f}")
        logger.info(f"Optuna best params: {best_params}")
        
        return best_model, best_params
        
    except Exception as e:
        logger.debug(f"Optuna optimization failed: {e}")
        return None, {}


def _get_optuna_params(trial, model_name: str) -> Optional[Dict]:
    """Get Optuna parameter suggestions for each model."""
    
    if 'RandomForest' in model_name:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'n_jobs': -1
        }
    
    elif 'GradientBoosting' in model_name:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
    
    elif 'XGB' in model_name:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
        }
    
    elif 'LGBM' in model_name:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', -1, 30),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
        }
    
    elif 'CatBoost' in model_name:
        return {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 1)
        }
    
    elif 'TabNet' in model_name:
        return {
            'n_d': trial.suggest_int('n_d', 8, 64),
            'n_a': trial.suggest_int('n_a', 8, 64),
            'n_steps': trial.suggest_int('n_steps', 3, 10),
            'gamma': trial.suggest_float('gamma', 1.0, 2.0),
            'n_independent': trial.suggest_int('n_independent', 1, 5),
            'n_shared': trial.suggest_int('n_shared', 1, 5),
            'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-2, log=True)
        }
    
    elif 'SVC' in model_name or 'SVR' in model_name:
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        params = {
            'C': trial.suggest_float('C', 0.01, 100, log=True),
            'kernel': kernel
        }
        
        if kernel in ['poly', 'rbf', 'sigmoid']:
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
        
        if kernel == 'poly':
            params['degree'] = trial.suggest_int('degree', 2, 5)
        
        if 'SVR' in model_name:
            params['epsilon'] = trial.suggest_float('epsilon', 0.01, 1.0)
        
        return params
    
    elif 'Logistic' in model_name:
        return {
            'C': trial.suggest_float('C', 0.001, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 100, 1000)
        }
    
    elif 'KNeighbors' in model_name:
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
            'p': trial.suggest_int('p', 1, 3)
        }
    
    return None


def _create_model_from_params(model_name: str, params: Dict, task: str):
    """Create model instance from parameters."""
    
    try:
        if task == 'classification':
            if 'XGB' in model_name:
                from xgboost import XGBClassifier
                return XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss', verbosity=0)
            elif 'LGBM' in model_name:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(**params, verbosity=-1, force_col_wise=True)
            elif 'CatBoost' in model_name:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(**params, verbose=False, allow_writing_files=False)
            elif 'TabNet' in model_name:
                from pytorch_tabnet.tab_model import TabNetClassifier
                return TabNetClassifier(**params, verbose=0)
            elif 'RandomForest' in model_name:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**params)
            elif 'GradientBoosting' in model_name:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(**params)
            elif 'SVC' in model_name:
                from sklearn.svm import SVC
                return SVC(**params)
            elif 'Logistic' in model_name:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(**params)
            elif 'KNeighbors' in model_name:
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier(**params)
                
        else:  # regression
            if 'XGB' in model_name:
                from xgboost import XGBRegressor
                return XGBRegressor(**params, verbosity=0)
            elif 'LGBM' in model_name:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(**params, verbosity=-1, force_col_wise=True)
            elif 'CatBoost' in model_name:
                from catboost import CatBoostRegressor
                return CatBoostRegressor(**params, verbose=False, allow_writing_files=False)
            elif 'TabNet' in model_name:
                from pytorch_tabnet.tab_model import TabNetRegressor
                return TabNetRegressor(**params, verbose=0)
            elif 'RandomForest' in model_name:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(**params)
            elif 'GradientBoosting' in model_name:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(**params)
            elif 'SVR' in model_name:
                from sklearn.svm import SVR
                return SVR(**params)
            elif 'KNeighbors' in model_name:
                from sklearn.neighbors import KNeighborsRegressor
                return KNeighborsRegressor(**params)
                
    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {e}")
        
    return None
