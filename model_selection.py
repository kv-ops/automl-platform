"""Model selection with exhaustive model testing."""

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


def get_available_models(task: str = 'classification') -> Dict[str, Any]:
    """Get all available models for the task."""
    models = {}
    
    # Get sklearn models
    if task == 'classification':
        estimators = all_estimators(type_filter='classifier')
    else:
        estimators = all_estimators(type_filter='regressor')
    
    # Filter and instantiate safe models
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
    
    for name, EstimatorClass in estimators:
        if name in safe_models:
            try:
                # Instantiate with minimal params
                if 'SVC' in name or 'SVR' in name or 'NuSVC' in name:
                    model = EstimatorClass(max_iter=1000)
                elif 'SGD' in name:
                    model = EstimatorClass(max_iter=1000, tol=1e-3)
                elif 'Logistic' in name:
                    model = EstimatorClass(max_iter=1000)
                elif 'RandomForest' in name or 'ExtraTrees' in name:
                    model = EstimatorClass(n_estimators=100, n_jobs=-1)
                elif 'GradientBoosting' in name:
                    model = EstimatorClass(n_estimators=100)
                elif 'HistGradientBoosting' in name:
                    model = EstimatorClass(max_iter=100)
                else:
                    model = EstimatorClass()
                    
                models[name] = model
            except Exception as e:
                logger.debug(f"Could not instantiate {name}: {e}")
    
    # Try external boosting libraries
    if task == 'classification':
        try:
            from xgboost import XGBClassifier
            models['XGBClassifier'] = XGBClassifier(
                n_estimators=100, use_label_encoder=False, 
                eval_metric='logloss', verbosity=0
            )
        except ImportError:
            pass
        
        try:
            from lightgbm import LGBMClassifier
            models['LGBMClassifier'] = LGBMClassifier(
                n_estimators=100, verbosity=-1
            )
        except ImportError:
            pass
        
        try:
            from catboost import CatBoostClassifier
            models['CatBoostClassifier'] = CatBoostClassifier(
                iterations=100, verbose=False
            )
        except ImportError:
            pass
        
        # Imbalanced-learn models
        try:
            from imblearn.ensemble import (
                BalancedRandomForestClassifier, BalancedBaggingClassifier,
                RUSBoostClassifier, EasyEnsembleClassifier
            )
            models['BalancedRandomForestClassifier'] = BalancedRandomForestClassifier(
                n_estimators=100, n_jobs=-1
            )
            models['BalancedBaggingClassifier'] = BalancedBaggingClassifier(
                n_estimators=10, n_jobs=-1
            )
        except ImportError:
            pass
    
    else:  # Regression
        try:
            from xgboost import XGBRegressor
            models['XGBRegressor'] = XGBRegressor(
                n_estimators=100, verbosity=0
            )
        except ImportError:
            pass
        
        try:
            from lightgbm import LGBMRegressor
            models['LGBMRegressor'] = LGBMRegressor(
                n_estimators=100, verbosity=-1
            )
        except ImportError:
            pass
        
        try:
            from catboost import CatBoostRegressor
            models['CatBoostRegressor'] = CatBoostRegressor(
                iterations=100, verbose=False
            )
        except ImportError:
            pass
    
    logger.info(f"Found {len(models)} available models for {task}")
    return models


def get_param_grid(model_name: str) -> Dict[str, List]:
    """Get hyperparameter grid for model."""
    
    # Minimal grids for speed
    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
        },
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
        },
        'RandomForestRegressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
        },
        'GradientBoostingClassifier': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
        },
        'GradientBoostingRegressor': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
        },
        'XGBClassifier': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6],
        },
        'XGBRegressor': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6],
        },
        'LGBMClassifier': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50],
        },
        'LGBMRegressor': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [31, 50],
        },
        'SVC': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
        },
        'SVR': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
        },
    }
    
    return param_grids.get(model_name, {})


def get_cv_splitter(task: str, n_splits: int = 5, random_state: int = 42):
    """Get appropriate CV splitter for task."""
    if task == 'classification':
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif task == 'regression':
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif task == 'timeseries':
        return TimeSeriesSplit(n_splits=n_splits)
    else:
        raise ValueError(f"Unknown task: {task}")


def tune_model(model: Any, X: np.ndarray, y: np.ndarray,
               param_grid: Dict[str, List], cv: Any,
               scoring: str, n_iter: int = 10) -> Tuple[Any, Dict]:
    """Tune model hyperparameters."""
    
    if not param_grid:
        # No tuning needed
        return model, {}
    
    try:
        # Use RandomizedSearchCV for efficiency
        if len(param_grid) > 0 and n_iter > 0:
            search = RandomizedSearchCV(
                model, param_grid, n_iter=min(n_iter, 10),
                cv=cv, scoring=scoring, n_jobs=-1,
                random_state=42, error_score='raise'
            )
            search.fit(X, y)
            return search.best_estimator_, search.best_params_
    except Exception as e:
        logger.debug(f"Tuning failed: {e}")
    
    # Fallback to default model
    return model, {}


def try_optuna(model_name: str, X: np.ndarray, y: np.ndarray,
               task: str, cv: Any, scoring: str,
               n_trials: int = 20) -> Tuple[Any, Dict]:
    """Try Optuna for HPO if available."""
    try:
        import optuna
        from optuna.samplers import TPESampler
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            # Get params based on model
            params = {}
            
            if 'RandomForest' in model_name:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'n_jobs': -1
                }
            elif 'GradientBoosting' in model_name:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                }
            elif 'XGB' in model_name:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                }
            elif 'LGBM' in model_name:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                }
            else:
                return 0  # Skip optuna for this model
            
            # Create and evaluate model
            from sklearn.model_selection import cross_val_score
            
            if task == 'classification':
                if 'XGB' in model_name:
                    from xgboost import XGBClassifier
                    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
                elif 'LGBM' in model_name:
                    from lightgbm import LGBMClassifier
                    model = LGBMClassifier(**params, verbosity=-1)
                elif 'RandomForest' in model_name:
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(**params)
                elif 'GradientBoosting' in model_name:
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(**params)
                else:
                    return 0
            else:  # regression
                if 'XGB' in model_name:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(**params)
                elif 'LGBM' in model_name:
                    from lightgbm import LGBMRegressor
                    model = LGBMRegressor(**params, verbosity=-1)
                elif 'RandomForest' in model_name:
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(**params)
                elif 'GradientBoosting' in model_name:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(**params)
                else:
                    return 0
            
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()
        
        # Create study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=60)  # 1 minute timeout
        
        # Get best params and recreate model
        best_params = study.best_params
        
        if task == 'classification':
            if 'XGB' in model_name:
                from xgboost import XGBClassifier
                best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
            elif 'LGBM' in model_name:
                from lightgbm import LGBMClassifier
                best_model = LGBMClassifier(**best_params, verbosity=-1)
            elif 'RandomForest' in model_name:
                from sklearn.ensemble import RandomForestClassifier
                best_model = RandomForestClassifier(**best_params, n_jobs=-1)
            elif 'GradientBoosting' in model_name:
                from sklearn.ensemble import GradientBoostingClassifier
                best_model = GradientBoostingClassifier(**best_params)
        else:
            if 'XGB' in model_name:
                from xgboost import XGBRegressor
                best_model = XGBRegressor(**best_params)
            elif 'LGBM' in model_name:
                from lightgbm import LGBMRegressor
                best_model = LGBMRegressor(**best_params, verbosity=-1)
            elif 'RandomForest' in model_name:
                from sklearn.ensemble import RandomForestRegressor
                best_model = RandomForestRegressor(**best_params, n_jobs=-1)
            elif 'GradientBoosting' in model_name:
                from sklearn.ensemble import GradientBoostingRegressor
                best_model = GradientBoostingRegressor(**best_params)
        
        return best_model, best_params
        
    except Exception as e:
        logger.debug(f"Optuna not available or failed: {e}")
        return None, {}
