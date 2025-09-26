"""
Model Selection Module for AutoML Platform
==========================================
Provides model selection, hyperparameter tuning, and cross-validation utilities.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import (
    StratifiedKFold, KFold, TimeSeriesSplit,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.dummy import DummyRegressor, DummyClassifier
import logging
import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = [
    'get_available_models',
    'get_param_grid', 
    'get_cv_splitter',
    'tune_model',
    'try_optuna'
]


def get_available_models(
    task: str = 'classification',
    include_neural: bool = False,
    include_ensemble: bool = True,
    include_timeseries: bool = False,
    include_incremental: bool = False
) -> Dict[str, Any]:
    """
    Get all available models for the specified task.
    
    Args:
        task: Task type ('classification', 'regression', 'timeseries')
        include_neural: Include neural network models
        include_ensemble: Include ensemble models
        include_timeseries: Include time series specific models
        include_incremental: Include incremental learning models
        
    Returns:
        Dictionary of model name to model instance
    """
    models = {}
    
    if task == 'classification':
        # Basic models
        models['LogisticRegression'] = LogisticRegression(max_iter=1000, random_state=42)
        models['DecisionTreeClassifier'] = DecisionTreeClassifier(random_state=42)
        models['GaussianNB'] = GaussianNB()
        models['KNeighborsClassifier'] = KNeighborsClassifier()
        models['SVC'] = SVC(probability=True, random_state=42)
        
        # Ensemble models
        if include_ensemble:
            models['RandomForestClassifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
            models['GradientBoostingClassifier'] = GradientBoostingClassifier(random_state=42)
            models['ExtraTreesClassifier'] = ExtraTreesClassifier(n_estimators=100, random_state=42)
            models['AdaBoostClassifier'] = AdaBoostClassifier(random_state=42)
            
            if XGBOOST_AVAILABLE:
                models['XGBClassifier'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            if LIGHTGBM_AVAILABLE:
                models['LGBMClassifier'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        # Neural networks
        if include_neural:
            models['MLPClassifier'] = MLPClassifier(max_iter=1000, random_state=42)
        
        # Incremental models
        if include_incremental:
            models['SGDClassifier'] = SGDClassifier(loss='log_loss', random_state=42)
            models['MultinomialNB'] = MultinomialNB()
            models['BernoulliNB'] = BernoulliNB()
            
    elif task == 'regression':
        # Basic models
        models['LinearRegression'] = LinearRegression()
        models['Ridge'] = Ridge(random_state=42)
        models['Lasso'] = Lasso(random_state=42)
        models['ElasticNet'] = ElasticNet(random_state=42)
        models['DecisionTreeRegressor'] = DecisionTreeRegressor(random_state=42)
        models['KNeighborsRegressor'] = KNeighborsRegressor()
        models['SVR'] = SVR()
        
        # Ensemble models
        if include_ensemble:
            models['RandomForestRegressor'] = RandomForestRegressor(n_estimators=100, random_state=42)
            models['GradientBoostingRegressor'] = GradientBoostingRegressor(random_state=42)
            models['ExtraTreesRegressor'] = ExtraTreesRegressor(n_estimators=100, random_state=42)
            models['AdaBoostRegressor'] = AdaBoostRegressor(random_state=42)
            
            if XGBOOST_AVAILABLE:
                models['XGBRegressor'] = xgb.XGBRegressor(random_state=42)
            if LIGHTGBM_AVAILABLE:
                models['LGBMRegressor'] = lgb.LGBMRegressor(random_state=42, verbose=-1)
        
        # Neural networks
        if include_neural:
            models['MLPRegressor'] = MLPRegressor(max_iter=1000, random_state=42)
        
        # Incremental models
        if include_incremental:
            models['SGDRegressor'] = SGDRegressor(random_state=42)
            
    elif task == 'timeseries':
        # For time series, we'll use regular regression models
        # In real implementation, would use ARIMA, Prophet, etc.
        models['Ridge'] = Ridge(random_state=42)
        models['LinearRegression'] = LinearRegression()
        
        if include_ensemble:
            models['RandomForestRegressor'] = RandomForestRegressor(n_estimators=100, random_state=42)
            models['GradientBoostingRegressor'] = GradientBoostingRegressor(random_state=42)
        
        if include_timeseries:
            # Placeholder for time series specific models
            models['DummyRegressor'] = DummyRegressor(strategy='mean')
    
    logger.info(f"Found {len(models)} available models for {task}")
    return models


def get_param_grid(model_name: str) -> Dict[str, List]:
    """
    Get hyperparameter grid for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of hyperparameters to search
    """
    param_grids = {
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        },
        'Ridge': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'Lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        'ElasticNet': {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.2, 0.5, 0.8]
        },
        'DecisionTreeClassifier': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'DecisionTreeRegressor': {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        },
        'RandomForestRegressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        },
        'GradientBoostingClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'SVC': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        },
        'SVR': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        },
        'KNeighborsClassifier': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'KNeighborsRegressor': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'MLPClassifier': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01]
        },
        'MLPRegressor': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01]
        },
        'XGBClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        },
        'XGBRegressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        },
        'LGBMClassifier': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 50, 100]
        },
        'LGBMRegressor': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 50, 100]
        },
        'SGDClassifier': {
            'loss': ['hinge', 'log_loss', 'modified_huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01]
        },
        'SGDRegressor': {
            'loss': ['squared_error', 'huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01]
        }
    }
    
    return param_grids.get(model_name, {})


def get_cv_splitter(
    task: str,
    n_splits: int = 5,
    random_state: Optional[int] = None
) -> Union[StratifiedKFold, KFold, TimeSeriesSplit]:
    """
    Get cross-validation splitter based on task type.
    
    Args:
        task: Task type ('classification', 'regression', 'timeseries')
        n_splits: Number of CV folds
        random_state: Random state for reproducibility
        
    Returns:
        CV splitter object
    """
    if task == 'classification':
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif task == 'timeseries':
        return TimeSeriesSplit(n_splits=n_splits)
    else:  # regression
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def tune_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List],
    cv: Any,
    scoring: str,
    n_iter: int = 10,
    random_state: int = 42
) -> Tuple[Any, Dict]:
    """
    Tune model hyperparameters using GridSearchCV or RandomizedSearchCV.
    
    Args:
        model: Model to tune
        X: Training features
        y: Training labels
        param_grid: Parameter grid to search
        cv: Cross-validation splitter
        scoring: Scoring metric
        n_iter: Number of iterations for random search
        random_state: Random state
        
    Returns:
        Tuple of (best model, best parameters)
    """
    if not param_grid:
        # No parameters to tune
        return model, {}
    
    # Calculate total number of combinations
    n_combinations = 1
    for values in param_grid.values():
        n_combinations *= len(values)
    
    # Use GridSearchCV for small grids, RandomizedSearchCV for large ones
    if n_combinations <= 20:
        search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
    else:
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=min(n_iter, n_combinations),
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            verbose=0
        )
    
    try:
        search.fit(X, y)
        return search.best_estimator_, search.best_params_
    except Exception as e:
        logger.warning(f"Hyperparameter tuning failed: {e}")
        return model, {}


def try_optuna(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    cv: Any,
    scoring: str,
    n_trials: int = 20,
    timeout: Optional[int] = None
) -> Tuple[Optional[Any], Dict]:
    """
    Try hyperparameter optimization using Optuna if available.
    
    Args:
        model_name: Name of the model to optimize
        X: Training features
        y: Training labels
        task: Task type
        cv: Cross-validation splitter
        scoring: Scoring metric
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (best model, best parameters) or (None, {})
    """
    if not OPTUNA_AVAILABLE:
        logger.debug("Optuna not available")
        return None, {}
    
    # Only support specific models for Optuna optimization
    supported_models = [
        'RandomForestClassifier', 'RandomForestRegressor',
        'GradientBoostingClassifier', 'GradientBoostingRegressor',
        'XGBClassifier', 'XGBRegressor',
        'LGBMClassifier', 'LGBMRegressor'
    ]
    
    if model_name not in supported_models:
        logger.debug(f"Optuna optimization not supported for {model_name}")
        return None, {}
    
    def objective(trial):
        params = {}
        
        if model_name in ['RandomForestClassifier', 'RandomForestRegressor']:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
            if model_name == 'RandomForestClassifier':
                model = RandomForestClassifier(**params, random_state=42)
            else:
                model = RandomForestRegressor(**params, random_state=42)
                
        elif model_name in ['GradientBoostingClassifier', 'GradientBoostingRegressor']:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
            if model_name == 'GradientBoostingClassifier':
                model = GradientBoostingClassifier(**params, random_state=42)
            else:
                model = GradientBoostingRegressor(**params, random_state=42)
        
        elif XGBOOST_AVAILABLE and model_name in ['XGBClassifier', 'XGBRegressor']:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
            if model_name == 'XGBClassifier':
                model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
            else:
                model = xgb.XGBRegressor(**params, random_state=42)
        
        elif LIGHTGBM_AVAILABLE and model_name in ['LGBMClassifier', 'LGBMRegressor']:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }
            if model_name == 'LGBMClassifier':
                model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            else:
                model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        else:
            return 0
        
        # Cross-validation
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return scores.mean()
    
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Create model with best parameters
        best_params = study.best_params
        
        if model_name == 'RandomForestClassifier':
            best_model = RandomForestClassifier(**best_params, random_state=42)
        elif model_name == 'RandomForestRegressor':
            best_model = RandomForestRegressor(**best_params, random_state=42)
        elif model_name == 'GradientBoostingClassifier':
            best_model = GradientBoostingClassifier(**best_params, random_state=42)
        elif model_name == 'GradientBoostingRegressor':
            best_model = GradientBoostingRegressor(**best_params, random_state=42)
        elif XGBOOST_AVAILABLE and model_name == 'XGBClassifier':
            best_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
        elif XGBOOST_AVAILABLE and model_name == 'XGBRegressor':
            best_model = xgb.XGBRegressor(**best_params, random_state=42)
        elif LIGHTGBM_AVAILABLE and model_name == 'LGBMClassifier':
            best_model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
        elif LIGHTGBM_AVAILABLE and model_name == 'LGBMRegressor':
            best_model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
        else:
            return None, {}
        
        best_model.fit(X, y)
        return best_model, best_params
        
    except Exception as e:
        logger.warning(f"Optuna optimization failed: {e}")
        return None, {}
