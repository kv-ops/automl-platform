"""
Model Selection - Sections modifiées pour intégrer l'apprentissage incrémental
==============================================================================
Dans le fichier automl_platform/model_selection.py, ajouter ces modifications :
"""

# ============= SECTION 1: Ajouter aux imports en début de fichier =============

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

# Import incremental learning models (AJOUTER CES LIGNES)
try:
    from river import linear_model as river_linear
    from river import tree as river_tree
    from river import ensemble as river_ensemble
    from river import neural_net as river_nn
    from river import naive_bayes as river_nb
    from sklearn.linear_model import SGDClassifier, SGDRegressor
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    INCREMENTAL_AVAILABLE = True
except ImportError:
    INCREMENTAL_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============= SECTION 2: Modifier la signature de get_available_models =============

def get_available_models(task: str = 'classification', 
                        include_neural: bool = False,
                        include_ensemble: bool = True,
                        include_timeseries: bool = False,
                        include_incremental: bool = False) -> Dict[str, Any]:  # AJOUTER include_incremental
    """Get all available models for the task including advanced models and incremental learning."""
    models = {}
    
    # Code existant pour les modèles sklearn...
    
    # Add incremental learning models if requested (AJOUTER CES LIGNES)
    if include_incremental and INCREMENTAL_AVAILABLE:
        models.update(_get_incremental_models(task))
    
    # Reste du code existant...
    
    logger.info(f"Found {len(models)} available models for {task}")
    return models


# ============= SECTION 3: Ajouter la nouvelle fonction _get_incremental_models =============

def _get_incremental_models(task: str) -> Dict[str, Any]:
    """Get incremental learning models for streaming/large data."""
    models = {}
    
    if not INCREMENTAL_AVAILABLE:
        return models
    
    if task == 'classification':
        # Sklearn SGD models (partial_fit support)
        models['SGDClassifier_Incremental'] = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            max_iter=1,
            tol=None,
            warm_start=True,
            learning_rate='adaptive',
            eta0=0.01
        )
        
        # Naive Bayes models (partial_fit support)
        models['MultinomialNB_Incremental'] = MultinomialNB(alpha=1.0)
        models['BernoulliNB_Incremental'] = BernoulliNB(alpha=1.0)
        
        # Neural network with warm start
        models['MLPClassifier_Incremental'] = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1,
            warm_start=True,
            early_stopping=False
        )
        
        # River models wrapped for sklearn compatibility
        try:
            from river.compat import convert_river_to_sklearn
            
            # Logistic regression
            river_lr = river_linear.LogisticRegression(
                optimizer=river_linear.optim.SGD(0.01),
                l2=0.01
            )
            models['River_LogisticRegression'] = convert_river_to_sklearn(river_lr)
            
            # Perceptron
            river_perceptron = river_linear.Perceptron(
                l2=0.01,
                clip_gradient=5
            )
            models['River_Perceptron'] = convert_river_to_sklearn(river_perceptron)
            
            # Hoeffding Tree
            river_ht = river_tree.HoeffdingTreeClassifier(
                grace_period=200,
                split_confidence=1e-5,
                max_depth=20,
                leaf_prediction='mc'
            )
            models['River_HoeffdingTree'] = convert_river_to_sklearn(river_ht)
            
            # Hoeffding Adaptive Tree
            river_hat = river_tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=200,
                split_confidence=1e-5,
                max_depth=20,
                leaf_prediction='mc'
            )
            models['River_HoeffdingAdaptiveTree'] = convert_river_to_sklearn(river_hat)
            
            # Adaptive Random Forest
            river_arf = river_ensemble.AdaptiveRandomForestClassifier(
                n_models=10,
                max_features='sqrt',
                lambda_value=6,
                grace_period=200,
                split_confidence=1e-5,
                seed=42
            )
            models['River_AdaptiveRandomForest'] = convert_river_to_sklearn(river_arf)
            
            # Streaming Random Patches
            river_srp = river_ensemble.SRPClassifier(
                n_models=10,
                drift_detector=True,
                warning_detector=True,
                seed=42
            )
            models['River_StreamingRandomPatches'] = convert_river_to_sklearn(river_srp)
            
            # Naive Bayes
            river_gnb = river_nb.GaussianNB()
            models['River_GaussianNB'] = convert_river_to_sklearn(river_gnb)
            
        except ImportError as e:
            logger.warning(f"Could not import River models: {e}")
            
    else:  # Regression
        # Sklearn SGD regressor
        models['SGDRegressor_Incremental'] = SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=0.0001,
            max_iter=1,
            tol=None,
            warm_start=True,
            learning_rate='adaptive',
            eta0=0.01
        )
        
        # Neural network regressor with warm start
        models['MLPRegressor_Incremental'] = MLPRegressor(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1,
            warm_start=True,
            early_stopping=False
        )
        
        # River models for regression
        try:
            from river.compat import convert_river_to_sklearn
            
            # Linear regression
            river_lr = river_linear.LinearRegression(
                optimizer=river_linear.optim.SGD(0.01),
                l2=0.01,
                intercept_lr=0.01
            )
            models['River_LinearRegression'] = convert_river_to_sklearn(river_lr)
            
            # Passive-Aggressive Regressor
            river_pa = river_linear.PARegressor(
                C=1.0,
                mode=2,
                eps=0.1
            )
            models['River_PARegressor'] = convert_river_to_sklearn(river_pa)
            
            # Hoeffding Tree Regressor
            river_htr = river_tree.HoeffdingTreeRegressor(
                grace_period=200,
                split_confidence=1e-5,
                max_depth=20,
                leaf_prediction='adaptive'
            )
            models['River_HoeffdingTreeRegressor'] = convert_river_to_sklearn(river_htr)
            
            # Hoeffding Adaptive Tree Regressor
            river_hatr = river_tree.HoeffdingAdaptiveTreeRegressor(
                grace_period=200,
                split_confidence=1e-5,
                max_depth=20,
                leaf_prediction='adaptive'
            )
            models['River_HoeffdingAdaptiveTreeRegressor'] = convert_river_to_sklearn(river_hatr)
            
            # Adaptive Random Forest Regressor
            river_arfr = river_ensemble.AdaptiveRandomForestRegressor(
                n_models=10,
                max_features='sqrt',
                lambda_value=6,
                grace_period=200,
                split_confidence=1e-5,
                seed=42
            )
            models['River_AdaptiveRandomForestRegressor'] = convert_river_to_sklearn(river_arfr)
            
        except ImportError as e:
            logger.warning(f"Could not import River regression models: {e}")
    
    return models


# ============= SECTION 4: Ajouter/Modifier les param_grids pour les modèles incrémentaux =============

def get_param_grid(model_name: str, search_type: str = 'random') -> Dict[str, List]:
    """Get hyperparameter grid for model with expanded options including incremental models."""
    
    # Comprehensive parameter grids including incremental models
    param_grids = {
        # ... (garder tous les param_grids existants) ...
        
        # Incremental Learning Models (AJOUTER CES GRIDS)
        'SGDClassifier_Incremental': {
            'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.00001, 0.0001, 0.001, 0.01],
            'l1_ratio': [0.15, 0.25, 0.5, 0.75],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'eta0': [0.001, 0.01, 0.1],
            'power_t': [0.25, 0.5],
            'average': [False, True]
        },
        'SGDRegressor_Incremental': {
            'loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.00001, 0.0001, 0.001, 0.01],
            'l1_ratio': [0.15, 0.25, 0.5, 0.75],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'eta0': [0.001, 0.01, 0.1],
            'power_t': [0.25, 0.5],
            'epsilon': [0.01, 0.1, 0.5]
        },
        'MultinomialNB_Incremental': {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
            'fit_prior': [True, False]
        },
        'BernoulliNB_Incremental': {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
            'binarize': [0.0, 0.5, 1.0],
            'fit_prior': [True, False]
        },
        'MLPClassifier_Incremental': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'batch_size': [32, 64, 128, 256],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'momentum': [0.9, 0.95, 0.99],
            'nesterovs_momentum': [True, False]
        },
        'MLPRegressor_Incremental': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'batch_size': [32, 64, 128, 256],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'momentum': [0.9, 0.95, 0.99]
        },
        'River_HoeffdingTree': {
            'grace_period': [50, 100, 200, 500],
            'split_confidence': [1e-7, 1e-5, 1e-3],
            'max_depth': [10, 20, 30, None],
            'tau': [0.05, 0.1, 0.2],
            'leaf_prediction': ['mc', 'nb', 'nba']
        },
        'River_AdaptiveRandomForest': {
            'n_models': [5, 10, 20],
            'max_features': ['sqrt', 'log2', 0.5, 0.7],
            'lambda_value': [1, 3, 6],
            'grace_period': [50, 100, 200],
            'split_confidence': [1e-7, 1e-5, 1e-3],
            'max_depth': [10, 20, None]
        }
    }
    
    # Return reduced grid for grid search
    if search_type == 'grid' and model_name in param_grids:
        grid = param_grids[model_name].copy()
        for param, values in grid.items():
            if len(values) > 3:
                if isinstance(values[0], (int, float)):
                    grid[param] = [values[0], values[len(values)//2], values[-1]]
                else:
                    grid[param] = values[:3]
        return grid
    
    return param_grids.get(model_name, {})
