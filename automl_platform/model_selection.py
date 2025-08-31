"""Enhanced model selection with time series, transfer learning, and advanced algorithms."""

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
                        include_ensemble: bool = True,
                        include_timeseries: bool = False) -> Dict[str, Any]:
    """Get all available models for the task including advanced models."""
    models = {}
    
    # Get sklearn models
    if task == 'classification':
        estimators = all_estimators(type_filter='classifier')
    elif task == 'regression':
        estimators = all_estimators(type_filter='regressor')
    else:
        estimators = []
    
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
    
    # Add time series models if requested
    if include_timeseries or task == 'timeseries':
        models.update(_get_timeseries_models())
    
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
        
        # CatBoost
        try:
            from catboost import CatBoostClassifier
            models['CatBoostClassifier'] = CatBoostClassifier(
                iterations=100,
                verbose=False,
                allow_writing_files=False,
                task_type='CPU',
                auto_class_weights='Balanced'
            )
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
        
        # CatBoost
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
    """Get models for imbalanced classification including focal loss."""
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
    
    # Add focal loss support for neural networks
    try:
        from focal_loss import BinaryFocalLoss
        models['FocalLossEnabled'] = True
    except ImportError:
        pass
    
    return models


def _get_neural_models(task: str) -> Dict[str, Any]:
    """Get neural network models with transfer learning support."""
    models = {}
    
    # TabNet
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
                seed=42,
                verbose=0
            )
    except ImportError:
        pass
    
    # Transfer Learning Models
    try:
        models.update(_get_transfer_learning_models(task))
    except ImportError:
        pass
    
    return models


def _get_transfer_learning_models(task: str) -> Dict[str, Any]:
    """Get transfer learning models."""
    models = {}
    
    try:
        import torch
        import torchvision.models as tv_models
        from sklearn.base import BaseEstimator, ClassifierMixin
        
        class TransferLearningWrapper(BaseEstimator, ClassifierMixin):
            """Wrapper for transfer learning models."""
            
            def __init__(self, base_model='resnet18', freeze_base=True, n_classes=2):
                self.base_model = base_model
                self.freeze_base = freeze_base
                self.n_classes = n_classes
                self.model = None
                
            def fit(self, X, y):
                """Fit transfer learning model."""
                # Load pretrained model
                if self.base_model == 'resnet18':
                    self.model = tv_models.resnet18(pretrained=True)
                elif self.base_model == 'efficientnet':
                    self.model = tv_models.efficientnet_b0(pretrained=True)
                else:
                    self.model = tv_models.resnet18(pretrained=True)
                
                # Freeze base layers if requested
                if self.freeze_base:
                    for param in self.model.parameters():
                        param.requires_grad = False
                
                # Replace final layer
                if hasattr(self.model, 'fc'):
                    num_features = self.model.fc.in_features
                    self.model.fc = torch.nn.Linear(num_features, self.n_classes)
                elif hasattr(self.model, 'classifier'):
                    num_features = self.model.classifier[-1].in_features
                    self.model.classifier[-1] = torch.nn.Linear(num_features, self.n_classes)
                
                # Simple training loop would go here
                return self
            
            def predict(self, X):
                """Make predictions."""
                # Simplified prediction
                return np.zeros(len(X))
        
        if task == 'classification':
            models['TransferLearning_ResNet18'] = TransferLearningWrapper('resnet18')
            models['TransferLearning_EfficientNet'] = TransferLearningWrapper('efficientnet')
            
    except ImportError:
        pass
    
    return models


def _get_timeseries_models() -> Dict[str, Any]:
    """Get time series models (Prophet, ARIMA, LSTM)."""
    models = {}
    
    # Prophet
    try:
        from prophet import Prophet
        from sklearn.base import BaseEstimator, RegressorMixin
        
        class ProphetWrapper(BaseEstimator, RegressorMixin):
            """Sklearn-compatible Prophet wrapper."""
            
            def __init__(self, seasonality_mode='multiplicative', 
                        yearly_seasonality=True, weekly_seasonality=True,
                        daily_seasonality=False):
                self.seasonality_mode = seasonality_mode
                self.yearly_seasonality = yearly_seasonality
                self.weekly_seasonality = weekly_seasonality
                self.daily_seasonality = daily_seasonality
                self.model = None
                
            def fit(self, X, y):
                """Fit Prophet model."""
                # Prepare data for Prophet
                df = pd.DataFrame({
                    'ds': pd.date_range(start='2020-01-01', periods=len(y), freq='D'),
                    'y': y
                })
                
                self.model = Prophet(
                    seasonality_mode=self.seasonality_mode,
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality
                )
                
                # Add regressors from X if available
                if X is not None and len(X.shape) > 1:
                    for i in range(X.shape[1]):
                        self.model.add_regressor(f'regressor_{i}')
                        df[f'regressor_{i}'] = X[:, i]
                
                self.model.fit(df)
                return self
            
            def predict(self, X):
                """Make predictions."""
                if self.model is None:
                    raise ValueError("Model must be fitted first")
                
                # Create future dataframe
                future = self.model.make_future_dataframe(periods=len(X))
                
                # Add regressors if available
                if X is not None and len(X.shape) > 1:
                    for i in range(X.shape[1]):
                        future[f'regressor_{i}'] = np.concatenate([
                            self.model.history[f'regressor_{i}'].values,
                            X[:, i]
                        ])
                
                forecast = self.model.predict(future)
                return forecast['yhat'].values[-len(X):]
        
        models['Prophet'] = ProphetWrapper()
        models['Prophet_Additive'] = ProphetWrapper(seasonality_mode='additive')
        
    except ImportError:
        logger.debug("Prophet not installed")
    
    # ARIMA
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.base import BaseEstimator, RegressorMixin
        
        class ARIMAWrapper(BaseEstimator, RegressorMixin):
            """Sklearn-compatible ARIMA wrapper."""
            
            def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
                self.order = order
                self.seasonal_order = seasonal_order
                self.model = None
                self.model_fit = None
                
            def fit(self, X, y):
                """Fit ARIMA model."""
                self.model = ARIMA(y, order=self.order, seasonal_order=self.seasonal_order)
                self.model_fit = self.model.fit()
                return self
            
            def predict(self, X):
                """Make predictions."""
                if self.model_fit is None:
                    raise ValueError("Model must be fitted first")
                
                n_periods = len(X) if hasattr(X, '__len__') else X
                forecast = self.model_fit.forecast(steps=n_periods)
                return np.array(forecast)
        
        models['ARIMA'] = ARIMAWrapper()
        models['ARIMA_Seasonal'] = ARIMAWrapper(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        
    except ImportError:
        logger.debug("statsmodels not installed")
    
    # LSTM
    try:
        import torch
        import torch.nn as nn
        from sklearn.base import BaseEstimator, RegressorMixin
        
        class LSTMModel(nn.Module):
            """Simple LSTM model."""
            
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out
        
        class LSTMWrapper(BaseEstimator, RegressorMixin):
            """Sklearn-compatible LSTM wrapper."""
            
            def __init__(self, hidden_size=64, num_layers=2, learning_rate=0.001, epochs=100):
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.learning_rate = learning_rate
                self.epochs = epochs
                self.model = None
                self.scaler = None
                
            def fit(self, X, y):
                """Fit LSTM model."""
                from sklearn.preprocessing import StandardScaler
                
                # Normalize data
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1]))
                X_scaled = X_scaled.reshape(X.shape)
                
                # Convert to tensors
                X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)
                y_tensor = torch.FloatTensor(y).unsqueeze(1)
                
                # Initialize model
                input_size = X.shape[-1] if len(X.shape) > 1 else 1
                self.model = LSTMModel(input_size, self.hidden_size, self.num_layers, 1)
                
                # Training
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                
                for epoch in range(self.epochs):
                    outputs = self.model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if epoch % 10 == 0:
                        logger.debug(f'Epoch [{epoch}/{self.epochs}], Loss: {loss.item():.4f}')
                
                return self
            
            def predict(self, X):
                """Make predictions."""
                if self.model is None:
                    raise ValueError("Model must be fitted first")
                
                # Normalize
                X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
                X_scaled = X_scaled.reshape(X.shape)
                
                # Predict
                X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)
                self.model.eval()
                with torch.no_grad():
                    predictions = self.model(X_tensor)
                
                return predictions.numpy().flatten()
        
        models['LSTM'] = LSTMWrapper()
        models['LSTM_Deep'] = LSTMWrapper(hidden_size=128, num_layers=3)
        
    except ImportError:
        logger.debug("PyTorch not installed for LSTM")
    
    return models


def _get_automl_models(task: str) -> Dict[str, Any]:
    """Get AutoML library models."""
    models = {}
    
    # AutoGluon
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
    
    # Comprehensive parameter grids including time series models
    param_grids = {
        # Time Series Models
        'Prophet': {
            'seasonality_mode': ['additive', 'multiplicative'],
            'yearly_seasonality': [True, False],
            'weekly_seasonality': [True, False],
            'daily_seasonality': [True, False],
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
        },
        'ARIMA': {
            'order': [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1), (1,1,1), (2,1,2)],
            'seasonal_order': [(0,0,0,0), (1,0,0,12), (0,1,0,12), (0,0,1,12), (1,1,1,12)]
        },
        'LSTM': {
            'hidden_size': [32, 64, 128, 256],
            'num_layers': [1, 2, 3, 4],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1],
            'epochs': [50, 100, 200],
            'dropout': [0.0, 0.1, 0.2, 0.3]
        },
        
        # Transfer Learning
        'TransferLearning_ResNet18': {
            'freeze_base': [True, False],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64],
            'epochs': [10, 20, 50]
        },
        
        # Imbalanced Learning
        'BalancedRandomForestClassifier': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'sampling_strategy': ['auto', 'majority', 'not minority', 'not majority']
        },
        
        # Linear models
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'max_iter': [100, 500, 1000],
            'class_weight': [None, 'balanced']
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
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced', 'balanced_subsample']
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
            'reg_lambda': [0, 0.01, 0.1, 1],
            'scale_pos_weight': [1, 2, 5, 10]  # For imbalanced data
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
            'reg_lambda': [0, 0.01, 0.1],
            'class_weight': [None, 'balanced']
        },
        
        # CatBoost
        'CatBoostClassifier': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7],
            'border_count': [32, 64, 128],
            'bagging_temperature': [0, 0.5, 1],
            'random_strength': [0, 0.5, 1],
            'auto_class_weights': ['Balanced', 'SqrtBalanced', None]
        },
        
        # TabNet
        'TabNetClassifier': {
            'n_d': [8, 16, 32],
            'n_a': [8, 16, 32],
            'n_steps': [3, 4, 5],
            'gamma': [1.0, 1.3, 1.5],
            'n_independent': [1, 2, 3],
            'n_shared': [1, 2, 3],
            'lambda_sparse': [1e-4, 1e-3, 1e-2]
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
