"""
Advanced ensemble methods inspired by AutoGluon
Implements multi-layer stacking, weighted ensemble, and dynamic selection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import logging
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AutoMLEnsemble(BaseEstimator):
    """
    Advanced ensemble with multi-layer stacking inspired by AutoGluon.
    Supports both classification and regression.
    """
    
    def __init__(self,
                 base_models: List[Tuple[str, Any]],
                 ensemble_method: str = 'stacking',
                 n_layers: int = 2,
                 cv_folds: int = 5,
                 use_probabilities: bool = True,
                 weighted: bool = True,
                 task: str = 'classification',
                 random_state: int = 42):
        """
        Initialize ensemble.
        
        Args:
            base_models: List of (name, model) tuples
            ensemble_method: 'stacking', 'voting', 'blending', 'dynamic'
            n_layers: Number of stacking layers
            cv_folds: Folds for generating meta-features
            use_probabilities: Use probabilities for classification
            weighted: Use weighted average based on CV performance
            task: 'classification' or 'regression'
            random_state: Random seed
        """
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.n_layers = n_layers
        self.cv_folds = cv_folds
        self.use_probabilities = use_probabilities
        self.weighted = weighted
        self.task = task
        self.random_state = random_state
        
        self.fitted_models_ = []
        self.meta_models_ = []
        self.weights_ = None
        self.feature_importances_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AutoMLEnsemble':
        """Fit ensemble models."""
        
        if self.ensemble_method == 'stacking':
            return self._fit_stacking(X, y)
        elif self.ensemble_method == 'voting':
            return self._fit_voting(X, y)
        elif self.ensemble_method == 'blending':
            return self._fit_blending(X, y)
        elif self.ensemble_method == 'dynamic':
            return self._fit_dynamic(X, y)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _fit_stacking(self, X: np.ndarray, y: np.ndarray) -> 'AutoMLEnsemble':
        """Fit multi-layer stacking ensemble."""
        
        logger.info(f"Training {self.n_layers}-layer stacking ensemble with {len(self.base_models)} models")
        
        # Setup CV
        if self.task == 'classification':
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        current_X = X
        layer_models = []
        
        for layer in range(self.n_layers):
            logger.info(f"Training layer {layer + 1}/{self.n_layers}")
            
            if layer == 0:
                # First layer: use base models
                models_to_use = self.base_models
            else:
                # Subsequent layers: use subset of best models
                models_to_use = self._select_models_for_layer(layer_models[-1], current_X, y)
            
            # Generate meta-features
            meta_features = []
            fitted_models = []
            
            for name, model in models_to_use:
                logger.debug(f"Training {name}")
                
                try:
                    # Clone model to avoid modifying original
                    model_clone = clone(model)
                    
                    # Generate out-of-fold predictions
                    if self.task == 'classification' and self.use_probabilities and hasattr(model_clone, 'predict_proba'):
                        oof_preds = cross_val_predict(
                            model_clone, current_X, y, cv=cv,
                            method='predict_proba', n_jobs=-1
                        )
                        # For binary classification, use only positive class probabilities
                        if oof_preds.shape[1] == 2:
                            oof_preds = oof_preds[:, 1].reshape(-1, 1)
                    else:
                        oof_preds = cross_val_predict(
                            model_clone, current_X, y, cv=cv,
                            method='predict', n_jobs=-1
                        ).reshape(-1, 1)
                    
                    meta_features.append(oof_preds)
                    
                    # Fit model on full data for later use
                    model_clone.fit(current_X, y)
                    fitted_models.append((name, model_clone))
                    
                except Exception as e:
                    logger.warning(f"Failed to train {name}: {e}")
                    continue
            
            if not meta_features:
                raise ValueError("No models could be trained")
            
            # Combine meta-features
            meta_X = np.hstack(meta_features)
            
            # Add original features if first layer (optional)
            if layer == 0 and current_X.shape[1] < 100:  # Only for small feature sets
                meta_X = np.hstack([current_X, meta_X])
            
            layer_models.append(fitted_models)
            
            # Prepare for next layer or final meta-model
            if layer < self.n_layers - 1:
                current_X = meta_X
            else:
                # Train final meta-model
                self._train_meta_model(meta_X, y)
        
        self.fitted_models_ = layer_models
        return self
    
    def _fit_voting(self, X: np.ndarray, y: np.ndarray) -> 'AutoMLEnsemble':
        """Fit voting ensemble."""
        
        logger.info(f"Training voting ensemble with {len(self.base_models)} models")
        
        if self.weighted:
            # Calculate weights based on CV performance
            weights = self._calculate_weights(X, y)
        else:
            weights = None
        
        # Create voting ensemble
        if self.task == 'classification':
            voting = 'soft' if self.use_probabilities else 'hard'
            self.ensemble_ = VotingClassifier(
                estimators=self.base_models,
                voting=voting,
                weights=weights,
                n_jobs=-1
            )
        else:
            self.ensemble_ = VotingRegressor(
                estimators=self.base_models,
                weights=weights,
                n_jobs=-1
            )
        
        self.ensemble_.fit(X, y)
        self.weights_ = weights
        
        return self
    
    def _fit_blending(self, X: np.ndarray, y: np.ndarray) -> 'AutoMLEnsemble':
        """Fit blending ensemble (holdout stacking)."""
        
        logger.info("Training blending ensemble")
        
        # Split data into blend and holdout
        from sklearn.model_selection import train_test_split
        X_blend, X_holdout, y_blend, y_holdout = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state,
            stratify=y if self.task == 'classification' else None
        )
        
        # Train base models on blend set
        fitted_models = []
        holdout_preds = []
        
        for name, model in self.base_models:
            try:
                model_clone = clone(model)
                model_clone.fit(X_blend, y_blend)
                
                # Get predictions on holdout
                if self.task == 'classification' and self.use_probabilities and hasattr(model_clone, 'predict_proba'):
                    preds = model_clone.predict_proba(X_holdout)
                    if preds.shape[1] == 2:
                        preds = preds[:, 1].reshape(-1, 1)
                else:
                    preds = model_clone.predict(X_holdout).reshape(-1, 1)
                
                holdout_preds.append(preds)
                fitted_models.append((name, model_clone))
                
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        # Train meta-model on holdout predictions
        meta_X = np.hstack(holdout_preds)
        self._train_meta_model(meta_X, y_holdout)
        
        # Retrain all models on full data
        self.fitted_models_ = []
        for name, model in self.base_models:
            model_clone = clone(model)
            model_clone.fit(X, y)
            self.fitted_models_.append((name, model_clone))
        
        return self
    
    def _fit_dynamic(self, X: np.ndarray, y: np.ndarray) -> 'AutoMLEnsemble':
        """Fit dynamic ensemble selection."""
        
        logger.info("Training dynamic ensemble selection")
        
        # Train all base models
        self.fitted_models_ = []
        for name, model in self.base_models:
            try:
                model_clone = clone(model)
                model_clone.fit(X, y)
                self.fitted_models_.append((name, model_clone))
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
        
        # Store training data for dynamic selection
        self.X_train_ = X
        self.y_train_ = y
        
        # Calculate competence regions for each model
        self._calculate_competence_regions(X, y)
        
        return self
    
    def _train_meta_model(self, meta_X: np.ndarray, y: np.ndarray):
        """Train the meta-model for stacking."""
        
        # Scale meta-features
        self.scaler_ = StandardScaler()
        meta_X_scaled = self.scaler_.fit_transform(meta_X)
        
        if self.task == 'classification':
            # Use LogisticRegression or simple neural network
            self.meta_model_ = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                C=1.0
            )
        else:
            # Use Ridge regression
            self.meta_model_ = Ridge(
                alpha=1.0,
                random_state=self.random_state
            )
        
        self.meta_model_.fit(meta_X_scaled, y)
        
        # Try to get feature importances from meta-model
        if hasattr(self.meta_model_, 'coef_'):
            self.feature_importances_ = np.abs(self.meta_model_.coef_).ravel()
    
    def _calculate_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate weights for voting based on CV performance."""
        
        weights = []
        
        if self.task == 'classification':
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.base_models:
            scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                try:
                    model_clone = clone(model)
                    model_clone.fit(X_train, y_train)
                    y_pred = model_clone.predict(X_val)
                    
                    if self.task == 'classification':
                        score = accuracy_score(y_val, y_pred)
                    else:
                        score = -mean_squared_error(y_val, y_pred)
                    
                    scores.append(score)
                except:
                    scores.append(0)
            
            avg_score = np.mean(scores) if scores else 0
            weights.append(max(0, avg_score))
        
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights
    
    def _select_models_for_layer(self, 
                                 prev_layer_models: List[Tuple[str, Any]],
                                 X: np.ndarray,
                                 y: np.ndarray) -> List[Tuple[str, Any]]:
        """Select best models for next stacking layer."""
        
        # Use top 50% of models from previous layer
        n_select = max(3, len(prev_layer_models) // 2)
        
        # Evaluate models
        scores = []
        for name, model in prev_layer_models:
            try:
                # Quick evaluation using a single split
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state
                )
                
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                y_pred = model_clone.predict(X_val)
                
                if self.task == 'classification':
                    score = accuracy_score(y_val, y_pred)
                else:
                    score = -mean_squared_error(y_val, y_pred)
                
                scores.append((score, (name, model)))
            except:
                continue
        
        # Sort by score and select top models
        scores.sort(key=lambda x: x[0], reverse=True)
        selected = [model for _, model in scores[:n_select]]
        
        # Add a simple meta-model if we have few models
        if len(selected) < 3:
            if self.task == 'classification':
                from sklearn.linear_model import LogisticRegression
                selected.append(('MetaLR', LogisticRegression(max_iter=1000)))
            else:
                from sklearn.linear_model import Ridge
                selected.append(('MetaRidge', Ridge()))
        
        return selected
    
    def _calculate_competence_regions(self, X: np.ndarray, y: np.ndarray):
        """Calculate competence regions for dynamic selection."""
        
        # For each model, calculate local accuracy in neighborhoods
        from sklearn.neighbors import NearestNeighbors
        
        self.nn_ = NearestNeighbors(n_neighbors=10)
        self.nn_.fit(X)
        
        # Store model competences
        self.competences_ = {}
        
        for name, model in self.fitted_models_:
            y_pred = model.predict(X)
            
            # Calculate local accuracy for each point
            competences = []
            for i in range(len(X)):
                # Find neighbors
                neighbors = self.nn_.kneighbors([X[i]], return_distance=False)[0]
                
                # Calculate local accuracy
                if self.task == 'classification':
                    local_acc = accuracy_score(y[neighbors], y_pred[neighbors])
                else:
                    local_acc = -mean_squared_error(y[neighbors], y_pred[neighbors])
                
                competences.append(local_acc)
            
            self.competences_[name] = np.array(competences)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        
        if self.ensemble_method == 'stacking':
            return self._predict_stacking(X)
        elif self.ensemble_method == 'voting':
            return self.ensemble_.predict(X)
        elif self.ensemble_method == 'blending':
            return self._predict_blending(X)
        elif self.ensemble_method == 'dynamic':
            return self._predict_dynamic(X)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking."""
        
        current_X = X
        
        # Pass through all layers
        for layer_idx, layer_models in enumerate(self.fitted_models_):
            predictions = []
            
            for name, model in layer_models:
                if self.task == 'classification' and self.use_probabilities and hasattr(model, 'predict_proba'):
                    preds = model.predict_proba(current_X)
                    if preds.shape[1] == 2:
                        preds = preds[:, 1].reshape(-1, 1)
                else:
                    preds = model.predict(current_X).reshape(-1, 1)
                
                predictions.append(preds)
            
            meta_X = np.hstack(predictions)
            
            # Add original features if first layer
            if layer_idx == 0 and X.shape[1] < 100:
                meta_X = np.hstack([current_X, meta_X])
            
            current_X = meta_X
        
        # Final prediction with meta-model
        meta_X_scaled = self.scaler_.transform(current_X)
        return self.meta_model_.predict(meta_X_scaled)
    
    def _predict_blending(self, X: np.ndarray) -> np.ndarray:
        """Predict using blending."""
        
        predictions = []
        
        for name, model in self.fitted_models_:
            if self.task == 'classification' and self.use_probabilities and hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)
                if preds.shape[1] == 2:
                    preds = preds[:, 1].reshape(-1, 1)
            else:
                preds = model.predict(X).reshape(-1, 1)
            
            predictions.append(preds)
        
        meta_X = np.hstack(predictions)
        meta_X_scaled = self.scaler_.transform(meta_X)
        
        return self.meta_model_.predict(meta_X_scaled)
    
    def _predict_dynamic(self, X: np.ndarray) -> np.ndarray:
        """Predict using dynamic selection."""
        
        predictions = []
        
        for i in range(len(X)):
            # Find most competent model for this instance
            neighbors = self.nn_.kneighbors([X[i]], return_distance=False)[0]
            
            best_model = None
            best_score = -np.inf
            
            for name, model in self.fitted_models_:
                # Get average competence in neighborhood
                competence = np.mean(self.competences_[name][neighbors])
                
                if competence > best_score:
                    best_score = competence
                    best_model = model
            
            # Predict with best model
            pred = best_model.predict([X[i]])
            predictions.append(pred[0])
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (classification only)."""
        
        if self.task != 'classification':
            raise ValueError("predict_proba is only for classification")
        
        if self.ensemble_method == 'voting':
            if hasattr(self.ensemble_, 'predict_proba'):
                return self.ensemble_.predict_proba(X)
        
        # For other methods, we need to implement probability prediction
        # This is a simplified version
        predictions = self.predict(X)
        
        # Convert to probability format
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        lb.fit(predictions)
        return lb.transform(predictions)


class AutoGluonEnsemble(AutoMLEnsemble):
    """
    AutoGluon-style ensemble with bagging and multi-layer stacking.
    """
    
    def __init__(self,
                 base_models: List[Tuple[str, Any]],
                 n_bags: int = 5,
                 n_layers: int = 3,
                 n_folds: int = 5,
                 task: str = 'classification',
                 random_state: int = 42):
        """
        Initialize AutoGluon-style ensemble.
        
        Args:
            base_models: List of (name, model) tuples
            n_bags: Number of bagging iterations
            n_layers: Number of stacking layers
            n_folds: Number of CV folds for stacking
            task: 'classification' or 'regression'
            random_state: Random seed
        """
        super().__init__(
            base_models=base_models,
            ensemble_method='stacking',
            n_layers=n_layers,
            cv_folds=n_folds,
            use_probabilities=True,
            weighted=True,
            task=task,
            random_state=random_state
        )
        
        self.n_bags = n_bags
        self.bagged_models_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AutoGluonEnsemble':
        """Fit with bagging and stacking."""
        
        logger.info(f"Training AutoGluon-style ensemble with {self.n_bags} bags")
        
        # Bagging phase
        for bag in range(self.n_bags):
            logger.info(f"Training bag {bag + 1}/{self.n_bags}")
            
            # Bootstrap sample
            indices = np.random.RandomState(self.random_state + bag).choice(
                len(X), size=len(X), replace=True
            )
            X_bag = X[indices]
            y_bag = y[indices]
            
            # Train stacking ensemble on bag
            bag_ensemble.fit(X_bag, y_bag)
            self.bagged_models_.append(bag_ensemble)
        
        # Final stacking on out-of-bag predictions
        self._fit_final_stacking(X, y)
        
        return self
    
    def _fit_final_stacking(self, X: np.ndarray, y: np.ndarray):
        """Fit final stacking layer on bagged predictions."""
        
        # Get out-of-bag predictions from all bags
        oob_predictions = []
        
        for bag_ensemble in self.bagged_models_:
            preds = bag_ensemble.predict(X)
            oob_predictions.append(preds.reshape(-1, 1))
        
        # Stack predictions
        meta_X = np.hstack(oob_predictions)
        
        # Train final meta-model
        self._train_meta_model(meta_X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using bagged ensemble."""
        
        # Get predictions from all bags
        bag_predictions = []
        
        for bag_ensemble in self.bagged_models_:
            preds = bag_ensemble.predict(X)
            bag_predictions.append(preds.reshape(-1, 1))
        
        # Stack and predict with meta-model
        meta_X = np.hstack(bag_predictions)
        meta_X_scaled = self.scaler_.transform(meta_X)
        
        return self.meta_model_.predict(meta_X_scaled)


class WeightedEnsemble(BaseEstimator):
    """
    Simple weighted ensemble with automatic weight optimization.
    """
    
    def __init__(self,
                 models: List[Tuple[str, Any]],
                 optimization_method: str = 'nelder-mead',
                 task: str = 'classification',
                 metric: str = 'auto'):
        """
        Initialize weighted ensemble.
        
        Args:
            models: List of (name, model) tuples
            optimization_method: Method for weight optimization
            task: 'classification' or 'regression'
            metric: Metric to optimize
        """
        self.models = models
        self.optimization_method = optimization_method
        self.task = task
        self.metric = metric
        self.weights_ = None
        self.fitted_models_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightedEnsemble':
        """Fit models and optimize weights."""
        
        # Fit all models
        self.fitted_models_ = []
        predictions = []
        
        for name, model in self.models:
            logger.info(f"Training {name}")
            model_clone = clone(model)
            model_clone.fit(X, y)
            self.fitted_models_.append((name, model_clone))
            
            # Get predictions for weight optimization
            pred = model_clone.predict(X)
            predictions.append(pred)
        
        # Optimize weights
        predictions = np.array(predictions).T
        self.weights_ = self._optimize_weights(predictions, y)
        
        logger.info(f"Optimized weights: {dict(zip([n for n, _ in self.models], self.weights_))}")
        
        return self
    
    def _optimize_weights(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Optimize ensemble weights."""
        
        from scipy.optimize import minimize
        
        n_models = predictions.shape[1]
        
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / weights.sum()
            
            # Weighted prediction
            y_pred = np.average(predictions, axis=1, weights=weights)
            
            # Calculate loss
            if self.task == 'classification':
                from sklearn.metrics import log_loss
                try:
                    loss = log_loss(y_true, y_pred)
                except:
                    loss = np.mean((y_true - y_pred) ** 2)
            else:
                loss = np.mean((y_true - y_pred) ** 2)
            
            return loss
        
        # Initial weights (uniform)
        x0 = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            objective, x0,
            method=self.optimization_method,
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted predictions."""
        
        predictions = []
        
        for (name, model), weight in zip(self.fitted_models_, self.weights_):
            pred = model.predict(X) * weight
            predictions.append(pred)
        
        return np.sum(predictions, axis=0)


# Utility functions for ensemble creation
def create_diverse_ensemble(X: np.ndarray, y: np.ndarray,
                           task: str = 'classification',
                           max_models: int = 10) -> List[Tuple[str, Any]]:
    """
    Create a diverse set of models for ensemble.
    
    Args:
        X: Training features
        y: Training labels
        task: Task type
        max_models: Maximum number of models
        
    Returns:
        List of (name, model) tuples
    """
    models = []
    
    if task == 'classification':
        # Linear models
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        models.append(('LogisticRegression', LogisticRegression(max_iter=1000)))
        models.append(('RidgeClassifier', RidgeClassifier()))
        
        # Tree-based
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        models.append(('RandomForest', RandomForestClassifier(n_estimators=100, n_jobs=-1)))
        models.append(('ExtraTrees', ExtraTreesClassifier(n_estimators=100, n_jobs=-1)))
        
        # Boosting
        try:
            from xgboost import XGBClassifier
            models.append(('XGBoost', XGBClassifier(n_estimators=100, verbosity=0)))
        except ImportError:
            pass
        
        try:
            from lightgbm import LGBMClassifier
            models.append(('LightGBM', LGBMClassifier(n_estimators=100, verbosity=-1)))
        except ImportError:
            pass
        
        try:
            from catboost import CatBoostClassifier
            models.append(('CatBoost', CatBoostClassifier(iterations=100, verbose=False)))
        except ImportError:
            pass
        
        # KNN
        from sklearn.neighbors import KNeighborsClassifier
        models.append(('KNN', KNeighborsClassifier(n_neighbors=10)))
        
        # Naive Bayes
        from sklearn.naive_bayes import GaussianNB
        models.append(('NaiveBayes', GaussianNB()))
        
        # SVM (if dataset is small)
        if len(X) < 5000:
            from sklearn.svm import SVC
            models.append(('SVM', SVC(probability=True)))
            
    else:  # regression
        # Linear models
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        models.append(('Ridge', Ridge()))
        models.append(('Lasso', Lasso()))
        models.append(('ElasticNet', ElasticNet()))
        
        # Tree-based
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        models.append(('RandomForest', RandomForestRegressor(n_estimators=100, n_jobs=-1)))
        models.append(('ExtraTrees', ExtraTreesRegressor(n_estimators=100, n_jobs=-1)))
        
        # Boosting
        try:
            from xgboost import XGBRegressor
            models.append(('XGBoost', XGBRegressor(n_estimators=100, verbosity=0)))
        except ImportError:
            pass
        
        try:
            from lightgbm import LGBMRegressor
            models.append(('LightGBM', LGBMRegressor(n_estimators=100, verbosity=-1)))
        except ImportError:
            pass
        
        try:
            from catboost import CatBoostRegressor
            models.append(('CatBoost', CatBoostRegressor(iterations=100, verbose=False)))
        except ImportError:
            pass
        
        # KNN
        from sklearn.neighbors import KNeighborsRegressor
        models.append(('KNN', KNeighborsRegressor(n_neighbors=10)))
        
        # SVR (if dataset is small)
        if len(X) < 5000:
            from sklearn.svm import SVR
            models.append(('SVR', SVR()))
    
    # Limit to max_models
    if len(models) > max_models:
        models = models[:max_models]
    
    return models


def create_ensemble_pipeline(base_models: List[Tuple[str, Any]],
                            ensemble_type: str = 'auto',
                            task: str = 'classification') -> BaseEstimator:
    """
    Create an ensemble pipeline based on the number and type of models.
    
    Args:
        base_models: List of (name, model) tuples
        ensemble_type: Type of ensemble ('auto', 'voting', 'stacking', 'autogluon')
        task: Task type
        
    Returns:
        Configured ensemble model
    """
    n_models = len(base_models)
    
    if ensemble_type == 'auto':
        # Automatically choose ensemble type
        if n_models < 3:
            ensemble_type = 'voting'
        elif n_models < 10:
            ensemble_type = 'stacking'
        else:
            ensemble_type = 'autogluon'
    
    if ensemble_type == 'voting':
        return AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='voting',
            weighted=True,
            task=task
        )
    elif ensemble_type == 'stacking':
        return AutoMLEnsemble(
            base_models=base_models,
            ensemble_method='stacking',
            n_layers=2,
            cv_folds=5,
            task=task
        )
    elif ensemble_type == 'autogluon':
        return AutoGluonEnsemble(
            base_models=base_models,
            n_bags=3,
            n_layers=2,
            n_folds=5,
            task=task
        )
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create diverse models
    models = create_diverse_ensemble(X_train, y_train, task='classification', max_models=5)
    
    # Create ensemble
    ensemble = AutoMLEnsemble(
        base_models=models,
        ensemble_method='stacking',
        n_layers=2,
        cv_folds=5,
        task='classification'
    )
    
    # Train
    ensemble.fit(X_train, y_train)
    
    # Predict
    y_pred = ensemble.predict(X_test)
    
    # Evaluate
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"Ensemble accuracy: {acc:.4f}") = AutoMLEnsemble(
                base_models=self.base_models.copy(),
                ensemble_method='stacking',
                n_layers=self.n_layers,
                cv_folds=self.cv_folds,
                use_probabilities=self.use_probabilities,
                weighted=self.weighted,
                task=self.task,
                random_state=self.random_state + bag
            )
            
            bag_ensemble
