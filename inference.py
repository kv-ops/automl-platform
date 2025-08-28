"""Inference utilities for AutoML platform."""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
import json
import logging

logger = logging.getLogger(__name__)


def load_pipeline(filepath: Union[str, Path]) -> Tuple[Any, Dict]:
    """
    Load saved pipeline with metadata.
    
    Args:
        filepath: Path to saved pipeline file
        
    Returns:
        Tuple of (pipeline, metadata)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pipeline not found: {filepath}")
    
    # Load pipeline
    pipeline = joblib.load(filepath)
    
    # Load metadata if available
    metadata_path = filepath.with_suffix('.meta.json')
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            logger.info(f"Loaded metadata: task={metadata.get('task')}, "
                       f"best_model={metadata.get('best_model')}")
    
    logger.info(f"Pipeline loaded from {filepath}")
    return pipeline, metadata


def predict(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using pipeline.
    
    Args:
        pipeline: Trained sklearn pipeline
        X: Features dataframe
        
    Returns:
        Array of predictions
    """
    if not hasattr(pipeline, 'predict'):
        raise ValueError("Pipeline doesn't have predict method")
    
    predictions = pipeline.predict(X)
    logger.info(f"Generated {len(predictions)} predictions")
    return predictions


def predict_proba(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Get probability predictions.
    
    Args:
        pipeline: Trained sklearn pipeline
        X: Features dataframe
        
    Returns:
        Array of probabilities
    """
    if not hasattr(pipeline, 'predict_proba'):
        raise ValueError("Pipeline doesn't support probability predictions")
    
    probabilities = pipeline.predict_proba(X)
    logger.info(f"Generated probabilities with shape {probabilities.shape}")
    return probabilities


def predict_batch(pipeline: Any, X: pd.DataFrame, 
                 batch_size: int = 1000) -> np.ndarray:
    """
    Make predictions in batches for large datasets.
    
    Args:
        pipeline: Trained sklearn pipeline
        X: Features dataframe
        batch_size: Size of each batch
        
    Returns:
        Array of predictions
    """
    n_samples = len(X)
    predictions = []
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch = X.iloc[i:batch_end]
        batch_pred = pipeline.predict(batch)
        predictions.append(batch_pred)
        logger.info(f"Processed batch {i//batch_size + 1}/{(n_samples-1)//batch_size + 1}")
    
    return np.concatenate(predictions)


def save_predictions(predictions: np.ndarray, 
                    filepath: Union[str, Path],
                    ids: Optional[np.ndarray] = None,
                    probabilities: Optional[np.ndarray] = None) -> None:
    """
    Save predictions to file.
    
    Args:
        predictions: Array of predictions
        filepath: Output file path
        ids: Optional array of IDs
        probabilities: Optional probability array
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Create dataframe
    data = {'prediction': predictions}
    
    if ids is not None:
        data['id'] = ids
        data = {'id': ids, 'prediction': predictions}
    
    if probabilities is not None:
        if len(probabilities.shape) > 1:
            # Multi-class probabilities
            for i in range(probabilities.shape[1]):
                data[f'prob_class_{i}'] = probabilities[:, i]
        else:
            data['probability'] = probabilities
    
    df = pd.DataFrame(data)
    
    # Save based on extension
    if filepath.suffix == '.csv':
        df.to_csv(filepath, index=False)
    elif filepath.suffix == '.parquet':
        df.to_parquet(filepath, index=False)
    elif filepath.suffix == '.json':
        df.to_json(filepath, orient='records', indent=2)
    else:
        # Default to CSV
        df.to_csv(filepath, index=False)
    
    logger.info(f"Predictions saved to {filepath}")


def explain_prediction(pipeline: Any, X: pd.DataFrame, 
                       index: int = 0) -> Dict[str, Any]:
    """
    Explain a single prediction using SHAP or feature importance.
    
    Args:
        pipeline: Trained pipeline
        X: Features dataframe
        index: Index of instance to explain
        
    Returns:
        Dictionary with explanation
    """
    explanation = {}
    
    try:
        import shap
        
        # Extract model from pipeline
        if hasattr(pipeline, 'named_steps'):
            model = pipeline.named_steps.get('model')
            preprocessor = pipeline.named_steps.get('preprocessor')
            
            if preprocessor and model:
                X_transformed = preprocessor.transform(X)
                
                # Try tree explainer for tree-based models
                model_type = type(model).__name__
                if 'Tree' in model_type or 'Forest' in model_type or 'Boost' in model_type:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_transformed[index:index+1])
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    
                    explanation['method'] = 'shap_tree'
                    explanation['values'] = shap_values[0].tolist()
                    explanation['base_value'] = explainer.expected_value
                    
                else:
                    # Use kernel explainer for other models
                    sample = X_transformed[:100] if len(X_transformed) > 100 else X_transformed
                    explainer = shap.KernelExplainer(model.predict, sample)
                    shap_values = explainer.shap_values(X_transformed[index:index+1])
                    
                    explanation['method'] = 'shap_kernel'
                    explanation['values'] = shap_values[0].tolist()
                    explanation['base_value'] = explainer.expected_value
    
    except ImportError:
        logger.warning("SHAP not available for explanations")
        
        # Fall back to feature importance if available
        if hasattr(pipeline, 'named_steps'):
            model = pipeline.named_steps.get('model')
            if hasattr(model, 'feature_importances_'):
                explanation['method'] = 'feature_importance'
                explanation['values'] = model.feature_importances_.tolist()
    
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        explanation['error'] = str(e)
    
    return explanation


def validate_input(X: pd.DataFrame, expected_features: Optional[List[str]] = None,
                  expected_dtypes: Optional[Dict[str, str]] = None) -> bool:
    """
    Validate input data before prediction.
    
    Args:
        X: Input features
        expected_features: Expected feature names
        expected_dtypes: Expected data types
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check for empty dataframe
    if X.empty:
        raise ValueError("Input dataframe is empty")
    
    # Check expected features
    if expected_features:
        missing = set(expected_features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        extra = set(X.columns) - set(expected_features)
        if extra:
            logger.warning(f"Extra features will be ignored: {extra}")
    
    # Check data types
    if expected_dtypes:
        for col, expected_dtype in expected_dtypes.items():
            if col in X.columns:
                actual_dtype = str(X[col].dtype)
                if not actual_dtype.startswith(expected_dtype):
                    logger.warning(f"Column {col} has dtype {actual_dtype}, "
                                 f"expected {expected_dtype}")
    
    # Check for all null columns
    null_cols = X.columns[X.isnull().all()].tolist()
    if null_cols:
        logger.warning(f"Columns with all null values: {null_cols}")
    
    return True
