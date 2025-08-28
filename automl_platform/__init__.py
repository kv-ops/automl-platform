"""
AutoML Platform - Production-ready AutoML with no data leakage.

A comprehensive machine learning automation platform that provides:
- Automated model selection and hyperparameter optimization
- No data leakage guarantee through proper CV pipelines
- Support for 30+ sklearn models plus XGBoost, LightGBM, CatBoost
- Automatic feature engineering and preprocessing
- Model explainability with SHAP and LIME
- REST API for deployment
"""

__version__ = "3.0.0"
__author__ = "AutoML Platform Team"

# Import main components
from .config import AutoMLConfig
from .orchestrator import AutoMLOrchestrator
from .data_prep import DataPreprocessor, validate_data, handle_imbalance
from .model_selection import (
    get_available_models,
    get_cv_splitter,
    get_param_grid,
    tune_model
)
from .metrics import detect_task, calculate_metrics
from .inference import (
    load_pipeline,
    predict,
    predict_proba,
    predict_batch,
    save_predictions
)

# Define public API
__all__ = [
    # Version
    "__version__",
    
    # Main classes
    "AutoMLConfig",
    "AutoMLOrchestrator",
    "DataPreprocessor",
    
    # Functions
    "validate_data",
    "handle_imbalance",
    "get_available_models",
    "get_cv_splitter",
    "get_param_grid",
    "tune_model",
    "detect_task",
    "calculate_metrics",
    "load_pipeline",
    "predict",
    "predict_proba",
    "predict_batch",
    "save_predictions",
]

# Module level docstring for help()
def get_info():
    """
    Get information about the AutoML Platform.
    
    Returns:
        dict: Platform information including version, features, and usage
    """
    return {
        "version": __version__,
        "description": "Production-ready AutoML with no data leakage",
        "features": [
            "30+ sklearn models + XGBoost/LightGBM/CatBoost",
            "Automatic preprocessing with no data leakage",
            "Hyperparameter optimization with Optuna",
            "Model explainability (SHAP/LIME)",
            "Imbalanced data handling",
            "REST API for deployment",
            "Comprehensive testing suite"
        ],
        "usage": """
        from automl_platform import AutoMLConfig, AutoMLOrchestrator
        import pandas as pd
        
        # Load data
        df = pd.read_csv('data.csv')
        X = df.drop(columns=['target'])
        y = df['target']
        
        # Configure and train
        config = AutoMLConfig()
        orchestrator = AutoMLOrchestrator(config)
        orchestrator.fit(X, y)
        
        # Get results
        leaderboard = orchestrator.get_leaderboard()
        predictions = orchestrator.predict(X_test)
        """,
        "documentation": "https://github.com/automl-platform/automl-platform",
        "license": "MIT"
    }
