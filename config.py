import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AutoMLConfig:
    """AutoML configuration with all hyperparameters."""
    
    # General settings
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 1
    
    # Data preprocessing
    max_missing_ratio: float = 0.5
    rare_category_threshold: float = 0.01
    high_cardinality_threshold: int = 20
    outlier_method: str = "iqr"  # "iqr", "isolation_forest", "none"
    outlier_threshold: float = 1.5
    scaling_method: str = "robust"  # "standard", "robust", "minmax", "none"
    
    # Feature engineering
    create_polynomial: bool = False
    polynomial_degree: int = 2
    create_interactions: bool = False
    create_datetime_features: bool = True
    create_lag_features: bool = False
    lag_periods: List[int] = field(default_factory=lambda: [1, 7, 30])
    
    # Text processing
    text_max_features: int = 100
    text_ngram_range: tuple = (1, 2)
    text_min_df: int = 2
    
    # Model selection
    task: str = "auto"  # "classification", "regression", "timeseries", "auto"
    cv_folds: int = 5
    validation_strategy: str = "auto"  # "stratified", "kfold", "timeseries", "auto"
    scoring: str = "auto"  # Will be determined based on task
    
    # Hyperparameter tuning
    hpo_method: str = "optuna"  # "grid", "random", "optuna", "none"
    hpo_n_iter: int = 20
    hpo_time_budget: int = 3600
    early_stopping_rounds: int = 50
    
    # Model training
    algorithms: List[str] = field(default_factory=lambda: ["all"])
    exclude_algorithms: List[str] = field(default_factory=list)
    ensemble_method: str = "voting"  # "voting", "stacking", "none"
    calibrate_probabilities: bool = False
    
    # Class imbalance
    handle_imbalance: bool = True
    imbalance_method: str = "class_weight"  # "class_weight", "smote", "adasyn", "none"
    
    # Performance thresholds
    min_accuracy: float = 0.6
    min_auc: float = 0.6
    min_r2: float = 0.0
    
    # Output settings
    output_dir: str = "./automl_output"
    save_pipeline: bool = True
    save_predictions: bool = True
    save_feature_importance: bool = True
    generate_report: bool = True
    
    # API settings
    api_enabled: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "AutoMLConfig":
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Handle any mismatched types
        if config_dict:
            # Ensure lists are properly converted
            if 'algorithms' in config_dict and not isinstance(config_dict['algorithms'], list):
                config_dict['algorithms'] = [config_dict['algorithms']]
            if 'exclude_algorithms' in config_dict and not isinstance(config_dict['exclude_algorithms'], list):
                config_dict['exclude_algorithms'] = [config_dict['exclude_algorithms']]
            if 'lag_periods' in config_dict and not isinstance(config_dict['lag_periods'], list):
                config_dict['lag_periods'] = [config_dict['lag_periods']]
        return cls(**config_dict) if config_dict else cls()
    
    def to_yaml(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        config_dict = asdict(self)
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        try:
            assert 0 <= self.max_missing_ratio <= 1, "max_missing_ratio must be between 0 and 1"
            assert 0 <= self.rare_category_threshold <= 1, "rare_category_threshold must be between 0 and 1"
            assert self.cv_folds > 1, "cv_folds must be greater than 1"
            assert self.hpo_n_iter > 0, "hpo_n_iter must be positive"
            assert self.outlier_method in ["iqr", "isolation_forest", "none"], f"Invalid outlier_method: {self.outlier_method}"
            assert self.scaling_method in ["standard", "robust", "minmax", "none"], f"Invalid scaling_method: {self.scaling_method}"
            assert self.hpo_method in ["grid", "random", "optuna", "none"], f"Invalid hpo_method: {self.hpo_method}"
            assert self.ensemble_method in ["voting", "stacking", "none"], f"Invalid ensemble_method: {self.ensemble_method}"
            assert self.imbalance_method in ["class_weight", "smote", "adasyn", "none"], f"Invalid imbalance_method: {self.imbalance_method}"
            
            # Create output directory if it doesn't exist
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            
            return True
        except AssertionError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
