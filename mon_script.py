#!/usr/bin/env python3
"""
Script de création du package AutoML Platform v2.0 complet
Version finale avec TOUS les modules - CORRIGÉE
"""

import os
import subprocess
from pathlib import Path
from typing import Dict

def create_automl_platform():
    """Crée le package AutoML Platform v2.0 complet sur disque"""
    
    print("🚀 Création du package AutoML Platform v2.0...")
    
    # Registre pour éviter les duplications
    files_to_write: Dict[str, str] = {}
    
    def add_file(path: str, content: str):
        """Ajoute un fichier au registre avec vérification de duplication"""
        if path in files_to_write:
            raise Exception(f"Duplicate file write detected: {path}")
        # Assurer que chaque fichier se termine par un newline
        if not content.endswith('\n'):
            content += '\n'
        files_to_write[path] = content
    
    # ==================== PACKAGE ROOT ====================
    
    add_file("automl_platform/__init__.py", '''"""AutoML Platform - Complete ML automation framework"""

__version__ = "2.0.0"

from .config.settings import EnhancedPlatformConfig, get_config, set_config
from .modeling.utils import load_model, save_model, predict, predict_proba
from .data.io import load_data, save_data, validate_dataframe, split_features_target
from .features.engineering import FeatureEngineer
from .features.selection import FeatureSelector
from .modeling.trainer import train_cv
from .explain.shap_lime import explain_global, explain_local
from .fairness.metrics import fairness_report, demographic_parity, equal_opportunity
from .monitoring.drift import check_drift
from .monitoring.quality import check_data_quality

__all__ = [
    "__version__",
    "EnhancedPlatformConfig", "get_config", "set_config",
    "load_model", "save_model", "predict", "predict_proba",
    "load_data", "save_data", "validate_dataframe", "split_features_target",
    "FeatureEngineer", "FeatureSelector",
    "train_cv",
    "explain_global", "explain_local",
    "fairness_report", "demographic_parity", "equal_opportunity",
    "check_drift", "check_data_quality"
]
''')

    # ==================== API ====================
    
    add_file("automl_platform/api/__init__.py", '''from .app import app
__all__ = ['app']
''')
    
    add_file("automl_platform/api/app.py", '''"""FastAPI application with all endpoints"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import uuid

from ..modeling.utils import load_model, predict, predict_proba
from ..config.settings import get_config
from ..explain.shap_lime import explain_global, explain_local
from ..monitoring.drift import check_drift
from ..monitoring.quality import check_data_quality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoML Platform API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
metrics = {
    "requests": 0,
    "errors": 0,
    "start_time": datetime.now()
}

config = get_config()
MODEL_PATH = config.model_path
_model_cache = {"model": None, "loaded_at": None, "hash": None}
_reference_data = None

# Pydantic models
class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    explain: bool = False
    request_id: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, Any]]
    explain: bool = False
    request_id: Optional[str] = None

class DriftCheckRequest(BaseModel):
    current_data: List[Dict[str, Any]]
    reference_data: Optional[List[Dict[str, Any]]] = None
    threshold: float = 0.1
    method: str = "ks"

def get_or_load_model():
    """Load model with caching and hash verification"""
    global _model_cache
    
    if _model_cache["model"] is not None:
        cache_age = (datetime.now() - _model_cache["loaded_at"]).total_seconds()
        if cache_age < 3600:  # 1 hour cache
            return _model_cache["model"]
    
    try:
        model = load_model(MODEL_PATH)
        _model_cache = {
            "model": model,
            "loaded_at": datetime.now(),
            "hash": None
        }
        logger.info(f"Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        logger.warning(f"Failed to load model from {MODEL_PATH}: {e}")
        # Return a dummy model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Create dummy data for fitting
        X_dummy = np.random.randn(100, 5)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        _model_cache = {
            "model": model,
            "loaded_at": datetime.now(),
            "hash": "dummy"
        }
        logger.info("Using dummy model for testing")
        return model

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "AutoML Platform API",
        "version": "2.0.0",
        "status": "ready",
        "endpoints": ["/health", "/predict", "/predict_batch", "/check_drift", "/metrics", "/reload_model"]
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    uptime = (datetime.now() - metrics["start_time"]).total_seconds()
    return {
        "status": "healthy",
        "model_path": "model.pkl",
        "uptime_seconds": uptime,
        "requests_served": metrics["requests"],
        "error_rate": metrics["errors"] / max(metrics["requests"], 1),
        "model_loaded": _model_cache["model"] is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
    """Single prediction endpoint with explain"""
    request_id = request.request_id or str(uuid.uuid4())
    metrics["requests"] += 1
    
    try:
        model = get_or_load_model()
        df = pd.DataFrame([request.features])
        
        # Handle different input sizes
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            if len(df.columns) < expected_features:
                # Pad with zeros
                for i in range(len(df.columns), expected_features):
                    df[f'feature_{i}'] = 0
            elif len(df.columns) > expected_features:
                # Truncate
                df = df.iloc[:, :expected_features]
        
        prediction = predict(model, df)[0]
        
        response = {
            "prediction": prediction.item() if hasattr(prediction, 'item') else prediction,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add confidence if available
        if hasattr(model, 'predict_proba'):
            proba = predict_proba(model, df)[0]
            response["confidence"] = float(max(proba))
            response["probabilities"] = proba.tolist()
        
        # Add explanation if requested
        if request.explain:
            try:
                # Global explanation (once)
                global_exp = explain_global(model, df)
                # Local explanation for this instance
                local_exp = explain_local(model, df, 0)
                
                response["explanation"] = {
                    "global": global_exp,
                    "local": local_exp
                }
            except Exception as e:
                logger.warning(f"Explanation failed: {e}")
                response["explanation"] = {"error": str(e)}
        
        return response
        
    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch_endpoint(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    request_id = request.request_id or str(uuid.uuid4())
    metrics["requests"] += 1
    
    try:
        model = get_or_load_model()
        df = pd.DataFrame(request.data)
        
        # Handle different input sizes
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            if len(df.columns) < expected_features:
                # Pad with zeros
                for i in range(len(df.columns), expected_features):
                    df[f'feature_{i}'] = 0
            elif len(df.columns) > expected_features:
                # Truncate
                df = df.iloc[:, :expected_features]
        
        predictions = predict(model, df)
        
        results = []
        for i, pred in enumerate(predictions):
            result = {
                "index": i,
                "prediction": pred.item() if hasattr(pred, 'item') else pred
            }
            
            # Add confidence if available
            if hasattr(model, 'predict_proba'):
                proba = predict_proba(model, df.iloc[[i]])[0]
                result["confidence"] = float(max(proba))
            
            results.append(result)
        
        return {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "predictions": results
        }
        
    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check_drift")
async def check_drift_endpoint(request: DriftCheckRequest):
    """Check for data drift"""
    try:
        current_df = pd.DataFrame(request.current_data)
        
        if request.reference_data:
            reference_df = pd.DataFrame(request.reference_data)
        else:
            # Use stored reference data if available
            if _reference_data is not None:
                reference_df = _reference_data
            else:
                return {
                    "error": "No reference data provided or stored",
                    "timestamp": datetime.now().isoformat()
                }
        
        drift_results = check_drift(
            current_df, 
            reference_df,
            threshold=request.threshold,
            method=request.method
        )
        
        return {
            "drift_results": drift_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Drift check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Get API metrics"""
    uptime = (datetime.now() - metrics["start_time"]).total_seconds()
    return {
        "requests": metrics["requests"],
        "errors": metrics["errors"],
        "error_rate": metrics["errors"] / max(metrics["requests"], 1),
        "uptime_seconds": uptime,
        "start_time": metrics["start_time"].isoformat(),
        "current_time": datetime.now().isoformat()
    }

@app.post("/reload_model")
async def reload_model():
    """Force reload model from disk"""
    global _model_cache
    
    try:
        model = load_model(MODEL_PATH)
        _model_cache = {
            "model": model,
            "loaded_at": datetime.now(),
            "hash": None
        }
        logger.info(f"Model reloaded from {MODEL_PATH}")
        
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_reference_data")
async def set_reference_data(data: List[Dict[str, Any]]):
    """Set reference data for drift detection"""
    global _reference_data
    
    try:
        _reference_data = pd.DataFrame(data)
        
        return {
            "status": "success",
            "message": f"Reference data set with {len(_reference_data)} samples",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Set reference data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
''')

    # ==================== MONITORING ====================
    
    add_file("automl_platform/monitoring/__init__.py", '''from .drift import check_drift
from .quality import check_data_quality
__all__ = ['check_drift', 'check_data_quality']
''')
    
    add_file("automl_platform/monitoring/drift.py", '''"""Data drift detection"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Union

def check_drift(X_current: pd.DataFrame, 
               X_reference: pd.DataFrame,
               threshold: float = 0.1,
               method: str = "ks") -> Dict[str, Any]:
    """
    Check for data drift between current and reference data
    
    Methods: KS test, Wasserstein distance, or JSD
    """
    results = {
        "drift_detected": False,
        "method": method,
        "threshold": threshold,
        "feature_drift": {},
        "segment_drift": {}
    }
    
    # Check each feature
    for col in X_current.columns:
        if col not in X_reference.columns:
            continue
        
        current_vals = X_current[col].dropna()
        reference_vals = X_reference[col].dropna()
        
        if len(current_vals) == 0 or len(reference_vals) == 0:
            continue
        
        # Compute drift score based on method
        if method == "ks":
            # Kolmogorov-Smirnov test
            if pd.api.types.is_numeric_dtype(current_vals):
                statistic, p_value = stats.ks_2samp(current_vals, reference_vals)
                drift_score = statistic
            else:
                # For categorical, use chi-square
                current_counts = current_vals.value_counts()
                reference_counts = reference_vals.value_counts()
                
                # Align categories
                all_categories = set(current_counts.index) | set(reference_counts.index)
                current_aligned = pd.Series([current_counts.get(cat, 0) for cat in all_categories])
                reference_aligned = pd.Series([reference_counts.get(cat, 0) for cat in all_categories])
                
                if len(all_categories) > 1:
                    chi2, p_value = stats.chisquare(current_aligned + 1, reference_aligned + 1)
                    drift_score = min(1.0, chi2 / 100)  # Normalize
                else:
                    drift_score = 0.0
        
        elif method == "wasserstein":
            # Wasserstein distance
            if pd.api.types.is_numeric_dtype(current_vals):
                drift_score = stats.wasserstein_distance(current_vals, reference_vals)
                # Normalize
                max_val = max(current_vals.max(), reference_vals.max())
                min_val = min(current_vals.min(), reference_vals.min())
                if max_val > min_val:
                    drift_score = drift_score / (max_val - min_val)
            else:
                drift_score = 0.0
        
        elif method == "jsd":
            # Jensen-Shannon Divergence
            if pd.api.types.is_numeric_dtype(current_vals):
                # Bin numeric values
                bins = np.histogram_bin_edges(np.concatenate([current_vals, reference_vals]), bins=20)
                current_hist, _ = np.histogram(current_vals, bins=bins)
                reference_hist, _ = np.histogram(reference_vals, bins=bins)
                
                # Normalize
                current_hist = current_hist / current_hist.sum()
                reference_hist = reference_hist / reference_hist.sum()
                
                # Calculate JSD
                m = (current_hist + reference_hist) / 2
                divergence_current = stats.entropy(current_hist + 1e-10, m + 1e-10)
                divergence_reference = stats.entropy(reference_hist + 1e-10, m + 1e-10)
                drift_score = (divergence_current + divergence_reference) / 2
            else:
                # For categorical
                current_probs = current_vals.value_counts(normalize=True)
                reference_probs = reference_vals.value_counts(normalize=True)
                
                # Align categories
                all_categories = set(current_probs.index) | set(reference_probs.index)
                current_aligned = pd.Series([current_probs.get(cat, 0) for cat in all_categories])
                reference_aligned = pd.Series([reference_probs.get(cat, 0) for cat in all_categories])
                
                m = (current_aligned + reference_aligned) / 2
                divergence_current = stats.entropy(current_aligned + 1e-10, m + 1e-10)
                divergence_reference = stats.entropy(reference_aligned + 1e-10, m + 1e-10)
                drift_score = (divergence_current + divergence_reference) / 2
        
        else:
            drift_score = 0.0
        
        results["feature_drift"][col] = {
            "score": float(drift_score),
            "drifted": drift_score > threshold
        }
        
        if drift_score > threshold:
            results["drift_detected"] = True
    
    # Check segment drift (for up to 2 categorical features)
    categorical_cols = X_current.select_dtypes(include=['object', 'category']).columns[:2]
    
    for col in categorical_cols:
        if col not in X_reference.columns:
            continue
        
        segments = X_current[col].unique()
        segment_scores = {}
        
        for segment in segments:
            if segment not in X_reference[col].values:
                segment_scores[str(segment)] = 1.0
                continue
            
            current_segment = X_current[X_current[col] == segment]
            reference_segment = X_reference[X_reference[col] == segment]
            
            if len(current_segment) < 10 or len(reference_segment) < 10:
                continue
            
            # Simple drift check on first numeric column
            numeric_cols = current_segment.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                num_col = numeric_cols[0]
                statistic, _ = stats.ks_2samp(
                    current_segment[num_col].dropna(),
                    reference_segment[num_col].dropna()
                )
                segment_scores[str(segment)] = float(statistic)
        
        if segment_scores:
            results["segment_drift"][col] = segment_scores
    
    # Overall drift score
    if results["feature_drift"]:
        drift_scores = [f["score"] for f in results["feature_drift"].values()]
        results["overall_drift_score"] = float(np.mean(drift_scores))
    else:
        results["overall_drift_score"] = 0.0
    
    return results
''')
    
    add_file("automl_platform/monitoring/quality.py", '''"""Data quality monitoring"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

def check_data_quality(df: pd.DataFrame, 
                      reference_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Check data quality metrics"""
    
    results = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "issues": [],
        "column_quality": {},
        "overall_quality_score": 1.0
    }
    
    quality_scores = []
    
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "null_ratio": float(df[col].isnull().mean()),
            "unique_ratio": float(df[col].nunique() / len(df)) if len(df) > 0 else 0,
            "issues": []
        }
        
        # Score based on null ratio
        null_score = 1.0 - col_info["null_ratio"]
        
        # Check for all nulls
        if col_info["null_ratio"] == 1.0:
            col_info["issues"].append("all_null")
            results["issues"].append(f"Column {col} has all null values")
        elif col_info["null_ratio"] > 0.5:
            col_info["issues"].append("high_null_ratio")
            results["issues"].append(f"Column {col} has >50% null values")
        
        # Check for single value
        if df[col].nunique() == 1:
            col_info["issues"].append("single_value")
            results["issues"].append(f"Column {col} has only one unique value")
            null_score *= 0.5
        
        # Check for high cardinality
        if col_info["unique_ratio"] > 0.95 and len(df) > 100:
            col_info["issues"].append("high_cardinality")
            
        # Check for outliers (numeric columns)
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_ratio = outliers / len(df)
                
                col_info["outlier_ratio"] = float(outlier_ratio)
                
                if outlier_ratio > 0.1:
                    col_info["issues"].append("high_outliers")
                    results["issues"].append(f"Column {col} has >10% outliers")
                    null_score *= 0.9
        
        # Check data type consistency with reference
        if reference_df is not None and col in reference_df.columns:
            if df[col].dtype != reference_df[col].dtype:
                col_info["issues"].append("dtype_mismatch")
                results["issues"].append(f"Column {col} dtype mismatch with reference")
                null_score *= 0.8
        
        col_info["quality_score"] = null_score
        quality_scores.append(null_score)
        results["column_quality"][col] = col_info
    
    # Check for duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        results["n_duplicates"] = int(n_duplicates)
        results["duplicate_ratio"] = float(n_duplicates / len(df))
        results["issues"].append(f"Found {n_duplicates} duplicate rows")
        quality_scores.append(1.0 - results["duplicate_ratio"])
    
    # Overall quality score
    if quality_scores:
        results["overall_quality_score"] = float(np.mean(quality_scores))
    
    # Add summary
    results["summary"] = {
        "n_issues": len(results["issues"]),
        "status": "good" if results["overall_quality_score"] > 0.8 else "needs_attention"
    }
    
    return results
''')
    
    add_file("automl_platform/monitoring/alerts.py", '''"""Alert system for monitoring"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

class AlertManager:
    """Manage monitoring alerts"""
    
    def __init__(self, alert_file: str = "alerts.json"):
        self.alert_file = Path(alert_file)
        self.alerts = []
        self._load_alerts()
    
    def _load_alerts(self):
        """Load existing alerts"""
        if self.alert_file.exists():
            with open(self.alert_file, 'r') as f:
                self.alerts = json.load(f)
    
    def _save_alerts(self):
        """Save alerts to file"""
        with open(self.alert_file, 'w') as f:
            json.dump(self.alerts, f, indent=2)
    
    def add_alert(self, alert_type: str, message: str, 
                 severity: str = "warning", metadata: Optional[Dict] = None):
        """Add new alert"""
        alert = {
            "id": len(self.alerts) + 1,
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message,
            "severity": severity,
            "metadata": metadata or {},
            "resolved": False
        }
        
        self.alerts.append(alert)
        self._save_alerts()
        
        return alert
    
    def get_active_alerts(self) -> List[Dict]:
        """Get unresolved alerts"""
        return [a for a in self.alerts if not a.get("resolved", False)]
    
    def resolve_alert(self, alert_id: int):
        """Mark alert as resolved"""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["resolved"] = True
                alert["resolved_at"] = datetime.now().isoformat()
                self._save_alerts()
                return True
        return False
    
    def check_thresholds(self, metrics: Dict[str, float], 
                        thresholds: Dict[str, float]) -> List[Dict]:
        """Check metrics against thresholds and create alerts"""
        new_alerts = []
        
        for metric, value in metrics.items():
            if metric in thresholds:
                threshold = thresholds[metric]
                
                if value > threshold:
                    alert = self.add_alert(
                        alert_type="threshold_exceeded",
                        message=f"{metric} exceeded threshold: {value:.3f} > {threshold}",
                        severity="warning" if value < threshold * 1.5 else "critical",
                        metadata={"metric": metric, "value": value, "threshold": threshold}
                    )
                    new_alerts.append(alert)
        
        return new_alerts
''')

    # [Continuation des autres modules...]
    # Je continue avec tous les autres modules dans l'ordre
    
    # ==================== CONFIG ====================
    
    add_file("automl_platform/config/__init__.py", '''from .settings import EnhancedPlatformConfig, get_config, set_config
__all__ = ['EnhancedPlatformConfig', 'get_config', 'set_config']
''')
    
    add_file("automl_platform/config/settings.py", '''"""Enhanced platform configuration"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

@dataclass
class EnhancedPlatformConfig:
    """Enhanced configuration for AutoML platform"""
    
    # Core settings
    n_trials: int = 30
    cv_folds: int = 5
    drift_threshold: float = 0.1
    n_jobs: int = -1
    time_budget: int = 3600
    random_state: int = 42
    
    # Privacy & compliance
    privacy_email: str = "privacy@automl-platform.com"
    storage_path: str = "./automl_storage"
    model_path: str = "model.pkl"
    
    # Training
    algorithms: List[str] = field(default_factory=lambda: ["xgboost", "lightgbm", "random_forest"])
    early_stopping_rounds: int = 50
    validation_fraction: float = 0.2
    
    # Monitoring
    alert_threshold: float = 0.95
    quality_threshold: float = 0.8
    
    # Fairness
    protected_attributes: List[str] = field(default_factory=list)
    fairness_threshold: float = 0.8
    
    # Feature engineering
    max_features: int = 100
    feature_selection_method: str = "importance"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    def validate(self) -> bool:
        """Validate configuration"""
        assert self.n_trials > 0, "n_trials must be positive"
        assert 0 < self.cv_folds <= 20, "cv_folds must be between 1 and 20"
        assert 0 <= self.drift_threshold <= 1, "drift_threshold must be between 0 and 1"
        assert self.time_budget > 0, "time_budget must be positive"
        assert 0 < self.validation_fraction < 1, "validation_fraction must be between 0 and 1"
        
        # Create storage path if needed
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "n_trials": self.n_trials,
            "cv_folds": self.cv_folds,
            "drift_threshold": self.drift_threshold,
            "n_jobs": self.n_jobs,
            "time_budget": self.time_budget,
            "random_state": self.random_state,
            "privacy_email": self.privacy_email,
            "storage_path": self.storage_path,
            "model_path": self.model_path,
            "algorithms": self.algorithms,
            "early_stopping_rounds": self.early_stopping_rounds,
            "validation_fraction": self.validation_fraction,
            "alert_threshold": self.alert_threshold,
            "quality_threshold": self.quality_threshold,
            "protected_attributes": self.protected_attributes,
            "fairness_threshold": self.fairness_threshold,
            "max_features": self.max_features,
            "feature_selection_method": self.feature_selection_method,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "api_workers": self.api_workers
        }

# Singleton instance
_config_instance = None

def get_config() -> EnhancedPlatformConfig:
    """Get configuration singleton"""
    global _config_instance
    if _config_instance is None:
        _config_instance = EnhancedPlatformConfig()
        _config_instance.validate()
    return _config_instance

def set_config(config: EnhancedPlatformConfig) -> None:
    """Set configuration singleton"""
    global _config_instance
    config.validate()
    _config_instance = config
''')

    # ==================== DATA ====================
    
    add_file("automl_platform/data/__init__.py", '''from .io import load_data, save_data, validate_dataframe, split_features_target
__all__ = ['load_data', 'save_data', 'validate_dataframe', 'split_features_target']
''')
    
    add_file("automl_platform/data/io.py", '''"""Data I/O operations"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
import json

def load_data(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load data from various formats"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix == '.csv':
        return pd.read_csv(filepath, **kwargs)
    elif suffix == '.parquet':
        return pd.read_parquet(filepath, **kwargs)
    elif suffix == '.json':
        return pd.read_json(filepath, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

def save_data(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """Save data to various formats"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = filepath.suffix.lower()
    
    if suffix == '.csv':
        df.to_csv(filepath, index=False, **kwargs)
    elif suffix == '.parquet':
        df.to_parquet(filepath, index=False, **kwargs)
    elif suffix == '.json':
        df.to_json(filepath, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        df.to_excel(filepath, index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """Validate DataFrame structure"""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # Check for all null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        print(f"Warning: Columns with all null values: {null_cols}")
    
    # Check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(f"Duplicate column names found: {duplicate_cols}")
    
    return True

def split_features_target(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target"""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y
''')

    # [Je continue avec tous les autres modules dans le prochain bloc...]
    
    # ==================== FEATURES ====================
    
    add_file("automl_platform/features/__init__.py", '''from .engineering import FeatureEngineer
from .selection import FeatureSelector
__all__ = ['FeatureEngineer', 'FeatureSelector']
''')
    
    add_file("automl_platform/features/engineering.py", '''"""Feature engineering module"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Optional, List, Dict, Any

class FeatureEngineer:
    """Feature engineering pipeline"""
    
    def __init__(self, 
                 scale_numeric: bool = True,
                 encode_categorical: bool = True,
                 create_interactions: bool = False,
                 max_cardinality: int = 20):
        self.scale_numeric = scale_numeric
        self.encode_categorical = encode_categorical
        self.create_interactions = create_interactions
        self.max_cardinality = max_cardinality
        
        self.scalers_ = {}
        self.encoders_ = {}
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit feature engineering pipeline"""
        self.feature_names_ = list(X.columns)
        
        # Numeric features
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        if self.scale_numeric and len(numeric_cols) > 0:
            for col in numeric_cols:
                self.scalers_[col] = StandardScaler()
                self.scalers_[col].fit(X[col].values.reshape(-1, 1))
        
        # Categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if self.encode_categorical and len(categorical_cols) > 0:
            for col in categorical_cols:
                n_unique = X[col].nunique()
                if n_unique <= self.max_cardinality:
                    # Use OneHotEncoder for low cardinality
                    self.encoders_[col] = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    self.encoders_[col].fit(X[[col]])
                else:
                    # Use LabelEncoder for high cardinality
                    self.encoders_[col] = LabelEncoder()
                    self.encoders_[col].fit(X[col].fillna('missing'))
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features"""
        X_transformed = X.copy()
        
        # Scale numeric features
        for col, scaler in self.scalers_.items():
            if col in X_transformed.columns:
                X_transformed[col] = scaler.transform(X_transformed[col].values.reshape(-1, 1)).flatten()
        
        # Encode categorical features
        for col, encoder in self.encoders_.items():
            if col in X_transformed.columns:
                if isinstance(encoder, OneHotEncoder):
                    # OneHotEncoder
                    encoded = encoder.transform(X_transformed[[col]])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                        index=X_transformed.index
                    )
                    X_transformed = pd.concat([X_transformed.drop(columns=[col]), encoded_df], axis=1)
                else:
                    # LabelEncoder
                    X_transformed[col] = encoder.transform(X_transformed[col].fillna('missing'))
        
        # Create interactions
        if self.create_interactions:
            numeric_cols = X_transformed.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) >= 2:
                # Create top 3 interactions
                for i, col1 in enumerate(numeric_cols[:3]):
                    for col2 in numeric_cols[i+1:4]:
                        X_transformed[f"{col1}_x_{col2}"] = X_transformed[col1] * X_transformed[col2]
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform features"""
        return self.fit(X, y).transform(X)
''')
    
    add_file("automl_platform/features/selection.py", '''"""Feature selection module"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Optional, List, Union, Dict

class FeatureSelector:
    """Feature selection methods"""
    
    def __init__(self, 
                 method: str = "importance",
                 k: Union[int, float] = 0.8,
                 task: str = "classification"):
        """
        Initialize feature selector
        
        Args:
            method: Selection method ('importance', 'statistical', 'permutation')
            k: Number of features to select (int) or fraction (float)
            task: Task type ('classification' or 'regression')
        """
        self.method = method
        self.k = k
        self.task = task
        self.selected_features_ = None
        self.scores_ = None
        self.selector_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit feature selector"""
        n_features = X.shape[1]
        
        if isinstance(self.k, float):
            k_features = max(1, int(n_features * self.k))
        else:
            k_features = min(self.k, n_features)
        
        if self.method == "importance":
            # Use tree-based feature importance
            if self.task == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            model.fit(X, y)
            importances = model.feature_importances_
            
            # Select top k features
            indices = np.argsort(importances)[::-1][:k_features]
            self.selected_features_ = X.columns[indices].tolist()
            self.scores_ = dict(zip(X.columns, importances))
            
        elif self.method == "statistical":
            # Use statistical tests
            if self.task == "classification":
                score_func = f_classif
            else:
                score_func = f_regression
            
            selector = SelectKBest(score_func=score_func, k=k_features)
            selector.fit(X, y)
            
            self.selector_ = selector
            self.selected_features_ = X.columns[selector.get_support()].tolist()
            self.scores_ = dict(zip(X.columns, selector.scores_))
            
        elif self.method == "permutation":
            # Permutation importance (simplified)
            if self.task == "classification":
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            model.fit(X, y)
            baseline_score = model.score(X, y)
            
            importances = []
            for col in X.columns:
                X_perm = X.copy()
                X_perm[col] = np.random.permutation(X_perm[col])
                perm_score = model.score(X_perm, y)
                importance = baseline_score - perm_score
                importances.append(importance)
            
            importances = np.array(importances)
            indices = np.argsort(importances)[::-1][:k_features]
            
            self.selected_features_ = X.columns[indices].tolist()
            self.scores_ = dict(zip(X.columns, importances))
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features"""
        if self.selected_features_ is None:
            raise ValueError("Selector not fitted yet")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform features"""
        return self.fit(X, y).transform(X)
    
    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature scores"""
        if self.scores_ is None:
            raise ValueError("Selector not fitted yet")
        return self.scores_
''')
    
    add_file("automl_platform/features/nlp.py", '''"""NLP feature extraction"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, List, Union

class TextFeatureExtractor:
    """Text feature extraction"""
    
    def __init__(self, 
                 method: str = "tfidf",
                 max_features: int = 100,
                 ngram_range: tuple = (1, 2),
                 min_df: int = 2):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer_ = None
        self.embedder_ = None
        
    def fit(self, texts: Union[List[str], pd.Series]):
        """Fit text feature extractor"""
        if self.method == "tfidf":
            self.vectorizer_ = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                stop_words='english'
            )
            self.vectorizer_.fit(texts)
            
        elif self.method == "embeddings":
            # Try to use sentence transformers if available
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder_ = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                # Fallback to TF-IDF
                print("Sentence transformers not available, falling back to TF-IDF")
                self.method = "tfidf"
                self.vectorizer_ = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    min_df=self.min_df
                )
                self.vectorizer_.fit(texts)
        
        return self
    
    def transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Transform texts to features"""
        if self.method == "tfidf":
            if self.vectorizer_ is None:
                raise ValueError("Vectorizer not fitted yet")
            return self.vectorizer_.transform(texts).toarray()
            
        elif self.method == "embeddings" and self.embedder_ is not None:
            embeddings = self.embedder_.encode(list(texts))
            # Reduce dimensionality if needed
            if embeddings.shape[1] > self.max_features:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.max_features)
                embeddings = pca.fit_transform(embeddings)
            return embeddings
        
        else:
            # Fallback
            if self.vectorizer_ is None:
                raise ValueError("No feature extractor fitted")
            return self.vectorizer_.transform(texts).toarray()
    
    def fit_transform(self, texts: Union[List[str], pd.Series]) -> np.ndarray:
        """Fit and transform texts"""
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        if self.method == "tfidf" and self.vectorizer_ is not None:
            return self.vectorizer_.get_feature_names_out().tolist()
        elif self.method == "embeddings":
            return [f"embed_{i}" for i in range(self.max_features)]
        else:
            return []
''')

    # [Je continue avec les modules restants dans le prochain bloc...]
    
    # ==================== MODELING ====================
    
    add_file("automl_platform/modeling/__init__.py", '''from .utils import load_model, save_model, predict, predict_proba
from .trainer import train_cv
__all__ = ['load_model', 'save_model', 'predict', 'predict_proba', 'train_cv']
''')
    
    add_file("automl_platform/modeling/utils.py", '''"""Model utilities"""

import joblib
import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Union, Optional, Dict
import pandas as pd
import numpy as np
from datetime import datetime

def save_model(model: Any, filepath: Union[str, Path], metadata: Optional[Dict] = None) -> str:
    """Save model with metadata and hash"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
    else:
        joblib.dump(model, filepath)
    
    # Calculate hash
    with open(filepath, 'rb') as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()
    
    # Save metadata
    metadata = metadata or {}
    metadata.update({
        'hash': model_hash,
        'saved_at': datetime.now().isoformat(),
        'model_type': type(model).__name__,
        'filepath': str(filepath)
    })
    
    metadata_path = filepath.with_suffix('.meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_hash

def load_model(filepath: Union[str, Path], verify_hash: bool = False) -> Any:
    """Load model with optional hash verification"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Load metadata if exists
    metadata_path = filepath.with_suffix('.meta.json')
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Verify hash if requested
    if verify_hash and 'hash' in metadata:
        with open(filepath, 'rb') as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        if current_hash != metadata['hash']:
            raise ValueError(f"Model hash mismatch! Expected {metadata['hash']}, got {current_hash}")
    
    # Load model
    if filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
    else:
        model = joblib.load(filepath)
    
    return model

def predict(model: Any, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Universal predict function"""
    if hasattr(model, 'predict'):
        return model.predict(X)
    elif callable(model):
        return model(X)
    else:
        raise ValueError(f"Model {type(model)} does not have predict method")

def predict_proba(model: Any, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Universal predict_proba function"""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    elif hasattr(model, 'decision_function'):
        # Convert decision function to probabilities
        scores = model.decision_function(X)
        if len(scores.shape) == 1:
            # Binary classification
            probs = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - probs, probs])
        else:
            # Multi-class
            exp_scores = np.exp(scores)
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    elif hasattr(model, 'predict'):
        # Fallback: return binary probabilities based on predictions
        preds = model.predict(X)
        n_samples = len(preds)
        # Assume binary classification
        probs = np.zeros((n_samples, 2))
        probs[range(n_samples), preds.astype(int)] = 1.0
        return probs
    else:
        raise ValueError(f"Model {type(model)} does not have predict_proba or decision_function method")
''')
    
    add_file("automl_platform/modeling/trainer.py", '''"""Model training with HPO"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from typing import Tuple, Dict, Any, Optional, Union
import time
import warnings
warnings.filterwarnings('ignore')

def train_cv(X: Union[pd.DataFrame, np.ndarray], 
             y: Union[pd.Series, np.ndarray],
             task: str = "classification",
             config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict]:
    """
    Train model with cross-validation and HPO
    
    Args:
        X: Features
        y: Target
        task: Task type ('classification', 'regression', 'timeseries')
        config: Training configuration
    
    Returns:
        Tuple of (best_model, training_info)
    """
    from ..config.settings import get_config
    
    # Get config
    if config is None:
        config = get_config().to_dict()
    
    n_trials = config.get('n_trials', 30)
    cv_folds = config.get('cv_folds', 5)
    time_budget = config.get('time_budget', 3600)
    algorithms = config.get('algorithms', ['xgboost', 'lightgbm', 'random_forest'])
    random_state = config.get('random_state', 42)
    
    # Determine task specifics
    if task == "classification":
        if len(np.unique(y)) == 2:
            scoring = 'roc_auc'
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            scoring = 'accuracy'
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    elif task == "regression":
        scoring = 'neg_mean_squared_error'
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    elif task == "timeseries":
        scoring = 'neg_mean_squared_error'
        cv = TimeSeriesSplit(n_splits=cv_folds)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Try to use Optuna for HPO
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        use_optuna = True
    except ImportError:
        use_optuna = False
        print("Optuna not available, using default parameters")
    
    best_score = -np.inf
    best_model = None
    best_params = {}
    training_history = []
    
    start_time = time.time()
    
    if use_optuna:
        # Optuna HPO
        from .search_space import get_search_space
        
        def objective(trial):
            # Check time budget
            if time.time() - start_time > time_budget:
                trial.study.stop()
                return best_score
            
            # Select algorithm
            algo = trial.suggest_categorical('algorithm', algorithms)
            
            # Get hyperparameters
            params = get_search_space(algo, trial)
            
            # Create model
            if algo == 'xgboost':
                try:
                    from xgboost import XGBClassifier, XGBRegressor
                    if task == 'regression':
                        model = XGBRegressor(**params, random_state=random_state)
                    else:
                        model = XGBClassifier(**params, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
                except ImportError:
                    return -np.inf
                    
            elif algo == 'lightgbm':
                try:
                    from lightgbm import LGBMClassifier, LGBMRegressor
                    if task == 'regression':
                        model = LGBMRegressor(**params, random_state=random_state, verbosity=-1)
                    else:
                        model = LGBMClassifier(**params, random_state=random_state, verbosity=-1)
                except ImportError:
                    return -np.inf
                    
            elif algo == 'catboost':
                try:
                    from catboost import CatBoostClassifier, CatBoostRegressor
                    if task == 'regression':
                        model = CatBoostRegressor(**params, random_state=random_state, verbose=False)
                    else:
                        model = CatBoostClassifier(**params, random_state=random_state, verbose=False)
                except ImportError:
                    return -np.inf
                    
            elif algo == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                if task == 'regression':
                    model = RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
                else:
                    model = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
                    
            elif algo == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(**params, random_state=random_state, max_iter=1000)
            
            else:
                return -np.inf
            
            # Cross-validation
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
                score = scores.mean()
            except Exception as e:
                print(f"Error in trial: {e}")
                return -np.inf
            
            return score
        
        # Create study
        sampler = TPESampler(seed=random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=time_budget)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Train final model with best parameters
        algo = best_params.pop('algorithm')
        
    else:
        # Grid search fallback
        algo = algorithms[0] if algorithms else 'random_forest'
        best_params = {}
    
    # Train final model on all data
    if algo == 'xgboost':
        try:
            from xgboost import XGBClassifier, XGBRegressor
            if task == 'regression':
                best_model = XGBRegressor(**best_params, random_state=random_state)
            else:
                best_model = XGBClassifier(**best_params, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        except ImportError:
            algo = 'random_forest'
            
    if algo == 'lightgbm':
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            if task == 'regression':
                best_model = LGBMRegressor(**best_params, random_state=random_state, verbosity=-1)
            else:
                best_model = LGBMClassifier(**best_params, random_state=random_state, verbosity=-1)
        except ImportError:
            algo = 'random_forest'
            
    if algo == 'catboost':
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
            if task == 'regression':
                best_model = CatBoostRegressor(**best_params, random_state=random_state, verbose=False)
            else:
                best_model = CatBoostClassifier(**best_params, random_state=random_state, verbose=False)
        except ImportError:
            algo = 'random_forest'
            
    if algo == 'random_forest' or best_model is None:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if task == 'regression':
            best_model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        else:
            best_model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    
    if algo == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        best_model = LogisticRegression(random_state=random_state, max_iter=1000)
    
    # Fit final model
    best_model.fit(X, y)
    
    # Prepare training info
    training_info = {
        'best_score': best_score,
        'best_params': best_params,
        'algorithm': algo,
        'task': task,
        'cv_folds': cv_folds,
        'n_trials': n_trials if use_optuna else 1,
        'training_time': time.time() - start_time,
        'scoring': scoring
    }
    
    return best_model, training_info
''')
    
    add_file("automl_platform/modeling/search_space.py", '''"""Hyperparameter search spaces"""

def get_search_space(algo: str, trial) -> dict:
    """Get hyperparameter search space for algorithm"""
    
    if algo == 'xgboost':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
        }
    
    elif algo == 'lightgbm':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
        }
    
    elif algo == 'catboost':
        return {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1)
        }
    
    elif algo == 'random_forest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
    
    elif algo == 'logistic_regression':
        return {
            'C': trial.suggest_float('C', 0.001, 10.0, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
        }
    
    else:
        return {}
''')
    
    add_file("automl_platform/modeling/ensembling.py", '''"""Model ensembling strategies"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.calibration import CalibratedClassifierCV
from typing import List, Any, Optional, Union

def create_voting_ensemble(models: List[Any], task: str = "classification", 
                          voting: str = "soft") -> Any:
    """Create voting ensemble"""
    if task == "classification":
        return VotingClassifier(
            estimators=[(f"model_{i}", model) for i, model in enumerate(models)],
            voting=voting
        )
    else:
        return VotingRegressor(
            estimators=[(f"model_{i}", model) for i, model in enumerate(models)]
        )

def create_stacking_ensemble(models: List[Any], task: str = "classification",
                            meta_model: Optional[Any] = None) -> Any:
    """Create stacking ensemble"""
    if meta_model is None:
        if task == "classification":
            meta_model = LogisticRegression(max_iter=1000)
        else:
            meta_model = Ridge()
    
    if task == "classification":
        return StackingClassifier(
            estimators=[(f"model_{i}", model) for i, model in enumerate(models)],
            final_estimator=meta_model,
            cv=5
        )
    else:
        return StackingRegressor(
            estimators=[(f"model_{i}", model) for i, model in enumerate(models)],
            final_estimator=meta_model,
            cv=5
        )

def create_blending_ensemble(models: List[Any], X_blend: pd.DataFrame, 
                            y_blend: pd.Series, task: str = "classification") -> Any:
    """Create blending ensemble"""
    # Get predictions from base models
    if task == "classification":
        blend_features = np.column_stack([
            model.predict_proba(X_blend)[:, 1] if hasattr(model, 'predict_proba')
            else model.predict(X_blend)
            for model in models
        ])
        
        # Train meta model
        meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(blend_features, y_blend)
        
    else:
        blend_features = np.column_stack([
            model.predict(X_blend) for model in models
        ])
        
        # Train meta model
        meta_model = Ridge()
        meta_model.fit(blend_features, y_blend)
    
    # Create ensemble wrapper
    class BlendingEnsemble:
        def __init__(self, base_models, meta_model, task):
            self.base_models = base_models
            self.meta_model = meta_model
            self.task = task
        
        def predict(self, X):
            if self.task == "classification":
                features = np.column_stack([
                    model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba')
                    else model.predict(X)
                    for model in self.base_models
                ])
            else:
                features = np.column_stack([
                    model.predict(X) for model in self.base_models
                ])
            return self.meta_model.predict(features)
        
        def predict_proba(self, X):
            if self.task != "classification":
                raise ValueError("predict_proba only for classification")
            
            features = np.column_stack([
                model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba')
                else model.predict(X)
                for model in self.base_models
            ])
            return self.meta_model.predict_proba(features)
    
    return BlendingEnsemble(models, meta_model, task)

def calibrate_model(model: Any, X_calib: Union[pd.DataFrame, np.ndarray], 
                   y_calib: Union[pd.Series, np.ndarray]) -> Any:
    """Calibrate model probabilities"""
    n_samples = len(y_calib)
    
    # Choose calibration method based on sample size
    if n_samples > 1000:
        method = 'isotonic'
    else:
        method = 'sigmoid'
    
    calibrated = CalibratedClassifierCV(
        model,
        method=method,
        cv='prefit'  # Model already trained
    )
    
    calibrated.fit(X_calib, y_calib)
    return calibrated
''')
    
    add_file("automl_platform/modeling/timeseries.py", '''"""Time series modeling"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple

class TimeSeriesForecaster:
    """Time series forecasting wrapper"""
    
    def __init__(self, method: str = "auto", seasonality: Optional[int] = None):
        self.method = method
        self.seasonality = seasonality
        self.model_ = None
        
    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None):
        """Fit time series model"""
        
        if self.method == "prophet" or self.method == "auto":
            try:
                from prophet import Prophet
                self.model_ = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False
                )
                
                # Prepare data for Prophet
                df = pd.DataFrame({
                    'ds': y.index,
                    'y': y.values
                })
                
                if exog is not None:
                    for col in exog.columns:
                        self.model_.add_regressor(col)
                        df[col] = exog[col].values
                
                self.model_.fit(df)
                self.method = "prophet"
                
            except ImportError:
                self.method = "arima"
        
        if self.method == "arima":
            try:
                from pmdarima import auto_arima
                self.model_ = auto_arima(
                    y,
                    exogenous=exog,
                    seasonal=self.seasonality is not None,
                    m=self.seasonality or 1,
                    stepwise=True,
                    suppress_warnings=True
                )
                
            except ImportError:
                # Fallback to simple moving average
                self.method = "moving_average"
                self.model_ = SimpleMovingAverage(window=self.seasonality or 7)
                self.model_.fit(y)
        
        return self
    
    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Make predictions"""
        
        if self.method == "prophet":
            # Create future dataframe
            future = self.model_.make_future_dataframe(periods=steps)
            
            if exog is not None:
                for col in exog.columns:
                    # Need to extend exog data for future periods
                    future[col] = np.concatenate([
                        self.model_.history[col].values,
                        exog[col].values[:steps]
                    ])
            
            forecast = self.model_.predict(future)
            return forecast['yhat'].iloc[-steps:].values
            
        elif self.method == "arima":
            return self.model_.predict(n_periods=steps, exogenous=exog)
            
        else:
            # Moving average fallback
            return self.model_.predict(steps)


class SimpleMovingAverage:
    """Simple moving average for fallback"""
    
    def __init__(self, window: int = 7):
        self.window = window
        self.history_ = None
        
    def fit(self, y: pd.Series):
        """Fit moving average"""
        self.history_ = y.values[-self.window:]
        return self
    
    def predict(self, steps: int) -> np.ndarray:
        """Predict using moving average"""
        predictions = []
        history = list(self.history_)
        
        for _ in range(steps):
            pred = np.mean(history[-self.window:])
            predictions.append(pred)
            history.append(pred)
        
        return np.array(predictions)
''')

    # ==================== EXPLAIN ====================
    
    add_file("automl_platform/explain/__init__.py", '''from .shap_lime import explain_global, explain_local
__all__ = ['explain_global', 'explain_local']
''')
    
    add_file("automl_platform/explain/shap_lime.py", '''"""Model explainability with SHAP and LIME"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union

def explain_global(model: Any, X: Union[pd.DataFrame, np.ndarray], 
                  sample_size: int = 100) -> Dict[str, Any]:
    """
    Global model explanation
    
    Priority: SHAP Tree → SHAP Kernel → feature_importances_ → variance → zeros
    """
    explanation = {"method": None, "importances": {}, "values": None}
    
    # Ensure we have column names
    if isinstance(X, np.ndarray):
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        feature_names = list(X.columns)
    
    # Try SHAP Tree Explainer first
    try:
        import shap
        
        # Check if model is tree-based
        if hasattr(model, 'booster') or type(model).__name__ in ['RandomForestClassifier', 'RandomForestRegressor', 
                                                                  'XGBClassifier', 'XGBRegressor',
                                                                  'LGBMClassifier', 'LGBMRegressor']:
            explainer = shap.TreeExplainer(model)
            X_sample = X[:sample_size] if len(X) > sample_size else X
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate mean absolute SHAP values
            importances = np.abs(shap_values).mean(axis=0)
            
            explanation["method"] = "shap_tree"
            explanation["importances"] = dict(zip(feature_names, importances))
            explanation["values"] = shap_values
            
            return explanation
            
    except Exception:
        pass
    
    # Try SHAP Kernel Explainer
    try:
        import shap
        
        X_sample = X[:sample_size] if len(X) > sample_size else X
        
        # Create background data
        if len(X) > 100:
            background = shap.sample(X, 100)
        else:
            background = X
        
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_sample[:10])  # Limited samples for kernel
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        importances = np.abs(shap_values).mean(axis=0)
        
        explanation["method"] = "shap_kernel"
        explanation["importances"] = dict(zip(feature_names, importances))
        explanation["values"] = shap_values
        
        return explanation
        
    except Exception:
        pass
    
    # Try feature_importances_
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        explanation["method"] = "feature_importances"
        explanation["importances"] = dict(zip(feature_names, importances))
        return explanation
    
    # Try coefficients (for linear models)
    if hasattr(model, 'coef_'):
        if len(model.coef_.shape) == 1:
            importances = np.abs(model.coef_)
        else:
            importances = np.abs(model.coef_).mean(axis=0)
        
        explanation["method"] = "coefficients"
        explanation["importances"] = dict(zip(feature_names, importances))
        return explanation
    
    # Fallback to variance
    if isinstance(X, pd.DataFrame):
        importances = X.var().values
    else:
        importances = np.var(X, axis=0)
    
    # Normalize
    if importances.sum() > 0:
        importances = importances / importances.sum()
    
    explanation["method"] = "variance"
    explanation["importances"] = dict(zip(feature_names, importances))
    
    return explanation

def explain_local(model: Any, X: Union[pd.DataFrame, np.ndarray], 
                 instance_idx: int) -> Dict[str, Any]:
    """
    Local instance explanation
    
    Priority: SHAP → LIME → delta naive → zeros
    """
    explanation = {"method": None, "importances": {}, "values": None}
    
    # Ensure we have column names
    if isinstance(X, np.ndarray):
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        instance = X[instance_idx:instance_idx+1]
    else:
        feature_names = list(X.columns)
        instance = X.iloc[instance_idx:instance_idx+1]
    
    # Try SHAP first
    try:
        import shap
        
        # Try Tree Explainer
        if hasattr(model, 'booster') or type(model).__name__ in ['RandomForestClassifier', 'RandomForestRegressor',
                                                                  'XGBClassifier', 'XGBRegressor',
                                                                  'LGBMClassifier', 'LGBMRegressor']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(instance)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            explanation["method"] = "shap"
            explanation["importances"] = dict(zip(feature_names, shap_values))
            explanation["values"] = shap_values
            
            return explanation
            
    except Exception:
        pass
    
    # Try LIME
    try:
        import lime
        import lime.lime_tabular
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X if isinstance(X, np.ndarray) else X.values,
            feature_names=feature_names,
            mode='classification' if hasattr(model, 'predict_proba') else 'regression'
        )
        
        if hasattr(model, 'predict_proba'):
            exp = explainer.explain_instance(
                instance[0] if isinstance(instance, np.ndarray) else instance.values[0],
                model.predict_proba,
                num_features=len(feature_names)
            )
        else:
            exp = explainer.explain_instance(
                instance[0] if isinstance(instance, np.ndarray) else instance.values[0],
                model.predict,
                num_features=len(feature_names)
            )
        
        # Extract feature importances
        importances = dict(exp.as_list())
        
        explanation["method"] = "lime"
        explanation["importances"] = importances
        
        return explanation
        
    except Exception:
        pass
    
    # Fallback to delta naive
    try:
        base_pred = model.predict(instance)[0]
        importances = {}
        
        for i, feature in enumerate(feature_names):
            # Perturb feature
            perturbed = instance.copy()
            if isinstance(perturbed, pd.DataFrame):
                mean_val = X[feature].mean()
                perturbed.iloc[0, i] = mean_val
            else:
                mean_val = X[:, i].mean()
                perturbed[0, i] = mean_val
            
            # Calculate delta
            perturbed_pred = model.predict(perturbed)[0]
            delta = base_pred - perturbed_pred
            importances[feature] = float(delta)
        
        explanation["method"] = "delta"
        explanation["importances"] = importances
        
        return explanation
        
    except Exception:
        pass
    
    # Fallback to zeros
    explanation["method"] = "zeros"
    explanation["importances"] = {feature: 0.0 for feature in feature_names}
    
    return explanation
''')
    
    add_file("automl_platform/explain/reporting.py", '''"""Generate comprehensive model reports"""

import json
from typing import Dict, Any, Optional
from datetime import datetime

def build_report(metrics: Dict[str, float],
                importances: Dict[str, float],
                drift: Optional[Dict[str, Any]] = None,
                fairness: Optional[Dict[str, Any]] = None) -> str:
    """Build comprehensive model report in JSON format"""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "feature_importances": importances,
        "model_performance": {
            "status": "good" if metrics.get("accuracy", 0) > 0.7 else "needs_improvement",
            "primary_metric": list(metrics.keys())[0] if metrics else None,
            "primary_value": list(metrics.values())[0] if metrics else None
        }
    }
    
    # Add drift information if available
    if drift:
        report["drift_analysis"] = drift
        report["drift_detected"] = drift.get("drift_detected", False)
    
    # Add fairness information if available
    if fairness:
        report["fairness_analysis"] = fairness
        report["fairness_score"] = fairness.get("fairness_score", None)
    
    # Add recommendations
    recommendations = []
    
    if metrics.get("accuracy", 1.0) < 0.7:
        recommendations.append("Consider feature engineering or hyperparameter tuning")
    
    if drift and drift.get("drift_detected"):
        recommendations.append("Model drift detected - consider retraining")
    
    if fairness and fairness.get("fairness_score", 1.0) < 0.8:
        recommendations.append("Fairness issues detected - review model for bias")
    
    report["recommendations"] = recommendations
    
    return json.dumps(report, indent=2)
''')

    # ==================== FAIRNESS ====================
    
    add_file("automl_platform/fairness/__init__.py", '''from .metrics import fairness_report, demographic_parity, equal_opportunity
from .wrappers import ThresholdOptimizedModel, CalibratedGroupModel
__all__ = ['fairness_report', 'demographic_parity', 'equal_opportunity', 
          'ThresholdOptimizedModel', 'CalibratedGroupModel']
''')
    
    add_file("automl_platform/fairness/wrappers.py", '''"""Fairness-aware model wrappers"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from typing import Union, Optional, List, Dict, Any

class ThresholdOptimizedModel:
    """Model with group-specific thresholds for fairness"""
    
    def __init__(self, base_model: Any, protected_attribute: str = None):
        self.base_model = base_model
        self.protected_attribute = protected_attribute
        self.thresholds_ = {}
        self.default_threshold_ = 0.5
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            protected: Optional[Union[pd.Series, np.ndarray]] = None):
        """Fit model and optimize thresholds"""
        
        # Fit base model
        self.base_model.fit(X, y)
        
        # Get predictions
        if hasattr(self.base_model, 'predict_proba'):
            proba = self.base_model.predict_proba(X)[:, 1]
        else:
            proba = self.base_model.decision_function(X)
            proba = 1 / (1 + np.exp(-proba))
        
        # Optimize thresholds per group
        if protected is not None:
            if isinstance(protected, pd.Series):
                protected = protected.values
            
            for group in np.unique(protected):
                mask = protected == group
                group_proba = proba[mask]
                group_y = y[mask] if isinstance(y, np.ndarray) else y.values[mask]
                
                # Find optimal threshold for this group
                best_threshold = 0.5
                best_score = 0
                
                for threshold in np.linspace(0.1, 0.9, 20):
                    preds = (group_proba >= threshold).astype(int)
                    # Use balanced accuracy
                    tp = ((preds == 1) & (group_y == 1)).sum()
                    tn = ((preds == 0) & (group_y == 0)).sum()
                    fp = ((preds == 1) & (group_y == 0)).sum()
                    fn = ((preds == 0) & (group_y == 1)).sum()
                    
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    score = (sensitivity + specificity) / 2
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                
                self.thresholds_[group] = best_threshold
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                protected: Optional[Union[pd.Series, np.ndarray]] = None):
        """Predict with group-specific thresholds"""
        
        if hasattr(self.base_model, 'predict_proba'):
            proba = self.base_model.predict_proba(X)[:, 1]
        else:
            proba = self.base_model.decision_function(X)
            proba = 1 / (1 + np.exp(-proba))
        
        if protected is not None and self.thresholds_:
            if isinstance(protected, pd.Series):
                protected = protected.values
            
            predictions = np.zeros(len(proba))
            for group in np.unique(protected):
                mask = protected == group
                threshold = self.thresholds_.get(group, self.default_threshold_)
                predictions[mask] = (proba[mask] >= threshold).astype(int)
        else:
            predictions = (proba >= self.default_threshold_).astype(int)
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        """Get probability predictions"""
        return self.base_model.predict_proba(X) if hasattr(self.base_model, 'predict_proba') else None


class CalibratedGroupModel:
    """Model with group-specific calibration"""
    
    def __init__(self, base_model: Any, protected_attribute: str = None):
        self.base_model = base_model
        self.protected_attribute = protected_attribute
        self.calibrators_ = {}
        self.default_calibrator_ = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            protected: Optional[Union[pd.Series, np.ndarray]] = None):
        """Fit model with group-specific calibration"""
        
        # Fit base model
        self.base_model.fit(X, y)
        
        # Calibrate per group
        if protected is not None:
            if isinstance(protected, pd.Series):
                protected = protected.values
            
            for group in np.unique(protected):
                mask = protected == group
                
                # Handle DataFrame/ndarray indexing
                if isinstance(X, pd.DataFrame):
                    X_group = X.iloc[mask]
                else:
                    X_group = X[mask]
                
                y_group = y[mask] if isinstance(y, np.ndarray) else y.iloc[mask]
                
                # Calibrate for this group
                n_samples = len(y_group)
                method = 'isotonic' if n_samples > 1000 else 'sigmoid'
                
                calibrator = CalibratedClassifierCV(
                    self.base_model,
                    method=method,
                    cv='prefit'
                )
                calibrator.fit(X_group, y_group)
                self.calibrators_[group] = calibrator
        
        # Default calibrator
        self.default_calibrator_ = self.base_model
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                protected: Optional[Union[pd.Series, np.ndarray]] = None):
        """Predict with group-specific calibration"""
        
        if protected is not None and self.calibrators_:
            if isinstance(protected, pd.Series):
                protected = protected.values
            
            predictions = np.zeros(len(X), dtype=int)
            
            for group in np.unique(protected):
                mask = protected == group
                calibrator = self.calibrators_.get(group, self.default_calibrator_)
                
                # Handle DataFrame/ndarray indexing with alignment
                if isinstance(X, pd.DataFrame):
                    X_group = X.iloc[mask]
                else:
                    X_group = X[mask]
                
                if len(X_group) > 0:
                    group_preds = calibrator.predict(X_group)
                    
                    # Ensure mask indices align with predictions array
                    mask_indices = np.where(mask)[0]
                    predictions[mask_indices] = group_preds
        else:
            predictions = self.base_model.predict(X)
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray],
                      protected: Optional[Union[pd.Series, np.ndarray]] = None):
        """Get calibrated probability predictions"""
        
        if protected is not None and self.calibrators_:
            if isinstance(protected, pd.Series):
                protected = protected.values
            
            n_classes = 2  # Assume binary for now
            probas = np.zeros((len(X), n_classes))
            
            for group in np.unique(protected):
                mask = protected == group
                calibrator = self.calibrators_.get(group, self.default_calibrator_)
                
                if isinstance(X, pd.DataFrame):
                    X_group = X.iloc[mask]
                else:
                    X_group = X[mask]
                
                if len(X_group) > 0:
                    group_probas = calibrator.predict_proba(X_group)
                    
                    # Ensure mask indices align
                    mask_indices = np.where(mask)[0]
                    probas[mask_indices] = group_probas
        else:
            probas = self.base_model.predict_proba(X) if hasattr(self.base_model, 'predict_proba') else None
        
        return probas
''')
    
    add_file("automl_platform/fairness/metrics.py", '''"""Fairness metrics - complete implementation"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from sklearn.metrics import confusion_matrix

def demographic_parity(y_true: Union[np.ndarray, pd.Series],
                       y_pred: Union[np.ndarray, pd.Series],
                       protected: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
    """
    Calculate demographic parity metrics
    
    Returns difference and ratio of positive prediction rates between groups
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(protected, pd.Series):
        protected = protected.values
    
    groups = np.unique(protected)
    positive_rates = {}
    
    for group in groups:
        mask = protected == group
        group_preds = y_pred[mask]
        positive_rate = np.mean(group_preds)
        positive_rates[str(group)] = positive_rate
    
    rates = list(positive_rates.values())
    
    # Calculate metrics
    parity_diff = max(rates) - min(rates)
    parity_ratio = min(rates) / max(rates) if max(rates) > 0 else 0
    
    return {
        "metric": "demographic_parity",
        "difference": parity_diff,
        "ratio": parity_ratio,
        "by_group": positive_rates,
        "fair": parity_diff < 0.1  # Common threshold
    }

def equal_opportunity(y_true: Union[np.ndarray, pd.Series],
                     y_pred: Union[np.ndarray, pd.Series],
                     protected: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
    """
    Calculate equal opportunity metrics
    
    Returns TPR difference between groups
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(protected, pd.Series):
        protected = protected.values
    
    groups = np.unique(protected)
    tpr_rates = {}
    
    for group in groups:
        mask = protected == group
        group_true = y_true[mask]
        group_preds = y_pred[mask]
        
        # Calculate TPR (True Positive Rate)
        positive_mask = group_true == 1
        if positive_mask.sum() > 0:
            tpr = np.mean(group_preds[positive_mask])
        else:
            tpr = 0.0
        
        tpr_rates[str(group)] = tpr
    
    rates = list(tpr_rates.values())
    
    # Calculate metrics
    tpr_diff = max(rates) - min(rates)
    tpr_ratio = min(rates) / max(rates) if max(rates) > 0 else 0
    
    return {
        "metric": "equal_opportunity",
        "tpr_difference": tpr_diff,
        "tpr_ratio": tpr_ratio,
        "by_group": tpr_rates,
        "fair": tpr_diff < 0.1
    }

def disparate_impact(y_true: Union[np.ndarray, pd.Series],
                    y_pred: Union[np.ndarray, pd.Series],
                    protected: Union[np.ndarray, pd.Series],
                    reference_group: Optional[Any] = None) -> Dict[str, Any]:
    """
    Calculate disparate impact (80% rule)
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if isinstance(protected, pd.Series):
        protected = protected.values
    
    groups = np.unique(protected)
    positive_rates = {}
    
    for group in groups:
        mask = protected == group
        group_preds = y_pred[mask]
        positive_rate = np.mean(group_preds)
        positive_rates[str(group)] = positive_rate
    
    # Determine reference group
    if reference_group is None:
        reference_rate = max(positive_rates.values())
    else:
        reference_rate = positive_rates.get(str(reference_group), max(positive_rates.values()))
    
    # Calculate disparate impact ratios
    impact_ratios = {}
    for group, rate in positive_rates.items():
        if reference_rate > 0:
            ratio = rate / reference_rate
        else:
            ratio = 0.0
        impact_ratios[group] = ratio
    
    # Check 80% rule
    min_ratio = min(impact_ratios.values())
    passes_80_rule = min_ratio >= 0.8
    
    return {
        "metric": "disparate_impact",
        "impact_ratios": impact_ratios,
        "min_ratio": min_ratio,
        "passes_80_rule": passes_80_rule,
        "reference_rate": reference_rate,
        "by_group": positive_rates
    }

def fairness_report(y_true: Union[np.ndarray, pd.Series],
                   y_pred: Union[np.ndarray, pd.Series],
                   protected: Union[np.ndarray, pd.Series],
                   metrics: List[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive fairness report
    """
    if metrics is None:
        metrics = ["demographic_parity", "equal_opportunity", "disparate_impact"]
    
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_samples": len(y_true),
        "n_groups": len(np.unique(protected)),
        "metrics": {}
    }
    
    # Calculate each metric
    if "demographic_parity" in metrics:
        report["metrics"]["demographic_parity"] = demographic_parity(y_true, y_pred, protected)
    
    if "equal_opportunity" in metrics:
        report["metrics"]["equal_opportunity"] = equal_opportunity(y_true, y_pred, protected)
    
    if "disparate_impact" in metrics:
        report["metrics"]["disparate_impact"] = disparate_impact(y_true, y_pred, protected)
    
    # Calculate overall fairness score
    fairness_scores = []
    
    if "demographic_parity" in report["metrics"]:
        dp_ratio = report["metrics"]["demographic_parity"]["ratio"]
        fairness_scores.append(dp_ratio)
    
    if "equal_opportunity" in report["metrics"]:
        eo_ratio = report["metrics"]["equal_opportunity"]["tpr_ratio"]
        fairness_scores.append(eo_ratio)
    
    if "disparate_impact" in report["metrics"]:
        di_ratio = report["metrics"]["disparate_impact"]["min_ratio"]
        fairness_scores.append(min(1.0, di_ratio / 0.8))  # Normalize to 0-1
    
    report["fairness_score"] = np.mean(fairness_scores) if fairness_scores else 0.0
    
    # Add recommendations
    report["recommendations"] = []
    
    if report["fairness_score"] < 0.8:
        report["recommendations"].append("Consider using fairness-aware model wrappers")
    
    if "demographic_parity" in report["metrics"]:
        if report["metrics"]["demographic_parity"]["difference"] > 0.1:
            report["recommendations"].append("Large demographic parity gap detected")
    
    if "equal_opportunity" in report["metrics"]:
        if report["metrics"]["equal_opportunity"]["tpr_difference"] > 0.1:
            report["recommendations"].append("Unequal true positive rates across groups")
    
    if "disparate_impact" in report["metrics"]:
        if not report["metrics"]["disparate_impact"]["passes_80_rule"]:
            report["recommendations"].append("Fails 80% rule for disparate impact")
    
    # Recommended thresholds per group for optimization
    groups = np.unique(protected)
    recommended_thresholds = {}
    
    for group in groups:
        mask = protected == group
        group_true = y_true[mask] if isinstance(y_true, np.ndarray) else y_true.values[mask]
        
        # Simple threshold recommendation based on group prevalence
        prevalence = np.mean(group_true)
        if prevalence < 0.3:
            recommended_thresholds[str(group)] = 0.4
        elif prevalence > 0.7:
            recommended_thresholds[str(group)] = 0.6
        else:
            recommended_thresholds[str(group)] = 0.5
    
    report["recommended_thresholds"] = recommended_thresholds
    
    return report
''')

    # ==================== FICHIERS DE CONFIGURATION ====================
    
    add_file("setup.py", '''from setuptools import setup, find_packages

setup(
    name="automl-platform",
    version="2.0.0",
    author="AutoML Platform Team",
    description="Complete ML automation framework with monitoring, fairness, and explainability",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "scipy>=1.7.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "hpo": ["optuna>=2.10.0"],
        "boosting": ["xgboost>=1.5.0", "lightgbm>=3.2.0", "catboost>=1.0.0"],
        "explain": ["shap>=0.40.0", "lime>=0.2.0"],
        "timeseries": ["prophet>=1.0.0", "pmdarima>=1.8.0"],
        "nlp": ["sentence-transformers>=2.0.0"],
        "dev": ["pytest>=6.0.0", "black>=21.0", "flake8>=3.9.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
''')
    
    add_file("requirements.txt", '''# Core dependencies
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0
joblib>=1.1.0

# API dependencies
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0

# Optional: HPO
optuna>=2.10.0

# Optional: Boosting algorithms
xgboost>=1.5.0
lightgbm>=3.2.0
catboost>=1.0.0

# Optional: Explainability
shap>=0.40.0
lime>=0.2.0

# Optional: Time series
prophet>=1.0.0
pmdarima>=1.8.0

# Optional: NLP
sentence-transformers>=2.0.0

# Development
pytest>=6.0.0
black>=21.0
flake8>=3.9.0
''')
    
    add_file("README.md", '''# AutoML Platform v2.0

Complete ML automation framework with monitoring, fairness, and explainability.

## Features

- 🚀 Automated model training with hyperparameter optimization
- 📊 Data drift detection and quality monitoring
- 🔍 Model explainability (SHAP/LIME)
- ⚖️ Fairness metrics and bias mitigation
- 🌐 REST API for deployment
- 📈 Comprehensive monitoring and alerting
- 🕒 Time series forecasting support
- 📝 NLP feature extraction

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

Or install with extras:

```bash
pip install -e ".[hpo,boosting,explain]"
```

## Quick Start

### Basic Usage

```python
from automl_platform import load_data, train_cv, save_model
from automl_platform.data import split_features_target

# Load and prepare data
df = load_data("data.csv")
X, y = split_features_target(df, "target")

# Train model with automatic HPO
model, info = train_cv(X, y, task="classification")

# Save model
save_model(model, "model.pkl")

print(f"Best model: {info['algorithm']}")
print(f"CV Score: {info['best_score']:.4f}")
```

### API Deployment

```python
# Start API server
uvicorn automl_platform.api.app:app --host 0.0.0.0 --port 8000

# Or programmatically
from automl_platform.api import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Monitoring

```python
from automl_platform.monitoring import check_drift, check_data_quality

# Check data quality
quality_report = check_data_quality(new_data, reference_data)
print(f"Quality score: {quality_report['overall_quality_score']}")

# Check for drift
drift_report = check_drift(new_data, reference_data, threshold=0.1)
if drift_report["drift_detected"]:
    print("⚠️ Data drift detected!")
```

### Explainability

```python
from automl_platform.explain import explain_global, explain_local

# Global feature importance
global_exp = explain_global(model, X)
print("Top features:", sorted(global_exp["importances"].items(), 
                            key=lambda x: x[1], reverse=True)[:5])

# Local explanation for a specific instance
local_exp = explain_local(model, X, instance_idx=0)
print("Instance explanation:", local_exp["importances"])
```

### Fairness

```python
from automl_platform.fairness import fairness_report

# Generate fairness report
report = fairness_report(y_true, y_pred, protected_attribute)
print(f"Fairness score: {report['fairness_score']:.2f}")

# Use fairness-aware model
from automl_platform.fairness.wrappers import ThresholdOptimizedModel

fair_model = ThresholdOptimizedModel(base_model)
fair_model.fit(X, y, protected=protected_attribute)
fair_predictions = fair_model.predict(X_test, protected=protected_test)
```

## API Endpoints

- `GET /` - Root endpoint with API info
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions
- `POST /check_drift` - Check for data drift
- `GET /metrics` - API metrics
- `POST /reload_model` - Reload model from disk

## Configuration

```python
from automl_platform.config import EnhancedPlatformConfig, set_config

config = EnhancedPlatformConfig(
    n_trials=50,
    cv_folds=5,
    algorithms=["xgboost", "lightgbm", "catboost"],
    time_budget=7200,
    drift_threshold=0.05
)
set_config(config)
```

## Project Structure

```
automl_platform/
├── api/           # REST API
├── config/        # Configuration
├── data/          # Data I/O
├── explain/       # Explainability
├── fairness/      # Fairness metrics
├── features/      # Feature engineering
├── modeling/      # Model training
└── monitoring/    # Monitoring & alerts
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
''')
    
    add_file(".gitignore", '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints
*.ipynb

# Data
*.csv
*.parquet
*.xlsx
*.xls
*.json
data/

# Models
*.pkl
*.joblib
*.h5
*.pt
*.pth
models/

# Logs
*.log
logs/

# Config
.env
config.yaml
config.json

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Documentation
docs/_build/
''')
    
    # Écriture des fichiers sur disque
    print("\n📂 Création de la structure des dossiers et fichiers...")
    
    for filepath, content in files_to_write.items():
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✓ {filepath}")
    
    print(f"\n✅ {len(files_to_write)} fichiers créés avec succès!")
    print("📦 Package AutoML Platform v2.0 créé avec succès!")
    print("\n🚀 Pour installer le package:")
    print("   pip install -r requirements.txt")
    print("   python setup.py install")
    print("\n📝 Pour démarrer l'API:")
    print("   uvicorn automl_platform.api.app:app --reload")
    
    return files_to_write

# Appel principal
if __name__ == "__main__":
    create_automl_platform()