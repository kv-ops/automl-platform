"""
Enhanced FastAPI application for AutoML Platform
Production-ready with rate limiting, monitoring, and comprehensive endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks, WebSocket, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
import json
import asyncio
import aiofiles
from datetime import datetime, timedelta
import hashlib
import jwt
import os
from pathlib import Path
import uuid
import logging
from io import BytesIO
import time

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
import uvicorn

# AutoML Platform imports - Updated paths
from automl_platform.config import AutoMLConfig, load_config
from automl_platform.orchestrator import AutoMLOrchestrator
from automl_platform.data_prep import DataPreprocessor, validate_data
from automl_platform.model_selection import get_available_models, get_param_grid, get_cv_splitter
from automl_platform.metrics import calculate_metrics, detect_task
from automl_platform.feature_engineering import AutoFeatureEngineer
from automl_platform.ensemble import AutoMLEnsemble, create_diverse_ensemble
from automl_platform.inference import load_pipeline, predict, predict_proba, save_predictions
from automl_platform.data_quality_agent import IntelligentDataQualityAgent, DataQualityAssessment

# Storage and monitoring
from automl_platform.storage import StorageManager
from automl_platform.monitoring import MonitoringService, DataQualityMonitor

# LLM integration
from automl_platform.llm import AutoMLLLMAssistant, DataCleaningAgent
from automl_platform.prompts import PromptTemplates, PromptOptimizer

# Infrastructure and billing
from automl_platform.api.infrastructure import TenantManager, SecurityManager, DeploymentManager
from automl_platform.api.billing import BillingManager, UsageTracker, PlanType, BillingPeriod

# Data connectors
from automl_platform.api.connectors import ConnectorFactory, ConnectionConfig

# Streaming
from automl_platform.api.streaming import StreamingOrchestrator, StreamConfig, MLStreamProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

config = load_config(os.getenv("CONFIG_PATH", "config.yaml"))

# ============================================================================
# Metrics with Custom Registry to Avoid Duplicates
# ============================================================================

# Create a custom registry for this instance
metrics_registry = CollectorRegistry()

# Create metrics with the custom registry
request_count = Counter(
    'automl_api_requests_total', 
    'Total API requests', 
    ['method', 'endpoint', 'status'],
    registry=metrics_registry
)

request_duration = Histogram(
    'automl_api_request_duration_seconds', 
    'API request duration', 
    ['method', 'endpoint'],
    registry=metrics_registry
)

active_models = Gauge(
    'automl_active_models', 
    'Number of active models',
    registry=metrics_registry
)

training_jobs = Gauge(
    'automl_training_jobs', 
    'Number of training jobs', 
    ['status'],
    registry=metrics_registry
)

prediction_count = Counter(
    'automl_predictions_total', 
    'Total predictions made',
    registry=metrics_registry
)

upload_size = Histogram(
    'automl_upload_size_bytes', 
    'Size of uploaded files',
    registry=metrics_registry
)

# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting AutoML API...")
    
    # Initialize storage with proper parameters based on backend
    if config.storage.backend == "local":
        app.state.storage = StorageManager(backend="local")
    elif config.storage.backend == "s3":
        app.state.storage = StorageManager(
            backend="s3",
            endpoint=config.storage.endpoint,
            access_key=config.storage.access_key,
            secret_key=config.storage.secret_key
        )
    elif config.storage.backend == "gcs":
        app.state.storage = StorageManager(
            backend="gcs",
            project_id=getattr(config.storage, 'project_id', None),
            credentials_path=getattr(config.storage, 'credentials_path', None)
        )
    else:
        app.state.storage = None
    
    # Initialize monitoring
    app.state.monitoring = MonitoringService(app.state.storage) if config.monitoring.enabled else None
    
    # Initialize infrastructure components
    app.state.tenant_manager = TenantManager(db_url=getattr(config, 'database_url', 'sqlite:///automl.db'))
    app.state.security_manager = SecurityManager(secret_key=getattr(config, 'secret_key', 'default-secret'))
    app.state.billing_manager = BillingManager(tenant_manager=app.state.tenant_manager)
    app.state.usage_tracker = UsageTracker(billing_manager=app.state.billing_manager)
    
    # Initialize deployment manager
    app.state.deployment_manager = DeploymentManager(app.state.tenant_manager)
    
    # Initialize LLM assistant if configured
    if hasattr(config, 'llm') and config.llm.enabled:
        llm_config = {
            'provider': config.llm.provider,
            'api_key': config.llm.api_key,
            'model_name': config.llm.model_name,
            'enable_rag': config.llm.enable_rag,
            'cache_responses': config.llm.cache_responses
        }
        app.state.llm_assistant = AutoMLLLMAssistant(llm_config)
    else:
        app.state.llm_assistant = None
    
    # Initialize data quality agent
    app.state.quality_agent = IntelligentDataQualityAgent(llm_provider=app.state.llm_assistant.llm if app.state.llm_assistant else None)
    
    # Initialize orchestrators pool
    app.state.orchestrators = {}
    
    # Initialize WebSocket connections
    app.state.websocket_connections = {}
    
    logger.info("AutoML API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AutoML API...")
    
    # Clean up resources
    if app.state.monitoring:
        app.state.monitoring.save_monitoring_data()
    
    # Close WebSocket connections
    for ws in app.state.websocket_connections.values():
        await ws.close()
    
    logger.info("AutoML API shutdown complete")

# ============================================================================
# Application Setup
# ============================================================================

# Create FastAPI app
app = FastAPI(
    title="AutoML Platform API",
    description="Production-ready AutoML platform with monitoring and storage",
    version="2.0.0",
    docs_url="/docs" if getattr(config.api, 'enable_docs', True) else None,
    redoc_url="/redoc" if getattr(config.api, 'enable_docs', True) else None,
    lifespan=lifespan
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
if getattr(config.api, 'enable_cors', True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=getattr(config.api, 'cors_origins', ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ============================================================================
# Security
# ============================================================================

security = HTTPBearer()

def create_token(user_id: str) -> str:
    """Create JWT token"""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(minutes=getattr(config.api, 'jwt_expiration_minutes', 60))
    }
    return jwt.encode(payload, getattr(config.api, 'jwt_secret', 'secret'), algorithm=getattr(config.api, 'jwt_algorithm', 'HS256'))

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token"""
    if not getattr(config.api, 'enable_auth', False):
        return "anonymous"
    
    token = credentials.credentials
    try:
        payload = jwt.decode(token, getattr(config.api, 'jwt_secret', 'secret'), algorithms=[getattr(config.api, 'jwt_algorithm', 'HS256')])
        return payload["user_id"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================================================================
# Pydantic Models
# ============================================================================

class TrainRequest(BaseModel):
    """Training request model"""
    experiment_name: Optional[str] = Field(None, description="Name for the experiment")
    task: Optional[str] = Field("auto", description="Task type: classification, regression, auto")
    algorithms: Optional[List[str]] = Field(None, description="Algorithms to use")
    max_runtime_seconds: Optional[int] = Field(3600, description="Maximum runtime in seconds")
    optimize_metric: Optional[str] = Field("auto", description="Metric to optimize")
    validation_split: Optional[float] = Field(0.2, description="Validation split ratio")
    enable_monitoring: Optional[bool] = Field(True, description="Enable monitoring")
    enable_feature_engineering: Optional[bool] = Field(True, description="Enable feature engineering")

class PredictRequest(BaseModel):
    """Prediction request model"""
    model_id: str = Field(..., description="Model ID to use for prediction")
    data: Dict[str, Any] = Field(..., description="Input data for prediction")
    track: Optional[bool] = Field(True, description="Track predictions for monitoring")

class ModelInfo(BaseModel):
    """Model information response"""
    model_id: str
    algorithm: str
    metrics: Dict[str, float]
    version: str
    created_at: str
    status: str

class ExperimentInfo(BaseModel):
    """Experiment information response"""
    experiment_id: str
    status: str
    progress: float
    models_trained: int
    best_score: Optional[float]
    elapsed_time: float
    eta: Optional[float]

class DataQualityReport(BaseModel):
    """Data quality report"""
    quality_score: float
    issues: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    statistics: Dict[str, Any]

class FeatureEngineeringRequest(BaseModel):
    """Feature engineering request"""
    dataset_id: str
    target_column: str
    max_features: int = 20
    feature_types: str = "all"

class TenantRequest(BaseModel):
    """Tenant creation request"""
    name: str
    plan: str = "free"
    features: Optional[Dict[str, bool]] = None

class ConnectorRequest(BaseModel):
    """Data connector request"""
    connection_type: str
    connection_config: Dict[str, Any]
    query: Optional[str] = None
    table_name: Optional[str] = None

# ============================================================================
# Middleware
# ============================================================================

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Add metrics to all requests"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(duration)
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    
    return response

# ============================================================================
# Health & Monitoring Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "environment": getattr(config, 'environment', 'development')
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        BytesIO(generate_latest(metrics_registry)),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/api/v1/status")
@limiter.limit("10/minute")
async def get_status(request: Request, user_id: str = Depends(verify_token)):
    """Get system status"""
    return {
        "status": "operational",
        "active_experiments": len(app.state.orchestrators),
        "models_available": active_models._value.get() if hasattr(active_models._value, 'get') else 0,
        "storage_backend": config.storage.backend if hasattr(config, 'storage') else 'none',
        "monitoring_enabled": config.monitoring.enabled if hasattr(config, 'monitoring') else False,
        "user_id": user_id
    }

# ============================================================================
# Data Management Endpoints
# ============================================================================

@app.post("/api/v1/data/upload")
@limiter.limit("5/minute")
async def upload_data(
    request: Request,
    file: UploadFile = File(...),
    dataset_name: Optional[str] = None,
    user_id: str = Depends(verify_token)
):
    """Upload dataset"""
    # Validate file
    allowed_extensions = getattr(config.api, 'allowed_extensions', ['.csv', '.parquet', '.json', '.xlsx'])
    if not file.filename.endswith(tuple(allowed_extensions)):
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed_extensions}")
    
    # Check file size
    contents = await file.read()
    file_size = len(contents)
    
    max_upload_size = getattr(config.api, 'max_upload_size_mb', 100) * 1024 * 1024
    if file_size > max_upload_size:
        raise HTTPException(413, f"File too large. Maximum size: {max_upload_size // (1024*1024)} MB")
    
    upload_size.observe(file_size)
    
    # Parse data
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(contents))
        elif file.filename.endswith('.json'):
            df = pd.read_json(BytesIO(contents))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(BytesIO(contents))
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")
    
    # Generate dataset ID
    dataset_id = dataset_name or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"
    
    # Save to storage
    if app.state.storage:
        try:
            path = app.state.storage.save_dataset(df, dataset_id, tenant_id=user_id)
            logger.info(f"Dataset saved: {dataset_id}")
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise HTTPException(500, "Failed to save dataset")
    
    # Data quality check
    quality_report = None
    if app.state.quality_agent:
        try:
            assessment = app.state.quality_agent.assess(df)
            quality_report = {
                "quality_score": assessment.quality_score,
                "alerts": assessment.alerts[:5],  # Limit for response size
                "warnings": assessment.warnings[:5]
            }
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
    
    return {
        "dataset_id": dataset_id,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "quality_report": quality_report,
        "storage_path": path if app.state.storage else None
    }

@app.get("/api/v1/data/{dataset_id}")
@limiter.limit("10/minute")
async def get_dataset_info(
    request: Request,
    dataset_id: str,
    user_id: str = Depends(verify_token)
):
    """Get dataset information"""
    if not app.state.storage:
        raise HTTPException(503, "Storage not configured")
    
    try:
        df = app.state.storage.load_dataset(dataset_id, tenant_id=user_id)
        return {
            "dataset_id": dataset_id,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "head": df.head(10).to_dict('records'),
            "statistics": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    except Exception as e:
        raise HTTPException(500, f"Failed to load dataset: {str(e)}")

@app.post("/api/v1/data/{dataset_id}/quality")
@limiter.limit("5/minute")
async def check_data_quality(
    request: Request,
    dataset_id: str,
    user_id: str = Depends(verify_token)
) -> DataQualityReport:
    """Check data quality"""
    if not app.state.storage:
        raise HTTPException(503, "Storage not configured")
    
    try:
        df = app.state.storage.load_dataset(dataset_id, tenant_id=user_id)
        assessment = app.state.quality_agent.assess(df)
        
        return DataQualityReport(
            quality_score=assessment.quality_score,
            issues=assessment.alerts,
            warnings=assessment.warnings,
            statistics={
                "rows": assessment.statistics.get("rows", len(df)),
                "columns": assessment.statistics.get("columns", len(df.columns))
            }
        )
    except Exception as e:
        raise HTTPException(500, f"Quality check failed: {str(e)}")

# ============================================================================
# Feature Engineering Endpoints
# ============================================================================

@app.post("/api/v1/features/engineer")
@limiter.limit("5/minute")
async def engineer_features(
    request: Request,
    feature_request: FeatureEngineeringRequest,
    user_id: str = Depends(verify_token)
):
    """Automatically engineer features"""
    if not app.state.storage:
        raise HTTPException(503, "Storage not configured")
    
    try:
        # Load dataset
        df = app.state.storage.load_dataset(feature_request.dataset_id, tenant_id=user_id)
        
        # Initialize feature engineer
        engineer = AutoFeatureEngineer(
            max_features=feature_request.max_features,
            feature_types=feature_request.feature_types,
            task='auto'
        )
        
        # Separate features and target
        if feature_request.target_column in df.columns:
            X = df.drop(columns=[feature_request.target_column])
            y = df[feature_request.target_column]
        else:
            X = df
            y = None
        
        # Engineer features
        engineer.fit(X, y)
        X_engineered = engineer.transform(X)
        
        # Create new dataset
        if y is not None:
            df_engineered = pd.concat([X_engineered, y], axis=1)
        else:
            df_engineered = X_engineered
        
        # Save engineered dataset
        new_dataset_id = f"{feature_request.dataset_id}_engineered"
        path = app.state.storage.save_dataset(df_engineered, new_dataset_id, tenant_id=user_id)
        
        return {
            "original_dataset_id": feature_request.dataset_id,
            "new_dataset_id": new_dataset_id,
            "original_features": len(X.columns),
            "engineered_features": len(X_engineered.columns),
            "features_added": len(X_engineered.columns) - len(X.columns),
            "selected_features": engineer.selected_features_[:10] if engineer.selected_features_ else []
        }
        
    except Exception as e:
        raise HTTPException(500, f"Feature engineering failed: {str(e)}")

# ============================================================================
# Training Endpoints
# ============================================================================

@app.post("/api/v1/train")
@limiter.limit("2/minute")
async def start_training(
    request: Request,
    background_tasks: BackgroundTasks,
    dataset_id: str,
    target_column: str,
    train_request: TrainRequest,
    user_id: str = Depends(verify_token)
) -> ExperimentInfo:
    """Start training job"""
    # Load dataset
    if not app.state.storage:
        raise HTTPException(503, "Storage not configured")
    
    try:
        df = app.state.storage.load_dataset(dataset_id, tenant_id=user_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    
    # Validate target column
    if target_column not in df.columns:
        raise HTTPException(400, f"Target column {target_column} not found")
    
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Create experiment ID
    experiment_id = train_request.experiment_name or f"exp_{uuid.uuid4().hex[:8]}"
    
    # Configure orchestrator
    train_config = AutoMLConfig()
    if train_request.algorithms:
        train_config.algorithms = train_request.algorithms
    if train_request.optimize_metric:
        train_config.scoring = train_request.optimize_metric
    
    # Create orchestrator
    orchestrator = AutoMLOrchestrator(train_config)
    app.state.orchestrators[experiment_id] = {
        "orchestrator": orchestrator,
        "status": "running",
        "start_time": time.time(),
        "user_id": user_id
    }
    
    # Update metrics
    training_jobs.labels(status="running").inc()
    
    # Start training in background
    background_tasks.add_task(
        run_training,
        orchestrator,
        X,
        y,
        experiment_id,
        train_request.task
    )
    
    return ExperimentInfo(
        experiment_id=experiment_id,
        status="running",
        progress=0.0,
        models_trained=0,
        best_score=None,
        elapsed_time=0.0,
        eta=train_request.max_runtime_seconds
    )

async def run_training(orchestrator, X, y, experiment_id, task):
    """Run training in background"""
    try:
        # Train models
        orchestrator.fit(X, y, task=task)
        
        # Update status
        app.state.orchestrators[experiment_id]["status"] = "completed"
        training_jobs.labels(status="running").dec()
        training_jobs.labels(status="completed").inc()
        
        # Update active models count
        active_models.inc()
        
        logger.info(f"Training completed for experiment {experiment_id}")
        
    except Exception as e:
        logger.error(f"Training failed for experiment {experiment_id}: {e}")
        app.state.orchestrators[experiment_id]["status"] = "failed"
        app.state.orchestrators[experiment_id]["error"] = str(e)
        training_jobs.labels(status="running").dec()
        training_jobs.labels(status="failed").inc()

@app.get("/api/v1/experiments/{experiment_id}")
@limiter.limit("30/minute")
async def get_experiment_status(
    request: Request,
    experiment_id: str,
    user_id: str = Depends(verify_token)
) -> ExperimentInfo:
    """Get experiment status"""
    if experiment_id not in app.state.orchestrators:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    
    exp = app.state.orchestrators[experiment_id]
    
    # Check ownership
    if exp["user_id"] != user_id and user_id != "admin":
        raise HTTPException(403, "Access denied")
    
    orchestrator = exp["orchestrator"]
    elapsed_time = time.time() - exp["start_time"]
    
    # Calculate progress
    if exp["status"] == "completed":
        progress = 1.0
    elif hasattr(orchestrator, 'total_models_trained') and orchestrator.total_models_trained > 0:
        progress = min(orchestrator.total_models_trained / len(orchestrator.leaderboard), 1.0)
    else:
        progress = 0.0
    
    # Get best score
    best_score = None
    if hasattr(orchestrator, 'leaderboard') and orchestrator.leaderboard:
        best_score = orchestrator.leaderboard[0]["cv_score"]
    
    return ExperimentInfo(
        experiment_id=experiment_id,
        status=exp["status"],
        progress=progress,
        models_trained=getattr(orchestrator, 'total_models_trained', 0),
        best_score=best_score,
        elapsed_time=elapsed_time,
        eta=None if exp["status"] == "completed" else (elapsed_time / progress - elapsed_time) if progress > 0 else None
    )

@app.get("/api/v1/experiments/{experiment_id}/leaderboard")
@limiter.limit("10/minute")
async def get_leaderboard(
    request: Request,
    experiment_id: str,
    top_n: Optional[int] = 10,
    user_id: str = Depends(verify_token)
):
    """Get experiment leaderboard"""
    if experiment_id not in app.state.orchestrators:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    
    exp = app.state.orchestrators[experiment_id]
    if exp["user_id"] != user_id and user_id != "admin":
        raise HTTPException(403, "Access denied")
    
    orchestrator = exp["orchestrator"]
    
    try:
        leaderboard = orchestrator.get_leaderboard(top_n=top_n)
        return {
            "experiment_id": experiment_id,
            "leaderboard": leaderboard.to_dict('records') if not leaderboard.empty else []
        }
    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        return {
            "experiment_id": experiment_id,
            "leaderboard": []
        }

# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.get("/api/v1/models")
@limiter.limit("10/minute")
async def list_models(
    request: Request,
    user_id: str = Depends(verify_token)
):
    """List available models"""
    if not app.state.storage:
        raise HTTPException(503, "Storage not configured")
    
    try:
        models = app.state.storage.list_models(tenant_id=user_id)
        
        return {
            "models": [
                ModelInfo(
                    model_id=m.get("model_id", "unknown"),
                    algorithm=m.get("algorithm", "unknown"),
                    metrics=m.get("metrics", {}),
                    version=m.get("version", "1.0.0"),
                    created_at=m.get("created_at", datetime.now().isoformat()),
                    status="active"
                )
                for m in models
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return {"models": []}

@app.get("/api/v1/models/{model_id}")
@limiter.limit("10/minute")
async def get_model_info(
    request: Request,
    model_id: str,
    version: Optional[str] = None,
    user_id: str = Depends(verify_token)
):
    """Get model information"""
    if not app.state.storage:
        raise HTTPException(503, "Storage not configured")
    
    try:
        model, metadata = app.state.storage.load_model(model_id, version, tenant_id=user_id)
        return metadata
    except FileNotFoundError:
        raise HTTPException(404, f"Model {model_id} not found")
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {str(e)}")

@app.delete("/api/v1/models/{model_id}")
@limiter.limit("5/minute")
async def delete_model(
    request: Request,
    model_id: str,
    version: Optional[str] = None,
    user_id: str = Depends(verify_token)
):
    """Delete model"""
    if not app.state.storage:
        raise HTTPException(503, "Storage not configured")
    
    try:
        success = app.state.storage.delete_model(model_id, version, tenant_id=user_id)
        
        if success:
            active_models.dec()
            return {"message": f"Model {model_id} deleted successfully"}
        else:
            raise HTTPException(404, f"Model {model_id} not found")
    except Exception as e:
        raise HTTPException(500, f"Failed to delete model: {str(e)}")

# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/api/v1/predict")
@limiter.limit("100/minute")
async def predict_single(
    request: Request,
    predict_request: PredictRequest,
    user_id: str = Depends(verify_token)
):
    """Make predictions"""
    # Load model
    if not app.state.storage:
        raise HTTPException(503, "Storage not configured")
    
    try:
        model, metadata = app.state.storage.load_model(
            predict_request.model_id,
            tenant_id=user_id
        )
    except FileNotFoundError:
        raise HTTPException(404, f"Model {predict_request.model_id} not found")
    
    # Prepare data
    df = pd.DataFrame([predict_request.data])
    
    # Make prediction
    start_time = time.time()
    try:
        predictions = model.predict(df)
        prediction_time = time.time() - start_time
        
        # Track if monitoring enabled
        if app.state.monitoring and predict_request.track:
            monitor = app.state.monitoring.get_monitor(predict_request.model_id)
            if monitor:
                monitor.log_prediction(df, predictions, None, prediction_time)
        
        # Update metrics
        prediction_count.inc()
        
        # Return predictions
        if len(predictions) == 1:
            prediction_value = float(predictions[0]) if isinstance(predictions[0], (np.integer, np.floating)) else predictions[0]
        else:
            prediction_value = predictions.tolist()
        
        return {
            "model_id": predict_request.model_id,
            "prediction": prediction_value,
            "prediction_time": prediction_time,
            "metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/api/v1/predict/batch")
@limiter.limit("10/minute")
async def predict_batch(
    request: Request,
    model_id: str,
    file: UploadFile = File(...),
    user_id: str = Depends(verify_token)
):
    """Batch predictions"""
    # Parse input file
    contents = await file.read()
    
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents))
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(BytesIO(contents))
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")
    
    # Load model
    if not app.state.storage:
        raise HTTPException(503, "Storage not configured")
    
    try:
        model, metadata = app.state.storage.load_model(model_id, tenant_id=user_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model {model_id} not found")
    
    # Make predictions
    try:
        predictions = model.predict(df)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        
        # Convert to CSV
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Update metrics
        prediction_count.inc(len(predictions))
        
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{model_id}.csv"}
        )
        
    except Exception as e:
        raise HTTPException(500, f"Batch prediction failed: {str(e)}")

# ============================================================================
# Data Connectors Endpoints
# ============================================================================

@app.post("/api/v1/connectors/test")
@limiter.limit("5/minute")
async def test_connection(
    request: Request,
    connector_request: ConnectorRequest,
    user_id: str = Depends(verify_token)
):
    """Test database connection"""
    try:
        # Create connection config
        config = ConnectionConfig(
            connection_type=connector_request.connection_type,
            **connector_request.connection_config
        )
        
        # Create connector
        connector = ConnectorFactory.create_connector(config)
        
        # Test connection
        success = connector.test_connection()
        
        return {
            "connection_type": connector_request.connection_type,
            "success": success,
            "message": "Connection successful" if success else "Connection failed"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Connection test failed: {str(e)}")

@app.post("/api/v1/connectors/query")
@limiter.limit("10/minute")
async def execute_query(
    request: Request,
    connector_request: ConnectorRequest,
    user_id: str = Depends(verify_token)
):
    """Execute query on external database"""
    try:
        # Create connection config
        config = ConnectionConfig(
            connection_type=connector_request.connection_type,
            **connector_request.connection_config
        )
        
        # Create connector
        connector = ConnectorFactory.create_connector(config)
        
        # Execute query or read table
        if connector_request.query:
            df = connector.query(connector_request.query)
        elif connector_request.table_name:
            df = connector.read_table(connector_request.table_name)
        else:
            raise HTTPException(400, "Either query or table_name must be provided")
        
        # Disconnect
        connector.disconnect()
        
        # Save as dataset
        dataset_id = f"external_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if app.state.storage:
            app.state.storage.save_dataset(df, dataset_id, tenant_id=user_id)
        
        return {
            "dataset_id": dataset_id,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "preview": df.head(5).to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(500, f"Query execution failed: {str(e)}")

# ============================================================================
# Tenant Management Endpoints
# ============================================================================

@app.post("/api/v1/tenants")
@limiter.limit("5/minute")
async def create_tenant(
    request: Request,
    tenant_request: TenantRequest,
    user_id: str = Depends(verify_token)
):
    """Create new tenant"""
    try:
        tenant = app.state.tenant_manager.create_tenant(
            name=tenant_request.name,
            plan=tenant_request.plan
        )
        
        # Generate API key
        api_key = app.state.security_manager.generate_api_key(tenant.tenant_id)
        
        return {
            "tenant_id": tenant.tenant_id,
            "name": tenant.name,
            "plan": tenant.plan,
            "api_key": api_key,
            "created_at": tenant.created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to create tenant: {str(e)}")

@app.get("/api/v1/tenants/{tenant_id}")
@limiter.limit("10/minute")
async def get_tenant(
    request: Request,
    tenant_id: str,
    user_id: str = Depends(verify_token)
):
    """Get tenant information"""
    try:
        tenant = app.state.tenant_manager.get_tenant(tenant_id)
        if not tenant:
            raise HTTPException(404, "Tenant not found")
        
        return {
            "tenant_id": tenant.tenant_id,
            "name": tenant.name,
            "plan": tenant.plan,
            "max_cpu_cores": tenant.max_cpu_cores,
            "max_memory_gb": tenant.max_memory_gb,
            "max_storage_gb": tenant.max_storage_gb,
            "features": tenant.features
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to get tenant: {str(e)}")

# ============================================================================
# Billing Endpoints
# ============================================================================

@app.post("/api/v1/billing/subscription")
@limiter.limit("5/minute")
async def create_subscription(
    request: Request,
    tenant_id: str,
    plan: str,
    billing_period: str = "monthly",
    user_id: str = Depends(verify_token)
):
    """Create subscription for tenant"""
    try:
        plan_type = PlanType(plan)
        period_type = BillingPeriod(billing_period)
        
        result = app.state.billing_manager.create_subscription(
            tenant_id=tenant_id,
            plan_type=plan_type,
            billing_period=period_type
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Failed to create subscription: {str(e)}")

@app.get("/api/v1/billing/usage/{tenant_id}")
@limiter.limit("10/minute")
async def get_usage(
    request: Request,
    tenant_id: str,
    user_id: str = Depends(verify_token)
):
    """Get usage statistics for tenant"""
    try:
        usage = app.state.usage_tracker.get_usage(tenant_id)
        bill = app.state.billing_manager.calculate_bill(tenant_id)
        
        return {
            "tenant_id": tenant_id,
            "usage": usage,
            "billing": bill
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to get usage: {str(e)}")

# ============================================================================
# WebSocket Endpoints
# ============================================================================

@app.websocket("/ws/experiments/{experiment_id}")
async def websocket_experiment_updates(
    websocket: WebSocket,
    experiment_id: str
):
    """WebSocket for real-time experiment updates"""
    await websocket.accept()
    app.state.websocket_connections[f"exp_{experiment_id}"] = websocket
    
    try:
        while True:
            # Send updates every second
            await asyncio.sleep(1)
            
            if experiment_id in app.state.orchestrators:
                exp = app.state.orchestrators[experiment_id]
                orchestrator = exp["orchestrator"]
                
                update = {
                    "type": "experiment_update",
                    "experiment_id": experiment_id,
                    "status": exp["status"],
                    "models_trained": getattr(orchestrator, 'total_models_trained', 0),
                    "best_score": orchestrator.leaderboard[0]["cv_score"] if hasattr(orchestrator, 'leaderboard') and orchestrator.leaderboard else None
                }
                
                await websocket.send_json(update)
                
                if exp["status"] in ["completed", "failed"]:
                    break
            else:
                await websocket.send_json({"type": "error", "message": "Experiment not found"})
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if f"exp_{experiment_id}" in app.state.websocket_connections:
            del app.state.websocket_connections[f"exp_{experiment_id}"]

# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/api/v1/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, username: str, password: str):
    """Login endpoint"""
    # This is a simple example - implement proper authentication
    if username == "admin" and password == "admin":  # Replace with real auth
        token = create_token(username)
        return {"access_token": token, "token_type": "bearer"}
    else:
        raise HTTPException(401, "Invalid credentials")

@app.post("/api/v1/auth/refresh")
@limiter.limit("10/minute")
async def refresh_token(request: Request, user_id: str = Depends(verify_token)):
    """Refresh JWT token"""
    new_token = create_token(user_id)
    return {"access_token": new_token, "token_type": "bearer"}

# ============================================================================
# LLM Endpoints
# ============================================================================

@app.post("/api/v1/llm/chat")
@limiter.limit("20/minute")
async def llm_chat(
    request: Request,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    user_id: str = Depends(verify_token)
):
    """Chat with LLM assistant"""
    if not app.state.llm_assistant:
        raise HTTPException(503, "LLM not configured")
    
    try:
        response = await app.state.llm_assistant.chat(message, context)
        return {
            "message": message,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(500, f"LLM chat failed: {str(e)}")

@app.post("/api/v1/llm/features/suggest")
@limiter.limit("10/minute")
async def llm_suggest_features(
    request: Request,
    dataset_id: str,
    target_column: str,
    task_type: str = "classification",
    user_id: str = Depends(verify_token)
):
    """Get AI-powered feature suggestions"""
    if not app.state.llm_assistant:
        raise HTTPException(503, "LLM not configured")
    
    try:
        # Load dataset
        df = app.state.storage.load_dataset(dataset_id, tenant_id=user_id)
        
        # Get suggestions
        suggestions = await app.state.llm_assistant.suggest_features(
            df, target_column, task_type
        )
        
        return {
            "dataset_id": dataset_id,
            "target_column": target_column,
            "suggestions": suggestions[:10]  # Limit to top 10
        }
        
    except Exception as e:
        raise HTTPException(500, f"Feature suggestion failed: {str(e)}")

# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=getattr(config.api, 'host', '0.0.0.0'),
        port=getattr(config.api, 'port', 8000),
        reload=getattr(config, 'debug', False),
        log_level="info" if getattr(config, 'verbose', True) else "error"
    )
