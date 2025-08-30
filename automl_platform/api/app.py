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

# Import your modules
from automl_platform.config import AutoMLConfig, load_config
from automl_platform.enhanced_orchestrator import EnhancedAutoMLOrchestrator
from automl_platform.storage import StorageManager
from automl_platform.monitoring import MonitoringService, DataQualityMonitor

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
    docs_url="/docs" if config.api.enable_docs else None,
    redoc_url="/redoc" if config.api.enable_docs else None,
    lifespan=lifespan
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
if config.api.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
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
        "exp": datetime.utcnow() + timedelta(minutes=config.api.jwt_expiration_minutes)
    }
    return jwt.encode(payload, config.api.jwt_secret, algorithm=config.api.jwt_algorithm)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token"""
    if not config.api.enable_auth:
        return "anonymous"
    
    token = credentials.credentials
    try:
        payload = jwt.decode(token, config.api.jwt_secret, algorithms=[config.api.jwt_algorithm])
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
        "environment": config.environment
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        generate_latest(metrics_registry),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/api/v1/status")
@limiter.limit("10/minute")
async def get_status(request: Request, user_id: str = Depends(verify_token)):
    """Get system status"""
    return {
        "status": "operational",
        "active_experiments": len(app.state.orchestrators),
        "models_available": active_models._value.get(),
        "storage_backend": config.storage.backend,
        "monitoring_enabled": config.monitoring.enabled,
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
    if not file.filename.endswith(tuple(config.api.allowed_extensions)):
        raise HTTPException(400, f"Invalid file type. Allowed: {config.api.allowed_extensions}")
    
    # Check file size
    contents = await file.read()
    file_size = len(contents)
    
    if file_size > config.api.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large. Maximum size: {config.api.max_upload_size_mb} MB")
    
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
    if app.state.monitoring:
        quality_monitor = DataQualityMonitor()
        quality_report = quality_monitor.check_data_quality(df)
    
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
            "statistics": df.describe().to_dict()
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
        quality_monitor = DataQualityMonitor()
        report = quality_monitor.check_data_quality(df)
        
        return DataQualityReport(
            quality_score=report["quality_score"],
            issues=report.get("issues", []),
            warnings=report.get("warnings", []),
            statistics={
                "rows": report["rows"],
                "columns": report["columns"]
            }
        )
    except Exception as e:
        raise HTTPException(500, f"Quality check failed: {str(e)}")

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
    train_config = config.copy()
    if train_request.algorithms:
        train_config.algorithms = train_request.algorithms
    if train_request.optimize_metric:
        train_config.scoring = train_request.optimize_metric
    
    # Create orchestrator
    orchestrator = EnhancedAutoMLOrchestrator(train_config)
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
        orchestrator.fit(X, y, task=task, experiment_name=experiment_id)
        
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
    elif orchestrator.total_models_trained > 0:
        progress = min(orchestrator.total_models_trained / len(orchestrator.leaderboard), 1.0)
    else:
        progress = 0.0
    
    # Get best score
    best_score = None
    if orchestrator.leaderboard:
        best_score = orchestrator.leaderboard[0]["cv_score"]
    
    return ExperimentInfo(
        experiment_id=experiment_id,
        status=exp["status"],
        progress=progress,
        models_trained=orchestrator.total_models_trained,
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
    leaderboard = orchestrator.get_leaderboard(top_n=top_n)
    
    return {
        "experiment_id": experiment_id,
        "leaderboard": leaderboard.to_dict('records') if not leaderboard.empty else []
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
    
    models = app.state.storage.list_models(tenant_id=user_id)
    
    return {
        "models": [
            ModelInfo(
                model_id=m["model_id"],
                algorithm=m["algorithm"],
                metrics=m["metrics"],
                version=m["version"],
                created_at=m["created_at"],
                status="active"
            )
            for m in models
        ]
    }

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
    
    success = app.state.storage.delete_model(model_id, version, tenant_id=user_id)
    
    if success:
        active_models.dec()
        return {"message": f"Model {model_id} deleted successfully"}
    else:
        raise HTTPException(404, f"Model {model_id} not found")

# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/api/v1/predict")
@limiter.limit("100/minute")
async def predict(
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
# Monitoring Endpoints
# ============================================================================

@app.get("/api/v1/monitoring/drift/{model_id}")
@limiter.limit("10/minute")
async def get_drift_status(
    request: Request,
    model_id: str,
    user_id: str = Depends(verify_token)
):
    """Get drift status for a model"""
    if not app.state.monitoring:
        raise HTTPException(503, "Monitoring not configured")
    
    monitor = app.state.monitoring.get_monitor(model_id)
    if not monitor:
        raise HTTPException(404, f"Monitor for model {model_id} not found")
    
    # Get latest drift results
    if monitor.drift_detector.drift_history:
        latest_drift = monitor.drift_detector.drift_history[-1]
    else:
        latest_drift = {"drift_detected": False, "message": "No drift checks performed yet"}
    
    return {
        "model_id": model_id,
        "drift_status": latest_drift,
        "performance_summary": monitor.get_performance_summary()
    }

@app.get("/api/v1/monitoring/performance/{model_id}")
@limiter.limit("10/minute")
async def get_performance_metrics(
    request: Request,
    model_id: str,
    days: int = 7,
    user_id: str = Depends(verify_token)
):
    """Get performance metrics for a model"""
    if not app.state.monitoring:
        raise HTTPException(503, "Monitoring not configured")
    
    monitor = app.state.monitoring.get_monitor(model_id)
    if not monitor:
        raise HTTPException(404, f"Monitor for model {model_id} not found")
    
    return monitor.get_performance_summary(last_n_days=days)

@app.post("/api/v1/monitoring/alert")
@limiter.limit("5/minute")
async def configure_alert(
    request: Request,
    model_id: str,
    alert_type: str,
    threshold: float,
    notification_channel: str,
    user_id: str = Depends(verify_token)
):
    """Configure monitoring alert"""
    if not app.state.monitoring:
        raise HTTPException(503, "Monitoring not configured")
    
    # This would configure alerts in your monitoring system
    return {
        "message": "Alert configured successfully",
        "model_id": model_id,
        "alert_type": alert_type,
        "threshold": threshold,
        "channel": notification_channel
    }

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
                    "models_trained": orchestrator.total_models_trained,
                    "best_score": orchestrator.leaderboard[0]["cv_score"] if orchestrator.leaderboard else None
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
        del app.state.websocket_connections[f"exp_{experiment_id}"]
        await websocket.close()

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
# Admin Endpoints
# ============================================================================

@app.get("/api/v1/admin/experiments")
@limiter.limit("5/minute")
async def list_all_experiments(
    request: Request,
    user_id: str = Depends(verify_token)
):
    """List all experiments (admin only)"""
    if user_id != "admin":
        raise HTTPException(403, "Admin access required")
    
    experiments = []
    for exp_id, exp in app.state.orchestrators.items():
        experiments.append({
            "experiment_id": exp_id,
            "user_id": exp["user_id"],
            "status": exp["status"],
            "start_time": exp["start_time"]
        })
    
    return {"experiments": experiments}

@app.delete("/api/v1/admin/experiments/{experiment_id}")
@limiter.limit("5/minute")
async def delete_experiment(
    request: Request,
    experiment_id: str,
    user_id: str = Depends(verify_token)
):
    """Delete experiment (admin only)"""
    if user_id != "admin":
        raise HTTPException(403, "Admin access required")
    
    if experiment_id in app.state.orchestrators:
        del app.state.orchestrators[experiment_id]
        return {"message": f"Experiment {experiment_id} deleted"}
    else:
        raise HTTPException(404, f"Experiment {experiment_id} not found")

# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.debug,
        log_level="info" if config.verbose else "error"
    )
