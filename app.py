"""
Enhanced FastAPI application for AutoML Platform
Production-ready with rate limiting, monitoring, billing middleware and scheduler integration
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

# AutoML Platform imports
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

# NEW: Billing middleware and routes
from automl_platform.api.billing_middleware import BillingMiddleware, BillingEnforcer, InvoiceGenerator
from automl_platform.api.billing_routes import billing_router

# NEW: Scheduler integration
from automl_platform.scheduler import SchedulerFactory, JobRequest, QueueType

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
# Metrics with Custom Registry
# ============================================================================

metrics_registry = CollectorRegistry()

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
    logger.info("Starting AutoML API with Billing & Scheduler...")
    
    # Initialize storage
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
    
    # NEW: Initialize billing components
    app.state.billing_enforcer = BillingEnforcer(app.state.billing_manager)
    app.state.invoice_generator = InvoiceGenerator(app.state.billing_manager)
    
    # NEW: Initialize scheduler for distributed processing
    app.state.scheduler = SchedulerFactory.create_scheduler(config, app.state.billing_manager)
    logger.info(f"Scheduler initialized: {type(app.state.scheduler).__name__}")
    
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
    
    logger.info("AutoML API started successfully with all components")
    
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
    description="Production-ready AutoML platform with billing, monitoring and distributed processing",
    version="3.0.0",
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

# NEW: Add billing middleware for quota enforcement
@app.on_event("startup")
async def add_billing_middleware():
    """Add billing middleware after app state is initialized"""
    if hasattr(app.state, 'billing_manager'):
        app.add_middleware(BillingMiddleware, billing_manager=app.state.billing_manager)
        logger.info("Billing middleware added successfully")

# NEW: Include billing routes
app.include_router(billing_router)

# ============================================================================
# Security
# ============================================================================

security = HTTPBearer()

def create_token(user_id: str, tenant_id: str = None) -> str:
    """Create JWT token with tenant information"""
    payload = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "exp": datetime.utcnow() + timedelta(minutes=getattr(config.api, 'jwt_expiration_minutes', 60))
    }
    return jwt.encode(payload, getattr(config.api, 'jwt_secret', 'secret'), algorithm=getattr(config.api, 'jwt_algorithm', 'HS256'))

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, str]:
    """Verify JWT token and return user info"""
    if not getattr(config.api, 'enable_auth', False):
        return {"user_id": "anonymous", "tenant_id": "default"}
    
    token = credentials.credentials
    try:
        payload = jwt.decode(token, getattr(config.api, 'jwt_secret', 'secret'), algorithms=[getattr(config.api, 'jwt_algorithm', 'HS256')])
        return {"user_id": payload["user_id"], "tenant_id": payload.get("tenant_id", "default")}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================================================================
# Enhanced Pydantic Models
# ============================================================================

class TrainRequest(BaseModel):
    """Enhanced training request with scheduler options"""
    experiment_name: Optional[str] = Field(None, description="Name for the experiment")
    task: Optional[str] = Field("auto", description="Task type: classification, regression, auto")
    algorithms: Optional[List[str]] = Field(None, description="Algorithms to use")
    max_runtime_seconds: Optional[int] = Field(3600, description="Maximum runtime in seconds")
    optimize_metric: Optional[str] = Field("auto", description="Metric to optimize")
    validation_split: Optional[float] = Field(0.2, description="Validation split ratio")
    enable_monitoring: Optional[bool] = Field(True, description="Enable monitoring")
    enable_feature_engineering: Optional[bool] = Field(True, description="Enable feature engineering")
    use_gpu: Optional[bool] = Field(False, description="Use GPU for training")
    async_mode: Optional[bool] = Field(False, description="Run training asynchronously via scheduler")
    priority: Optional[str] = Field("default", description="Job priority: default, high, low")

class PredictRequest(BaseModel):
    """Prediction request model"""
    model_id: str = Field(..., description="Model ID to use for prediction")
    data: Dict[str, Any] = Field(..., description="Input data for prediction")
    track: Optional[bool] = Field(True, description="Track predictions for monitoring")

class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    queue: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]
    result: Optional[Dict[str, Any]]

# Keep existing models...
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
    
    # Add tenant context to request
    if hasattr(request.state, 'user'):
        request.headers.__dict__["_list"].append(
            (b"x-tenant-id", request.state.user.get("tenant_id", "default").encode())
        )
    
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
# NEW: Scheduler Endpoints
# ============================================================================

@app.post("/api/v1/jobs/submit")
@limiter.limit("10/minute")
async def submit_job(
    request: Request,
    job_type: str,
    payload: Dict[str, Any],
    use_gpu: bool = False,
    priority: str = "default",
    user_info: Dict = Depends(verify_token)
):
    """Submit job to scheduler"""
    
    # Determine queue type
    if use_gpu:
        queue_type = QueueType.GPU_TRAINING
    elif priority == "high":
        queue_type = QueueType.CPU_PRIORITY
    else:
        queue_type = QueueType.CPU_DEFAULT
    
    # Get user's plan
    tenant_id = user_info["tenant_id"]
    subscription = app.state.billing_manager.get_subscription(tenant_id)
    plan_type = subscription['plan'] if subscription else PlanType.FREE.value
    
    # Create job request
    job_request = JobRequest(
        tenant_id=tenant_id,
        user_id=user_info["user_id"],
        plan_type=plan_type,
        task_type=job_type,
        queue_type=queue_type,
        payload=payload,
        requires_gpu=use_gpu
    )
    
    # Submit to scheduler
    job_id = app.state.scheduler.submit_job(job_request)
    
    return {
        "job_id": job_id,
        "status": "submitted",
        "queue": queue_type.queue_name,
        "estimated_wait_time": "calculating..."
    }

@app.get("/api/v1/jobs/{job_id}")
@limiter.limit("30/minute")
async def get_job_status(
    request: Request,
    job_id: str,
    user_info: Dict = Depends(verify_token)
) -> JobStatusResponse:
    """Get job status from scheduler"""
    
    job = app.state.scheduler.get_job_status(job_id)
    
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    
    # Verify ownership
    if job.tenant_id != user_info["tenant_id"] and user_info["user_id"] != "admin":
        raise HTTPException(403, "Access denied")
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        queue=job.queue_type.queue_name,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error_message,
        result=job.result
    )

@app.delete("/api/v1/jobs/{job_id}")
@limiter.limit("10/minute")
async def cancel_job(
    request: Request,
    job_id: str,
    user_info: Dict = Depends(verify_token)
):
    """Cancel a job"""
    
    job = app.state.scheduler.get_job_status(job_id)
    
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    
    # Verify ownership
    if job.tenant_id != user_info["tenant_id"] and user_info["user_id"] != "admin":
        raise HTTPException(403, "Access denied")
    
    success = app.state.scheduler.cancel_job(job_id)
    
    if success:
        return {"message": f"Job {job_id} cancelled successfully"}
    else:
        raise HTTPException(500, f"Failed to cancel job {job_id}")

@app.get("/api/v1/scheduler/stats")
@limiter.limit("10/minute")
async def get_scheduler_stats(
    request: Request,
    user_info: Dict = Depends(verify_token)
):
    """Get scheduler statistics"""
    
    # Admin only
    if user_info["user_id"] != "admin":
        # Return limited stats for non-admin
        return {
            "status": "operational",
            "your_jobs": len([j for j in app.state.scheduler.active_jobs.values() 
                            if j.tenant_id == user_info["tenant_id"]])
        }
    
    stats = app.state.scheduler.get_queue_stats() if hasattr(app.state.scheduler, 'get_queue_stats') else {}
    
    return stats

# ============================================================================
# Enhanced Training Endpoints with Scheduler
# ============================================================================

@app.post("/api/v1/train")
@limiter.limit("2/minute")
async def start_training(
    request: Request,
    background_tasks: BackgroundTasks,
    dataset_id: str,
    target_column: str,
    train_request: TrainRequest,
    user_info: Dict = Depends(verify_token)
) -> Union[ExperimentInfo, Dict[str, str]]:
    """Start training job with optional async mode via scheduler"""
    
    # Load dataset
    if not app.state.storage:
        raise HTTPException(503, "Storage not configured")
    
    try:
        df = app.state.storage.load_dataset(dataset_id, tenant_id=user_info["tenant_id"])
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    
    # Validate target column
    if target_column not in df.columns:
        raise HTTPException(400, f"Target column {target_column} not found")
    
    # Create experiment ID
    experiment_id = train_request.experiment_name or f"exp_{uuid.uuid4().hex[:8]}"
    
    # Configure orchestrator
    train_config = AutoMLConfig()
    if train_request.algorithms:
        train_config.algorithms = train_request.algorithms
    if train_request.optimize_metric:
        train_config.scoring = train_request.optimize_metric
    train_config.tenant_id = user_info["tenant_id"]
    train_config.user_id = user_info["user_id"]
    
    # Check if async mode requested
    if train_request.async_mode and app.state.scheduler:
        # Submit to scheduler
        orchestrator = AutoMLOrchestrator(
            train_config, 
            scheduler=app.state.scheduler,
            billing_manager=app.state.billing_manager,
            async_mode=True
        )
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Submit training job
        orchestrator.fit(
            X, y,
            task=train_request.task,
            use_gpu=train_request.use_gpu,
            priority=train_request.priority
        )
        
        return {
            "experiment_id": experiment_id,
            "job_id": orchestrator.current_job_id,
            "status": "submitted",
            "message": "Training job submitted to scheduler",
            "tracking_url": f"/api/v1/jobs/{orchestrator.current_job_id}"
        }
    
    else:
        # Original synchronous mode
        orchestrator = AutoMLOrchestrator(train_config)
        app.state.orchestrators[experiment_id] = {
            "orchestrator": orchestrator,
            "status": "running",
            "start_time": time.time(),
            "user_id": user_info["user_id"],
            "tenant_id": user_info["tenant_id"]
        }
        
        # Update metrics
        training_jobs.labels(status="running").inc()
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
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

# Keep all existing endpoints...
# [Rest of the endpoints remain the same as in your original app.py]

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

# [Include all other existing endpoints from your original app.py here]
# I'm not repeating them all to save space, but they should all be included

# ============================================================================
# Health & Monitoring Endpoints (keep existing)
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "environment": getattr(config, 'environment', 'development'),
        "billing_enabled": True,
        "scheduler_enabled": True
    }

# [Rest of existing endpoints continue here...]

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
