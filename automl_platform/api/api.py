"""
Main API Entry Point for AutoML Platform
=========================================
Place in: automl_platform/api/api.py

Central FastAPI application that integrates all modules.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pathlib import Path

# Internal imports
from ..auth import (
    auth_router, 
    get_current_user, 
    User, 
    get_db,
    require_permission,
    require_plan,
    RateLimiter,
    PlanType,
    init_auth_system
)
from ..billing import BillingManager, UsageTracker
from ..billing_middleware import BillingMiddleware, setup_billing_middleware
from ..scheduler import SchedulerFactory, JobRequest, JobStatus, QueueType
from ..data_prep import EnhancedDataPreprocessor, validate_data
from ..inference import load_pipeline, predict, predict_batch, save_predictions
from ..streaming import StreamConfig, StreamingOrchestrator, MLStreamProcessor
from ..export_service import ModelExporter, ExportConfig
from ..ab_testing import ABTestingService
from ..auth_endpoints import create_auth_router

# Import additional routers if they exist
try:
    from ..connectors import connector_router
    CONNECTORS_AVAILABLE = True
except ImportError:
    CONNECTORS_AVAILABLE = False

try:
    from ..feature_store import feature_store_router
    FEATURE_STORE_AVAILABLE = True
except ImportError:
    FEATURE_STORE_AVAILABLE = False

try:
    from ..orchestrator import AutoMLOrchestrator
    from ..config import load_config
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="AutoML Platform API",
    description="Enterprise AutoML Platform with Advanced Features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Initialize Services
# ============================================================================

# Configuration
config = load_config() if ORCHESTRATOR_AVAILABLE else None

# Initialize core services
billing_manager = BillingManager()
scheduler = SchedulerFactory.create_scheduler(config, billing_manager) if config else None
ab_testing_service = ABTestingService()
model_exporter = ModelExporter()

# Setup billing middleware
billing_enforcer = setup_billing_middleware(app, billing_manager)

# Initialize auth system
auth_services = init_auth_system()

# ============================================================================
# Pydantic Models
# ============================================================================

class TrainRequest(BaseModel):
    """Model training request"""
    dataset_id: Optional[str] = None
    task: str = Field(..., description="Task type: classification, regression, clustering")
    target_column: str = Field(..., description="Target column name")
    feature_columns: Optional[List[str]] = Field(None, description="Feature columns to use")
    
    # Advanced options
    time_limit: int = Field(300, description="Training time limit in seconds")
    metric: Optional[str] = Field(None, description="Optimization metric")
    enable_gpu: bool = Field(False, description="Use GPU for training")
    enable_streaming: bool = Field(False, description="Enable incremental learning")
    
    # AutoML options
    include_neural: bool = Field(False, description="Include neural networks")
    include_ensemble: bool = Field(True, description="Include ensemble methods")
    max_models: int = Field(10, description="Maximum number of models to train")
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "dataset_123",
                "task": "classification",
                "target_column": "target",
                "time_limit": 600,
                "enable_gpu": False
            }
        }


class PredictRequest(BaseModel):
    """Prediction request"""
    model_id: str = Field(..., description="Model ID to use for prediction")
    data: Dict[str, Any] = Field(..., description="Input data for prediction")
    return_probabilities: bool = Field(False, description="Return prediction probabilities")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "model_abc123",
                "data": {"feature1": 1.0, "feature2": "A"},
                "return_probabilities": True
            }
        }


class BatchPredictRequest(BaseModel):
    """Batch prediction request"""
    model_id: str
    dataset_id: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    batch_size: int = Field(1000, description="Batch size for processing")
    output_format: str = Field("json", description="Output format: json, csv, parquet")


class ABTestRequest(BaseModel):
    """A/B test creation request"""
    model_name: str
    champion_version: int
    challenger_version: int
    traffic_split: float = Field(0.1, ge=0.0, le=1.0)
    min_samples: int = Field(100, ge=10)
    primary_metric: str = "accuracy"


class ExportRequest(BaseModel):
    """Model export request"""
    model_id: str
    format: str = Field("onnx", description="Export format: onnx, pmml, tflite, coreml")
    quantize: bool = Field(True, description="Apply quantization for smaller model size")
    optimize_for_edge: bool = Field(False, description="Optimize for edge deployment")


# ============================================================================
# Health and Status Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AutoML Platform API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "scheduler": "active" if scheduler else "not configured",
            "billing": "active",
            "auth": "active"
        }
    }


@app.get("/api/status")
async def get_status(current_user: User = Depends(get_current_user)):
    """Get platform status and user quotas"""
    
    # Get user's subscription
    subscription = billing_manager.get_subscription(current_user.tenant_id)
    
    # Get usage statistics
    usage = billing_manager.usage_tracker.get_usage(current_user.tenant_id)
    
    # Get queue stats if scheduler available
    queue_stats = scheduler.get_queue_stats() if scheduler else {}
    
    return {
        "user": {
            "id": str(current_user.id),
            "username": current_user.username,
            "plan": current_user.plan_type,
            "organization": current_user.organization
        },
        "subscription": subscription,
        "usage": usage,
        "queue_stats": queue_stats,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Data Management Endpoints
# ============================================================================

@app.post("/api/upload")
@billing_enforcer.require_quota("storage", 1)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = None,
    description: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    request: Request = None
):
    """Upload a dataset"""
    
    # Check file size
    file_size_mb = len(await file.read()) / (1024 * 1024)
    file.file.seek(0)  # Reset file pointer
    
    # Check storage quota
    if not billing_manager.check_limits(current_user.tenant_id, "storage", file_size_mb):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Storage quota exceeded"
        )
    
    # Save file
    upload_dir = Path(f"data/uploads/{current_user.tenant_id}")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Load and validate data
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file.filename.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Use CSV or Parquet."
        )
    
    # Validate data
    validation_report = validate_data(df)
    
    # Track storage usage
    billing_manager.usage_tracker.track_storage(current_user.tenant_id, file_size_mb)
    
    dataset_id = f"dataset_{current_user.tenant_id}_{datetime.utcnow().timestamp()}"
    
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "size_mb": round(file_size_mb, 2),
        "rows": len(df),
        "columns": len(df.columns),
        "validation": validation_report,
        "path": str(file_path)
    }


@app.get("/api/datasets")
async def list_datasets(
    current_user: User = Depends(get_current_user),
    limit: int = 100,
    offset: int = 0
):
    """List user's datasets"""
    
    upload_dir = Path(f"data/uploads/{current_user.tenant_id}")
    
    if not upload_dir.exists():
        return {"datasets": [], "total": 0}
    
    datasets = []
    for file_path in upload_dir.glob("*.csv"):
        stat = file_path.stat()
        datasets.append({
            "name": file_path.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
        })
    
    for file_path in upload_dir.glob("*.parquet"):
        stat = file_path.stat()
        datasets.append({
            "name": file_path.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat()
        })
    
    return {
        "datasets": datasets[offset:offset+limit],
        "total": len(datasets)
    }


# ============================================================================
# Model Training Endpoints
# ============================================================================

@app.post("/api/train")
@billing_enforcer.require_quota("models", 1)
async def train_model(
    train_request: TrainRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    request: Request = None
):
    """Train a new model"""
    
    if not scheduler:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not configured"
        )
    
    # Determine queue type based on request
    if train_request.enable_gpu:
        queue_type = QueueType.GPU_TRAINING
    elif current_user.plan_type in [PlanType.PRO.value, PlanType.ENTERPRISE.value]:
        queue_type = QueueType.CPU_PRIORITY
    else:
        queue_type = QueueType.CPU_DEFAULT
    
    # Create job request
    job = JobRequest(
        tenant_id=current_user.tenant_id,
        user_id=str(current_user.id),
        plan_type=current_user.plan_type,
        task_type="train",
        queue_type=queue_type,
        payload=train_request.dict(),
        requires_gpu=train_request.enable_gpu,
        estimated_time_minutes=train_request.time_limit // 60
    )
    
    # Submit to scheduler
    job_id = scheduler.submit_job(job)
    
    # Track model count
    billing_manager.increment_model_count(
        current_user.tenant_id,
        "gpu" if train_request.enable_gpu else "standard"
    )
    
    return {
        "job_id": job_id,
        "status": "submitted",
        "queue": queue_type.queue_name,
        "estimated_time_minutes": job.estimated_time_minutes
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get job status"""
    
    if not scheduler:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not configured"
        )
    
    job = scheduler.get_job_status(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # Check ownership
    if job.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return {
        "job_id": job_id,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "result": job.result,
        "error": job.error_message
    }


# ============================================================================
# Prediction Endpoints
# ============================================================================

@app.post("/api/predict")
@billing_enforcer.require_quota("predictions", 1)
async def make_prediction(
    predict_request: PredictRequest,
    current_user: User = Depends(get_current_user),
    request: Request = None
):
    """Make a single prediction"""
    
    # Load model
    model_path = Path(f"models/{current_user.tenant_id}/{predict_request.model_id}")
    
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    pipeline, metadata = load_pipeline(model_path)
    
    # Prepare data
    df = pd.DataFrame([predict_request.data])
    
    # Make prediction
    if predict_request.return_probabilities and hasattr(pipeline, 'predict_proba'):
        probabilities = pipeline.predict_proba(df)
        prediction = pipeline.predict(df)[0]
        result = {
            "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            "probabilities": probabilities[0].tolist()
        }
    else:
        prediction = predict(pipeline, df)[0]
        result = {
            "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction
        }
    
    # Track usage
    billing_manager.usage_tracker.track_predictions(current_user.tenant_id, 1)
    
    return result


@app.post("/api/predict/batch")
@billing_enforcer.require_quota("predictions", 100)
async def batch_predict(
    batch_request: BatchPredictRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    request: Request = None
):
    """Submit batch prediction job"""
    
    if not scheduler:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not configured"
        )
    
    # Create job request
    job = JobRequest(
        tenant_id=current_user.tenant_id,
        user_id=str(current_user.id),
        plan_type=current_user.plan_type,
        task_type="batch_predict",
        queue_type=QueueType.BATCH,
        payload=batch_request.dict()
    )
    
    # Submit to scheduler
    job_id = scheduler.submit_job(job)
    
    return {
        "job_id": job_id,
        "status": "submitted",
        "message": "Batch prediction job submitted"
    }


# ============================================================================
# A/B Testing Endpoints
# ============================================================================

@app.post("/api/ab-tests")
@require_plan(PlanType.PRO)
async def create_ab_test(
    ab_request: ABTestRequest,
    current_user: User = Depends(get_current_user),
    request: Request = None
):
    """Create A/B test between models"""
    
    request.state.user = current_user
    request.state.db = next(get_db())
    
    test_id = ab_testing_service.create_ab_test(
        model_name=ab_request.model_name,
        champion_version=ab_request.champion_version,
        challenger_version=ab_request.challenger_version,
        traffic_split=ab_request.traffic_split,
        min_samples=ab_request.min_samples,
        primary_metric=ab_request.primary_metric
    )
    
    return {
        "test_id": test_id,
        "status": "active",
        "message": "A/B test created successfully"
    }


@app.get("/api/ab-tests")
async def list_ab_tests(
    current_user: User = Depends(get_current_user)
):
    """List active A/B tests"""
    
    tests = ab_testing_service.get_active_tests()
    
    return {
        "tests": tests,
        "total": len(tests)
    }


@app.get("/api/ab-tests/{test_id}/results")
async def get_ab_test_results(
    test_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get A/B test results"""
    
    results = ab_testing_service.get_test_results(test_id)
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="A/B test not found"
        )
    
    return results


@app.post("/api/ab-tests/{test_id}/conclude")
async def conclude_ab_test(
    test_id: str,
    promote_winner: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Conclude A/B test"""
    
    results = ab_testing_service.conclude_test(test_id, promote_winner)
    
    return results


# ============================================================================
# Model Export Endpoints
# ============================================================================

@app.post("/api/export")
async def export_model(
    export_request: ExportRequest,
    current_user: User = Depends(get_current_user)
):
    """Export model to various formats"""
    
    # Load model
    model_path = Path(f"models/{current_user.tenant_id}/{export_request.model_id}")
    
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    pipeline, metadata = load_pipeline(model_path)
    
    # Create sample input for shape inference
    sample_input = np.random.randn(1, 10)  # Adjust based on actual model
    
    # Export based on format
    if export_request.format == "onnx":
        result = model_exporter.export_to_onnx(
            pipeline,
            sample_input,
            model_name=export_request.model_id,
            output_path=f"exports/{current_user.tenant_id}/{export_request.model_id}.onnx"
        )
    elif export_request.format == "pmml":
        # Need sample output for PMML
        sample_output = np.array([0])
        result = model_exporter.export_to_pmml(
            pipeline,
            sample_input,
            sample_output,
            model_name=export_request.model_id
        )
    elif export_request.optimize_for_edge:
        result = model_exporter.export_for_edge(
            pipeline,
            sample_input,
            model_name=export_request.model_id,
            formats=[export_request.format]
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported export format: {export_request.format}"
        )
    
    return result


# ============================================================================
# Streaming Endpoints
# ============================================================================

@app.post("/api/streaming/start")
@require_plan(PlanType.PROFESSIONAL)
async def start_streaming(
    stream_config: StreamConfig,
    current_user: User = Depends(get_current_user),
    request: Request = None
):
    """Start streaming pipeline"""
    
    request.state.user = current_user
    request.state.db = next(get_db())
    
    # Create streaming orchestrator
    orchestrator = StreamingOrchestrator(stream_config)
    
    # Create ML processor
    processor = MLStreamProcessor(stream_config)
    
    # Set processor
    orchestrator.set_processor(processor)
    
    # Start streaming (in background)
    asyncio.create_task(orchestrator.start())
    
    return {
        "status": "started",
        "platform": stream_config.platform,
        "topic": stream_config.topic
    }


# ============================================================================
# Include Routers
# ============================================================================

# Include authentication router
app.include_router(auth_router)

# Include SSO/RGPD router
auth_full_router = create_auth_router()
app.include_router(auth_full_router)

# Include optional routers
if CONNECTORS_AVAILABLE:
    app.include_router(connector_router, prefix="/api")

if FEATURE_STORE_AVAILABLE:
    app.include_router(feature_store_router, prefix="/api")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting AutoML Platform API...")
    
    # Initialize services
    logger.info("Services initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down AutoML Platform API...")
    
    # Cleanup resources
    if scheduler:
        scheduler.stop() if hasattr(scheduler, 'stop') else None


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
