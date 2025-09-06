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

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, BackgroundTasks, Request, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Prometheus metrics imports
# ============================================================================

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    generate_latest = None
    CONTENT_TYPE_LATEST = None
    CollectorRegistry = None
    REGISTRY = None

# ============================================================================
# Conditional Imports with Availability Flags
# ============================================================================

# Core auth imports (assumed to be always available)
try:
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
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    auth_router = None
    get_current_user = None
    User = None
    get_db = None
    require_permission = None
    require_plan = None
    RateLimiter = None
    PlanType = None
    init_auth_system = None

# Billing imports
try:
    from ..billing import BillingManager, UsageTracker
    from ..billing_middleware import BillingMiddleware, setup_billing_middleware
    BILLING_AVAILABLE = True
except ImportError:
    BILLING_AVAILABLE = False
    BillingManager = None
    UsageTracker = None
    BillingMiddleware = None
    setup_billing_middleware = None

# Scheduler imports
try:
    from ..scheduler import SchedulerFactory, JobRequest, JobStatus, QueueType
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    SchedulerFactory = None
    JobRequest = None
    JobStatus = None
    QueueType = None

# Data preparation imports
try:
    from ..data_prep import EnhancedDataPreprocessor, validate_data
    DATA_PREP_AVAILABLE = True
except ImportError:
    DATA_PREP_AVAILABLE = False
    EnhancedDataPreprocessor = None
    validate_data = None

# Inference imports
try:
    from ..inference import load_pipeline, predict, predict_batch, save_predictions
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    load_pipeline = None
    predict = None
    predict_batch = None
    save_predictions = None

# Streaming imports
try:
    from ..streaming import StreamConfig, StreamingOrchestrator, MLStreamProcessor
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    StreamConfig = None
    StreamingOrchestrator = None
    MLStreamProcessor = None

# Export service imports
try:
    from ..export_service import ModelExporter, ExportConfig
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False
    ModelExporter = None
    ExportConfig = None

# A/B Testing imports
try:
    from ..ab_testing import ABTestingService
    AB_TESTING_AVAILABLE = True
except ImportError:
    AB_TESTING_AVAILABLE = False
    ABTestingService = None

# Auth endpoints imports
try:
    from ..auth_endpoints import create_auth_router
    AUTH_ENDPOINTS_AVAILABLE = True
except ImportError:
    AUTH_ENDPOINTS_AVAILABLE = False
    create_auth_router = None

# Connector router imports
try:
    from ..connectors import connector_router
    CONNECTORS_AVAILABLE = True
except ImportError:
    CONNECTORS_AVAILABLE = False
    connector_router = None

# Feature store router imports
try:
    from ..feature_store import feature_store_router
    FEATURE_STORE_AVAILABLE = True
except ImportError:
    FEATURE_STORE_AVAILABLE = False
    feature_store_router = None

# Orchestrator imports
try:
    from ..orchestrator import AutoMLOrchestrator
    from ..config import load_config
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    AutoMLOrchestrator = None
    load_config = None

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
# Initialize Services (only if available)
# ============================================================================

# Configuration
config = load_config() if ORCHESTRATOR_AVAILABLE else None

# Initialize core services based on availability
billing_manager = BillingManager() if BILLING_AVAILABLE else None
scheduler = SchedulerFactory.create_scheduler(config, billing_manager) if (SCHEDULER_AVAILABLE and config) else None
ab_testing_service = ABTestingService() if AB_TESTING_AVAILABLE else None
model_exporter = ModelExporter() if EXPORT_AVAILABLE else None

# Setup billing middleware if available
if BILLING_AVAILABLE and billing_manager:
    billing_enforcer = setup_billing_middleware(app, billing_manager)
else:
    # Create a dummy billing enforcer that doesn't check quotas
    class DummyBillingEnforcer:
        def require_quota(self, resource: str, amount: int):
            def decorator(func):
                return func
            return decorator
    billing_enforcer = DummyBillingEnforcer()

# Initialize auth system if available
auth_services = init_auth_system() if AUTH_AVAILABLE else None

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


# Only define StreamConfig-dependent models if streaming is available
if STREAMING_AVAILABLE:
    # StreamConfig should be available from the streaming module
    pass

# ============================================================================
# Metrics Endpoint
# ============================================================================

@app.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus metrics endpoint for monitoring.
    
    Returns:
        Response: Prometheus formatted metrics
    """
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prometheus client not installed. Install with: pip install prometheus-client"
        )
    
    # Collect metrics from all available modules
    registries = []
    
    # Add default registry
    registries.append(REGISTRY)
    
    # Add UI metrics if available
    try:
        from ..ui import ui_registry, METRICS_AVAILABLE as UI_METRICS_AVAILABLE
        if UI_METRICS_AVAILABLE and ui_registry:
            registries.append(ui_registry)
            logger.info("UI metrics included")
    except ImportError:
        pass
    
    # Add monitoring service metrics if available
    try:
        from ..monitoring import MonitoringService
        if hasattr(MonitoringService, 'registry') and MonitoringService.registry:
            registries.append(MonitoringService.registry)
            logger.info("Monitoring service metrics included")
    except ImportError:
        pass
    
    # Add scheduler metrics if available
    if SCHEDULER_AVAILABLE and scheduler:
        try:
            if hasattr(scheduler, 'metrics_registry'):
                registries.append(scheduler.metrics_registry)
                logger.info("Scheduler metrics included")
        except AttributeError:
            pass
    
    # Add billing metrics if available
    if BILLING_AVAILABLE and billing_manager:
        try:
            if hasattr(billing_manager, 'metrics_registry'):
                registries.append(billing_manager.metrics_registry)
                logger.info("Billing metrics included")
        except AttributeError:
            pass
    
    # Add feature store metrics if available
    try:
        from ..feature_store import feature_store_registry
        if feature_store_registry:
            registries.append(feature_store_registry)
            logger.info("Feature store metrics included")
    except ImportError:
        pass
    
    # Add streaming metrics if available
    try:
        from ..streaming import streaming_registry
        if streaming_registry:
            registries.append(streaming_registry)
            logger.info("Streaming metrics included")
    except ImportError:
        pass
    
    # Add connector metrics if available
    try:
        from ..connectors import connector_registry
        if connector_registry:
            registries.append(connector_registry)
            logger.info("Connector metrics included")
    except ImportError:
        pass
    
    # Combine all metrics from different registries
    combined_metrics = []
    for registry in registries:
        try:
            metrics_data = generate_latest(registry)
            if metrics_data:
                combined_metrics.append(metrics_data.decode('utf-8') if isinstance(metrics_data, bytes) else metrics_data)
        except Exception as e:
            logger.warning(f"Error collecting metrics from registry: {e}")
    
    # Combine all metrics, removing duplicate TYPE and HELP lines
    if combined_metrics:
        # Parse and deduplicate metrics
        seen_metrics = set()
        final_metrics = []
        
        for metrics_block in combined_metrics:
            lines = metrics_block.split('\n')
            for line in lines:
                if line.startswith('#'):
                    # It's a TYPE or HELP line
                    metric_name = line.split(' ')[2] if ' ' in line else None
                    if metric_name and metric_name not in seen_metrics:
                        final_metrics.append(line)
                        seen_metrics.add(metric_name)
                elif line.strip():  # Non-empty, non-comment lines
                    final_metrics.append(line)
        
        metrics_output = '\n'.join(final_metrics)
    else:
        # Return empty metrics if none available
        metrics_output = "# No metrics available\n"
    
    return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/status")
async def metrics_status():
    """
    Get status of available metrics sources.
    
    Returns:
        dict: Status of each metrics source
    """
    status = {
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "sources": {}
    }
    
    # Check UI metrics
    try:
        from ..ui import METRICS_AVAILABLE as UI_METRICS_AVAILABLE
        status["sources"]["ui"] = UI_METRICS_AVAILABLE
    except ImportError:
        status["sources"]["ui"] = False
    
    # Check monitoring service
    try:
        from ..monitoring import MonitoringService
        status["sources"]["monitoring"] = hasattr(MonitoringService, 'registry')
    except ImportError:
        status["sources"]["monitoring"] = False
    
    # Check scheduler
    status["sources"]["scheduler"] = SCHEDULER_AVAILABLE and scheduler and hasattr(scheduler, 'metrics_registry')
    
    # Check billing
    status["sources"]["billing"] = BILLING_AVAILABLE and billing_manager and hasattr(billing_manager, 'metrics_registry')
    
    # Check feature store
    try:
        from ..feature_store import feature_store_registry
        status["sources"]["feature_store"] = feature_store_registry is not None
    except ImportError:
        status["sources"]["feature_store"] = False
    
    # Check streaming
    try:
        from ..streaming import streaming_registry
        status["sources"]["streaming"] = streaming_registry is not None
    except ImportError:
        status["sources"]["streaming"] = False
    
    # Check connectors
    try:
        from ..connectors import connector_registry
        status["sources"]["connectors"] = connector_registry is not None
    except ImportError:
        status["sources"]["connectors"] = False
    
    # Count active sources
    active_sources = sum(1 for v in status["sources"].values() if v)
    total_sources = len(status["sources"])
    status["summary"] = f"{active_sources}/{total_sources} metrics sources active"
    
    return status


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
        "documentation": "/docs",
        "metrics": "/metrics" if PROMETHEUS_AVAILABLE else "not available",
        "available_features": {
            "auth": AUTH_AVAILABLE,
            "billing": BILLING_AVAILABLE,
            "scheduler": SCHEDULER_AVAILABLE,
            "data_prep": DATA_PREP_AVAILABLE,
            "inference": INFERENCE_AVAILABLE,
            "streaming": STREAMING_AVAILABLE,
            "export": EXPORT_AVAILABLE,
            "ab_testing": AB_TESTING_AVAILABLE,
            "connectors": CONNECTORS_AVAILABLE,
            "feature_store": FEATURE_STORE_AVAILABLE,
            "metrics": PROMETHEUS_AVAILABLE
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected" if AUTH_AVAILABLE else "not available",
            "scheduler": "active" if (scheduler and SCHEDULER_AVAILABLE) else "not configured",
            "billing": "active" if BILLING_AVAILABLE else "not available",
            "auth": "active" if AUTH_AVAILABLE else "not available",
            "streaming": "available" if STREAMING_AVAILABLE else "not available",
            "metrics": "available" if PROMETHEUS_AVAILABLE else "not available"
        }
    }


@app.get("/api/status")
async def get_status(current_user: User = Depends(get_current_user) if AUTH_AVAILABLE else None):
    """Get platform status and user quotas"""
    
    if not AUTH_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not available"
        )
    
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    response = {
        "user": {
            "id": str(current_user.id),
            "username": current_user.username,
            "plan": current_user.plan_type if hasattr(current_user, 'plan_type') else "unknown",
            "organization": current_user.organization if hasattr(current_user, 'organization') else None
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add subscription info if billing is available
    if BILLING_AVAILABLE and billing_manager:
        subscription = billing_manager.get_subscription(current_user.tenant_id)
        usage = billing_manager.usage_tracker.get_usage(current_user.tenant_id)
        response["subscription"] = subscription
        response["usage"] = usage
    
    # Add queue stats if scheduler is available
    if SCHEDULER_AVAILABLE and scheduler:
        response["queue_stats"] = scheduler.get_queue_stats()
    
    return response


# ============================================================================
# Data Management Endpoints (only if data_prep is available)
# ============================================================================

if DATA_PREP_AVAILABLE:
    @app.post("/api/upload")
    @billing_enforcer.require_quota("storage", 1)
    async def upload_dataset(
        file: UploadFile = File(...),
        name: Optional[str] = None,
        description: Optional[str] = None,
        current_user: User = Depends(get_current_user) if AUTH_AVAILABLE else None,
        request: Request = None
    ):
        """Upload a dataset"""
        
        if not AUTH_AVAILABLE or not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Check file size
        file_size_mb = len(await file.read()) / (1024 * 1024)
        file.file.seek(0)  # Reset file pointer
        
        # Check storage quota if billing is available
        if BILLING_AVAILABLE and billing_manager:
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
        
        # Track storage usage if billing is available
        if BILLING_AVAILABLE and billing_manager:
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
        current_user: User = Depends(get_current_user) if AUTH_AVAILABLE else None,
        limit: int = 100,
        offset: int = 0
    ):
        """List user's datasets"""
        
        if not AUTH_AVAILABLE or not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
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
# Model Training Endpoints (only if scheduler is available)
# ============================================================================

if SCHEDULER_AVAILABLE:
    @app.post("/api/train")
    @billing_enforcer.require_quota("models", 1)
    async def train_model(
        train_request: TrainRequest,
        background_tasks: BackgroundTasks,
        current_user: User = Depends(get_current_user) if AUTH_AVAILABLE else None,
        request: Request = None
    ):
        """Train a new model"""
        
        if not AUTH_AVAILABLE or not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        if not scheduler:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Scheduler not configured"
            )
        
        # Determine queue type based on request
        if train_request.enable_gpu:
            queue_type = QueueType.GPU_TRAINING
        elif hasattr(current_user, 'plan_type') and current_user.plan_type in [PlanType.PRO.value, PlanType.ENTERPRISE.value]:
            queue_type = QueueType.CPU_PRIORITY
        else:
            queue_type = QueueType.CPU_DEFAULT
        
        # Create job request
        job = JobRequest(
            tenant_id=current_user.tenant_id,
            user_id=str(current_user.id),
            plan_type=current_user.plan_type if hasattr(current_user, 'plan_type') else 'free',
            task_type="train",
            queue_type=queue_type,
            payload=train_request.dict(),
            requires_gpu=train_request.enable_gpu,
            estimated_time_minutes=train_request.time_limit // 60
        )
        
        # Submit to scheduler
        job_id = scheduler.submit_job(job)
        
        # Track model count if billing is available
        if BILLING_AVAILABLE and billing_manager:
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
        current_user: User = Depends(get_current_user) if AUTH_AVAILABLE else None
    ):
        """Get job status"""
        
        if not AUTH_AVAILABLE or not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
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


# [SUITE DU FICHIER INCHANGÉE - Les autres endpoints restent identiques]
# ... (tout le reste du code reste identique jusqu'à la fin)


# ============================================================================
# Include Routers (only if available)
# ============================================================================

# Include authentication router if available
if AUTH_AVAILABLE and auth_router:
    app.include_router(auth_router)

# Include SSO/RGPD router if available
if AUTH_ENDPOINTS_AVAILABLE and create_auth_router:
    auth_full_router = create_auth_router()
    app.include_router(auth_full_router)

# Include connectors router if available
if CONNECTORS_AVAILABLE and connector_router:
    app.include_router(connector_router, prefix="/api")

# Include feature store router if available
if FEATURE_STORE_AVAILABLE and feature_store_router:
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
    
    # Log available features
    features = {
        "auth": AUTH_AVAILABLE,
        "billing": BILLING_AVAILABLE,
        "scheduler": SCHEDULER_AVAILABLE,
        "data_prep": DATA_PREP_AVAILABLE,
        "inference": INFERENCE_AVAILABLE,
        "streaming": STREAMING_AVAILABLE,
        "export": EXPORT_AVAILABLE,
        "ab_testing": AB_TESTING_AVAILABLE,
        "connectors": CONNECTORS_AVAILABLE,
        "feature_store": FEATURE_STORE_AVAILABLE,
        "metrics": PROMETHEUS_AVAILABLE
    }
    
    logger.info(f"Available features: {features}")
    
    if PROMETHEUS_AVAILABLE:
        logger.info("Prometheus metrics endpoint available at /metrics")
    
    logger.info("Services initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down AutoML Platform API...")
    
    # Cleanup resources
    if SCHEDULER_AVAILABLE and scheduler:
        if hasattr(scheduler, 'stop'):
            scheduler.stop()
    
    logger.info("Shutdown complete")


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
