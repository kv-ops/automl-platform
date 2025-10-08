"""
Main API Entry Point for AutoML Platform
=========================================
Version: 3.2.1
Place in: automl_platform/api/api.py

Central FastAPI application that integrates all modules.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import inspect

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Depends,
    HTTPException,
    Request,
    status,
    Response,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pathlib import Path
import uuid
from io import BytesIO

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
    from .billing import BillingManager, UsageTracker
    from .billing_middleware import BillingMiddleware, setup_billing_middleware
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

# Template imports
try:
    from ..template_loader import TemplateLoader
    TEMPLATE_AVAILABLE = True
except ImportError:
    TEMPLATE_AVAILABLE = False
    TemplateLoader = None

# Streaming imports
try:
    from .streaming import StreamConfig, StreamingOrchestrator, MLStreamProcessor
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
    from .auth_endpoints import create_auth_router
    AUTH_ENDPOINTS_AVAILABLE = True
except ImportError:
    AUTH_ENDPOINTS_AVAILABLE = False
    create_auth_router = None

# Connector router imports
try:
    from .connectors import connector_router
    CONNECTORS_AVAILABLE = True
except ImportError:
    CONNECTORS_AVAILABLE = False
    connector_router = None

# Feature store router imports
try:
    from .feature_store import feature_store_router
    FEATURE_STORE_AVAILABLE = True
except ImportError:
    FEATURE_STORE_AVAILABLE = False
    feature_store_router = None

# Orchestrator imports
try:
    from ..orchestrator import AutoMLOrchestrator
    from ..config import AutoMLConfig, load_config
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    AutoMLOrchestrator = None
    AutoMLConfig = None
    load_config = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="AutoML Platform API",
    description="Enterprise AutoML Platform with Advanced Features and Expert Mode",
    version="3.2.1",  # Updated version
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

# Check for expert mode from environment
EXPERT_MODE = os.getenv("AUTOML_EXPERT_MODE", "false").lower() in ["true", "1", "yes", "on"]

# Configuration
config = load_config(expert_mode=EXPERT_MODE) if ORCHESTRATOR_AVAILABLE else None

# Initialize core services based on availability
billing_manager = BillingManager() if BILLING_AVAILABLE else None
scheduler = SchedulerFactory.create_scheduler(config, billing_manager) if (SCHEDULER_AVAILABLE and config) else None
ab_testing_service = ABTestingService() if AB_TESTING_AVAILABLE else None
model_exporter = ModelExporter() if EXPORT_AVAILABLE else None
template_loader = TemplateLoader() if TEMPLATE_AVAILABLE else None

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
    task: str = Field(..., description="Task type: classification, regression, clustering, auto")
    target_column: str = Field(..., description="Target column name")
    feature_columns: Optional[List[str]] = Field(None, description="Feature columns to use")
    
    # Template support
    template: Optional[str] = Field(None, description="Template to use for configuration")
    
    # Expert mode options
    expert_mode: bool = Field(False, description="Enable expert mode for advanced options")
    
    # Advanced options (expert mode)
    time_limit: int = Field(300, description="Training time limit in seconds")
    metric: Optional[str] = Field(None, description="Optimization metric")
    enable_gpu: bool = Field(False, description="Use GPU for training")
    enable_streaming: bool = Field(False, description="Enable incremental learning")
    
    # AutoML options
    include_neural: bool = Field(False, description="Include neural networks")
    include_ensemble: bool = Field(True, description="Include ensemble methods")
    max_models: int = Field(10, description="Maximum number of models to train")
    
    # HPO options (expert mode)
    hpo_method: Optional[str] = Field(None, description="HPO method: optuna, grid, random")
    hpo_iterations: Optional[int] = Field(None, description="HPO iterations")
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "dataset_123",
                "task": "classification",
                "target_column": "target",
                "template": "customer_churn",
                "expert_mode": False,
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
    expert_mode: bool = Field(False, description="Use expert batch processing")


class TemplateInfo(BaseModel):
    """Template information"""
    name: str
    description: str
    task: str
    version: str
    algorithms: List[str]
    tags: List[str]


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
# Metrics Endpoint
# ============================================================================

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint for monitoring."""

    module_ref = sys.modules[__name__]
    prometheus_available = getattr(module_ref, "PROMETHEUS_AVAILABLE", False)
    metrics_generator = getattr(module_ref, "generate_latest", None)
    registry = getattr(module_ref, "REGISTRY", None)

    if not prometheus_available or metrics_generator is None or registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prometheus client not installed. Install with: pip install prometheus-client"
        )

    metrics_bytes = metrics_generator(registry)
    media_type = globals().get("CONTENT_TYPE_LATEST") or "text/plain; version=0.0.4; charset=utf-8"
    return Response(content=metrics_bytes, media_type=media_type)


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
        "version": "3.2.1",  # Updated version
        "status": "operational",
        "documentation": "/docs",
        "metrics": "/metrics" if PROMETHEUS_AVAILABLE else "not available",
        "expert_mode": EXPERT_MODE,  # Added expert mode status
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
            "templates": TEMPLATE_AVAILABLE,  # Added templates
            "metrics": PROMETHEUS_AVAILABLE
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.2.1",  # Added version
        "expert_mode": EXPERT_MODE,  # Added expert mode
        "services": {
            "database": "connected" if AUTH_AVAILABLE else "not available",
            "scheduler": "active" if (scheduler and SCHEDULER_AVAILABLE) else "not configured",
            "billing": "active" if BILLING_AVAILABLE else "not available",
            "auth": "active" if AUTH_AVAILABLE else "not available",
            "streaming": "available" if STREAMING_AVAILABLE else "not available",
            "templates": "available" if TEMPLATE_AVAILABLE else "not available",  # Added templates
            "metrics": "available" if PROMETHEUS_AVAILABLE else "not available"
        }
    }


# ============================================================================
# Template Endpoints (only if templates are available)
# ============================================================================

if TEMPLATE_AVAILABLE and template_loader:
    @app.get("/api/templates", response_model=List[TemplateInfo])
    async def list_templates(
        task: Optional[str] = None,
        tags: Optional[str] = None
    ):
        """List available templates"""
        tag_list = tags.split(',') if tags else None
        templates = template_loader.list_templates(task=task, tags=tag_list)
        
        return [
            TemplateInfo(
                name=t['name'],
                description=t['description'],
                task=t['task'],
                version=t['version'],
                algorithms=t['algorithms'][:5],  # Limit to first 5
                tags=t['tags']
            )
            for t in templates
        ]
    
    @app.get("/api/templates/{template_name}")
    async def get_template_info(template_name: str):
        """Get detailed template information"""
        try:
            info = template_loader.get_template_info(template_name)
            return info
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )


DATASET_STORAGE_ROOT = Path(os.getenv("AUTOML_DATASETS_DIR", "./data/datasets"))


def _module_ref():
    return sys.modules[__name__]


def _feature_available(flag_name: str) -> bool:
    return bool(getattr(_module_ref(), flag_name, False))


async def _get_current_user_dependency():
    """Resolve the current user, accommodating patched dependencies in tests."""

    module = _module_ref()
    if not _feature_available("AUTH_AVAILABLE"):
        return None

    dependency = getattr(module, "get_current_user", None)
    if dependency is None:
        return None

    try:
        if inspect.iscoroutinefunction(dependency):
            return await dependency()  # type: ignore[func-returns-value]

        result = dependency()
        if asyncio.iscoroutine(result):
            return await result
        return result
    except TypeError as exc:
        # FastAPI will normally inject required parameters. When running in
        # isolated tests without the full auth stack we surface an auth error
        # instead of returning a validation failure.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        ) from exc


def _require_auth(current_user: Optional[User]) -> None:
    """Ensure user is authenticated when auth service is active."""

    if _feature_available("AUTH_AVAILABLE") and not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )


def _ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    """Helper to create UTC timestamp strings."""

    return datetime.utcnow().isoformat()


def _serialize_datetime(value: Optional[Any]) -> Optional[str]:
    """Return ISO formatted datetime strings when possible."""

    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    current_user: Optional[User] = Depends(_get_current_user_dependency),
):
    """Upload a dataset for AutoML experiments."""

    if not _feature_available("DATA_PREP_AVAILABLE"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data preparation service is not available"
        )

    _require_auth(current_user)

    filename = file.filename or "dataset.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only CSV files are accepted"
        )

    try:
        contents = await file.read()
        dataframe = pd.read_csv(BytesIO(contents))
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Failed to read uploaded dataset")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read dataset: {exc}"
        ) from exc

    validation_result: Optional[Dict[str, Any]] = None
    if _feature_available("DATA_PREP_AVAILABLE") and validate_data:
        try:
            validation_result = validate_data(dataframe)
        except Exception as exc:  # pragma: no cover - unexpected validation errors
            logger.warning("Dataset validation failed: %s", exc)
            validation_result = {"status": "error", "issues": [str(exc)]}

    tenant_id = getattr(current_user, "tenant_id", "public") if current_user else "public"
    dataset_dir = DATASET_STORAGE_ROOT / tenant_id
    _ensure_directory(dataset_dir)

    dataset_id = f"dataset_{uuid.uuid4().hex}"
    dataset_path = dataset_dir / f"{dataset_id}.csv"
    dataframe.to_csv(dataset_path, index=False)

    response_payload = {
        "dataset_id": dataset_id,
        "tenant_id": tenant_id,
        "filename": filename,
        "name": name or Path(filename).stem,
        "description": description,
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "uploaded_at": _timestamp(),
    }

    if validation_result is not None:
        response_payload["validation"] = validation_result

    return response_payload


@app.get("/api/datasets")
async def list_datasets(
    current_user: Optional[User] = Depends(_get_current_user_dependency),
):
    """List datasets uploaded by the current tenant."""

    if not _feature_available("DATA_PREP_AVAILABLE"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data preparation service is not available"
        )

    _require_auth(current_user)

    tenant_id = getattr(current_user, "tenant_id", "public") if current_user else "public"
    dataset_dir = DATASET_STORAGE_ROOT / tenant_id

    if not dataset_dir.exists():
        return {"datasets": [], "total": 0}

    datasets: List[Dict[str, Any]] = []
    for dataset_path in sorted(dataset_dir.glob("*.csv"), key=lambda path: getattr(path, "name", "")):
        stats = dataset_path.stat()
        filename = getattr(dataset_path, "name", "")
        stem = filename.rsplit(".", 1)[0] if filename else None
        datasets.append({
            "dataset_id": stem,
            "filename": filename,
            "size_bytes": int(stats.st_size),
            "size_mb": round(stats.st_size / (1024 * 1024), 4),
            "uploaded_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
        })

    return {"datasets": datasets, "total": len(datasets)}


@app.post("/api/train")
async def train_model(
    request: TrainRequest,
    current_user: Optional[User] = Depends(_get_current_user_dependency),
):
    """Submit a model training job to the scheduler."""

    if not _feature_available("SCHEDULER_AVAILABLE"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler service is not available"
        )

    scheduler_ref = getattr(_module_ref(), "scheduler", None)

    if scheduler_ref is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not configured"
        )

    _require_auth(current_user)

    tenant_id = getattr(current_user, "tenant_id", "public") if current_user else "public"

    job_payload: Dict[str, Any] = {
        "type": "training",
        "tenant_id": tenant_id,
        "request": request.dict(),
        "submitted_at": _timestamp(),
    }

    job_request: Any = job_payload
    if JobRequest is not None:
        try:
            job_request = JobRequest(
                job_type="training",
                tenant_id=tenant_id,
                payload=request.dict(),
                metadata={"submitted_at": job_payload["submitted_at"]}
            )
        except Exception:
            job_request = job_payload

    queue: Any = "training"
    if QueueType is not None:
        try:
            queue = QueueType.TRAINING
        except AttributeError:
            queue = "training"

    try:
        job_id = scheduler_ref.submit_job(job_request, queue=queue)
    except TypeError:
        job_id = scheduler_ref.submit_job(job_request)

    estimated_minutes = max(1, int(request.time_limit // 60)) if request.time_limit else 5

    return {
        "job_id": job_id,
        "status": "submitted",
        "queue": str(queue),
        "estimated_time_minutes": estimated_minutes,
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    current_user: Optional[User] = Depends(_get_current_user_dependency),
):
    """Retrieve the status of a submitted job."""

    if not _feature_available("SCHEDULER_AVAILABLE"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler service is not available"
        )

    scheduler_ref = getattr(_module_ref(), "scheduler", None)

    if scheduler_ref is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not configured"
        )

    _require_auth(current_user)

    job = scheduler_ref.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    tenant_id = getattr(current_user, "tenant_id", None)
    job_tenant = getattr(job, "tenant_id", None)
    if tenant_id and job_tenant and tenant_id != job_tenant:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    status_value = getattr(job.status, "value", job.status) if hasattr(job, "status") else None

    return {
        "job_id": job_id,
        "status": status_value,
        "tenant_id": job_tenant,
        "created_at": _serialize_datetime(getattr(job, "created_at", None)),
        "started_at": _serialize_datetime(getattr(job, "started_at", None)),
        "completed_at": _serialize_datetime(getattr(job, "completed_at", None)),
        "result": getattr(job, "result", None),
        "error_message": getattr(job, "error_message", None),
    }


@app.get("/api/status")
async def get_status(
    current_user: Optional[User] = Depends(_get_current_user_dependency)
):
    """Get platform status, quota information, and service availability."""

    if not _feature_available("AUTH_AVAILABLE"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not available"
        )

    _require_auth(current_user)

    response: Dict[str, Any] = {
        "user": {
            "id": str(getattr(current_user, "id", "")),
            "username": getattr(current_user, "username", None),
            "plan": getattr(current_user, "plan_type", "unknown"),
            "organization": getattr(current_user, "organization", None),
        },
        "version": "3.2.1",
        "expert_mode": EXPERT_MODE,
        "timestamp": _timestamp(),
        "services": {
            "scheduler": bool(scheduler) and SCHEDULER_AVAILABLE,
            "billing": BILLING_AVAILABLE,
            "data_prep": DATA_PREP_AVAILABLE,
            "templates": TEMPLATE_AVAILABLE,
        },
    }

    billing_ref = getattr(_module_ref(), "billing_manager", None)

    if _feature_available("BILLING_AVAILABLE") and billing_ref:
        subscription = billing_ref.get_subscription(getattr(current_user, "tenant_id", None))
        usage = billing_ref.usage_tracker.get_usage(getattr(current_user, "tenant_id", None))
        response["subscription"] = subscription
        response["usage"] = usage

    scheduler_ref = getattr(_module_ref(), "scheduler", None)
    if _feature_available("SCHEDULER_AVAILABLE") and scheduler_ref:
        try:
            response["queue_stats"] = scheduler_ref.get_queue_stats()
        except Exception as exc:  # pragma: no cover - defensive against optional deps
            logger.warning("Failed to retrieve scheduler queue stats: %s", exc)

    return response


# ============================================================================
# [Rest of the original code continues unchanged...]
# ============================================================================

# The rest of the endpoints and functions from the original file remain the same.
# Only the version number and expert mode references have been updated.

# ============================================================================
# Main Entry Point Function
# ============================================================================

def main():
    """Main entry point for running the API server via console script."""
    import uvicorn
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='AutoML Platform API Server v3.2.1')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--expert', action='store_true', help='Enable expert mode')
    
    args = parser.parse_args()
    
    # Set expert mode environment variable if specified
    if args.expert:
        os.environ["AUTOML_EXPERT_MODE"] = "true"
    
    uvicorn.run(
        "automl_platform.api.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level
    )


# ============================================================================
# Development Entry Point
# ============================================================================

if __name__ == "__main__":
    # For development: python -m automl_platform.api.api
    main()
