"""
Enhanced FastAPI application for AutoML Platform
Production-ready with rate limiting, monitoring, and comprehensive endpoints
Version: 3.0.0 - Full Enterprise Features
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
from enum import Enum

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
import uvicorn

# Distributed computing
from celery import Celery
import ray
from dask.distributed import Client as DaskClient

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

# Data connectors
from automl_platform.api.connectors import ConnectorFactory, ConnectionConfig

# Streaming
from automl_platform.api.streaming import StreamingOrchestrator, StreamConfig, MLStreamProcessor

# MLOps
from automl_platform.api.mlops import MLflowIntegration, ModelRegistry, AutoRetrainer
from automl_platform.api.export import ONNXExporter, PMMLExporter, EdgeDeploymentOptimizer

# A/B Testing
from automl_platform.api.ab_testing import ABTestManager, ModelComparator

# SSO and Audit
from automl_platform.api.auth import SSOManager, AuditLogger, GDPRCompliance

# Autoscaling
from automl_platform.api.autoscaling import JobScheduler, ResourceManager, GPUScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

config = load_config(os.getenv("CONFIG_PATH", "config.yaml"))

# ============================================================================
# Celery Configuration for Distributed Tasks
# ============================================================================

celery_app = Celery(
    'automl_tasks',
    broker=config.worker.broker_url if config.worker.enabled else 'redis://localhost:6379/0',
    backend=config.worker.result_backend if config.worker.enabled else 'redis://localhost:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_time_limit=config.worker.task_time_limit if config.worker.enabled else 3600,
    task_soft_time_limit=config.worker.task_time_limit - 60 if config.worker.enabled else 3540,
)

# ============================================================================
# Ray Configuration for Distributed Training
# ============================================================================

if config.distributed.enabled:
    ray.init(
        address=config.distributed.ray_address,
        num_cpus=config.distributed.num_cpus,
        num_gpus=config.distributed.num_gpus,
        object_store_memory=config.distributed.object_store_memory_gb * 1024 * 1024 * 1024
    )

# ============================================================================
# Plan Limits Configuration
# ============================================================================

class PlanLimits:
    """Define resource limits per plan"""
    LIMITS = {
        "free": {
            "max_concurrent_jobs": 1,
            "max_file_size_mb": 10,
            "max_models_per_month": 10,
            "max_predictions_per_day": 100,
            "max_llm_calls_per_day": 10,
            "gpu_access": False,
            "distributed_training": False,
            "priority": 0
        },
        "trial": {
            "max_concurrent_jobs": 3,
            "max_file_size_mb": 100,
            "max_models_per_month": 100,
            "max_predictions_per_day": 1000,
            "max_llm_calls_per_day": 100,
            "gpu_access": False,
            "distributed_training": False,
            "priority": 1
        },
        "pro": {
            "max_concurrent_jobs": 10,
            "max_file_size_mb": 1000,
            "max_models_per_month": -1,  # Unlimited
            "max_predictions_per_day": -1,
            "max_llm_calls_per_day": 1000,
            "gpu_access": True,
            "distributed_training": True,
            "priority": 2
        },
        "enterprise": {
            "max_concurrent_jobs": -1,  # Unlimited
            "max_file_size_mb": -1,
            "max_models_per_month": -1,
            "max_predictions_per_day": -1,
            "max_llm_calls_per_day": -1,
            "gpu_access": True,
            "distributed_training": True,
            "priority": 3
        }
    }

# ============================================================================
# Metrics with Custom Registry
# ============================================================================

metrics_registry = CollectorRegistry()

request_count = Counter(
    'automl_api_requests_total', 
    'Total API requests', 
    ['method', 'endpoint', 'status', 'tenant'],
    registry=metrics_registry
)

request_duration = Histogram(
    'automl_api_request_duration_seconds', 
    'API request duration', 
    ['method', 'endpoint', 'tenant'],
    registry=metrics_registry
)

active_models = Gauge(
    'automl_active_models', 
    'Number of active models',
    ['tenant'],
    registry=metrics_registry
)

training_jobs = Gauge(
    'automl_training_jobs', 
    'Number of training jobs', 
    ['status', 'tenant', 'plan'],
    registry=metrics_registry
)

gpu_utilization = Gauge(
    'automl_gpu_utilization',
    'GPU utilization percentage',
    ['gpu_id'],
    registry=metrics_registry
)

llm_calls = Counter(
    'automl_llm_calls_total',
    'Total LLM API calls',
    ['tenant', 'model'],
    registry=metrics_registry
)

# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting AutoML API v3.0.0...")
    
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
    
    # Initialize deployment manager
    app.state.deployment_manager = DeploymentManager(app.state.tenant_manager)
    
    # Initialize job scheduler and resource manager
    app.state.job_scheduler = JobScheduler(
        max_workers=config.worker.max_workers if config.worker.enabled else 4,
        celery_app=celery_app if config.worker.enabled else None
    )
    app.state.resource_manager = ResourceManager(plan_limits=PlanLimits.LIMITS)
    app.state.gpu_scheduler = GPUScheduler(num_gpus=config.distributed.num_gpus if config.distributed.enabled else 0)
    
    # Initialize MLOps components
    app.state.mlflow = MLflowIntegration(
        tracking_uri=config.mlops.mlflow_tracking_uri if hasattr(config, 'mlops') else "sqlite:///mlflow.db"
    )
    app.state.model_registry = ModelRegistry(mlflow=app.state.mlflow)
    app.state.auto_retrainer = AutoRetrainer(
        scheduler_backend=config.mlops.scheduler_backend if hasattr(config, 'mlops') else "celery"
    )
    
    # Initialize A/B testing
    app.state.ab_test_manager = ABTestManager()
    app.state.model_comparator = ModelComparator()
    
    # Initialize SSO and Audit
    if config.sso.enabled:
        app.state.sso_manager = SSOManager(
            provider=config.sso.provider,
            client_id=config.sso.client_id,
            client_secret=config.sso.client_secret,
            redirect_uri=config.sso.redirect_uri
        )
    else:
        app.state.sso_manager = None
    
    app.state.audit_logger = AuditLogger(
        backend=config.audit.backend if hasattr(config, 'audit') else "database",
        retention_days=config.audit.retention_days if hasattr(config, 'audit') else 90
    )
    app.state.gdpr_compliance = GDPRCompliance()
    
    # Initialize exporters
    app.state.onnx_exporter = ONNXExporter()
    app.state.pmml_exporter = PMMLExporter()
    app.state.edge_optimizer = EdgeDeploymentOptimizer()
    
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
    app.state.quality_agent = IntelligentDataQualityAgent(
        llm_provider=app.state.llm_assistant.llm if app.state.llm_assistant else None
    )
    
    # Initialize Dask client for distributed processing
    if config.distributed.dask_enabled:
        app.state.dask_client = DaskClient(config.distributed.dask_scheduler_address)
    else:
        app.state.dask_client = None
    
    # Initialize orchestrators pool
    app.state.orchestrators = {}
    
    # Initialize WebSocket connections
    app.state.websocket_connections = {}
    
    # Initialize cache for model pipelines
    app.state.pipeline_cache = {}
    
    logger.info("AutoML API v3.0.0 started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AutoML API...")
    
    # Clean up resources
    if app.state.monitoring:
        app.state.monitoring.save_monitoring_data()
    
    # Close WebSocket connections
    for ws in app.state.websocket_connections.values():
        await ws.close()
    
    # Shutdown distributed computing
    if config.distributed.enabled:
        ray.shutdown()
    
    if app.state.dask_client:
        await app.state.dask_client.close()
    
    # Flush audit logs
    app.state.audit_logger.flush()
    
    logger.info("AutoML API shutdown complete")

# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="AutoML Platform API",
    description="Enterprise AutoML platform with MLOps, distributed training, and comprehensive monitoring",
    version="3.0.0",
    docs_url="/docs" if getattr(config.api, 'enable_docs', True) else None,
    redoc_url="/redoc" if getattr(config.api, 'enable_docs', True) else None,
    lifespan=lifespan
)

# Rate limiter with plan-based limits
def get_rate_limit_key(request: Request):
    """Get rate limit key based on user and plan"""
    # This would be enhanced to get actual user plan
    return get_remote_address(request)

limiter = Limiter(key_func=get_rate_limit_key)
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
# Security & Authentication
# ============================================================================

security = HTTPBearer()

async def get_current_tenant(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Get current tenant with plan information"""
    if not getattr(config.api, 'enable_auth', False):
        return {
            "tenant_id": "default",
            "user_id": "anonymous",
            "plan": "free",
            "limits": PlanLimits.LIMITS["free"]
        }
    
    token = credentials.credentials
    try:
        # Verify JWT token
        payload = jwt.decode(
            token, 
            getattr(config.api, 'jwt_secret', 'secret'), 
            algorithms=[getattr(config.api, 'jwt_algorithm', 'HS256')]
        )
        
        # Get tenant information
        tenant = app.state.tenant_manager.get_tenant(payload.get("tenant_id", "default"))
        
        return {
            "tenant_id": tenant.tenant_id,
            "user_id": payload["user_id"],
            "plan": tenant.plan,
            "limits": PlanLimits.LIMITS.get(tenant.plan, PlanLimits.LIMITS["free"])
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def create_token(user_id: str, tenant_id: str) -> str:
    """Create JWT token with tenant information"""
    payload = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "exp": datetime.utcnow() + timedelta(minutes=getattr(config.api, 'jwt_expiration_minutes', 60))
    }
    return jwt.encode(payload, getattr(config.api, 'jwt_secret', 'secret'), algorithm=getattr(config.api, 'jwt_algorithm', 'HS256'))

# ============================================================================
# Middleware for Billing & Quotas
# ============================================================================

@app.middleware("http")
async def billing_quota_middleware(request: Request, call_next):
    """Check billing quotas before processing requests"""
    # Skip for health checks and metrics
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    # Get tenant from token if auth is enabled
    if getattr(config.api, 'enable_auth', False):
        # Extract tenant_id from JWT (simplified - should use proper auth)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(
                    token,
                    getattr(config.api, 'jwt_secret', 'secret'),
                    algorithms=[getattr(config.api, 'jwt_algorithm', 'HS256')]
                )
                tenant_id = payload.get("tenant_id", "default")
                
                # Check if account is suspended
                if app.state.billing_manager.is_suspended(tenant_id):
                    return JSONResponse(
                        status_code=402,
                        content={"detail": "Account suspended due to billing issues"}
                    )
                
                # Check quotas for specific endpoints
                if "/train" in request.url.path:
                    if not app.state.resource_manager.can_start_job(tenant_id):
                        return JSONResponse(
                            status_code=429,
                            content={"detail": "Concurrent job limit reached for your plan"}
                        )
                
                if "/llm" in request.url.path:
                    if not app.state.resource_manager.can_use_llm(tenant_id):
                        return JSONResponse(
                            status_code=429,
                            content={"detail": "Daily LLM call limit reached for your plan"}
                        )
                        
            except:
                pass
    
    response = await call_next(request)
    return response

@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    """Audit all API calls"""
    start_time = time.time()
    
    # Get user/tenant info
    tenant_info = {"tenant_id": "unknown", "user_id": "unknown"}
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(
                token,
                getattr(config.api, 'jwt_secret', 'secret'),
                algorithms=[getattr(config.api, 'jwt_algorithm', 'HS256')]
            )
            tenant_info = {
                "tenant_id": payload.get("tenant_id", "unknown"),
                "user_id": payload.get("user_id", "unknown")
            }
        except:
            pass
    
    # Process request
    response = await call_next(request)
    
    # Log to audit
    duration = time.time() - start_time
    app.state.audit_logger.log(
        action=f"{request.method} {request.url.path}",
        tenant_id=tenant_info["tenant_id"],
        user_id=tenant_info["user_id"],
        details={
            "status_code": response.status_code,
            "duration": duration,
            "ip_address": request.client.host if request.client else "unknown"
        }
    )
    
    # Record metrics
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        tenant=tenant_info["tenant_id"]
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path,
        tenant=tenant_info["tenant_id"]
    ).observe(duration)
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(duration)
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    
    return response

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
    use_gpu: Optional[bool] = Field(False, description="Use GPU for training")
    distributed: Optional[bool] = Field(False, description="Use distributed training")
    num_workers: Optional[int] = Field(1, description="Number of workers for distributed training")

class PredictRequest(BaseModel):
    """Prediction request model"""
    model_id: str = Field(..., description="Model ID to use for prediction")
    data: Dict[str, Any] = Field(..., description="Input data for prediction")
    track: Optional[bool] = Field(True, description="Track predictions for monitoring")

class ExportRequest(BaseModel):
    """Model export request"""
    model_id: str
    format: str = Field(..., description="Export format: onnx, pmml, tensorflow_lite")
    optimize_for_edge: bool = False
    quantize: bool = False
    target_device: Optional[str] = None

class ABTestRequest(BaseModel):
    """A/B test request"""
    name: str
    model_a_id: str
    model_b_id: str
    traffic_split: float = 0.5
    duration_hours: int = 24
    metrics: List[str] = ["accuracy", "latency"]

class RetrainingSchedule(BaseModel):
    """Retraining schedule configuration"""
    model_id: str
    schedule_type: str = Field("cron", description="Schedule type: cron, interval, trigger")
    schedule_config: Dict[str, Any]
    data_source: str
    retrain_threshold: Optional[float] = None
    notification_emails: Optional[List[str]] = None

# ============================================================================
# Health & Monitoring Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "environment": getattr(config, 'environment', 'development'),
        "components": {
            "storage": "healthy" if app.state.storage else "not configured",
            "monitoring": "healthy" if app.state.monitoring else "disabled",
            "mlflow": "healthy" if app.state.mlflow else "disabled",
            "celery": "healthy" if config.worker.enabled else "disabled",
            "ray": "healthy" if config.distributed.enabled else "disabled",
            "gpu": f"{app.state.gpu_scheduler.available_gpus()}/{config.distributed.num_gpus} available" if config.distributed.enabled else "disabled"
        }
    }
    
    # Check if any component is unhealthy
    if any(v == "unhealthy" for v in health_status["components"].values()):
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        BytesIO(generate_latest(metrics_registry)),
        media_type=CONTENT_TYPE_LATEST
    )

# ============================================================================
# Training Endpoints with Autoscaling
# ============================================================================

@app.post("/api/v1/train")
async def start_training(
    request: Request,
    background_tasks: BackgroundTasks,
    dataset_id: str,
    target_column: str,
    train_request: TrainRequest,
    tenant: Dict = Depends(get_current_tenant)
):
    """Start training job with autoscaling and resource management"""
    
    # Check concurrent job limits
    current_jobs = app.state.job_scheduler.get_tenant_jobs(tenant["tenant_id"])
    if len(current_jobs) >= tenant["limits"]["max_concurrent_jobs"]:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum concurrent jobs ({tenant['limits']['max_concurrent_jobs']}) reached for {tenant['plan']} plan"
        )
    
    # Check GPU access
    if train_request.use_gpu and not tenant["limits"]["gpu_access"]:
        raise HTTPException(
            status_code=403,
            detail=f"GPU access not available for {tenant['plan']} plan. Please upgrade to Pro or Enterprise."
        )
    
    # Check distributed training access
    if train_request.distributed and not tenant["limits"]["distributed_training"]:
        raise HTTPException(
            status_code=403,
            detail=f"Distributed training not available for {tenant['plan']} plan. Please upgrade to Pro or Enterprise."
        )
    
    # Load dataset
    if not app.state.storage:
        raise HTTPException(503, "Storage not configured")
    
    try:
        df = app.state.storage.load_dataset(dataset_id, tenant_id=tenant["tenant_id"])
    except FileNotFoundError:
        raise HTTPException(404, f"Dataset {dataset_id} not found")
    
    # Validate target column
    if target_column not in df.columns:
        raise HTTPException(400, f"Target column {target_column} not found")
    
    # Create experiment with MLflow tracking
    experiment_id = train_request.experiment_name or f"exp_{uuid.uuid4().hex[:8]}"
    mlflow_run = app.state.mlflow.create_run(
        experiment_name=f"{tenant['tenant_id']}/{experiment_id}",
        tags={
            "tenant_id": tenant["tenant_id"],
            "user_id": tenant["user_id"],
            "plan": tenant["plan"],
            "dataset_id": dataset_id,
            "target_column": target_column
        }
    )
    
    # Schedule job based on plan priority
    job_config = {
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "target_column": target_column,
        "train_request": train_request.dict(),
        "tenant": tenant,
        "mlflow_run_id": mlflow_run.info.run_id,
        "priority": tenant["limits"]["priority"]
    }
    
    if train_request.use_gpu:
        # Schedule GPU job
        job_id = app.state.gpu_scheduler.schedule_job(
            job_config,
            num_gpus=1,
            priority=tenant["limits"]["priority"]
        )
    elif train_request.distributed:
        # Schedule distributed job
        job_id = celery_app.send_task(
            'automl_tasks.distributed_train',
            args=[job_config],
            priority=tenant["limits"]["priority"],
            queue=f"priority_{tenant['limits']['priority']}"
        )
    else:
        # Schedule regular CPU job
        job_id = celery_app.send_task(
            'automl_tasks.train',
            args=[job_config],
            priority=tenant["limits"]["priority"],
            queue=f"priority_{tenant['limits']['priority']}"
        )
    
    # Track job
    app.state.job_scheduler.register_job(
        job_id=str(job_id),
        tenant_id=tenant["tenant_id"],
        experiment_id=experiment_id,
        job_type="training",
        resources={
            "gpu": train_request.use_gpu,
            "distributed": train_request.distributed,
            "workers": train_request.num_workers if train_request.distributed else 1
        }
    )
    
    # Update metrics
    training_jobs.labels(
        status="queued",
        tenant=tenant["tenant_id"],
        plan=tenant["plan"]
    ).inc()
    
    # Track usage for billing
    app.state.usage_tracker.track_training(
        tenant_id=tenant["tenant_id"],
        model_type="automl",
        duration_estimate=train_request.max_runtime_seconds,
        use_gpu=train_request.use_gpu
    )
    
    return {
        "job_id": str(job_id),
        "experiment_id": experiment_id,
        "mlflow_run_id": mlflow_run.info.run_id,
        "status": "queued",
        "estimated_wait_time": app.state.job_scheduler.estimate_wait_time(tenant["limits"]["priority"]),
        "queue_position": app.state.job_scheduler.get_queue_position(str(job_id))
    }

# ============================================================================
# Model Export Endpoints
# ============================================================================

@app.post("/api/v1/models/export")
async def export_model(
    request: Request,
    export_request: ExportRequest,
    tenant: Dict = Depends(get_current_tenant)
):
    """Export model to ONNX, PMML, or TensorFlow Lite"""
    
    # Load model
    try:
        model, metadata = app.state.storage.load_model(
            export_request.model_id,
            tenant_id=tenant["tenant_id"]
        )
    except FileNotFoundError:
        raise HTTPException(404, f"Model {export_request.model_id} not found")
    
    # Export based on format
    try:
        if export_request.format.lower() == "onnx":
            exported_model = app.state.onnx_exporter.export(
                model,
                optimize=export_request.optimize_for_edge,
                quantize=export_request.quantize
            )
            file_extension = ".onnx"
        elif export_request.format.lower() == "pmml":
            exported_model = app.state.pmml_exporter.export(model)
            file_extension = ".pmml"
        elif export_request.format.lower() == "tensorflow_lite":
            exported_model = app.state.edge_optimizer.convert_to_tflite(
                model,
                quantize=export_request.quantize,
                target_device=export_request.target_device
            )
            file_extension = ".tflite"
        else:
            raise HTTPException(400, f"Unsupported export format: {export_request.format}")
        
        # Save exported model
        export_path = f"exports/{tenant['tenant_id']}/{export_request.model_id}{file_extension}"
        app.state.storage.save_artifact(exported_model, export_path)
        
        # Track usage
        app.state.usage_tracker.track_export(
            tenant_id=tenant["tenant_id"],
            model_id=export_request.model_id,
            format=export_request.format
        )
        
        return {
            "model_id": export_request.model_id,
            "export_format": export_request.format,
            "export_path": export_path,
            "optimized": export_request.optimize_for_edge,
            "quantized": export_request.quantize,
            "file_size_mb": len(exported_model) / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(500, f"Export failed: {str(e)}")

# ============================================================================
# A/B Testing Endpoints
# ============================================================================

@app.post("/api/v1/ab_tests")
async def create_ab_test(
    request: Request,
    ab_test_request: ABTestRequest,
    tenant: Dict = Depends(get_current_tenant)
):
    """Create A/B test between two models"""
    
    # Verify both models exist
    for model_id in [ab_test_request.model_a_id, ab_test_request.model_b_id]:
        try:
            app.state.storage.load_model(model_id, tenant_id=tenant["tenant_id"])
        except FileNotFoundError:
            raise HTTPException(404, f"Model {model_id} not found")
    
    # Create A/B test
    test_id = app.state.ab_test_manager.create_test(
        name=ab_test_request.name,
        tenant_id=tenant["tenant_id"],
        model_a_id=ab_test_request.model_a_id,
        model_b_id=ab_test_request.model_b_id,
        traffic_split=ab_test_request.traffic_split,
        duration_hours=ab_test_request.duration_hours,
        metrics=ab_test_request.metrics
    )
    
    return {
        "test_id": test_id,
        "name": ab_test_request.name,
        "status": "active",
        "start_time": datetime.now().isoformat(),
        "end_time": (datetime.now() + timedelta(hours=ab_test_request.duration_hours)).isoformat()
    }

@app.get("/api/v1/ab_tests/{test_id}/results")
async def get_ab_test_results(
    request: Request,
    test_id: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get A/B test results with statistical significance"""
    
    results = app.state.ab_test_manager.get_results(test_id, tenant_id=tenant["tenant_id"])
    
    if not results:
        raise HTTPException(404, f"A/B test {test_id} not found")
    
    # Calculate statistical significance
    comparison = app.state.model_comparator.compare(
        results["model_a_metrics"],
        results["model_b_metrics"],
        metrics=results["metrics"]
    )
    
    return {
        "test_id": test_id,
        "status": results["status"],
        "model_a": {
            "id": results["model_a_id"],
            "requests": results["model_a_requests"],
            "metrics": results["model_a_metrics"]
        },
        "model_b": {
            "id": results["model_b_id"],
            "requests": results["model_b_requests"],
            "metrics": results["model_b_metrics"]
        },
        "statistical_analysis": comparison,
        "winner": comparison.get("winner"),
        "confidence_level": comparison.get("confidence_level")
    }

# ============================================================================
# MLOps Endpoints
# ============================================================================

@app.post("/api/v1/models/{model_id}/retrain")
async def schedule_retraining(
    request: Request,
    model_id: str,
    schedule: RetrainingSchedule,
    tenant: Dict = Depends(get_current_tenant)
):
    """Schedule automatic model retraining"""
    
    # Verify model exists
    try:
        model, metadata = app.state.storage.load_model(model_id, tenant_id=tenant["tenant_id"])
    except FileNotFoundError:
        raise HTTPException(404, f"Model {model_id} not found")
    
    # Schedule retraining
    schedule_id = app.state.auto_retrainer.schedule(
        model_id=model_id,
        tenant_id=tenant["tenant_id"],
        schedule_type=schedule.schedule_type,
        schedule_config=schedule.schedule_config,
        data_source=schedule.data_source,
        retrain_threshold=schedule.retrain_threshold,
        notification_emails=schedule.notification_emails
    )
    
    return {
        "schedule_id": schedule_id,
        "model_id": model_id,
        "schedule_type": schedule.schedule_type,
        "next_run": app.state.auto_retrainer.get_next_run_time(schedule_id),
        "status": "scheduled"
    }

@app.get("/api/v1/models/{model_id}/versions")
async def get_model_versions(
    request: Request,
    model_id: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get all versions of a model with metrics"""
    
    versions = app.state.model_registry.get_versions(
        model_id=model_id,
        tenant_id=tenant["tenant_id"]
    )
    
    if not versions:
        raise HTTPException(404, f"Model {model_id} not found")
    
    return {
        "model_id": model_id,
        "versions": [
            {
                "version": v["version"],
                "created_at": v["created_at"],
                "metrics": v["metrics"],
                "tags": v["tags"],
                "stage": v["stage"],  # staging, production, archived
                "mlflow_run_id": v["mlflow_run_id"]
            }
            for v in versions
        ],
        "current_production": next((v for v in versions if v["stage"] == "production"), None)
    }

@app.post("/api/v1/models/{model_id}/rollback")
async def rollback_model(
    request: Request,
    model_id: str,
    target_version: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Rollback model to a previous version"""
    
    try:
        app.state.model_registry.transition_stage(
            model_id=model_id,
            version=target_version,
            stage="production",
            tenant_id=tenant["tenant_id"]
        )
        
        # Archive current production version
        current_prod = app.state.model_registry.get_production_version(model_id, tenant["tenant_id"])
        if current_prod and current_prod["version"] != target_version:
            app.state.model_registry.transition_stage(
                model_id=model_id,
                version=current_prod["version"],
                stage="archived",
                tenant_id=tenant["tenant_id"]
            )
        
        return {
            "model_id": model_id,
            "rolled_back_to": target_version,
            "previous_version": current_prod["version"] if current_prod else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(500, f"Rollback failed: {str(e)}")

# ============================================================================
# SSO Endpoints
# ============================================================================

@app.get("/api/v1/auth/sso/login")
async def sso_login(request: Request):
    """Initiate SSO login"""
    if not app.state.sso_manager:
        raise HTTPException(503, "SSO not configured")
    
    auth_url = app.state.sso_manager.get_authorization_url()
    return {"auth_url": auth_url}

@app.get("/api/v1/auth/sso/callback")
async def sso_callback(request: Request, code: str, state: str):
    """Handle SSO callback"""
    if not app.state.sso_manager:
        raise HTTPException(503, "SSO not configured")
    
    try:
        user_info = await app.state.sso_manager.handle_callback(code, state)
        
        # Create or update user
        tenant = app.state.tenant_manager.get_or_create_tenant_for_sso_user(user_info)
        
        # Generate JWT token
        token = create_token(user_info["sub"], tenant.tenant_id)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_info": user_info,
            "tenant_id": tenant.tenant_id
        }
        
    except Exception as e:
        logger.error(f"SSO callback failed: {e}")
        raise HTTPException(400, f"SSO authentication failed: {str(e)}")

# ============================================================================
# GDPR Compliance Endpoints
# ============================================================================

@app.get("/api/v1/gdpr/export")
async def export_user_data(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Export all user data (GDPR compliance)"""
    
    data = app.state.gdpr_compliance.export_user_data(
        tenant_id=tenant["tenant_id"],
        user_id=tenant["user_id"]
    )
    
    # Create ZIP file with all data
    export_path = f"gdpr_exports/{tenant['tenant_id']}/{tenant['user_id']}_{datetime.now().strftime('%Y%m%d')}.zip"
    app.state.storage.save_artifact(data, export_path)
    
    return FileResponse(
        export_path,
        media_type="application/zip",
        filename=f"user_data_export_{tenant['user_id']}.zip"
    )

@app.delete("/api/v1/gdpr/delete")
async def delete_user_data(
    request: Request,
    confirm: bool = False,
    tenant: Dict = Depends(get_current_tenant)
):
    """Delete all user data (GDPR right to be forgotten)"""
    
    if not confirm:
        raise HTTPException(400, "Please confirm data deletion by setting confirm=true")
    
    # Schedule deletion (with 30-day grace period)
    deletion_id = app.state.gdpr_compliance.schedule_deletion(
        tenant_id=tenant["tenant_id"],
        user_id=tenant["user_id"],
        grace_period_days=30
    )
    
    return {
        "deletion_id": deletion_id,
        "scheduled_deletion_date": (datetime.now() + timedelta(days=30)).isoformat(),
        "message": "Your data deletion has been scheduled. You have 30 days to cancel this request."
    }

# ============================================================================
# Billing Endpoints
# ============================================================================

@app.get("/api/v1/billing/invoice/{tenant_id}")
async def generate_invoice(
    request: Request,
    tenant_id: str,
    month: Optional[int] = None,
    year: Optional[int] = None,
    tenant: Dict = Depends(get_current_tenant)
):
    """Generate monthly invoice"""
    
    # Check authorization
    if tenant["tenant_id"] != tenant_id and not tenant.get("is_admin"):
        raise HTTPException(403, "Access denied")
    
    # Generate invoice
    invoice = app.state.billing_manager.generate_invoice(
        tenant_id=tenant_id,
        month=month or datetime.now().month,
        year=year or datetime.now().year
    )
    
    return {
        "invoice_id": invoice["invoice_id"],
        "tenant_id": tenant_id,
        "period": f"{invoice['year']}-{invoice['month']:02d}",
        "plan": invoice["plan"],
        "usage": invoice["usage"],
        "charges": invoice["charges"],
        "total": invoice["total"],
        "currency": "USD",
        "due_date": invoice["due_date"],
        "status": invoice["status"]
    }

# ============================================================================
# GPU Monitoring Endpoints
# ============================================================================

@app.get("/api/v1/gpu/status")
async def get_gpu_status(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get GPU cluster status"""
    
    if not config.distributed.enabled:
        raise HTTPException(503, "GPU cluster not configured")
    
    gpu_status = app.state.gpu_scheduler.get_status()
    
    return {
        "total_gpus": gpu_status["total"],
        "available_gpus": gpu_status["available"],
        "gpu_queue_length": gpu_status["queue_length"],
        "gpu_utilization": gpu_status["utilization"],
        "gpu_memory": gpu_status["memory"],
        "your_gpu_access": tenant["limits"]["gpu_access"],
        "your_queue_position": app.state.gpu_scheduler.get_tenant_position(tenant["tenant_id"]) if tenant["limits"]["gpu_access"] else None
    }

# ============================================================================
# WebSocket for Real-time Updates
# ============================================================================

@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_updates(
    websocket: WebSocket,
    job_id: str
):
    """WebSocket for real-time job updates"""
    await websocket.accept()
    
    try:
        while True:
            # Get job status
            job_status = app.state.job_scheduler.get_job_status(job_id)
            
            if job_status:
                await websocket.send_json({
                    "type": "job_update",
                    "job_id": job_id,
                    "status": job_status["status"],
                    "progress": job_status.get("progress", 0),
                    "metrics": job_status.get("metrics", {}),
                    "eta": job_status.get("eta"),
                    "logs": job_status.get("recent_logs", [])
                })
                
                if job_status["status"] in ["completed", "failed", "cancelled"]:
                    break
            
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=getattr(config.api, 'host', '0.0.0.0'),
        port=getattr(config.api, 'port', 8000),
        reload=getattr(config, 'debug', False),
        log_level="info" if getattr(config, 'verbose', True) else "error",
        workers=1 if getattr(config, 'debug', False) else 4
    )
