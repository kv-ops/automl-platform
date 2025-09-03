"""
Enhanced FastAPI application for AutoML Platform
Production-ready with rate limiting, monitoring, and comprehensive endpoints
Version: 3.0.0 - Full Enterprise Features with Autoscaling
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
from automl_platform.infrastructure import TenantManager, SecurityManager, DeploymentManager
from automl_platform.api.billing import BillingManager, UsageTracker, PlanType, BillingPeriod

# Data connectors
from automl_platform.api.connectors import ConnectorFactory, ConnectionConfig

# Streaming
from automl_platform.api.streaming import StreamingOrchestrator, StreamConfig, MLStreamProcessor

# MLOps
from automl_platform.mlops_service import MLflowRegistry, RetrainingService, ModelExporter
from automl_platform.export_service import ModelExporter as EnhancedModelExporter
from automl_platform.ab_testing import ABTestingService, MetricsComparator

# Authentication
from automl_platform.auth import TokenService, RBACService, QuotaService, AuditService, auth_router

# Autoscaling - CORRECTED IMPORT
from automl_platform.autoscaling import (
    AutoscalingService, 
    ResourceManager as AutoscaleResourceManager,
    GPUScheduler as AutoscaleGPUScheduler,
    JobScheduler as AutoscaleJobScheduler,
    JobRequest, 
    JobStatus, 
    QueueType
)

# Scheduler - Import from existing scheduler.py
from automl_platform.scheduler import (
    SchedulerFactory,
    PLAN_LIMITS,
    PlanType as SchedulerPlanType
)

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

if hasattr(config, 'distributed') and config.distributed.enabled:
    ray.init(
        address=config.distributed.ray_address,
        num_cpus=config.distributed.num_cpus,
        num_gpus=config.distributed.num_gpus,
        object_store_memory=config.distributed.object_store_memory_gb * 1024 * 1024 * 1024
    )

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
    app.state.billing_manager = BillingManager()
    app.state.usage_tracker = UsageTracker()
    
    # Initialize deployment manager
    app.state.deployment_manager = DeploymentManager(app.state.tenant_manager)
    
    # Initialize scheduler using SchedulerFactory
    app.state.scheduler = SchedulerFactory.create_scheduler(config, app.state.billing_manager)
    
    # Initialize autoscaling service
    app.state.autoscaling = AutoscalingService(
        config=config,
        tenant_manager=app.state.tenant_manager,
        billing_manager=app.state.billing_manager
    )
    
    # Initialize MLOps components
    app.state.mlflow_registry = MLflowRegistry(config)
    app.state.model_exporter = ModelExporter(config)
    app.state.enhanced_exporter = EnhancedModelExporter()
    app.state.ab_testing_service = ABTestingService(app.state.mlflow_registry)
    
    # Initialize authentication services
    app.state.token_service = TokenService()
    app.state.rbac_service = RBACService(app.state.tenant_manager.Session())
    app.state.quota_service = QuotaService(
        app.state.tenant_manager.Session(), 
        app.state.billing_manager.redis_client if hasattr(app.state.billing_manager, 'redis_client') else None
    )
    app.state.audit_service = AuditService(app.state.tenant_manager.Session())
    
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
    if hasattr(config, 'distributed') and config.distributed.dask_enabled:
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
    if hasattr(config, 'distributed') and config.distributed.enabled:
        ray.shutdown()
    
    if app.state.dask_client:
        await app.state.dask_client.close()
    
    # Flush audit logs
    app.state.audit_service.log_action(
        user_id=None,
        tenant_id=None,
        action="system_shutdown",
        response_status=200
    )
    
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

# Include auth router
app.include_router(auth_router)

# Rate limiter with plan-based limits
def get_rate_limit_key(request: Request):
    """Get rate limit key based on user and plan"""
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
            "limits": PLAN_LIMITS["free"]
        }
    
    token = credentials.credentials
    try:
        # Verify JWT token using TokenService
        payload = app.state.token_service.verify_token(token)
        
        # Get tenant information
        tenant = app.state.tenant_manager.get_tenant(payload.get("tenant_id", "default"))
        
        return {
            "tenant_id": tenant.tenant_id,
            "user_id": payload["sub"],
            "plan": tenant.plan,
            "limits": PLAN_LIMITS.get(tenant.plan, PLAN_LIMITS["free"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

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
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                token = auth_header.split(" ")[1]
                payload = app.state.token_service.verify_token(token)
                tenant_id = payload.get("tenant_id", "default")
                
                # Check quotas using QuotaService
                if "/train" in request.url.path:
                    if not app.state.quota_service.check_quota(
                        app.state.tenant_manager.get_tenant(tenant_id), 
                        "concurrent_jobs", 
                        1
                    ):
                        return JSONResponse(
                            status_code=429,
                            content={"detail": "Concurrent job limit reached for your plan"}
                        )
                
                if "/llm" in request.url.path:
                    if not app.state.quota_service.check_quota(
                        app.state.tenant_manager.get_tenant(tenant_id),
                        "api_calls",
                        1
                    ):
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
            payload = app.state.token_service.verify_token(token)
            tenant_info = {
                "tenant_id": payload.get("tenant_id", "unknown"),
                "user_id": payload.get("sub", "unknown")
            }
        except:
            pass
    
    # Process request
    response = await call_next(request)
    
    # Log to audit
    duration = time.time() - start_time
    app.state.audit_service.log_action(
        user_id=tenant_info["user_id"],
        tenant_id=tenant_info["tenant_id"],
        action=f"{request.method} {request.url.path}",
        response_status=response.status_code,
        ip_address=request.client.host if request.client else "unknown"
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
            "mlflow": "healthy" if app.state.mlflow_registry else "disabled",
            "celery": "healthy" if config.worker.enabled else "disabled",
            "ray": "healthy" if hasattr(config, 'distributed') and config.distributed.enabled else "disabled",
            "autoscaling": "healthy" if app.state.autoscaling else "disabled"
        }
    }
    
    # Check autoscaling status
    if app.state.autoscaling:
        system_status = app.state.autoscaling.get_system_status()
        health_status["components"]["gpu"] = f"{len(system_status['gpus'])} GPUs available"
        health_status["components"]["cluster"] = f"{system_status['cluster']['nodes']} nodes"
    
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
    
    # Check concurrent job limits using autoscaling service
    current_status = app.state.autoscaling.get_system_status()
    tenant_jobs = [j for j in current_status["scheduler"]["running_jobs"] 
                   if j.get("tenant_id") == tenant["tenant_id"]]
    
    if len(tenant_jobs) >= tenant["limits"]["max_concurrent_jobs"]:
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
    
    # Create job request for autoscaling
    job = JobRequest(
        tenant_id=tenant["tenant_id"],
        user_id=tenant["user_id"],
        plan_type=tenant["plan"],
        task_type="train",
        queue_type=QueueType.GPU_TRAINING if train_request.use_gpu else QueueType.CPU_PRIORITY,
        payload={
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "target_column": target_column,
            "train_request": train_request.dict()
        },
        estimated_memory_gb=4.0,  # Estimate based on dataset size
        estimated_time_minutes=train_request.max_runtime_seconds // 60,
        requires_gpu=train_request.use_gpu,
        num_gpus=1 if train_request.use_gpu else 0,
        gpu_memory_gb=8.0 if train_request.use_gpu else 0,
        priority=tenant["limits"]["priority"]
    )
    
    # Submit job through autoscaling service
    job_id = app.state.autoscaling.submit_job(job)
    
    # Update metrics
    training_jobs.labels(
        status="queued",
        tenant=tenant["tenant_id"],
        plan=tenant["plan"]
    ).inc()
    
    # Track usage for billing
    app.state.quota_service.consume_quota(
        app.state.tenant_manager.get_tenant(tenant["tenant_id"]),
        "compute_minutes",
        train_request.max_runtime_seconds // 60
    )
    
    return {
        "job_id": job_id,
        "experiment_id": experiment_id,
        "status": "queued",
        "queue_type": job.queue_type.queue_name,
        "estimated_wait_time": f"{job.estimated_time_minutes} minutes",
        "cluster_status": app.state.autoscaling.get_system_status()["cluster"]
    }

@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(
    request: Request,
    job_id: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get job status from autoscaling service"""
    
    job = app.state.autoscaling.get_job_status(job_id)
    
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    
    # Verify tenant owns this job
    if job.tenant_id != tenant["tenant_id"]:
        raise HTTPException(403, "Access denied")
    
    return {
        "job_id": job_id,
        "status": job.status.value,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "queue_type": job.queue_type.queue_name,
        "error_message": job.error_message,
        "result": job.result
    }

# ============================================================================
# Autoscaling Management Endpoints
# ============================================================================

@app.get("/api/v1/autoscaling/status")
async def get_autoscaling_status(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get autoscaling system status"""
    
    status = app.state.autoscaling.get_system_status()
    
    # Filter sensitive information based on plan
    if tenant["plan"] not in ["pro", "enterprise"]:
        # Limited view for free/trial users
        return {
            "timestamp": status["timestamp"],
            "your_jobs": {
                "pending": len([j for j in status["scheduler"]["pending_jobs"] 
                               if j.get("tenant_id") == tenant["tenant_id"]]),
                "running": len([j for j in status["scheduler"]["running_jobs"] 
                               if j.get("tenant_id") == tenant["tenant_id"]])
            },
            "cluster_utilization": status["cluster"]["utilization"]
        }
    
    return status

@app.post("/api/v1/autoscaling/scale")
async def scale_cluster(
    request: Request,
    target_workers: int,
    tenant: Dict = Depends(get_current_tenant)
):
    """Manually scale the cluster (admin only)"""
    
    # Check if user is admin
    if not app.state.rbac_service.check_permission(
        app.state.tenant_manager.get_tenant(tenant["tenant_id"]),
        "cluster",
        "scale"
    ):
        raise HTTPException(403, "Admin access required")
    
    success = await app.state.autoscaling.scale_workers(target_workers)
    
    return {
        "success": success,
        "target_workers": target_workers,
        "current_status": app.state.autoscaling.get_system_status()["cluster"]
    }

@app.get("/api/v1/autoscaling/optimize")
async def optimize_resources(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get resource optimization recommendations"""
    
    if tenant["plan"] not in ["enterprise"]:
        raise HTTPException(403, "Enterprise plan required")
    
    optimizations = await app.state.autoscaling.optimize_resources()
    
    return optimizations

# ============================================================================
# GPU Management Endpoints
# ============================================================================

@app.get("/api/v1/gpu/status")
async def get_gpu_status(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get GPU cluster status"""
    
    system_status = app.state.autoscaling.get_system_status()
    gpu_status = system_status.get("gpus", [])
    
    # Check GPU access for tenant
    has_gpu_access = tenant["limits"].get("gpu_access", False)
    
    return {
        "total_gpus": len(gpu_status),
        "available_gpus": sum(1 for gpu in gpu_status if not gpu.get("allocated")),
        "your_gpu_access": has_gpu_access,
        "gpu_details": gpu_status if has_gpu_access else None,
        "upgrade_required": not has_gpu_access,
        "upgrade_message": "Upgrade to Pro or Enterprise plan for GPU access" if not has_gpu_access else None
    }

@app.post("/api/v1/gpu/request")
async def request_gpu(
    request: Request,
    duration_hours: int = 1,
    tenant: Dict = Depends(get_current_tenant)
):
    """Request GPU allocation"""
    
    if not tenant["limits"].get("gpu_access", False):
        raise HTTPException(403, "GPU access not available for your plan")
    
    # Check GPU hours quota
    if not app.state.quota_service.check_quota(
        app.state.tenant_manager.get_tenant(tenant["tenant_id"]),
        "gpu_hours",
        duration_hours
    ):
        raise HTTPException(429, "GPU hours quota exceeded")
    
    # Create GPU job request
    job = JobRequest(
        tenant_id=tenant["tenant_id"],
        user_id=tenant["user_id"],
        plan_type=tenant["plan"],
        task_type="gpu_allocation",
        queue_type=QueueType.GPU_INFERENCE,
        requires_gpu=True,
        num_gpus=1,
        gpu_memory_gb=8.0,
        estimated_time_minutes=duration_hours * 60
    )
    
    job_id = app.state.autoscaling.submit_job(job)
    
    # Consume quota
    app.state.quota_service.consume_quota(
        app.state.tenant_manager.get_tenant(tenant["tenant_id"]),
        "gpu_hours",
        duration_hours
    )
    
    return {
        "allocation_id": job_id,
        "duration_hours": duration_hours,
        "status": "pending",
        "estimated_wait_time": "Check status for updates"
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
        model = app.state.mlflow_registry.get_production_model(
            export_request.model_id
        )
        if not model:
            raise FileNotFoundError()
    except:
        raise HTTPException(404, f"Model {export_request.model_id} not found")
    
    # Export based on format
    try:
        if export_request.format.lower() == "onnx":
            # Create sample input for ONNX conversion
            sample_input = np.random.randn(1, 10).astype(np.float32)  # Adjust based on your model
            
            success = app.state.model_exporter.export_to_onnx(
                model,
                sample_input,
                f"exports/{tenant['tenant_id']}/{export_request.model_id}.onnx"
            )
            
            if not success:
                raise Exception("ONNX export failed")
            
            file_extension = ".onnx"
            
        elif export_request.format.lower() == "pmml":
            success = app.state.model_exporter.export_to_pmml(
                model,
                f"exports/{tenant['tenant_id']}/{export_request.model_id}.pmml"
            )
            
            if not success:
                raise Exception("PMML export failed")
            
            file_extension = ".pmml"
            
        elif export_request.format.lower() == "tensorflow_lite":
            sample_input = np.random.randn(1, 10).astype(np.float32)
            
            success = app.state.model_exporter.export_to_tensorflow_lite(
                model,
                sample_input,
                f"exports/{tenant['tenant_id']}/{export_request.model_id}.tflite"
            )
            
            if not success:
                raise Exception("TensorFlow Lite export failed")
            
            file_extension = ".tflite"
        else:
            raise HTTPException(400, f"Unsupported export format: {export_request.format}")
        
        export_path = f"exports/{tenant['tenant_id']}/{export_request.model_id}{file_extension}"
        
        # Track usage
        app.state.audit_service.log_action(
            user_id=tenant["user_id"],
            tenant_id=tenant["tenant_id"],
            action=f"model_export_{export_request.format}",
            resource_type="model",
            resource_id=export_request.model_id,
            response_status=200
        )
        
        return {
            "model_id": export_request.model_id,
            "export_format": export_request.format,
            "export_path": export_path,
            "optimized": export_request.optimize_for_edge,
            "quantized": export_request.quantize
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
    
    # Create A/B test
    test_id = app.state.ab_testing_service.create_ab_test(
        model_name=ab_test_request.name,
        champion_version=ab_test_request.model_a_id,
        challenger_version=ab_test_request.model_b_id,
        traffic_split=ab_test_request.traffic_split
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
    
    results = app.state.ab_testing_service.get_test_results(test_id)
    
    if not results:
        raise HTTPException(404, f"A/B test {test_id} not found")
    
    return results

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
            # Get job status from autoscaling service
            job_status = app.state.autoscaling.get_job_status(job_id)
            
            if job_status:
                await websocket.send_json({
                    "type": "job_update",
                    "job_id": job_id,
                    "status": job_status.status.value,
                    "queue_type": job_status.queue_type.queue_name,
                    "started_at": job_status.started_at.isoformat() if job_status.started_at else None,
                    "error_message": job_status.error_message
                })
                
                if job_status.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
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
