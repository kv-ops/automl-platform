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

# Streaming - CORRECTED IMPORT
from automl_platform.api.streaming import (
    StreamingOrchestrator, 
    StreamConfig, 
    MLStreamProcessor,
    StreamMessage,
    KafkaStreamHandler
)

# MLOps
from automl_platform.mlops_service import MLflowRegistry, RetrainingService, ModelExporter
from automl_platform.export_service import ModelExporter as EnhancedModelExporter
from automl_platform.ab_testing import ABTestingService, MetricsComparator

# Authentication
from automl_platform.auth import TokenService, RBACService, QuotaService, AuditService, auth_router

# Scheduler - CORRECTED IMPORT
from automl_platform.scheduler import (
    SchedulerFactory,
    JobRequest,
    JobStatus,
    QueueType,
    PLAN_LIMITS,
    PlanType as SchedulerPlanType,
    CeleryScheduler,
    RayScheduler,
    LocalScheduler
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Management Endpoints
# ============================================================================

@app.get("/api/v1/config")
async def get_configuration(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get current configuration (filtered by permissions)"""
    
    # Only admins can see full config
    if tenant.get("role") == "admin":
        return app.state.config_manager.export_config(format="json", include_secrets=False)
    
    # Regular users get summary
    return app.state.config_manager.get_config_summary()

@app.post("/api/v1/config/reload")
async def reload_configuration(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Reload configuration from file (admin only)"""
    
    if tenant.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    
    success = app.state.config_manager.reload_config()
    
    if success:
        # Update global config reference
        global config
        config = app.state.config_manager.config
        
        return {"status": "success", "message": "Configuration reloaded"}
    else:
        return {"status": "no_change", "message": "Configuration unchanged"}

@app.get("/api/v1/config/features")
async def get_feature_flags(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get feature flags for current configuration"""
    return app.state.config_manager.get_feature_flags()

@app.get("/api/v1/config/limits")
async def get_limits(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get limits for current plan"""
    return app.state.config_manager.get_limits()

# ============================================================================
# Service Registry Endpoints
# ============================================================================

@app.get("/api/v1/services")
async def list_services(
    request: Request,
    service_type: Optional[str] = None,
    tenant: Dict = Depends(get_current_tenant)
):
    """List all registered services"""
    
    services = app.state.service_registry.list_services(service_type)
    
    # Get detailed info for each service
    service_info = []
    for service_name in services:
        info = app.state.service_registry.get_info(service_name)
        service_info.append({
            "name": service_name,
            "type": info.service_type,
            "status": info.status.value,
            "dependencies": info.dependencies,
            "registered_at": info.registered_at.isoformat()
        })
    
    return {
        "services": service_info,
        "total": len(service_info),
        "statistics": app.state.service_registry.get_statistics()
    }

@app.get("/api/v1/services/{service_name}/status")
async def get_service_status(
    request: Request,
    service_name: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get status of a specific service"""
    
    info = app.state.service_registry.get_info(service_name)
    
    if not info:
        raise HTTPException(404, f"Service {service_name} not found")
    
    return {
        "name": service_name,
        "type": info.service_type,
        "status": info.status.value,
        "dependencies": info.dependencies,
        "dependents": app.state.service_registry._get_dependent_services(service_name),
        "registered_at": info.registered_at.isoformat(),
        "last_health_check": info.last_health_check.isoformat() if info.last_health_check else None
    }

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

streaming_messages = Counter(
    'automl_streaming_messages_total',
    'Total streaming messages processed',
    ['platform', 'topic', 'status'],
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
    
    # Initialize scheduler using SchedulerFactory - CORRECTED
    app.state.scheduler = SchedulerFactory.create_scheduler(config, app.state.billing_manager)
    
    # Initialize streaming if configured
    app.state.streaming_orchestrators = {}
    
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
    
    # Initialize WebSocket connections manager
    app.state.websocket_manager = WebSocketManager()
    
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
    await app.state.websocket_manager.disconnect_all()
    
    # Stop streaming orchestrators
    for orchestrator in app.state.streaming_orchestrators.values():
        orchestrator.stop()
    
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
# WebSocket Manager
# ============================================================================

class WebSocketManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = []
        self.active_connections[client_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].remove(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
    
    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_text(message)
    
    async def broadcast(self, message: str):
        for connections in self.active_connections.values():
            for connection in connections:
                await connection.send_text(message)
    
    async def disconnect_all(self):
        for connections in self.active_connections.values():
            for connection in connections:
                await connection.close()
        self.active_connections.clear()

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
            "plan": PlanType.FREE.value,
            "limits": PLAN_LIMITS[PlanType.FREE.value]
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
            "limits": PLAN_LIMITS.get(tenant.plan, PLAN_LIMITS[PlanType.FREE.value])
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

class StreamingConfig(BaseModel):
    """Streaming configuration"""
    platform: str = Field("kafka", description="Streaming platform: kafka, flink, pulsar, redis")
    brokers: List[str] = Field(..., description="Broker addresses")
    topic: str = Field(..., description="Topic name")
    consumer_group: Optional[str] = Field("automl-consumer", description="Consumer group")
    batch_size: Optional[int] = Field(100, description="Batch size")
    window_size: Optional[int] = Field(60, description="Window size in seconds")

class ExportRequest(BaseModel):
    """Model export request"""
    model_id: str
    format: str = Field(..., description="Export format: onnx, pmml, tensorflow_lite")
    optimize_for_edge: bool = False
    quantize: bool = False
    target_device: Optional[str] = None

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
            "scheduler": "healthy" if app.state.scheduler else "disabled",
            "streaming": f"{len(app.state.streaming_orchestrators)} active" if app.state.streaming_orchestrators else "none"
        }
    }
    
    # Check scheduler status
    if app.state.scheduler and hasattr(app.state.scheduler, 'get_queue_stats'):
        queue_stats = app.state.scheduler.get_queue_stats()
        health_status["components"]["workers"] = f"{queue_stats.get('workers', 0)} workers"
        health_status["components"]["gpu_workers"] = f"{queue_stats.get('gpu_workers', 0)} GPU workers"
    
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
# Training Endpoints with Scheduler
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
    """Start training job with scheduler"""
    
    # Check concurrent job limits
    if app.state.scheduler and hasattr(app.state.scheduler, 'get_queue_stats'):
        queue_stats = app.state.scheduler.get_queue_stats()
        active_jobs = queue_stats.get('active_jobs', 0)
        
        if active_jobs >= tenant["limits"]["max_concurrent_jobs"]:
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
    
    # Create job request for scheduler
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
        estimated_memory_gb=4.0,
        estimated_time_minutes=train_request.max_runtime_seconds // 60,
        requires_gpu=train_request.use_gpu,
        num_gpus=1 if train_request.use_gpu else 0,
        gpu_memory_gb=8.0 if train_request.use_gpu else 0,
        priority=tenant["limits"]["queue_priority"]
    )
    
    # Submit job through scheduler
    job_id = app.state.scheduler.submit_job(job)
    
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
        "estimated_wait_time": f"{job.estimated_time_minutes} minutes"
    }

@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(
    request: Request,
    job_id: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get job status from scheduler"""
    
    job = app.state.scheduler.get_job_status(job_id)
    
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

@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(
    request: Request,
    job_id: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Cancel a running job"""
    
    job = app.state.scheduler.get_job_status(job_id)
    
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    
    if job.tenant_id != tenant["tenant_id"]:
        raise HTTPException(403, "Access denied")
    
    success = app.state.scheduler.cancel_job(job_id)
    
    if success:
        training_jobs.labels(
            status="cancelled",
            tenant=tenant["tenant_id"],
            plan=tenant["plan"]
        ).inc()
    
    return {"success": success, "job_id": job_id}

# ============================================================================
# Streaming Endpoints
# ============================================================================

@app.post("/api/v1/streaming/start")
async def start_streaming(
    request: Request,
    config: StreamingConfig,
    model_id: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Start streaming pipeline for real-time predictions"""
    
    # Check if streaming is enabled for plan
    if tenant["plan"] not in [PlanType.PRO.value, PlanType.ENTERPRISE.value]:
        raise HTTPException(403, "Streaming requires Pro or Enterprise plan")
    
    # Load model
    try:
        model = app.state.mlflow_registry.get_production_model(model_id)
        if not model:
            raise FileNotFoundError()
    except:
        raise HTTPException(404, f"Model {model_id} not found")
    
    # Create streaming configuration
    stream_config = StreamConfig(
        platform=config.platform,
        brokers=config.brokers,
        topic=config.topic,
        consumer_group=config.consumer_group or f"automl-{tenant['tenant_id']}",
        batch_size=config.batch_size,
        window_size=config.window_size
    )
    
    # Create ML processor
    processor = MLStreamProcessor(stream_config, model=model)
    
    # Create orchestrator
    orchestrator = StreamingOrchestrator(stream_config)
    orchestrator.set_processor(processor)
    
    # Store orchestrator
    stream_id = f"stream_{uuid.uuid4().hex[:8]}"
    app.state.streaming_orchestrators[stream_id] = orchestrator
    
    # Start streaming in background
    asyncio.create_task(orchestrator.start(output_topic=f"{config.topic}_predictions"))
    
    return {
        "stream_id": stream_id,
        "status": "started",
        "platform": config.platform,
        "topic": config.topic,
        "output_topic": f"{config.topic}_predictions"
    }

@app.get("/api/v1/streaming/{stream_id}/status")
async def get_streaming_status(
    request: Request,
    stream_id: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get streaming pipeline status"""
    
    if stream_id not in app.state.streaming_orchestrators:
        raise HTTPException(404, f"Stream {stream_id} not found")
    
    orchestrator = app.state.streaming_orchestrators[stream_id]
    metrics = orchestrator.get_metrics()
    
    return {
        "stream_id": stream_id,
        "status": metrics.get("status", "unknown"),
        "platform": metrics.get("platform"),
        "topic": metrics.get("topic"),
        "messages_processed": metrics.get("messages_processed", 0),
        "messages_failed": metrics.get("messages_failed", 0),
        "throughput_per_sec": metrics.get("throughput_per_sec", 0)
    }

@app.delete("/api/v1/streaming/{stream_id}")
async def stop_streaming(
    request: Request,
    stream_id: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Stop streaming pipeline"""
    
    if stream_id not in app.state.streaming_orchestrators:
        raise HTTPException(404, f"Stream {stream_id} not found")
    
    orchestrator = app.state.streaming_orchestrators[stream_id]
    orchestrator.stop()
    
    del app.state.streaming_orchestrators[stream_id]
    
    return {"stream_id": stream_id, "status": "stopped"}

# ============================================================================
# WebSocket for Real-time Updates
# ============================================================================

@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_updates(
    websocket: WebSocket,
    job_id: str
):
    """WebSocket for real-time job updates"""
    await app.state.websocket_manager.connect(websocket, job_id)
    
    try:
        while True:
            # Get job status from scheduler
            job_status = app.state.scheduler.get_job_status(job_id)
            
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
        app.state.websocket_manager.disconnect(websocket, job_id)

@app.websocket("/ws/streaming/{stream_id}")
async def websocket_streaming_updates(
    websocket: WebSocket,
    stream_id: str
):
    """WebSocket for real-time streaming updates"""
    await app.state.websocket_manager.connect(websocket, stream_id)
    
    try:
        while True:
            if stream_id not in app.state.streaming_orchestrators:
                await websocket.send_json({
                    "type": "stream_stopped",
                    "stream_id": stream_id
                })
                break
            
            orchestrator = app.state.streaming_orchestrators[stream_id]
            metrics = orchestrator.get_metrics()
            
            await websocket.send_json({
                "type": "stream_update",
                "stream_id": stream_id,
                "metrics": metrics
            })
            
            # Update Prometheus metrics
            streaming_messages.labels(
                platform=metrics.get("platform", "unknown"),
                topic=metrics.get("topic", "unknown"),
                status="processed"
            ).inc(metrics.get("messages_processed", 0))
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket streaming error: {e}")
    finally:
        app.state.websocket_manager.disconnect(websocket, stream_id)

# ============================================================================
# Configuration Management Endpoints
# ============================================================================

@app.get("/api/v1/config")
async def get_configuration(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get current configuration (filtered by permissions)"""
    
    # Only admins can see full config
    if tenant.get("role") == "admin":
        return app.state.config_manager.export_config(format="json", include_secrets=False)
    
    # Regular users get summary
    return app.state.config_manager.get_config_summary()

@app.post("/api/v1/config/reload")
async def reload_configuration(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Reload configuration from file (admin only)"""
    
    if tenant.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    
    success = app.state.config_manager.reload_config()
    
    if success:
        # Update global config reference
        global config
        config = app.state.config_manager.config
        
        return {"status": "success", "message": "Configuration reloaded"}
    else:
        return {"status": "no_change", "message": "Configuration unchanged"}

@app.get("/api/v1/config/features")
async def get_feature_flags(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get feature flags for current configuration"""
    return app.state.config_manager.get_feature_flags()

@app.get("/api/v1/config/limits")
async def get_limits(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get limits for current plan"""
    return app.state.config_manager.get_limits()

# ============================================================================
# Service Registry Endpoints
# ============================================================================

@app.get("/api/v1/services")
async def list_services(
    request: Request,
    service_type: Optional[str] = None,
    tenant: Dict = Depends(get_current_tenant)
):
    """List all registered services"""
    
    services = app.state.service_registry.list_services(service_type)
    
    # Get detailed info for each service
    service_info = []
    for service_name in services:
        info = app.state.service_registry.get_info(service_name)
        service_info.append({
            "name": service_name,
            "type": info.service_type,
            "status": info.status.value,
            "dependencies": info.dependencies,
            "registered_at": info.registered_at.isoformat()
        })
    
    return {
        "services": service_info,
        "total": len(service_info),
        "statistics": app.state.service_registry.get_statistics()
    }

@app.get("/api/v1/services/{service_name}/status")
async def get_service_status(
    request: Request,
    service_name: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get status of a specific service"""
    
    info = app.state.service_registry.get_info(service_name)
    
    if not info:
        raise HTTPException(404, f"Service {service_name} not found")
    
    return {
        "name": service_name,
        "type": info.service_type,
        "status": info.status.value,
        "dependencies": info.dependencies,
        "dependents": app.state.service_registry._get_dependent_services(service_name),
        "registered_at": info.registered_at.isoformat(),
        "last_health_check": info.last_health_check.isoformat() if info.last_health_check else None
    }

# ============================================================================
# Queue Management Endpoints
# ============================================================================

@app.get("/api/v1/queues/stats")
async def get_queue_statistics(
    request: Request,
    tenant: Dict = Depends(get_current_tenant)
):
    """Get queue statistics from scheduler"""
    
    if not hasattr(app.state.scheduler, 'get_queue_stats'):
        raise HTTPException(501, "Queue statistics not available")
    
    stats = app.state.scheduler.get_queue_stats()
    
    # Filter based on plan
    if tenant["plan"] == PlanType.FREE.value:
        # Limited view for free users
        return {
            "your_jobs": {
                "active": len([j for j in app.state.scheduler.active_jobs.values() 
                             if j.tenant_id == tenant["tenant_id"] and j.status == JobStatus.RUNNING]),
                "queued": len([j for j in app.state.scheduler.active_jobs.values() 
                             if j.tenant_id == tenant["tenant_id"] and j.status == JobStatus.QUEUED])
            }
        }
    
    return stats

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
            sample_input = np.random.randn(1, 10).astype(np.float32)
            
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
