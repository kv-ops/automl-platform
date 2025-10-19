"""
Enhanced FastAPI application for AutoML Platform
Production-ready with rate limiting, monitoring, and comprehensive endpoints
Version: 3.2.1 - Full Enterprise Features with SSO, Audit, and RGPD
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks, WebSocket, Request, status, APIRouter
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
import secrets
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

try:
    import ray
except ImportError:  # pragma: no cover - optional dependency
    ray = None  # type: ignore[assignment]

try:
    from dask.distributed import Client as DaskClient
except ImportError:  # pragma: no cover - optional dependency
    DaskClient = None  # type: ignore[assignment]

# AutoML Platform imports
from automl_platform.config import (
    AutoMLConfig,
    load_config,
    validate_secret_value,
)

try:
    from automl_platform.config import InsecureEnvironmentVariableError
except ImportError:  # pragma: no cover - fallback for legacy versions
    class InsecureEnvironmentVariableError(RuntimeError):
        """Raised when an environment variable fails security validation."""
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
from automl_platform.api.billing import BillingManager, UsageTracker, BillingPeriod
from automl_platform.plans import PlanType

# Data connectors
from automl_platform.api.connectors import ConnectorFactory, ConnectionConfig

# Streaming
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
from automl_platform.auth import TokenService, RBACService, QuotaService, AuditService as BasicAuditService, auth_router
from automl_platform.auth import get_current_user, permission_dependency, require_plan

# SSO, Advanced Audit, and RGPD imports
from automl_platform.sso_service import SSOService, SSOProvider
from automl_platform.audit_service import AuditService, AuditEventType, AuditSeverity
from automl_platform.rgpd_compliance_service import RGPDComplianceService, GDPRRequestType, ConsentType

# Scheduler
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
    if ray is None:
        logger.warning(
            "Ray distributed runtime requested but 'ray' is not installed. "
            "Install the 'distributed' extra to enable Ray support."
        )
    else:
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

# New metrics for SSO and RGPD
automl_sso_requests_total = Counter(
    'automl_sso_requests_total',
    'Total SSO authentication requests',
    ['provider', 'status', 'tenant'],
    registry=metrics_registry
)

automl_rgpd_requests_total = Counter(
    'automl_rgpd_requests_total',
    'Total RGPD compliance requests',
    ['request_type', 'status', 'tenant'],
    registry=metrics_registry
)

# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting AutoML API v3.2.1...")
    
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
            credentials_path=getattr(config.storage, 'credentials_path', None),
            models_bucket=getattr(config.storage, 'models_bucket', 'models'),
            datasets_bucket=getattr(config.storage, 'datasets_bucket', 'datasets'),
            artifacts_bucket=getattr(config.storage, 'artifacts_bucket', 'artifacts')
        )
    else:
        app.state.storage = None
    
    # Initialize monitoring
    app.state.monitoring = MonitoringService(app.state.storage) if config.monitoring.enabled else None
    
    # Initialize infrastructure components
    database_cfg = getattr(config, 'database', None)
    security_cfg = getattr(config, 'security', None)

    primary_db_url = getattr(database_cfg, 'url', None) or getattr(config, 'database_url', 'sqlite:///automl.db')
    app.state.tenant_manager = TenantManager(db_url=primary_db_url)

    # Run database migrations
    logger.info("Running database migrations...")
    from alembic import command
    from alembic.config import Config as AlembicConfig
    import os
    alembic_ini_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic.ini")
    alembic_cfg = AlembicConfig(alembic_ini_path)
    command.upgrade(alembic_cfg, "head")
    logger.info("Database migrations completed")

    secret_key = (
        os.getenv("AUTOML_SECRET_KEY")
        or getattr(security_cfg, 'secret_key', None)
        or getattr(config, 'secret_key', None)
    )
    if not secret_key:
        if os.getenv("ENV", "development").lower() == "production":
            raise RuntimeError("AUTOML_SECRET_KEY must be defined for API startup in production.")
        secret_key = secrets.token_urlsafe(32)
        logger.warning(
            "Generated ephemeral AUTOML secret key for development. Set AUTOML_SECRET_KEY for persistence."
        )
    os.environ.setdefault("AUTOML_SECRET_KEY", secret_key)
    try:
        validate_secret_value("AUTOML_SECRET_KEY", secret_key)
    except InsecureEnvironmentVariableError as exc:
        raise RuntimeError(
            "AUTOML_SECRET_KEY uses an insecure default value. Provide a newly generated secret."
        ) from exc
    app.state.security_manager = SecurityManager(secret_key=secret_key)
    app.state.billing_manager = BillingManager()
    app.state.usage_tracker = UsageTracker()
    
    # Initialize deployment manager
    app.state.deployment_manager = DeploymentManager(app.state.tenant_manager)
    
    # Initialize scheduler using SchedulerFactory
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
    app.state.basic_audit_service = BasicAuditService(app.state.tenant_manager.Session())
    
    # Initialize SSO Service
    logger.info("Initializing SSO Service...")
    app.state.sso_service = SSOService(
        redis_client=app.state.billing_manager.redis_client if hasattr(app.state.billing_manager, 'redis_client') else None
    )
    
    # Initialize Advanced Audit Service
    logger.info("Initializing Advanced Audit Service...")
    audit_db_url = (
        getattr(database_cfg, 'audit_url', None)
        or getattr(config, 'audit_database_url', None)
        or primary_db_url
        or 'postgresql://user:pass@localhost/audit'
    )
    audit_encryption_key = getattr(security_cfg, 'audit_encryption_key', None) or getattr(config, 'audit_encryption_key', None)
    app.state.audit_service = AuditService(
        database_url=audit_db_url,
        redis_client=app.state.billing_manager.redis_client if hasattr(app.state.billing_manager, 'redis_client') else None,
        encryption_key=audit_encryption_key
    )
    
    # Initialize RGPD Compliance Service
    logger.info("Initializing RGPD Compliance Service...")
    rgpd_db_url = (
        getattr(database_cfg, 'rgpd_url', None)
        or getattr(config, 'rgpd_database_url', None)
        or primary_db_url
        or 'postgresql://user:pass@localhost/rgpd'
    )
    rgpd_encryption_key = getattr(security_cfg, 'rgpd_encryption_key', None) or getattr(config, 'rgpd_encryption_key', None)
    app.state.rgpd_service = RGPDComplianceService(
        database_url=rgpd_db_url,
        redis_client=app.state.billing_manager.redis_client if hasattr(app.state.billing_manager, 'redis_client') else None,
        audit_service=app.state.audit_service,
        encryption_key=rgpd_encryption_key
    )
    
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
        if DaskClient is None:
            logger.warning(
                "Dask distributed scheduler requested but 'dask.distributed' is not installed. "
                "Install the 'distributed' extra to enable Dask integration."
            )
            app.state.dask_client = None
        else:
            app.state.dask_client = DaskClient(config.distributed.dask_scheduler_address)
    else:
        app.state.dask_client = None
    
    # Initialize orchestrators pool
    app.state.orchestrators = {}
    
    # Initialize WebSocket connections manager
    app.state.websocket_manager = WebSocketManager()
    
    # Initialize cache for model pipelines
    app.state.pipeline_cache = {}
    
    logger.info("AutoML API v3.2.1 started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AutoML API...")
    
    # Flush audit logs
    if app.state.audit_service:
        app.state.audit_service.flush()
    
    # Clean up resources
    if app.state.monitoring:
        app.state.monitoring.save_monitoring_data()
    
    # Close WebSocket connections
    await app.state.websocket_manager.disconnect_all()
    
    # Stop streaming orchestrators
    for orchestrator in app.state.streaming_orchestrators.values():
        orchestrator.stop()
    
    # Shutdown distributed computing
    if hasattr(config, 'distributed') and config.distributed.enabled and ray is not None:
        ray.shutdown()
    
    if app.state.dask_client:
        await app.state.dask_client.close()
    
    # Final audit log
    app.state.audit_service.log_event(
        event_type=AuditEventType.SECURITY_ALERT,
        action="system_shutdown",
        severity=AuditSeverity.INFO,
        description="AutoML API shutdown complete"
    )
    app.state.audit_service.flush()
    
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
    description="Enterprise AutoML platform with MLOps, distributed training, SSO, Audit, and RGPD compliance",
    version="3.2.1",
    docs_url="/docs" if getattr(config.api, 'enable_docs', True) else None,
    redoc_url="/redoc" if getattr(config.api, 'enable_docs', True) else None,
    lifespan=lifespan
)

@app.get("/health")
async def health():
    return {"status": "ok"}

# Include auth router
app.include_router(auth_router)

# ============================================================================
# SSO Router
# ============================================================================

sso_router = APIRouter(prefix="/auth/sso", tags=["authentication"])

class SSOLoginRequest(BaseModel):
    provider: str = Field(..., description="SSO provider (keycloak, auth0, okta, azure_ad)")
    redirect_uri: str = Field(..., description="Redirect URI after authentication")

class SSOCallbackRequest(BaseModel):
    provider: str
    code: str
    state: str
    redirect_uri: str

@sso_router.post("/login")
async def sso_login(
    request: SSOLoginRequest,
    req: Request
):
    """Initiate SSO login"""
    try:
        result = await app.state.sso_service.get_authorization_url(
            provider=request.provider,
            redirect_uri=request.redirect_uri
        )
        
        # Track metrics
        automl_sso_requests_total.labels(
            provider=request.provider,
            status="initiated",
            tenant="unknown"
        ).inc()
        
        return result
    except Exception as e:
        automl_sso_requests_total.labels(
            provider=request.provider,
            status="failed",
            tenant="unknown"
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))

@sso_router.post("/callback")
async def sso_callback(
    request: SSOCallbackRequest,
    req: Request
):
    """Handle SSO callback"""
    try:
        result = await app.state.sso_service.handle_callback(
            provider=request.provider,
            code=request.code,
            state=request.state,
            redirect_uri=request.redirect_uri
        )
        
        # Log audit event
        app.state.audit_service.log_event(
            event_type=AuditEventType.LOGIN,
            action="sso_login_success",
            user_id=result["user"]["sub"],
            metadata={"provider": request.provider},
            ip_address=req.client.host if req.client else None
        )
        
        automl_sso_requests_total.labels(
            provider=request.provider,
            status="success",
            tenant=result["user"].get("tenant_id", "unknown")
        ).inc()
        
        return result
    except Exception as e:
        automl_sso_requests_total.labels(
            provider=request.provider,
            status="failed",
            tenant="unknown"
        ).inc()
        raise HTTPException(status_code=400, detail=str(e))

@sso_router.post("/logout")
async def sso_logout(
    session_id: str,
    provider: str,
    redirect_uri: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Logout from SSO provider"""
    logout_url = await app.state.sso_service.logout(
        provider=provider,
        session_id=session_id,
        redirect_uri=redirect_uri
    )
    
    # Log audit event
    app.state.audit_service.log_event(
        event_type=AuditEventType.LOGOUT,
        action="sso_logout",
        user_id=current_user["user_id"]
    )
    
    return {"logout_url": logout_url}

app.include_router(sso_router)

# ============================================================================
# RGPD Router
# ============================================================================

rgpd_router = APIRouter(prefix="/compliance/rgpd", tags=["compliance"])

class RGPDRequestCreate(BaseModel):
    request_type: str = Field(..., description="Type of GDPR request (access, rectification, erasure, portability, restriction, objection)")
    reason: Optional[str] = Field(None, description="Reason for the request")
    requested_data: Optional[Dict] = Field(None, description="Specific data requested")

class ConsentUpdate(BaseModel):
    consent_type: str = Field(..., description="Type of consent (marketing, analytics, cookies, data_processing, third_party, profiling, automated_decision)")
    granted: bool = Field(..., description="Whether consent is granted")
    purpose: Optional[str] = Field(None, description="Purpose of data processing")
    data_categories: Optional[List[str]] = Field(None, description="Categories of data")
    expires_in_days: Optional[int] = Field(365, description="Consent expiration in days")

@rgpd_router.post(
    "/requests",
    dependencies=[Depends(permission_dependency("rgpd", "create"))],
)
async def create_rgpd_request(
    request: RGPDRequestCreate,
    current_user: Dict = Depends(get_current_user)
):
    """Create a GDPR data subject request"""
    try:
        request_type = GDPRRequestType[request.request_type.upper()]
        
        request_id = app.state.rgpd_service.create_data_request(
            user_id=current_user["user_id"],
            request_type=request_type,
            tenant_id=current_user.get("tenant_id"),
            reason=request.reason,
            requested_data=request.requested_data
        )
        
        # Track metrics
        automl_rgpd_requests_total.labels(
            request_type=request.request_type,
            status="created",
            tenant=current_user.get("tenant_id", "unknown")
        ).inc()
        
        # Log audit event
        app.state.audit_service.log_event(
            event_type=AuditEventType.GDPR_REQUEST,
            action=f"gdpr_request_created",
            user_id=current_user["user_id"],
            tenant_id=current_user.get("tenant_id"),
            resource_type="gdpr_request",
            resource_id=request_id,
            gdpr_relevant=True
        )
        
        return {"request_id": request_id, "status": "pending"}
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid request type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rgpd_router.get("/requests/{request_id}")
async def get_rgpd_request_status(
    request_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get status of a GDPR request"""
    # Implementation would check the request status
    # For now, return a mock response
    return {
        "request_id": request_id,
        "status": "processing",
        "created_at": datetime.utcnow().isoformat(),
        "deadline": (datetime.utcnow() + timedelta(days=30)).isoformat()
    }

@rgpd_router.post(
    "/consent",
    dependencies=[Depends(permission_dependency("rgpd", "consent"))],
)
async def update_consent(
    consent: ConsentUpdate,
    req: Request,
    current_user: Dict = Depends(get_current_user)
):
    """Update user consent"""
    try:
        consent_type = ConsentType[consent.consent_type.upper()]
        
        consent_id = app.state.rgpd_service.record_consent(
            user_id=current_user["user_id"],
            consent_type=consent_type,
            granted=consent.granted,
            tenant_id=current_user.get("tenant_id"),
            purpose=consent.purpose,
            data_categories=consent.data_categories,
            expires_in_days=consent.expires_in_days,
            ip_address=req.client.host if req.client else None,
            user_agent=req.headers.get("user-agent")
        )
        
        # Log audit event
        app.state.audit_service.log_event(
            event_type=AuditEventType.CONSENT_UPDATE,
            action=f"consent_{'granted' if consent.granted else 'revoked'}",
            user_id=current_user["user_id"],
            tenant_id=current_user.get("tenant_id"),
            resource_type="consent",
            resource_id=str(consent_id),
            gdpr_relevant=True
        )
        
        return {"consent_id": str(consent_id), "status": "updated"}
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid consent type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@rgpd_router.get("/consent")
async def get_user_consents(
    current_user: Dict = Depends(get_current_user)
):
    """Get all consents for current user"""
    consents = app.state.rgpd_service.get_user_consents(current_user["user_id"])
    return {"consents": consents}

@rgpd_router.get(
    "/data-mapping",
    dependencies=[Depends(permission_dependency("rgpd", "admin"))],
)
async def get_data_mapping(
    tenant_id: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Get data mapping for personal data (admin only)"""
    if not tenant_id:
        tenant_id = current_user.get("tenant_id")
    
    mapping = app.state.rgpd_service.get_data_mapping(tenant_id)
    return {"data_mapping": mapping}

@rgpd_router.get(
    "/compliance-report",
    dependencies=[Depends(permission_dependency("rgpd", "admin"))],
)
async def get_compliance_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Generate GDPR compliance report (admin only)"""
    report = app.state.rgpd_service.generate_compliance_report(
        tenant_id=current_user.get("tenant_id"),
        start_date=start_date,
        end_date=end_date
    )
    return report

app.include_router(rgpd_router)

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
        allow_origins=["http://localhost:5173"],
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
            "plan_type": PlanType.FREE.value,
            "plan": PlanType.FREE.value,
            "limits": PLAN_LIMITS[PlanType.FREE.value]
        }

    token = credentials.credentials
    try:
        # Verify JWT token using TokenService
        payload = app.state.token_service.verify_token(token)

        # Get tenant information
        payload_tenant_id = payload.get("tenant_id")
        tenant = (
            app.state.tenant_manager.get_tenant(payload_tenant_id)
            if payload_tenant_id
            else None
        )

        if tenant is None:
            plan_type = PlanType.FREE.value
            tenant_identifier = str(payload_tenant_id or "default")
        else:
            plan_type = tenant.plan_type
            tenant_identifier = tenant.id

        return {
            "tenant_id": tenant_identifier,
            "user_id": payload["sub"],
            "plan_type": plan_type,
            "plan": plan_type,
            "limits": PLAN_LIMITS.get(plan_type, PLAN_LIMITS[PlanType.FREE.value])
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
    
    # Log to advanced audit service
    duration = time.time() - start_time
    
    # Determine event type based on path
    event_type = AuditEventType.DATA_READ  # Default
    if request.method == "POST":
        if "/train" in request.url.path:
            event_type = AuditEventType.MODEL_TRAIN
        elif "/predict" in request.url.path:
            event_type = AuditEventType.MODEL_PREDICT
        elif "/export" in request.url.path:
            event_type = AuditEventType.MODEL_EXPORT
        else:
            event_type = AuditEventType.DATA_CREATE
    elif request.method == "DELETE":
        event_type = AuditEventType.DATA_DELETE
    elif request.method == "PUT" or request.method == "PATCH":
        event_type = AuditEventType.DATA_UPDATE
    
    app.state.audit_service.log_event(
        event_type=event_type,
        action=f"{request.method} {request.url.path}",
        user_id=tenant_info["user_id"],
        tenant_id=tenant_info["tenant_id"],
        request_method=request.method,
        request_path=request.url.path,
        response_status=response.status_code,
        response_time_ms=duration * 1000,
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
        "version": "3.2.1",
        "environment": getattr(config, 'environment', 'development'),
        "components": {
            "storage": "healthy" if app.state.storage else "not configured",
            "monitoring": "healthy" if app.state.monitoring else "disabled",
            "mlflow": "healthy" if app.state.mlflow_registry else "disabled",
            "celery": "healthy" if config.worker.enabled else "disabled",
            "ray": "healthy" if hasattr(config, 'distributed') and config.distributed.enabled else "disabled",
            "scheduler": "healthy" if app.state.scheduler else "disabled",
            "streaming": f"{len(app.state.streaming_orchestrators)} active" if app.state.streaming_orchestrators else "none",
            "sso": "healthy" if app.state.sso_service else "disabled",
            "audit": "healthy" if app.state.audit_service else "disabled",
            "rgpd": "healthy" if app.state.rgpd_service else "disabled"
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
# Training Endpoints with Scheduler and Audit
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
    """Start training job with scheduler and audit logging"""
    
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
            detail=(
                f"GPU access not available for {tenant['plan']} plan. "
                "Please upgrade to the Professional tier or higher."
            ),
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
    
    # Log audit event for training start
    app.state.audit_service.log_event(
        event_type=AuditEventType.MODEL_TRAIN,
        action="training_started",
        user_id=tenant["user_id"],
        tenant_id=tenant["tenant_id"],
        resource_type="model",
        resource_id=experiment_id,
        metadata={
            "dataset_id": dataset_id,
            "target_column": target_column,
            "algorithms": train_request.algorithms,
            "use_gpu": train_request.use_gpu
        }
    )
    
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

@app.post("/api/v1/predict")
async def predict(
    request: Request,
    predict_request: PredictRequest,
    tenant: Dict = Depends(get_current_tenant)
):
    """Make prediction with audit logging"""
    
    # Log audit event
    app.state.audit_service.log_event(
        event_type=AuditEventType.MODEL_PREDICT,
        action="prediction_requested",
        user_id=tenant["user_id"],
        tenant_id=tenant["tenant_id"],
        resource_type="model",
        resource_id=predict_request.model_id,
        metadata={"track": predict_request.track}
    )
    
    # Load model
    try:
        pipeline = load_pipeline(predict_request.model_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Model {predict_request.model_id} not found")
    
    # Make prediction
    try:
        prediction = predict(pipeline, predict_request.data)
        
        # Track prediction if requested
        if predict_request.track and app.state.monitoring:
            app.state.monitoring.track_prediction(
                model_id=predict_request.model_id,
                input_data=predict_request.data,
                prediction=prediction,
                timestamp=datetime.utcnow()
            )
        
        return {"prediction": prediction}
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@app.post("/api/v1/export")
async def export_model(
    request: Request,
    export_request: ExportRequest,
    tenant: Dict = Depends(get_current_tenant)
):
    """Export model with audit logging"""
    
    # Log audit event
    app.state.audit_service.log_event(
        event_type=AuditEventType.MODEL_EXPORT,
        action="model_export_requested",
        user_id=tenant["user_id"],
        tenant_id=tenant["tenant_id"],
        resource_type="model",
        resource_id=export_request.model_id,
        metadata={
            "format": export_request.format,
            "optimize_for_edge": export_request.optimize_for_edge,
            "quantize": export_request.quantize
        }
    )
    
    try:
        # Export model
        exported_path = app.state.enhanced_exporter.export(
            model_id=export_request.model_id,
            format=export_request.format,
            optimize_for_edge=export_request.optimize_for_edge,
            quantize=export_request.quantize,
            target_device=export_request.target_device
        )
        
        return FileResponse(
            path=exported_path,
            media_type="application/octet-stream",
            filename=f"{export_request.model_id}.{export_request.format}"
        )
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(500, f"Export failed: {str(e)}")

@app.delete("/api/v1/models/{model_id}")
async def delete_model(
    request: Request,
    model_id: str,
    tenant: Dict = Depends(get_current_tenant)
):
    """Delete model with audit logging"""
    
    # Log audit event
    app.state.audit_service.log_event(
        event_type=AuditEventType.MODEL_DELETE,
        action="model_deleted",
        user_id=tenant["user_id"],
        tenant_id=tenant["tenant_id"],
        resource_type="model",
        resource_id=model_id
    )
    
    try:
        # Delete from MLflow registry
        if app.state.mlflow_registry:
            app.state.mlflow_registry.delete_model(model_id)
        
        # Delete from storage
        if app.state.storage:
            app.state.storage.delete_model(model_id, tenant_id=tenant["tenant_id"])
        
        return {"status": "deleted", "model_id": model_id}
        
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(500, f"Delete failed: {str(e)}")

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
    if tenant["plan"] not in [
        PlanType.PRO.value,
        PlanType.PROFESSIONAL.value,
        PlanType.ENTERPRISE.value,
    ]:
        raise HTTPException(
            403,
            "Streaming requires Professional tier or higher",
        )
    
    # Load model
    try:
        model = app.state.mlflow_registry.load_production_model(model_id)
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
    
    # Start streaming
    orchestrator.start()
    
    # Store orchestrator
    orchestrator_id = f"stream_{uuid.uuid4().hex[:8]}"
    app.state.streaming_orchestrators[orchestrator_id] = orchestrator
    
    return {
        "orchestrator_id": orchestrator_id,
        "status": "running",
        "platform": config.platform,
        "topic": config.topic
    }
