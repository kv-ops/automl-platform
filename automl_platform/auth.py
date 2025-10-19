"""
Enterprise Authentication and Authorization Service
====================================================

Based on best practices from H2O AI Cloud (Keycloak/OpenID Connect) and DataRobot (custom roles).
Implements JWT tokens, SSO support, RBAC, API keys, and multi-tenant isolation.

Author: MLOps Platform Team
Date: 2024
"""

import os
import jwt
import json
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
import asyncio
from functools import wraps
import uuid

import redis
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Security, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy import Column, String, DateTime, Boolean, Integer, JSON, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
import httpx
from prometheus_client import Counter, Histogram
import time

from automl_platform.database import get_app_engine, get_app_sessionmaker

# Import from automl_platform modules
try:
    from automl_platform.config import Config as MLOpsConfig
    from automl_platform.storage import StorageService
    from automl_platform.monitoring import MonitoringService
except ImportError:
    # Fallback if modules not yet created
    MLOpsConfig = None
    StorageService = None
    MonitoringService = None

from automl_platform.plans import PlanType, normalize_plan_type, plan_level

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class AuthConfig:
    """Authentication configuration aligned with enterprise standards"""
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRATION_HOURS = 24
    JWT_REFRESH_EXPIRATION_DAYS = 30
    
    # OAuth/SSO Configuration (like H2O AI Cloud with Keycloak)
    OAUTH_ENABLED = os.getenv("OAUTH_ENABLED", "false").lower() == "true"
    KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080")
    KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "mlops")
    KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "mlops-platform")
    KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "")
    
    # SAML Configuration
    SAML_ENABLED = os.getenv("SAML_ENABLED", "false").lower() == "true"
    SAML_IDP_URL = os.getenv("SAML_IDP_URL", "")
    SAML_CERT_FILE = os.getenv("SAML_CERT_FILE", "")
    
    # API Key Configuration
    API_KEY_HEADER = "X-API-Key"
    API_KEY_LENGTH = 32
    
    # Redis for session management
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    SESSION_TTL = 3600  # 1 hour
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/mlops")
    
    # Rate limiting (like DataRobot's worker limits)
    RATE_LIMIT_ENABLED = True
    DEFAULT_RATE_LIMIT = 100  # requests per minute
    
    # Multi-tenant configuration
    MULTI_TENANT_ENABLED = os.getenv("MULTI_TENANT_ENABLED", "true").lower() == "true"
    
    # Audit logging
    AUDIT_LOG_ENABLED = True
    AUDIT_LOG_TABLE = "audit_logs"


# ============================================================================
# Database Setup
# ============================================================================

# Create database engine
_auth_db_override = AuthConfig.DATABASE_URL if "DATABASE_URL" in os.environ else None
engine = get_app_engine(_auth_db_override)
SessionLocal = get_app_sessionmaker(_auth_db_override)
Base = declarative_base()

# ============================================================================
# Database Models (PostgreSQL)
# ============================================================================

# Association tables for many-to-many relationships
user_roles = Table('user_roles', Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id')),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id'))
)

role_permissions = Table('role_permissions', Base.metadata,
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id')),
    Column('permission_id', UUID(as_uuid=True), ForeignKey('permissions.id'))
)

project_users = Table('project_users', Base.metadata,
    Column('project_id', UUID(as_uuid=True), ForeignKey('projects.id')),
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id')),
    Column('role', String(50))  # project-specific role
)


class User(Base):
    """User model with multi-tenant support"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255))
    
    # Multi-tenant fields
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenants.id'), nullable=True)
    organization = Column(String(255))
    
    # Plan and quotas (like DataRobot's worker limits)
    plan_type = Column(String(50), default=PlanType.FREE.value)
    max_workers = Column(Integer, default=1)  # DataRobot: 4 workers for trial
    max_concurrent_jobs = Column(Integer, default=2)  # H2O: 2 jobs per node
    storage_quota_gb = Column(Integer, default=10)
    monthly_compute_minutes = Column(Integer, default=1000)
    
    # Status and metadata
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # SSO fields
    sso_provider = Column(String(50))  # keycloak, saml, google, etc.
    sso_id = Column(String(255))
    
    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    projects = relationship("Project", secondary=project_users, back_populates="users")
    audit_logs = relationship("AuditLog", back_populates="user")


from automl_platform.models.tenant import Tenant


class Role(Base):
    """Role model for RBAC (like DataRobot's custom roles)"""
    __tablename__ = "roles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(String(500))
    is_system = Column(Boolean, default=False)  # System roles can't be deleted
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenants.id'))
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")


class Permission(Base):
    """Permission model for fine-grained access control"""
    __tablename__ = "permissions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    resource = Column(String(100), nullable=False)  # datasets, models, projects, etc.
    action = Column(String(50), nullable=False)  # create, read, update, delete, execute
    scope = Column(String(50))  # own, team, all
    
    # Relationships
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")


class Project(Base):
    """Project model for workspace isolation"""
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenants.id'), nullable=True)
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Access control
    is_public = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    users = relationship("User", secondary=project_users, back_populates="projects")
    owner = relationship("User", foreign_keys=[owner_id])


class APIKey(Base):
    """API Key model for machine authentication"""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(255), unique=True, nullable=False)
    name = Column(String(100))
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Scopes and permissions
    scopes = Column(JSON)  # ["read:models", "write:datasets", etc.]
    
    # Rate limiting
    rate_limit = Column(Integer, default=1000)  # requests per hour
    
    # Expiration
    expires_at = Column(DateTime)
    last_used_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")


class AuditLog(Base):
    """Audit log for compliance and security"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenants.id'), nullable=True)
    
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(255))
    
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    request_data = Column(JSON)
    response_status = Column(Integer)
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")


# ============================================================================
# Security Services
# ============================================================================

class PasswordService:
    """Password hashing and verification"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """Validate password meets security requirements"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False, "Password must contain at least one special character"
        return True, "Password is valid"


class TokenService:
    """JWT token management"""
    
    def __init__(self):
        self.secret_key = AuthConfig.JWT_SECRET_KEY
        self.algorithm = AuthConfig.JWT_ALGORITHM
        self.redis_client = redis.from_url(AuthConfig.REDIS_URL)
    
    def create_access_token(
        self,
        user_id: str,
        tenant_id: Optional[str],
        roles: List[str],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=AuthConfig.JWT_EXPIRATION_HOURS)
        
        payload = {
            "sub": user_id,
            "tenant_id": tenant_id,
            "roles": roles,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())  # JWT ID for revocation
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Store in Redis for session management
        self.redis_client.setex(
            f"token:{payload['jti']}",
            AuthConfig.SESSION_TTL,
            json.dumps({"user_id": user_id, "tenant_id": tenant_id})
        )
        
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a refresh token"""
        expire = datetime.utcnow() + timedelta(days=AuthConfig.JWT_REFRESH_EXPIRATION_DAYS)
        payload = {
            "sub": user_id,
            "type": "refresh",
            "exp": expire,
            "jti": str(uuid.uuid4())
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti and not self.redis_client.exists(f"token:{jti}"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def revoke_token(self, token: str):
        """Revoke a token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            jti = payload.get("jti")
            if jti:
                self.redis_client.delete(f"token:{jti}")
        except jwt.JWTError:
            pass


class APIKeyService:
    """API Key management for machine authentication"""
    
    def generate_api_key(self) -> str:
        """Generate a new API key"""
        return f"mlops_{secrets.token_urlsafe(AuthConfig.API_KEY_LENGTH)}"
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key(self, api_key: str, key_hash: str) -> bool:
        """Verify an API key against its hash"""
        return self.hash_api_key(api_key) == key_hash


# ============================================================================
# OAuth/SSO Services (H2O AI Cloud style with Keycloak)
# ============================================================================

class OAuthService:
    """OAuth/OpenID Connect service for SSO"""
    
    def __init__(self):
        if AuthConfig.OAUTH_ENABLED:
            config = Config(environ={
                "KEYCLOAK_CLIENT_ID": AuthConfig.KEYCLOAK_CLIENT_ID,
                "KEYCLOAK_CLIENT_SECRET": AuthConfig.KEYCLOAK_CLIENT_SECRET
            })
            self.oauth = OAuth(config)
            
            # Register Keycloak provider
            self.oauth.register(
                name='keycloak',
                server_metadata_url=f'{AuthConfig.KEYCLOAK_URL}/realms/{AuthConfig.KEYCLOAK_REALM}/.well-known/openid-configuration',
                client_kwargs={'scope': 'openid profile email'}
            )
    
    async def get_authorization_url(self, redirect_uri: str) -> str:
        """Get OAuth authorization URL"""
        if not AuthConfig.OAUTH_ENABLED:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="OAuth is not enabled"
            )
        
        client = self.oauth.create_client('keycloak')
        return await client.create_authorization_url(redirect_uri)
    
    async def handle_callback(self, code: str, redirect_uri: str) -> Dict:
        """Handle OAuth callback and exchange code for tokens"""
        if not AuthConfig.OAUTH_ENABLED:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="OAuth is not enabled"
            )
        
        client = self.oauth.create_client('keycloak')
        token = await client.fetch_token(code=code, redirect_uri=redirect_uri)
        
        # Get user info
        resp = await client.get('userinfo', token=token)
        user_info = resp.json()
        
        return {
            "access_token": token['access_token'],
            "refresh_token": token.get('refresh_token'),
            "user_info": user_info
        }


# ============================================================================
# RBAC Service (DataRobot-style custom roles)
# ============================================================================

class RBACService:
    """Role-Based Access Control service"""
    
    def __init__(self, db: Session):
        self.db = db
        self._init_system_roles()
    
    def _init_system_roles(self):
        """Initialize system roles if they don't exist"""
        system_roles = [
            {
                "name": "admin",
                "description": "Full system access",
                "permissions": ["*:*:all"]
            },
            {
                "name": "data_scientist",
                "description": "Can create and manage ML projects",
                "permissions": [
                    "projects:*:own",
                    "datasets:*:team",
                    "models:*:team",
                    "predictions:read:all"
                ]
            },
            {
                "name": "viewer",
                "description": "Read-only access",
                "permissions": ["*:read:all"]
            },
            {
                "name": "trial",
                "description": "Limited trial access (DataRobot: 4 workers)",
                "permissions": [
                    "projects:create:own",
                    "datasets:create:own",
                    "models:create:own"
                ]
            }
        ]
        
        for role_data in system_roles:
            if not self.db.query(Role).filter_by(name=role_data["name"]).first():
                role = Role(
                    name=role_data["name"],
                    description=role_data["description"],
                    is_system=True
                )
                self.db.add(role)
        
        try:
            self.db.commit()
        except:
            self.db.rollback()
    
    def check_permission(
        self,
        user: User,
        resource: str,
        action: str,
        resource_owner_id: Optional[str] = None
    ) -> bool:
        """Check if user has permission for an action"""
        
        # Admins have full access
        if any(role.name == "admin" for role in user.roles):
            return True
        
        for role in user.roles:
            for permission in role.permissions:
                # Check exact match
                if permission.resource == resource and permission.action == action:
                    # Check scope
                    if permission.scope == "all":
                        return True
                    elif permission.scope == "team":
                        # Check if resource belongs to same tenant
                        return True  # Simplified - implement team logic
                    elif permission.scope == "own":
                        return str(resource_owner_id) == str(user.id)
                
                # Check wildcard permissions
                if permission.resource == "*" and permission.action == action:
                    return True
                if permission.resource == resource and permission.action == "*":
                    return True
                if permission.resource == "*" and permission.action == "*":
                    return True
        
        return False
    
    def get_user_permissions(self, user: User) -> List[str]:
        """Get all permissions for a user"""
        permissions = []
        for role in user.roles:
            for permission in role.permissions:
                permissions.append(
                    f"{permission.resource}:{permission.action}:{permission.scope}"
                )
        return permissions


# ============================================================================
# Quota and Metering Service
# ============================================================================

class QuotaService:
    """Service for managing user quotas and usage metering"""
    
    def __init__(self, db: Session, redis_client: redis.Redis):
        self.db = db
        self.redis = redis_client
    
    def check_quota(self, user: User, resource_type: str, amount: int = 1) -> bool:
        """Check if user has quota for a resource"""
        
        # Get current usage from Redis (for real-time tracking)
        usage_key = f"usage:{user.id}:{resource_type}"
        current_usage = int(self.redis.get(usage_key) or 0)
        
        # Check against limits based on plan
        limits = self._get_plan_limits(user.plan_type)
        limit = limits.get(resource_type, 0)
        
        return (current_usage + amount) <= limit
    
    def consume_quota(self, user: User, resource_type: str, amount: int = 1):
        """Consume quota for a resource"""
        if not self.check_quota(user, resource_type, amount):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Quota exceeded for {resource_type}"
            )
        
        # Increment usage
        usage_key = f"usage:{user.id}:{resource_type}"
        self.redis.incrby(usage_key, amount)
        
        # Set TTL to reset monthly
        ttl = self.redis.ttl(usage_key)
        if ttl == -1:  # No TTL set
            days_until_month_end = 30  # Simplified
            self.redis.expire(usage_key, days_until_month_end * 86400)
    
    def _get_plan_limits(self, plan_type: str) -> Dict[str, int]:
        """Get resource limits for a plan"""
        limits = {
            PlanType.FREE: {
                "workers": 1,
                "concurrent_jobs": 1,
                "storage_gb": 10,
                "compute_minutes": 100,
                "api_calls": 1000,
            },
            PlanType.TRIAL: {
                "workers": 4,  # DataRobot trial limit
                "concurrent_jobs": 2,  # H2O: 2 jobs per node
                "storage_gb": 50,
                "compute_minutes": 1000,
                "api_calls": 10000,
            },
            PlanType.STARTER: {
                "workers": 3,
                "concurrent_jobs": 3,
                "storage_gb": 25,
                "compute_minutes": 5000,
                "api_calls": 30000,
            },
            PlanType.PRO: {
                "workers": 4,  # Legacy Pro plan
                "concurrent_jobs": 4,
                "storage_gb": 100,
                "compute_minutes": 10000,
                "api_calls": 100000,
            },
            PlanType.PROFESSIONAL: {
                "workers": 8,
                "concurrent_jobs": 10,
                "storage_gb": 250,
                "compute_minutes": 50000,
                "api_calls": 500000,
            },
            PlanType.ENTERPRISE: {
                "workers": 999999,  # Unlimited
                "concurrent_jobs": 999999,
                "storage_gb": 999999,
                "compute_minutes": 999999,
                "api_calls": 999999,
            },
            PlanType.CUSTOM: {
                "workers": 999999,
                "concurrent_jobs": 999999,
                "storage_gb": 999999,
                "compute_minutes": 999999,
                "api_calls": 999999,
            },
        }
        resolved_plan = normalize_plan_type(plan_type, default=PlanType.FREE)
        return limits.get(resolved_plan, limits[PlanType.FREE])


# ============================================================================
# Audit Service
# ============================================================================

class AuditService:
    """Service for audit logging and compliance"""
    
    def __init__(self, db: Session):
        self.db = db
        
        # Metrics for monitoring
        self.auth_attempts = Counter('auth_attempts_total', 'Total authentication attempts')
        self.auth_failures = Counter('auth_failures_total', 'Total authentication failures')
        self.api_calls = Counter('api_calls_total', 'Total API calls', ['endpoint', 'method'])
        self.api_latency = Histogram('api_latency_seconds', 'API latency', ['endpoint'])
    
    def log_action(
        self,
        user_id: Optional[str],
        tenant_id: Optional[str],
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        request_data: Optional[Dict] = None,
        response_status: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Log an action for audit trail"""
        if not AuthConfig.AUDIT_LOG_ENABLED:
            return
        
        audit_log = AuditLog(
            user_id=uuid.UUID(user_id) if user_id else None,
            tenant_id=uuid.UUID(tenant_id) if tenant_id else None,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            request_data=request_data,
            response_status=response_status,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.add(audit_log)
        try:
            self.db.commit()
        except:
            self.db.rollback()
        
        # Update metrics
        if action == "login":
            self.auth_attempts.inc()
            if response_status != 200:
                self.auth_failures.inc()
    
    def get_audit_logs(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Retrieve audit logs with filtering"""
        query = self.db.query(AuditLog).filter_by(tenant_id=tenant_id)
        
        if user_id:
            query = query.filter_by(user_id=user_id)
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()


# ============================================================================
# Authentication Dependencies for FastAPI
# ============================================================================

security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
api_key_header = APIKeyHeader(name=AuthConfig.API_KEY_HEADER, auto_error=False)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db),
    api_key: Optional[str] = Security(api_key_header)
) -> User:
    """Dependency to get current authenticated user"""
    
    # Try API key first (for machine authentication)
    if api_key:
        api_key_service = APIKeyService()
        key_hash = api_key_service.hash_api_key(api_key)
        
        api_key_obj = db.query(APIKey).filter_by(
            key_hash=key_hash,
            is_active=True
        ).first()
        
        if not api_key_obj:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Check expiration
        if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key has expired"
            )
        
        # Update last used
        api_key_obj.last_used_at = datetime.utcnow()
        db.commit()
        
        return api_key_obj.user
    
    # Try JWT token
    if credentials:
        token_service = TokenService()
        payload = token_service.verify_token(credentials.credentials)
        
        user = db.query(User).filter_by(
            id=payload["sub"],
            is_active=True
        ).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated"
    )


def require_permission(resource: str, action: str):
    """Decorator to check permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from request context
            request = kwargs.get('request')
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request context not found"
                )
            
            user = request.state.user
            db = request.state.db
            
            rbac_service = RBACService(db)
            
            # Check permission
            resource_owner_id = kwargs.get('owner_id')
            if not rbac_service.check_permission(user, resource, action, resource_owner_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied for {action} on {resource}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def permission_dependency(resource: str, action: str):
    """Create a FastAPI dependency that enforces RBAC permissions."""

    async def dependency(
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> User:
        # Ensure state is populated for downstream handlers expecting it
        if not getattr(request.state, "user", None):
            request.state.user = current_user
        if not getattr(request.state, "db", None):
            request.state.db = db

        rbac_service = RBACService(db)
        resource_owner_id = request.path_params.get("owner_id")

        if not rbac_service.check_permission(
            current_user, resource, action, resource_owner_id
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied for {action} on {resource}",
            )

        return current_user

    return dependency


def require_plan(min_plan: PlanType):
    """Decorator to check user plan"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request context not found"
                )
            
            user = request.state.user
            
            required_plan = normalize_plan_type(min_plan, default=PlanType.FREE)
            user_plan_level = plan_level(getattr(user, "plan_type", None))
            required_plan_level = plan_level(required_plan)

            if user_plan_level < required_plan_level:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        f"This feature requires the {required_plan.value} plan or higher"
                    )
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Rate Limiting Middleware
# ============================================================================

class RateLimiter:
    """Rate limiting middleware (like DataRobot's worker limits)"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def check_rate_limit(self, user: User, endpoint: str) -> bool:
        """Check if user has exceeded rate limit"""
        if not AuthConfig.RATE_LIMIT_ENABLED:
            return True
        
        # Get user's rate limit based on plan
        rate_limit = self._get_rate_limit(user.plan_type)
        
        # Check current usage
        key = f"rate_limit:{user.id}:{endpoint}"
        current = self.redis.incr(key)
        
        if current == 1:
            # First request, set expiry
            self.redis.expire(key, 60)  # 1 minute window
        
        return current <= rate_limit
    
    def _get_rate_limit(self, plan_type: str) -> int:
        """Get rate limit for plan type"""
        limits = {
            PlanType.FREE: 10,  # 10 requests per minute
            PlanType.TRIAL: 60,  # 60 requests per minute
            PlanType.STARTER: 250,
            PlanType.PRO: 300,  # 300 requests per minute for legacy Pro
            PlanType.PROFESSIONAL: 500,
            PlanType.ENTERPRISE: 9999,  # Effectively unlimited
            PlanType.CUSTOM: 9999,
        }
        resolved_plan = normalize_plan_type(plan_type, default=None)
        if resolved_plan is None:
            return AuthConfig.DEFAULT_RATE_LIMIT

        return limits.get(resolved_plan, AuthConfig.DEFAULT_RATE_LIMIT)


# ============================================================================
# Authentication API Endpoints
# ============================================================================

from fastapi import APIRouter

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post("/register")
async def register(
    email: EmailStr,
    username: str,
    password: str,
    organization: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    # Check if user exists
    if db.query(User).filter_by(email=email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Validate password
    pwd_service = PasswordService()
    valid, message = pwd_service.validate_password_strength(password)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    # Create tenant if multi-tenant enabled
    tenant_id = None
    tenant = None
    if AuthConfig.MULTI_TENANT_ENABLED:
        tenant = Tenant(
            name=organization or username,
            subdomain=username.lower(),
            plan_type=PlanType.TRIAL.value,
            trial_ends_at=datetime.utcnow() + timedelta(days=14),
            k8s_namespace=f"mlops-{username.lower()}",
            minio_bucket=f"mlops-{username.lower()}",
            database_schema=f"tenant_{username.lower()}"
        )
        db.add(tenant)
        db.flush()
        tenant_id = tenant.id
    
    # Create user
    user = User(
        email=email,
        username=username,
        password_hash=pwd_service.hash_password(password),
        tenant_id=tenant_id,
        organization=organization,
        plan_type=PlanType.TRIAL.value,
        max_workers=4,  # DataRobot trial limit
        max_concurrent_jobs=2  # H2O: 2 jobs per node
    )
    
    db.add(user)
    
    # Assign default role
    trial_role = db.query(Role).filter_by(name="trial").first()
    if trial_role:
        user.roles.append(trial_role)
    
    db.commit()
    
    # Create tokens
    token_service = TokenService()
    access_token = token_service.create_access_token(
        str(user.id),
        str(tenant_id) if tenant_id else None,
        ["trial"]
    )
    refresh_token = token_service.create_refresh_token(str(user.id))
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user_id": str(user.id),
        "trial_ends_at": tenant.trial_ends_at if tenant else None
    }


@auth_router.post("/login")
async def login(
    username: str,
    password: str,
    db: Session = Depends(get_db),
    request: Request = None
):
    """Login with username/password"""
    # Find user
    user = db.query(User).filter(
        (User.username == username) | (User.email == username)
    ).first()
    
    if not user:
        # Log failed attempt
        audit_service = AuditService(db)
        audit_service.log_action(
            user_id=None,
            tenant_id=None,
            action="login",
            response_status=401,
            ip_address=request.client.host if request else None
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Verify password
    pwd_service = PasswordService()
    if not pwd_service.verify_password(password, user.password_hash):
        # Log failed attempt
        audit_service = AuditService(db)
        audit_service.log_action(
            user_id=str(user.id),
            tenant_id=str(user.tenant_id) if user.tenant_id else None,
            action="login",
            response_status=401,
            ip_address=request.client.host if request else None
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Log successful login
    audit_service = AuditService(db)
    audit_service.log_action(
        user_id=str(user.id),
        tenant_id=str(user.tenant_id) if user.tenant_id else None,
        action="login",
        response_status=200,
        ip_address=request.client.host if request else None
    )
    
    # Create tokens
    token_service = TokenService()
    roles = [role.name for role in user.roles]
    access_token = token_service.create_access_token(
        str(user.id),
        str(user.tenant_id) if user.tenant_id else None,
        roles
    )
    refresh_token = token_service.create_refresh_token(str(user.id))
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "username": user.username,
            "email": user.email,
            "roles": roles,
            "plan_type": user.plan_type
        }
    }


@auth_router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """Logout and revoke token"""
    token_service = TokenService()
    token_service.revoke_token(credentials.credentials)
    
    return {"message": "Successfully logged out"}


@auth_router.post("/refresh")
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """Refresh access token"""
    token_service = TokenService()
    
    # Verify refresh token
    try:
        payload = token_service.verify_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Get user
    user = db.query(User).filter_by(id=payload["sub"]).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Create new access token
    roles = [role.name for role in user.roles]
    access_token = token_service.create_access_token(
        str(user.id),
        str(user.tenant_id) if user.tenant_id else None,
        roles
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@auth_router.get("/sso/login")
async def sso_login(provider: str = "keycloak", redirect_uri: str = None):
    """Initiate SSO login"""
    if not AuthConfig.OAUTH_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="SSO is not enabled"
        )
    
    oauth_service = OAuthService()
    auth_url = await oauth_service.get_authorization_url(redirect_uri)
    
    return {"auth_url": auth_url}


@auth_router.get("/sso/callback")
async def sso_callback(
    code: str,
    state: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Handle SSO callback"""
    oauth_service = OAuthService()
    redirect_uri = "http://localhost:8000/auth/sso/callback"  # Should come from config
    result = await oauth_service.handle_callback(code, redirect_uri)
    
    user_info = result["user_info"]
    
    # Find or create user
    user = db.query(User).filter_by(sso_id=user_info["sub"]).first()
    
    if not user:
        # Create new user from SSO
        user = User(
            email=user_info.get("email"),
            username=user_info.get("preferred_username", user_info["sub"]),
            sso_provider="keycloak",
            sso_id=user_info["sub"],
            is_verified=True,
            plan_type=PlanType.TRIAL.value
        )
        db.add(user)
        db.commit()
    
    # Create tokens
    token_service = TokenService()
    roles = [role.name for role in user.roles]
    access_token = token_service.create_access_token(
        str(user.id),
        str(user.tenant_id) if user.tenant_id else None,
        roles
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "username": user.username,
            "email": user.email
        }
    }


@auth_router.post("/api-keys")
async def create_api_key(
    name: str,
    scopes: List[str],
    expires_in_days: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new API key"""
    api_key_service = APIKeyService()
    
    # Generate key
    api_key = api_key_service.generate_api_key()
    key_hash = api_key_service.hash_api_key(api_key)
    
    # Calculate expiration
    expires_at = None
    if expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
    
    # Create API key object
    api_key_obj = APIKey(
        key_hash=key_hash,
        name=name,
        user_id=current_user.id,
        scopes=scopes,
        expires_at=expires_at
    )
    
    db.add(api_key_obj)
    db.commit()
    
    return {
        "api_key": api_key,
        "key_id": str(api_key_obj.id),
        "expires_at": expires_at
    }


@auth_router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Revoke an API key"""
    api_key = db.query(APIKey).filter_by(
        id=key_id,
        user_id=current_user.id
    ).first()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    api_key.is_active = False
    db.commit()
    
    return {"message": "API key revoked"}


@auth_router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user information"""
    rbac_service = RBACService(db)
    permissions = rbac_service.get_user_permissions(current_user)
    
    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "organization": current_user.organization,
        "plan_type": current_user.plan_type,
        "roles": [role.name for role in current_user.roles],
        "permissions": permissions,
        "quotas": {
            "max_workers": current_user.max_workers,
            "max_concurrent_jobs": current_user.max_concurrent_jobs,
            "storage_quota_gb": current_user.storage_quota_gb,
            "monthly_compute_minutes": current_user.monthly_compute_minutes
        }
    }


# ============================================================================
# Helper Functions
# ============================================================================

def init_auth_system():
    """Initialize authentication system"""
    logger.info("Initializing authentication system...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Initialize services
    pwd_service = PasswordService()
    token_service = TokenService()
    api_key_service = APIKeyService()
    
    # Initialize system roles
    db = SessionLocal()
    rbac_service = RBACService(db)
    db.close()
    
    logger.info("Authentication system initialized successfully")
    
    return {
        "password_service": pwd_service,
        "token_service": token_service,
        "api_key_service": api_key_service
    }


if __name__ == "__main__":
    init_auth_system()
