"""
Unified Authentication Middleware
==================================
Place in: automl_platform/auth_middleware.py

Integrates auth.py, sso_service.py, audit_service.py, and rgpd_compliance_service.py
into a unified FastAPI middleware for complete authentication and compliance.
"""

import logging
import time
import json
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis

# Import existing services
from .auth import (
    TokenService, 
    RBACService, 
    QuotaService,
    AuditService as AuthAuditService,
    get_db,
    User
)
from .sso_service import SSOService
from .audit_service import AuditService, AuditEventType, AuditSeverity
from .rgpd_compliance_service import (
    RGPDComplianceService,
    ConsentType,
    GDPRRequestType
)
from .config import AutoMLConfig

logger = logging.getLogger(__name__)


class UnifiedAuthMiddleware(BaseHTTPMiddleware):
    """
    Unified middleware that handles:
    - Authentication (JWT + SSO)
    - Authorization (RBAC)
    - Audit logging
    - RGPD compliance
    - Rate limiting
    - Quota management
    """
    
    def __init__(
        self,
        app,
        config: AutoMLConfig = None,
        redis_client: redis.Redis = None
    ):
        super().__init__(app)
        
        # Load configuration
        self.config = config or AutoMLConfig()
        
        # Initialize Redis
        redis_url = self.config.api.redis_url or os.getenv("REDIS_URL")
        if not redis_url:
            raise RuntimeError(
                "Redis URL is required. Set api.redis_url in the configuration or provide the REDIS_URL environment variable."
            )

        self.redis_client = redis_client or redis.from_url(redis_url)
        
        # Initialize services
        self.token_service = TokenService()
        self.sso_service = SSOService(self.redis_client)
        self.audit_service = AuditService(redis_client=self.redis_client)
        self.rgpd_service = RGPDComplianceService(
            redis_client=self.redis_client,
            audit_service=self.audit_service
        )
        
        # Public endpoints that don't require authentication
        self.public_endpoints = [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/auth/login",
            "/auth/register",
            "/auth/sso/login",
            "/auth/sso/callback",
            "/rgpd/consent",
            "/rgpd/request"
        ]
        
        # Endpoints that require RGPD consent check
        self.consent_required_endpoints = [
            "/api/models/train",
            "/api/predictions",
            "/api/data/upload"
        ]
        
        logger.info("Unified Auth Middleware initialized")
    
    async def dispatch(self, request: Request, call_next):
        """
        Process each request through the middleware pipeline
        """
        start_time = time.time()
        
        # Extract request metadata
        client_ip = request.client.host
        user_agent = request.headers.get("User-Agent", "Unknown")
        request_id = request.headers.get("X-Request-ID", str(time.time()))
        
        # Store request metadata
        request.state.request_id = request_id
        request.state.client_ip = client_ip
        request.state.user_agent = user_agent
        
        try:
            # 1. Check if endpoint is public
            if self._is_public_endpoint(request.url.path):
                response = await call_next(request)
                return response
            
            # 2. Authenticate user
            user = await self._authenticate_user(request)
            if not user:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authentication required"}
                )
            
            # Store user in request state
            request.state.user = user
            request.state.user_id = str(user.id)
            request.state.tenant_id = str(user.tenant_id) if user.tenant_id else None
            
            # 3. Check rate limiting
            if not await self._check_rate_limit(user, request):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Rate limit exceeded"}
                )
            
            # 4. Check quotas
            if not await self._check_quotas(user, request):
                return JSONResponse(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    content={"detail": "Quota exceeded. Please upgrade your plan."}
                )
            
            # 5. Check RBAC permissions
            if not await self._check_permissions(user, request):
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Insufficient permissions"}
                )
            
            # 6. Check RGPD consent if required
            if not await self._check_rgpd_consent(user, request):
                return JSONResponse(
                    status_code=status.HTTP_451_UNAVAILABLE_FOR_LEGAL_REASONS,
                    content={"detail": "Consent required for this operation"}
                )
            
            # 7. Process request
            response = await call_next(request)
            
            # 8. Post-process response
            response = await self._post_process_response(response, user, request)
            
            # 9. Audit log successful request
            process_time = time.time() - start_time
            await self._audit_log_success(
                user, request, response, process_time
            )
            
            return response
            
        except HTTPException as e:
            # Audit log failed request
            await self._audit_log_failure(
                request, e.status_code, e.detail
            )
            raise
            
        except Exception as e:
            # Audit log error
            await self._audit_log_error(request, str(e))
            logger.error(f"Middleware error: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public"""
        for public_path in self.public_endpoints:
            if path.startswith(public_path):
                return True
        return False
    
    async def _authenticate_user(self, request: Request) -> Optional[User]:
        """
        Authenticate user using JWT or SSO
        """
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
            # Check if it's an SSO session
            sso_session = self.sso_service.get_session(token)
            if sso_session:
                # Get user from SSO session
                return await self._get_user_from_sso(sso_session)
            
            # Otherwise, verify JWT token
            try:
                payload = self.token_service.verify_token(token)
                return await self._get_user_by_id(payload["sub"])
            except:
                pass
        
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self._authenticate_api_key(api_key)
        
        return None
    
    async def _get_user_from_sso(self, session: Dict) -> Optional[User]:
        """Get user from SSO session"""
        from .auth import SessionLocal, User as AuthUser
        
        db = SessionLocal()
        try:
            user = db.query(AuthUser).filter_by(
                email=session["email"]
            ).first()
            return user
        finally:
            db.close()
    
    async def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        from .auth import SessionLocal, User as AuthUser
        
        db = SessionLocal()
        try:
            user = db.query(AuthUser).filter_by(
                id=user_id
            ).first()
            return user
        finally:
            db.close()
    
    async def _authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key"""
        from .auth import SessionLocal, APIKey, APIKeyService
        
        db = SessionLocal()
        try:
            service = APIKeyService()
            key_hash = service.hash_api_key(api_key)
            
            api_key_obj = db.query(APIKey).filter_by(
                key_hash=key_hash,
                is_active=True
            ).first()
            
            if api_key_obj:
                # Check expiration
                if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
                    return None
                
                # Update last used
                api_key_obj.last_used_at = datetime.utcnow()
                db.commit()
                
                return api_key_obj.user
            
            return None
        finally:
            db.close()
    
    async def _check_rate_limit(self, user: User, request: Request) -> bool:
        """Check rate limiting"""
        if not self.config.api.enable_rate_limit:
            return True
        
        # Get rate limit based on plan
        rate_limit = self.config.billing.quotas[user.plan_type].get(
            "api_rate_limit", 
            self.config.api.rate_limit_requests
        )
        
        # Check current usage
        key = f"rate_limit:{user.id}:{request.url.path}"
        current = self.redis_client.incr(key)
        
        if current == 1:
            # First request, set expiry
            self.redis_client.expire(key, self.config.api.rate_limit_period)
        
        return current <= rate_limit
    
    async def _check_quotas(self, user: User, request: Request) -> bool:
        """Check user quotas"""
        if not self.config.billing.enabled:
            return True
        
        # Map endpoints to quota types
        quota_mapping = {
            "/api/models/train": "max_models",
            "/api/predictions": "max_predictions_per_day",
            "/api/data/upload": "max_storage_gb"
        }
        
        for pattern, quota_key in quota_mapping.items():
            if request.url.path.startswith(pattern):
                # Check quota
                from .auth import SessionLocal, QuotaService
                
                db = SessionLocal()
                try:
                    quota_service = QuotaService(db, self.redis_client)
                    return quota_service.check_quota(user, quota_key)
                finally:
                    db.close()
        
        return True
    
    async def _check_permissions(self, user: User, request: Request) -> bool:
        """Check RBAC permissions"""
        # Map HTTP methods to actions
        method_to_action = {
            "GET": "read",
            "POST": "create",
            "PUT": "update",
            "PATCH": "update",
            "DELETE": "delete"
        }
        
        action = method_to_action.get(request.method, "read")
        
        # Extract resource from path
        path_parts = request.url.path.strip("/").split("/")
        if len(path_parts) >= 2:
            resource = path_parts[1]  # e.g., "models", "data", etc.
        else:
            resource = "*"
        
        # Check permission
        from .auth import SessionLocal, RBACService
        
        db = SessionLocal()
        try:
            rbac_service = RBACService(db)
            return rbac_service.check_permission(user, resource, action)
        finally:
            db.close()
    
    async def _check_rgpd_consent(self, user: User, request: Request) -> bool:
        """Check RGPD consent for sensitive operations"""
        # Check if endpoint requires consent
        requires_consent = False
        for endpoint in self.consent_required_endpoints:
            if request.url.path.startswith(endpoint):
                requires_consent = True
                break
        
        if not requires_consent:
            return True
        
        # Check consent
        has_consent = self.rgpd_service.check_consent(
            str(user.id),
            ConsentType.DATA_PROCESSING
        )
        
        if not has_consent:
            # Log consent check failure
            self.audit_service.log_event(
                event_type=AuditEventType.UNAUTHORIZED_ACCESS,
                action="consent_required",
                user_id=str(user.id),
                tenant_id=str(user.tenant_id) if user.tenant_id else None,
                resource_type="api",
                resource_id=request.url.path,
                severity=AuditSeverity.WARNING,
                gdpr_relevant=True
            )
        
        return has_consent
    
    async def _post_process_response(
        self,
        response: Response,
        user: User,
        request: Request
    ) -> Response:
        """Post-process response for compliance"""
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Add request ID for tracing
        response.headers["X-Request-ID"] = request.state.request_id
        
        # Check if response contains personal data
        if hasattr(response, "body"):
            try:
                # Parse response body
                body = json.loads(response.body)
                
                # Check for personal data and anonymize if needed
                if self._contains_personal_data(body):
                    # Check if user has consent for data exposure
                    if not self.rgpd_service.check_consent(
                        str(user.id),
                        ConsentType.DATA_PROCESSING
                    ):
                        # Anonymize response
                        body = self.rgpd_service.anonymize_data(body)
                        response.body = json.dumps(body).encode()
            except:
                pass  # Not JSON response
        
        return response
    
    def _contains_personal_data(self, data: Any) -> bool:
        """Check if data contains personal information"""
        if isinstance(data, dict):
            pii_fields = ['email', 'name', 'phone', 'address', 'ssn']
            for field in pii_fields:
                if field in data:
                    return True
            # Recursively check nested data
            for value in data.values():
                if self._contains_personal_data(value):
                    return True
        elif isinstance(data, list):
            for item in data:
                if self._contains_personal_data(item):
                    return True
        return False
    
    async def _audit_log_success(
        self,
        user: User,
        request: Request,
        response: Response,
        process_time: float
    ):
        """Log successful request"""
        self.audit_service.log_event(
            event_type=AuditEventType.DATA_READ,
            action=f"{request.method}_{request.url.path}",
            user_id=str(user.id),
            tenant_id=str(user.tenant_id) if user.tenant_id else None,
            resource_type="api",
            resource_id=request.url.path,
            request_method=request.method,
            request_path=request.url.path,
            response_status=response.status_code,
            response_time_ms=process_time * 1000,
            ip_address=request.state.client_ip,
            user_agent=request.state.user_agent,
            severity=AuditSeverity.INFO
        )
    
    async def _audit_log_failure(
        self,
        request: Request,
        status_code: int,
        detail: str
    ):
        """Log failed request"""
        self.audit_service.log_event(
            event_type=AuditEventType.UNAUTHORIZED_ACCESS,
            action=f"{request.method}_{request.url.path}",
            user_id=getattr(request.state, "user_id", None),
            tenant_id=getattr(request.state, "tenant_id", None),
            resource_type="api",
            resource_id=request.url.path,
            request_method=request.method,
            request_path=request.url.path,
            response_status=status_code,
            ip_address=getattr(request.state, "client_ip", None),
            user_agent=getattr(request.state, "user_agent", None),
            description=detail,
            severity=AuditSeverity.WARNING
        )
    
    async def _audit_log_error(self, request: Request, error: str):
        """Log error"""
        self.audit_service.log_event(
            event_type=AuditEventType.SECURITY_ALERT,
            action=f"{request.method}_{request.url.path}",
            user_id=getattr(request.state, "user_id", None),
            tenant_id=getattr(request.state, "tenant_id", None),
            resource_type="api",
            resource_id=request.url.path,
            request_method=request.method,
            request_path=request.url.path,
            response_status=500,
            ip_address=getattr(request.state, "client_ip", None),
            user_agent=getattr(request.state, "user_agent", None),
            description=error,
            severity=AuditSeverity.ERROR
        )


class RGPDMiddleware(BaseHTTPMiddleware):
    """
    Specific middleware for RGPD compliance
    """
    
    def __init__(self, app, rgpd_service: RGPDComplianceService = None):
        super().__init__(app)
        self.rgpd_service = rgpd_service or RGPDComplianceService()
    
    async def dispatch(self, request: Request, call_next):
        """
        Handle RGPD-specific requirements
        """
        # Check for RGPD headers
        gdpr_request = request.headers.get("X-GDPR-Request")
        
        if gdpr_request:
            # Handle GDPR request
            if gdpr_request == "data-access":
                return await self._handle_data_access(request)
            elif gdpr_request == "data-portability":
                return await self._handle_data_portability(request)
            elif gdpr_request == "data-erasure":
                return await self._handle_data_erasure(request)
        
        # Check for consent updates
        consent_header = request.headers.get("X-Consent-Update")
        if consent_header:
            await self._handle_consent_update(request, consent_header)
        
        # Process request
        response = await call_next(request)
        
        # Add RGPD compliance headers
        response.headers["X-GDPR-Compliant"] = "true"
        response.headers["X-Data-Controller"] = "AutoML Platform"
        response.headers["X-Privacy-Policy"] = "/privacy"
        
        return response
    
    async def _handle_data_access(self, request: Request) -> JSONResponse:
        """Handle data access request"""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authentication required"}
            )
        
        # Create GDPR request
        request_id = self.rgpd_service.create_data_request(
            user_id=user_id,
            request_type=GDPRRequestType.ACCESS
        )
        
        # Process immediately
        data = self.rgpd_service.process_access_request(request_id)
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "request_id": request_id,
                "data": data
            }
        )
    
    async def _handle_data_portability(self, request: Request) -> Response:
        """Handle data portability request"""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authentication required"}
            )
        
        # Create GDPR request
        request_id = self.rgpd_service.create_data_request(
            user_id=user_id,
            request_type=GDPRRequestType.PORTABILITY
        )
        
        # Process and return data
        data = self.rgpd_service.process_portability_request(request_id)
        
        return Response(
            content=data,
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=personal_data_{user_id}.json"
            }
        )
    
    async def _handle_data_erasure(self, request: Request) -> JSONResponse:
        """Handle data erasure request"""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authentication required"}
            )
        
        # Create GDPR request
        request_id = self.rgpd_service.create_data_request(
            user_id=user_id,
            request_type=GDPRRequestType.ERASURE
        )
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "request_id": request_id,
                "message": "Erasure request received and will be processed within 30 days"
            }
        )
    
    async def _handle_consent_update(self, request: Request, consent_data: str):
        """Handle consent update"""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            return
        
        try:
            # Parse consent data
            consents = json.loads(consent_data)
            
            for consent_type, granted in consents.items():
                self.rgpd_service.record_consent(
                    user_id=user_id,
                    consent_type=ConsentType[consent_type.upper()],
                    granted=granted,
                    ip_address=getattr(request.state, "client_ip", None),
                    user_agent=getattr(request.state, "user_agent", None)
                )
        except Exception as e:
            logger.error(f"Failed to update consent: {e}")


def setup_auth_middleware(app, config: AutoMLConfig = None):
    """
    Setup authentication middleware for FastAPI app
    """
    # Add unified auth middleware
    app.add_middleware(UnifiedAuthMiddleware, config=config)
    
    # Add RGPD middleware
    app.add_middleware(RGPDMiddleware)
    
    logger.info("Authentication middleware configured")
