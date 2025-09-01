"""
Authentication and RGPD API Endpoints
======================================
Place in: automl_platform/api/auth_endpoints.py

FastAPI endpoints for SSO authentication and RGPD compliance.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr, Field

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse

# Import services
from ..auth import get_current_user, User, get_db
from ..sso_service import SSOService, SSOProvider
from ..audit_service import AuditService, AuditEventType
from ..rgpd_compliance_service import (
    RGPDComplianceService,
    GDPRRequestType,
    ConsentType,
    get_rgpd_service
)

logger = logging.getLogger(__name__)

# Create routers
sso_router = APIRouter(prefix="/api/sso", tags=["SSO Authentication"])
rgpd_router = APIRouter(prefix="/api/rgpd", tags=["RGPD Compliance"])


# ==================== PYDANTIC MODELS ====================

class SSOLoginRequest(BaseModel):
    """SSO login request"""
    provider: str = Field(..., description="SSO provider (keycloak, auth0, okta)")
    redirect_uri: str = Field(..., description="Redirect URI after authentication")


class ConsentRequest(BaseModel):
    """Consent update request"""
    consent_type: str = Field(..., description="Type of consent")
    granted: bool = Field(..., description="Whether consent is granted")
    purpose: Optional[str] = Field(None, description="Purpose of data processing")
    data_categories: Optional[List[str]] = Field(None, description="Categories of data")
    expires_in_days: Optional[int] = Field(365, description="Consent expiration in days")


class GDPRDataRequest(BaseModel):
    """GDPR data request"""
    request_type: str = Field(..., description="Type of GDPR request")
    reason: Optional[str] = Field(None, description="Reason for request")
    format: Optional[str] = Field("json", description="Response format (json, csv, pdf)")


class RectificationRequest(BaseModel):
    """Data rectification request"""
    corrections: Dict[str, Any] = Field(..., description="Fields to correct with new values")
    reason: Optional[str] = Field(None, description="Reason for rectification")


class DataMappingResponse(BaseModel):
    """Response for data mapping request"""
    data_categories: List[Dict[str, Any]]
    storage_locations: List[str]
    purposes: List[str]
    retention_periods: Dict[str, int]


# ==================== SSO ENDPOINTS ====================

@sso_router.post("/login")
async def sso_login(
    request: SSOLoginRequest,
    background_tasks: BackgroundTasks,
    audit_service: AuditService = Depends(lambda: AuditService())
):
    """
    Initiate SSO login
    """
    try:
        sso_service = SSOService()
        
        # Get authorization URL
        auth_data = await sso_service.get_authorization_url(
            provider=request.provider,
            redirect_uri=request.redirect_uri
        )
        
        # Audit log
        background_tasks.add_task(
            audit_service.log_event,
            event_type=AuditEventType.LOGIN,
            action="sso_login_initiated",
            resource_type="authentication",
            metadata={"provider": request.provider}
        )
        
        return {
            "authorization_url": auth_data["authorization_url"],
            "state": auth_data["state"]
        }
        
    except Exception as e:
        logger.error(f"SSO login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SSO login initialization failed"
        )


@sso_router.get("/callback")
async def sso_callback(
    code: str,
    state: str,
    provider: str = "keycloak",
    background_tasks: BackgroundTasks = None,
    audit_service: AuditService = Depends(lambda: AuditService())
):
    """
    Handle SSO callback
    """
    try:
        sso_service = SSOService()
        
        # Handle callback
        result = await sso_service.handle_callback(
            provider=provider,
            code=code,
            state=state,
            redirect_uri=f"/api/sso/callback?provider={provider}"
        )
        
        # Audit log
        background_tasks.add_task(
            audit_service.log_event,
            event_type=AuditEventType.LOGIN,
            action="sso_login_completed",
            user_id=result["user"]["sub"],
            resource_type="authentication",
            metadata={"provider": provider}
        )
        
        return {
            "session_id": result["session_id"],
            "user": result["user"],
            "access_token": result["tokens"]["access_token"],
            "expires_in": result["tokens"]["expires_in"]
        }
        
    except Exception as e:
        logger.error(f"SSO callback failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SSO authentication failed"
        )


@sso_router.post("/logout")
async def sso_logout(
    session_id: str,
    provider: str = "keycloak",
    redirect_uri: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = None,
    audit_service: AuditService = Depends(lambda: AuditService())
):
    """
    SSO logout
    """
    try:
        sso_service = SSOService()
        
        # Logout from SSO provider
        logout_url = await sso_service.logout(
            provider=provider,
            session_id=session_id,
            redirect_uri=redirect_uri
        )
        
        # Audit log
        background_tasks.add_task(
            audit_service.log_event,
            event_type=AuditEventType.LOGOUT,
            action="sso_logout",
            user_id=str(current_user.id),
            resource_type="authentication"
        )
        
        return {"logout_url": logout_url}
        
    except Exception as e:
        logger.error(f"SSO logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SSO logout failed"
        )


@sso_router.post("/refresh")
async def refresh_sso_token(
    session_id: str,
    provider: str = "keycloak",
    current_user: User = Depends(get_current_user)
):
    """
    Refresh SSO token
    """
    try:
        sso_service = SSOService()
        
        # Get session
        session = sso_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Refresh token
        new_tokens = await sso_service.refresh_token(
            provider=provider,
            refresh_token=session["refresh_token"]
        )
        
        return {
            "access_token": new_tokens["access_token"],
            "expires_in": new_tokens.get("expires_in", 3600)
        }
        
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token refresh failed"
        )


# ==================== RGPD ENDPOINTS ====================

@rgpd_router.post("/consent")
async def update_consent(
    consent: ConsentRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    rgpd_service: RGPDComplianceService = Depends(get_rgpd_service),
    background_tasks: BackgroundTasks = None
):
    """
    Record or update user consent
    """
    try:
        # Record consent
        consent_id = rgpd_service.record_consent(
            user_id=str(current_user.id),
            consent_type=ConsentType[consent.consent_type.upper()],
            granted=consent.granted,
            tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None,
            purpose=consent.purpose,
            data_categories=consent.data_categories,
            expires_in_days=consent.expires_in_days,
            ip_address=request.client.host,
            user_agent=request.headers.get("User-Agent")
        )
        
        return {
            "consent_id": consent_id,
            "consent_type": consent.consent_type,
            "granted": consent.granted,
            "recorded_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Consent update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update consent"
        )


@rgpd_router.get("/consent")
async def get_user_consents(
    current_user: User = Depends(get_current_user),
    rgpd_service: RGPDComplianceService = Depends(get_rgpd_service)
):
    """
    Get all user consents
    """
    try:
        consents = rgpd_service.get_user_consents(str(current_user.id))
        return {"consents": consents}
        
    except Exception as e:
        logger.error(f"Failed to get consents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve consents"
        )


@rgpd_router.post("/request")
async def create_gdpr_request(
    gdpr_request: GDPRDataRequest,
    current_user: User = Depends(get_current_user),
    rgpd_service: RGPDComplianceService = Depends(get_rgpd_service),
    background_tasks: BackgroundTasks = None
):
    """
    Create a GDPR data subject request
    """
    try:
        # Create request
        request_id = rgpd_service.create_data_request(
            user_id=str(current_user.id),
            request_type=GDPRRequestType[gdpr_request.request_type.upper()],
            tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None,
            reason=gdpr_request.reason
        )
        
        # Process immediately for certain request types
        if gdpr_request.request_type.upper() in ["ACCESS", "PORTABILITY"]:
            background_tasks.add_task(
                process_gdpr_request_async,
                request_id,
                rgpd_service
            )
        
        return {
            "request_id": request_id,
            "request_type": gdpr_request.request_type,
            "status": "pending",
            "deadline": (datetime.utcnow() + timedelta(days=30)).isoformat(),
            "message": "Your request has been received and will be processed within 30 days"
        }
        
    except Exception as e:
        logger.error(f"GDPR request creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create GDPR request"
        )


@rgpd_router.get("/request/{request_id}")
async def get_gdpr_request_status(
    request_id: str,
    current_user: User = Depends(get_current_user),
    rgpd_service: RGPDComplianceService = Depends(get_rgpd_service)
):
    """
    Get GDPR request status
    """
    # This would query the request status
    return {
        "request_id": request_id,
        "status": "processing",
        "message": "Your request is being processed"
    }


@rgpd_router.get("/data/export")
async def export_personal_data(
    format: str = "json",
    current_user: User = Depends(get_current_user),
    rgpd_service: RGPDComplianceService = Depends(get_rgpd_service)
):
    """
    Export user's personal data (Article 20 - Data Portability)
    """
    try:
        # Create portability request
        request_id = rgpd_service.create_data_request(
            user_id=str(current_user.id),
            request_type=GDPRRequestType.PORTABILITY,
            tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None
        )
        
        # Process immediately
        data = rgpd_service.process_portability_request(request_id)
        
        # Return as file download
        return Response(
            content=data,
            media_type="application/json" if format == "json" else "text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=personal_data_{current_user.id}.{format}"
            }
        )
        
    except Exception as e:
        logger.error(f"Data export failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export personal data"
        )


@rgpd_router.delete("/data")
async def request_data_erasure(
    reason: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    rgpd_service: RGPDComplianceService = Depends(get_rgpd_service)
):
    """
    Request data erasure (Article 17 - Right to be forgotten)
    """
    try:
        # Create erasure request
        request_id = rgpd_service.create_data_request(
            user_id=str(current_user.id),
            request_type=GDPRRequestType.ERASURE,
            tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None,
            reason=reason
        )
        
        return {
            "request_id": request_id,
            "message": "Your erasure request has been received. We will process it within 30 days after verification.",
            "warning": "This action is irreversible. All your personal data will be permanently deleted or anonymized."
        }
        
    except Exception as e:
        logger.error(f"Erasure request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create erasure request"
        )


@rgpd_router.post("/data/rectify")
async def rectify_personal_data(
    rectification: RectificationRequest,
    current_user: User = Depends(get_current_user),
    rgpd_service: RGPDComplianceService = Depends(get_rgpd_service)
):
    """
    Request data rectification (Article 16)
    """
    try:
        # Create rectification request
        request_id = rgpd_service.create_data_request(
            user_id=str(current_user.id),
            request_type=GDPRRequestType.RECTIFICATION,
            tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None,
            reason=rectification.reason,
            requested_data=rectification.corrections
        )
        
        # Process rectification
        result = rgpd_service.process_rectification_request(
            request_id=request_id,
            corrections=rectification.corrections
        )
        
        return {
            "request_id": request_id,
            "status": result["status"],
            "rectified_fields": list(rectification.corrections.keys()),
            "message": "Your data has been successfully rectified"
        }
        
    except Exception as e:
        logger.error(f"Rectification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rectify data"
        )


@rgpd_router.get("/data/mapping")
async def get_data_mapping(
    current_user: User = Depends(get_current_user),
    rgpd_service: RGPDComplianceService = Depends(get_rgpd_service)
):
    """
    Get mapping of all personal data stored
    """
    try:
        mapping = rgpd_service.get_data_mapping(
            tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None
        )
        
        return DataMappingResponse(
            data_categories=[m for m in mapping],
            storage_locations=list(set(m["location"] for m in mapping)),
            purposes=list(set(m["purpose"] for m in mapping if m["purpose"])),
            retention_periods={m["type"]: m["retention_days"] for m in mapping if m["retention_days"]}
        )
        
    except Exception as e:
        logger.error(f"Failed to get data mapping: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve data mapping"
        )


@rgpd_router.get("/compliance/report")
async def get_compliance_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_user),
    rgpd_service: RGPDComplianceService = Depends(get_rgpd_service)
):
    """
    Get GDPR compliance report (for admins)
    """
    # Check if user is admin
    if not any(role.name == "admin" for role in current_user.roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        report = rgpd_service.generate_compliance_report(
            tenant_id=str(current_user.tenant_id) if current_user.tenant_id else None,
            start_date=start_date,
            end_date=end_date
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate compliance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate compliance report"
        )


# ==================== HELPER FUNCTIONS ====================

async def process_gdpr_request_async(request_id: str, rgpd_service: RGPDComplianceService):
    """
    Process GDPR request asynchronously
    """
    try:
        # This would be handled by a background worker in production
        logger.info(f"Processing GDPR request {request_id} asynchronously")
    except Exception as e:
        logger.error(f"Failed to process GDPR request {request_id}: {e}")


# ==================== COMBINED ROUTER ====================

def create_auth_router() -> APIRouter:
    """
    Create combined authentication router
    """
    router = APIRouter()
    router.include_router(sso_router)
    router.include_router(rgpd_router)
    return router
