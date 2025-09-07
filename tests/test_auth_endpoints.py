"""
Tests for Authentication API Endpoints
========================================
Tests for SSO authentication and RGPD compliance FastAPI endpoints.
"""

import pytest
import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from httpx import AsyncClient
import jwt

# Import auth components and endpoints
try:
    from automl_platform.api.auth_endpoints import (
        sso_router,
        rgpd_router,
        create_auth_router,
        SSOLoginRequest,
        ConsentRequest,
        GDPRDataRequest,
        RectificationRequest,
        process_gdpr_request_async
    )
    from automl_platform.auth import (
        User,
        get_current_user,
        require_permission,
        require_plan,
        PlanType,
        get_db
    )
    from automl_platform.sso_service import SSOService, SSOProvider
    from automl_platform.audit_service import AuditService, AuditEventType
    from automl_platform.rgpd_compliance_service import (
        RGPDComplianceService,
        GDPRRequestType,
        ConsentType,
        get_rgpd_service
    )
    AUTH_ENDPOINTS_AVAILABLE = True
except ImportError:
    AUTH_ENDPOINTS_AVAILABLE = False
    sso_router = None
    rgpd_router = None
    create_auth_router = None
    SSOLoginRequest = None
    ConsentRequest = None
    GDPRDataRequest = None
    RectificationRequest = None
    process_gdpr_request_async = None
    User = None
    get_current_user = None
    require_permission = None
    require_plan = None
    PlanType = None
    get_db = None
    SSOService = None
    SSOProvider = None
    AuditService = None
    AuditEventType = None
    RGPDComplianceService = None
    GDPRRequestType = None
    ConsentType = None
    get_rgpd_service = None


# ============================================================================
# Test Application Setup
# ============================================================================

@pytest.fixture
def app():
    """Create FastAPI test application with auth routes."""
    if not AUTH_ENDPOINTS_AVAILABLE:
        pytest.skip("Auth endpoints module not available")
    
    test_app = FastAPI()
    
    # Include auth routers
    auth_router = create_auth_router()
    test_app.include_router(auth_router)
    
    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
async def async_client(app):
    """Create async test client for async endpoints."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    if not AUTH_ENDPOINTS_AVAILABLE:
        pytest.skip("Auth module not available")
    
    user = Mock(spec=User)
    user.id = uuid.uuid4()
    user.username = "testuser"
    user.email = "test@example.com"
    user.tenant_id = uuid.uuid4()
    user.plan_type = PlanType.PRO.value
    user.is_active = True
    user.roles = [Mock(name="user")]
    return user


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user."""
    if not AUTH_ENDPOINTS_AVAILABLE:
        pytest.skip("Auth module not available")
    
    user = Mock(spec=User)
    user.id = uuid.uuid4()
    user.username = "admin"
    user.email = "admin@example.com"
    user.tenant_id = uuid.uuid4()
    user.plan_type = PlanType.ENTERPRISE.value
    user.is_active = True
    user.roles = [Mock(name="admin")]
    return user


@pytest.fixture
def mock_sso_service():
    """Create mock SSO service."""
    if not AUTH_ENDPOINTS_AVAILABLE:
        pytest.skip("Auth module not available")
    
    with patch('automl_platform.api.auth_endpoints.SSOService') as mock_sso:
        service = AsyncMock(spec=SSOService)
        mock_sso.return_value = service
        yield service


@pytest.fixture
def mock_audit_service():
    """Create mock audit service."""
    if not AUTH_ENDPOINTS_AVAILABLE:
        pytest.skip("Auth module not available")
    
    with patch('automl_platform.api.auth_endpoints.AuditService') as mock_audit:
        service = AsyncMock(spec=AuditService)
        mock_audit.return_value = service
        yield service


@pytest.fixture
def mock_rgpd_service():
    """Create mock RGPD compliance service."""
    if not AUTH_ENDPOINTS_AVAILABLE:
        pytest.skip("Auth module not available")
    
    with patch('automl_platform.api.auth_endpoints.get_rgpd_service') as mock_get:
        service = Mock(spec=RGPDComplianceService)
        mock_get.return_value = service
        yield service


@pytest.fixture
def mock_get_current_user(mock_user):
    """Mock get_current_user dependency."""
    if not AUTH_ENDPOINTS_AVAILABLE:
        pytest.skip("Auth module not available")
    
    with patch('automl_platform.api.auth_endpoints.get_current_user') as mock_get:
        mock_get.return_value = mock_user
        yield mock_get


@pytest.fixture
def valid_access_token():
    """Create a valid access token for testing."""
    payload = {
        "sub": str(uuid.uuid4()),
        "tenant_id": str(uuid.uuid4()),
        "roles": ["user"],
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4())
    }
    return jwt.encode(payload, "test_secret", algorithm="HS256")


# ============================================================================
# SSO Login Tests
# ============================================================================

class TestSSOLogin:
    """Tests for SSO login endpoint."""
    
    @pytest.mark.asyncio
    async def test_sso_login_success(
        self, client, mock_sso_service, mock_audit_service
    ):
        """Test successful SSO login initiation."""
        # Setup mock responses
        mock_sso_service.get_authorization_url.return_value = {
            "authorization_url": "https://sso.provider.com/authorize?client_id=test",
            "state": "random_state_token"
        }
        
        # Make request
        response = client.post(
            "/api/sso/login",
            json={
                "provider": "keycloak",
                "redirect_uri": "http://localhost:3000/callback"
            }
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "authorization_url" in data
        assert "state" in data
        assert data["authorization_url"] == "https://sso.provider.com/authorize?client_id=test"
        
        # Verify audit was logged
        mock_audit_service.log_event.assert_called_once()
        call_args = mock_audit_service.log_event.call_args[1]
        assert call_args["event_type"] == AuditEventType.LOGIN
        assert call_args["action"] == "sso_login_initiated"
    
    @pytest.mark.asyncio
    async def test_sso_login_invalid_provider(
        self, client, mock_sso_service, mock_audit_service
    ):
        """Test SSO login with invalid provider."""
        # Setup mock to raise exception
        mock_sso_service.get_authorization_url.side_effect = Exception("Invalid provider")
        
        # Make request
        response = client.post(
            "/api/sso/login",
            json={
                "provider": "invalid_provider",
                "redirect_uri": "http://localhost:3000/callback"
            }
        )
        
        # Assertions
        assert response.status_code == 500
        assert "SSO login initialization failed" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_sso_login_missing_redirect_uri(self, client):
        """Test SSO login without redirect URI."""
        response = client.post(
            "/api/sso/login",
            json={"provider": "keycloak"}
        )
        
        # Should fail validation
        assert response.status_code == 422


# ============================================================================
# SSO Callback Tests
# ============================================================================

class TestSSOCallback:
    """Tests for SSO callback endpoint."""
    
    @pytest.mark.asyncio
    async def test_sso_callback_success(
        self, client, mock_sso_service, mock_audit_service
    ):
        """Test successful SSO callback."""
        # Setup mock responses
        mock_sso_service.handle_callback.return_value = {
            "session_id": "session_123",
            "user": {
                "sub": str(uuid.uuid4()),
                "email": "user@example.com",
                "name": "Test User"
            },
            "tokens": {
                "access_token": "access_token_123",
                "refresh_token": "refresh_token_123",
                "expires_in": 3600
            }
        }
        
        # Make request
        response = client.get(
            "/api/sso/callback",
            params={
                "code": "auth_code_123",
                "state": "state_token_123",
                "provider": "keycloak"
            }
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "user" in data
        assert "access_token" in data
        assert data["session_id"] == "session_123"
        
        # Verify audit was logged
        mock_audit_service.log_event.assert_called_once()
        call_args = mock_audit_service.log_event.call_args[1]
        assert call_args["event_type"] == AuditEventType.LOGIN
        assert call_args["action"] == "sso_login_completed"
    
    @pytest.mark.asyncio
    async def test_sso_callback_invalid_code(
        self, client, mock_sso_service
    ):
        """Test SSO callback with invalid authorization code."""
        # Setup mock to raise exception
        mock_sso_service.handle_callback.side_effect = Exception("Invalid authorization code")
        
        # Make request
        response = client.get(
            "/api/sso/callback",
            params={
                "code": "invalid_code",
                "state": "state_token_123",
                "provider": "keycloak"
            }
        )
        
        # Assertions
        assert response.status_code == 400
        assert "SSO authentication failed" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_sso_callback_user_disabled(
        self, client, mock_sso_service
    ):
        """Test SSO callback when user account is disabled."""
        # Setup mock to return disabled user
        mock_sso_service.handle_callback.return_value = {
            "user": {
                "sub": str(uuid.uuid4()),
                "email": "disabled@example.com",
                "disabled": True
            }
        }
        
        # Make request
        response = client.get(
            "/api/sso/callback",
            params={
                "code": "auth_code_123",
                "state": "state_token_123"
            }
        )
        
        # Should handle disabled user appropriately
        # Implementation specific - adjust based on actual behavior
        assert response.status_code in [400, 403]


# ============================================================================
# Logout Tests
# ============================================================================

class TestLogout:
    """Tests for logout endpoint."""
    
    @pytest.mark.asyncio
    async def test_logout_success(
        self, client, mock_sso_service, mock_audit_service,
        mock_get_current_user, mock_user, valid_access_token
    ):
        """Test successful logout."""
        # Setup mock responses
        mock_sso_service.logout.return_value = "https://sso.provider.com/logout"
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make request with auth header
            response = client.post(
                "/api/sso/logout",
                json={
                    "session_id": "session_123",
                    "provider": "keycloak",
                    "redirect_uri": "http://localhost:3000/"
                },
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "logout_url" in data
        assert data["logout_url"] == "https://sso.provider.com/logout"
        
        # Verify audit was logged
        mock_audit_service.log_event.assert_called_once()
        call_args = mock_audit_service.log_event.call_args[1]
        assert call_args["event_type"] == AuditEventType.LOGOUT
        assert call_args["action"] == "sso_logout"
        assert call_args["user_id"] == str(mock_user.id)
    
    @pytest.mark.asyncio
    async def test_logout_without_auth(self, client):
        """Test logout without authentication."""
        response = client.post(
            "/api/sso/logout",
            json={
                "session_id": "session_123",
                "provider": "keycloak"
            }
        )
        
        # Should require authentication
        assert response.status_code in [401, 403]
    
    @pytest.mark.asyncio
    async def test_logout_invalid_token(self, client):
        """Test logout with invalid/expired token."""
        response = client.post(
            "/api/sso/logout",
            json={
                "session_id": "session_123",
                "provider": "keycloak"
            },
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        # Should reject invalid token
        assert response.status_code == 401


# ============================================================================
# Token Refresh Tests
# ============================================================================

class TestTokenRefresh:
    """Tests for token refresh endpoint."""
    
    @pytest.mark.asyncio
    async def test_refresh_token_success(
        self, client, mock_sso_service, mock_user, valid_access_token
    ):
        """Test successful token refresh."""
        # Setup mock responses
        mock_sso_service.get_session.return_value = {
            "refresh_token": "refresh_token_123",
            "user_id": str(mock_user.id)
        }
        mock_sso_service.refresh_token.return_value = {
            "access_token": "new_access_token_123",
            "expires_in": 3600
        }
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make request
            response = client.post(
                "/api/sso/refresh",
                json={
                    "session_id": "session_123",
                    "provider": "keycloak"
                },
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "expires_in" in data
        assert data["access_token"] == "new_access_token_123"
    
    @pytest.mark.asyncio
    async def test_refresh_token_session_not_found(
        self, client, mock_sso_service, mock_user, valid_access_token
    ):
        """Test token refresh with non-existent session."""
        # Setup mock to return None
        mock_sso_service.get_session.return_value = None
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make request
            response = client.post(
                "/api/sso/refresh",
                json={
                    "session_id": "nonexistent_session",
                    "provider": "keycloak"
                },
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_refresh_token_expired(
        self, client, mock_sso_service, mock_user, valid_access_token
    ):
        """Test token refresh with expired refresh token."""
        # Setup mock responses
        mock_sso_service.get_session.return_value = {
            "refresh_token": "expired_refresh_token"
        }
        mock_sso_service.refresh_token.side_effect = Exception("Refresh token expired")
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make request
            response = client.post(
                "/api/sso/refresh",
                json={
                    "session_id": "session_123",
                    "provider": "keycloak"
                },
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 400
        assert "Token refresh failed" in response.json()["detail"]


# ============================================================================
# RGPD Consent Tests
# ============================================================================

class TestRGPDConsent:
    """Tests for RGPD consent endpoints."""
    
    @pytest.mark.asyncio
    async def test_update_consent_grant(
        self, client, mock_rgpd_service, mock_user, valid_access_token
    ):
        """Test granting consent."""
        # Setup mock responses
        mock_rgpd_service.record_consent.return_value = "consent_123"
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make request
            response = client.post(
                "/api/rgpd/consent",
                json={
                    "consent_type": "marketing",
                    "granted": True,
                    "purpose": "Send promotional emails",
                    "data_categories": ["email", "name"],
                    "expires_in_days": 365
                },
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["consent_id"] == "consent_123"
        assert data["consent_type"] == "marketing"
        assert data["granted"] is True
        
        # Verify service was called correctly
        mock_rgpd_service.record_consent.assert_called_once()
        call_args = mock_rgpd_service.record_consent.call_args[1]
        assert call_args["user_id"] == str(mock_user.id)
        assert call_args["consent_type"] == ConsentType.MARKETING
        assert call_args["granted"] is True
    
    @pytest.mark.asyncio
    async def test_update_consent_revoke(
        self, client, mock_rgpd_service, mock_user, valid_access_token
    ):
        """Test revoking consent."""
        # Setup mock responses
        mock_rgpd_service.record_consent.return_value = "consent_124"
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make request
            response = client.post(
                "/api/rgpd/consent",
                json={
                    "consent_type": "analytics",
                    "granted": False
                },
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["granted"] is False
    
    @pytest.mark.asyncio
    async def test_get_user_consents(
        self, client, mock_rgpd_service, mock_user, valid_access_token
    ):
        """Test retrieving user consents."""
        # Setup mock responses
        mock_rgpd_service.get_user_consents.return_value = [
            {
                "consent_type": "marketing",
                "granted": True,
                "recorded_at": "2024-01-01T00:00:00Z"
            },
            {
                "consent_type": "analytics",
                "granted": False,
                "recorded_at": "2024-01-02T00:00:00Z"
            }
        ]
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make request
            response = client.get(
                "/api/rgpd/consent",
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "consents" in data
        assert len(data["consents"]) == 2
        assert data["consents"][0]["consent_type"] == "marketing"
        assert data["consents"][1]["granted"] is False


# ============================================================================
# GDPR Data Request Tests
# ============================================================================

class TestGDPRDataRequests:
    """Tests for GDPR data subject requests."""
    
    @pytest.mark.asyncio
    async def test_create_access_request(
        self, client, mock_rgpd_service, mock_user, valid_access_token
    ):
        """Test creating data access request."""
        # Setup mock responses
        mock_rgpd_service.create_data_request.return_value = "request_123"
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            with patch('automl_platform.api.auth_endpoints.BackgroundTasks') as mock_bg:
                # Make request
                response = client.post(
                    "/api/rgpd/request",
                    json={
                        "request_type": "access",
                        "reason": "Want to see my data",
                        "format": "json"
                    },
                    headers={"Authorization": f"Bearer {valid_access_token}"}
                )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "request_123"
        assert data["request_type"] == "access"
        assert data["status"] == "pending"
        assert "deadline" in data
        
        # Verify background task was scheduled
        mock_bg.return_value.add_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_erasure_request(
        self, client, mock_rgpd_service, mock_user, valid_access_token
    ):
        """Test creating data erasure request."""
        # Setup mock responses
        mock_rgpd_service.create_data_request.return_value = "request_124"
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make request
            response = client.delete(
                "/api/rgpd/data",
                params={"reason": "Closing my account"},
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == "request_124"
        assert "warning" in data
        assert "irreversible" in data["warning"].lower()
        
        # Verify service was called with erasure type
        mock_rgpd_service.create_data_request.assert_called_once()
        call_args = mock_rgpd_service.create_data_request.call_args[1]
        assert call_args["request_type"] == GDPRRequestType.ERASURE
    
    @pytest.mark.asyncio
    async def test_export_personal_data(
        self, client, mock_rgpd_service, mock_user, valid_access_token
    ):
        """Test exporting personal data (portability)."""
        # Setup mock responses
        mock_rgpd_service.create_data_request.return_value = "request_125"
        mock_rgpd_service.process_portability_request.return_value = json.dumps({
            "user_id": str(mock_user.id),
            "email": mock_user.email,
            "data": "user_data_here"
        })
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make request
            response = client.get(
                "/api/rgpd/data/export",
                params={"format": "json"},
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert "attachment" in response.headers.get("content-disposition", "")
        
        # Verify data contains user info
        data = response.json()
        assert data["user_id"] == str(mock_user.id)
        assert data["email"] == mock_user.email
    
    @pytest.mark.asyncio
    async def test_rectify_personal_data(
        self, client, mock_rgpd_service, mock_user, valid_access_token
    ):
        """Test rectifying personal data."""
        # Setup mock responses
        mock_rgpd_service.create_data_request.return_value = "request_126"
        mock_rgpd_service.process_rectification_request.return_value = {
            "status": "completed",
            "rectified_fields": ["email", "phone"]
        }
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make request
            response = client.post(
                "/api/rgpd/data/rectify",
                json={
                    "corrections": {
                        "email": "newemail@example.com",
                        "phone": "+1234567890"
                    },
                    "reason": "Updating contact information"
                },
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert "email" in data["rectified_fields"]
        assert "phone" in data["rectified_fields"]


# ============================================================================
# Permission and Plan Tests
# ============================================================================

class TestPermissionsAndPlans:
    """Tests for permission and plan-based access control."""
    
    @pytest.mark.asyncio
    async def test_endpoint_requires_permission(
        self, client, mock_user, valid_access_token
    ):
        """Test endpoint that requires specific permission."""
        # Setup user without required permission
        mock_user.roles = [Mock(name="user")]
        
        # Override dependencies
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            with patch('automl_platform.api.auth_endpoints.require_permission') as mock_require:
                mock_require.side_effect = HTTPException(
                    status_code=403,
                    detail="Insufficient permissions"
                )
                
                # Try to access admin-only endpoint (compliance report)
                response = client.get(
                    "/api/rgpd/compliance/report",
                    headers={"Authorization": f"Bearer {valid_access_token}"}
                )
        
        # Should be forbidden
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_endpoint_requires_plan(
        self, client, mock_user, valid_access_token
    ):
        """Test endpoint that requires specific plan."""
        # Setup user with FREE plan
        mock_user.plan_type = PlanType.FREE.value
        
        # Override dependencies
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            with patch('automl_platform.api.auth_endpoints.require_plan') as mock_require:
                mock_require.side_effect = HTTPException(
                    status_code=402,
                    detail="This feature requires PRO plan or higher"
                )
                
                # Try to access feature requiring higher plan
                response = client.get(
                    "/api/rgpd/data/mapping",
                    headers={"Authorization": f"Bearer {valid_access_token}"}
                )
        
        # Should require payment/upgrade
        assert response.status_code == 402
    
    @pytest.mark.asyncio
    async def test_admin_access_compliance_report(
        self, client, mock_rgpd_service, mock_admin_user, valid_access_token
    ):
        """Test admin accessing compliance report."""
        # Setup mock responses
        mock_rgpd_service.generate_compliance_report.return_value = {
            "total_requests": 100,
            "completed_requests": 95,
            "pending_requests": 5,
            "average_response_time": "2.5 days"
        }
        
        # Override dependency with admin user
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_admin_user):
            # Make request
            response = client.get(
                "/api/rgpd/compliance/report",
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert data["total_requests"] == 100


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in auth endpoints."""
    
    @pytest.mark.asyncio
    async def test_sso_service_unavailable(
        self, client, mock_sso_service
    ):
        """Test handling when SSO service is unavailable."""
        # Setup mock to simulate service unavailable
        mock_sso_service.get_authorization_url.side_effect = Exception("Service unavailable")
        
        # Make request
        response = client.post(
            "/api/sso/login",
            json={
                "provider": "keycloak",
                "redirect_uri": "http://localhost:3000/callback"
            }
        )
        
        # Should handle gracefully
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_rgpd_service_error(
        self, client, mock_rgpd_service, mock_user, valid_access_token
    ):
        """Test handling RGPD service errors."""
        # Setup mock to raise exception
        mock_rgpd_service.record_consent.side_effect = Exception("Database error")
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make request
            response = client.post(
                "/api/rgpd/consent",
                json={
                    "consent_type": "marketing",
                    "granted": True
                },
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Should handle error
        assert response.status_code == 500
        assert "Failed to update consent" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_malformed_request_body(self, client):
        """Test handling malformed request body."""
        # Send invalid JSON
        response = client.post(
            "/api/sso/login",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return validation error
        assert response.status_code == 422


# ============================================================================
# Integration Tests
# ============================================================================

class TestAuthEndpointsIntegration:
    """Integration tests for auth endpoints."""
    
    @pytest.mark.asyncio
    async def test_full_sso_flow(
        self, client, mock_sso_service, mock_audit_service
    ):
        """Test complete SSO login flow."""
        # Step 1: Initiate login
        mock_sso_service.get_authorization_url.return_value = {
            "authorization_url": "https://sso.provider.com/authorize",
            "state": "state_123"
        }
        
        response = client.post(
            "/api/sso/login",
            json={
                "provider": "keycloak",
                "redirect_uri": "http://localhost:3000/callback"
            }
        )
        assert response.status_code == 200
        state = response.json()["state"]
        
        # Step 2: Handle callback
        mock_sso_service.handle_callback.return_value = {
            "session_id": "session_123",
            "user": {"sub": str(uuid.uuid4())},
            "tokens": {
                "access_token": "access_123",
                "refresh_token": "refresh_123",
                "expires_in": 3600
            }
        }
        
        response = client.get(
            "/api/sso/callback",
            params={
                "code": "auth_code",
                "state": state,
                "provider": "keycloak"
            }
        )
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        
        # Step 3: Refresh token
        mock_sso_service.get_session.return_value = {
            "refresh_token": "refresh_123"
        }
        mock_sso_service.refresh_token.return_value = {
            "access_token": "new_access_123",
            "expires_in": 3600
        }
        
        # Need to mock current user for refresh
        with patch('automl_platform.api.auth_endpoints.get_current_user'):
            response = client.post(
                "/api/sso/refresh",
                json={
                    "session_id": session_id,
                    "provider": "keycloak"
                },
                headers={"Authorization": "Bearer access_123"}
            )
        
        # Verify audit trail
        assert mock_audit_service.log_event.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_gdpr_request_lifecycle(
        self, client, mock_rgpd_service, mock_user, valid_access_token
    ):
        """Test complete GDPR request lifecycle."""
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Step 1: Create data access request
            mock_rgpd_service.create_data_request.return_value = "request_123"
            
            response = client.post(
                "/api/rgpd/request",
                json={
                    "request_type": "access",
                    "format": "json"
                },
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
            assert response.status_code == 200
            request_id = response.json()["request_id"]
            
            # Step 2: Check request status
            response = client.get(
                f"/api/rgpd/request/{request_id}",
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
            assert response.status_code == 200
            assert response.json()["status"] == "processing"
            
            # Step 3: Export data when ready
            mock_rgpd_service.process_portability_request.return_value = json.dumps({
                "user_data": "exported"
            })
            
            response = client.get(
                "/api/rgpd/data/export",
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
            assert response.status_code == 200


# ============================================================================
# Async Endpoint Tests
# ============================================================================

class TestAsyncEndpoints:
    """Tests for async endpoints using httpx.AsyncClient."""
    
    @pytest.mark.asyncio
    async def test_async_sso_login(self, async_client, mock_sso_service):
        """Test async SSO login endpoint."""
        # Setup mock
        mock_sso_service.get_authorization_url.return_value = {
            "authorization_url": "https://sso.provider.com/authorize",
            "state": "state_async"
        }
        
        # Make async request
        response = await async_client.post(
            "/api/sso/login",
            json={
                "provider": "auth0",
                "redirect_uri": "http://localhost:3000/callback"
            }
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "state_async"
    
    @pytest.mark.asyncio
    async def test_async_consent_update(
        self, async_client, mock_rgpd_service, mock_user, valid_access_token
    ):
        """Test async consent update endpoint."""
        # Setup mock
        mock_rgpd_service.record_consent.return_value = "consent_async"
        
        # Override dependency
        with patch('automl_platform.api.auth_endpoints.get_current_user', return_value=mock_user):
            # Make async request
            response = await async_client.post(
                "/api/rgpd/consent",
                json={
                    "consent_type": "functional",
                    "granted": True
                },
                headers={"Authorization": f"Bearer {valid_access_token}"}
            )
        
        # Assertions
        assert response.status_code == 200
        assert response.json()["consent_id"] == "consent_async"
