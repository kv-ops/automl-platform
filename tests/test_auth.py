"""
Tests for Authentication and Authorization Service
====================================================
Tests for JWT tokens, RBAC, API keys, SSO, and multi-tenant isolation.
"""

import pytest
import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import jwt
import redis

# Import auth components
try:
    from automl_platform.auth import (
        PasswordService,
        TokenService,
        APIKeyService,
        RBACService,
        QuotaService,
        AuditService,
        User,
        Role,
        Permission,
        Tenant,
        APIKey,
        PlanType,
        get_current_user,
        require_permission,
        require_plan,
        RateLimiter,
        AuthConfig
    )
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    PasswordService = None
    TokenService = None
    APIKeyService = None
    RBACService = None
    QuotaService = None
    AuditService = None
    User = None
    Role = None
    Permission = None
    Tenant = None
    APIKey = None
    PlanType = None
    get_current_user = None
    require_permission = None
    require_plan = None
    RateLimiter = None
    AuthConfig = None

try:
    from automl_platform.audit_service import AuditEventType, AuditSeverity
except ImportError:
    AuditEventType = None
    AuditSeverity = None


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def password_service():
    """Create a PasswordService instance."""
    if not AUTH_AVAILABLE:
        pytest.skip("Auth module not available")
    return PasswordService()


@pytest.fixture
def token_service():
    """Create a TokenService instance with mocked Redis."""
    if not AUTH_AVAILABLE:
        pytest.skip("Auth module not available")
    
    with patch('automl_platform.auth.redis.from_url') as mock_redis:
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        service = TokenService()
        service.redis_client = mock_client
        return service


@pytest.fixture
def api_key_service():
    """Create an APIKeyService instance."""
    if not AUTH_AVAILABLE:
        pytest.skip("Auth module not available")
    return APIKeyService()


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = MagicMock()
    return session


@pytest.fixture
def sample_user():
    """Create a sample user object."""
    if not AUTH_AVAILABLE:
        pytest.skip("Auth module not available")
    
    user = Mock(spec=User)
    user.id = uuid.uuid4()
    user.username = "testuser"
    user.email = "test@example.com"
    user.tenant_id = uuid.uuid4()
    user.plan_type = PlanType.PRO.value
    user.is_active = True
    user.roles = []
    user.max_workers = 4
    user.max_concurrent_jobs = 4
    user.storage_quota_gb = 100
    user.monthly_compute_minutes = 10000
    return user


@pytest.fixture
def sample_tenant():
    """Create a sample tenant object."""
    if not AUTH_AVAILABLE:
        pytest.skip("Auth module not available")
    
    tenant = Mock(spec=Tenant)
    tenant.id = uuid.uuid4()
    tenant.name = "Test Tenant"
    tenant.subdomain = "test"
    tenant.plan_type = PlanType.ENTERPRISE.value
    tenant.is_active = True
    return tenant


@pytest.fixture
def rbac_service(mock_db_session):
    """Create an RBACService instance."""
    if not AUTH_AVAILABLE:
        pytest.skip("Auth module not available")
    
    # Mock the role query
    mock_db_session.query.return_value.filter_by.return_value.first.return_value = None
    return RBACService(mock_db_session)


@pytest.fixture
def quota_service(mock_db_session):
    """Create a QuotaService instance."""
    if not AUTH_AVAILABLE:
        pytest.skip("Auth module not available")

    with patch('automl_platform.auth.redis.from_url') as mock_redis:
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        service = QuotaService(mock_db_session, mock_client)
        return service


@pytest.fixture
def audit_backend():
    with patch('automl_platform.auth._get_audit_backend') as mock_get_backend:
        backend = MagicMock()
        mock_get_backend.return_value = backend
        yield backend


@pytest.fixture
def audit_service(mock_db_session, audit_backend):
    """Create an AuditService instance."""
    if not AUTH_AVAILABLE:
        pytest.skip("Auth module not available")
    with patch('automl_platform.auth.Counter') as mock_counter, patch('automl_platform.auth.Histogram') as mock_histogram:
        mock_counter.return_value = MagicMock()
        mock_histogram.return_value = MagicMock()
        return AuditService(mock_db_session)


@pytest.fixture
def rate_limiter():
    """Create a RateLimiter instance."""
    if not AUTH_AVAILABLE:
        pytest.skip("Auth module not available")
    
    mock_redis = MagicMock()
    return RateLimiter(mock_redis)


# ============================================================================
# Password Service Tests
# ============================================================================

class TestPasswordService:
    """Tests for PasswordService."""
    
    def test_hash_password(self, password_service):
        """Test password hashing."""
        password = "TestPassword123!"
        hashed = password_service.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert "$2b$" in hashed  # bcrypt hash prefix
    
    def test_verify_password_correct(self, password_service):
        """Test verifying correct password."""
        password = "TestPassword123!"
        hashed = password_service.hash_password(password)
        
        assert password_service.verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self, password_service):
        """Test verifying incorrect password."""
        password = "TestPassword123!"
        hashed = password_service.hash_password(password)
        
        assert password_service.verify_password("WrongPassword", hashed) is False
    
    def test_validate_password_strength_valid(self, password_service):
        """Test password strength validation with valid password."""
        valid, message = password_service.validate_password_strength("Test123!Pass")
        assert valid is True
        assert message == "Password is valid"
    
    def test_validate_password_strength_too_short(self, password_service):
        """Test password strength validation with short password."""
        valid, message = password_service.validate_password_strength("Test1!")
        assert valid is False
        assert "at least 8 characters" in message
    
    def test_validate_password_strength_no_uppercase(self, password_service):
        """Test password strength validation without uppercase."""
        valid, message = password_service.validate_password_strength("test123!pass")
        assert valid is False
        assert "uppercase letter" in message
    
    def test_validate_password_strength_no_lowercase(self, password_service):
        """Test password strength validation without lowercase."""
        valid, message = password_service.validate_password_strength("TEST123!PASS")
        assert valid is False
        assert "lowercase letter" in message
    
    def test_validate_password_strength_no_digit(self, password_service):
        """Test password strength validation without digit."""
        valid, message = password_service.validate_password_strength("TestPass!")
        assert valid is False
        assert "digit" in message
    
    def test_validate_password_strength_no_special(self, password_service):
        """Test password strength validation without special character."""
        valid, message = password_service.validate_password_strength("TestPass123")
        assert valid is False
        assert "special character" in message


# ============================================================================
# Token Service Tests
# ============================================================================

class TestTokenService:
    """Tests for TokenService."""
    
    def test_create_access_token(self, token_service):
        """Test creating access token."""
        user_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())
        roles = ["admin", "user"]
        
        token = token_service.create_access_token(user_id, tenant_id, roles)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode without verification for testing
        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == user_id
        assert payload["tenant_id"] == tenant_id
        assert payload["roles"] == roles
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload
    
    def test_create_access_token_with_custom_expiry(self, token_service):
        """Test creating access token with custom expiry."""
        user_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())
        roles = ["user"]
        expires_delta = timedelta(minutes=5)
        
        token = token_service.create_access_token(
            user_id, tenant_id, roles, expires_delta
        )
        
        payload = jwt.decode(token, options={"verify_signature": False})
        exp_time = datetime.fromtimestamp(payload["exp"])
        iat_time = datetime.fromtimestamp(payload["iat"])
        
        # Check that expiry is approximately 5 minutes after issued at
        delta = exp_time - iat_time
        assert 290 < delta.total_seconds() < 310  # Allow some tolerance
    
    def test_create_refresh_token(self, token_service):
        """Test creating refresh token."""
        user_id = str(uuid.uuid4())
        
        token = token_service.create_refresh_token(user_id)
        
        assert isinstance(token, str)
        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["sub"] == user_id
        assert payload["type"] == "refresh"
        assert "exp" in payload
        assert "jti" in payload
    
    def test_verify_token_valid(self, token_service):
        """Test verifying valid token."""
        user_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())
        roles = ["admin"]
        
        token = token_service.create_access_token(user_id, tenant_id, roles)
        
        # Mock Redis exists to return True
        token_service.redis_client.exists.return_value = True
        
        payload = token_service.verify_token(token)
        assert payload["sub"] == user_id
        assert payload["tenant_id"] == tenant_id
        assert payload["roles"] == roles
    
    def test_verify_token_expired(self, token_service):
        """Test verifying expired token."""
        # Create token with negative expiry
        user_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())
        roles = ["user"]
        
        token = token_service.create_access_token(
            user_id, tenant_id, roles, timedelta(seconds=-1)
        )
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            token_service.verify_token(token)
        
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail
    
    def test_verify_token_revoked(self, token_service):
        """Test verifying revoked token."""
        user_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())
        roles = ["admin"]
        
        token = token_service.create_access_token(user_id, tenant_id, roles)
        
        # Mock Redis exists to return False (token revoked)
        token_service.redis_client.exists.return_value = False
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            token_service.verify_token(token)
        
        assert exc_info.value.status_code == 401
        assert "revoked" in exc_info.value.detail
    
    def test_revoke_token(self, token_service):
        """Test revoking token."""
        user_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())
        roles = ["admin"]
        
        token = token_service.create_access_token(user_id, tenant_id, roles)
        
        token_service.revoke_token(token)
        
        # Verify Redis delete was called
        payload = jwt.decode(token, options={"verify_signature": False})
        token_service.redis_client.delete.assert_called_with(f"token:{payload['jti']}")


# ============================================================================
# API Key Service Tests
# ============================================================================

class TestAPIKeyService:
    """Tests for APIKeyService."""
    
    def test_generate_api_key(self, api_key_service):
        """Test generating API key."""
        key = api_key_service.generate_api_key()
        
        assert key.startswith("mlops_")
        assert len(key) > len("mlops_")
    
    def test_hash_api_key(self, api_key_service):
        """Test hashing API key."""
        api_key = "mlops_test_key_123"
        hashed = api_key_service.hash_api_key(api_key)
        
        assert hashed != api_key
        assert len(hashed) == 64  # SHA256 produces 64 character hex string
    
    def test_verify_api_key_correct(self, api_key_service):
        """Test verifying correct API key."""
        api_key = api_key_service.generate_api_key()
        key_hash = api_key_service.hash_api_key(api_key)
        
        assert api_key_service.verify_api_key(api_key, key_hash) is True
    
    def test_verify_api_key_incorrect(self, api_key_service):
        """Test verifying incorrect API key."""
        api_key = api_key_service.generate_api_key()
        wrong_key = "mlops_wrong_key"
        key_hash = api_key_service.hash_api_key(api_key)
        
        assert api_key_service.verify_api_key(wrong_key, key_hash) is False


# ============================================================================
# RBAC Service Tests
# ============================================================================

class TestRBACService:
    """Tests for RBACService."""
    
    def test_init_system_roles(self, rbac_service, mock_db_session):
        """Test initialization of system roles."""
        # Verify that system roles were created
        assert mock_db_session.add.called
        assert mock_db_session.commit.called or mock_db_session.rollback.called
    
    def test_check_permission_admin(self, rbac_service, sample_user):
        """Test permission check for admin user."""
        # Create admin role
        admin_role = Mock(spec=Role)
        admin_role.name = "admin"
        sample_user.roles = [admin_role]
        
        # Admin should have all permissions
        assert rbac_service.check_permission(
            sample_user, "models", "create"
        ) is True
        assert rbac_service.check_permission(
            sample_user, "datasets", "delete"
        ) is True
    
    def test_check_permission_specific(self, rbac_service, sample_user):
        """Test permission check for specific permission."""
        # Create role with specific permission
        role = Mock(spec=Role)
        role.name = "data_scientist"
        
        permission = Mock(spec=Permission)
        permission.resource = "models"
        permission.action = "create"
        permission.scope = "all"
        
        role.permissions = [permission]
        sample_user.roles = [role]
        
        # Should have the specific permission
        assert rbac_service.check_permission(
            sample_user, "models", "create"
        ) is True
        
        # Should not have other permissions
        assert rbac_service.check_permission(
            sample_user, "models", "delete"
        ) is False
    
    def test_check_permission_scope_own(self, rbac_service, sample_user):
        """Test permission check with 'own' scope."""
        role = Mock(spec=Role)
        role.name = "user"
        
        permission = Mock(spec=Permission)
        permission.resource = "models"
        permission.action = "update"
        permission.scope = "own"
        
        role.permissions = [permission]
        sample_user.roles = [role]
        
        # Should have permission for own resources
        assert rbac_service.check_permission(
            sample_user, "models", "update", str(sample_user.id)
        ) is True
        
        # Should not have permission for others' resources
        other_user_id = str(uuid.uuid4())
        assert rbac_service.check_permission(
            sample_user, "models", "update", other_user_id
        ) is False
    
    def test_check_permission_wildcard(self, rbac_service, sample_user):
        """Test permission check with wildcards."""
        role = Mock(spec=Role)
        role.name = "viewer"
        
        permission = Mock(spec=Permission)
        permission.resource = "*"
        permission.action = "read"
        permission.scope = "all"
        
        role.permissions = [permission]
        sample_user.roles = [role]
        
        # Should have read permission for all resources
        assert rbac_service.check_permission(
            sample_user, "models", "read"
        ) is True
        assert rbac_service.check_permission(
            sample_user, "datasets", "read"
        ) is True
        
        # Should not have write permissions
        assert rbac_service.check_permission(
            sample_user, "models", "create"
        ) is False
    
    def test_get_user_permissions(self, rbac_service, sample_user):
        """Test getting all user permissions."""
        role = Mock(spec=Role)
        role.name = "data_scientist"
        
        permission1 = Mock(spec=Permission)
        permission1.resource = "models"
        permission1.action = "create"
        permission1.scope = "team"
        
        permission2 = Mock(spec=Permission)
        permission2.resource = "datasets"
        permission2.action = "*"
        permission2.scope = "own"
        
        role.permissions = [permission1, permission2]
        sample_user.roles = [role]
        
        permissions = rbac_service.get_user_permissions(sample_user)
        
        assert len(permissions) == 2
        assert "models:create:team" in permissions
        assert "datasets:*:own" in permissions


# ============================================================================
# Quota Service Tests
# ============================================================================

class TestQuotaService:
    """Tests for QuotaService."""
    
    def test_check_quota_within_limit(self, quota_service, sample_user):
        """Test quota check within limits."""
        # Mock Redis to return current usage below limit
        quota_service.redis.get.return_value = b"5"
        
        assert quota_service.check_quota(sample_user, "api_calls", 10) is True
    
    def test_check_quota_exceeds_limit(self, quota_service, sample_user):
        """Test quota check exceeding limits."""
        # Mock Redis to return current usage at limit
        sample_user.plan_type = PlanType.FREE.value
        quota_service.redis.get.return_value = b"995"
        
        assert quota_service.check_quota(sample_user, "api_calls", 10) is False
    
    def test_consume_quota_success(self, quota_service, sample_user):
        """Test consuming quota successfully."""
        quota_service.redis.get.return_value = b"10"
        quota_service.redis.ttl.return_value = -1
        
        quota_service.consume_quota(sample_user, "api_calls", 5)
        
        # Verify Redis increment was called
        quota_service.redis.incrby.assert_called_with(
            f"usage:{sample_user.id}:api_calls", 5
        )
        
        # Verify TTL was set
        quota_service.redis.expire.assert_called()
    
    def test_consume_quota_exceeded(self, quota_service, sample_user):
        """Test consuming quota when limit exceeded."""
        sample_user.plan_type = PlanType.FREE.value
        quota_service.redis.get.return_value = b"999"
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            quota_service.consume_quota(sample_user, "api_calls", 5)
        
        assert exc_info.value.status_code == 429
        assert "Quota exceeded" in exc_info.value.detail
    
    def test_get_plan_limits(self, quota_service):
        """Test getting plan limits."""
        free_limits = quota_service._get_plan_limits(PlanType.FREE.value)
        assert free_limits["workers"] == 1
        assert free_limits["storage_gb"] == 10

        starter_limits = quota_service._get_plan_limits(PlanType.STARTER.value)
        assert starter_limits["workers"] == 3
        assert starter_limits["api_calls"] == 30000

        pro_limits = quota_service._get_plan_limits(PlanType.PRO.value)
        assert pro_limits["workers"] == 4
        assert pro_limits["storage_gb"] == 100

        professional_limits = quota_service._get_plan_limits(PlanType.PROFESSIONAL.value)
        assert professional_limits["workers"] == 8
        assert professional_limits["api_calls"] == 500000

        enterprise_limits = quota_service._get_plan_limits(PlanType.ENTERPRISE.value)
        assert enterprise_limits["workers"] == 999999  # Unlimited


# ============================================================================
# Audit Service Tests
# ============================================================================

class TestAuditService:
    """Tests for AuditService."""

    def test_log_action(self, audit_service, audit_backend):
        """Test logging an action."""
        user_id = str(uuid.uuid4())
        tenant_id = str(uuid.uuid4())

        audit_service.log_action(
            user_id=user_id,
            tenant_id=tenant_id,
            action="login",
            resource_type="auth",
            response_status=200,
            ip_address="127.0.0.1"
        )

        audit_backend.log_event.assert_called_once()
        kwargs = audit_backend.log_event.call_args.kwargs
        if AuditEventType is not None:
            assert kwargs["event_type"] == AuditEventType.LOGIN
        if AuditSeverity is not None:
            assert kwargs["severity"] == AuditSeverity.INFO
        assert kwargs["user_id"] == user_id
        assert kwargs["tenant_id"] == tenant_id

    def test_log_action_login_failure(self, audit_service, audit_backend):
        """Test logging failed login."""
        audit_service.log_action(
            user_id=None,
            tenant_id=None,
            action="login",
            response_status=401,
            ip_address="192.168.1.100"
        )

        assert audit_backend.log_event.called
        kwargs = audit_backend.log_event.call_args.kwargs
        if AuditSeverity is not None:
            assert kwargs["severity"] == AuditSeverity.WARNING

    def test_log_action_backend_failure_is_nonfatal(self, audit_service, audit_backend, caplog):
        """Audit backend failures should not break auth flows."""
        audit_backend.log_event.side_effect = redis.exceptions.ConnectionError("redis down")

        with caplog.at_level("WARNING"):
            audit_service.log_action(
                user_id=str(uuid.uuid4()),
                tenant_id=str(uuid.uuid4()),
                action="login",
                response_status=200,
                ip_address="10.0.0.1",
            )

        assert "Audit backend Redis logging failed" in caplog.text

    def test_log_action_generic_backend_failure_is_nonfatal(self, audit_service, audit_backend, caplog):
        """Unexpected backend errors should be logged and ignored."""
        audit_backend.log_event.side_effect = RuntimeError("boom")

        with caplog.at_level("ERROR"):
            audit_service.log_action(
                user_id=str(uuid.uuid4()),
                tenant_id=str(uuid.uuid4()),
                action="logout",
                response_status=200,
                ip_address="10.0.0.2",
            )

        assert "Audit backend logging failed" in caplog.text

    def test_get_audit_logs(self, audit_service, audit_backend):
        """Test retrieving audit logs."""
        tenant_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())

        audit_backend.search.return_value = []

        logs = audit_service.get_audit_logs(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=50
        )

        assert isinstance(logs, list)
        audit_backend.search.assert_called_once_with(
            tenant_id=tenant_id,
            user_id=user_id,
            start_date=None,
            end_date=None,
            limit=50,
        )

    def test_get_audit_logs_backend_failure_is_nonfatal(self, audit_service, audit_backend, caplog):
        """Audit backend failures while searching should not break auth flows."""
        tenant_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())

        audit_backend.search.side_effect = redis.exceptions.ConnectionError("redis down")

        with caplog.at_level("WARNING"):
            logs = audit_service.get_audit_logs(
                tenant_id=tenant_id,
                user_id=user_id,
                limit=10,
            )

        assert logs == []
        assert "Audit backend Redis search failed" in caplog.text

    def test_get_audit_logs_generic_backend_failure_is_nonfatal(self, audit_service, audit_backend, caplog):
        """Unexpected backend errors during search should be logged and ignored."""
        tenant_id = str(uuid.uuid4())
        user_id = str(uuid.uuid4())

        audit_backend.search.side_effect = RuntimeError("boom")

        with caplog.at_level("ERROR"):
            logs = audit_service.get_audit_logs(
                tenant_id=tenant_id,
                user_id=user_id,
                limit=5,
            )

        assert logs == []
        assert "Audit backend search failed" in caplog.text


# ============================================================================
# Rate Limiter Tests
# ============================================================================

class TestRateLimiter:
    """Tests for RateLimiter."""
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_within_limit(self, rate_limiter, sample_user):
        """Test rate limit check within limits."""
        rate_limiter.redis.incr.return_value = 5
        
        result = await rate_limiter.check_rate_limit(sample_user, "/api/predict")
        
        assert result is True
        rate_limiter.redis.incr.assert_called_with(
            f"rate_limit:{sample_user.id}:/api/predict"
        )
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, rate_limiter, sample_user):
        """Test rate limit check when exceeded."""
        sample_user.plan_type = PlanType.FREE.value
        rate_limiter.redis.incr.return_value = 15  # Exceeds FREE plan limit
        
        result = await rate_limiter.check_rate_limit(sample_user, "/api/predict")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_first_request(self, rate_limiter, sample_user):
        """Test rate limit check for first request."""
        rate_limiter.redis.incr.return_value = 1
        
        result = await rate_limiter.check_rate_limit(sample_user, "/api/predict")
        
        assert result is True
        rate_limiter.redis.expire.assert_called_with(
            f"rate_limit:{sample_user.id}:/api/predict", 60
        )
    
    def test_get_rate_limit_by_plan(self, rate_limiter):
        """Test getting rate limit by plan type."""
        assert rate_limiter._get_rate_limit(PlanType.FREE.value) == 10
        assert rate_limiter._get_rate_limit(PlanType.TRIAL.value) == 60
        assert rate_limiter._get_rate_limit(PlanType.STARTER.value) == 250
        assert rate_limiter._get_rate_limit(PlanType.PRO.value) == 300
        assert rate_limiter._get_rate_limit(PlanType.PROFESSIONAL.value) == 500
        assert rate_limiter._get_rate_limit(PlanType.ENTERPRISE.value) == 9999
        assert rate_limiter._get_rate_limit(PlanType.CUSTOM.value) == 9999


# ============================================================================
# Integration Tests
# ============================================================================

class TestAuthIntegration:
    """Integration tests for auth components."""
    
    def test_password_token_integration(self, password_service, token_service):
        """Test password and token services integration."""
        # Create user with hashed password
        password = "SecurePass123!"
        hashed = password_service.hash_password(password)
        
        # Verify password
        assert password_service.verify_password(password, hashed)
        
        # Create token for authenticated user
        user_id = str(uuid.uuid4())
        token = token_service.create_access_token(user_id, None, ["user"])
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_api_key_workflow(self, api_key_service):
        """Test complete API key workflow."""
        # Generate new API key
        api_key = api_key_service.generate_api_key()
        
        # Hash for storage
        key_hash = api_key_service.hash_api_key(api_key)
        
        # Verify the key
        assert api_key_service.verify_api_key(api_key, key_hash)
        
        # Wrong key should fail
        assert not api_key_service.verify_api_key("wrong_key", key_hash)
    
    @pytest.mark.asyncio
    async def test_quota_and_rate_limit_interaction(
        self, quota_service, rate_limiter, sample_user
    ):
        """Test quota and rate limiting interaction."""
        # Check quota
        quota_service.redis.get.return_value = b"10"
        assert quota_service.check_quota(sample_user, "api_calls", 5)
        
        # Check rate limit
        rate_limiter.redis.incr.return_value = 5
        assert await rate_limiter.check_rate_limit(sample_user, "/api/predict")
        
        # Both should use Redis for tracking
        assert quota_service.redis.get.called
        assert rate_limiter.redis.incr.called


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in auth services."""
    
    def test_token_service_invalid_token(self, token_service):
        """Test handling of invalid token."""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            token_service.verify_token("invalid.token.here")
        
        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail
    
    def test_quota_service_redis_error(self, quota_service, sample_user):
        """Test handling of Redis errors in quota service."""
        # Simulate Redis error
        quota_service.redis.get.side_effect = Exception("Redis connection error")
        
        # Should handle gracefully and default to denying
        with pytest.raises(Exception) as exc_info:
            quota_service.check_quota(sample_user, "api_calls", 1)
        
        assert "Redis connection error" in str(exc_info.value)
    
    def test_rbac_service_no_roles(self, rbac_service, sample_user):
        """Test RBAC with user having no roles."""
        sample_user.roles = []
        
        # Should deny all non-admin permissions
        assert rbac_service.check_permission(
            sample_user, "models", "create"
        ) is False
