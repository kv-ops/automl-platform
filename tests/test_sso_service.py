"""
Tests for SSO Service
=====================
Comprehensive tests for the SSO service with multiple providers.
"""

import pytest
import json
import secrets
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
import httpx
import jwt

from automl_platform.sso_service import (
    SSOService,
    SSOProvider,
    SSOConfig,
    SAMLService
)


class TestSSOConfig:
    """Test suite for SSOConfig."""
    
    def test_keycloak_config_initialization(self):
        """Test Keycloak configuration initialization."""
        config = SSOConfig(
            provider=SSOProvider.KEYCLOAK,
            client_id="test-client",
            client_secret="test-secret",
            domain="http://localhost:8080",
            realm="master"
        )
        
        assert config.provider == SSOProvider.KEYCLOAK
        assert config.client_id == "test-client"
        assert config.client_secret == "test-secret"
        assert config.domain == "http://localhost:8080"
        assert config.realm == "master"
        
        # Check URLs are built correctly
        base_url = "http://localhost:8080/realms/master"
        assert config.authorize_url == f"{base_url}/protocol/openid-connect/auth"
        assert config.token_url == f"{base_url}/protocol/openid-connect/token"
        assert config.userinfo_url == f"{base_url}/protocol/openid-connect/userinfo"
        assert config.jwks_url == f"{base_url}/protocol/openid-connect/certs"
        assert config.logout_url == f"{base_url}/protocol/openid-connect/logout"
    
    def test_auth0_config_initialization(self):
        """Test Auth0 configuration initialization."""
        config = SSOConfig(
            provider=SSOProvider.AUTH0,
            client_id="auth0-client",
            client_secret="auth0-secret",
            domain="test.auth0.com"
        )
        
        assert config.provider == SSOProvider.AUTH0
        assert config.client_id == "auth0-client"
        assert config.client_secret == "auth0-secret"
        assert config.domain == "test.auth0.com"
        
        # Check URLs are built correctly
        assert config.authorize_url == "https://test.auth0.com/authorize"
        assert config.token_url == "https://test.auth0.com/oauth/token"
        assert config.userinfo_url == "https://test.auth0.com/userinfo"
        assert config.jwks_url == "https://test.auth0.com/.well-known/jwks.json"
        assert config.logout_url == "https://test.auth0.com/v2/logout"
    
    def test_okta_config_initialization(self):
        """Test Okta configuration initialization."""
        config = SSOConfig(
            provider=SSOProvider.OKTA,
            client_id="okta-client",
            client_secret="okta-secret",
            domain="test.okta.com"
        )
        
        assert config.provider == SSOProvider.OKTA
        base_url = "https://test.okta.com/oauth2/default"
        assert config.authorize_url == f"{base_url}/v1/authorize"
        assert config.token_url == f"{base_url}/v1/token"
        assert config.userinfo_url == f"{base_url}/v1/userinfo"
        assert config.jwks_url == f"{base_url}/v1/keys"
        assert config.logout_url == f"{base_url}/v1/logout"
    
    def test_azure_ad_config_initialization(self):
        """Test Azure AD configuration initialization."""
        config = SSOConfig(
            provider=SSOProvider.AZURE_AD,
            client_id="azure-client",
            client_secret="azure-secret",
            tenant_id="test-tenant"
        )
        
        assert config.provider == SSOProvider.AZURE_AD
        assert config.tenant_id == "test-tenant"
        
        base_url = "https://login.microsoftonline.com/test-tenant"
        assert config.authorize_url == f"{base_url}/oauth2/v2.0/authorize"
        assert config.token_url == f"{base_url}/oauth2/v2.0/token"
        assert config.userinfo_url == "https://graph.microsoft.com/v1.0/me"
        assert config.jwks_url == f"{base_url}/discovery/v2.0/keys"
        assert config.logout_url == f"{base_url}/oauth2/v2.0/logout"


class TestSSOService:
    """Test suite for SSOService."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis_mock = Mock()
        redis_mock.setex = Mock()
        redis_mock.get = Mock()
        redis_mock.delete = Mock()
        return redis_mock
    
    @pytest.fixture
    def sso_service(self, mock_redis):
        """Create SSO service with mock Redis."""
        with patch.dict('os.environ', {
            'KEYCLOAK_ENABLED': 'true',
            'KEYCLOAK_CLIENT_ID': 'test-client',
            'KEYCLOAK_CLIENT_SECRET': 'test-secret',
            'KEYCLOAK_URL': 'http://localhost:8080',
            'KEYCLOAK_REALM': 'master',
            'AUTH0_ENABLED': 'true',
            'AUTH0_CLIENT_ID': 'auth0-client',
            'AUTH0_CLIENT_SECRET': 'auth0-secret',
            'AUTH0_DOMAIN': 'test.auth0.com'
        }):
            service = SSOService(redis_client=mock_redis)
            return service
    
    @pytest.mark.asyncio
    async def test_get_authorization_url_keycloak(self, sso_service):
        """Test getting authorization URL for Keycloak."""
        result = await sso_service.get_authorization_url(
            provider="keycloak",
            redirect_uri="http://localhost:3000/callback"
        )
        
        assert "authorization_url" in result
        assert "state" in result
        assert "http://localhost:8080/realms/master/protocol/openid-connect/auth" in result["authorization_url"]
        assert "client_id=test-client" in result["authorization_url"]
        assert "redirect_uri=http%3A%2F%2Flocalhost%3A3000%2Fcallback" in result["authorization_url"]
        
        # Verify state was stored in Redis
        sso_service.redis_client.setex.assert_called_once()
        call_args = sso_service.redis_client.setex.call_args
        assert call_args[0][0].startswith("sso:state:")
        assert call_args[0][1] == 300  # TTL
    
    @pytest.mark.asyncio
    async def test_get_authorization_url_auth0(self, sso_service):
        """Test getting authorization URL for Auth0."""
        with patch.dict('os.environ', {'AUTH0_AUDIENCE': 'https://api.example.com'}):
            result = await sso_service.get_authorization_url(
                provider="auth0",
                redirect_uri="http://localhost:3000/callback"
            )
            
            assert "authorization_url" in result
            assert "state" in result
            assert "https://test.auth0.com/authorize" in result["authorization_url"]
            assert "audience=https%3A%2F%2Fapi.example.com" in result["authorization_url"]
    
    @pytest.mark.asyncio
    async def test_get_authorization_url_invalid_provider(self, sso_service):
        """Test getting authorization URL with invalid provider."""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            await sso_service.get_authorization_url(
                provider="invalid",
                redirect_uri="http://localhost:3000/callback"
            )
        
        assert exc_info.value.status_code == 400
        assert "not configured" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_handle_callback_success(self, sso_service, mock_redis):
        """Test successful OAuth callback handling."""
        # Setup state in Redis
        state = "test-state"
        state_data = {
            "provider": "keycloak",
            "redirect_uri": "http://localhost:3000/callback",
            "created_at": datetime.utcnow().isoformat()
        }
        mock_redis.get.return_value = json.dumps(state_data).encode()
        
        # Mock token and userinfo responses
        mock_token_response = Mock()
        mock_token_response.status_code = 200
        mock_token_response.json.return_value = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "id_token": "test-id-token",
            "expires_in": 3600
        }
        
        mock_userinfo_response = Mock()
        mock_userinfo_response.status_code = 200
        mock_userinfo_response.json.return_value = {
            "sub": "user123",
            "email": "user@example.com",
            "name": "Test User",
            "email_verified": True
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_token_response
            mock_client.get.return_value = mock_userinfo_response
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            # Mock ID token validation
            with patch.object(sso_service, '_validate_id_token', new_callable=AsyncMock) as mock_validate:
                mock_validate.return_value = {"sub": "user123"}
                
                result = await sso_service.handle_callback(
                    provider="keycloak",
                    code="test-code",
                    state=state,
                    redirect_uri="http://localhost:3000/callback"
                )
        
        assert "session_id" in result
        assert "user" in result
        assert "tokens" in result
        assert result["user"]["sub"] == "user123"
        assert result["user"]["email"] == "user@example.com"
        assert result["tokens"]["access_token"] == "test-access-token"
        
        # Verify state was deleted
        mock_redis.delete.assert_called_with(f"sso:state:{state}")
    
    @pytest.mark.asyncio
    async def test_handle_callback_invalid_state(self, sso_service, mock_redis):
        """Test callback handling with invalid state."""
        from fastapi import HTTPException
        
        mock_redis.get.return_value = None  # State not found
        
        with pytest.raises(HTTPException) as exc_info:
            await sso_service.handle_callback(
                provider="keycloak",
                code="test-code",
                state="invalid-state",
                redirect_uri="http://localhost:3000/callback"
            )
        
        assert exc_info.value.status_code == 400
        assert "Invalid or expired state" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_handle_callback_token_exchange_failure(self, sso_service, mock_redis):
        """Test callback handling when token exchange fails."""
        from fastapi import HTTPException
        
        # Setup state in Redis
        state_data = {
            "provider": "keycloak",
            "redirect_uri": "http://localhost:3000/callback",
            "created_at": datetime.utcnow().isoformat()
        }
        mock_redis.get.return_value = json.dumps(state_data).encode()
        
        # Mock failed token response
        mock_token_response = Mock()
        mock_token_response.status_code = 400
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_token_response
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            with pytest.raises(HTTPException) as exc_info:
                await sso_service.handle_callback(
                    provider="keycloak",
                    code="test-code",
                    state="test-state",
                    redirect_uri="http://localhost:3000/callback"
                )
            
            assert exc_info.value.status_code == 400
            assert "Failed to exchange code" in str(exc_info.value.detail)
    
    def test_map_user_attributes_keycloak(self, sso_service):
        """Test mapping Keycloak user attributes."""
        user_info = {
            "sub": "keycloak-user",
            "email": "user@keycloak.org",
            "email_verified": True,
            "name": "Keycloak User",
            "given_name": "Keycloak",
            "family_name": "User",
            "realm_access": {"roles": ["admin", "user"]},
            "groups": ["group1", "group2"]
        }
        
        mapped = sso_service._map_user_attributes("keycloak", user_info)
        
        assert mapped["sub"] == "keycloak-user"
        assert mapped["email"] == "user@keycloak.org"
        assert mapped["email_verified"] == True
        assert mapped["name"] == "Keycloak User"
        assert mapped["given_name"] == "Keycloak"
        assert mapped["family_name"] == "User"
        assert mapped["roles"] == ["admin", "user"]
        assert mapped["groups"] == ["group1", "group2"]
    
    def test_map_user_attributes_auth0(self, sso_service):
        """Test mapping Auth0 user attributes."""
        user_info = {
            "sub": "auth0|123456",
            "email": "user@auth0.com",
            "email_verified": True,
            "name": "Auth0 User",
            "https://your-namespace/roles": ["viewer"],
            "https://your-namespace/permissions": ["read:data"]
        }
        
        mapped = sso_service._map_user_attributes("auth0", user_info)
        
        assert mapped["sub"] == "auth0|123456"
        assert mapped["email"] == "user@auth0.com"
        assert mapped["roles"] == ["viewer"]
        assert mapped["permissions"] == ["read:data"]
    
    def test_map_user_attributes_azure_ad(self, sso_service):
        """Test mapping Azure AD user attributes."""
        user_info = {
            "id": "azure-user-id",
            "mail": "user@microsoft.com",
            "displayName": "Azure User",
            "roles": ["contributor"],
            "groups": ["azure-group"],
            "tid": "tenant-123"
        }
        
        mapped = sso_service._map_user_attributes("azure_ad", user_info)
        
        assert mapped["sub"] == "azure-user-id"
        assert mapped["email"] == "user@microsoft.com"
        assert mapped["name"] == "Azure User"
        assert mapped["roles"] == ["contributor"]
        assert mapped["groups"] == ["azure-group"]
        assert mapped["tenant_id"] == "tenant-123"
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, sso_service):
        """Test refreshing access token."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            result = await sso_service.refresh_token("keycloak", "old-refresh-token")
            
            assert result["access_token"] == "new-access-token"
            assert result["refresh_token"] == "new-refresh-token"
    
    @pytest.mark.asyncio
    async def test_logout(self, sso_service, mock_redis):
        """Test logout functionality."""
        session_id = "test-session"
        session_data = {
            "provider": "keycloak",
            "user_id": "user123",
            "access_token": "test-token"
        }
        mock_redis.get.return_value = json.dumps(session_data).encode()
        
        logout_url = await sso_service.logout(
            provider="keycloak",
            session_id=session_id,
            redirect_uri="http://localhost:3000"
        )
        
        assert "http://localhost:8080/realms/master/protocol/openid-connect/logout" in logout_url
        assert "redirect_uri=http%3A%2F%2Flocalhost%3A3000" in logout_url
        
        # Verify session was deleted
        mock_redis.delete.assert_called_with(f"sso:session:{session_id}")
    
    def test_get_session_exists(self, sso_service, mock_redis):
        """Test getting existing session."""
        session_data = {
            "provider": "keycloak",
            "user_id": "user123",
            "email": "user@example.com"
        }
        mock_redis.get.return_value = json.dumps(session_data).encode()
        
        session = sso_service.get_session("test-session")
        
        assert session is not None
        assert session["user_id"] == "user123"
        assert session["email"] == "user@example.com"
    
    def test_get_session_not_exists(self, sso_service, mock_redis):
        """Test getting non-existent session."""
        mock_redis.get.return_value = None
        
        session = sso_service.get_session("non-existent")
        
        assert session is None
    
    def test_validate_session_valid(self, sso_service, mock_redis):
        """Test validating a valid session."""
        future_time = datetime.utcnow() + timedelta(hours=1)
        session_data = {
            "expires_at": future_time.isoformat()
        }
        mock_redis.get.return_value = json.dumps(session_data).encode()
        
        is_valid = sso_service.validate_session("test-session")
        
        assert is_valid == True
    
    def test_validate_session_expired(self, sso_service, mock_redis):
        """Test validating an expired session."""
        past_time = datetime.utcnow() - timedelta(hours=1)
        session_data = {
            "expires_at": past_time.isoformat()
        }
        mock_redis.get.return_value = json.dumps(session_data).encode()
        
        is_valid = sso_service.validate_session("test-session")
        
        assert is_valid == False
        # Verify expired session was deleted
        mock_redis.delete.assert_called_with("sso:session:test-session")
    
    def test_validate_session_not_exists(self, sso_service, mock_redis):
        """Test validating non-existent session."""
        mock_redis.get.return_value = None
        
        is_valid = sso_service.validate_session("non-existent")
        
        assert is_valid == False
    
    @pytest.mark.asyncio
    async def test_introspect_token_keycloak(self, sso_service):
        """Test token introspection for Keycloak."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "active": True,
            "sub": "user123",
            "email": "user@example.com"
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client_class.return_value = mock_client
            
            result = await sso_service.introspect_token("keycloak", "test-token")
            
            assert result["active"] == True
            assert result["sub"] == "user123"


class TestSAMLService:
    """Test suite for SAMLService."""
    
    def test_saml_service_initialization_with_library(self):
        """Test SAML service initialization with library available."""
        with patch('automl_platform.sso_service.Saml2Client'):
            service = SAMLService()
            assert service.saml2_available == True
    
    def test_saml_service_initialization_without_library(self):
        """Test SAML service initialization without library."""
        with patch('builtins.__import__', side_effect=ImportError):
            service = SAMLService()
            assert service.saml2_available == False
    
    def test_create_saml_client_not_available(self):
        """Test creating SAML client when library not available."""
        from fastapi import HTTPException
        
        service = SAMLService()
        service.saml2_available = False
        
        with pytest.raises(HTTPException) as exc_info:
            service.create_saml_client({})
        
        assert exc_info.value.status_code == 501
        assert "SAML support not available" in str(exc_info.value.detail)
    
    def test_create_auth_request(self):
        """Test creating SAML authentication request."""
        service = SAMLService()
        service.saml2_available = True
        service.BINDING_HTTP_REDIRECT = "redirect"
        
        mock_client = Mock()
        mock_client.prepare_for_authenticate.return_value = (
            "session123",
            {"https://idp.example.com/sso": "https://idp.example.com/sso?SAMLRequest=..."}
        )
        
        session_id, redirect_url = service.create_auth_request(
            mock_client,
            "https://idp.example.com/sso"
        )
        
        assert session_id == "session123"
        assert redirect_url == "https://idp.example.com/sso?SAMLRequest=..."
    
    def test_process_saml_response(self):
        """Test processing SAML response."""
        service = SAMLService()
        service.saml2_available = True
        service.BINDING_HTTP_POST = "post"
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.name_id.text = "user@example.com"
        mock_response.ava = {"email": ["user@example.com"], "name": ["Test User"]}
        mock_response.session_index = "idx123"
        
        mock_client.parse_authn_request_response.return_value = mock_response
        
        user_info = service.process_response(mock_client, "saml-response-data")
        
        assert user_info["sub"] == "user@example.com"
        assert user_info["attributes"]["email"] == ["user@example.com"]
        assert user_info["session_index"] == "idx123"


class TestSSOServiceIntegration:
    """Integration tests for SSO service."""
    
    @pytest.mark.asyncio
    async def test_full_oauth_flow(self):
        """Test complete OAuth flow from authorization to session."""
        mock_redis = Mock()
        mock_redis.setex = Mock()
        mock_redis.get = Mock()
        mock_redis.delete = Mock()
        
        with patch.dict('os.environ', {
            'KEYCLOAK_ENABLED': 'true',
            'KEYCLOAK_CLIENT_ID': 'test-client',
            'KEYCLOAK_CLIENT_SECRET': 'test-secret',
            'KEYCLOAK_URL': 'http://localhost:8080',
            'KEYCLOAK_REALM': 'master'
        }):
            service = SSOService(redis_client=mock_redis)
            
            # Step 1: Get authorization URL
            auth_result = await service.get_authorization_url(
                provider="keycloak",
                redirect_uri="http://localhost:3000/callback"
            )
            
            assert "authorization_url" in auth_result
            assert "state" in auth_result
            
            state = auth_result["state"]
            
            # Step 2: Simulate callback
            state_data = {
                "provider": "keycloak",
                "redirect_uri": "http://localhost:3000/callback",
                "created_at": datetime.utcnow().isoformat()
            }
            mock_redis.get.return_value = json.dumps(state_data).encode()
            
            # Mock successful token exchange
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                
                # Token response
                mock_client.post.return_value = Mock(
                    status_code=200,
                    json=Mock(return_value={
                        "access_token": "access-token",
                        "refresh_token": "refresh-token",
                        "id_token": "id-token",
                        "expires_in": 3600
                    })
                )
                
                # Userinfo response
                mock_client.get.return_value = Mock(
                    status_code=200,
                    json=Mock(return_value={
                        "sub": "user123",
                        "email": "user@example.com",
                        "name": "Test User"
                    })
                )
                
                mock_client.__aenter__.return_value = mock_client
                mock_client_class.return_value = mock_client
                
                with patch.object(service, '_validate_id_token', new_callable=AsyncMock) as mock_validate:
                    mock_validate.return_value = {"sub": "user123"}
                    
                    callback_result = await service.handle_callback(
                        provider="keycloak",
                        code="auth-code",
                        state=state,
                        redirect_uri="http://localhost:3000/callback"
                    )
            
            assert "session_id" in callback_result
            assert callback_result["user"]["sub"] == "user123"
            
            # Step 3: Validate session
            session_id = callback_result["session_id"]
            
            # Mock session retrieval
            future_time = datetime.utcnow() + timedelta(hours=1)
            session_data = {
                "expires_at": future_time.isoformat(),
                "user_id": "user123"
            }
            mock_redis.get.return_value = json.dumps(session_data).encode()
            
            is_valid = service.validate_session(session_id)
            assert is_valid == True
    
    @pytest.mark.asyncio
    async def test_multi_provider_configuration(self):
        """Test service with multiple providers configured."""
        mock_redis = Mock()
        
        with patch.dict('os.environ', {
            'KEYCLOAK_ENABLED': 'true',
            'KEYCLOAK_CLIENT_ID': 'kc-client',
            'KEYCLOAK_CLIENT_SECRET': 'kc-secret',
            'KEYCLOAK_URL': 'http://localhost:8080',
            'KEYCLOAK_REALM': 'master',
            'AUTH0_ENABLED': 'true',
            'AUTH0_CLIENT_ID': 'a0-client',
            'AUTH0_CLIENT_SECRET': 'a0-secret',
            'AUTH0_DOMAIN': 'test.auth0.com'
        }):
            service = SSOService(redis_client=mock_redis)
            
            # Check both providers are configured
            assert "keycloak" in service.providers
            assert "auth0" in service.providers
            
            # Test authorization URL for each provider
            kc_result = await service.get_authorization_url(
                provider="keycloak",
                redirect_uri="http://localhost:3000/callback"
            )
            assert "8080" in kc_result["authorization_url"]
            
            a0_result = await service.get_authorization_url(
                provider="auth0",
                redirect_uri="http://localhost:3000/callback"
            )
            assert "auth0.com" in a0_result["authorization_url"]
