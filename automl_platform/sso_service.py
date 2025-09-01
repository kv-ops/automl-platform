"""
Enterprise SSO Service with Keycloak and Auth0 Support
=======================================================
Place in: automl_platform/sso_service.py

Implements SSO integration with multiple providers, session management,
and multi-tenant support.
"""

import os
import jwt
import json
import logging
import httpx
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import base64
import secrets
from urllib.parse import urlencode, quote

import redis
from fastapi import HTTPException, status
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.jose import JsonWebKey, jwt as authlib_jwt

logger = logging.getLogger(__name__)


class SSOProvider(Enum):
    """Supported SSO providers"""
    KEYCLOAK = "keycloak"
    AUTH0 = "auth0"
    OKTA = "okta"
    AZURE_AD = "azure_ad"
    GOOGLE = "google"
    SAML = "saml"


@dataclass
class SSOConfig:
    """SSO provider configuration"""
    provider: SSOProvider
    client_id: str
    client_secret: str
    domain: str = None  # For Auth0
    realm: str = None  # For Keycloak
    tenant_id: str = None  # For Azure AD
    
    # URLs
    authorize_url: str = None
    token_url: str = None
    userinfo_url: str = None
    jwks_url: str = None
    logout_url: str = None
    
    # SAML specific
    saml_metadata_url: str = None
    saml_entity_id: str = None
    
    # Options
    scope: str = "openid profile email"
    response_type: str = "code"
    grant_type: str = "authorization_code"
    
    def __post_init__(self):
        """Build URLs based on provider"""
        if self.provider == SSOProvider.KEYCLOAK:
            base_url = f"{self.domain}/realms/{self.realm}"
            self.authorize_url = f"{base_url}/protocol/openid-connect/auth"
            self.token_url = f"{base_url}/protocol/openid-connect/token"
            self.userinfo_url = f"{base_url}/protocol/openid-connect/userinfo"
            self.jwks_url = f"{base_url}/protocol/openid-connect/certs"
            self.logout_url = f"{base_url}/protocol/openid-connect/logout"
            
        elif self.provider == SSOProvider.AUTH0:
            self.authorize_url = f"https://{self.domain}/authorize"
            self.token_url = f"https://{self.domain}/oauth/token"
            self.userinfo_url = f"https://{self.domain}/userinfo"
            self.jwks_url = f"https://{self.domain}/.well-known/jwks.json"
            self.logout_url = f"https://{self.domain}/v2/logout"
            
        elif self.provider == SSOProvider.OKTA:
            self.authorize_url = f"https://{self.domain}/oauth2/default/v1/authorize"
            self.token_url = f"https://{self.domain}/oauth2/default/v1/token"
            self.userinfo_url = f"https://{self.domain}/oauth2/default/v1/userinfo"
            self.jwks_url = f"https://{self.domain}/oauth2/default/v1/keys"
            self.logout_url = f"https://{self.domain}/oauth2/default/v1/logout"
            
        elif self.provider == SSOProvider.AZURE_AD:
            base_url = f"https://login.microsoftonline.com/{self.tenant_id}"
            self.authorize_url = f"{base_url}/oauth2/v2.0/authorize"
            self.token_url = f"{base_url}/oauth2/v2.0/token"
            self.userinfo_url = "https://graph.microsoft.com/v1.0/me"
            self.jwks_url = f"{base_url}/discovery/v2.0/keys"
            self.logout_url = f"{base_url}/oauth2/v2.0/logout"


class SSOService:
    """
    Unified SSO service supporting multiple providers
    """
    
    def __init__(self, redis_client: redis.Redis = None):
        self.redis_client = redis_client or redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        self.providers: Dict[str, SSOConfig] = {}
        self.oauth_clients: Dict[str, AsyncOAuth2Client] = {}
        self._init_providers()
    
    def _init_providers(self):
        """Initialize configured SSO providers"""
        
        # Keycloak
        if os.getenv("KEYCLOAK_ENABLED", "false").lower() == "true":
            self.providers["keycloak"] = SSOConfig(
                provider=SSOProvider.KEYCLOAK,
                client_id=os.getenv("KEYCLOAK_CLIENT_ID"),
                client_secret=os.getenv("KEYCLOAK_CLIENT_SECRET"),
                domain=os.getenv("KEYCLOAK_URL", "http://localhost:8080"),
                realm=os.getenv("KEYCLOAK_REALM", "master")
            )
        
        # Auth0
        if os.getenv("AUTH0_ENABLED", "false").lower() == "true":
            self.providers["auth0"] = SSOConfig(
                provider=SSOProvider.AUTH0,
                client_id=os.getenv("AUTH0_CLIENT_ID"),
                client_secret=os.getenv("AUTH0_CLIENT_SECRET"),
                domain=os.getenv("AUTH0_DOMAIN")
            )
        
        # Initialize OAuth clients
        for name, config in self.providers.items():
            self.oauth_clients[name] = AsyncOAuth2Client(
                client_id=config.client_id,
                client_secret=config.client_secret,
                scope=config.scope
            )
    
    async def get_authorization_url(
        self,
        provider: str,
        redirect_uri: str,
        state: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get authorization URL for SSO login
        
        Returns:
            Dictionary with authorization URL and state
        """
        if provider not in self.providers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"SSO provider {provider} not configured"
            )
        
        config = self.providers[provider]
        client = self.oauth_clients[provider]
        
        # Generate state for CSRF protection
        if not state:
            state = secrets.token_urlsafe(32)
        
        # Store state in Redis with TTL
        state_data = {
            "provider": provider,
            "redirect_uri": redirect_uri,
            "created_at": datetime.utcnow().isoformat()
        }
        self.redis_client.setex(
            f"sso:state:{state}",
            300,  # 5 minutes TTL
            json.dumps(state_data)
        )
        
        # Build authorization URL
        params = {
            "client_id": config.client_id,
            "redirect_uri": redirect_uri,
            "response_type": config.response_type,
            "scope": config.scope,
            "state": state
        }
        
        # Provider-specific parameters
        if config.provider == SSOProvider.AUTH0:
            params["audience"] = os.getenv("AUTH0_AUDIENCE", "")
        elif config.provider == SSOProvider.AZURE_AD:
            params["response_mode"] = "query"
        
        auth_url = f"{config.authorize_url}?{urlencode(params)}"
        
        return {
            "authorization_url": auth_url,
            "state": state
        }
    
    async def handle_callback(
        self,
        provider: str,
        code: str,
        state: str,
        redirect_uri: str
    ) -> Dict[str, Any]:
        """
        Handle OAuth callback and exchange code for tokens
        
        Returns:
            User information and tokens
        """
        # Verify state
        state_key = f"sso:state:{state}"
        state_data = self.redis_client.get(state_key)
        
        if not state_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired state"
            )
        
        # Delete state to prevent replay
        self.redis_client.delete(state_key)
        
        config = self.providers[provider]
        
        # Exchange code for tokens
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                config.token_url,
                data={
                    "grant_type": config.grant_type,
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": config.client_id,
                    "client_secret": config.client_secret
                }
            )
            
            if token_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange code for tokens"
                )
            
            tokens = token_response.json()
            
            # Get user info
            userinfo_response = await client.get(
                config.userinfo_url,
                headers={"Authorization": f"Bearer {tokens['access_token']}"}
            )
            
            if userinfo_response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user information"
                )
            
            user_info = userinfo_response.json()
        
        # Validate ID token if present
        if "id_token" in tokens:
            claims = await self._validate_id_token(provider, tokens["id_token"])
            user_info.update(claims)
        
        # Map user attributes based on provider
        user_data = self._map_user_attributes(provider, user_info)
        
        # Store session
        session_id = secrets.token_urlsafe(32)
        session_data = {
            "provider": provider,
            "user_id": user_data["sub"],
            "email": user_data["email"],
            "name": user_data.get("name"),
            "roles": user_data.get("roles", []),
            "groups": user_data.get("groups", []),
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token"),
            "expires_at": (
                datetime.utcnow() + timedelta(seconds=tokens.get("expires_in", 3600))
            ).isoformat()
        }
        
        # Store session with TTL
        self.redis_client.setex(
            f"sso:session:{session_id}",
            tokens.get("expires_in", 3600),
            json.dumps(session_data)
        )
        
        return {
            "session_id": session_id,
            "user": user_data,
            "tokens": {
                "access_token": tokens["access_token"],
                "refresh_token": tokens.get("refresh_token"),
                "expires_in": tokens.get("expires_in", 3600)
            }
        }
    
    async def _validate_id_token(self, provider: str, id_token: str) -> Dict:
        """Validate and decode ID token"""
        config = self.providers[provider]
        
        # Get JWKS
        async with httpx.AsyncClient() as client:
            jwks_response = await client.get(config.jwks_url)
            jwks = jwks_response.json()
        
        # Decode token header to get kid
        header = jwt.get_unverified_header(id_token)
        kid = header.get("kid")
        
        # Find matching key
        key = None
        for jwk in jwks.get("keys", []):
            if jwk.get("kid") == kid:
                key = jwk
                break
        
        if not key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to find matching key for token"
            )
        
        # Verify and decode token
        try:
            # Convert JWK to PEM
            public_key = JsonWebKey.import_key(key)
            
            # Decode and verify
            claims = authlib_jwt.decode(
                id_token,
                public_key,
                claims_options={
                    "iss": {"essential": True, "value": config.authorize_url},
                    "aud": {"essential": True, "value": config.client_id}
                }
            )
            
            return claims
            
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid ID token"
            )
    
    def _map_user_attributes(self, provider: str, user_info: Dict) -> Dict:
        """Map provider-specific attributes to standard format"""
        
        # Standard attributes
        mapped = {
            "sub": user_info.get("sub") or user_info.get("id") or user_info.get("user_id"),
            "email": user_info.get("email") or user_info.get("mail"),
            "email_verified": user_info.get("email_verified", False),
            "name": user_info.get("name") or user_info.get("displayName"),
            "given_name": user_info.get("given_name") or user_info.get("firstName"),
            "family_name": user_info.get("family_name") or user_info.get("lastName"),
            "picture": user_info.get("picture") or user_info.get("avatar_url"),
            "locale": user_info.get("locale"),
            "updated_at": user_info.get("updated_at")
        }
        
        # Provider-specific mappings
        if provider == "keycloak":
            mapped["roles"] = user_info.get("realm_access", {}).get("roles", [])
            mapped["groups"] = user_info.get("groups", [])
            
        elif provider == "auth0":
            mapped["roles"] = user_info.get("https://your-namespace/roles", [])
            mapped["permissions"] = user_info.get("https://your-namespace/permissions", [])
            
        elif provider == "azure_ad":
            mapped["roles"] = user_info.get("roles", [])
            mapped["groups"] = user_info.get("groups", [])
            mapped["tenant_id"] = user_info.get("tid")
        
        return mapped
    
    async def refresh_token(self, provider: str, refresh_token: str) -> Dict:
        """Refresh access token"""
        config = self.providers[provider]
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                config.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": config.client_id,
                    "client_secret": config.client_secret
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to refresh token"
                )
            
            return response.json()
    
    async def logout(self, provider: str, session_id: str, redirect_uri: str = None) -> str:
        """
        Logout user from SSO provider
        
        Returns:
            Logout URL for provider
        """
        config = self.providers[provider]
        
        # Get session
        session_key = f"sso:session:{session_id}"
        session_data = self.redis_client.get(session_key)
        
        if session_data:
            session = json.loads(session_data)
            
            # Revoke tokens if supported
            if provider == "auth0":
                await self._revoke_auth0_token(session["access_token"])
            
            # Delete session
            self.redis_client.delete(session_key)
        
        # Build logout URL
        logout_params = {}
        
        if provider == "keycloak":
            logout_params["redirect_uri"] = redirect_uri
        elif provider == "auth0":
            logout_params["returnTo"] = redirect_uri
            logout_params["client_id"] = config.client_id
        elif provider == "okta":
            logout_params["post_logout_redirect_uri"] = redirect_uri
        
        logout_url = f"{config.logout_url}?{urlencode(logout_params)}"
        
        return logout_url
    
    async def _revoke_auth0_token(self, token: str):
        """Revoke Auth0 token"""
        config = self.providers["auth0"]
        
        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://{config.domain}/oauth/revoke",
                data={
                    "client_id": config.client_id,
                    "client_secret": config.client_secret,
                    "token": token
                }
            )
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data"""
        session_key = f"sso:session:{session_id}"
        session_data = self.redis_client.get(session_key)
        
        if session_data:
            return json.loads(session_data)
        
        return None
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if session is active"""
        session = self.get_session(session_id)
        
        if not session:
            return False
        
        # Check expiration
        expires_at = datetime.fromisoformat(session["expires_at"])
        if expires_at < datetime.utcnow():
            # Session expired, delete it
            self.redis_client.delete(f"sso:session:{session_id}")
            return False
        
        return True
    
    async def introspect_token(self, provider: str, token: str) -> Dict:
        """Introspect token with provider"""
        config = self.providers[provider]
        
        # Build introspection URL
        introspect_url = None
        if provider == "keycloak":
            introspect_url = config.token_url.replace("/token", "/introspect")
        elif provider == "auth0":
            # Auth0 doesn't have standard introspection, use userinfo
            return await self._get_userinfo_with_token(provider, token)
        
        if not introspect_url:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Token introspection not supported for {provider}"
            )
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                introspect_url,
                data={
                    "token": token,
                    "client_id": config.client_id,
                    "client_secret": config.client_secret
                }
            )
            
            if response.status_code != 200:
                return {"active": False}
            
            return response.json()
    
    async def _get_userinfo_with_token(self, provider: str, token: str) -> Dict:
        """Get user info with access token"""
        config = self.providers[provider]
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                config.userinfo_url,
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code != 200:
                return {"active": False}
            
            user_info = response.json()
            return {
                "active": True,
                "sub": user_info.get("sub"),
                "email": user_info.get("email"),
                **user_info
            }


class SAMLService:
    """SAML 2.0 SSO service"""
    
    def __init__(self):
        try:
            from saml2 import BINDING_HTTP_POST, BINDING_HTTP_REDIRECT
            from saml2.client import Saml2Client
            from saml2.config import Config as Saml2Config
            
            self.saml2_available = True
            self.BINDING_HTTP_POST = BINDING_HTTP_POST
            self.BINDING_HTTP_REDIRECT = BINDING_HTTP_REDIRECT
            self.Saml2Client = Saml2Client
            self.Saml2Config = Saml2Config
        except ImportError:
            self.saml2_available = False
            logger.warning("python3-saml not installed. SAML support disabled.")
    
    def create_saml_client(self, config: Dict) -> Any:
        """Create SAML client"""
        if not self.saml2_available:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="SAML support not available"
            )
        
        saml_settings = self.Saml2Config()
        saml_settings.load(config)
        return self.Saml2Client(config=saml_settings)
    
    def create_auth_request(self, client: Any, sso_url: str) -> Tuple[str, str]:
        """Create SAML authentication request"""
        session_id, request_info = client.prepare_for_authenticate(
            relay_state="",
            binding=self.BINDING_HTTP_REDIRECT
        )
        
        redirect_url = None
        for key, value in request_info.items():
            if sso_url in key:
                redirect_url = value
                break
        
        return session_id, redirect_url
    
    def process_response(self, client: Any, saml_response: str) -> Dict:
        """Process SAML response"""
        authn_response = client.parse_authn_request_response(
            saml_response,
            self.BINDING_HTTP_POST
        )
        
        user_info = {
            "sub": authn_response.name_id.text,
            "attributes": authn_response.ava,
            "session_index": authn_response.session_index
        }
        
        return user_info
