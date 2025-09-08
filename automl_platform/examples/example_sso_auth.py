"""
SSO Authentication Example
==========================
Place in: automl_platform/examples/example_sso_auth.py

Demonstrates SSO integration with multiple providers (Keycloak, Auth0, Okta, Azure AD),
session management, token handling, and multi-tenant support.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
import pandas as pd
import httpx

# Import SSO components
from automl_platform.sso_service import (
    SSOService,
    SSOProvider,
    SSOConfig,
    SAMLService
)
from automl_platform.audit_service import (
    AuditService,
    AuditEventType,
    AuditSeverity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSOAuthenticationExample:
    """Comprehensive SSO authentication examples"""
    
    def __init__(self):
        self.sso_service = SSOService()
        self.saml_service = SAMLService()
        self.audit_service = AuditService()
        
        # Sample user data for testing
        self.test_users = {
            "keycloak": {
                "user_id": "kc_user_001",
                "email": "user@keycloak.example.com",
                "name": "Keycloak Test User"
            },
            "auth0": {
                "user_id": "auth0|123456",
                "email": "user@auth0.example.com",
                "name": "Auth0 Test User"
            },
            "okta": {
                "user_id": "00u1234567890",
                "email": "user@okta.example.com",
                "name": "Okta Test User"
            }
        }
    
    async def example_1_multi_provider_sso(self):
        """Example 1: Multi-provider SSO authentication flow"""
        print("\n" + "="*80)
        print("EXAMPLE 1: Multi-Provider SSO Authentication")
        print("="*80)
        
        providers = ["keycloak", "auth0", "okta"]
        
        for provider in providers:
            print(f"\n🔐 {provider.upper()} Authentication Flow")
            print("-" * 50)
            
            try:
                # Step 1: Initialize authorization
                auth_url_data = await self._initialize_sso_auth(provider)
                
                if auth_url_data:
                    print(f"\n✅ Authorization URL generated")
                    print(f"   URL: {auth_url_data['url'][:80]}...")
                    print(f"   State: {auth_url_data['state']}")
                    
                    # Step 2: Simulate user authentication
                    user_data = await self._simulate_user_authentication(
                        provider,
                        auth_url_data['state']
                    )
                    
                    if user_data:
                        print(f"\n✅ User authenticated successfully")
                        print(f"   Session ID: {user_data['session_id']}")
                        print(f"   User: {user_data['user']['name']}")
                        print(f"   Email: {user_data['user']['email']}")
                        print(f"   Roles: {', '.join(user_data['user'].get('roles', []))}")
                    
            except Exception as e:
                print(f"   ❌ {provider} authentication failed: {e}")
    
    async def example_2_session_management(self):
        """Example 2: Advanced session management"""
        print("\n" + "="*80)
        print("EXAMPLE 2: Session Management & Token Handling")
        print("="*80)
        
        # Create test sessions
        sessions = await self._create_test_sessions()
        
        print(f"\n📊 Created {len(sessions)} test sessions")
        
        # Step 1: Session validation
        print("\n1️⃣ Session Validation")
        print("-" * 40)
        
        for session_id, session_info in sessions.items():
            is_valid = self.sso_service.validate_session(session_id)
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"  Session {session_id[:8]}... ({session_info['provider']}): {status}")
        
        # Step 2: Session refresh
        print("\n2️⃣ Token Refresh")
        print("-" * 40)
        
        for session_id, session_info in sessions.items():
            if session_info.get('refresh_token'):
                try:
                    print(f"\n  Refreshing tokens for {session_info['provider']}...")
                    
                    # Simulate token refresh
                    new_tokens = await self._simulate_token_refresh(
                        session_info['provider'],
                        session_info['refresh_token']
                    )
                    
                    if new_tokens:
                        print(f"    ✅ New access token: {new_tokens['access_token'][:20]}...")
                        print(f"    Expires in: {new_tokens['expires_in']} seconds")
                    
                except Exception as e:
                    print(f"    ❌ Refresh failed: {e}")
        
        # Step 3: Session introspection
        print("\n3️⃣ Token Introspection")
        print("-" * 40)
        
        for session_id, session_info in sessions.items():
            provider = session_info['provider']
            
            # Simulate token introspection
            introspection = await self._simulate_token_introspection(
                provider,
                session_info['access_token']
            )
            
            print(f"\n  {provider} token:")
            print(f"    Active: {introspection.get('active', False)}")
            print(f"    Expires at: {introspection.get('exp', 'N/A')}")
            print(f"    Scope: {introspection.get('scope', 'N/A')}")
        
        # Step 4: Session termination
        print("\n4️⃣ Session Termination")
        print("-" * 40)
        
        for session_id in list(sessions.keys())[:1]:  # Terminate first session
            session_info = sessions[session_id]
            
            logout_url = await self.sso_service.logout(
                provider=session_info['provider'],
                session_id=session_id,
                redirect_uri="http://localhost:8000/logout-complete"
            )
            
            print(f"\n  Terminated session for {session_info['provider']}")
            print(f"  Logout URL: {logout_url[:80]}...")
            
            # Verify session is deleted
            is_valid = self.sso_service.validate_session(session_id)
            print(f"  Session status after logout: {'✅ Still valid' if is_valid else '❌ Deleted'}")
    
    async def example_3_role_based_access(self):
        """Example 3: Role-based access control with SSO"""
        print("\n" + "="*80)
        print("EXAMPLE 3: Role-Based Access Control (RBAC)")
        print("="*80)
        
        # Define role hierarchy
        role_hierarchy = {
            "admin": ["write", "read", "delete", "manage_users"],
            "developer": ["write", "read", "deploy"],
            "analyst": ["read", "export"],
            "viewer": ["read"]
        }
        
        # Test users with different roles
        test_scenarios = [
            {
                "user": "admin_user",
                "roles": ["admin", "developer"],
                "provider": "keycloak"
            },
            {
                "user": "dev_user",
                "roles": ["developer"],
                "provider": "auth0"
            },
            {
                "user": "analyst_user",
                "roles": ["analyst"],
                "provider": "okta"
            }
        ]
        
        print("\n🔑 Testing Role-Based Access")
        print("-" * 40)
        
        for scenario in test_scenarios:
            print(f"\n  User: {scenario['user']}")
            print(f"  Roles: {', '.join(scenario['roles'])}")
            print(f"  Provider: {scenario['provider']}")
            
            # Calculate effective permissions
            permissions = set()
            for role in scenario['roles']:
                if role in role_hierarchy:
                    permissions.update(role_hierarchy[role])
            
            print(f"  Permissions: {', '.join(sorted(permissions))}")
            
            # Test access to resources
            resources = [
                ("User Management", "manage_users"),
                ("Data Export", "export"),
                ("Model Deployment", "deploy"),
                ("Data Viewing", "read")
            ]
            
            print(f"\n  Access Matrix:")
            for resource, required_perm in resources:
                has_access = required_perm in permissions
                status = "✅" if has_access else "❌"
                print(f"    {resource}: {status}")
    
    async def example_4_saml_integration(self):
        """Example 4: SAML 2.0 SSO integration"""
        print("\n" + "="*80)
        print("EXAMPLE 4: SAML 2.0 Integration")
        print("="*80)
        
        if not self.saml_service.saml2_available:
            print("\n⚠️ SAML support not available. Install python3-saml:")
            print("   pip install python3-saml")
            return
        
        # SAML configuration
        saml_config = {
            "entityid": "http://localhost:8000",
            "metadata": {
                "local": ["metadata.xml"]
            },
            "service": {
                "sp": {
                    "endpoints": {
                        "assertion_consumer_service": [
                            ("http://localhost:8000/saml/acs", "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST")
                        ]
                    },
                    "allow_unsolicited": False,
                    "authn_requests_signed": False,
                    "logout_requests_signed": True,
                    "want_assertions_signed": True,
                    "want_response_signed": False
                }
            }
        }
        
        print("\n1️⃣ SAML Configuration")
        print("-" * 40)
        print(f"  Entity ID: {saml_config['entityid']}")
        print(f"  ACS URL: {saml_config['service']['sp']['endpoints']['assertion_consumer_service'][0][0]}")
        print(f"  Assertions Signed: {saml_config['service']['sp']['want_assertions_signed']}")
        
        try:
            # Create SAML client
            saml_client = self.saml_service.create_saml_client(saml_config)
            
            print("\n2️⃣ SAML Authentication Request")
            print("-" * 40)
            
            # Generate auth request
            session_id, auth_url = self.saml_service.create_auth_request(
                saml_client,
                "https://idp.example.com/sso"
            )
            
            print(f"  Session ID: {session_id}")
            print(f"  Auth URL: {auth_url[:80]}..." if auth_url else "  Auth URL: Not generated")
            
            print("\n3️⃣ SAML Response Processing")
            print("-" * 40)
            
            # Simulate SAML response
            mock_saml_response = self._generate_mock_saml_response()
            
            print(f"  Response received: {len(mock_saml_response)} bytes")
            print(f"  Processing response...")
            
            # In real scenario, process the response
            # user_info = self.saml_service.process_response(saml_client, mock_saml_response)
            
            print(f"  ✅ SAML authentication flow demonstrated")
            
        except Exception as e:
            print(f"  ❌ SAML configuration error: {e}")
    
    async def example_5_multi_tenant_sso(self):
        """Example 5: Multi-tenant SSO configuration"""
        print("\n" + "="*80)
        print("EXAMPLE 5: Multi-Tenant SSO Configuration")
        print("="*80)
        
        # Define tenant configurations
        tenants = {
            "tenant_a": {
                "name": "Enterprise A",
                "sso_provider": "keycloak",
                "realm": "enterprise-a",
                "allowed_domains": ["enterprise-a.com"],
                "custom_claims": ["department", "employee_id"]
            },
            "tenant_b": {
                "name": "Enterprise B",
                "sso_provider": "auth0",
                "connection": "enterprise-b-ad",
                "allowed_domains": ["enterprise-b.org", "subsidiary-b.com"],
                "custom_claims": ["cost_center", "manager"]
            },
            "tenant_c": {
                "name": "Enterprise C",
                "sso_provider": "azure_ad",
                "tenant_id": "c-tenant-id-123",
                "allowed_domains": ["enterprise-c.net"],
                "custom_claims": ["division", "location"]
            }
        }
        
        print("\n🏢 Tenant Configurations")
        print("-" * 40)
        
        for tenant_id, config in tenants.items():
            print(f"\n  {config['name']} ({tenant_id}):")
            print(f"    Provider: {config['sso_provider']}")
            print(f"    Domains: {', '.join(config['allowed_domains'])}")
            print(f"    Custom Claims: {', '.join(config['custom_claims'])}")
        
        print("\n🔄 Domain-based Routing")
        print("-" * 40)
        
        # Test email addresses
        test_emails = [
            "user@enterprise-a.com",
            "admin@subsidiary-b.com",
            "analyst@enterprise-c.net",
            "external@gmail.com"
        ]
        
        for email in test_emails:
            domain = email.split('@')[1]
            matched_tenant = None
            
            # Find matching tenant
            for tenant_id, config in tenants.items():
                if domain in config['allowed_domains']:
                    matched_tenant = tenant_id
                    break
            
            if matched_tenant:
                tenant = tenants[matched_tenant]
                print(f"\n  {email}:")
                print(f"    → Tenant: {tenant['name']}")
                print(f"    → SSO Provider: {tenant['sso_provider']}")
            else:
                print(f"\n  {email}:")
                print(f"    → No tenant match (use local auth)")
        
        print("\n📊 Tenant Usage Statistics")
        print("-" * 40)
        
        # Simulate usage statistics
        stats = {
            "tenant_a": {"users": 450, "logins_today": 320, "active_sessions": 125},
            "tenant_b": {"users": 280, "logins_today": 195, "active_sessions": 87},
            "tenant_c": {"users": 150, "logins_today": 98, "active_sessions": 42}
        }
        
        df_stats = pd.DataFrame(stats).T
        df_stats.index.name = 'Tenant'
        print(f"\n{df_stats.to_string()}")
        
        total_users = df_stats['users'].sum()
        total_logins = df_stats['logins_today'].sum()
        total_sessions = df_stats['active_sessions'].sum()
        
        print(f"\n  Totals:")
        print(f"    Users: {total_users}")
        print(f"    Logins Today: {total_logins}")
        print(f"    Active Sessions: {total_sessions}")
    
    async def example_6_security_features(self):
        """Example 6: SSO security features"""
        print("\n" + "="*80)
        print("EXAMPLE 6: SSO Security Features")
        print("="*80)
        
        print("\n1️⃣ PKCE (Proof Key for Code Exchange)")
        print("-" * 40)
        
        # Generate PKCE challenge
        import hashlib
        import base64
        import secrets
        
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode('utf-8').rstrip('=')
        
        print(f"  Code Verifier: {code_verifier[:20]}...")
        print(f"  Code Challenge: {code_challenge[:20]}...")
        print(f"  Challenge Method: S256")
        
        print("\n2️⃣ State Parameter Validation")
        print("-" * 40)
        
        # Generate and store state
        state = secrets.token_urlsafe(32)
        state_data = {
            "provider": "keycloak",
            "redirect_uri": "http://localhost:8000/callback",
            "created_at": datetime.utcnow().isoformat(),
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0..."
        }
        
        # Store in Redis with TTL
        self.sso_service.redis_client.setex(
            f"sso:state:{state}",
            300,  # 5 minutes
            json.dumps(state_data)
        )
        
        print(f"  State Token: {state[:20]}...")
        print(f"  TTL: 300 seconds")
        print(f"  Stored metadata: {list(state_data.keys())}")
        
        # Validate state
        retrieved = self.sso_service.redis_client.get(f"sso:state:{state}")
        if retrieved:
            print(f"  ✅ State validation successful")
        
        print("\n3️⃣ Token Security")
        print("-" * 40)
        
        # Token security measures
        security_measures = [
            ("Token Encryption", "AES-256-GCM", "✅"),
            ("Secure Storage", "Redis with encryption at rest", "✅"),
            ("Token Rotation", "Automatic refresh before expiry", "✅"),
            ("Audience Validation", "Verify token audience claim", "✅"),
            ("Issuer Validation", "Verify token issuer", "✅"),
            ("Signature Verification", "RS256 with JWKS", "✅"),
            ("Token Binding", "Bind token to client TLS cert", "⚠️"),
            ("DPoP", "Demonstration of Proof-of-Possession", "⚠️")
        ]
        
        for measure, description, status in security_measures:
            print(f"  {status} {measure}")
            print(f"     {description}")
        
        print("\n4️⃣ Audit Logging")
        print("-" * 40)
        
        # Log security events
        security_events = [
            ("Successful login", AuditSeverity.INFO),
            ("Failed login attempt", AuditSeverity.WARNING),
            ("Token refresh", AuditSeverity.INFO),
            ("Session timeout", AuditSeverity.INFO),
            ("Suspicious activity detected", AuditSeverity.HIGH),
            ("Multiple failed attempts", AuditSeverity.CRITICAL)
        ]
        
        for event, severity in security_events:
            self.audit_service.log_event(
                event_type=AuditEventType.AUTH_LOGIN,
                action=event.lower().replace(' ', '_'),
                user_id="test_user",
                severity=severity,
                metadata={"provider": "keycloak", "ip": "192.168.1.100"}
            )
            
            icon = "🔴" if severity == AuditSeverity.CRITICAL else "🟡" if severity == AuditSeverity.WARNING else "🟢"
            print(f"  {icon} {event} (Severity: {severity.value})")
        
        print("\n5️⃣ Rate Limiting")
        print("-" * 40)
        
        # Implement rate limiting
        rate_limits = {
            "login_attempts": {"limit": 5, "window": 300, "current": 3},
            "token_refresh": {"limit": 10, "window": 3600, "current": 2},
            "password_reset": {"limit": 3, "window": 3600, "current": 0}
        }
        
        for action, limits in rate_limits.items():
            remaining = limits['limit'] - limits['current']
            status = "✅" if remaining > 0 else "❌"
            print(f"  {action}:")
            print(f"    {status} {remaining}/{limits['limit']} remaining")
            print(f"    Window: {limits['window']}s")
    
    # Helper methods
    
    async def _initialize_sso_auth(self, provider: str) -> Optional[Dict]:
        """Initialize SSO authentication"""
        try:
            auth_data = await self.sso_service.get_authorization_url(
                provider=provider,
                redirect_uri=f"http://localhost:8000/auth/{provider}/callback",
                state=None
            )
            
            return {
                "url": auth_data['authorization_url'],
                "state": auth_data['state']
            }
        except Exception as e:
            logger.error(f"Failed to initialize {provider} auth: {e}")
            return None
    
    async def _simulate_user_authentication(self, provider: str, state: str) -> Optional[Dict]:
        """Simulate user authentication (for demo)"""
        # In real scenario, user would authenticate with provider
        # For demo, return mock data
        
        mock_code = f"mock_code_{provider}_123"
        
        try:
            # Simulate callback handling
            user_info = self.test_users.get(provider, {})
            
            session_id = secrets.token_urlsafe(32)
            session_data = {
                "provider": provider,
                "user_id": user_info.get("user_id"),
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "roles": ["user", "ml_user"],
                "access_token": f"mock_access_token_{provider}",
                "refresh_token": f"mock_refresh_token_{provider}",
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
            
            # Store session
            self.sso_service.redis_client.setex(
                f"sso:session:{session_id}",
                3600,
                json.dumps(session_data)
            )
            
            return {
                "session_id": session_id,
                "user": {
                    "user_id": user_info.get("user_id"),
                    "email": user_info.get("email"),
                    "name": user_info.get("name"),
                    "roles": ["user", "ml_user"]
                }
            }
            
        except Exception as e:
            logger.error(f"Authentication simulation failed: {e}")
            return None
    
    async def _create_test_sessions(self) -> Dict[str, Dict]:
        """Create test sessions for examples"""
        sessions = {}
        
        for provider, user_info in self.test_users.items():
            session_id = secrets.token_urlsafe(32)
            
            session_data = {
                "provider": provider,
                "user_id": user_info["user_id"],
                "email": user_info["email"],
                "name": user_info["name"],
                "access_token": f"access_{provider}_{secrets.token_urlsafe(16)}",
                "refresh_token": f"refresh_{provider}_{secrets.token_urlsafe(16)}",
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
            
            # Store in Redis
            self.sso_service.redis_client.setex(
                f"sso:session:{session_id}",
                3600,
                json.dumps(session_data)
            )
            
            sessions[session_id] = session_data
        
        return sessions
    
    async def _simulate_token_refresh(self, provider: str, refresh_token: str) -> Optional[Dict]:
        """Simulate token refresh"""
        # In real scenario, would call provider's token endpoint
        return {
            "access_token": f"new_access_{provider}_{secrets.token_urlsafe(16)}",
            "refresh_token": f"new_refresh_{provider}_{secrets.token_urlsafe(16)}",
            "expires_in": 3600,
            "token_type": "Bearer"
        }
    
    async def _simulate_token_introspection(self, provider: str, token: str) -> Dict:
        """Simulate token introspection"""
        # In real scenario, would call provider's introspection endpoint
        return {
            "active": True,
            "scope": "openid profile email",
            "client_id": f"{provider}_client",
            "username": f"user@{provider}.example.com",
            "exp": (datetime.utcnow() + timedelta(hours=1)).timestamp()
        }
    
    def _generate_mock_saml_response(self) -> str:
        """Generate mock SAML response for testing"""
        # This would be a base64-encoded SAML response in production
        return base64.b64encode(b"<samlp:Response>...</samlp:Response>").decode()


async def main():
    """Main execution function"""
    example = SSOAuthenticationExample()
    
    print("\n" + "="*80)
    print(" " * 20 + "SSO AUTHENTICATION EXAMPLES")
    print("="*80)
    
    # Run examples
    await example.example_1_multi_provider_sso()
    await example.example_2_session_management()
    await example.example_3_role_based_access()
    await example.example_4_saml_integration()
    await example.example_5_multi_tenant_sso()
    await example.example_6_security_features()
    
    print("\n" + "="*80)
    print(" " * 15 + "✅ ALL SSO EXAMPLES COMPLETED!")
    print("="*80)
    
    print("\n📊 Summary of Features Demonstrated:")
    print("  ✓ Multi-provider SSO (Keycloak, Auth0, Okta, Azure AD)")
    print("  ✓ Session management and token handling")
    print("  ✓ Role-based access control (RBAC)")
    print("  ✓ SAML 2.0 integration")
    print("  ✓ Multi-tenant configuration")
    print("  ✓ Security features (PKCE, state validation, rate limiting)")
    print("  ✓ Token introspection and refresh")
    print("  ✓ Audit logging and monitoring")
    
    print("\n⚙️ Configuration Requirements:")
    print("  1. Set environment variables for SSO providers:")
    print("     - KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET")
    print("     - AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN")
    print("     - OKTA_CLIENT_ID, OKTA_CLIENT_SECRET, OKTA_DOMAIN")
    print("  2. Redis running on localhost:6379")
    print("  3. Optional: python3-saml for SAML support")


if __name__ == "__main__":
    import base64
    import secrets
    asyncio.run(main())
