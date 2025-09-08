"""
SSO and RGPD/GDPR Compliance Example
====================================
Place in: automl_platform/examples/sso_rgpd_example.py

Demonstrates SSO integration (Keycloak/Auth0) and GDPR compliance
features including consent management and data subject requests.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import httpx

# Import SSO and RGPD components
from automl_platform.sso_service import (
    SSOService,
    SSOProvider,
    SSOConfig
)
from automl_platform.rgpd_compliance_service import (
    RGPDComplianceService,
    GDPRRequestType,
    ConsentType,
    DataCategory,
    get_rgpd_service
)
from automl_platform.audit_service import (
    AuditService,
    AuditEventType,
    AuditSeverity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSOAndRGPDExample:
    """SSO and RGPD compliance examples"""
    
    def __init__(self):
        self.sso_service = SSOService()
        self.rgpd_service = get_rgpd_service()
        self.audit_service = AuditService()
    
    async def example_1_sso_authentication(self):
        """Example 1: SSO authentication flow"""
        print("\n" + "="*80)
        print("EXAMPLE 1: SSO Authentication Flow")
        print("="*80)
        
        # Simulate SSO login with different providers
        providers = ["keycloak", "auth0"]
        
        for provider in providers:
            print(f"\n🔐 {provider.upper()} Authentication Flow")
            print("-" * 40)
            
            try:
                # Step 1: Get authorization URL
                print(f"\n1️⃣ Getting authorization URL for {provider}...")
                
                auth_data = await self.sso_service.get_authorization_url(
                    provider=provider,
                    redirect_uri="http://localhost:8000/auth/callback",
                    state=None  # Auto-generated
                )
                
                print(f"   Authorization URL: {auth_data['authorization_url'][:80]}...")
                print(f"   State token: {auth_data['state']}")
                
                # Step 2: Simulate OAuth callback
                print(f"\n2️⃣ Simulating OAuth callback...")
                
                # In real scenario, user would authenticate and provider redirects back
                mock_auth_code = "mock_authorization_code_12345"
                
                try:
                    # Handle callback (would normally receive real code)
                    user_data = await self._simulate_callback(
                        provider,
                        mock_auth_code,
                        auth_data['state'],
                        "http://localhost:8000/auth/callback"
                    )
                    
                    print(f"\n3️⃣ User authenticated successfully!")
                    print(f"   User ID: {user_data.get('user_id', 'N/A')}")
                    print(f"   Email: {user_data.get('email', 'N/A')}")
                    print(f"   Session ID: {user_data.get('session_id', 'N/A')}")
                    
                    # Log authentication event
                    self.audit_service.log_event(
                        event_type=AuditEventType.AUTH_LOGIN,
                        action=f"sso_login_{provider}",
                        user_id=user_data.get('user_id', 'unknown'),
                        resource_type="authentication",
                        metadata={
                            "provider": provider,
                            "email": user_data.get('email')
                        }
                    )
                    
                except Exception as e:
                    print(f"   ⚠️ Callback simulation failed: {e}")
                
            except Exception as e:
                print(f"   ❌ {provider} not configured: {e}")
        
        # Step 3: Session management
        print("\n\n4️⃣ Session Management")
        print("-" * 40)
        
        # Create mock session
        session_id = "test_session_123"
        session_data = {
            "provider": "keycloak",
            "user_id": "user_001",
            "email": "test@example.com",
            "roles": ["user", "ml_engineer"],
            "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
        }
        
        # Store session
        self.sso_service.redis_client.setex(
            f"sso:session:{session_id}",
            3600,
            json.dumps(session_data)
        )
        
        # Validate session
        is_valid = self.sso_service.validate_session(session_id)
        print(f"Session validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        # Get session data
        retrieved_session = self.sso_service.get_session(session_id)
        if retrieved_session:
            print(f"Session data retrieved:")
            print(f"  - User: {retrieved_session['user_id']}")
            print(f"  - Roles: {retrieved_session['roles']}")
            print(f"  - Provider: {retrieved_session['provider']}")
    
    def example_2_consent_management(self):
        """Example 2: GDPR consent management"""
        print("\n" + "="*80)
        print("EXAMPLE 2: GDPR Consent Management")
        print("="*80)
        
        user_id = "user_001"
        tenant_id = "tenant_123"
        
        # Step 1: Record various consents
        print("\n1️⃣ Recording User Consents")
        print("-" * 40)
        
        consent_scenarios = [
            {
                "type": ConsentType.DATA_PROCESSING,
                "granted": True,
                "purpose": "Process personal data for ML model training",
                "categories": ["behavioral", "technical"]
            },
            {
                "type": ConsentType.MARKETING,
                "granted": False,
                "purpose": "Send promotional emails",
                "categories": ["contact"]
            },
            {
                "type": ConsentType.ANALYTICS,
                "granted": True,
                "purpose": "Analyze usage patterns to improve service",
                "categories": ["behavioral", "technical"]
            },
            {
                "type": ConsentType.THIRD_PARTY,
                "granted": False,
                "purpose": "Share data with partners",
                "categories": ["basic", "contact"]
            }
        ]
        
        for consent in consent_scenarios:
            consent_id = self.rgpd_service.record_consent(
                user_id=user_id,
                consent_type=consent["type"],
                granted=consent["granted"],
                tenant_id=tenant_id,
                purpose=consent["purpose"],
                data_categories=consent["categories"],
                expires_in_days=365,
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0"
            )
            
            status = "✅ Granted" if consent["granted"] else "❌ Denied"
            print(f"  {consent['type'].value}: {status}")
            print(f"    Purpose: {consent['purpose']}")
            print(f"    Categories: {', '.join(consent['categories'])}")
        
        # Step 2: Check consent status
        print("\n2️⃣ Checking Consent Status")
        print("-" * 40)
        
        for consent_type in ConsentType:
            has_consent = self.rgpd_service.check_consent(user_id, consent_type)
            status = "✅" if has_consent else "❌"
            print(f"  {consent_type.value}: {status}")
        
        # Step 3: Get all user consents
        print("\n3️⃣ Retrieving All User Consents")
        print("-" * 40)
        
        all_consents = self.rgpd_service.get_user_consents(user_id)
        
        print(f"Total consent records: {len(all_consents)}")
        for consent in all_consents:
            print(f"\n  Type: {consent['type']}")
            print(f"  Active: {consent['active']}")
            print(f"  Granted at: {consent.get('granted_at', 'N/A')}")
            print(f"  Expires at: {consent.get('expires_at', 'N/A')}")
        
        # Step 4: Revoke consent
        print("\n4️⃣ Revoking Consent")
        print("-" * 40)
        
        # Revoke analytics consent
        self.rgpd_service.record_consent(
            user_id=user_id,
            consent_type=ConsentType.ANALYTICS,
            granted=False,
            tenant_id=tenant_id
        )
        
        print("  Analytics consent revoked")
        
        # Verify revocation
        has_analytics = self.rgpd_service.check_consent(user_id, ConsentType.ANALYTICS)
        print(f"  Analytics consent after revocation: {'✅' if has_analytics else '❌ Revoked'}")
    
    def example_3_data_subject_requests(self):
        """Example 3: GDPR data subject requests"""
        print("\n" + "="*80)
        print("EXAMPLE 3: GDPR Data Subject Requests")
        print("="*80)
        
        user_id = "user_002"
        tenant_id = "tenant_123"
        
        # Step 1: Data Access Request (Article 15)
        print("\n1️⃣ Data Access Request (Article 15)")
        print("-" * 40)
        
        access_request_id = self.rgpd_service.create_data_request(
            user_id=user_id,
            request_type=GDPRRequestType.ACCESS,
            tenant_id=tenant_id,
            reason="User wants to review all personal data",
            requested_data={"include_ml_predictions": True}
        )
        
        print(f"  Request ID: {access_request_id}")
        print(f"  Status: Pending")
        print(f"  Legal deadline: 30 days")
        
        # Process the request
        print("\n  Processing access request...")
        user_data = self.rgpd_service.process_access_request(access_request_id)
        
        print(f"  ✅ Data package prepared:")
        print(f"    - Personal data records: {len(user_data.get('personal_data', []))}")
        print(f"    - Consents: {len(user_data.get('consents', []))}")
        print(f"    - Processing activities: {len(user_data.get('processing_activities', []))}")
        
        # Step 2: Data Portability Request (Article 20)
        print("\n2️⃣ Data Portability Request (Article 20)")
        print("-" * 40)
        
        portability_request_id = self.rgpd_service.create_data_request(
            user_id=user_id,
            request_type=GDPRRequestType.PORTABILITY,
            tenant_id=tenant_id,
            reason="User switching to competitor service"
        )
        
        print(f"  Request ID: {portability_request_id}")
        
        # Process portability request
        portable_data = self.rgpd_service.process_portability_request(portability_request_id)
        
        print(f"  ✅ Portable data package created:")
        print(f"    - Format: JSON/CSV")
        print(f"    - Size: {len(portable_data)} bytes")
        print(f"    - Machine-readable: Yes")
        
        # Step 3: Rectification Request (Article 16)
        print("\n3️⃣ Data Rectification Request (Article 16)")
        print("-" * 40)
        
        rectification_request_id = self.rgpd_service.create_data_request(
            user_id=user_id,
            request_type=GDPRRequestType.RECTIFICATION,
            tenant_id=tenant_id,
            reason="Incorrect personal information"
        )
        
        corrections = {
            "email": "newemail@example.com",
            "phone": "+1-555-0123",
            "address": "123 New Street, City"
        }
        
        print(f"  Request ID: {rectification_request_id}")
        print(f"  Corrections requested: {list(corrections.keys())}")
        
        rectification_result = self.rgpd_service.process_rectification_request(
            rectification_request_id,
            corrections
        )
        
        print(f"  ✅ Rectification completed:")
        print(f"    - Fields updated: {len(rectification_result['rectified_items'])}")
        
        # Step 4: Erasure Request (Article 17 - Right to be forgotten)
        print("\n4️⃣ Erasure Request (Article 17 - Right to be forgotten)")
        print("-" * 40)
        
        erasure_request_id = self.rgpd_service.create_data_request(
            user_id="user_003",  # Different user for safety
            request_type=GDPRRequestType.ERASURE,
            tenant_id=tenant_id,
            reason="User account closure"
        )
        
        print(f"  Request ID: {erasure_request_id}")
        print(f"  Checking erasure eligibility...")
        
        # Process erasure (with verification)
        erasure_result = self.rgpd_service.process_erasure_request(
            erasure_request_id,
            verify_legal_basis=True
        )
        
        if erasure_result['status'] == 'completed':
            print(f"  ✅ Erasure completed:")
            print(f"    - Personal data: {erasure_result['erased_items']['personal_data']} records")
            print(f"    - Consents: {erasure_result['erased_items']['consents']} records")
            print(f"    - ML models: {erasure_result['erased_items']['ml_models']} references")
            print(f"    - Logs: {erasure_result['erased_items']['logs']} entries")
        else:
            print(f"  ❌ Erasure rejected: {erasure_result.get('reason', 'Unknown')}")
    
    def example_4_data_protection(self):
        """Example 4: Data protection and anonymization"""
        print("\n" + "="*80)
        print("EXAMPLE 4: Data Protection & Anonymization")
        print("="*80)
        
        # Sample personal data
        personal_data = {
            "user_id": "user_004",
            "email": "john.doe@example.com",
            "name": "John Doe",
            "phone": "+1-555-1234",
            "address": "123 Main St, City",
            "age": 35,
            "income": 75000,
            "credit_score": 720
        }
        
        print("\n1️⃣ Original Personal Data")
        print("-" * 40)
        for key, value in personal_data.items():
            print(f"  {key}: {value}")
        
        # Step 1: Anonymization
        print("\n2️⃣ Data Anonymization")
        print("-" * 40)
        
        anonymized_data = self.rgpd_service.anonymize_data(personal_data)
        
        print("Anonymized data:")
        for key, value in anonymized_data.items():
            if value != personal_data[key]:
                print(f"  {key}: {personal_data[key]} → {value}")
            else:
                print(f"  {key}: {value} (unchanged)")
        
        # Step 2: Pseudonymization
        print("\n3️⃣ Data Pseudonymization")
        print("-" * 40)
        
        pseudonymized_data, pseudonym = self.rgpd_service.pseudonymize_data(
            personal_data,
            personal_data['user_id']
        )
        
        print(f"Pseudonym generated: {pseudonym}")
        print("Pseudonymized data:")
        for key, value in pseudonymized_data.items():
            if value != personal_data[key]:
                print(f"  {key}: {personal_data[key]} → {value}")
        
        # Step 3: Encryption
        print("\n4️⃣ Sensitive Data Encryption")
        print("-" * 40)
        
        sensitive_info = "SSN: 123-45-6789, Credit Card: 4111-1111-1111-1111"
        
        encrypted = self.rgpd_service.encrypt_sensitive_data(sensitive_info)
        print(f"Original: {sensitive_info}")
        print(f"Encrypted: {encrypted[:50]}...")
        
        decrypted = self.rgpd_service.decrypt_sensitive_data(encrypted)
        print(f"Decrypted: {decrypted}")
        print(f"✅ Encryption/Decryption successful: {decrypted == sensitive_info}")
    
    def example_5_compliance_reporting(self):
        """Example 5: GDPR compliance reporting"""
        print("\n" + "="*80)
        print("EXAMPLE 5: GDPR Compliance Reporting")
        print("="*80)
        
        tenant_id = "tenant_123"
        
        # Generate compliance report
        print("\n1️⃣ Generating Compliance Report")
        print("-" * 40)
        
        report = self.rgpd_service.generate_compliance_report(
            tenant_id=tenant_id,
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow()
        )
        
        print(f"Report Period: {report['period']['start']} to {report['period']['end']}")
        
        print("\n📊 Request Statistics:")
        for request_type, stats in report['requests'].items():
            print(f"\n  {request_type.upper()}:")
            print(f"    Total: {stats['total']}")
            print(f"    Completed: {stats['completed']}")
            print(f"    Completion Rate: {stats['completion_rate']:.1f}%")
            print(f"    Avg Processing: {stats['avg_processing_days']:.1f} days")
        
        print("\n📊 Consent Statistics:")
        print(f"  Total: {report['consents']['total']}")
        print(f"  Granted: {report['consents']['granted']}")
        print(f"  Revoked: {report['consents']['revoked']}")
        print(f"  Grant Rate: {report['consents']['grant_rate']:.1f}%")
        
        print(f"\n🏆 Compliance Score: {report['compliance_score']:.1f}/100")
        
        # Data mapping
        print("\n2️⃣ Personal Data Mapping")
        print("-" * 40)
        
        # Create sample data records
        from automl_platform.rgpd_compliance_service import PersonalDataRecord
        
        session = self.rgpd_service.SessionLocal()
        
        sample_records = [
            PersonalDataRecord(
                user_id="user_001",
                tenant_id=tenant_id,
                data_category="basic",
                data_type="email",
                storage_location="postgresql",
                table_name="users",
                column_name="email",
                purpose="Account management",
                legal_basis="contract",
                retention_period_days=730,
                encrypted=True,
                anonymized=False
            ),
            PersonalDataRecord(
                user_id="user_001",
                tenant_id=tenant_id,
                data_category="behavioral",
                data_type="usage_patterns",
                storage_location="redis",
                purpose="Service improvement",
                legal_basis="legitimate_interest",
                retention_period_days=90,
                encrypted=False,
                anonymized=True
            )
        ]
        
        for record in sample_records:
            session.add(record)
        session.commit()
        session.close()
        
        # Get data mapping
        data_mapping = self.rgpd_service.get_data_mapping(tenant_id)
        
        print("Personal Data Categories:")
        df_mapping = pd.DataFrame(data_mapping)
        if not df_mapping.empty:
            print(df_mapping[['category', 'type', 'location', 'retention_days', 'encrypted']].to_string())
        
        # Audit trail
        print("\n3️⃣ GDPR-Relevant Audit Trail")
        print("-" * 40)
        
        # Search for GDPR-relevant events
        gdpr_events = self.audit_service.search(
            gdpr_only=True,
            limit=5
        )
        
        print(f"Recent GDPR events: {len(gdpr_events)}")
        for event in gdpr_events[:3]:
            print(f"\n  Event: {event.get('action', 'N/A')}")
            print(f"  Type: {event.get('event_type', 'N/A')}")
            print(f"  User: {event.get('user_id', 'N/A')}")
            print(f"  Time: {event.get('timestamp', 'N/A')}")
    
    async def _simulate_callback(self, provider: str, code: str, state: str, redirect_uri: str) -> Dict:
        """Simulate OAuth callback (for demo purposes)"""
        
        # In real scenario, this would exchange code for tokens
        # For demo, return mock user data
        return {
            "session_id": f"session_{provider}_123",
            "user_id": f"user_{provider}_001",
            "email": f"user@{provider}.example.com",
            "name": f"Test User ({provider})",
            "roles": ["user", "ml_user"],
            "provider": provider
        }


async def main():
    """Main execution function"""
    example = SSOAndRGPDExample()
    
    print("\n" + "="*80)
    print("SSO AND RGPD/GDPR COMPLIANCE EXAMPLES")
    print("="*80)
    
    # Example 1: SSO Authentication
    await example.example_1_sso_authentication()
    
    # Example 2: Consent Management
    example.example_2_consent_management()
    
    # Example 3: Data Subject Requests
    example.example_3_data_subject_requests()
    
    # Example 4: Data Protection
    example.example_4_data_protection()
    
    # Example 5: Compliance Reporting
    example.example_5_compliance_reporting()
    
    print("\n" + "="*80)
    print("✅ ALL SSO AND RGPD EXAMPLES COMPLETED!")
    print("="*80)
    
    print("\n📊 Summary:")
    print("  - SSO authentication flows demonstrated")
    print("  - Consent management implemented")
    print("  - Data subject requests processed")
    print("  - Data protection measures applied")
    print("  - Compliance reporting generated")
    
    print("\n⚠️ Notes:")
    print("  - Configure SSO providers in environment variables")
    print("  - Ensure PostgreSQL is running for RGPD storage")
    print("  - Redis required for session management")
    print("  - All personal data in examples is fictional")


if __name__ == "__main__":
    asyncio.run(main())
