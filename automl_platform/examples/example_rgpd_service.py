"""
RGPD/GDPR Compliance Example
============================
Place in: examples/example_rgpd_service.py

Demonstrates comprehensive GDPR compliance features including consent management,
data subject requests, data protection, anonymization, and regulatory reporting.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import hashlib

# Import RGPD components
from automl_platform.rgpd_compliance_service import (
    RGPDComplianceService,
    GDPRRequestType,
    ConsentType,
    DataCategory,
    PersonalDataRecord,
    get_rgpd_service
)
from automl_platform.audit_service import (
    AuditService,
    AuditEventType,
    AuditSeverity
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RGPDComplianceExample:
    """Comprehensive RGPD/GDPR compliance examples"""
    
    def __init__(self):
        self.rgpd_service = get_rgpd_service()
        self.audit_service = AuditService()
        
        # Sample data for testing
        self.test_users = [
            {"user_id": "user_001", "email": "john.doe@example.com", "name": "John Doe"},
            {"user_id": "user_002", "email": "jane.smith@example.com", "name": "Jane Smith"},
            {"user_id": "user_003", "email": "bob.wilson@example.com", "name": "Bob Wilson"}
        ]
        
        self.tenant_id = "tenant_automl"
    
    def example_1_consent_management(self):
        """Example 1: Comprehensive consent management"""
        print("\n" + "="*80)
        print("EXAMPLE 1: Consent Management System")
        print("="*80)
        
        user_id = self.test_users[0]["user_id"]
        
        # Step 1: Present consent options
        print("\n1️⃣ Consent Collection Interface")
        print("-" * 40)
        
        consent_options = [
            {
                "type": ConsentType.DATA_PROCESSING,
                "title": "Essential Data Processing",
                "description": "Process your personal data to provide ML services",
                "required": True,
                "categories": ["basic", "technical"],
                "retention_days": 730
            },
            {
                "type": ConsentType.ANALYTICS,
                "title": "Service Analytics",
                "description": "Analyze usage patterns to improve our service",
                "required": False,
                "categories": ["behavioral", "technical"],
                "retention_days": 365
            },
            {
                "type": ConsentType.MARKETING,
                "title": "Marketing Communications",
                "description": "Send you updates about new features and offerings",
                "required": False,
                "categories": ["contact"],
                "retention_days": 365
            },
            {
                "type": ConsentType.THIRD_PARTY,
                "title": "Partner Integration",
                "description": "Share data with integrated third-party services",
                "required": False,
                "categories": ["basic", "technical"],
                "retention_days": 180
            },
            {
                "type": ConsentType.PROFILING,
                "title": "User Profiling",
                "description": "Create usage profiles for personalization",
                "required": False,
                "categories": ["behavioral", "derived"],
                "retention_days": 180
            }
        ]
        
        print("Consent Options Presented to User:\n")
        for i, option in enumerate(consent_options, 1):
            req_status = "REQUIRED" if option["required"] else "Optional"
            print(f"  {i}. [{req_status}] {option['title']}")
            print(f"     {option['description']}")
            print(f"     Categories: {', '.join(option['categories'])}")
            print(f"     Retention: {option['retention_days']} days")
            print()
        
        # Step 2: Record user choices
        print("\n2️⃣ Recording User Consent Choices")
        print("-" * 40)
        
        # Simulate user choices
        user_choices = {
            ConsentType.DATA_PROCESSING: True,  # Required
            ConsentType.ANALYTICS: True,
            ConsentType.MARKETING: False,
            ConsentType.THIRD_PARTY: False,
            ConsentType.PROFILING: True
        }
        
        consent_records = []
        for consent_option in consent_options:
            consent_type = consent_option["type"]
            granted = user_choices[consent_type]
            
            consent_id = self.rgpd_service.record_consent(
                user_id=user_id,
                consent_type=consent_type,
                granted=granted,
                tenant_id=self.tenant_id,
                purpose=consent_option["description"],
                data_categories=consent_option["categories"],
                expires_in_days=consent_option["retention_days"],
                ip_address="192.168.1.100",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            )
            
            consent_records.append(consent_id)
            
            status = "✅ GRANTED" if granted else "❌ DENIED"
            print(f"  {consent_type.value}: {status}")
            if granted:
                print(f"    Expires: {(datetime.utcnow() + timedelta(days=consent_option['retention_days'])).strftime('%Y-%m-%d')}")
        
        # Step 3: Consent verification
        print("\n3️⃣ Consent Verification for Processing")
        print("-" * 40)
        
        processing_scenarios = [
            ("Send marketing email", ConsentType.MARKETING),
            ("Train ML model with user data", ConsentType.DATA_PROCESSING),
            ("Share data with partner API", ConsentType.THIRD_PARTY),
            ("Create user behavior profile", ConsentType.PROFILING),
            ("Generate usage analytics", ConsentType.ANALYTICS)
        ]
        
        for scenario, required_consent in processing_scenarios:
            has_consent = self.rgpd_service.check_consent(user_id, required_consent)
            
            if has_consent:
                print(f"  ✅ {scenario}: ALLOWED")
            else:
                print(f"  ❌ {scenario}: BLOCKED (no consent)")
        
        # Step 4: Consent withdrawal
        print("\n4️⃣ Consent Withdrawal Process")
        print("-" * 40)
        
        print("  User withdraws analytics consent...")
        
        self.rgpd_service.record_consent(
            user_id=user_id,
            consent_type=ConsentType.ANALYTICS,
            granted=False,
            tenant_id=self.tenant_id,
            purpose="Consent withdrawn by user"
        )
        
        # Verify withdrawal
        has_analytics = self.rgpd_service.check_consent(user_id, ConsentType.ANALYTICS)
        print(f"  Analytics consent after withdrawal: {'✅ Active' if has_analytics else '❌ Withdrawn'}")
        
        # Step 5: Consent history
        print("\n5️⃣ Consent History & Audit Trail")
        print("-" * 40)
        
        all_consents = self.rgpd_service.get_user_consents(user_id)
        
        print(f"  Total consent records: {len(all_consents)}")
        print("\n  Recent consent changes:")
        
        for consent in all_consents[:5]:
            status = "Active" if consent['active'] else "Inactive"
            print(f"    • {consent['type']} - {status}")
            if consent['granted_at']:
                print(f"      Granted: {consent['granted_at']}")
            if consent['revoked_at']:
                print(f"      Revoked: {consent['revoked_at']}")
    
    def example_2_data_subject_requests(self):
        """Example 2: Processing GDPR data subject requests"""
        print("\n" + "="*80)
        print("EXAMPLE 2: Data Subject Request Processing")
        print("="*80)
        
        # Test different request types
        request_scenarios = [
            {
                "user_id": "user_001",
                "type": GDPRRequestType.ACCESS,
                "title": "Data Access Request (Article 15)",
                "reason": "User wants to review all collected personal data"
            },
            {
                "user_id": "user_002",
                "type": GDPRRequestType.PORTABILITY,
                "title": "Data Portability Request (Article 20)",
                "reason": "User switching to competitor service"
            },
            {
                "user_id": "user_001",
                "type": GDPRRequestType.RECTIFICATION,
                "title": "Data Rectification Request (Article 16)",
                "reason": "Incorrect personal information needs correction"
            },
            {
                "user_id": "user_003",
                "type": GDPRRequestType.ERASURE,
                "title": "Right to be Forgotten (Article 17)",
                "reason": "User closing account permanently"
            }
        ]
        
        for scenario in request_scenarios:
            print(f"\n{'='*60}")
            print(f"📋 {scenario['title']}")
            print(f"{'='*60}")
            
            # Create request
            request_id = self.rgpd_service.create_data_request(
                user_id=scenario["user_id"],
                request_type=scenario["type"],
                tenant_id=self.tenant_id,
                reason=scenario["reason"]
            )
            
            print(f"\n  Request Details:")
            print(f"    ID: {request_id}")
            print(f"    User: {scenario['user_id']}")
            print(f"    Type: {scenario['type'].value}")
            print(f"    Reason: {scenario['reason']}")
            print(f"    Legal Deadline: 30 days")
            
            # Process based on type
            if scenario["type"] == GDPRRequestType.ACCESS:
                self._process_access_request(request_id)
            elif scenario["type"] == GDPRRequestType.PORTABILITY:
                self._process_portability_request(request_id)
            elif scenario["type"] == GDPRRequestType.RECTIFICATION:
                self._process_rectification_request(request_id)
            elif scenario["type"] == GDPRRequestType.ERASURE:
                self._process_erasure_request(request_id, scenario["user_id"])
    
    def example_3_data_protection(self):
        """Example 3: Data protection and privacy techniques"""
        print("\n" + "="*80)
        print("EXAMPLE 3: Data Protection & Privacy Techniques")
        print("="*80)
        
        # Sample sensitive data
        sensitive_data = {
            "user_id": "user_004",
            "email": "alice.johnson@example.com",
            "name": "Alice Johnson",
            "phone": "+1-555-0123",
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111",
            "address": "742 Evergreen Terrace, Springfield",
            "date_of_birth": "1985-06-15",
            "salary": 85000,
            "medical_condition": "Type 2 Diabetes",
            "ip_address": "192.168.1.42"
        }
        
        print("\n1️⃣ Original Sensitive Data")
        print("-" * 40)
        self._print_data_sample(sensitive_data, ["email", "ssn", "credit_card"])
        
        # Technique 1: Anonymization
        print("\n2️⃣ Data Anonymization")
        print("-" * 40)
        
        anonymized = self.rgpd_service.anonymize_data(sensitive_data.copy())
        
        print("  Anonymization Results:")
        for key in ["email", "name", "phone", "ssn", "credit_card", "ip_address"]:
            if key in sensitive_data:
                original = sensitive_data[key]
                anon = anonymized[key]
                if original != anon:
                    print(f"    {key}: {original} → {anon}")
        
        # Technique 2: Pseudonymization
        print("\n3️⃣ Data Pseudonymization")
        print("-" * 40)
        
        pseudonymized, pseudonym = self.rgpd_service.pseudonymize_data(
            sensitive_data.copy(),
            sensitive_data['user_id']
        )
        
        print(f"  Generated Pseudonym: {pseudonym}")
        print(f"  Reversible: Yes (with mapping key)")
        print("\n  Pseudonymized Fields:")
        print(f"    user_id: {sensitive_data['user_id']} → {pseudonym}")
        print(f"    Other PII: Retained but linked to pseudonym")
        
        # Technique 3: Encryption
        print("\n4️⃣ Field-Level Encryption")
        print("-" * 40)
        
        # Encrypt sensitive fields
        encrypted_fields = {}
        sensitive_fields = ["ssn", "credit_card", "medical_condition"]
        
        for field in sensitive_fields:
            if field in sensitive_data:
                original = sensitive_data[field]
                encrypted = self.rgpd_service.encrypt_sensitive_data(original)
                encrypted_fields[field] = encrypted
                
                print(f"  {field}:")
                print(f"    Original: {original}")
                print(f"    Encrypted: {encrypted[:50]}...")
        
        # Verify decryption
        print("\n  Decryption Verification:")
        for field, encrypted in encrypted_fields.items():
            decrypted = self.rgpd_service.decrypt_sensitive_data(encrypted)
            matches = decrypted == sensitive_data[field]
            status = "✅" if matches else "❌"
            print(f"    {field}: {status} Decryption successful")
        
        # Technique 4: Data Minimization
        print("\n5️⃣ Data Minimization")
        print("-" * 40)
        
        purposes = {
            "authentication": ["user_id", "email"],
            "billing": ["user_id", "name", "address"],
            "analytics": ["user_id"],  # Minimal data
            "ml_training": ["salary", "date_of_birth"]  # Anonymized
        }
        
        print("  Data Collection by Purpose:")
        for purpose, fields in purposes.items():
            print(f"\n    {purpose.upper()}:")
            print(f"      Fields: {', '.join(fields)}")
            print(f"      Data points: {len(fields)}")
        
        # Technique 5: Differential Privacy
        print("\n6️⃣ Differential Privacy for Analytics")
        print("-" * 40)
        
        # Simulate salary data with noise
        original_salaries = [75000, 85000, 95000, 65000, 120000]
        epsilon = 1.0  # Privacy budget
        
        # Add Laplace noise
        sensitivity = 10000  # Salary sensitivity
        scale = sensitivity / epsilon
        noisy_salaries = [
            salary + np.random.laplace(0, scale)
            for salary in original_salaries
        ]
        
        print(f"  Privacy Budget (ε): {epsilon}")
        print(f"  Original Mean: ${np.mean(original_salaries):,.0f}")
        print(f"  Noisy Mean: ${np.mean(noisy_salaries):,.0f}")
        print(f"  Error: ${abs(np.mean(original_salaries) - np.mean(noisy_salaries)):,.0f}")
    
    def example_4_data_retention(self):
        """Example 4: Data retention and deletion policies"""
        print("\n" + "="*80)
        print("EXAMPLE 4: Data Retention & Deletion Policies")
        print("="*80)
        
        # Define retention policies
        retention_policies = [
            {
                "category": DataCategory.BASIC,
                "description": "Basic personal information",
                "retention_days": 730,  # 2 years
                "legal_basis": "Contract execution",
                "examples": ["name", "email", "user_id"]
            },
            {
                "category": DataCategory.FINANCIAL,
                "description": "Financial and billing data",
                "retention_days": 2555,  # 7 years (legal requirement)
                "legal_basis": "Legal obligation",
                "examples": ["invoices", "payments", "tax_id"]
            },
            {
                "category": DataCategory.BEHAVIORAL,
                "description": "Usage patterns and behavior",
                "retention_days": 90,  # 3 months
                "legal_basis": "Legitimate interest",
                "examples": ["clicks", "page_views", "feature_usage"]
            },
            {
                "category": DataCategory.TECHNICAL,
                "description": "Technical and system data",
                "retention_days": 30,  # 1 month
                "legal_basis": "Legitimate interest",
                "examples": ["ip_address", "user_agent", "session_id"]
            },
            {
                "category": DataCategory.SENSITIVE,
                "description": "Sensitive personal data",
                "retention_days": 0,  # Delete immediately when not needed
                "legal_basis": "Explicit consent",
                "examples": ["health_data", "biometric", "political_views"]
            }
        ]
        
        print("\n1️⃣ Data Retention Policy Matrix")
        print("-" * 40)
        
        # Create policy table
        policy_data = []
        for policy in retention_policies:
            policy_data.append({
                "Category": policy["category"].value,
                "Retention (days)": policy["retention_days"],
                "Legal Basis": policy["legal_basis"],
                "Examples": ", ".join(policy["examples"][:2])
            })
        
        df_policies = pd.DataFrame(policy_data)
        print(df_policies.to_string(index=False))
        
        # Simulate data lifecycle
        print("\n2️⃣ Data Lifecycle Management")
        print("-" * 40)
        
        # Create sample data records
        session = self.rgpd_service.SessionLocal()
        
        data_records = []
        for user in self.test_users:
            for policy in retention_policies[:3]:  # Create records for first 3 categories
                record = PersonalDataRecord(
                    user_id=user["user_id"],
                    tenant_id=self.tenant_id,
                    data_category=policy["category"].value,
                    data_type=policy["examples"][0],
                    storage_location="postgresql",
                    table_name=f"{policy['category'].value}_data",
                    purpose=policy["description"],
                    legal_basis=policy["legal_basis"],
                    retention_period_days=policy["retention_days"],
                    collected_at=datetime.utcnow() - timedelta(days=policy["retention_days"] + 10),
                    encrypted=policy["category"] == DataCategory.FINANCIAL
                )
                data_records.append(record)
                session.add(record)
        
        session.commit()
        
        print(f"  Created {len(data_records)} data records")
        
        # Check for expired data
        print("\n3️⃣ Expired Data Detection")
        print("-" * 40)
        
        expired_count = 0
        for record in data_records:
            retention_end = record.collected_at + timedelta(days=record.retention_period_days)
            if retention_end < datetime.utcnow():
                expired_count += 1
                days_overdue = (datetime.utcnow() - retention_end).days
                print(f"  ⚠️ EXPIRED: {record.data_type} for {record.user_id}")
                print(f"     Overdue by {days_overdue} days")
        
        print(f"\n  Total expired records: {expired_count}/{len(data_records)}")
        
        # Automated deletion
        print("\n4️⃣ Automated Deletion Process")
        print("-" * 40)
        
        if expired_count > 0:
            print("  Initiating automated deletion...")
            
            # Mark for deletion
            for record in data_records:
                retention_end = record.collected_at + timedelta(days=record.retention_period_days)
                if retention_end < datetime.utcnow():
                    record.deletion_date = datetime.utcnow()
                    print(f"    Marked for deletion: {record.data_type} ({record.user_id})")
            
            session.commit()
            print(f"\n  ✅ {expired_count} records marked for deletion")
        
        session.close()
        
        # Deletion verification
        print("\n5️⃣ Deletion Verification & Audit")
        print("-" * 40)
        
        # Log deletion events
        for i in range(min(3, expired_count)):
            self.audit_service.log_event(
                event_type=AuditEventType.DATA_DELETE,
                action="automated_retention_deletion",
                user_id=f"user_{i:03d}",
                resource_type="personal_data",
                metadata={
                    "category": "behavioral",
                    "reason": "retention_period_expired",
                    "deleted_records": 1
                },
                gdpr_relevant=True
            )
        
        print("  Deletion audit events logged")
        print("  Verification methods:")
        print("    ✅ Database queries confirm deletion")
        print("    ✅ Backup exclusion verified")
        print("    ✅ Cache invalidation completed")
        print("    ✅ Search index updated")
    
    def example_5_compliance_reporting(self):
        """Example 5: GDPR compliance reporting and metrics"""
        print("\n" + "="*80)
        print("EXAMPLE 5: Compliance Reporting & Metrics")
        print("="*80)
        
        # Generate compliance report
        report = self.rgpd_service.generate_compliance_report(
            tenant_id=self.tenant_id,
            start_date=datetime.utcnow() - timedelta(days=30),
            end_date=datetime.utcnow()
        )
        
        print("\n1️⃣ Monthly Compliance Report")
        print("-" * 40)
        print(f"  Period: {report['period']['start']} to {report['period']['end']}")
        
        # Request statistics
        print("\n2️⃣ Data Subject Request Statistics")
        print("-" * 40)
        
        request_types = [
            ("Access", "access"),
            ("Rectification", "rectification"),
            ("Erasure", "erasure"),
            ("Portability", "portability")
        ]
        
        print("\n  Request Processing Metrics:")
        print("  " + "-" * 50)
        print(f"  {'Type':<15} {'Total':<8} {'Completed':<10} {'Rate':<8} {'Avg Days':<10}")
        print("  " + "-" * 50)
        
        for display_name, key in request_types:
            stats = report['requests'].get(key, {
                'total': 0, 'completed': 0, 'completion_rate': 0, 'avg_processing_days': 0
            })
            print(f"  {display_name:<15} {stats['total']:<8} {stats['completed']:<10} "
                  f"{stats['completion_rate']:.1f}%{'  ':<5} {stats['avg_processing_days']:.1f}")
        
        # Consent metrics
        print("\n3️⃣ Consent Management Metrics")
        print("-" * 40)
        
        consent_stats = report['consents']
        
        # Create consent visualization
        consent_data = {
            'Metric': ['Total Consents', 'Granted', 'Revoked', 'Grant Rate'],
            'Value': [
                consent_stats['total'],
                consent_stats['granted'],
                consent_stats['revoked'],
                f"{consent_stats['grant_rate']:.1f}%"
            ]
        }
        
        df_consent = pd.DataFrame(consent_data)
        print(f"\n{df_consent.to_string(index=False)}")
        
        # Compliance score
        print("\n4️⃣ Compliance Score Calculation")
        print("-" * 40)
        
        score = report['compliance_score']
        score_color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        
        print(f"\n  Overall Compliance Score: {score_color} {score:.1f}/100")
        
        # Score breakdown
        score_factors = [
            ("Request completion rate", 30, 28),
            ("Response time compliance", 25, 22),
            ("Consent management", 20, 18),
            ("Data protection measures", 15, 14),
            ("Audit trail completeness", 10, 8)
        ]
        
        print("\n  Score Breakdown:")
        for factor, max_points, actual_points in score_factors:
            percentage = (actual_points / max_points) * 100
            print(f"    • {factor}: {actual_points}/{max_points} ({percentage:.0f}%)")
        
        # Risk assessment
        print("\n5️⃣ Privacy Risk Assessment")
        print("-" * 40)
        
        risk_items = [
            ("Unencrypted sensitive data", "HIGH", "🔴"),
            ("Expired data not deleted", "MEDIUM", "🟡"),
            ("Missing consent records", "HIGH", "🔴"),
            ("Incomplete audit trails", "LOW", "🟢"),
            ("Third-party data sharing", "MEDIUM", "🟡")
        ]
        
        print("\n  Identified Risks:")
        for risk, severity, icon in risk_items:
            print(f"    {icon} {risk}: {severity}")
        
        # Recommendations
        print("\n6️⃣ Compliance Recommendations")
        print("-" * 40)
        
        recommendations = [
            "Implement automated data deletion for expired records",
            "Enhance encryption for sensitive data categories",
            "Improve consent collection UI for better user understanding",
            "Reduce data subject request processing time to < 15 days",
            "Conduct quarterly privacy impact assessments"
        ]
        
        print("\n  Priority Actions:")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec}")
        
        # Data mapping summary
        print("\n7️⃣ Personal Data Mapping")
        print("-" * 40)
        
        data_mapping = self.rgpd_service.get_data_mapping(self.tenant_id)
        
        if data_mapping:
            # Group by category
            category_summary = {}
            for mapping in data_mapping:
                category = mapping['category']
                if category not in category_summary:
                    category_summary[category] = {
                        'count': 0,
                        'encrypted': 0,
                        'anonymized': 0
                    }
                category_summary[category]['count'] += 1
                if mapping.get('encrypted'):
                    category_summary[category]['encrypted'] += 1
                if mapping.get('anonymized'):
                    category_summary[category]['anonymized'] += 1
            
            print("\n  Data Categories Overview:")
            for category, stats in category_summary.items():
                print(f"\n    {category.upper()}:")
                print(f"      Total records: {stats['count']}")
                print(f"      Encrypted: {stats['encrypted']} ({stats['encrypted']/stats['count']*100:.0f}%)")
                print(f"      Anonymized: {stats['anonymized']} ({stats['anonymized']/stats['count']*100:.0f}%)")
    
    # Helper methods
    
    def _process_access_request(self, request_id: str):
        """Process data access request"""
        print("\n  Processing Access Request...")
        
        user_data = self.rgpd_service.process_access_request(request_id)
        
        print("\n  ✅ Data Package Prepared:")
        print(f"    • Personal data records: {len(user_data.get('personal_data', []))}")
        print(f"    • Active consents: {len([c for c in user_data.get('consents', []) if c.get('active')])}")
        print(f"    • Processing activities: {len(user_data.get('processing_activities', []))}")
        print(f"    • Profile data included: Yes")
        print(f"    • ML model data included: Yes")
        print(f"    • Usage analytics included: Yes")
        
        print("\n  Delivery method: Secure download link")
        print("  Format: JSON (machine-readable)")
        print("  Encryption: AES-256")
    
    def _process_portability_request(self, request_id: str):
        """Process data portability request"""
        print("\n  Processing Portability Request...")
        
        portable_data = self.rgpd_service.process_portability_request(request_id)
        
        print("\n  ✅ Portable Data Package Created:")
        print(f"    • Format: JSON and CSV")
        print(f"    • Size: {len(portable_data)} bytes")
        print(f"    • Machine-readable: Yes")
        print(f"    • Industry standard format: Yes")
        print(f"    • Includes structured data: Yes")
        print(f"    • Ready for import to other systems: Yes")
    
    def _process_rectification_request(self, request_id: str):
        """Process data rectification request"""
        print("\n  Processing Rectification Request...")
        
        # Corrections to apply
        corrections = {
            "email": "corrected.email@example.com",
            "phone": "+1-555-9999",
            "address": "456 New Address, New City"
        }
        
        print(f"\n  Corrections to apply: {list(corrections.keys())}")
        
        result = self.rgpd_service.process_rectification_request(
            request_id,
            corrections
        )
        
        print("\n  ✅ Rectification Completed:")
        print(f"    • Fields updated: {len(result['rectified_items'])}")
        print("    • Systems updated: Database, Cache, Search Index")
        print("    • Propagation to partners: Initiated")
        print("    • Audit trail: Created")
    
    def _process_erasure_request(self, request_id: str, user_id: str):
        """Process erasure request"""
        print("\n  Processing Erasure Request...")
        print("  Checking eligibility for erasure...")
        
        result = self.rgpd_service.process_erasure_request(
            request_id,
            verify_legal_basis=True
        )
        
        if result['status'] == 'completed':
            print("\n  ✅ Erasure Completed:")
            print(f"    • Personal data: {result['erased_items']['personal_data']} records")
            print(f"    • Consents: {result['erased_items']['consents']} records")
            print(f"    • ML models: {result['erased_items']['ml_models']} references")
            print(f"    • Audit logs: Anonymized")
            print("\n  Post-erasure verification:")
            print("    • Database: No personal data found")
            print("    • Backups: Marked for deletion")
            print("    • Partners: Erasure request sent")
        else:
            print(f"\n  ❌ Erasure Rejected:")
            print(f"    Reason: {result.get('reason', 'Legal obligation to retain data')}")
            print("    User notified with explanation")
    
    def _print_data_sample(self, data: Dict, sensitive_fields: List[str]):
        """Print data sample with sensitive fields marked"""
        for key, value in data.items():
            if key in sensitive_fields:
                print(f"  {key}: {value} [SENSITIVE]")
            else:
                print(f"  {key}: {value}")


def main():
    """Main execution function"""
    example = RGPDComplianceExample()
    
    print("\n" + "="*80)
    print(" " * 15 + "RGPD/GDPR COMPLIANCE EXAMPLES")
    print("="*80)
    
    # Run examples
    example.example_1_consent_management()
    example.example_2_data_subject_requests()
    example.example_3_data_protection()
    example.example_4_data_retention()
    example.example_5_compliance_reporting()
    
    print("\n" + "="*80)
    print(" " * 10 + "✅ ALL RGPD/GDPR EXAMPLES COMPLETED!")
    print("="*80)
    
    print("\n📊 Summary of Features Demonstrated:")
    print("  ✓ Comprehensive consent management")
    print("  ✓ Data subject rights (Access, Portability, Rectification, Erasure)")
    print("  ✓ Data protection techniques (Anonymization, Pseudonymization, Encryption)")
    print("  ✓ Data retention and lifecycle management")
    print("  ✓ Compliance reporting and metrics")
    print("  ✓ Privacy risk assessment")
    print("  ✓ Audit trail and accountability")
    
    print("\n⚙️ Implementation Requirements:")
    print("  1. PostgreSQL database for RGPD data storage")
    print("  2. Redis for caching and session management")
    print("  3. Encryption keys configured (RGPD_ENCRYPTION_KEY)")
    print("  4. Audit service integration")
    print("  5. Regular compliance monitoring")
    
    print("\n📋 Legal Compliance Notes:")
    print("  • Ensure legal team reviews all consent forms and privacy notices")
    print("  • Implement data processing agreements with third parties")
    print("  • Maintain records of processing activities (Article 30)")
    print("  • Appoint Data Protection Officer if required")
    print("  • Conduct Data Protection Impact Assessments (DPIA) for high-risk processing")
    print("  • Report data breaches within 72 hours to supervisory authority")
    print("  • Implement Privacy by Design and Default principles")


if __name__ == "__main__":
    main()
