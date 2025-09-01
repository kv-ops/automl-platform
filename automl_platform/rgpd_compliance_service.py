"""
RGPD/GDPR Compliance Service
=============================
Place in: automl_platform/rgpd_compliance_service.py

Comprehensive GDPR compliance with data privacy, consent management,
and regulatory reporting. Integrates with audit_service and sso_service.
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64

import redis
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, JSON, Text, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
import pandas as pd

# Import from existing services
from .audit_service import AuditService, AuditEventType, AuditSeverity

logger = logging.getLogger(__name__)

Base = declarative_base()


class GDPRRequestType(Enum):
    """Types of GDPR requests"""
    ACCESS = "access"  # Article 15
    RECTIFICATION = "rectification"  # Article 16
    ERASURE = "erasure"  # Article 17 - Right to be forgotten
    PORTABILITY = "portability"  # Article 20
    RESTRICTION = "restriction"  # Article 18
    OBJECTION = "objection"  # Article 21


class ConsentType(Enum):
    """Types of consent"""
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    COOKIES = "cookies"
    DATA_PROCESSING = "data_processing"
    THIRD_PARTY = "third_party"
    PROFILING = "profiling"
    AUTOMATED_DECISION = "automated_decision"


class DataCategory(Enum):
    """Categories of personal data"""
    BASIC = "basic"  # Name, email
    CONTACT = "contact"  # Address, phone
    FINANCIAL = "financial"  # Payment info
    BEHAVIORAL = "behavioral"  # Usage patterns
    TECHNICAL = "technical"  # IP, device info
    SENSITIVE = "sensitive"  # Health, biometric
    DERIVED = "derived"  # ML predictions


@dataclass
class GDPRRequest:
    """GDPR request data structure"""
    request_id: str
    user_id: str
    tenant_id: Optional[str]
    request_type: GDPRRequestType
    
    # Request details
    requested_at: datetime
    requested_by: str  # user or authorized representative
    reason: Optional[str]
    
    # Processing
    status: str  # pending, processing, completed, rejected
    processed_at: Optional[datetime]
    processed_by: Optional[str]
    
    # Response
    response_data: Optional[Dict]
    response_format: str = "json"  # json, csv, pdf
    
    # Verification
    identity_verified: bool = False
    verification_method: Optional[str]
    
    # Legal basis
    legal_basis: Optional[str]
    retention_override: Optional[bool]


class ConsentRecord(Base):
    """Database model for consent records"""
    __tablename__ = "consent_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    tenant_id = Column(String(255), index=True)
    
    # Consent details
    consent_type = Column(String(50), nullable=False)
    granted = Column(Boolean, nullable=False)
    
    # Timestamps
    granted_at = Column(DateTime)
    revoked_at = Column(DateTime)
    expires_at = Column(DateTime)
    
    # Context
    ip_address = Column(String(45))
    user_agent = Column(Text)
    consent_text = Column(Text)
    version = Column(String(50))
    
    # Purpose and scope
    purpose = Column(Text)
    data_categories = Column(JSON)
    third_parties = Column(JSON)
    
    # Audit
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DataSubjectRequest(Base):
    """Database model for GDPR requests"""
    __tablename__ = "data_subject_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    tenant_id = Column(String(255), index=True)
    
    # Request info
    request_type = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")
    
    # Details
    reason = Column(Text)
    requested_data = Column(JSON)
    
    # Processing
    requested_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    deadline = Column(DateTime)  # Legal deadline (30 days)
    
    # Response
    response_data = Column(Text)  # Encrypted
    response_format = Column(String(20))
    
    # Verification
    identity_verified = Column(Boolean, default=False)
    verification_method = Column(String(100))
    
    # Metadata
    metadata = Column(JSON)


class PersonalDataRecord(Base):
    """Database model for tracking personal data"""
    __tablename__ = "personal_data_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    tenant_id = Column(String(255), index=True)
    
    # Data classification
    data_category = Column(String(50), nullable=False)
    data_type = Column(String(100), nullable=False)  # email, name, etc.
    
    # Storage location
    storage_location = Column(String(255))  # database, file, third-party
    table_name = Column(String(100))
    column_name = Column(String(100))
    
    # Purpose and legal basis
    purpose = Column(Text)
    legal_basis = Column(String(100))
    
    # Retention
    collected_at = Column(DateTime, default=datetime.utcnow)
    retention_period_days = Column(Integer)
    deletion_date = Column(DateTime)
    
    # Data flow
    source = Column(String(255))  # Where data came from
    shared_with = Column(JSON)  # Third parties
    
    # Security
    encrypted = Column(Boolean, default=False)
    anonymized = Column(Boolean, default=False)
    pseudonymized = Column(Boolean, default=False)


class RGPDComplianceService:
    """
    Comprehensive RGPD/GDPR compliance service
    """
    
    def __init__(
        self,
        database_url: str = None,
        redis_client: redis.Redis = None,
        audit_service: AuditService = None,
        encryption_key: bytes = None
    ):
        # Database setup
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://user:pass@localhost/rgpd"
        )
        self.engine = create_engine(self.database_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Redis for caching
        self.redis_client = redis_client or redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        
        # Audit service integration
        self.audit_service = audit_service or AuditService()
        
        # Encryption
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for sensitive data"""
        key_string = os.getenv("RGPD_ENCRYPTION_KEY")
        if key_string:
            return base64.b64decode(key_string)
        
        # Generate new key
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'rgpd_salt',  # In production, use proper salt management
            iterations=100000,
        )
        return kdf.derive(b'rgpd_key')  # In production, use secure key
    
    # ==================== CONSENT MANAGEMENT ====================
    
    def record_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        granted: bool,
        tenant_id: Optional[str] = None,
        purpose: str = None,
        data_categories: List[str] = None,
        expires_in_days: int = 365,
        ip_address: str = None,
        user_agent: str = None
    ) -> str:
        """
        Record user consent
        
        Returns:
            Consent record ID
        """
        session = self.SessionLocal()
        try:
            # Check for existing consent
            existing = session.query(ConsentRecord).filter_by(
                user_id=user_id,
                consent_type=consent_type.value,
                revoked_at=None
            ).first()
            
            if existing:
                # Update existing consent
                if not granted:
                    existing.revoked_at = datetime.utcnow()
                    existing.granted = False
                else:
                    existing.granted = True
                    existing.granted_at = datetime.utcnow()
                    existing.expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
                
                consent_id = existing.id
            else:
                # Create new consent record
                consent = ConsentRecord(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    consent_type=consent_type.value,
                    granted=granted,
                    granted_at=datetime.utcnow() if granted else None,
                    expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if granted else None,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    purpose=purpose,
                    data_categories=data_categories,
                    version="1.0"
                )
                session.add(consent)
                session.flush()
                consent_id = consent.id
            
            session.commit()
            
            # Audit log
            self.audit_service.log_event(
                event_type=AuditEventType.CONSENT_UPDATE,
                action=f"consent_{('granted' if granted else 'revoked')}",
                user_id=user_id,
                tenant_id=tenant_id,
                resource_type="consent",
                resource_id=str(consent_id),
                metadata={
                    "consent_type": consent_type.value,
                    "granted": granted
                }
            )
            
            logger.info(f"Recorded consent for user {user_id}: {consent_type.value} = {granted}")
            return str(consent_id)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to record consent: {e}")
            raise
        finally:
            session.close()
    
    def get_user_consents(self, user_id: str) -> List[Dict]:
        """Get all consents for a user"""
        session = self.SessionLocal()
        try:
            consents = session.query(ConsentRecord).filter_by(
                user_id=user_id
            ).order_by(ConsentRecord.created_at.desc()).all()
            
            return [
                {
                    "id": str(c.id),
                    "type": c.consent_type,
                    "granted": c.granted,
                    "granted_at": c.granted_at.isoformat() if c.granted_at else None,
                    "revoked_at": c.revoked_at.isoformat() if c.revoked_at else None,
                    "expires_at": c.expires_at.isoformat() if c.expires_at else None,
                    "purpose": c.purpose,
                    "active": c.granted and not c.revoked_at and (not c.expires_at or c.expires_at > datetime.utcnow())
                }
                for c in consents
            ]
        finally:
            session.close()
    
    def check_consent(
        self,
        user_id: str,
        consent_type: ConsentType
    ) -> bool:
        """Check if user has given specific consent"""
        session = self.SessionLocal()
        try:
            consent = session.query(ConsentRecord).filter_by(
                user_id=user_id,
                consent_type=consent_type.value,
                granted=True,
                revoked_at=None
            ).first()
            
            if not consent:
                return False
            
            # Check expiration
            if consent.expires_at and consent.expires_at < datetime.utcnow():
                return False
            
            return True
            
        finally:
            session.close()
    
    # ==================== DATA SUBJECT REQUESTS ====================
    
    def create_data_request(
        self,
        user_id: str,
        request_type: GDPRRequestType,
        tenant_id: Optional[str] = None,
        reason: str = None,
        requested_data: Dict = None
    ) -> str:
        """
        Create a GDPR data subject request
        
        Returns:
            Request ID
        """
        session = self.SessionLocal()
        try:
            request_id = f"GDPR-{uuid.uuid4().hex[:8].upper()}"
            
            # Create request
            request = DataSubjectRequest(
                request_id=request_id,
                user_id=user_id,
                tenant_id=tenant_id,
                request_type=request_type.value,
                status="pending",
                reason=reason,
                requested_data=requested_data,
                deadline=datetime.utcnow() + timedelta(days=30)  # Legal deadline
            )
            
            session.add(request)
            session.commit()
            
            # Audit log
            self.audit_service.log_event(
                event_type=AuditEventType.GDPR_REQUEST,
                action=f"gdpr_request_{request_type.value}",
                user_id=user_id,
                tenant_id=tenant_id,
                resource_type="gdpr_request",
                resource_id=request_id,
                gdpr_relevant=True
            )
            
            # Process immediately for some request types
            if request_type in [GDPRRequestType.ACCESS, GDPRRequestType.PORTABILITY]:
                self._process_request_async(request_id)
            
            logger.info(f"Created GDPR request {request_id} for user {user_id}")
            return request_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create GDPR request: {e}")
            raise
        finally:
            session.close()
    
    def process_access_request(self, request_id: str) -> Dict[str, Any]:
        """
        Process data access request (Article 15)
        
        Returns:
            User's personal data
        """
        session = self.SessionLocal()
        try:
            request = session.query(DataSubjectRequest).filter_by(
                request_id=request_id
            ).first()
            
            if not request:
                raise ValueError(f"Request {request_id} not found")
            
            user_data = {}
            
            # 1. Get personal data records
            data_records = session.query(PersonalDataRecord).filter_by(
                user_id=request.user_id
            ).all()
            
            user_data["personal_data"] = [
                {
                    "category": record.data_category,
                    "type": record.data_type,
                    "purpose": record.purpose,
                    "legal_basis": record.legal_basis,
                    "collected_at": record.collected_at.isoformat(),
                    "retention_days": record.retention_period_days,
                    "shared_with": record.shared_with
                }
                for record in data_records
            ]
            
            # 2. Get consent records
            consents = self.get_user_consents(request.user_id)
            user_data["consents"] = consents
            
            # 3. Get processing activities from audit logs
            audit_logs = self.audit_service.search(
                user_id=request.user_id,
                gdpr_only=True,
                limit=1000
            )
            user_data["processing_activities"] = audit_logs
            
            # 4. Get actual data from various sources
            user_data["profile_data"] = self._get_user_profile_data(request.user_id)
            user_data["ml_data"] = self._get_user_ml_data(request.user_id)
            user_data["usage_data"] = self._get_user_usage_data(request.user_id)
            
            # Encrypt response
            encrypted_data = self.fernet.encrypt(
                json.dumps(user_data, default=str).encode()
            )
            
            # Update request
            request.status = "completed"
            request.processed_at = datetime.utcnow()
            request.response_data = encrypted_data.decode()
            
            session.commit()
            
            return user_data
            
        finally:
            session.close()
    
    def process_erasure_request(
        self,
        request_id: str,
        verify_legal_basis: bool = True
    ) -> Dict[str, Any]:
        """
        Process data erasure request (Article 17 - Right to be forgotten)
        
        Returns:
            Erasure confirmation
        """
        session = self.SessionLocal()
        try:
            request = session.query(DataSubjectRequest).filter_by(
                request_id=request_id
            ).first()
            
            if not request:
                raise ValueError(f"Request {request_id} not found")
            
            # Check if erasure is allowed
            if verify_legal_basis:
                can_erase, reason = self._check_erasure_eligibility(request.user_id)
                if not can_erase:
                    request.status = "rejected"
                    request.response_data = reason
                    session.commit()
                    return {"status": "rejected", "reason": reason}
            
            # Perform erasure
            erased_items = {
                "personal_data": 0,
                "consents": 0,
                "ml_models": 0,
                "logs": 0
            }
            
            # 1. Anonymize personal data
            data_records = session.query(PersonalDataRecord).filter_by(
                user_id=request.user_id
            ).all()
            
            for record in data_records:
                self._anonymize_data_record(record)
                erased_items["personal_data"] += 1
            
            # 2. Delete consents
            consents = session.query(ConsentRecord).filter_by(
                user_id=request.user_id
            ).all()
            
            for consent in consents:
                session.delete(consent)
                erased_items["consents"] += 1
            
            # 3. Anonymize ML data
            erased_items["ml_models"] = self._anonymize_ml_data(request.user_id)
            
            # 4. Anonymize audit logs
            erased_items["logs"] = self._anonymize_audit_logs(request.user_id)
            
            # Update request
            request.status = "completed"
            request.processed_at = datetime.utcnow()
            request.response_data = self.fernet.encrypt(
                json.dumps(erased_items).encode()
            ).decode()
            
            session.commit()
            
            # Final audit log
            self.audit_service.log_event(
                event_type=AuditEventType.DATA_DELETE,
                action="gdpr_erasure_completed",
                user_id="ANONYMIZED",
                resource_type="user_data",
                resource_id=request.user_id,
                metadata=erased_items,
                gdpr_relevant=True
            )
            
            return {
                "status": "completed",
                "erased_items": erased_items,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        finally:
            session.close()
    
    def process_portability_request(self, request_id: str) -> bytes:
        """
        Process data portability request (Article 20)
        
        Returns:
            Portable data package
        """
        # Get all user data
        user_data = self.process_access_request(request_id)
        
        # Convert to portable format
        df = pd.DataFrame(user_data["personal_data"])
        
        # Create CSV export
        csv_buffer = df.to_csv(index=False)
        
        # Create JSON export
        json_data = json.dumps(user_data, indent=2, default=str)
        
        # Package both formats
        return json_data.encode()
    
    def process_rectification_request(
        self,
        request_id: str,
        corrections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process data rectification request (Article 16)
        
        Returns:
            Rectification confirmation
        """
        session = self.SessionLocal()
        try:
            request = session.query(DataSubjectRequest).filter_by(
                request_id=request_id
            ).first()
            
            if not request:
                raise ValueError(f"Request {request_id} not found")
            
            rectified_items = []
            
            # Apply corrections
            for field, new_value in corrections.items():
                # This would update the actual data in various systems
                # For now, we'll track what was changed
                rectified_items.append({
                    "field": field,
                    "old_value": "[REDACTED]",
                    "new_value": new_value,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Update request
            request.status = "completed"
            request.processed_at = datetime.utcnow()
            request.response_data = self.fernet.encrypt(
                json.dumps(rectified_items).encode()
            ).decode()
            
            session.commit()
            
            # Audit log
            self.audit_service.log_event(
                event_type=AuditEventType.DATA_UPDATE,
                action="gdpr_rectification",
                user_id=request.user_id,
                resource_type="personal_data",
                metadata={"fields_updated": len(corrections)},
                gdpr_relevant=True
            )
            
            return {
                "status": "completed",
                "rectified_items": rectified_items
            }
            
        finally:
            session.close()
    
    # ==================== DATA PROTECTION ====================
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize personal data
        
        Returns:
            Anonymized data
        """
        anonymized = data.copy()
        
        # Define PII fields
        pii_fields = [
            'email', 'name', 'first_name', 'last_name', 'phone',
            'address', 'ssn', 'credit_card', 'ip_address'
        ]
        
        for field in pii_fields:
            if field in anonymized:
                if field == 'email':
                    # Hash email for consistency
                    anonymized[field] = hashlib.sha256(
                        anonymized[field].encode()
                    ).hexdigest()[:8] + "@anonymous.com"
                elif field in ['name', 'first_name', 'last_name']:
                    anonymized[field] = "ANONYMIZED"
                elif field == 'ip_address':
                    anonymized[field] = "0.0.0.0"
                else:
                    anonymized[field] = "[REDACTED]"
        
        return anonymized
    
    def pseudonymize_data(
        self,
        data: Dict[str, Any],
        user_id: str
    ) -> Tuple[Dict[str, Any], str]:
        """
        Pseudonymize personal data
        
        Returns:
            Tuple of (pseudonymized data, pseudonym)
        """
        # Generate consistent pseudonym
        pseudonym = hashlib.sha256(
            f"{user_id}:{self.encryption_key}".encode()
        ).hexdigest()[:16]
        
        pseudonymized = data.copy()
        
        # Replace identifiers with pseudonym
        id_fields = ['user_id', 'customer_id', 'account_id']
        for field in id_fields:
            if field in pseudonymized:
                pseudonymized[field] = pseudonym
        
        # Store mapping securely
        self.redis_client.hset(
            "pseudonym_mapping",
            pseudonym,
            user_id
        )
        
        return pseudonymized, pseudonym
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    # ==================== COMPLIANCE REPORTING ====================
    
    def generate_compliance_report(
        self,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate GDPR compliance report
        
        Returns:
            Compliance report
        """
        session = self.SessionLocal()
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()
            
            report = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "requests": {},
                "consents": {},
                "breaches": [],
                "compliance_score": 0
            }
            
            # Count requests by type
            for request_type in GDPRRequestType:
                query = session.query(DataSubjectRequest).filter(
                    DataSubjectRequest.request_type == request_type.value,
                    DataSubjectRequest.requested_at >= start_date,
                    DataSubjectRequest.requested_at <= end_date
                )
                
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                
                total = query.count()
                completed = query.filter_by(status="completed").count()
                avg_processing_time = self._calculate_avg_processing_time(query.all())
                
                report["requests"][request_type.value] = {
                    "total": total,
                    "completed": completed,
                    "completion_rate": (completed / total * 100) if total > 0 else 0,
                    "avg_processing_days": avg_processing_time
                }
            
            # Consent statistics
            consent_query = session.query(ConsentRecord).filter(
                ConsentRecord.created_at >= start_date,
                ConsentRecord.created_at <= end_date
            )
            
            if tenant_id:
                consent_query = consent_query.filter_by(tenant_id=tenant_id)
            
            total_consents = consent_query.count()
            granted_consents = consent_query.filter_by(granted=True).count()
            
            report["consents"] = {
                "total": total_consents,
                "granted": granted_consents,
                "revoked": total_consents - granted_consents,
                "grant_rate": (granted_consents / total_consents * 100) if total_consents > 0 else 0
            }
            
            # Calculate compliance score
            report["compliance_score"] = self._calculate_compliance_score(report)
            
            return report
            
        finally:
            session.close()
    
    def get_data_mapping(self, tenant_id: Optional[str] = None) -> List[Dict]:
        """
        Get data mapping for all personal data
        
        Returns:
            List of data mappings
        """
        session = self.SessionLocal()
        try:
            query = session.query(PersonalDataRecord)
            
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            
            records = query.all()
            
            return [
                {
                    "category": record.data_category,
                    "type": record.data_type,
                    "location": record.storage_location,
                    "purpose": record.purpose,
                    "legal_basis": record.legal_basis,
                    "retention_days": record.retention_period_days,
                    "encrypted": record.encrypted,
                    "anonymized": record.anonymized,
                    "shared_with": record.shared_with
                }
                for record in records
            ]
            
        finally:
            session.close()
    
    # ==================== HELPER METHODS ====================
    
    def _check_erasure_eligibility(self, user_id: str) -> Tuple[bool, str]:
        """Check if user data can be erased"""
        # Check for legal obligations to retain data
        # This is simplified - real implementation would check various conditions
        
        # Check active contracts
        if self._has_active_contracts(user_id):
            return False, "Active contracts require data retention"
        
        # Check legal holds
        if self._has_legal_holds(user_id):
            return False, "Legal hold prevents data erasure"
        
        # Check financial records retention
        if self._has_recent_financial_records(user_id):
            return False, "Financial records must be retained for legal period"
        
        return True, "Eligible for erasure"
    
    def _has_active_contracts(self, user_id: str) -> bool:
        """Check if user has active contracts"""
        # Implementation would check actual contracts
        return False
    
    def _has_legal_holds(self, user_id: str) -> bool:
        """Check if user data has legal holds"""
        # Implementation would check legal hold database
        return False
    
    def _has_recent_financial_records(self, user_id: str) -> bool:
        """Check if user has recent financial records"""
        # Implementation would check financial records
        return False
    
    def _get_user_profile_data(self, user_id: str) -> Dict:
        """Get user profile data"""
        # This would fetch from actual user database
        return {
            "user_id": user_id,
            "data": "Profile data would be fetched here"
        }
    
    def _get_user_ml_data(self, user_id: str) -> Dict:
        """Get user ML-related data"""
        # This would fetch ML predictions, models, etc.
        return {
            "models_trained": 0,
            "predictions_made": 0,
            "data": "ML data would be fetched here"
        }
    
    def _get_user_usage_data(self, user_id: str) -> Dict:
        """Get user usage data"""
        # This would fetch usage logs, analytics, etc.
        return {
            "last_activity": datetime.utcnow().isoformat(),
            "data": "Usage data would be fetched here"
        }
    
    def _anonymize_data_record(self, record: PersonalDataRecord):
        """Anonymize a data record"""
        record.anonymized = True
        record.user_id = f"ANON_{hashlib.sha256(record.user_id.encode()).hexdigest()[:8]}"
    
    def _anonymize_ml_data(self, user_id: str) -> int:
        """Anonymize ML data for user"""
        # This would anonymize ML models, predictions, etc.
        return 0
    
    def _anonymize_audit_logs(self, user_id: str) -> int:
        """Anonymize audit logs for user"""
        # This would anonymize audit trail entries
        return 0
    
    def _calculate_avg_processing_time(self, requests: List[DataSubjectRequest]) -> float:
        """Calculate average processing time in days"""
        if not requests:
            return 0
        
        total_time = 0
        processed_count = 0
        
        for request in requests:
            if request.processed_at:
                delta = request.processed_at - request.requested_at
                total_time += delta.days
                processed_count += 1
        
        return total_time / processed_count if processed_count > 0 else 0
    
    def _calculate_compliance_score(self, report: Dict) -> float:
        """Calculate overall compliance score"""
        score = 100.0
        
        # Deduct points for incomplete requests
        for request_type, stats in report["requests"].items():
            if stats["total"] > 0:
                incompletion_rate = (1 - stats["completed"] / stats["total"]) * 100
                score -= incompletion_rate * 0.5
                
                # Deduct for slow processing
                if stats["avg_processing_days"] > 30:
                    score -= 10
        
        # Bonus for high consent grant rate
        if report["consents"]["total"] > 0:
            grant_rate = report["consents"]["grant_rate"]
            if grant_rate > 80:
                score += 5
        
        return max(0, min(100, score))
    
    def _process_request_async(self, request_id: str):
        """Process request asynchronously"""
        # This would trigger async processing
        # For now, we'll just log it
        logger.info(f"Async processing triggered for request {request_id}")


# Create singleton instance
rgpd_service = None

def get_rgpd_service() -> RGPDComplianceService:
    """Get or create RGPD service instance"""
    global rgpd_service
    if rgpd_service is None:
        rgpd_service = RGPDComplianceService()
    return rgpd_service
