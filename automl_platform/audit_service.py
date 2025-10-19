"""
Enterprise Audit Trail Service
===============================
Place in: automl_platform/audit_service.py

Comprehensive audit logging with search, compliance reporting,
and immutable storage.
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import gzip
import base64

import redis
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    Text,
)

from automl_platform.database import get_audit_engine, get_audit_sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import pandas as pd
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from automl_platform.config import DatabaseConfig
from automl_platform.models.base import (
    AuditBase,
    audit_table_args,
    database_supports_schemas,
)

logger = logging.getLogger(__name__)

_AUDIT_URL_OVERRIDE = (
    os.getenv("AUTOML_AUDIT_DATABASE_URL")
    or os.getenv("AUDIT_DATABASE_URL")
)

if _AUDIT_URL_OVERRIDE:
    _AUDIT_SUPPORTS_SCHEMAS = database_supports_schemas(_AUDIT_URL_OVERRIDE)
else:
    _db_config = DatabaseConfig()
    _AUDIT_URL_FALLBACK = getattr(_db_config, "audit_url", None) or getattr(_db_config, "url", None)
    _AUDIT_SUPPORTS_SCHEMAS = database_supports_schemas(_AUDIT_URL_FALLBACK)

# Base declarative registry for audit tables
Base = AuditBase

_REMOTE_USERS = Table(
    'users',
    MetaData(),
    Column('id', UUID(as_uuid=True)),
)
_REMOTE_TENANTS = Table(
    'tenants',
    MetaData(),
    Column('id', UUID(as_uuid=True)),
)


def _configure_audit_schema(supports_schemas: bool) -> None:
    """Update audit table metadata for the current database backend."""

    schema_args = audit_table_args(supports_schemas)
    schema = schema_args.get("schema")

    # Align declarative metadata
    AuditBase.metadata.schema = schema

    # Ensure the mapped table reflects the new schema configuration
    table = AuditLogModel.__table__ if 'AuditLogModel' in globals() else None
    if table is not None:
        table.schema = schema

    # Foreign-key targets should live in the public schema when supported
    remote_schema = 'public' if supports_schemas else None
    for remote in (_REMOTE_USERS, _REMOTE_TENANTS):
        remote.schema = remote_schema

    if table is not None:
        table_args = list(AuditLogModel._base_table_args)
        if schema_args:
            table_args.append(schema_args)
        AuditLogModel.__table_args__ = tuple(table_args)



class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    TOKEN_CREATED = "token_created"
    TOKEN_REVOKED = "token_revoked"
    PASSWORD_CHANGED = "password_changed"
    
    # Data operations
    DATA_CREATE = "data_create"
    DATA_READ = "data_read"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    
    # Model operations
    MODEL_TRAIN = "model_train"
    MODEL_PREDICT = "model_predict"
    MODEL_DEPLOY = "model_deploy"
    MODEL_DELETE = "model_delete"
    MODEL_EXPORT = "model_export"
    
    # Administrative
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"
    
    # Security
    SECURITY_ALERT = "security_alert"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    
    # Compliance
    GDPR_REQUEST = "gdpr_request"
    DATA_RETENTION = "data_retention"
    CONSENT_UPDATE = "consent_update"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity

    
    # Actor information
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    
    # Event details
    action: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Request/Response
    request_id: Optional[str] = None
    request_method: Optional[str] = None
    request_path: Optional[str] = None
    request_data: Optional[Dict] = None
    response_status: Optional[int] = None
    response_time_ms: Optional[float] = None
    
    # Compliance
    gdpr_relevant: bool = False
    retention_days: int = 2555  # 7 years default
    
    # Security
    hash_chain: Optional[str] = None
    signature: Optional[str] = None


class AuditLogModel(Base):
    """SQLAlchemy model for audit logs"""
    __tablename__ = "audit_logs_v2"

    # Primary key
    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Timestamp with index for range queries
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Event information
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    
    # Actor information with indexes
    user_id = Column(UUID(as_uuid=True), index=True)
    tenant_id = Column(UUID(as_uuid=True), index=True)
    session_id = Column(String(255))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Resource information with indexes
    resource_type = Column(String(50), index=True)
    resource_id = Column(String(255), index=True)
    resource_name = Column(String(255))
    
    # Event details
    action = Column(String(255), nullable=False)
    description = Column(Text)
    event_metadata = Column("metadata", JSON)
    
    # Request/Response
    request_id = Column(String(255), index=True)
    request_method = Column(String(10))
    request_path = Column(String(500))
    request_data = Column(JSON)  # Encrypted
    response_status = Column(Integer)
    response_time_ms = Column(Integer)
    
    # Compliance
    gdpr_relevant = Column(Boolean, default=False, index=True)
    retention_days = Column(Integer, default=2555)
    retention_expires = Column(DateTime, index=True)
    
    # Security - for tamper detection
    hash_chain = Column(String(255))  # Hash of previous event + current
    signature = Column(Text)  # Digital signature
    
    # Indexes for common queries
    _base_table_args = [
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_tenant_timestamp', 'tenant_id', 'timestamp'),
        Index('idx_resource', 'resource_type', 'resource_id'),
        Index('idx_retention', 'retention_expires'),
    ]

    _base_table_args.append(
        ForeignKeyConstraint(
            ['user_id'],
            [_REMOTE_USERS.c.id],
            link_to_name=True,
            use_alter=True,
            name='fk_audit_logs_user_id',
        )
    )
    _base_table_args.append(
        ForeignKeyConstraint(
            ['tenant_id'],
            [_REMOTE_TENANTS.c.id],
            link_to_name=True,
            use_alter=True,
            name='fk_audit_logs_tenant_id',
        )
    )

    __table_args__ = tuple(_base_table_args)


# Ensure table metadata reflects the resolved schema support at import time
_configure_audit_schema(_AUDIT_SUPPORTS_SCHEMAS)


class AuditService:
    """
    Comprehensive audit service with immutable logging
    """
    
    def __init__(
        self,
        database_url: str = None,
        redis_client: redis.Redis = None,
        encryption_key: bytes = None
    ):
        # Database setup
        explicit_database_url = database_url
        if explicit_database_url is not None:
            resolved_database_url = explicit_database_url
            self.engine = get_audit_engine(explicit_database_url)
            self.SessionLocal = get_audit_sessionmaker(explicit_database_url)
        else:
            resolved_database_url = (
                os.getenv("AUTOML_AUDIT_DATABASE_URL")
                or os.getenv("AUDIT_DATABASE_URL")
                or getattr(DatabaseConfig(), "audit_url", None)
                or getattr(DatabaseConfig(), "url", None)
                or "postgresql://user:pass@localhost/audit"
            )

            os.environ.setdefault("AUTOML_AUDIT_DATABASE_URL", resolved_database_url)
            self.engine = get_audit_engine()
            self.SessionLocal = get_audit_sessionmaker()

        self.database_url = resolved_database_url

        # Align metadata, table args, and remote schema references with the
        # resolved database URL. This is required when explicit URLs override
        # the import-time configuration or when environment variables change
        # after module import.
        actual_support = database_supports_schemas(resolved_database_url)
        _configure_audit_schema(actual_support)

        Base.metadata.create_all(self.engine)
        
        # Redis for caching and real-time alerts
        self.redis_client = redis_client or redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        
        # Encryption for sensitive data
        self.encryption_key = encryption_key or self._generate_encryption_key()
        
        # Hash chain for tamper detection
        self.last_hash = self._get_last_hash()
        
        # Audit buffer for batch writing
        self.audit_buffer = []
        self.buffer_size = 100
        self.flush_interval = 5  # seconds
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key from environment or create new"""
        key_string = os.getenv("AUTOML_AUDIT_ENCRYPTION_KEY") or os.getenv("AUDIT_ENCRYPTION_KEY")
        if key_string:
            return base64.b64decode(key_string)
        
        # Generate new key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',  # In production, use proper salt management
            iterations=100000,
        )
        return kdf.derive(b'default_key')  # In production, use secure key
    
    def _get_last_hash(self) -> str:
        """Get the hash of the last audit event"""
        session = self.SessionLocal()
        try:
            last_event = session.query(AuditLogModel).order_by(
                AuditLogModel.timestamp.desc()
            ).first()
            
            if last_event and last_event.hash_chain:
                return last_event.hash_chain
            
            # Genesis hash
            return hashlib.sha256(b"AUDIT_GENESIS").hexdigest()
            
        finally:
            session.close()
    
    def _calculate_hash_chain(self, event: AuditEvent) -> str:
        """Calculate hash chain for tamper detection"""
        # Serialize event data
        event_data = {
            "event_id": str(event.event_id),
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "user_id": event.user_id,
            "resource_id": event.resource_id,
            "action": event.action
        }
        
        # Combine with previous hash
        combined = f"{self.last_hash}:{json.dumps(event_data, sort_keys=True)}"
        
        # Calculate new hash
        new_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        # Update last hash
        self.last_hash = new_hash
        
        return new_hash
    
    def _encrypt_sensitive_data(self, data: Dict) -> str:
        """Encrypt sensitive data in audit logs"""
        if not data:
            return None
        
        from cryptography.fernet import Fernet
        
        # Remove sensitive fields before encryption
        safe_data = data.copy()
        sensitive_fields = ['password', 'token', 'secret', 'key', 'credential']
        
        for field in sensitive_fields:
            if field in safe_data:
                safe_data[field] = "***REDACTED***"
        
        # Encrypt
        f = Fernet(base64.urlsafe_b64encode(self.encryption_key))
        encrypted = f.encrypt(json.dumps(safe_data).encode())
        
        return base64.b64encode(encrypted).decode()
    
    def _decrypt_sensitive_data(self, encrypted_data: str) -> Dict:
        """Decrypt sensitive data from audit logs"""
        if not encrypted_data:
            return None
        
        from cryptography.fernet import Fernet
        
        f = Fernet(base64.urlsafe_b64encode(self.encryption_key))
        decrypted = f.decrypt(base64.b64decode(encrypted_data))
        
        return json.loads(decrypted)
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        description: str = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        user_id: str = None,
        tenant_id: str = None,
        resource_type: str = None,
        resource_id: str = None,
        resource_name: str = None,
        metadata: Dict = None,
        request_data: Dict = None,
        response_status: int = None,
        gdpr_relevant: bool = False,
        **kwargs
    ) -> str:
        """
        Log an audit event
        
        Returns:
            Event ID
        """
        # Create event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            tenant_id=tenant_id,
            action=action,
            description=description or action,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            metadata=metadata or {},
            request_data=request_data,
            response_status=response_status,
            gdpr_relevant=gdpr_relevant,
            **kwargs
        )
        
        # Calculate hash chain
        event.hash_chain = self._calculate_hash_chain(event)
        
        # Add to buffer
        self.audit_buffer.append(event)
        
        # Flush if buffer is full
        if len(self.audit_buffer) >= self.buffer_size:
            self.flush()
        
        # Real-time alerting for critical events
        if severity == AuditSeverity.CRITICAL:
            self._send_alert(event)
        
        # Track in Redis for real-time analytics
        self._track_in_redis(event)
        
        return event.event_id
    
    def flush(self):
        """Flush audit buffer to database"""
        if not self.audit_buffer:
            return
        
        session = self.SessionLocal()
        try:
            for event in self.audit_buffer:
                # Create database model
                db_event = AuditLogModel(
                    event_id=uuid.UUID(event.event_id),
                    timestamp=event.timestamp,
                    event_type=event.event_type.value,
                    severity=event.severity.value,
                    user_id=event.user_id,
                    tenant_id=event.tenant_id,
                    session_id=event.session_id,
                    ip_address=event.ip_address,
                    user_agent=event.user_agent,
                    resource_type=event.resource_type,
                    resource_id=event.resource_id,
                    resource_name=event.resource_name,
                    action=event.action,
                    description=event.description,
                    event_metadata=event.metadata,
                    request_id=event.request_id,
                    request_method=event.request_method,
                    request_path=event.request_path,
                    request_data=self._encrypt_sensitive_data(event.request_data),
                    response_status=event.response_status,
                    response_time_ms=event.response_time_ms,
                    gdpr_relevant=event.gdpr_relevant,
                    retention_days=event.retention_days,
                    retention_expires=event.timestamp + timedelta(days=event.retention_days),
                    hash_chain=event.hash_chain,
                    signature=event.signature
                )
                
                session.add(db_event)
            
            session.commit()
            self.audit_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush audit logs: {e}")
            session.rollback()
            # Store failed logs for retry
            self._store_failed_logs(self.audit_buffer)
        finally:
            session.close()
    
    def _store_failed_logs(self, events: List[AuditEvent]):
        """Store failed logs for later retry"""
        for event in events:
            self.redis_client.lpush(
                "audit:failed_logs",
                json.dumps(asdict(event), default=str)
            )
    
    def _track_in_redis(self, event: AuditEvent):
        """Track event in Redis for real-time analytics"""
        # Increment counters
        pipe = self.redis_client.pipeline()
        
        # Event type counter
        pipe.hincrby(
            f"audit:counts:{datetime.utcnow().strftime('%Y-%m-%d')}",
            event.event_type.value,
            1
        )
        
        # User activity
        if event.user_id:
            pipe.zadd(
                f"audit:user_activity:{event.user_id}",
                {event.event_id: event.timestamp.timestamp()}
            )
        
        # Tenant activity
        if event.tenant_id:
            pipe.zadd(
                f"audit:tenant_activity:{event.tenant_id}",
                {event.event_id: event.timestamp.timestamp()}
            )
        
        pipe.execute()
    
    def _send_alert(self, event: AuditEvent):
        """Send alert for critical events"""
        alert = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "user_id": event.user_id,
            "description": event.description
        }
        
        # Publish to Redis for real-time subscribers
        self.redis_client.publish("audit:alerts", json.dumps(alert))
        
        # Store in alert queue
        self.redis_client.lpush("audit:alert_queue", json.dumps(alert))
    
    def search(
        self,
        tenant_id: str = None,
        user_id: str = None,
        event_type: str = None,
        resource_type: str = None,
        resource_id: str = None,
        severity: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        gdpr_only: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Search audit logs with filters
        
        Returns:
            List of audit events
        """
        session = self.SessionLocal()
        try:
            query = session.query(AuditLogModel)
            
            # Apply filters
            if tenant_id:
                query = query.filter(AuditLogModel.tenant_id == tenant_id)
            if user_id:
                query = query.filter(AuditLogModel.user_id == user_id)
            if event_type:
                query = query.filter(AuditLogModel.event_type == event_type)
            if resource_type:
                query = query.filter(AuditLogModel.resource_type == resource_type)
            if resource_id:
                query = query.filter(AuditLogModel.resource_id == resource_id)
            if severity:
                query = query.filter(AuditLogModel.severity == severity)
            if start_date:
                query = query.filter(AuditLogModel.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditLogModel.timestamp <= end_date)
            if gdpr_only:
                query = query.filter(AuditLogModel.gdpr_relevant == True)
            
            # Order and paginate
            query = query.order_by(AuditLogModel.timestamp.desc())
            query = query.limit(limit).offset(offset)
            
            # Execute and convert to dict
            results = []
            for event in query.all():
                event_dict = {
                    "event_id": str(event.event_id),
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "user_id": event.user_id,
                    "tenant_id": event.tenant_id,
                    "resource_type": event.resource_type,
                    "resource_id": event.resource_id,
                    "resource_name": event.resource_name,
                    "action": event.action,
                    "description": event.description,
                    "metadata": event.event_metadata,
                    "response_status": event.response_status,
                    "gdpr_relevant": event.gdpr_relevant
                }
                
                # Decrypt request data if needed
                if event.request_data:
                    try:
                        event_dict["request_data"] = self._decrypt_sensitive_data(
                            event.request_data
                        )
                    except:
                        event_dict["request_data"] = "***ENCRYPTED***"
                
                results.append(event_dict)
            
            return results
            
        finally:
            session.close()
    
    def get_user_activity(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get user activity summary"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        session = self.SessionLocal()
        try:
            # Get event counts by type
            events = session.query(
                AuditLogModel.event_type,
                func.count(AuditLogModel.event_id).label('count')
            ).filter(
                AuditLogModel.user_id == user_id,
                AuditLogModel.timestamp >= start_date
            ).group_by(AuditLogModel.event_type).all()
            
            # Get recent events
            recent = session.query(AuditLogModel).filter(
                AuditLogModel.user_id == user_id
            ).order_by(
                AuditLogModel.timestamp.desc()
            ).limit(10).all()
            
            return {
                "user_id": user_id,
                "period_days": days,
                "event_counts": {e.event_type: e.count for e in events},
                "total_events": sum(e.count for e in events),
                "recent_events": [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "event_type": e.event_type,
                        "action": e.action,
                        "resource": f"{e.resource_type}:{e.resource_id}"
                    }
                    for e in recent
                ]
            }
            
        finally:
            session.close()
    
    def verify_integrity(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """
        Verify audit log integrity using hash chain
        
        Returns:
            Integrity verification results
        """
        session = self.SessionLocal()
        try:
            query = session.query(AuditLogModel).order_by(
                AuditLogModel.timestamp
            )
            
            if start_date:
                query = query.filter(AuditLogModel.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditLogModel.timestamp <= end_date)
            
            events = query.all()
            
            if not events:
                return {
                    "status": "no_events",
                    "message": "No events found in specified range"
                }
            
            # Verify hash chain
            previous_hash = self._get_hash_before(events[0].timestamp)
            broken_links = []
            
            for i, event in enumerate(events):
                # Recalculate expected hash
                event_data = {
                    "event_id": str(event.event_id),
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "user_id": event.user_id,
                    "resource_id": event.resource_id,
                    "action": event.action
                }
                
                combined = f"{previous_hash}:{json.dumps(event_data, sort_keys=True)}"
                expected_hash = hashlib.sha256(combined.encode()).hexdigest()
                
                if event.hash_chain != expected_hash:
                    broken_links.append({
                        "position": i,
                        "event_id": str(event.event_id),
                        "timestamp": event.timestamp.isoformat(),
                        "expected_hash": expected_hash,
                        "actual_hash": event.hash_chain
                    })
                
                previous_hash = event.hash_chain
            
            return {
                "status": "verified" if not broken_links else "tampered",
                "events_checked": len(events),
                "broken_links": broken_links,
                "integrity_score": (len(events) - len(broken_links)) / len(events) * 100
            }
            
        finally:
            session.close()
    
    def _get_hash_before(self, timestamp: datetime) -> str:
        """Get the hash of the event before given timestamp"""
        session = self.SessionLocal()
        try:
            event = session.query(AuditLogModel).filter(
                AuditLogModel.timestamp < timestamp
            ).order_by(
                AuditLogModel.timestamp.desc()
            ).first()
            
            if event and event.hash_chain:
                return event.hash_chain
            
            return hashlib.sha256(b"AUDIT_GENESIS").hexdigest()
            
        finally:
            session.close()
    
    def export_for_compliance(
        self,
        tenant_id: str,
        format: str = "json",
        gdpr_only: bool = True
    ) -> bytes:
        """
        Export audit logs for compliance reporting
        
        Returns:
            Exported data as bytes
        """
        # Get relevant logs
        logs = self.search(
            tenant_id=tenant_id,
            gdpr_only=gdpr_only,
            limit=10000  # Adjust as needed
        )
        
        if format == "json":
            return json.dumps(logs, indent=2).encode()
        
        elif format == "csv":
            df = pd.DataFrame(logs)
            return df.to_csv(index=False).encode()
        
        elif format == "compressed":
            data = json.dumps(logs).encode()
            return gzip.compress(data)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired audit logs based on retention policy
        
        Returns:
            Number of deleted records
        """
        session = self.SessionLocal()
        try:
            # Find expired records
            expired = session.query(AuditLogModel).filter(
                AuditLogModel.retention_expires <= datetime.utcnow()
            ).all()
            
            count = len(expired)
            
            # Archive before deletion (optional)
            for event in expired:
                self._archive_event(event)
            
            # Delete expired records
            session.query(AuditLogModel).filter(
                AuditLogModel.retention_expires <= datetime.utcnow()
            ).delete()
            
            session.commit()
            
            logger.info(f"Cleaned up {count} expired audit logs")
            return count
            
        finally:
            session.close()
    
    def _archive_event(self, event: AuditLogModel):
        """Archive event before deletion"""
        # This could write to cold storage, S3, etc.
        archive_data = {
            "event_id": str(event.event_id),
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type,
            "user_id": event.user_id,
            "tenant_id": event.tenant_id,
            "hash_chain": event.hash_chain
        }
        
        # Store in Redis with long TTL as example
        self.redis_client.setex(
            f"audit:archive:{event.event_id}",
            86400 * 365,  # 1 year
            json.dumps(archive_data)
        )


# Import for backward compatibility
from sqlalchemy import func
