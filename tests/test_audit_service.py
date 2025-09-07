"""
Tests for Audit Service
=======================
Comprehensive tests for the audit trail service.
"""

import pytest
import json
import hashlib
import base64
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timedelta
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from automl_platform.audit_service import (
    AuditService,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    AuditLogModel,
    Base
)


class TestAuditEvent:
    """Test suite for AuditEvent dataclass."""
    
    def test_audit_event_creation(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_id="test-123",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.LOGIN,
            severity=AuditSeverity.INFO,
            user_id="user123",
            tenant_id="tenant456",
            action="user_login",
            description="User logged in successfully"
        )
        
        assert event.event_id == "test-123"
        assert event.event_type == AuditEventType.LOGIN
        assert event.severity == AuditSeverity.INFO
        assert event.user_id == "user123"
        assert event.tenant_id == "tenant456"
        assert event.action == "user_login"
        assert event.gdpr_relevant == False  # Default value
        assert event.retention_days == 2555  # Default 7 years
    
    def test_audit_event_with_metadata(self):
        """Test audit event with metadata."""
        metadata = {"ip": "192.168.1.1", "browser": "Chrome"}
        event = AuditEvent(
            event_id="test-456",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.DATA_CREATE,
            severity=AuditSeverity.INFO,
            user_id="user123",
            tenant_id=None,
            action="create_dataset",
            description="Dataset created",
            metadata=metadata,
            gdpr_relevant=True
        )
        
        assert event.metadata == metadata
        assert event.gdpr_relevant == True


class TestAuditService:
    """Test suite for AuditService."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis_mock = Mock()
        redis_mock.lpush = Mock()
        redis_mock.hset = Mock()
        redis_mock.hincrby = Mock()
        redis_mock.zadd = Mock()
        redis_mock.pipeline = Mock(return_value=Mock(execute=Mock()))
        redis_mock.publish = Mock()
        return redis_mock
    
    @pytest.fixture
    def in_memory_db(self):
        """Create in-memory SQLite database for testing."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine
    
    @pytest.fixture
    def audit_service(self, mock_redis, in_memory_db):
        """Create audit service with mock Redis and in-memory DB."""
        service = AuditService(
            database_url="sqlite:///:memory:",
            redis_client=mock_redis
        )
        service.engine = in_memory_db
        service.SessionLocal = sessionmaker(bind=in_memory_db)
        return service
    
    def test_initialization(self, audit_service):
        """Test audit service initialization."""
        assert audit_service.redis_client is not None
        assert audit_service.encryption_key is not None
        assert audit_service.last_hash is not None
        assert audit_service.buffer_size == 100
        assert audit_service.flush_interval == 5
    
    def test_generate_encryption_key(self, audit_service):
        """Test encryption key generation."""
        key = audit_service._generate_encryption_key()
        assert key is not None
        assert len(key) == 32  # 256 bits
    
    def test_calculate_hash_chain(self, audit_service):
        """Test hash chain calculation."""
        event = AuditEvent(
            event_id="test-123",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.LOGIN,
            severity=AuditSeverity.INFO,
            user_id="user123",
            tenant_id="tenant456",
            action="login",
            description="User login"
        )
        
        initial_hash = audit_service.last_hash
        new_hash = audit_service._calculate_hash_chain(event)
        
        assert new_hash is not None
        assert new_hash != initial_hash
        assert len(new_hash) == 64  # SHA256 hex digest
        assert audit_service.last_hash == new_hash
    
    def test_encrypt_decrypt_sensitive_data(self, audit_service):
        """Test encryption and decryption of sensitive data."""
        sensitive_data = {
            "username": "testuser",
            "password": "should_be_redacted",
            "email": "user@example.com",
            "token": "secret_token"
        }
        
        # Encrypt
        encrypted = audit_service._encrypt_sensitive_data(sensitive_data)
        assert encrypted is not None
        assert isinstance(encrypted, str)
        
        # Decrypt
        decrypted = audit_service._decrypt_sensitive_data(encrypted)
        assert decrypted is not None
        assert decrypted["username"] == "testuser"
        assert decrypted["password"] == "***REDACTED***"  # Should be redacted
        assert decrypted["token"] == "***REDACTED***"  # Should be redacted
        assert decrypted["email"] == "user@example.com"
    
    def test_log_event(self, audit_service):
        """Test logging an audit event."""
        event_id = audit_service.log_event(
            event_type=AuditEventType.DATA_CREATE,
            action="create_model",
            description="ML model created",
            severity=AuditSeverity.INFO,
            user_id="user123",
            tenant_id="tenant456",
            resource_type="model",
            resource_id="model789",
            metadata={"model_type": "random_forest"}
        )
        
        assert event_id is not None
        assert len(audit_service.audit_buffer) == 1
        
        event = audit_service.audit_buffer[0]
        assert event.event_type == AuditEventType.DATA_CREATE
        assert event.action == "create_model"
        assert event.user_id == "user123"
        assert event.hash_chain is not None
    
    def test_log_critical_event_sends_alert(self, audit_service, mock_redis):
        """Test that critical events trigger alerts."""
        event_id = audit_service.log_event(
            event_type=AuditEventType.SECURITY_ALERT,
            action="unauthorized_access",
            description="Unauthorized access attempt",
            severity=AuditSeverity.CRITICAL,
            user_id="attacker",
            tenant_id="tenant123"
        )
        
        # Verify alert was published to Redis
        mock_redis.publish.assert_called()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "audit:alerts"
        
        alert_data = json.loads(call_args[0][1])
        assert alert_data["event_id"] == event_id
        assert alert_data["severity"] == "critical"
        
        # Verify alert was queued
        mock_redis.lpush.assert_called()
    
    def test_flush_buffer_to_database(self, audit_service):
        """Test flushing audit buffer to database."""
        # Add multiple events to buffer
        for i in range(3):
            audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                action=f"read_data_{i}",
                user_id=f"user{i}",
                tenant_id="tenant123"
            )
        
        assert len(audit_service.audit_buffer) == 3
        
        # Flush buffer
        audit_service.flush()
        
        # Verify buffer is cleared
        assert len(audit_service.audit_buffer) == 0
        
        # Verify events are in database
        session = audit_service.SessionLocal()
        try:
            events = session.query(AuditLogModel).all()
            assert len(events) == 3
            
            # Check events are properly stored
            for i, event in enumerate(events):
                assert event.action == f"read_data_{i}"
                assert event.user_id == f"user{i}"
                assert event.hash_chain is not None
        finally:
            session.close()
    
    def test_auto_flush_when_buffer_full(self, audit_service):
        """Test automatic flush when buffer reaches size limit."""
        audit_service.buffer_size = 5  # Set small buffer for testing
        
        # Add events to exceed buffer size
        for i in range(6):
            audit_service.log_event(
                event_type=AuditEventType.DATA_CREATE,
                action=f"action_{i}",
                user_id="user123"
            )
        
        # Buffer should have been flushed at 5 events
        assert len(audit_service.audit_buffer) == 1  # Only the 6th event
        
        # Check database has the first 5 events
        session = audit_service.SessionLocal()
        try:
            events = session.query(AuditLogModel).all()
            assert len(events) == 5
        finally:
            session.close()
    
    def test_search_audit_logs(self, audit_service):
        """Test searching audit logs with filters."""
        # Add some test events
        now = datetime.utcnow()
        
        # Create events directly in database
        session = audit_service.SessionLocal()
        try:
            events = [
                AuditLogModel(
                    event_id=uuid.uuid4(),
                    timestamp=now - timedelta(hours=1),
                    event_type=AuditEventType.LOGIN.value,
                    severity=AuditSeverity.INFO.value,
                    user_id="user1",
                    tenant_id="tenant1",
                    action="login",
                    description="User login",
                    gdpr_relevant=False
                ),
                AuditLogModel(
                    event_id=uuid.uuid4(),
                    timestamp=now - timedelta(minutes=30),
                    event_type=AuditEventType.DATA_CREATE.value,
                    severity=AuditSeverity.INFO.value,
                    user_id="user2",
                    tenant_id="tenant1",
                    action="create_dataset",
                    description="Dataset created",
                    gdpr_relevant=True
                ),
                AuditLogModel(
                    event_id=uuid.uuid4(),
                    timestamp=now - timedelta(minutes=10),
                    event_type=AuditEventType.SECURITY_ALERT.value,
                    severity=AuditSeverity.CRITICAL.value,
                    user_id="user1",
                    tenant_id="tenant2",
                    action="suspicious_activity",
                    description="Suspicious activity detected",
                    gdpr_relevant=False
                )
            ]
            
            for event in events:
                session.add(event)
            session.commit()
        finally:
            session.close()
        
        # Search by user
        results = audit_service.search(user_id="user1")
        assert len(results) == 2
        
        # Search by tenant
        results = audit_service.search(tenant_id="tenant1")
        assert len(results) == 2
        
        # Search by event type
        results = audit_service.search(event_type=AuditEventType.LOGIN.value)
        assert len(results) == 1
        
        # Search by severity
        results = audit_service.search(severity=AuditSeverity.CRITICAL.value)
        assert len(results) == 1
        
        # Search GDPR relevant only
        results = audit_service.search(gdpr_only=True)
        assert len(results) == 1
        assert results[0]["action"] == "create_dataset"
        
        # Search with date range
        results = audit_service.search(
            start_date=now - timedelta(hours=2),
            end_date=now - timedelta(minutes=20)
        )
        assert len(results) == 2  # Excludes the most recent event
    
    def test_get_user_activity(self, audit_service):
        """Test getting user activity summary."""
        # Add events for a user
        user_id = "test_user"
        
        session = audit_service.SessionLocal()
        try:
            # Import func here to avoid issues
            from sqlalchemy import func
            
            # Add various events
            for i in range(5):
                event = AuditLogModel(
                    event_id=uuid.uuid4(),
                    timestamp=datetime.utcnow() - timedelta(days=i),
                    event_type=AuditEventType.LOGIN.value if i % 2 == 0 else AuditEventType.DATA_READ.value,
                    severity=AuditSeverity.INFO.value,
                    user_id=user_id,
                    tenant_id="tenant123",
                    action=f"action_{i}",
                    description=f"Event {i}"
                )
                session.add(event)
            session.commit()
            
            # Get activity summary
            activity = audit_service.get_user_activity(user_id, days=30)
            
            assert activity["user_id"] == user_id
            assert activity["period_days"] == 30
            assert "event_counts" in activity
            assert "recent_events" in activity
            assert activity["total_events"] >= 5
        finally:
            session.close()
    
    def test_verify_integrity(self, audit_service):
        """Test audit log integrity verification."""
        # Add events with proper hash chain
        events_to_add = []
        previous_hash = audit_service.last_hash
        
        session = audit_service.SessionLocal()
        try:
            for i in range(3):
                event_id = str(uuid.uuid4())
                timestamp = datetime.utcnow()
                
                # Calculate hash
                event_data = {
                    "event_id": event_id,
                    "timestamp": timestamp.isoformat(),
                    "event_type": AuditEventType.DATA_READ.value,
                    "user_id": f"user{i}",
                    "resource_id": f"resource{i}",
                    "action": f"read_{i}"
                }
                
                combined = f"{previous_hash}:{json.dumps(event_data, sort_keys=True)}"
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                
                event = AuditLogModel(
                    event_id=uuid.UUID(event_id),
                    timestamp=timestamp,
                    event_type=AuditEventType.DATA_READ.value,
                    severity=AuditSeverity.INFO.value,
                    user_id=f"user{i}",
                    resource_id=f"resource{i}",
                    action=f"read_{i}",
                    description=f"Read operation {i}",
                    hash_chain=new_hash
                )
                session.add(event)
                previous_hash = new_hash
            
            session.commit()
            
            # Verify integrity - should pass
            result = audit_service.verify_integrity()
            assert result["status"] == "verified"
            assert result["events_checked"] == 3
            assert len(result["broken_links"]) == 0
            assert result["integrity_score"] == 100.0
            
            # Now tamper with an event
            event_to_tamper = session.query(AuditLogModel).first()
            event_to_tamper.hash_chain = "tampered_hash"
            session.commit()
            
            # Verify integrity - should fail
            result = audit_service.verify_integrity()
            assert result["status"] == "tampered"
            assert len(result["broken_links"]) > 0
            assert result["integrity_score"] < 100.0
        finally:
            session.close()
    
    def test_export_for_compliance(self, audit_service):
        """Test exporting audit logs for compliance."""
        # Add some GDPR-relevant events
        session = audit_service.SessionLocal()
        try:
            for i in range(3):
                event = AuditLogModel(
                    event_id=uuid.uuid4(),
                    timestamp=datetime.utcnow(),
                    event_type=AuditEventType.DATA_DELETE.value,
                    severity=AuditSeverity.INFO.value,
                    user_id=f"user{i}",
                    tenant_id="tenant123",
                    action=f"delete_{i}",
                    description=f"Data deletion {i}",
                    gdpr_relevant=True
                )
                session.add(event)
            session.commit()
        finally:
            session.close()
        
        # Export as JSON
        json_export = audit_service.export_for_compliance(
            tenant_id="tenant123",
            format="json",
            gdpr_only=True
        )
        assert json_export is not None
        data = json.loads(json_export)
        assert len(data) == 3
        
        # Export as CSV
        csv_export = audit_service.export_for_compliance(
            tenant_id="tenant123",
            format="csv",
            gdpr_only=True
        )
        assert b"event_id" in csv_export  # CSV header
        
        # Export as compressed
        compressed_export = audit_service.export_for_compliance(
            tenant_id="tenant123",
            format="compressed",
            gdpr_only=True
        )
        assert compressed_export is not None
        
        # Decompress and verify
        import gzip
        decompressed = gzip.decompress(compressed_export)
        data = json.loads(decompressed)
        assert len(data) == 3
    
    def test_cleanup_expired_logs(self, audit_service):
        """Test cleanup of expired audit logs."""
        session = audit_service.SessionLocal()
        try:
            now = datetime.utcnow()
            
            # Add expired events
            for i in range(3):
                event = AuditLogModel(
                    event_id=uuid.uuid4(),
                    timestamp=now - timedelta(days=365),
                    event_type=AuditEventType.DATA_READ.value,
                    severity=AuditSeverity.INFO.value,
                    user_id=f"user{i}",
                    tenant_id="tenant123",
                    action=f"read_{i}",
                    description=f"Old event {i}",
                    retention_days=1,
                    retention_expires=now - timedelta(days=1)  # Already expired
                )
                session.add(event)
            
            # Add non-expired event
            event = AuditLogModel(
                event_id=uuid.uuid4(),
                timestamp=now,
                event_type=AuditEventType.DATA_CREATE.value,
                severity=AuditSeverity.INFO.value,
                user_id="user_current",
                tenant_id="tenant123",
                action="create",
                description="Current event",
                retention_days=7,
                retention_expires=now + timedelta(days=7)
            )
            session.add(event)
            session.commit()
            
            # Run cleanup
            deleted = audit_service.cleanup_expired()
            assert deleted == 3
            
            # Verify only non-expired event remains
            remaining = session.query(AuditLogModel).all()
            assert len(remaining) == 1
            assert remaining[0].user_id == "user_current"
        finally:
            session.close()
    
    def test_track_in_redis(self, audit_service, mock_redis):
        """Test tracking events in Redis for real-time analytics."""
        event = AuditEvent(
            event_id="test-123",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.LOGIN,
            severity=AuditSeverity.INFO,
            user_id="user123",
            tenant_id="tenant456",
            action="login",
            description="User login"
        )
        
        # Track event
        audit_service._track_in_redis(event)
        
        # Verify Redis operations
        pipe_mock = mock_redis.pipeline.return_value
        
        # Should increment counter
        pipe_mock.hincrby.assert_called()
        
        # Should add to user activity
        pipe_mock.zadd.assert_called()
        
        # Pipeline should be executed
        pipe_mock.execute.assert_called_once()
    
    def test_send_alert(self, audit_service, mock_redis):
        """Test sending alerts for critical events."""
        event = AuditEvent(
            event_id="alert-123",
            timestamp=datetime.utcnow(),
            event_type=AuditEventType.SECURITY_ALERT,
            severity=AuditSeverity.CRITICAL,
            user_id="user123",
            tenant_id="tenant456",
            action="security_breach",
            description="Security breach detected"
        )
        
        audit_service._send_alert(event)
        
        # Verify alert was published
        mock_redis.publish.assert_called_once()
        channel, message = mock_redis.publish.call_args[0]
        assert channel == "audit:alerts"
        
        alert_data = json.loads(message)
        assert alert_data["event_id"] == "alert-123"
        assert alert_data["severity"] == "critical"
        
        # Verify alert was queued
        mock_redis.lpush.assert_called_once()
        queue, message = mock_redis.lpush.call_args[0]
        assert queue == "audit:alert_queue"


class TestAuditEventTypes:
    """Test audit event types enum."""
    
    def test_authentication_events(self):
        """Test authentication-related event types."""
        assert AuditEventType.LOGIN.value == "login"
        assert AuditEventType.LOGOUT.value == "logout"
        assert AuditEventType.LOGIN_FAILED.value == "login_failed"
        assert AuditEventType.TOKEN_CREATED.value == "token_created"
        assert AuditEventType.TOKEN_REVOKED.value == "token_revoked"
        assert AuditEventType.PASSWORD_CHANGED.value == "password_changed"
    
    def test_data_operation_events(self):
        """Test data operation event types."""
        assert AuditEventType.DATA_CREATE.value == "data_create"
        assert AuditEventType.DATA_READ.value == "data_read"
        assert AuditEventType.DATA_UPDATE.value == "data_update"
        assert AuditEventType.DATA_DELETE.value == "data_delete"
        assert AuditEventType.DATA_EXPORT.value == "data_export"
        assert AuditEventType.DATA_IMPORT.value == "data_import"
    
    def test_model_operation_events(self):
        """Test model operation event types."""
        assert AuditEventType.MODEL_TRAIN.value == "model_train"
        assert AuditEventType.MODEL_PREDICT.value == "model_predict"
        assert AuditEventType.MODEL_DEPLOY.value == "model_deploy"
        assert AuditEventType.MODEL_DELETE.value == "model_delete"
        assert AuditEventType.MODEL_EXPORT.value == "model_export"
    
    def test_security_events(self):
        """Test security event types."""
        assert AuditEventType.SECURITY_ALERT.value == "security_alert"
        assert AuditEventType.UNAUTHORIZED_ACCESS.value == "unauthorized_access"
        assert AuditEventType.SUSPICIOUS_ACTIVITY.value == "suspicious_activity"
    
    def test_compliance_events(self):
        """Test compliance event types."""
        assert AuditEventType.GDPR_REQUEST.value == "gdpr_request"
        assert AuditEventType.DATA_RETENTION.value == "data_retention"
        assert AuditEventType.CONSENT_UPDATE.value == "consent_update"


class TestAuditSeverityLevels:
    """Test audit severity levels."""
    
    def test_severity_levels(self):
        """Test all severity levels."""
        assert AuditSeverity.INFO.value == "info"
        assert AuditSeverity.WARNING.value == "warning"
        assert AuditSeverity.ERROR.value == "error"
        assert AuditSeverity.CRITICAL.value == "critical"
