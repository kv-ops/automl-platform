"""
WebSocket API for Real-time ML Platform Communication
======================================================
Place in: automl_platform/api/websocket.py

Provides real-time updates for:
- Training progress
- Model metrics
- Drift detection alerts
- Cache invalidation events
- Pipeline status
- Billing updates
"""

import json
import asyncio
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBearer
import redis.asyncio as aioredis

# Internal imports
from ..auth import get_current_user, User, TokenService
from ..monitoring import ModelMonitor, DriftDetector, AlertManager
from ..orchestrator import AutoMLOrchestrator
from ..mlops_service import MLflowRegistry, RetrainingService
from ..pipeline_cache import PipelineCache
from ..scheduler import JobRequest, JobStatus

logger = logging.getLogger(__name__)

# ============================================================================
# WebSocket Event Types
# ============================================================================

class EventType(Enum):
    """WebSocket event types"""
    # Training events
    TRAINING_STARTED = "training_started"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    
    # Model events
    MODEL_REGISTERED = "model_registered"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_RETRAINED = "model_retrained"
    
    # Monitoring events
    METRICS_UPDATE = "metrics_update"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADED = "performance_degraded"
    ALERT_TRIGGERED = "alert_triggered"
    
    # Cache events
    CACHE_INVALIDATED = "cache_invalidated"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    
    # Job events
    JOB_QUEUED = "job_queued"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    
    # System events
    CONNECTION_ESTABLISHED = "connection_established"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    
    # Billing events
    QUOTA_WARNING = "quota_warning"
    QUOTA_EXCEEDED = "quota_exceeded"
    BILLING_UPDATE = "billing_update"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: str = None
    message_id: str = None
    tenant_id: str = None
    user_id: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id
        })


# ============================================================================
# Connection Manager
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections with authentication and rooms"""
    
    def __init__(self):
        # Active connections by user ID
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Room subscriptions (room_id -> set of user_ids)
        self.rooms: Dict[str, Set[str]] = {}
        
        # User metadata
        self.user_metadata: Dict[str, Dict] = {}
        
        # Redis for distributed WebSocket support
        self.redis_client = None
        self.pubsub = None
        
    async def initialize_redis(self, redis_url: str = "redis://localhost:6379"):
        """Initialize Redis for distributed WebSocket support"""
        try:
            self.redis_client = await aioredis.from_url(redis_url)
            self.pubsub = self.redis_client.pubsub()
            logger.info("Redis initialized for WebSocket distribution")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}. Using in-memory only.")
    
    async def connect(self, websocket: WebSocket, user: User):
        """Accept WebSocket connection"""
        await websocket.accept()
        
        user_id = str(user.id)
        self.active_connections[user_id] = websocket
        self.user_metadata[user_id] = {
            "username": user.username,
            "tenant_id": str(user.tenant_id) if user.tenant_id else None,
            "plan_type": user.plan_type,
            "connected_at": datetime.utcnow().isoformat()
        }
        
        # Auto-join tenant room
        if user.tenant_id:
            await self.join_room(user_id, f"tenant_{user.tenant_id}")
        
        # Send connection confirmation
        await self.send_personal_message(
            WebSocketMessage(
                event_type=EventType.CONNECTION_ESTABLISHED,
                data={"message": "Connected to ML Platform WebSocket"},
                user_id=user_id,
                tenant_id=str(user.tenant_id) if user.tenant_id else None
            ),
            user_id
        )
        
        logger.info(f"User {user.username} connected via WebSocket")
    
    async def disconnect(self, user_id: str):
        """Disconnect WebSocket"""
        if user_id in self.active_connections:
            # Leave all rooms
            for room_id in list(self.rooms.keys()):
                if user_id in self.rooms[room_id]:
                    self.rooms[room_id].discard(user_id)
                    if not self.rooms[room_id]:
                        del self.rooms[room_id]
            
            # Remove connection
            del self.active_connections[user_id]
            del self.user_metadata[user_id]
            
            logger.info(f"User {user_id} disconnected from WebSocket")
    
    async def join_room(self, user_id: str, room_id: str):
        """Join a room for group messaging"""
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        
        self.rooms[room_id].add(user_id)
        
        # Subscribe to Redis channel if available
        if self.pubsub:
            await self.pubsub.subscribe(room_id)
        
        logger.debug(f"User {user_id} joined room {room_id}")
    
    async def leave_room(self, user_id: str, room_id: str):
        """Leave a room"""
        if room_id in self.rooms:
            self.rooms[room_id].discard(user_id)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
                
                # Unsubscribe from Redis channel
                if self.pubsub:
                    await self.pubsub.unsubscribe(room_id)
        
        logger.debug(f"User {user_id} left room {room_id}")
    
    async def send_personal_message(self, message: WebSocketMessage, user_id: str):
        """Send message to specific user"""
        if user_id in self.active_connections:
            connection = self.active_connections[user_id]
            try:
                await connection.send_text(message.to_json())
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")
                await self.disconnect(user_id)
    
    async def send_to_room(self, message: WebSocketMessage, room_id: str):
        """Send message to all users in a room"""
        if room_id in self.rooms:
            # Publish to Redis for distributed support
            if self.redis_client:
                await self.redis_client.publish(room_id, message.to_json())
            
            # Send to local connections
            for user_id in self.rooms[room_id]:
                await self.send_personal_message(message, user_id)
    
    async def broadcast(self, message: WebSocketMessage):
        """Broadcast message to all connected users"""
        disconnected = []
        
        for user_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message.to_json())
            except:
                disconnected.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected:
            await self.disconnect(user_id)
    
    async def send_to_tenant(self, message: WebSocketMessage, tenant_id: str):
        """Send message to all users in a tenant"""
        room_id = f"tenant_{tenant_id}"
        await self.send_to_room(message, room_id)


# Global connection manager instance
connection_manager = ConnectionManager()


# ============================================================================
# Event Publishers
# ============================================================================

class TrainingEventPublisher:
    """Publish training-related events"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def publish_training_started(self, 
                                      training_id: str,
                                      tenant_id: str,
                                      config: Dict):
        """Publish training started event"""
        message = WebSocketMessage(
            event_type=EventType.TRAINING_STARTED,
            data={
                "training_id": training_id,
                "config": config,
                "status": "started"
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)
    
    async def publish_training_progress(self,
                                       training_id: str,
                                       tenant_id: str,
                                       progress: float,
                                       current_model: str,
                                       metrics: Dict):
        """Publish training progress update"""
        message = WebSocketMessage(
            event_type=EventType.TRAINING_PROGRESS,
            data={
                "training_id": training_id,
                "progress": progress,
                "current_model": current_model,
                "current_metrics": metrics
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)
    
    async def publish_training_completed(self,
                                        training_id: str,
                                        tenant_id: str,
                                        best_model: str,
                                        final_metrics: Dict,
                                        leaderboard: List[Dict]):
        """Publish training completion event"""
        message = WebSocketMessage(
            event_type=EventType.TRAINING_COMPLETED,
            data={
                "training_id": training_id,
                "best_model": best_model,
                "final_metrics": final_metrics,
                "leaderboard": leaderboard[:5]  # Top 5 models
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)
    
    async def publish_training_failed(self,
                                     training_id: str,
                                     tenant_id: str,
                                     error: str):
        """Publish training failure event"""
        message = WebSocketMessage(
            event_type=EventType.TRAINING_FAILED,
            data={
                "training_id": training_id,
                "error": error
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)


class MonitoringEventPublisher:
    """Publish monitoring-related events"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def publish_metrics_update(self,
                                    model_id: str,
                                    tenant_id: str,
                                    metrics: Dict):
        """Publish model metrics update"""
        message = WebSocketMessage(
            event_type=EventType.METRICS_UPDATE,
            data={
                "model_id": model_id,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat()
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)
    
    async def publish_drift_detected(self,
                                    model_id: str,
                                    tenant_id: str,
                                    drift_info: Dict):
        """Publish drift detection event"""
        message = WebSocketMessage(
            event_type=EventType.DRIFT_DETECTED,
            data={
                "model_id": model_id,
                "drift_type": drift_info.get("type", "data_drift"),
                "drifted_features": drift_info.get("drifted_features", []),
                "drift_scores": drift_info.get("drift_scores", {}),
                "severity": "high"
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)
    
    async def publish_alert(self,
                          alert_type: str,
                          severity: str,
                          message_text: str,
                          tenant_id: str,
                          metadata: Dict = None):
        """Publish alert event"""
        message = WebSocketMessage(
            event_type=EventType.ALERT_TRIGGERED,
            data={
                "alert_type": alert_type,
                "severity": severity,
                "message": message_text,
                "metadata": metadata or {}
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)


class CacheEventPublisher:
    """Publish cache-related events"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def publish_cache_invalidated(self,
                                       pipeline_id: str,
                                       reason: str,
                                       tenant_id: str):
        """Publish cache invalidation event"""
        message = WebSocketMessage(
            event_type=EventType.CACHE_INVALIDATED,
            data={
                "pipeline_id": pipeline_id,
                "reason": reason,
                "action": "Model cache has been invalidated"
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)
    
    async def publish_cache_stats(self,
                                 stats: Dict,
                                 tenant_id: str):
        """Publish cache statistics"""
        message = WebSocketMessage(
            event_type=EventType.METRICS_UPDATE,
            data={
                "type": "cache_stats",
                "hit_rate": stats.get("hit_rate", 0),
                "total_hits": stats.get("hits", 0),
                "total_misses": stats.get("misses", 0),
                "cache_size_mb": stats.get("size_bytes", 0) / (1024 * 1024)
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)


class JobEventPublisher:
    """Publish job-related events"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def publish_job_status(self,
                                job: JobRequest,
                                tenant_id: str):
        """Publish job status update"""
        event_map = {
            JobStatus.QUEUED: EventType.JOB_QUEUED,
            JobStatus.RUNNING: EventType.JOB_STARTED,
            JobStatus.COMPLETED: EventType.JOB_COMPLETED,
            JobStatus.FAILED: EventType.JOB_FAILED
        }
        
        event_type = event_map.get(job.status, EventType.JOB_QUEUED)
        
        message = WebSocketMessage(
            event_type=event_type,
            data={
                "job_id": job.job_id,
                "status": job.status.value,
                "task_type": job.task_type,
                "queue_type": job.queue_type.queue_name,
                "progress": getattr(job, 'progress', 0),
                "estimated_time_remaining": getattr(job, 'estimated_time_remaining', None)
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)


class BillingEventPublisher:
    """Publish billing-related events"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
    
    async def publish_quota_warning(self,
                                   resource: str,
                                   usage_percent: float,
                                   tenant_id: str):
        """Publish quota warning event"""
        message = WebSocketMessage(
            event_type=EventType.QUOTA_WARNING,
            data={
                "resource": resource,
                "usage_percent": usage_percent,
                "message": f"Warning: {resource} usage at {usage_percent:.1f}%"
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)
    
    async def publish_quota_exceeded(self,
                                    resource: str,
                                    tenant_id: str):
        """Publish quota exceeded event"""
        message = WebSocketMessage(
            event_type=EventType.QUOTA_EXCEEDED,
            data={
                "resource": resource,
                "message": f"Quota exceeded for {resource}. Please upgrade your plan."
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)
    
    async def publish_billing_update(self,
                                    current_usage: Dict,
                                    estimated_cost: float,
                                    tenant_id: str):
        """Publish billing update"""
        message = WebSocketMessage(
            event_type=EventType.BILLING_UPDATE,
            data={
                "current_usage": current_usage,
                "estimated_cost": estimated_cost,
                "billing_period": datetime.utcnow().strftime("%Y-%m")
            },
            tenant_id=tenant_id
        )
        
        await self.connection_manager.send_to_tenant(message, tenant_id)


# ============================================================================
# WebSocket Endpoint Handler
# ============================================================================

class WebSocketHandler:
    """Main WebSocket handler"""
    
    def __init__(self):
        self.connection_manager = connection_manager
        self.training_publisher = TrainingEventPublisher(connection_manager)
        self.monitoring_publisher = MonitoringEventPublisher(connection_manager)
        self.cache_publisher = CacheEventPublisher(connection_manager)
        self.job_publisher = JobEventPublisher(connection_manager)
        self.billing_publisher = BillingEventPublisher(connection_manager)
        
        # Background tasks
        self.background_tasks = set()
    
    async def handle_connection(self, 
                               websocket: WebSocket,
                               user: User):
        """Handle WebSocket connection lifecycle"""
        
        user_id = str(user.id)
        
        try:
            # Accept connection
            await self.connection_manager.connect(websocket, user)
            
            # Start heartbeat
            heartbeat_task = asyncio.create_task(
                self._send_heartbeat(user_id)
            )
            self.background_tasks.add(heartbeat_task)
            
            # Listen for messages
            while True:
                data = await websocket.receive_text()
                await self._handle_message(data, user)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for user {user_id}")
        except Exception as e:
            logger.error(f"WebSocket error for user {user_id}: {e}")
        finally:
            # Cleanup
            await self.connection_manager.disconnect(user_id)
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
    
    async def _handle_message(self, message: str, user: User):
        """Handle incoming WebSocket message"""
        
        try:
            data = json.loads(message)
            command = data.get("command")
            
            if command == "subscribe":
                # Subscribe to specific events
                room_id = data.get("room_id")
                if room_id:
                    await self.connection_manager.join_room(str(user.id), room_id)
                    logger.debug(f"User {user.username} subscribed to {room_id}")
            
            elif command == "unsubscribe":
                # Unsubscribe from events
                room_id = data.get("room_id")
                if room_id:
                    await self.connection_manager.leave_room(str(user.id), room_id)
                    logger.debug(f"User {user.username} unsubscribed from {room_id}")
            
            elif command == "get_status":
                # Send current status
                await self._send_status(user)
            
            elif command == "ping":
                # Respond to ping
                await self.connection_manager.send_personal_message(
                    WebSocketMessage(
                        event_type=EventType.HEARTBEAT,
                        data={"message": "pong"}
                    ),
                    str(user.id)
                )
            
            else:
                # Unknown command
                await self.connection_manager.send_personal_message(
                    WebSocketMessage(
                        event_type=EventType.ERROR,
                        data={"error": f"Unknown command: {command}"}
                    ),
                    str(user.id)
                )
                
        except json.JSONDecodeError:
            await self.connection_manager.send_personal_message(
                WebSocketMessage(
                    event_type=EventType.ERROR,
                    data={"error": "Invalid JSON message"}
                ),
                str(user.id)
            )
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _send_heartbeat(self, user_id: str):
        """Send periodic heartbeat to keep connection alive"""
        
        while user_id in self.connection_manager.active_connections:
            try:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
                await self.connection_manager.send_personal_message(
                    WebSocketMessage(
                        event_type=EventType.HEARTBEAT,
                        data={"timestamp": datetime.utcnow().isoformat()}
                    ),
                    user_id
                )
                
            except Exception as e:
                logger.error(f"Heartbeat error for user {user_id}: {e}")
                break
    
    async def _send_status(self, user: User):
        """Send current system status to user"""
        
        status_data = {
            "user": {
                "id": str(user.id),
                "username": user.username,
                "plan_type": user.plan_type
            },
            "connection": {
                "connected_at": self.connection_manager.user_metadata.get(
                    str(user.id), {}
                ).get("connected_at"),
                "rooms": [
                    room for room, users in self.connection_manager.rooms.items()
                    if str(user.id) in users
                ]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.connection_manager.send_personal_message(
            WebSocketMessage(
                event_type=EventType.METRICS_UPDATE,
                data=status_data
            ),
            str(user.id)
        )


# ============================================================================
# Integration Hooks
# ============================================================================

class WebSocketIntegration:
    """Integration hooks for other services to publish events"""
    
    @staticmethod
    async def notify_training_progress(orchestrator: AutoMLOrchestrator,
                                      progress: float):
        """Hook for orchestrator to send training progress"""
        
        if not connection_manager.active_connections:
            return
        
        publisher = TrainingEventPublisher(connection_manager)
        
        tenant_id = orchestrator.config.tenant_id if hasattr(orchestrator.config, 'tenant_id') else "default"
        
        current_model = orchestrator.leaderboard[0]['model'] if orchestrator.leaderboard else "initializing"
        metrics = orchestrator.leaderboard[0]['metrics'] if orchestrator.leaderboard else {}
        
        await publisher.publish_training_progress(
            orchestrator.training_id,
            tenant_id,
            progress,
            current_model,
            metrics
        )
    
    @staticmethod
    async def notify_drift_detected(monitor: ModelMonitor,
                                   drift_info: Dict):
        """Hook for monitor to send drift alerts"""
        
        if not connection_manager.active_connections:
            return
        
        publisher = MonitoringEventPublisher(connection_manager)
        
        await publisher.publish_drift_detected(
            monitor.model_id,
            monitor.tenant_id,
            drift_info
        )
    
    @staticmethod
    async def notify_cache_invalidation(cache: PipelineCache,
                                       pipeline_id: str,
                                       reason: str):
        """Hook for cache to send invalidation events"""
        
        if not connection_manager.active_connections:
            return
        
        publisher = CacheEventPublisher(connection_manager)
        
        # Extract tenant_id from pipeline_id if formatted as "tenant_id_pipeline_name"
        parts = pipeline_id.split("_")
        tenant_id = parts[0] if len(parts) > 1 else "default"
        
        await publisher.publish_cache_invalidated(
            pipeline_id,
            reason,
            tenant_id
        )
    
    @staticmethod
    async def notify_job_update(job: JobRequest):
        """Hook for scheduler to send job updates"""
        
        if not connection_manager.active_connections:
            return
        
        publisher = JobEventPublisher(connection_manager)
        
        await publisher.publish_job_status(
            job,
            job.tenant_id
        )
    
    @staticmethod
    async def notify_quota_status(tenant_id: str,
                                 resource: str,
                                 usage_percent: float):
        """Hook for billing to send quota warnings"""
        
        if not connection_manager.active_connections:
            return
        
        publisher = BillingEventPublisher(connection_manager)
        
        if usage_percent >= 90:
            await publisher.publish_quota_exceeded(resource, tenant_id)
        elif usage_percent >= 80:
            await publisher.publish_quota_warning(resource, usage_percent, tenant_id)


# ============================================================================
# FastAPI WebSocket Route
# ============================================================================

async def websocket_endpoint(websocket: WebSocket, token: str):
    """
    Main WebSocket endpoint
    
    Usage:
        ws://localhost:8000/ws?token=YOUR_JWT_TOKEN
    """
    
    # Authenticate user
    try:
        token_service = TokenService()
        payload = token_service.verify_token(token)
        
        # Get user from database
        from ..auth import get_db
        db = next(get_db())
        user = db.query(User).filter_by(id=payload["sub"]).first()
        
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
    except Exception as e:
        logger.error(f"WebSocket authentication failed: {e}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    # Handle connection
    handler = WebSocketHandler()
    await handler.handle_connection(websocket, user)


# ============================================================================
# Startup and Shutdown
# ============================================================================

async def initialize_websocket_service(redis_url: Optional[str] = None):
    """Initialize WebSocket service on startup"""
    
    if redis_url:
        await connection_manager.initialize_redis(redis_url)
    
    logger.info("WebSocket service initialized")


async def shutdown_websocket_service():
    """Cleanup WebSocket service on shutdown"""
    
    # Close all connections
    for user_id in list(connection_manager.active_connections.keys()):
        await connection_manager.disconnect(user_id)
    
    # Close Redis connection
    if connection_manager.redis_client:
        await connection_manager.redis_client.close()
    
    logger.info("WebSocket service shutdown")
