"""
WebSocket Service for Real-time MLOps Platform
==============================================

This module provides real-time communication capabilities including:
- Real-time chat interface with LLM
- Push notifications for training, drift detection, and errors
- Live monitoring during model training
- Multi-user collaboration on projects

Author: MLOps Team
Date: 2025
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import websockets
from websockets.server import WebSocketServerProtocol
import aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
import numpy as np
from collections import defaultdict
import jwt
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class MessageType(str, Enum):
    """Types of WebSocket messages"""
    CHAT = "chat"
    NOTIFICATION = "notification"
    METRICS = "metrics"
    COLLABORATION = "collaboration"
    HEARTBEAT = "heartbeat"
    AUTH = "auth"
    ERROR = "error"
    TRAINING_UPDATE = "training_update"
    DRIFT_ALERT = "drift_alert"
    MODEL_PERFORMANCE = "model_performance"
    SYSTEM_STATUS = "system_status"


class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class User:
    """User model for WebSocket connections"""
    id: str
    username: str
    email: str
    roles: List[str]
    projects: List[str]
    connection_id: Optional[str] = None
    connected_at: Optional[datetime] = None


@dataclass
class Message:
    """WebSocket message structure"""
    id: str
    type: MessageType
    sender_id: str
    recipient_id: Optional[str]
    project_id: Optional[str]
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    """Chat message model"""
    content: str
    sender_id: str
    project_id: str
    parent_message_id: Optional[str] = None
    attachments: Optional[List[Dict]] = None


class Notification(BaseModel):
    """Notification model"""
    title: str
    message: str
    priority: NotificationPriority
    source: str
    project_id: Optional[str] = None
    user_ids: Optional[List[str]] = None
    data: Optional[Dict[str, Any]] = None
    action_required: bool = False
    action_url: Optional[str] = None


class TrainingMetrics(BaseModel):
    """Training metrics for live monitoring"""
    epoch: int
    batch: int
    loss: float
    accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float
    time_elapsed: float
    eta: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None


# ============================================================================
# Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections and routing"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocketServerProtocol] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        self.project_rooms: Dict[str, Set[str]] = defaultdict(set)
        self.connection_metadata: Dict[str, User] = {}
        
    async def connect(self, websocket: WebSocketServerProtocol, user: User):
        """Register new WebSocket connection"""
        connection_id = str(uuid.uuid4())
        user.connection_id = connection_id
        user.connected_at = datetime.utcnow()
        
        self.active_connections[connection_id] = websocket
        self.user_connections[user.id].add(connection_id)
        self.connection_metadata[connection_id] = user
        
        # Add user to project rooms
        for project_id in user.projects:
            self.project_rooms[project_id].add(connection_id)
            
        logger.info(f"User {user.username} connected (ID: {connection_id})")
        
        # Send connection confirmation
        await self.send_personal_message(
            connection_id,
            Message(
                id=str(uuid.uuid4()),
                type=MessageType.AUTH,
                sender_id="system",
                recipient_id=user.id,
                project_id=None,
                timestamp=datetime.utcnow(),
                data={"status": "connected", "user_id": user.id}
            )
        )
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Remove WebSocket connection"""
        if connection_id in self.connection_metadata:
            user = self.connection_metadata[connection_id]
            
            # Remove from user connections
            self.user_connections[user.id].discard(connection_id)
            if not self.user_connections[user.id]:
                del self.user_connections[user.id]
            
            # Remove from project rooms
            for project_id in user.projects:
                self.project_rooms[project_id].discard(connection_id)
                if not self.project_rooms[project_id]:
                    del self.project_rooms[project_id]
            
            # Clean up connection data
            del self.connection_metadata[connection_id]
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            
            logger.info(f"User {user.username} disconnected (ID: {connection_id})")
    
    async def send_personal_message(self, connection_id: str, message: Message):
        """Send message to specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send(json.dumps(asdict(message), default=str))
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                await self.disconnect(connection_id)
    
    async def send_to_user(self, user_id: str, message: Message):
        """Send message to all connections of a user"""
        if user_id in self.user_connections:
            for connection_id in self.user_connections[user_id]:
                await self.send_personal_message(connection_id, message)
    
    async def broadcast_to_project(self, project_id: str, message: Message):
        """Broadcast message to all users in a project"""
        if project_id in self.project_rooms:
            tasks = []
            for connection_id in self.project_rooms[project_id]:
                tasks.append(self.send_personal_message(connection_id, message))
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def broadcast_to_all(self, message: Message):
        """Broadcast message to all connected users"""
        tasks = []
        for connection_id in self.active_connections:
            tasks.append(self.send_personal_message(connection_id, message))
        await asyncio.gather(*tasks, return_exceptions=True)


# ============================================================================
# Chat Service with LLM Integration
# ============================================================================

class ChatService:
    """Handles real-time chat with LLM integration"""
    
    def __init__(self, llm_endpoint: str = "http://localhost:8000/llm/chat"):
        self.llm_endpoint = llm_endpoint
        self.conversation_history: Dict[str, List[ChatMessage]] = defaultdict(list)
        self.active_sessions: Dict[str, Dict] = {}
        
    async def process_chat_message(
        self,
        message: ChatMessage,
        connection_manager: ConnectionManager
    ) -> Dict[str, Any]:
        """Process incoming chat message and get LLM response"""
        
        # Store message in conversation history
        self.conversation_history[message.project_id].append(message)
        
        # Broadcast user message to project room
        await connection_manager.broadcast_to_project(
            message.project_id,
            Message(
                id=str(uuid.uuid4()),
                type=MessageType.CHAT,
                sender_id=message.sender_id,
                recipient_id=None,
                project_id=message.project_id,
                timestamp=datetime.utcnow(),
                data=message.dict()
            )
        )
        
        # Get LLM response
        llm_response = await self._get_llm_response(message)
        
        # Create LLM message
        llm_message = ChatMessage(
            content=llm_response["content"],
            sender_id="llm_assistant",
            project_id=message.project_id,
            parent_message_id=message.parent_message_id
        )
        
        # Store LLM response
        self.conversation_history[message.project_id].append(llm_message)
        
        # Broadcast LLM response
        await connection_manager.broadcast_to_project(
            message.project_id,
            Message(
                id=str(uuid.uuid4()),
                type=MessageType.CHAT,
                sender_id="llm_assistant",
                recipient_id=None,
                project_id=message.project_id,
                timestamp=datetime.utcnow(),
                data=llm_message.dict(),
                metadata=llm_response.get("metadata", {})
            )
        )
        
        return llm_response
    
    async def _get_llm_response(self, message: ChatMessage) -> Dict[str, Any]:
        """Get response from LLM service"""
        try:
            async with httpx.AsyncClient() as client:
                # Prepare context from conversation history
                context = self._prepare_context(message.project_id)
                
                response = await client.post(
                    self.llm_endpoint,
                    json={
                        "message": message.content,
                        "context": context,
                        "project_id": message.project_id,
                        "attachments": message.attachments
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "content": "I apologize, but I'm unable to process your request at the moment.",
                        "error": f"LLM service returned status {response.status_code}"
                    }
                    
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            return {
                "content": "I encountered an error while processing your request.",
                "error": str(e)
            }
    
    def _prepare_context(self, project_id: str, max_messages: int = 10) -> List[Dict]:
        """Prepare conversation context for LLM"""
        history = self.conversation_history.get(project_id, [])
        recent_messages = history[-max_messages:] if len(history) > max_messages else history
        
        return [
            {
                "role": "assistant" if msg.sender_id == "llm_assistant" else "user",
                "content": msg.content,
                "timestamp": msg.dict().get("timestamp")
            }
            for msg in recent_messages
        ]


# ============================================================================
# Notification Service
# ============================================================================

class NotificationService:
    """Handles push notifications for various events"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.notification_history: List[Notification] = []
        
    async def send_notification(self, notification: Notification):
        """Send notification to specified users or project"""
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.NOTIFICATION,
            sender_id="system",
            recipient_id=None,
            project_id=notification.project_id,
            timestamp=datetime.utcnow(),
            data=notification.dict(),
            metadata={"priority": notification.priority.value}
        )
        
        if notification.user_ids:
            # Send to specific users
            for user_id in notification.user_ids:
                await self.connection_manager.send_to_user(user_id, message)
        elif notification.project_id:
            # Send to project room
            await self.connection_manager.broadcast_to_project(
                notification.project_id, message
            )
        else:
            # Broadcast to all
            await self.connection_manager.broadcast_to_all(message)
        
        # Store in history
        self.notification_history.append(notification)
        
        # Log high priority notifications
        if notification.priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL]:
            logger.warning(f"High priority notification: {notification.title}")
    
    async def notify_training_event(
        self,
        project_id: str,
        event_type: str,
        details: Dict[str, Any]
    ):
        """Send training-related notifications"""
        priority_map = {
            "started": NotificationPriority.MEDIUM,
            "completed": NotificationPriority.HIGH,
            "failed": NotificationPriority.CRITICAL,
            "checkpoint": NotificationPriority.LOW
        }
        
        notification = Notification(
            title=f"Training {event_type.capitalize()}",
            message=f"Model training has {event_type}",
            priority=priority_map.get(event_type, NotificationPriority.MEDIUM),
            source="training_service",
            project_id=project_id,
            data=details,
            action_required=event_type == "failed",
            action_url=f"/projects/{project_id}/training"
        )
        
        await self.send_notification(notification)
    
    async def notify_drift_detection(
        self,
        project_id: str,
        drift_type: str,
        severity: str,
        details: Dict[str, Any]
    ):
        """Send drift detection alerts"""
        priority_map = {
            "low": NotificationPriority.LOW,
            "medium": NotificationPriority.MEDIUM,
            "high": NotificationPriority.HIGH,
            "critical": NotificationPriority.CRITICAL
        }
        
        notification = Notification(
            title=f"Data Drift Detected ({drift_type})",
            message=f"Severity: {severity}. Immediate action may be required.",
            priority=priority_map.get(severity, NotificationPriority.HIGH),
            source="drift_detection",
            project_id=project_id,
            data=details,
            action_required=severity in ["high", "critical"],
            action_url=f"/projects/{project_id}/monitoring/drift"
        )
        
        await self.send_notification(notification)
    
    async def notify_error(
        self,
        error_type: str,
        message: str,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        """Send error notifications"""
        notification = Notification(
            title=f"Error: {error_type}",
            message=message,
            priority=NotificationPriority.HIGH,
            source="error_handler",
            project_id=project_id,
            user_ids=[user_id] if user_id else None,
            data=details or {},
            action_required=True
        )
        
        await self.send_notification(notification)


# ============================================================================
# Live Monitoring Service
# ============================================================================

class LiveMonitoringService:
    """Provides real-time monitoring during model training"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_trainings: Dict[str, Dict] = {}
        self.metrics_buffer: Dict[str, List[TrainingMetrics]] = defaultdict(list)
        
    async def start_monitoring(self, project_id: str, training_id: str, config: Dict):
        """Initialize monitoring for a training session"""
        self.active_trainings[training_id] = {
            "project_id": project_id,
            "started_at": datetime.utcnow(),
            "config": config,
            "status": "running"
        }
        
        logger.info(f"Started monitoring for training {training_id}")
        
        # Notify users
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.TRAINING_UPDATE,
            sender_id="system",
            recipient_id=None,
            project_id=project_id,
            timestamp=datetime.utcnow(),
            data={
                "training_id": training_id,
                "status": "started",
                "config": config
            }
        )
        
        await self.connection_manager.broadcast_to_project(project_id, message)
    
    async def update_metrics(self, training_id: str, metrics: TrainingMetrics):
        """Update and broadcast training metrics"""
        if training_id not in self.active_trainings:
            logger.warning(f"Training {training_id} not found in active sessions")
            return
        
        project_id = self.active_trainings[training_id]["project_id"]
        
        # Store metrics
        self.metrics_buffer[training_id].append(metrics)
        
        # Prepare aggregated metrics
        aggregated = self._aggregate_metrics(training_id)
        
        # Broadcast metrics update
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.METRICS,
            sender_id="system",
            recipient_id=None,
            project_id=project_id,
            timestamp=datetime.utcnow(),
            data={
                "training_id": training_id,
                "current": metrics.dict(),
                "aggregated": aggregated
            }
        )
        
        await self.connection_manager.broadcast_to_project(project_id, message)
        
        # Check for anomalies
        await self._check_training_anomalies(training_id, metrics)
    
    async def stop_monitoring(self, training_id: str, status: str = "completed"):
        """Stop monitoring for a training session"""
        if training_id not in self.active_trainings:
            return
        
        project_id = self.active_trainings[training_id]["project_id"]
        training_info = self.active_trainings[training_id]
        
        # Calculate final statistics
        final_stats = self._calculate_final_stats(training_id)
        
        # Send completion message
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.TRAINING_UPDATE,
            sender_id="system",
            recipient_id=None,
            project_id=project_id,
            timestamp=datetime.utcnow(),
            data={
                "training_id": training_id,
                "status": status,
                "duration": (datetime.utcnow() - training_info["started_at"]).total_seconds(),
                "final_stats": final_stats
            }
        )
        
        await self.connection_manager.broadcast_to_project(project_id, message)
        
        # Clean up
        del self.active_trainings[training_id]
        if training_id in self.metrics_buffer:
            del self.metrics_buffer[training_id]
        
        logger.info(f"Stopped monitoring for training {training_id} (status: {status})")
    
    def _aggregate_metrics(self, training_id: str) -> Dict:
        """Aggregate metrics for visualization"""
        metrics_list = self.metrics_buffer[training_id]
        if not metrics_list:
            return {}
        
        latest_metrics = metrics_list[-1]
        
        # Calculate moving averages
        window_size = min(10, len(metrics_list))
        recent_metrics = metrics_list[-window_size:]
        
        return {
            "total_batches": latest_metrics.batch,
            "total_epochs": latest_metrics.epoch,
            "avg_loss": np.mean([m.loss for m in recent_metrics]),
            "avg_accuracy": np.mean([m.accuracy for m in recent_metrics if m.accuracy]),
            "loss_trend": self._calculate_trend([m.loss for m in recent_metrics]),
            "learning_rate": latest_metrics.learning_rate,
            "time_elapsed": latest_metrics.time_elapsed,
            "eta": latest_metrics.eta
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
        
        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        previous_avg = np.mean(values[:-3]) if len(values) > 3 else values[0]
        
        change = (recent_avg - previous_avg) / previous_avg if previous_avg != 0 else 0
        
        if change < -0.05:
            return "decreasing"
        elif change > 0.05:
            return "increasing"
        else:
            return "stable"
    
    def _calculate_final_stats(self, training_id: str) -> Dict:
        """Calculate final training statistics"""
        metrics_list = self.metrics_buffer[training_id]
        if not metrics_list:
            return {}
        
        return {
            "total_epochs": metrics_list[-1].epoch,
            "total_batches": metrics_list[-1].batch,
            "final_loss": metrics_list[-1].loss,
            "final_accuracy": metrics_list[-1].accuracy,
            "best_loss": min(m.loss for m in metrics_list),
            "best_accuracy": max(m.accuracy for m in metrics_list if m.accuracy),
            "total_time": metrics_list[-1].time_elapsed,
            "avg_batch_time": metrics_list[-1].time_elapsed / metrics_list[-1].batch
        }
    
    async def _check_training_anomalies(self, training_id: str, metrics: TrainingMetrics):
        """Check for training anomalies and send alerts"""
        metrics_list = self.metrics_buffer[training_id]
        
        # Check for loss explosion
        if len(metrics_list) > 5:
            recent_losses = [m.loss for m in metrics_list[-5:]]
            if metrics.loss > 10 * np.mean(recent_losses):
                await self._send_anomaly_alert(
                    training_id,
                    "Loss Explosion",
                    f"Loss increased dramatically to {metrics.loss}"
                )
        
        # Check for stagnation
        if len(metrics_list) > 20:
            recent_losses = [m.loss for m in metrics_list[-20:]]
            if np.std(recent_losses) < 0.001:
                await self._send_anomaly_alert(
                    training_id,
                    "Training Stagnation",
                    "Loss has not changed significantly in recent batches"
                )
        
        # Check for NaN values
        if np.isnan(metrics.loss) or (metrics.accuracy and np.isnan(metrics.accuracy)):
            await self._send_anomaly_alert(
                training_id,
                "NaN Values Detected",
                "Training metrics contain NaN values"
            )
    
    async def _send_anomaly_alert(self, training_id: str, anomaly_type: str, message: str):
        """Send anomaly alert notification"""
        project_id = self.active_trainings[training_id]["project_id"]
        
        notification = Notification(
            title=f"Training Anomaly: {anomaly_type}",
            message=message,
            priority=NotificationPriority.HIGH,
            source="training_monitor",
            project_id=project_id,
            data={"training_id": training_id},
            action_required=True,
            action_url=f"/projects/{project_id}/training/{training_id}"
        )
        
        notification_service = NotificationService(self.connection_manager)
        await notification_service.send_notification(notification)


# ============================================================================
# Collaboration Service
# ============================================================================

class CollaborationService:
    """Handles multi-user collaboration features"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_collaborations: Dict[str, Dict] = {}
        self.document_locks: Dict[str, str] = {}  # document_id -> user_id
        self.cursor_positions: Dict[str, Dict] = {}  # user_id -> position
        
    async def join_collaboration(
        self,
        user_id: str,
        project_id: str,
        document_id: str
    ):
        """User joins a collaborative session"""
        session_key = f"{project_id}:{document_id}"
        
        if session_key not in self.active_collaborations:
            self.active_collaborations[session_key] = {
                "project_id": project_id,
                "document_id": document_id,
                "users": set(),
                "started_at": datetime.utcnow()
            }
        
        self.active_collaborations[session_key]["users"].add(user_id)
        
        # Notify other users
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.COLLABORATION,
            sender_id=user_id,
            recipient_id=None,
            project_id=project_id,
            timestamp=datetime.utcnow(),
            data={
                "action": "user_joined",
                "document_id": document_id,
                "user_id": user_id,
                "active_users": list(self.active_collaborations[session_key]["users"])
            }
        )
        
        await self.connection_manager.broadcast_to_project(project_id, message)
    
    async def leave_collaboration(
        self,
        user_id: str,
        project_id: str,
        document_id: str
    ):
        """User leaves a collaborative session"""
        session_key = f"{project_id}:{document_id}"
        
        if session_key in self.active_collaborations:
            self.active_collaborations[session_key]["users"].discard(user_id)
            
            # Release any locks held by the user
            if document_id in self.document_locks and self.document_locks[document_id] == user_id:
                await self.release_lock(user_id, project_id, document_id)
            
            # Remove cursor position
            if user_id in self.cursor_positions:
                del self.cursor_positions[user_id]
            
            # Notify other users
            message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.COLLABORATION,
                sender_id=user_id,
                recipient_id=None,
                project_id=project_id,
                timestamp=datetime.utcnow(),
                data={
                    "action": "user_left",
                    "document_id": document_id,
                    "user_id": user_id,
                    "active_users": list(self.active_collaborations[session_key]["users"])
                }
            )
            
            await self.connection_manager.broadcast_to_project(project_id, message)
            
            # Clean up empty sessions
            if not self.active_collaborations[session_key]["users"]:
                del self.active_collaborations[session_key]
    
    async def acquire_lock(
        self,
        user_id: str,
        project_id: str,
        document_id: str,
        section: Optional[str] = None
    ) -> bool:
        """Acquire edit lock on document or section"""
        lock_key = f"{document_id}:{section}" if section else document_id
        
        if lock_key not in self.document_locks:
            self.document_locks[lock_key] = user_id
            
            # Notify users about lock acquisition
            message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.COLLABORATION,
                sender_id=user_id,
                recipient_id=None,
                project_id=project_id,
                timestamp=datetime.utcnow(),
                data={
                    "action": "lock_acquired",
                    "document_id": document_id,
                    "section": section,
                    "user_id": user_id
                }
            )
            
            await self.connection_manager.broadcast_to_project(project_id, message)
            return True
        
        return False
    
    async def release_lock(
        self,
        user_id: str,
        project_id: str,
        document_id: str,
        section: Optional[str] = None
    ):
        """Release edit lock"""
        lock_key = f"{document_id}:{section}" if section else document_id
        
        if lock_key in self.document_locks and self.document_locks[lock_key] == user_id:
            del self.document_locks[lock_key]
            
            # Notify users about lock release
            message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.COLLABORATION,
                sender_id=user_id,
                recipient_id=None,
                project_id=project_id,
                timestamp=datetime.utcnow(),
                data={
                    "action": "lock_released",
                    "document_id": document_id,
                    "section": section,
                    "user_id": user_id
                }
            )
            
            await self.connection_manager.broadcast_to_project(project_id, message)
    
    async def broadcast_cursor_position(
        self,
        user_id: str,
        project_id: str,
        document_id: str,
        position: Dict[str, Any]
    ):
        """Broadcast user's cursor position"""
        self.cursor_positions[user_id] = position
        
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.COLLABORATION,
            sender_id=user_id,
            recipient_id=None,
            project_id=project_id,
            timestamp=datetime.utcnow(),
            data={
                "action": "cursor_move",
                "document_id": document_id,
                "user_id": user_id,
                "position": position
            }
        )
        
        await self.connection_manager.broadcast_to_project(project_id, message)
    
    async def broadcast_document_change(
        self,
        user_id: str,
        project_id: str,
        document_id: str,
        change: Dict[str, Any]
    ):
        """Broadcast document changes for real-time sync"""
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.COLLABORATION,
            sender_id=user_id,
            recipient_id=None,
            project_id=project_id,
            timestamp=datetime.utcnow(),
            data={
                "action": "document_change",
                "document_id": document_id,
                "user_id": user_id,
                "change": change
            }
        )
        
        await self.connection_manager.broadcast_to_project(project_id, message)


# ============================================================================
# WebSocket Server
# ============================================================================

class WebSocketServer:
    """Main WebSocket server handling all real-time features"""
    
    def __init__(self, redis_url: str = "redis://localhost"):
        self.connection_manager = ConnectionManager()
        self.chat_service = ChatService()
        self.notification_service = NotificationService(self.connection_manager)
        self.monitoring_service = LiveMonitoringService(self.connection_manager)
        self.collaboration_service = CollaborationService(self.connection_manager)
        self.redis_url = redis_url
        self.redis_client: Optional[aioredis.Redis] = None
        
    async def initialize(self):
        """Initialize server components"""
        try:
            self.redis_client = await aioredis.create_redis_pool(self.redis_url)
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    async def cleanup(self):
        """Cleanup server resources"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
    
    async def authenticate_user(self, websocket: WebSocketServerProtocol) -> Optional[User]:
        """Authenticate WebSocket connection"""
        try:
            # Wait for authentication message
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            auth_data = json.loads(auth_message)
            
            if auth_data.get("type") != "auth":
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Authentication required"
                }))
                return None
            
            # Verify JWT token
            token = auth_data.get("token")
            if not token:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Token required"
                }))
                return None
            
            # Decode and verify token (simplified - use proper JWT verification in production)
            try:
                payload = jwt.decode(token, "secret_key", algorithms=["HS256"])
                user = User(
                    id=payload["user_id"],
                    username=payload["username"],
                    email=payload.get("email", ""),
                    roles=payload.get("roles", ["user"]),
                    projects=payload.get("projects", [])
                )
                return user
                
            except jwt.InvalidTokenError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid token"
                }))
                return None
                
        except asyncio.TimeoutError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Authentication timeout"
            }))
            return None
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connection"""
        connection_id = None
        
        try:
            # Authenticate user
            user = await self.authenticate_user(websocket)
            if not user:
                await websocket.close()
                return
            
            # Register connection
            connection_id = await self.connection_manager.connect(websocket, user)
            
            # Handle messages
            async for message in websocket:
                await self.handle_message(connection_id, message)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if connection_id:
                await self.connection_manager.disconnect(connection_id)
    
    async def handle_message(self, connection_id: str, raw_message: str):
        """Route incoming messages to appropriate handlers"""
        try:
            data = json.loads(raw_message)
            message_type = MessageType(data.get("type"))
            
            user = self.connection_manager.connection_metadata.get(connection_id)
            if not user:
                logger.error(f"User not found for connection {connection_id}")
                return
            
            # Route message based on type
            if message_type == MessageType.CHAT:
                await self.handle_chat_message(connection_id, data)
                
            elif message_type == MessageType.COLLABORATION:
                await self.handle_collaboration_message(connection_id, data)
                
            elif message_type == MessageType.HEARTBEAT:
                await self.handle_heartbeat(connection_id)
                
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {raw_message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def handle_chat_message(self, connection_id: str, data: Dict):
        """Handle chat messages"""
        user = self.connection_manager.connection_metadata[connection_id]
        
        chat_message = ChatMessage(
            content=data["content"],
            sender_id=user.id,
            project_id=data["project_id"],
            parent_message_id=data.get("parent_message_id"),
            attachments=data.get("attachments")
        )
        
        await self.chat_service.process_chat_message(
            chat_message,
            self.connection_manager
        )
    
    async def handle_collaboration_message(self, connection_id: str, data: Dict):
        """Handle collaboration messages"""
        user = self.connection_manager.connection_metadata[connection_id]
        action = data.get("action")
        
        if action == "join":
            await self.collaboration_service.join_collaboration(
                user.id,
                data["project_id"],
                data["document_id"]
            )
        elif action == "leave":
            await self.collaboration_service.leave_collaboration(
                user.id,
                data["project_id"],
                data["document_id"]
            )
        elif action == "acquire_lock":
            result = await self.collaboration_service.acquire_lock(
                user.id,
                data["project_id"],
                data["document_id"],
                data.get("section")
            )
            # Send result back to user
            await self.connection_manager.send_personal_message(
                connection_id,
                Message(
                    id=str(uuid.uuid4()),
                    type=MessageType.COLLABORATION,
                    sender_id="system",
                    recipient_id=user.id,
                    project_id=data["project_id"],
                    timestamp=datetime.utcnow(),
                    data={"action": "lock_result", "success": result}
                )
            )
        elif action == "release_lock":
            await self.collaboration_service.release_lock(
                user.id,
                data["project_id"],
                data["document_id"],
                data.get("section")
            )
        elif action == "cursor_position":
            await self.collaboration_service.broadcast_cursor_position(
                user.id,
                data["project_id"],
                data["document_id"],
                data["position"]
            )
        elif action == "document_change":
            await self.collaboration_service.broadcast_document_change(
                user.id,
                data["project_id"],
                data["document_id"],
                data["change"]
            )
    
    async def handle_heartbeat(self, connection_id: str):
        """Handle heartbeat messages"""
        await self.connection_manager.send_personal_message(
            connection_id,
            Message(
                id=str(uuid.uuid4()),
                type=MessageType.HEARTBEAT,
                sender_id="system",
                recipient_id=None,
                project_id=None,
                timestamp=datetime.utcnow(),
                data={"status": "alive"}
            )
        )
    
    async def start(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server"""
        await self.initialize()
        
        logger.info(f"Starting WebSocket server on {host}:{port}")
        
        async with websockets.serve(self.handle_connection, host, port):
            await asyncio.Future()  # Run forever


# ============================================================================
# FastAPI Integration
# ============================================================================

app = FastAPI(title="MLOps WebSocket Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global server instance
ws_server = WebSocketServer()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """FastAPI WebSocket endpoint"""
    await websocket.accept()
    await ws_server.handle_connection(websocket, "/ws")


@app.post("/api/notifications/send")
async def send_notification(notification: Notification):
    """REST endpoint to send notifications"""
    await ws_server.notification_service.send_notification(notification)
    return {"status": "sent", "notification_id": str(uuid.uuid4())}


@app.post("/api/training/{training_id}/metrics")
async def update_training_metrics(training_id: str, metrics: TrainingMetrics):
    """REST endpoint to update training metrics"""
    await ws_server.monitoring_service.update_metrics(training_id, metrics)
    return {"status": "updated"}


@app.post("/api/training/{training_id}/start")
async def start_training_monitoring(
    training_id: str,
    project_id: str = None,
    config: Dict = None
):
    """REST endpoint to start training monitoring"""
    await ws_server.monitoring_service.start_monitoring(
        project_id or "default",
        training_id,
        config or {}
    )
    return {"status": "monitoring_started"}


@app.post("/api/training/{training_id}/stop")
async def stop_training_monitoring(training_id: str, status: str = "completed"):
    """REST endpoint to stop training monitoring"""
    await ws_server.monitoring_service.stop_monitoring(training_id, status)
    return {"status": "monitoring_stopped"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_connections": len(ws_server.connection_manager.active_connections),
        "active_trainings": len(ws_server.monitoring_service.active_trainings)
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "fastapi":
        # Run FastAPI server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    else:
        # Run standalone WebSocket server
        server = WebSocketServer()
        
        try:
            asyncio.run(server.start())
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        finally:
            asyncio.run(server.cleanup())
