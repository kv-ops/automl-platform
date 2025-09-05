"""
WebSocket Service for Real-time MLOps Platform
==============================================

This module provides real-time communication capabilities including:
- Real-time chat interface with LLM
- Push notifications for training, drift detection, and errors
- Live monitoring during model training
- Multi-user collaboration on projects
- Presence system and typing indicators
- Message persistence and rate limiting

Author: MLOps Team
Date: 2025
Version: 3.0.1
"""

import asyncio
import json
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import hashlib
import traceback

import websockets
from websockets.server import WebSocketServerProtocol
import redis.asyncio as redis  # Updated from aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from pydantic import BaseModel, Field, validator
import numpy as np
from collections import defaultdict, deque
import jwt
import httpx

# Configuration
from ..config import load_config

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class WebSocketConfig:
   """WebSocket service configuration"""
   
   def __init__(self):
       self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
       self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
       self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
       self.llm_endpoint = os.getenv("LLM_ENDPOINT", "http://localhost:8000/llm/chat")
       self.max_connections_per_user = int(os.getenv("MAX_CONNECTIONS_PER_USER", "5"))
       self.rate_limit_messages = int(os.getenv("RATE_LIMIT_MESSAGES", "100"))
       self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
       self.message_persistence_ttl = int(os.getenv("MESSAGE_PERSISTENCE_TTL", "86400"))  # 24 hours
       self.heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL", "30"))  # seconds
       self.reconnect_timeout = int(os.getenv("RECONNECT_TIMEOUT", "300"))  # 5 minutes
       self.max_message_size = int(os.getenv("MAX_MESSAGE_SIZE", "1048576"))  # 1MB
       self.enable_message_persistence = os.getenv("ENABLE_MESSAGE_PERSISTENCE", "true").lower() == "true"
       self.enable_presence = os.getenv("ENABLE_PRESENCE", "true").lower() == "true"
       self.enable_typing_indicators = os.getenv("ENABLE_TYPING_INDICATORS", "true").lower() == "true"

config = WebSocketConfig()

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
   PRESENCE = "presence"
   TYPING = "typing"
   ACK = "acknowledgment"


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
   last_activity: Optional[datetime] = None
   is_online: bool = False


@dataclass
class Message:
   """WebSocket message structure"""
   id: str = field(default_factory=lambda: str(uuid.uuid4()))
   type: MessageType = MessageType.CHAT
   sender_id: str = ""
   recipient_id: Optional[str] = None
   project_id: Optional[str] = None
   timestamp: datetime = field(default_factory=datetime.utcnow)
   data: Dict[str, Any] = field(default_factory=dict)
   metadata: Optional[Dict[str, Any]] = None
   ack_required: bool = False


class ChatMessage(BaseModel):
   """Chat message model"""
   content: str
   sender_id: str
   project_id: str
   parent_message_id: Optional[str] = None
   attachments: Optional[List[Dict]] = None
   
   @validator('content')
   def validate_content_length(cls, v):
       if len(v) > config.max_message_size:
           raise ValueError(f"Message too large. Max size: {config.max_message_size} bytes")
       return v


class Notification(BaseModel):
   """Notification model"""
   title: str
   message: str
   priority: NotificationPriority = NotificationPriority.MEDIUM
   source: str = "system"
   project_id: Optional[str] = None
   user_ids: Optional[List[str]] = None
   data: Optional[Dict[str, Any]] = None
   action_required: bool = False
   action_url: Optional[str] = None
   expires_at: Optional[datetime] = None


class TrainingMetrics(BaseModel):
   """Training metrics for live monitoring"""
   training_id: str
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
# Rate Limiter
# ============================================================================

class RateLimiter:
   """Rate limiting for WebSocket connections"""
   
   def __init__(self, max_messages: int = 100, window_seconds: int = 60):
       self.max_messages = max_messages
       self.window_seconds = window_seconds
       self.user_messages: Dict[str, deque] = defaultdict(deque)
   
   def check_rate_limit(self, user_id: str) -> Tuple[bool, Optional[int]]:
       """
       Check if user has exceeded rate limit
       Returns: (is_allowed, seconds_until_reset)
       """
       now = time.time()
       user_queue = self.user_messages[user_id]
       
       # Remove old messages outside window
       while user_queue and user_queue[0] < now - self.window_seconds:
           user_queue.popleft()
       
       if len(user_queue) >= self.max_messages:
           # Calculate when the oldest message will expire
           oldest = user_queue[0]
           reset_in = int(oldest + self.window_seconds - now)
           return False, reset_in
       
       # Add current message timestamp
       user_queue.append(now)
       return True, None
   
   def reset_user(self, user_id: str):
       """Reset rate limit for a user"""
       if user_id in self.user_messages:
           del self.user_messages[user_id]


# ============================================================================
# Connection Manager
# ============================================================================

class ConnectionManager:
   """Manages WebSocket connections and routing"""
   
   def __init__(self, redis_client: Optional[redis.Redis] = None):
       self.active_connections: Dict[str, WebSocketServerProtocol] = {}
       self.user_connections: Dict[str, Set[str]] = defaultdict(set)
       self.project_rooms: Dict[str, Set[str]] = defaultdict(set)
       self.connection_metadata: Dict[str, User] = {}
       self.rate_limiter = RateLimiter(config.rate_limit_messages, config.rate_limit_window)
       self.redis_client = redis_client
       self.presence_tracker: Dict[str, Dict] = {}  # user_id -> {status, last_seen}
       
   async def connect(self, websocket: WebSocketServerProtocol, user: User) -> str:
       """Register new WebSocket connection"""
       # Check connection limit
       if len(self.user_connections[user.id]) >= config.max_connections_per_user:
           raise ValueError(f"Maximum connections ({config.max_connections_per_user}) exceeded for user")
       
       connection_id = str(uuid.uuid4())
       user.connection_id = connection_id
       user.connected_at = datetime.utcnow()
       user.last_activity = datetime.utcnow()
       user.is_online = True
       
       self.active_connections[connection_id] = websocket
       self.user_connections[user.id].add(connection_id)
       self.connection_metadata[connection_id] = user
       
       # Add user to project rooms
       for project_id in user.projects:
           self.project_rooms[project_id].add(connection_id)
       
       # Update presence
       if config.enable_presence:
           await self._update_presence(user.id, "online")
       
       # Store connection in Redis if available
       if self.redis_client:
           await self._store_connection_redis(connection_id, user)
       
       logger.info(f"User {user.username} connected (ID: {connection_id})")
       
       # Send connection confirmation
       await self.send_personal_message(
           connection_id,
           Message(
               type=MessageType.AUTH,
               sender_id="system",
               recipient_id=user.id,
               data={"status": "connected", "user_id": user.id, "connection_id": connection_id}
           )
       )
       
       # Notify others about presence
       if config.enable_presence:
           await self._broadcast_presence_update(user.id, "online", user.projects)
       
       return connection_id
   
   async def disconnect(self, connection_id: str):
       """Remove WebSocket connection"""
       if connection_id not in self.connection_metadata:
           return
       
       user = self.connection_metadata[connection_id]
       
       # Remove from user connections
       self.user_connections[user.id].discard(connection_id)
       
       # Check if user has no more connections
       if not self.user_connections[user.id]:
           del self.user_connections[user.id]
           if config.enable_presence:
               await self._update_presence(user.id, "offline")
               await self._broadcast_presence_update(user.id, "offline", user.projects)
       
       # Remove from project rooms
       for project_id in user.projects:
           self.project_rooms[project_id].discard(connection_id)
           if not self.project_rooms[project_id]:
               del self.project_rooms[project_id]
       
       # Clean up connection data
       del self.connection_metadata[connection_id]
       if connection_id in self.active_connections:
           del self.active_connections[connection_id]
       
       # Remove from Redis if available
       if self.redis_client:
           await self._remove_connection_redis(connection_id)
       
       logger.info(f"User {user.username} disconnected (ID: {connection_id})")
   
   async def send_personal_message(self, connection_id: str, message: Message) -> bool:
       """Send message to specific connection"""
       if connection_id not in self.active_connections:
           return False
       
       websocket = self.active_connections[connection_id]
       try:
           message_data = asdict(message)
           message_data['timestamp'] = message.timestamp.isoformat()
           await websocket.send(json.dumps(message_data))
           
           # Store message if persistence enabled
           if config.enable_message_persistence and self.redis_client:
               await self._store_message_redis(message)
           
           return True
       except Exception as e:
           logger.error(f"Error sending message to {connection_id}: {e}")
           await self.disconnect(connection_id)
           return False
   
   async def send_to_user(self, user_id: str, message: Message) -> int:
       """Send message to all connections of a user"""
       sent_count = 0
       if user_id in self.user_connections:
           for connection_id in self.user_connections[user_id].copy():
               if await self.send_personal_message(connection_id, message):
                   sent_count += 1
       return sent_count
   
   async def broadcast_to_project(self, project_id: str, message: Message) -> int:
       """Broadcast message to all users in a project"""
       sent_count = 0
       if project_id in self.project_rooms:
           tasks = []
           for connection_id in self.project_rooms[project_id].copy():
               tasks.append(self.send_personal_message(connection_id, message))
           results = await asyncio.gather(*tasks, return_exceptions=True)
           sent_count = sum(1 for r in results if r is True)
       return sent_count
   
   async def broadcast_to_all(self, message: Message) -> int:
       """Broadcast message to all connected users"""
       tasks = []
       for connection_id in self.active_connections.copy():
           tasks.append(self.send_personal_message(connection_id, message))
       results = await asyncio.gather(*tasks, return_exceptions=True)
       return sum(1 for r in results if r is True)
   
   async def _update_presence(self, user_id: str, status: str):
       """Update user presence status"""
       self.presence_tracker[user_id] = {
           "status": status,
           "last_seen": datetime.utcnow().isoformat()
       }
       
       if self.redis_client:
           key = f"presence:{user_id}"
           await self.redis_client.setex(
               key,
               config.reconnect_timeout,
               json.dumps(self.presence_tracker[user_id])
           )
   
   async def _broadcast_presence_update(self, user_id: str, status: str, projects: List[str]):
       """Broadcast presence update to relevant users"""
       message = Message(
           type=MessageType.PRESENCE,
           sender_id="system",
           data={
               "user_id": user_id,
               "status": status,
               "timestamp": datetime.utcnow().isoformat()
           }
       )
       
       for project_id in projects:
           await self.broadcast_to_project(project_id, message)
   
   async def _store_connection_redis(self, connection_id: str, user: User):
       """Store connection info in Redis"""
       try:
           key = f"connection:{connection_id}"
           data = {
               "user_id": user.id,
               "username": user.username,
               "connected_at": user.connected_at.isoformat(),
               "projects": user.projects
           }
           await self.redis_client.setex(key, config.reconnect_timeout, json.dumps(data))
       except Exception as e:
           logger.error(f"Failed to store connection in Redis: {e}")
   
   async def _remove_connection_redis(self, connection_id: str):
       """Remove connection info from Redis"""
       try:
           key = f"connection:{connection_id}"
           await self.redis_client.delete(key)
       except Exception as e:
           logger.error(f"Failed to remove connection from Redis: {e}")
   
   async def _store_message_redis(self, message: Message):
       """Store message in Redis for persistence"""
       try:
           key = f"message:{message.id}"
           message_data = asdict(message)
           message_data['timestamp'] = message.timestamp.isoformat()
           await self.redis_client.setex(
               key,
               config.message_persistence_ttl,
               json.dumps(message_data)
           )
           
           # Add to project message list
           if message.project_id:
               list_key = f"messages:{message.project_id}"
               await self.redis_client.lpush(list_key, message.id)
               await self.redis_client.ltrim(list_key, 0, 999)  # Keep last 1000 messages
       except Exception as e:
           logger.error(f"Failed to store message in Redis: {e}")
   
   def check_rate_limit(self, user_id: str) -> Tuple[bool, Optional[int]]:
       """Check if user has exceeded rate limit"""
       return self.rate_limiter.check_rate_limit(user_id)
   
   async def get_user_presence(self, user_id: str) -> Optional[Dict]:
       """Get user presence status"""
       if user_id in self.presence_tracker:
           return self.presence_tracker[user_id]
       
       if self.redis_client:
           try:
               key = f"presence:{user_id}"
               data = await self.redis_client.get(key)
               if data:
                   return json.loads(data)
           except Exception as e:
               logger.error(f"Failed to get presence from Redis: {e}")
       
       return None
   
   async def get_project_presence(self, project_id: str) -> List[Dict]:
       """Get presence status for all users in a project"""
       presence_list = []
       if project_id in self.project_rooms:
           for connection_id in self.project_rooms[project_id]:
               if connection_id in self.connection_metadata:
                   user = self.connection_metadata[connection_id]
                   presence = await self.get_user_presence(user.id)
                   if presence:
                       presence['user_id'] = user.id
                       presence['username'] = user.username
                       presence_list.append(presence)
       return presence_list


# ============================================================================
# Chat Service with LLM Integration
# ============================================================================

class ChatService:
   """Handles real-time chat with LLM integration"""
   
   def __init__(self, redis_client: Optional[redis.Redis] = None):
       self.llm_endpoint = config.llm_endpoint
       self.conversation_history: Dict[str, List[ChatMessage]] = defaultdict(list)
       self.active_sessions: Dict[str, Dict] = {}
       self.redis_client = redis_client
       self.typing_status: Dict[str, Set[str]] = defaultdict(set)  # project_id -> Set[user_id]
       
   async def process_chat_message(
       self,
       message: ChatMessage,
       connection_manager: ConnectionManager
   ) -> Dict[str, Any]:
       """Process incoming chat message and get LLM response"""
       
       # Store message in conversation history
       self.conversation_history[message.project_id].append(message)
       
       # Store in Redis if available
       if self.redis_client:
           await self._store_message_history(message)
       
       # Broadcast user message to project room
       await connection_manager.broadcast_to_project(
           message.project_id,
           Message(
               type=MessageType.CHAT,
               sender_id=message.sender_id,
               project_id=message.project_id,
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
               type=MessageType.CHAT,
               sender_id="llm_assistant",
               project_id=message.project_id,
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
               context = await self._prepare_context(message.project_id)
               
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
   
   async def _prepare_context(self, project_id: str, max_messages: int = 10) -> List[Dict]:
       """Prepare conversation context for LLM"""
       # Try to get from Redis first
       if self.redis_client:
           try:
               history = await self._get_message_history(project_id, max_messages)
               if history:
                   return history
           except Exception as e:
               logger.error(f"Failed to get message history from Redis: {e}")
       
       # Fallback to in-memory history
       history = self.conversation_history.get(project_id, [])
       recent_messages = history[-max_messages:] if len(history) > max_messages else history
       
       return [
           {
               "role": "assistant" if msg.sender_id == "llm_assistant" else "user",
               "content": msg.content,
               "timestamp": datetime.utcnow().isoformat()
           }
           for msg in recent_messages
       ]
   
   async def _store_message_history(self, message: ChatMessage):
       """Store message history in Redis"""
       try:
           key = f"chat_history:{message.project_id}"
           data = message.json()
           await self.redis_client.lpush(key, data)
           await self.redis_client.ltrim(key, 0, 99)  # Keep last 100 messages
           await self.redis_client.expire(key, config.message_persistence_ttl)
       except Exception as e:
           logger.error(f"Failed to store message history: {e}")
   
   async def _get_message_history(self, project_id: str, limit: int = 10) -> List[Dict]:
       """Get message history from Redis"""
       try:
           key = f"chat_history:{project_id}"
           messages = await self.redis_client.lrange(key, 0, limit - 1)
           return [json.loads(msg) for msg in messages]
       except Exception as e:
           logger.error(f"Failed to get message history: {e}")
           return []
   
   async def handle_typing_indicator(
       self,
       user_id: str,
       project_id: str,
       is_typing: bool,
       connection_manager: ConnectionManager
   ):
       """Handle typing indicator updates"""
       if not config.enable_typing_indicators:
           return
       
       if is_typing:
           self.typing_status[project_id].add(user_id)
       else:
           self.typing_status[project_id].discard(user_id)
       
       # Broadcast typing status
       await connection_manager.broadcast_to_project(
           project_id,
           Message(
               type=MessageType.TYPING,
               sender_id=user_id,
               project_id=project_id,
               data={
                   "user_id": user_id,
                   "is_typing": is_typing,
                   "typing_users": list(self.typing_status[project_id])
               }
           )
       )


# ============================================================================
# Notification Service
# ============================================================================

class NotificationService:
   """Handles push notifications for various events"""
   
   def __init__(self, connection_manager: ConnectionManager, redis_client: Optional[redis.Redis] = None):
       self.connection_manager = connection_manager
       self.notification_queue: asyncio.Queue = asyncio.Queue()
       self.notification_history: List[Notification] = []
       self.redis_client = redis_client
       
   async def send_notification(self, notification: Notification) -> str:
       """Send notification to specified users or project"""
       notification_id = str(uuid.uuid4())
       
       message = Message(
           id=notification_id,
           type=MessageType.NOTIFICATION,
           sender_id="system",
           project_id=notification.project_id,
           data=notification.dict(),
           metadata={"priority": notification.priority.value}
       )
       
       sent_count = 0
       
       if notification.user_ids:
           # Send to specific users
           for user_id in notification.user_ids:
               sent_count += await self.connection_manager.send_to_user(user_id, message)
       elif notification.project_id:
           # Send to project room
           sent_count = await self.connection_manager.broadcast_to_project(
               notification.project_id, message
           )
       else:
           # Broadcast to all
           sent_count = await self.connection_manager.broadcast_to_all(message)
       
       # Store in history
       self.notification_history.append(notification)
       if len(self.notification_history) > 1000:
           self.notification_history = self.notification_history[-1000:]
       
       # Store in Redis if available
       if self.redis_client:
           await self._store_notification(notification_id, notification)
       
       # Log high priority notifications
       if notification.priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL]:
           logger.warning(f"High priority notification: {notification.title}")
       
       logger.info(f"Notification {notification_id} sent to {sent_count} connections")
       return notification_id
   
   async def _store_notification(self, notification_id: str, notification: Notification):
       """Store notification in Redis"""
       try:
           key = f"notification:{notification_id}"
           data = notification.json()
           ttl = config.message_persistence_ttl
           
           if notification.expires_at:
               ttl = int((notification.expires_at - datetime.utcnow()).total_seconds())
           
           await self.redis_client.setex(key, ttl, data)
           
           # Add to user notification lists
           if notification.user_ids:
               for user_id in notification.user_ids:
                   list_key = f"notifications:{user_id}"
                   await self.redis_client.lpush(list_key, notification_id)
                   await self.redis_client.ltrim(list_key, 0, 99)
       except Exception as e:
           logger.error(f"Failed to store notification: {e}")
   
   async def get_user_notifications(self, user_id: str, limit: int = 20) -> List[Dict]:
       """Get notifications for a user"""
       notifications = []
       
       if self.redis_client:
           try:
               list_key = f"notifications:{user_id}"
               notification_ids = await self.redis_client.lrange(list_key, 0, limit - 1)
               
               for nid in notification_ids:
                   key = f"notification:{nid.decode() if isinstance(nid, bytes) else nid}"
                   data = await self.redis_client.get(key)
                   if data:
                       notifications.append(json.loads(data))
           except Exception as e:
               logger.error(f"Failed to get notifications: {e}")
       
       return notifications
   
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
           type=MessageType.TRAINING_UPDATE,
           sender_id="system",
           project_id=project_id,
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
       if len(self.metrics_buffer[training_id]) > 1000:
           self.metrics_buffer[training_id] = self.metrics_buffer[training_id][-1000:]
       
       # Prepare aggregated metrics
       aggregated = self._aggregate_metrics(training_id)
       
       # Broadcast metrics update
       message = Message(
           type=MessageType.METRICS,
           sender_id="system",
           project_id=project_id,
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
           type=MessageType.TRAINING_UPDATE,
           sender_id="system",
           project_id=project_id,
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
           "avg_loss": float(np.mean([m.loss for m in recent_metrics])),
           "avg_accuracy": float(np.mean([m.accuracy for m in recent_metrics if m.accuracy])) if any(m.accuracy for m in recent_metrics) else None,
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
           "final_loss": float(metrics_list[-1].loss),
           "final_accuracy": float(metrics_list[-1].accuracy) if metrics_list[-1].accuracy else None,
           "best_loss": float(min(m.loss for m in metrics_list)),
           "best_accuracy": float(max(m.accuracy for m in metrics_list if m.accuracy)) if any(m.accuracy for m in metrics_list) else None,
           "total_time": metrics_list[-1].time_elapsed,
           "avg_batch_time": metrics_list[-1].time_elapsed / metrics_list[-1].batch if metrics_list[-1].batch > 0 else 0
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
           type=MessageType.COLLABORATION,
           sender_id=user_id,
           project_id=project_id,
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
           locks_to_release = [k for k, v in self.document_locks.items() if v == user_id and k.startswith(document_id)]
           for lock_key in locks_to_release:
               del self.document_locks[lock_key]
           
           # Remove cursor position
           if user_id in self.cursor_positions:
               del self.cursor_positions[user_id]
           
           # Notify other users
           message = Message(
               type=MessageType.COLLABORATION,
               sender_id=user_id,
               project_id=project_id,
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


# ============================================================================
# WebSocket Server
# ============================================================================

class WebSocketServer:
   """Main WebSocket server handling all real-time features"""
   
   def __init__(self, redis_url: str = None):
       self.redis_url = redis_url or config.redis_url
       self.redis_client: Optional[redis.Redis] = None
       self.connection_manager: Optional[ConnectionManager] = None
       self.chat_service: Optional[ChatService] = None
       self.notification_service: Optional[NotificationService] = None
       self.monitoring_service: Optional[LiveMonitoringService] = None
       self.collaboration_service: Optional[CollaborationService] = None
       self._initialized = False
       
   async def initialize(self):
       """Initialize server components"""
       if self._initialized:
           return
       
       try:
           # Initialize Redis with retry logic
           self.redis_client = await self._connect_redis_with_retry()
           
           # Initialize services
           self.connection_manager = ConnectionManager(self.redis_client)
           self.chat_service = ChatService(self.redis_client)
           self.notification_service = NotificationService(self.connection_manager, self.redis_client)
           self.monitoring_service = LiveMonitoringService(self.connection_manager)
           self.collaboration_service = CollaborationService(self.connection_manager)
           
           self._initialized = True
           logger.info("WebSocket server initialized successfully")
           
       except Exception as e:
           logger.error(f"Failed to initialize WebSocket server: {e}")
           # Continue without Redis if it's not available
           self.connection_manager = ConnectionManager(None)
           self.chat_service = ChatService(None)
           self.notification_service = NotificationService(self.connection_manager, None)
           self.monitoring_service = LiveMonitoringService(self.connection_manager)
           self.collaboration_service = CollaborationService(self.connection_manager)
           self._initialized = True
   
   async def _connect_redis_with_retry(self, max_retries: int = 3, delay: int = 2) -> Optional[redis.Redis]:
       """Connect to Redis with retry logic"""
       for attempt in range(max_retries):
           try:
               client = redis.from_url(self.redis_url)
               await client.ping()
               logger.info("Connected to Redis")
               return client
           except Exception as e:
               logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
               if attempt < max_retries - 1:
                   await asyncio.sleep(delay)
       
       logger.error("Failed to connect to Redis after all retries")
       return None
   
   async def cleanup(self):
       """Cleanup server resources"""
       if self.redis_client:
           await self.redis_client.close()
       self._initialized = False
       logger.info("WebSocket server cleaned up")
   
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
           
           # Decode and verify token
           try:
               payload = jwt.decode(
                   token,
                   config.jwt_secret,
                   algorithms=[config.jwt_algorithm]
               )
               user = User(
                   id=payload["user_id"],
                   username=payload["username"],
                   email=payload.get("email", ""),
                   roles=payload.get("roles", ["user"]),
                   projects=payload.get("projects", [])
               )
               return user
               
           except jwt.InvalidTokenError as e:
               await websocket.send(json.dumps({
                   "type": "error",
                   "message": f"Invalid token: {str(e)}"
               }))
               return None
               
       except asyncio.TimeoutError:
           await websocket.send(json.dumps({
               "type": "error",
               "message": "Authentication timeout"
           }))
           return None
       except Exception as e:
           logger.error(f"Authentication error: {e}\n{traceback.format_exc()}")
           return None
   
   async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
       """Handle incoming WebSocket connection"""
       if not self._initialized:
           await self.initialize()
       
       connection_id = None
       heartbeat_task = None
       
       try:
           # Authenticate user
           user = await self.authenticate_user(websocket)
           if not user:
               await websocket.close()
               return
           
           # Register connection
           connection_id = await self.connection_manager.connect(websocket, user)
           
           # Start heartbeat
           heartbeat_task = asyncio.create_task(self._send_heartbeat(connection_id))
           
           # Handle messages
           async for message in websocket:
               # Check rate limit
               is_allowed, reset_in = self.connection_manager.check_rate_limit(user.id)
               if not is_allowed:
                   await self.connection_manager.send_personal_message(
                       connection_id,
                       Message(
                           type=MessageType.ERROR,
                           sender_id="system",
                           data={
                               "error": "Rate limit exceeded",
                               "reset_in_seconds": reset_in
                           }
                       )
                   )
                   continue
               
               await self.handle_message(connection_id, message)
               
       except WebSocketDisconnect:
           logger.info(f"WebSocket disconnected: {connection_id}")
       except Exception as e:
           logger.error(f"WebSocket error: {e}\n{traceback.format_exc()}")
       finally:
           if heartbeat_task:
               heartbeat_task.cancel()
           if connection_id:
               await self.connection_manager.disconnect(connection_id)
   
   async def _send_heartbeat(self, connection_id: str):
       """Send periodic heartbeat to keep connection alive"""
       while True:
           try:
               await asyncio.sleep(config.heartbeat_interval)
               await self.connection_manager.send_personal_message(
                   connection_id,
                   Message(
                       type=MessageType.HEARTBEAT,
                       sender_id="system",
                       data={"status": "alive", "timestamp": datetime.utcnow().isoformat()}
                   )
               )
           except asyncio.CancelledError:
               break
           except Exception as e:
               logger.error(f"Heartbeat error: {e}")
               break
   
   async def handle_message(self, connection_id: str, raw_message: str):
       """Route incoming messages to appropriate handlers"""
       try:
           data = json.loads(raw_message)
           message_type = MessageType(data.get("type"))
           
           user = self.connection_manager.connection_metadata.get(connection_id)
           if not user:
               logger.error(f"User not found for connection {connection_id}")
               return
           
           # Update last activity
           user.last_activity = datetime.utcnow()
           
           # Route message based on type
           if message_type == MessageType.CHAT:
               await self.handle_chat_message(connection_id, data)
           elif message_type == MessageType.COLLABORATION:
               await self.handle_collaboration_message(connection_id, data)
           elif message_type == MessageType.HEARTBEAT:
               await self.handle_heartbeat(connection_id)
           elif message_type == MessageType.TYPING:
               await self.handle_typing_indicator(connection_id, data)
           else:
               logger.warning(f"Unknown message type: {message_type}")
               
       except json.JSONDecodeError:
           logger.error(f"Invalid JSON received: {raw_message[:100]}")
       except Exception as e:
           logger.error(f"Error handling message: {e}\n{traceback.format_exc()}")
   
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
   
   async def handle_typing_indicator(self, connection_id: str, data: Dict):
       """Handle typing indicator"""
       user = self.connection_manager.connection_metadata[connection_id]
       
       await self.chat_service.handle_typing_indicator(
           user.id,
           data["project_id"],
           data["is_typing"],
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
   
   async def handle_heartbeat(self, connection_id: str):
       """Handle heartbeat messages"""
       await self.connection_manager.send_personal_message(
           connection_id,
           Message(
               type=MessageType.ACK,
               sender_id="system",
               data={"status": "alive", "timestamp": datetime.utcnow().isoformat()}
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

# Security
security = HTTPBearer()

# Create FastAPI app
app = FastAPI(
   title="MLOps WebSocket Service",
   version="3.0.1",
   description="Real-time WebSocket service for MLOps platform"
)

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

# Global server instance
ws_server: Optional[WebSocketServer] = None

async def get_ws_server() -> WebSocketServer:
   """Get or create WebSocket server instance"""
   global ws_server
   if not ws_server:
       ws_server = WebSocketServer()
       await ws_server.initialize()
   return ws_server

@app.on_event("startup")
async def startup_event():
   """Initialize WebSocket server on startup"""
   await get_ws_server()
   logger.info("WebSocket service started")

@app.on_event("shutdown")
async def shutdown_event():
   """Cleanup on shutdown"""
   global ws_server
   if ws_server:
       await ws_server.cleanup()
   logger.info("WebSocket service stopped")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
   """FastAPI WebSocket endpoint"""
   server = await get_ws_server()
   await websocket.accept()
   await server.handle_connection(websocket, "/ws")

@app.post("/api/notifications/send")
async def send_notification(
   notification: Notification,
   credentials: HTTPAuthorizationCredentials = Depends(security)
):
   """REST endpoint to send notifications"""
   server = await get_ws_server()
   notification_id = await server.notification_service.send_notification(notification)
   return {"status": "sent", "notification_id": notification_id}

@app.get("/api/notifications/{user_id}")
async def get_notifications(
   user_id: str,
   limit: int = 20,
   credentials: HTTPAuthorizationCredentials = Depends(security)
):
   """Get notifications for a user"""
   server = await get_ws_server()
   notifications = await server.notification_service.get_user_notifications(user_id, limit)
   return {"notifications": notifications}

@app.post("/api/training/{training_id}/metrics")
async def update_training_metrics(
   training_id: str,
   metrics: TrainingMetrics,
   credentials: HTTPAuthorizationCredentials = Depends(security)
):
   """REST endpoint to update training metrics"""
   server = await get_ws_server()
   await server.monitoring_service.update_metrics(training_id, metrics)
   return {"status": "updated"}

@app.post("/api/training/{training_id}/start")
async def start_training_monitoring(
   training_id: str,
   project_id: str = "default",
   config: Dict = None,
   credentials: HTTPAuthorizationCredentials = Depends(security)
):
   """REST endpoint to start training monitoring"""
   server = await get_ws_server()
   await server.monitoring_service.start_monitoring(
       project_id,
       training_id,
       config or {}
   )
   return {"status": "monitoring_started"}

@app.post("/api/training/{training_id}/stop")
async def stop_training_monitoring(
   training_id: str,
   status: str = "completed",
   credentials: HTTPAuthorizationCredentials = Depends(security)
):
   """REST endpoint to stop training monitoring"""
   server = await get_ws_server()
   await server.monitoring_service.stop_monitoring(training_id, status)
   return {"status": "monitoring_stopped"}

@app.get("/api/presence/{project_id}")
async def get_project_presence(
   project_id: str,
   credentials: HTTPAuthorizationCredentials = Depends(security)
):
   """Get presence information for a project"""
   server = await get_ws_server()
   presence = await server.connection_manager.get_project_presence(project_id)
   return {"project_id": project_id, "presence": presence}

@app.get("/api/health")
async def health_check():
   """Health check endpoint"""
   server = await get_ws_server()
   return {
       "status": "healthy",
       "timestamp": datetime.utcnow().isoformat(),
       "active_connections": len(server.connection_manager.active_connections),
       "active_trainings": len(server.monitoring_service.active_trainings),
       "redis_connected": server.redis_client is not None
   }

@app.get("/api/stats")
async def get_stats(credentials: HTTPAuthorizationCredentials = Depends(security)):
   """Get WebSocket service statistics"""
   server = await get_ws_server()
   return {
       "timestamp": datetime.utcnow().isoformat(),
       "connections": {
           "total": len(server.connection_manager.active_connections),
           "by_user": {
               uid: len(conns) 
               for uid, conns in server.connection_manager.user_connections.items()
           },
           "by_project": {
               pid: len(conns) 
               for pid, conns in server.connection_manager.project_rooms.items()
           }
       },
       "trainings": {
           "active": len(server.monitoring_service.active_trainings),
           "metrics_buffered": sum(
               len(metrics) 
               for metrics in server.monitoring_service.metrics_buffer.values()
           )
       },
       "notifications": {
           "history_size": len(server.notification_service.notification_history)
       }
   }

# ============================================================================
# Export Functions for __init__.py
# ============================================================================

# Global connection manager for import
connection_manager: Optional[ConnectionManager] = None

async def initialize_websocket_service(redis_url: str = None):
   """Initialize WebSocket service for import in __init__.py"""
   global ws_server, connection_manager
   ws_server = WebSocketServer(redis_url)
   await ws_server.initialize()
   connection_manager = ws_server.connection_manager
   return ws_server

async def shutdown_websocket_service():
   """Cleanup WebSocket service"""
   global ws_server, connection_manager
   if ws_server is not None:
       await ws_server.cleanup()
   ws_server = None  # Always assign to satisfy the global declaration
   connection_manager = None

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
