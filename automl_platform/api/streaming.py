"""
Streaming module for real-time data processing
Supports Kafka, Flink, Pulsar and other streaming platforms
WITH PROMETHEUS METRICS INTEGRATION
Place in: automl_platform/api/streaming.py
"""

import json
import logging
import asyncio
import os
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import deque
import threading
import time
from abc import ABC, abstractmethod

# Métriques Prometheus
from prometheus_client import Counter, Histogram, Gauge

# Kafka support
try:
    from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
    from kafka.admin import NewTopic
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

# Flink support
try:
    from pyflink.datastream import StreamExecutionEnvironment
    from pyflink.table import StreamTableEnvironment, DataTypes
    from pyflink.table.descriptors import Schema, Kafka, Json
    from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
    FLINK_AVAILABLE = True
except ImportError:
    FLINK_AVAILABLE = False

# Pulsar support
try:
    import pulsar
    PULSAR_AVAILABLE = True
except ImportError:
    PULSAR_AVAILABLE = False

# Redis Streams support
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Déclaration des métriques Prometheus
ml_streaming_messages_total = Counter(
    'ml_streaming_messages_total',
    'Total number of streaming messages processed',
    ['tenant_id', 'platform', 'topic', 'status']  # status: success, failure
)

ml_streaming_throughput = Gauge(
    'ml_streaming_throughput',
    'Current message throughput per second',
    ['tenant_id', 'platform', 'topic']
)

ml_streaming_lag = Gauge(
    'ml_streaming_lag',
    'Consumer lag in messages',
    ['tenant_id', 'platform', 'topic', 'partition']
)

ml_streaming_processing_latency_seconds = Histogram(
    'ml_streaming_processing_latency_seconds',
    'Message processing latency',
    ['tenant_id', 'platform', 'processor_type'],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0)
)

ml_streaming_batch_size = Histogram(
    'ml_streaming_batch_size',
    'Size of processed batches',
    ['tenant_id', 'platform'],
    buckets=(1, 10, 50, 100, 500, 1000, 5000)
)

ml_streaming_errors_total = Counter(
    'ml_streaming_errors_total',
    'Total number of streaming errors',
    ['tenant_id', 'platform', 'error_type']
)

ml_streaming_active_consumers = Gauge(
    'ml_streaming_active_consumers',
    'Number of active streaming consumers',
    ['platform', 'topic']
)


@dataclass
class StreamConfig:
    """Configuration for streaming pipeline."""
    platform: str  # kafka, flink, pulsar, redis
    brokers: List[str]
    topic: str
    consumer_group: str = "automl-consumer"
    batch_size: int = 100
    window_size: int = 60  # seconds
    checkpoint_interval: int = 30  # seconds
    max_latency_ms: int = 1000
    enable_exactly_once: bool = False
    
    # Authentication
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str = None
    sasl_username: str = None
    sasl_password: str = None
    
    # Schema
    schema: Dict[str, str] = None
    
    # Tenant info for metrics
    tenant_id: str = "default"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class StreamMessage:
    """Message format for streaming."""
    key: str
    value: Dict[str, Any]
    timestamp: datetime
    partition: int = 0
    offset: int = 0
    headers: Dict[str, str] = None


class StreamProcessor(ABC):
    """Base class for stream processing with metrics."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.running = False
        self.processed_count = 0
        self.error_count = 0
        self.last_checkpoint = datetime.now()
        self.last_throughput_calc = datetime.now()
        self.messages_since_last_calc = 0
        self.metrics = {
            "messages_processed": 0,
            "messages_failed": 0,
            "avg_latency_ms": 0,
            "throughput_per_sec": 0
        }
        
        # Increment active consumers gauge
        ml_streaming_active_consumers.labels(
            platform=config.platform,
            topic=config.topic
        ).inc()
        
    @abstractmethod
    async def process(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process single message."""
        pass
    
    @abstractmethod
    async def process_batch(self, messages: List[StreamMessage]) -> List[StreamMessage]:
        """Process batch of messages."""
        pass
    
    def checkpoint(self):
        """Save checkpoint."""
        self.last_checkpoint = datetime.now()
        logger.info(f"Checkpoint: processed={self.processed_count}, errors={self.error_count}")
    
    def update_throughput_metric(self):
        """Update throughput gauge metric."""
        now = datetime.now
