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
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

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

# Créer un registre local pour les métriques de streaming
streaming_registry = CollectorRegistry()

# Déclaration des métriques Prometheus avec le registre local
ml_streaming_messages_total = Counter(
    'ml_streaming_messages_total',
    'Total number of streaming messages processed',
    ['tenant_id', 'platform', 'topic', 'status'],  # status: success, failure
    registry=streaming_registry
)

ml_streaming_throughput = Gauge(
    'ml_streaming_throughput',
    'Current message throughput per second',
    ['tenant_id', 'platform', 'topic'],
    registry=streaming_registry
)

ml_streaming_lag = Gauge(
    'ml_streaming_lag',
    'Consumer lag in messages',
    ['tenant_id', 'platform', 'topic', 'partition'],
    registry=streaming_registry
)

ml_streaming_processing_latency_seconds = Histogram(
    'ml_streaming_processing_latency_seconds',
    'Message processing latency',
    ['tenant_id', 'platform', 'processor_type'],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0),
    registry=streaming_registry
)

ml_streaming_batch_size = Histogram(
    'ml_streaming_batch_size',
    'Size of processed batches',
    ['tenant_id', 'platform'],
    buckets=(1, 10, 50, 100, 500, 1000, 5000),
    registry=streaming_registry
)

ml_streaming_errors_total = Counter(
    'ml_streaming_errors_total',
    'Total number of streaming errors',
    ['tenant_id', 'platform', 'error_type'],
    registry=streaming_registry
)

ml_streaming_active_consumers = Gauge(
    'ml_streaming_active_consumers',
    'Number of active streaming consumers',
    ['platform', 'topic'],
    registry=streaming_registry
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
        now = datetime.now()
        time_diff = (now - self.last_throughput_calc).total_seconds()
        
        if time_diff > 0:
            throughput = self.messages_since_last_calc / time_diff
            ml_streaming_throughput.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                topic=self.config.topic
            ).set(throughput)
            
            self.metrics["throughput_per_sec"] = throughput
            self.last_throughput_calc = now
            self.messages_since_last_calc = 0
    
    def __del__(self):
        """Decrement active consumers when processor is destroyed."""
        try:
            ml_streaming_active_consumers.labels(
                platform=self.config.platform,
                topic=self.config.topic
            ).dec()
        except:
            pass  # Ignore errors during cleanup


class KafkaStreamProcessor(StreamProcessor):
    """Kafka-based stream processor with metrics."""
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python not installed. Install with: pip install kafka-python")
        
        # Consumer configuration
        self.consumer_config = {
            'bootstrap_servers': config.brokers,
            'group_id': config.consumer_group,
            'auto_offset_reset': 'latest',
            'enable_auto_commit': not config.enable_exactly_once,
            'max_poll_records': config.batch_size,
            'max_poll_interval_ms': 300000,
            'session_timeout_ms': 10000,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')) if m else None
        }
        
        # Add authentication if configured
        if config.security_protocol != "PLAINTEXT":
            self.consumer_config['security_protocol'] = config.security_protocol
            if config.sasl_mechanism:
                self.consumer_config['sasl_mechanism'] = config.sasl_mechanism
                self.consumer_config['sasl_plain_username'] = config.sasl_username
                self.consumer_config['sasl_plain_password'] = config.sasl_password
        
        self.consumer = None
        self.producer = None
        self._init_clients()
    
    def _init_clients(self):
        """Initialize Kafka clients."""
        try:
            # Create consumer
            self.consumer = KafkaConsumer(
                self.config.topic,
                **self.consumer_config
            )
            
            # Create producer for output
            producer_config = {
                'bootstrap_servers': self.config.brokers,
                'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
                'acks': 'all' if self.config.enable_exactly_once else 1,
                'retries': 3,
                'max_in_flight_requests_per_connection': 1 if self.config.enable_exactly_once else 5
            }
            
            self.producer = KafkaProducer(**producer_config)
            
            logger.info(f"Initialized Kafka clients for topic: {self.config.topic}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka clients: {e}")
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,  # Utilisation de self.config.platform au lieu de "kafka"
                error_type="initialization"
            ).inc()
            raise
    
    async def process(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process single Kafka message with metrics."""
        start_time = time.time()
        
        try:
            # Process message (to be implemented by subclass)
            result = await self._process_message_impl(message)
            
            # Update metrics
            self.processed_count += 1
            self.messages_since_last_calc += 1
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,  # Utilisation de self.config.platform au lieu de "kafka"
                topic=self.config.topic,
                status="success"
            ).inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.error_count += 1
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,  # Utilisation de self.config.platform au lieu de "kafka"
                topic=self.config.topic,
                status="failure"
            ).inc()
            
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,  # Utilisation de self.config.platform au lieu de "kafka"
                error_type=type(e).__name__
            ).inc()
            
            return None
            
        finally:
            # Record processing latency
            ml_streaming_processing_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,  # Utilisation de self.config.platform au lieu de "kafka"
                processor_type=self.__class__.__name__
            ).observe(time.time() - start_time)
    
    async def process_batch(self, messages: List[StreamMessage]) -> List[StreamMessage]:
        """Process batch of Kafka messages with metrics."""
        start_time = time.time()
        
        # Record batch size
        ml_streaming_batch_size.labels(
            tenant_id=self.config.tenant_id,
            platform=self.config.platform  # Utilisation de self.config.platform au lieu de "kafka"
        ).observe(len(messages))
        
        try:
            # Process batch (to be implemented by subclass)
            results = await self._process_batch_impl(messages)
            
            # Update metrics
            self.processed_count += len(messages)
            self.messages_since_last_calc += len(messages)
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,  # Utilisation de self.config.platform au lieu de "kafka"
                topic=self.config.topic,
                status="success"
            ).inc(len(messages))
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.error_count += len(messages)
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,  # Utilisation de self.config.platform au lieu de "kafka"
                topic=self.config.topic,
                status="failure"
            ).inc(len(messages))
            
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,  # Utilisation de self.config.platform au lieu de "kafka"
                error_type=type(e).__name__
            ).inc()
            
            return []
            
        finally:
            # Record processing latency
            ml_streaming_processing_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,  # Utilisation de self.config.platform au lieu de "kafka"
                processor_type=self.__class__.__name__
            ).observe(time.time() - start_time)
    
    async def _process_message_impl(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Actual message processing logic - to be implemented by subclass."""
        # Default implementation - just pass through
        return message
    
    async def _process_batch_impl(self, messages: List[StreamMessage]) -> List[StreamMessage]:
        """Actual batch processing logic - to be implemented by subclass."""
        # Default implementation - process each message individually
        results = []
        for msg in messages:
            result = await self._process_message_impl(msg)
            if result:
                results.append(result)
        return results
    
    def run(self):
        """Run the Kafka consumer loop."""
        self.running = True
        
        try:
            while self.running:
                # Poll for messages
                records = self.consumer.poll(timeout_ms=1000)
                
                if records:
                    # Process messages in batches
                    for topic_partition, messages in records.items():
                        # Convert to StreamMessage format
                        stream_messages = []
                        for msg in messages:
                            stream_msg = StreamMessage(
                                key=msg.key.decode('utf-8') if msg.key else None,
                                value=msg.value,
                                timestamp=datetime.fromtimestamp(msg.timestamp / 1000),
                                partition=msg.partition,
                                offset=msg.offset,
                                headers=dict(msg.headers) if msg.headers else None
                            )
                            stream_messages.append(stream_msg)
                        
                        # Process batch
                        asyncio.run(self.process_batch(stream_messages))
                        
                        # Update lag metric
                        highwater = self.consumer.highwater(topic_partition)
                        lag = highwater - msg.offset if highwater else 0
                        
                        ml_streaming_lag.labels(
                            tenant_id=self.config.tenant_id,
                            platform=self.config.platform,  # Utilisation de self.config.platform au lieu de "kafka"
                            topic=topic_partition.topic,
                            partition=str(topic_partition.partition)
                        ).set(lag)
                
                # Update throughput periodically
                if (datetime.now() - self.last_throughput_calc).seconds > 5:
                    self.update_throughput_metric()
                
                # Checkpoint periodically
                if (datetime.now() - self.last_checkpoint).seconds > self.config.checkpoint_interval:
                    self.checkpoint()
                    if not self.config.enable_exactly_once:
                        self.consumer.commit()
                        
        except Exception as e:
            logger.error(f"Error in Kafka consumer loop: {e}")
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,  # Utilisation de self.config.platform au lieu de "kafka"
                error_type="consumer_loop"
            ).inc()
            
        finally:
            self.stop()
    
    def stop(self):
        """Stop the Kafka consumer."""
        self.running = False
        
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.close()
        
        logger.info("Kafka stream processor stopped")


class PulsarStreamProcessor(StreamProcessor):
    """Pulsar-based stream processor with metrics."""
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        
        if not PULSAR_AVAILABLE:
            raise ImportError("pulsar-client not installed. Install with: pip install pulsar-client")
        
        self.client = None
        self.consumer = None
        self.producer = None
        self._init_clients()
    
    def _init_clients(self):
        """Initialize Pulsar clients."""
        try:
            # Create Pulsar client
            self.client = pulsar.Client(
                service_url=f'pulsar://{self.config.brokers[0]}',
                authentication=None  # Add authentication if needed
            )
            
            # Create consumer
            self.consumer = self.client.subscribe(
                topic=self.config.topic,
                subscription_name=self.config.consumer_group,
                consumer_type=pulsar.ConsumerType.Shared
            )
            
            # Create producer
            self.producer = self.client.create_producer(
                topic=f'{self.config.topic}-output'
            )
            
            logger.info(f"Initialized Pulsar clients for topic: {self.config.topic}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pulsar clients: {e}")
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                error_type="initialization"
            ).inc()
            raise
    
    async def process(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process single Pulsar message with metrics."""
        start_time = time.time()
        
        try:
            result = await self._process_message_impl(message)
            
            self.processed_count += 1
            self.messages_since_last_calc += 1
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                topic=self.config.topic,
                status="success"
            ).inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.error_count += 1
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                topic=self.config.topic,
                status="failure"
            ).inc()
            
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                error_type=type(e).__name__
            ).inc()
            
            return None
            
        finally:
            ml_streaming_processing_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                processor_type=self.__class__.__name__
            ).observe(time.time() - start_time)
    
    async def process_batch(self, messages: List[StreamMessage]) -> List[StreamMessage]:
        """Process batch of Pulsar messages with metrics."""
        start_time = time.time()
        
        ml_streaming_batch_size.labels(
            tenant_id=self.config.tenant_id,
            platform=self.config.platform
        ).observe(len(messages))
        
        try:
            results = await self._process_batch_impl(messages)
            
            self.processed_count += len(messages)
            self.messages_since_last_calc += len(messages)
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                topic=self.config.topic,
                status="success"
            ).inc(len(messages))
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.error_count += len(messages)
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                topic=self.config.topic,
                status="failure"
            ).inc(len(messages))
            
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                error_type=type(e).__name__
            ).inc()
            
            return []
            
        finally:
            ml_streaming_processing_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                processor_type=self.__class__.__name__
            ).observe(time.time() - start_time)
    
    async def _process_message_impl(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Actual message processing logic - to be implemented by subclass."""
        return message
    
    async def _process_batch_impl(self, messages: List[StreamMessage]) -> List[StreamMessage]:
        """Actual batch processing logic - to be implemented by subclass."""
        results = []
        for msg in messages:
            result = await self._process_message_impl(msg)
            if result:
                results.append(result)
        return results
    
    def run(self):
        """Run the Pulsar consumer loop."""
        self.running = True
        
        try:
            while self.running:
                try:
                    # Receive message with timeout
                    msg = self.consumer.receive(timeout_millis=1000)
                    
                    # Convert to StreamMessage
                    stream_msg = StreamMessage(
                        key=msg.partition_key() if msg.partition_key() else None,
                        value=json.loads(msg.data().decode('utf-8')),
                        timestamp=datetime.fromtimestamp(msg.publish_timestamp() / 1000),
                        partition=0,  # Pulsar doesn't expose partition directly
                        offset=0,  # Use message ID instead
                        headers=msg.properties() if msg.properties() else None
                    )
                    
                    # Process message
                    asyncio.run(self.process(stream_msg))
                    
                    # Acknowledge message
                    self.consumer.acknowledge(msg)
                    
                except Exception as timeout:
                    # Timeout is normal, continue
                    pass
                
                # Update throughput periodically
                if (datetime.now() - self.last_throughput_calc).seconds > 5:
                    self.update_throughput_metric()
                
                # Checkpoint periodically
                if (datetime.now() - self.last_checkpoint).seconds > self.config.checkpoint_interval:
                    self.checkpoint()
                    
        except Exception as e:
            logger.error(f"Error in Pulsar consumer loop: {e}")
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                error_type="consumer_loop"
            ).inc()
            
        finally:
            self.stop()
    
    def stop(self):
        """Stop the Pulsar consumer."""
        self.running = False
        
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.close()
        
        if self.client:
            self.client.close()
        
        logger.info("Pulsar stream processor stopped")


class FlinkStreamProcessor(StreamProcessor):
    """Flink-based stream processor with metrics."""
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        
        if not FLINK_AVAILABLE:
            raise ImportError("pyflink not installed. Install with: pip install apache-flink")
        
        self.env = None
        self.table_env = None
        self._init_environment()
    
    def _init_environment(self):
        """Initialize Flink environment."""
        try:
            # Create execution environment
            self.env = StreamExecutionEnvironment.get_execution_environment()
            self.table_env = StreamTableEnvironment.create(self.env)
            
            # Configure checkpointing
            self.env.enable_checkpointing(self.config.checkpoint_interval * 1000)
            
            logger.info("Initialized Flink environment")
            
        except Exception as e:
            logger.error(f"Failed to initialize Flink environment: {e}")
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                error_type="initialization"
            ).inc()
            raise
    
    async def process(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process single Flink message with metrics."""
        start_time = time.time()
        
        try:
            result = await self._process_message_impl(message)
            
            self.processed_count += 1
            self.messages_since_last_calc += 1
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                topic=self.config.topic,
                status="success"
            ).inc()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.error_count += 1
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                topic=self.config.topic,
                status="failure"
            ).inc()
            
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                error_type=type(e).__name__
            ).inc()
            
            return None
            
        finally:
            ml_streaming_processing_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                processor_type=self.__class__.__name__
            ).observe(time.time() - start_time)
    
    async def process_batch(self, messages: List[StreamMessage]) -> List[StreamMessage]:
        """Process batch of Flink messages with metrics."""
        start_time = time.time()
        
        ml_streaming_batch_size.labels(
            tenant_id=self.config.tenant_id,
            platform=self.config.platform
        ).observe(len(messages))
        
        try:
            results = await self._process_batch_impl(messages)
            
            self.processed_count += len(messages)
            self.messages_since_last_calc += len(messages)
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                topic=self.config.topic,
                status="success"
            ).inc(len(messages))
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.error_count += len(messages)
            
            ml_streaming_messages_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                topic=self.config.topic,
                status="failure"
            ).inc(len(messages))
            
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                error_type=type(e).__name__
            ).inc()
            
            return []
            
        finally:
            ml_streaming_processing_latency_seconds.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                processor_type=self.__class__.__name__
            ).observe(time.time() - start_time)
    
    async def _process_message_impl(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Actual message processing logic - to be implemented by subclass."""
        return message
    
    async def _process_batch_impl(self, messages: List[StreamMessage]) -> List[StreamMessage]:
        """Actual batch processing logic - to be implemented by subclass."""
        results = []
        for msg in messages:
            result = await self._process_message_impl(msg)
            if result:
                results.append(result)
        return results
    
    def run(self):
        """Run the Flink job."""
        self.running = True
        
        try:
            # Execute Flink job
            self.env.execute(f"Stream Processing - {self.config.topic}")
            
        except Exception as e:
            logger.error(f"Error in Flink job: {e}")
            ml_streaming_errors_total.labels(
                tenant_id=self.config.tenant_id,
                platform=self.config.platform,
                error_type="job_execution"
            ).inc()
            
        finally:
            self.stop()


class RedisStreamHandler(StreamProcessor):
    """Redis Streams handler for lightweight streaming scenarios."""

    def __init__(self, config: StreamConfig):
        super().__init__(config)

        if not REDIS_AVAILABLE:
            raise ImportError("redis package is required for RedisStreamHandler. Install with: pip install redis")

        # Support redis-py classic and asyncio versions via simple constructor
        broker = config.brokers[0] if config.brokers else "localhost:6379"
        host, _, port = broker.partition(":")
        self.client = redis.Redis(host=host or "localhost", port=int(port) if port else 6379, decode_responses=True)
        self.stream_key = config.topic
        self.consumer_group = config.consumer_group
        self.consumer_name = f"consumer-{os.getpid()}"

    async def start_consumer(self, processor: StreamProcessor, output_stream: Optional[str] = None):
        """Start consuming messages from Redis Streams."""

        try:
            self.client.xgroup_create(
                name=self.stream_key,
                groupname=self.consumer_group,
                id="0",
                mkstream=True
            )
        except Exception as exc:  # pragma: no cover - redis raises ResponseError for existing group
            if "BUSYGROUP" not in str(exc):
                raise

        while True:
            messages = self.client.xreadgroup(
                groupname=self.consumer_group,
                consumername=self.consumer_name,
                streams={self.stream_key: ">"},
                count=self.config.batch_size,
                block=1000
            )

            if not messages:
                await asyncio.sleep(0)
                continue

            for _, entries in messages:
                for message_id, payload in entries:
                    stream_message = StreamMessage(
                        key=payload.get("key", message_id),
                        value=json.loads(payload.get("value", "{}")) if payload.get("value") else payload,
                        timestamp=datetime.now(),
                    )

                    await processor.process(stream_message)

                    if output_stream:
                        self.client.xadd(output_stream, {"processed": json.dumps(stream_message.value)})

                    self.client.xack(self.stream_key, self.consumer_group, message_id)

    async def process(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Default Redis processing simply returns the message."""
        return message

    async def process_batch(self, messages: List[StreamMessage]) -> List[StreamMessage]:
        results = []
        for message in messages:
            processed = await self.process(message)
            if processed:
                results.append(processed)
        return results


    def stop(self):
        """Stop the Flink job."""
        self.running = False
        logger.info("Flink stream processor stopped")


# Backwards-compatible aliases expected by tests and legacy code
KafkaStreamHandler = KafkaStreamProcessor
PulsarStreamHandler = PulsarStreamProcessor
FlinkStreamHandler = FlinkStreamProcessor
RedisStreamProcessor = RedisStreamHandler


class WindowedAggregator:
    """Simple windowed aggregator for time-series metrics."""

    def __init__(self, window_size: int = 60, slide_interval: int = 15):
        self.window_size = window_size
        self.slide_interval = slide_interval
        self.windows: Dict[str, Dict[str, Any]] = {}

    def _window_start(self, timestamp: datetime) -> datetime:
        aligned_seconds = (timestamp.second // self.slide_interval) * self.slide_interval
        return timestamp.replace(second=aligned_seconds, microsecond=0)

    def _window_key(self, metric: str, timestamp: datetime) -> str:
        return f"{metric}_{self._window_start(timestamp).isoformat()}"

    def add(self, metric: str, value: float, timestamp: Optional[datetime] = None):
        timestamp = timestamp or datetime.utcnow()
        key = self._window_key(metric, timestamp)
        window = self.windows.setdefault(key, {"start": self._window_start(timestamp), "values": []})
        window["values"].append(float(value))
        return key

    def get_aggregates(self, metric: str, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        timestamp = timestamp or datetime.utcnow()
        key = self._window_key(metric, timestamp)
        window = self.windows.get(key)
        if not window or not window["values"]:
            return {"count": 0, "sum": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}

        values = window["values"]
        total = sum(values)
        return {
            "count": len(values),
            "sum": total,
            "mean": total / len(values),
            "min": min(values),
            "max": max(values),
        }

    def cleanup_old_windows(self, current_time: Optional[datetime] = None):
        current_time = current_time or datetime.utcnow()
        cutoff = current_time - timedelta(seconds=self.window_size)
        to_remove = [
            key
            for key, window in self.windows.items()
            if window["start"] < cutoff
        ]
        for key in to_remove:
            del self.windows[key]


class MLStreamProcessor(KafkaStreamProcessor):
    """Machine Learning stream processor for real-time inference."""
    
    def __init__(self, config: StreamConfig, model_path: str = None):
        super().__init__(config)
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load ML model for inference."""
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    async def _process_message_impl(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process message with ML inference."""
        if not self.model:
            return message
        
        try:
            # Extract features from message
            features = self._extract_features(message.value)
            
            # Make prediction
            prediction = self.model.predict([features])[0]
            
            # Add prediction to message
            result_value = message.value.copy()
            result_value['prediction'] = prediction.tolist() if hasattr(prediction, 'tolist') else prediction
            result_value['prediction_timestamp'] = datetime.now().isoformat()
            result_value['model_version'] = getattr(self.model, 'version', 'unknown')
            
            # Create result message
            result = StreamMessage(
                key=message.key,
                value=result_value,
                timestamp=datetime.now(),
                partition=message.partition,
                offset=message.offset,
                headers=message.headers
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ML processing: {e}")
            return None
    
    def _extract_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract features from message data."""
        # Default implementation - extract numeric values
        features = []
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                features.extend([float(v) for v in value])
        
        return features


class StreamingOrchestrator:
    """Orchestrator for managing multiple streaming pipelines."""
    
    def __init__(self):
        self.processors = {}
        self.threads = {}
        self.lock = threading.Lock()
    
    def add_processor(self, name: str, processor: StreamProcessor):
        """Add a stream processor."""
        with self.lock:
            self.processors[name] = processor
            logger.info(f"Added processor: {name}")
    
    def start_processor(self, name: str):
        """Start a stream processor in a separate thread."""
        with self.lock:
            if name not in self.processors:
                logger.error(f"Processor {name} not found")
                return
            
            if name in self.threads and self.threads[name].is_alive():
                logger.warning(f"Processor {name} is already running")
                return
            
            processor = self.processors[name]
            thread = threading.Thread(target=processor.run, name=f"processor-{name}")
            thread.daemon = True
            thread.start()
            
            self.threads[name] = thread
            logger.info(f"Started processor: {name}")
    
    def stop_processor(self, name: str):
        """Stop a stream processor."""
        with self.lock:
            if name in self.processors:
                self.processors[name].stop()
                
            if name in self.threads:
                thread = self.threads[name]
                thread.join(timeout=10)
                del self.threads[name]
                
            logger.info(f"Stopped processor: {name}")
    
    def stop_all(self):
        """Stop all processors."""
        for name in list(self.processors.keys()):
            self.stop_processor(name)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from all processors."""
        metrics = {}
        
        with self.lock:
            for name, processor in self.processors.items():
                metrics[name] = {
                    "processed": processor.processed_count,
                    "errors": processor.error_count,
                    "throughput": processor.metrics.get("throughput_per_sec", 0),
                    "running": name in self.threads and self.threads[name].is_alive()
                }
        
        return metrics


# Factory function to create appropriate processor based on platform
def create_stream_processor(config: StreamConfig) -> StreamProcessor:
    """
    Create a stream processor based on the platform specified in config.
    
    Args:
        config: StreamConfig with platform specification
        
    Returns:
        Appropriate StreamProcessor instance
    """
    platform = config.platform.lower()
    
    if platform == "kafka":
        return KafkaStreamProcessor(config)
    if platform == "pulsar":
        return PulsarStreamProcessor(config)
    if platform == "flink":
        return FlinkStreamProcessor(config)
    if platform == "redis":
        return RedisStreamHandler(config)

    raise ValueError(f"Unsupported streaming platform: {platform}")


# Export the registry and components so they can be imported by api.py
__all__ = [
    'streaming_registry',
    'StreamConfig',
    'StreamMessage',
    'StreamProcessor',
    'KafkaStreamProcessor',
    'PulsarStreamProcessor',
    'FlinkStreamProcessor',
    'MLStreamProcessor',
    'StreamingOrchestrator',
    'create_stream_processor'
]
