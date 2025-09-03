"""
Streaming module for real-time data processing
Supports Kafka, Flink, Pulsar and other streaming platforms
Place in: automl_platform/api/streaming.py
"""

import json
import logging
import asyncio
import os  # ADDED: Missing import - FIXED
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import deque
import threading
import time
from abc import ABC, abstractmethod

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
    """Base class for stream processing."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.running = False
        self.processed_count = 0
        self.error_count = 0
        self.last_checkpoint = datetime.now()
        self.metrics = {
            "messages_processed": 0,
            "messages_failed": 0,
            "avg_latency_ms": 0,
            "throughput_per_sec": 0
        }
        
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
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        runtime = (datetime.now() - self.last_checkpoint).total_seconds()
        self.metrics["throughput_per_sec"] = self.processed_count / runtime if runtime > 0 else 0
        self.metrics["messages_processed"] = self.processed_count
        self.metrics["messages_failed"] = self.error_count
        return self.metrics


class MLStreamProcessor(StreamProcessor):
    """Stream processor for ML predictions."""
    
    def __init__(self, config: StreamConfig, model=None):
        super().__init__(config)
        self.model = model
        self.feature_buffer = deque(maxlen=1000)
        
    async def process(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process single message for ML prediction."""
        try:
            # Extract features
            features = message.value.get("features", {})
            
            # Buffer for aggregation
            self.feature_buffer.append(features)
            
            # Make prediction if model available
            if self.model:
                df = pd.DataFrame([features])
                prediction = self.model.predict(df)[0]
                
                # Create response message
                response = StreamMessage(
                    key=f"prediction_{message.key}",
                    value={
                        "original": message.value,
                        "prediction": float(prediction),
                        "timestamp": datetime.now().isoformat(),
                        "model_id": getattr(self.model, "model_id", "unknown")
                    },
                    timestamp=datetime.now()
                )
                
                self.processed_count += 1
                return response
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.error_count += 1
            return None
    
    async def process_batch(self, messages: List[StreamMessage]) -> List[StreamMessage]:
        """Process batch of messages for ML prediction."""
        if not self.model or not messages:
            return []
        
        try:
            # Extract features from all messages
            features_list = [msg.value.get("features", {}) for msg in messages]
            df = pd.DataFrame(features_list)
            
            # Make batch predictions
            predictions = self.model.predict(df)
            
            # Create response messages
            responses = []
            for msg, pred in zip(messages, predictions):
                response = StreamMessage(
                    key=f"prediction_{msg.key}",
                    value={
                        "original": msg.value,
                        "prediction": float(pred),
                        "timestamp": datetime.now().isoformat(),
                        "model_id": getattr(self.model, "model_id", "unknown")
                    },
                    timestamp=datetime.now()
                )
                responses.append(response)
            
            self.processed_count += len(responses)
            return responses
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.error_count += len(messages)
            return []


class KafkaStreamHandler:
    """Kafka streaming handler."""
    
    def __init__(self, config: StreamConfig):
        if not KAFKA_AVAILABLE:
            raise ImportError("Kafka not installed. Install with: pip install kafka-python")
        
        self.config = config
        self.running = False
        
        # Configure Kafka
        self.kafka_config = {
            'bootstrap_servers': config.brokers,
            'security_protocol': config.security_protocol
        }
        
        if config.sasl_mechanism:
            self.kafka_config.update({
                'sasl_mechanism': config.sasl_mechanism,
                'sasl_plain_username': config.sasl_username,
                'sasl_plain_password': config.sasl_password
            })
        
        self.consumer = None
        self.producer = None
        
    def create_topic(self, partitions: int = 3, replication_factor: int = 1):
        """Create Kafka topic if it doesn't exist."""
        try:
            admin_client = KafkaAdminClient(**self.kafka_config)
            
            topic = NewTopic(
                name=self.config.topic,
                num_partitions=partitions,
                replication_factor=replication_factor
            )
            
            admin_client.create_topics([topic], validate_only=False)
            logger.info(f"Created Kafka topic: {self.config.topic}")
            
        except Exception as e:
            logger.debug(f"Topic might already exist: {e}")
    
    async def start_consumer(self, processor: StreamProcessor, output_topic: str = None):
        """Start Kafka consumer with processor."""
        self.consumer = KafkaConsumer(
            self.config.topic,
            group_id=self.config.consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda m: m.decode('utf-8') if m else None,
            auto_offset_reset='latest',
            enable_auto_commit=not self.config.enable_exactly_once,
            max_poll_records=self.config.batch_size,
            **self.kafka_config
        )
        
        # Initialize producer if output topic specified
        if output_topic:
            self.producer = KafkaProducer(
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                **self.kafka_config
            )
        
        self.running = True
        batch = []
        
        try:
            while self.running:
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, records in messages.items():
                    for record in records:
                        # Convert to StreamMessage
                        msg = StreamMessage(
                            key=record.key,
                            value=record.value,
                            timestamp=datetime.fromtimestamp(record.timestamp / 1000),
                            partition=record.partition,
                            offset=record.offset,
                            headers=dict(record.headers) if record.headers else {}
                        )
                        batch.append(msg)
                        
                        # Process batch when full
                        if len(batch) >= self.config.batch_size:
                            results = await processor.process_batch(batch)
                            
                            # Send results if producer available
                            if self.producer and output_topic:
                                for result in results:
                                    self.producer.send(
                                        output_topic,
                                        key=result.key,
                                        value=result.value
                                    )
                            
                            batch = []
                            
                            # Commit offsets if exactly-once
                            if self.config.enable_exactly_once:
                                self.consumer.commit()
                
                # Process remaining messages
                if batch:
                    results = await processor.process_batch(batch)
                    if self.producer and output_topic:
                        for result in results:
                            self.producer.send(output_topic, key=result.key, value=result.value)
                    batch = []
                
                # Checkpoint periodically
                if (datetime.now() - processor.last_checkpoint).seconds > self.config.checkpoint_interval:
                    processor.checkpoint()
                    
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop Kafka consumer and producer."""
        self.running = False
        
        if self.consumer:
            self.consumer.close()
        
        if self.producer:
            self.producer.flush()
            self.producer.close()


class FlinkStreamHandler:
    """Apache Flink streaming handler."""
    
    def __init__(self, config: StreamConfig):
        if not FLINK_AVAILABLE:
            raise ImportError("PyFlink not installed. Install with: pip install apache-flink")
        
        self.config = config
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.table_env = StreamTableEnvironment.create(self.env)
        
        # Configure checkpointing
        self.env.enable_checkpointing(config.checkpoint_interval * 1000)
        
    def create_pipeline(self, processor: StreamProcessor):
        """Create Flink streaming pipeline."""
        
        # Define source table (Kafka)
        source_ddl = f"""
        CREATE TABLE source_table (
            `key` STRING,
            `value` STRING,
            `timestamp` TIMESTAMP(3),
            WATERMARK FOR `timestamp` AS `timestamp` - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = '{self.config.topic}',
            'properties.bootstrap.servers' = '{",".join(self.config.brokers)}',
            'properties.group.id' = '{self.config.consumer_group}',
            'scan.startup.mode' = 'latest-offset',
            'format' = 'json'
        )
        """
        
        self.table_env.execute_sql(source_ddl)
        
        # Define sink table
        sink_ddl = f"""
        CREATE TABLE sink_table (
            `key` STRING,
            `prediction` DOUBLE,
            `timestamp` TIMESTAMP(3)
        ) WITH (
            'connector' = 'kafka',
            'topic' = '{self.config.topic}_predictions',
            'properties.bootstrap.servers' = '{",".join(self.config.brokers)}',
            'format' = 'json'
        )
        """
        
        self.table_env.execute_sql(sink_ddl)
        
        # Create processing logic
        self.table_env.from_path("source_table") \
            .select("key, value, timestamp") \
            .insert_into("sink_table")
        
    def start(self):
        """Start Flink job."""
        try:
            self.env.execute("AutoML Streaming Pipeline")
        except Exception as e:
            logger.error(f"Flink execution failed: {e}")


class PulsarStreamHandler:
    """Apache Pulsar streaming handler."""
    
    def __init__(self, config: StreamConfig):
        if not PULSAR_AVAILABLE:
            raise ImportError("Pulsar not installed. Install with: pip install pulsar-client")
        
        self.config = config
        self.client = pulsar.Client(f"pulsar://{config.brokers[0]}")
        self.consumer = None
        self.producer = None
        
    async def start_consumer(self, processor: StreamProcessor, output_topic: str = None):
        """Start Pulsar consumer."""
        self.consumer = self.client.subscribe(
            self.config.topic,
            subscription_name=self.config.consumer_group,
            consumer_type=pulsar.ConsumerType.Shared
        )
        
        if output_topic:
            self.producer = self.client.create_producer(output_topic)
        
        batch = []
        
        try:
            while True:
                msg = self.consumer.receive(timeout_millis=1000)
                
                try:
                    # Parse message
                    data = json.loads(msg.data().decode('utf-8'))
                    
                    stream_msg = StreamMessage(
                        key=msg.partition_key(),
                        value=data,
                        timestamp=datetime.fromtimestamp(msg.publish_timestamp() / 1000)
                    )
                    
                    batch.append(stream_msg)
                    
                    # Process batch
                    if len(batch) >= self.config.batch_size:
                        results = await processor.process_batch(batch)
                        
                        # Send results
                        if self.producer:
                            for result in results:
                                self.producer.send(
                                    json.dumps(result.value).encode('utf-8'),
                                    partition_key=result.key
                                )
                        
                        batch = []
                    
                    # Acknowledge message
                    self.consumer.acknowledge(msg)
                    
                except Exception as e:
                    logger.error(f"Error processing Pulsar message: {e}")
                    self.consumer.negative_acknowledge(msg)
                    
        except Exception as e:
            logger.error(f"Pulsar consumer error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop Pulsar client."""
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()
        self.client.close()


class RedisStreamHandler:
    """Redis Streams handler."""
    
    def __init__(self, config: StreamConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not installed. Install with: pip install redis")
        
        self.config = config
        self.redis_client = redis.Redis(
            host=config.brokers[0].split(':')[0],
            port=int(config.brokers[0].split(':')[1]) if ':' in config.brokers[0] else 6379,
            decode_responses=True
        )
        
    async def start_consumer(self, processor: StreamProcessor, output_stream: str = None):
        """Start Redis Streams consumer."""
        consumer_group = self.config.consumer_group
        consumer_name = f"{consumer_group}-{os.getpid()}"  # FIXED: os is now imported
        
        # Create consumer group
        try:
            self.redis_client.xgroup_create(self.config.topic, consumer_group, id='0')
        except redis.ResponseError:
            pass  # Group already exists
        
        batch = []
        
        try:
            while True:
                # Read messages
                messages = self.redis_client.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {self.config.topic: '>'},
                    count=self.config.batch_size,
                    block=1000
                )
                
                for stream_name, stream_messages in messages:
                    for msg_id, data in stream_messages:
                        stream_msg = StreamMessage(
                            key=msg_id,
                            value=data,
                            timestamp=datetime.now()
                        )
                        batch.append(stream_msg)
                
                # Process batch
                if batch:
                    results = await processor.process_batch(batch)
                    
                    # Send results to output stream
                    if output_stream:
                        for result in results:
                            self.redis_client.xadd(
                                output_stream,
                                result.value
                            )
                    
                    # Acknowledge messages
                    for msg in batch:
                        self.redis_client.xack(self.config.topic, consumer_group, msg.key)
                    
                    batch = []
                    
        except Exception as e:
            logger.error(f"Redis Streams error: {e}")


class StreamingOrchestrator:
    """Orchestrates streaming pipelines."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.handler = None
        self.processor = None
        
        # Select appropriate handler
        if config.platform == "kafka":
            self.handler = KafkaStreamHandler(config)
        elif config.platform == "flink":
            self.handler = FlinkStreamHandler(config)
        elif config.platform == "pulsar":
            self.handler = PulsarStreamHandler(config)
        elif config.platform == "redis":
            self.handler = RedisStreamHandler(config)
        else:
            raise ValueError(f"Unsupported streaming platform: {config.platform}")
    
    def set_processor(self, processor: StreamProcessor):
        """Set the stream processor."""
        self.processor = processor
    
    async def start(self, output_topic: str = None):
        """Start streaming pipeline."""
        if not self.processor:
            raise ValueError("No processor set")
        
        logger.info(f"Starting {self.config.platform} streaming pipeline")
        
        if self.config.platform == "kafka":
            await self.handler.start_consumer(self.processor, output_topic)
        elif self.config.platform == "flink":
            self.handler.create_pipeline(self.processor)
            self.handler.start()
        elif self.config.platform == "pulsar":
            await self.handler.start_consumer(self.processor, output_topic)
        elif self.config.platform == "redis":
            await self.handler.start_consumer(self.processor, output_topic)
    
    def stop(self):
        """Stop streaming pipeline."""
        if self.handler:
            self.handler.stop()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        metrics = {
            "platform": self.config.platform,
            "topic": self.config.topic,
            "status": "running" if self.handler else "stopped"
        }
        
        if self.processor:
            metrics.update(self.processor.get_metrics())
        
        return metrics


class WindowedAggregator:
    """Aggregates streaming data over time windows."""
    
    def __init__(self, window_size: int = 60, slide_interval: int = 10):
        """
        Initialize windowed aggregator.
        
        Args:
            window_size: Window size in seconds
            slide_interval: Slide interval in seconds
        """
        self.window_size = window_size
        self.slide_interval = slide_interval
        self.windows = {}
        
    def add(self, key: str, value: Any, timestamp: datetime):
        """Add value to window."""
        window_start = timestamp.replace(second=0, microsecond=0)
        window_key = f"{key}_{window_start.isoformat()}"
        
        if window_key not in self.windows:
            self.windows[window_key] = {
                "values": [],
                "start": window_start,
                "end": window_start + timedelta(seconds=self.window_size)
            }
        
        self.windows[window_key]["values"].append(value)
    
    def get_aggregates(self, key: str, timestamp: datetime) -> Dict[str, Any]:
        """Get aggregates for a window."""
        window_start = timestamp.replace(second=0, microsecond=0)
        window_key = f"{key}_{window_start.isoformat()}"
        
        if window_key not in self.windows:
            return {}
        
        values = self.windows[window_key]["values"]
        
        if not values:
            return {}
        
        # Calculate aggregates
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        
        aggregates = {
            "count": len(values),
            "window_start": self.windows[window_key]["start"].isoformat(),
            "window_end": self.windows[window_key]["end"].isoformat()
        }
        
        if numeric_values:
            aggregates.update({
                "sum": sum(numeric_values),
                "mean": np.mean(numeric_values),
                "std": np.std(numeric_values),
                "min": min(numeric_values),
                "max": max(numeric_values),
                "median": np.median(numeric_values)
            })
        
        return aggregates
    
    def cleanup_old_windows(self, current_time: datetime):
        """Remove old windows."""
        cutoff = current_time - timedelta(seconds=self.window_size * 2)
        
        old_keys = [
            key for key, window in self.windows.items()
            if window["end"] < cutoff
        ]
        
        for key in old_keys:
            del self.windows[key]


# Example usage
async def main():
    """Example streaming pipeline."""
    
    # Configure streaming
    config = StreamConfig(
        platform="kafka",
        brokers=["localhost:9092"],
        topic="ml_predictions",
        batch_size=10,
        window_size=30
    )
    
    # Create ML processor
    processor = MLStreamProcessor(config)
    
    # Create orchestrator
    orchestrator = StreamingOrchestrator(config)
    orchestrator.set_processor(processor)
    
    # Start streaming
    await orchestrator.start(output_topic="ml_results")
    
    # Monitor metrics
    while True:
        await asyncio.sleep(10)
        metrics = orchestrator.get_metrics()
        logger.info(f"Streaming metrics: {metrics}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
