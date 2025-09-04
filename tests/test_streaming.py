"""
Test Suite for Streaming ML Module
===================================
Tests for Kafka, Flink, Pulsar, Redis Streams integration and ML stream processing.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
import numpy as np
import pandas as pd
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl_platform.api.streaming import (
    StreamConfig,
    StreamMessage,
    StreamProcessor,
    MLStreamProcessor,
    KafkaStreamHandler,
    FlinkStreamHandler,
    PulsarStreamHandler,
    RedisStreamHandler,
    StreamingOrchestrator,
    WindowedAggregator
)


class TestStreamConfig(unittest.TestCase):
    """Test streaming configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="test_topic"
        )
        
        self.assertEqual(config.platform, "kafka")
        self.assertEqual(config.brokers, ["localhost:9092"])
        self.assertEqual(config.topic, "test_topic")
        self.assertEqual(config.consumer_group, "automl-consumer")
        self.assertEqual(config.batch_size, 100)
        self.assertEqual(config.window_size, 60)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamConfig(
            platform="pulsar",
            brokers=["pulsar://localhost:6650"],
            topic="ml_stream",
            batch_size=500,
            window_size=120,
            enable_exactly_once=True
        )
        
        self.assertEqual(config.platform, "pulsar")
        self.assertEqual(config.batch_size, 500)
        self.assertEqual(config.window_size, 120)
        self.assertTrue(config.enable_exactly_once)
    
    def test_to_dict(self):
        """Test configuration serialization."""
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="test",
            schema={"feature1": "float", "feature2": "int"}
        )
        
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn("schema", config_dict)


class TestStreamMessage(unittest.TestCase):
    """Test stream message format."""
    
    def test_message_creation(self):
        """Test creating a stream message."""
        msg = StreamMessage(
            key="msg_001",
            value={"feature1": 1.0, "feature2": 2.0},
            timestamp=datetime.now(),
            partition=0,
            offset=100
        )
        
        self.assertEqual(msg.key, "msg_001")
        self.assertIn("feature1", msg.value)
        self.assertEqual(msg.partition, 0)
        self.assertEqual(msg.offset, 100)
    
    def test_message_with_headers(self):
        """Test message with headers."""
        headers = {"content-type": "application/json", "model-version": "v1"}
        msg = StreamMessage(
            key="msg_002",
            value={"data": "test"},
            timestamp=datetime.now(),
            headers=headers
        )
        
        self.assertEqual(msg.headers["content-type"], "application/json")
        self.assertEqual(msg.headers["model-version"], "v1")


class TestMLStreamProcessor(unittest.TestCase):
    """Test ML stream processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="ml_data"
        )
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.predict = Mock(return_value=np.array([0.8]))
        self.mock_model.model_id = "test_model_001"
        
        self.processor = MLStreamProcessor(self.config, model=self.mock_model)
    
    def test_initialization(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.config, self.config)
        self.assertEqual(self.processor.model, self.mock_model)
        self.assertEqual(self.processor.processed_count, 0)
        self.assertEqual(self.processor.error_count, 0)
    
    async def test_process_single_message(self):
        """Test processing single message."""
        msg = StreamMessage(
            key="test_001",
            value={"features": {"f1": 1.0, "f2": 2.0}},
            timestamp=datetime.now()
        )
        
        result = await self.processor.process(msg)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.key, "prediction_test_001")
        self.assertIn("prediction", result.value)
        self.assertEqual(result.value["prediction"], 0.8)
        self.assertEqual(self.processor.processed_count, 1)
    
    async def test_process_batch(self):
        """Test batch processing."""
        messages = [
            StreamMessage(
                key=f"test_{i}",
                value={"features": {"f1": float(i), "f2": float(i*2)}},
                timestamp=datetime.now()
            )
            for i in range(5)
        ]
        
        # Mock batch prediction
        self.mock_model.predict.return_value = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        
        results = await self.processor.process_batch(messages)
        
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0].value["prediction"], 0.5)
        self.assertEqual(results[4].value["prediction"], 0.9)
        self.assertEqual(self.processor.processed_count, 5)
    
    async def test_process_error_handling(self):
        """Test error handling during processing."""
        # Make model raise exception
        self.mock_model.predict.side_effect = Exception("Model error")
        
        msg = StreamMessage(
            key="error_test",
            value={"features": {}},
            timestamp=datetime.now()
        )
        
        result = await self.processor.process(msg)
        
        self.assertIsNone(result)
        self.assertEqual(self.processor.error_count, 1)
    
    def test_get_metrics(self):
        """Test metrics retrieval."""
        # Simulate some processing
        self.processor.processed_count = 100
        self.processor.error_count = 5
        
        metrics = self.processor.get_metrics()
        
        self.assertEqual(metrics["messages_processed"], 100)
        self.assertEqual(metrics["messages_failed"], 5)
        self.assertIn("throughput_per_sec", metrics)


class TestKafkaStreamHandler(unittest.TestCase):
    """Test Kafka stream handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="test_topic",
            consumer_group="test_group"
        )
    
    @patch('automl_platform.api.streaming.KAFKA_AVAILABLE', True)
    @patch('automl_platform.api.streaming.KafkaConsumer')
    @patch('automl_platform.api.streaming.KafkaProducer')
    def test_initialization(self, mock_producer, mock_consumer):
        """Test Kafka handler initialization."""
        handler = KafkaStreamHandler(self.config)
        
        self.assertEqual(handler.config, self.config)
        self.assertFalse(handler.running)
        self.assertIn('bootstrap_servers', handler.kafka_config)
    
    @patch('automl_platform.api.streaming.KAFKA_AVAILABLE', False)
    def test_kafka_not_available(self):
        """Test error when Kafka is not installed."""
        with self.assertRaises(ImportError):
            KafkaStreamHandler(self.config)
    
    @patch('automl_platform.api.streaming.KAFKA_AVAILABLE', True)
    @patch('automl_platform.api.streaming.KafkaAdminClient')
    def test_create_topic(self, mock_admin):
        """Test Kafka topic creation."""
        handler = KafkaStreamHandler(self.config)
        
        mock_admin_instance = Mock()
        mock_admin.return_value = mock_admin_instance
        
        handler.create_topic(partitions=3, replication_factor=1)
        
        mock_admin.assert_called_once()
        mock_admin_instance.create_topics.assert_called_once()
    
    @patch('automl_platform.api.streaming.KAFKA_AVAILABLE', True)
    @patch('automl_platform.api.streaming.KafkaConsumer')
    @patch('automl_platform.api.streaming.KafkaProducer')
    async def test_start_consumer(self, mock_producer_class, mock_consumer_class):
        """Test starting Kafka consumer."""
        handler = KafkaStreamHandler(self.config)
        
        # Create mock consumer
        mock_consumer = Mock()
        mock_consumer_class.return_value = mock_consumer
        
        # Mock poll to return empty then stop
        handler.running = False
        mock_consumer.poll.return_value = {}
        
        # Create mock processor
        processor = AsyncMock(spec=StreamProcessor)
        processor.last_checkpoint = datetime.now() - timedelta(seconds=100)
        
        # Start consumer (will exit immediately due to running=False)
        await handler.start_consumer(processor, output_topic="output")
        
        # Verify consumer was created
        mock_consumer_class.assert_called_once()
    
    @patch('automl_platform.api.streaming.KAFKA_AVAILABLE', True)
    def test_stop(self):
        """Test stopping Kafka handler."""
        handler = KafkaStreamHandler(self.config)
        handler.running = True
        
        # Mock consumer and producer
        handler.consumer = Mock()
        handler.producer = Mock()
        
        handler.stop()
        
        self.assertFalse(handler.running)
        handler.consumer.close.assert_called_once()
        handler.producer.flush.assert_called_once()
        handler.producer.close.assert_called_once()


class TestFlinkStreamHandler(unittest.TestCase):
    """Test Flink stream handler."""
    
    @patch('automl_platform.api.streaming.FLINK_AVAILABLE', True)
    @patch('automl_platform.api.streaming.StreamExecutionEnvironment')
    @patch('automl_platform.api.streaming.StreamTableEnvironment')
    def test_initialization(self, mock_table_env, mock_stream_env):
        """Test Flink handler initialization."""
        config = StreamConfig(
            platform="flink",
            brokers=["localhost:9092"],
            topic="test_topic"
        )
        
        handler = FlinkStreamHandler(config)
        
        self.assertEqual(handler.config, config)
        mock_stream_env.get_execution_environment.assert_called_once()
    
    @patch('automl_platform.api.streaming.FLINK_AVAILABLE', False)
    def test_flink_not_available(self):
        """Test error when Flink is not installed."""
        config = StreamConfig(platform="flink", brokers=["localhost"], topic="test")
        
        with self.assertRaises(ImportError):
            FlinkStreamHandler(config)
    
    @patch('automl_platform.api.streaming.FLINK_AVAILABLE', True)
    @patch('automl_platform.api.streaming.StreamExecutionEnvironment')
    @patch('automl_platform.api.streaming.StreamTableEnvironment')
    def test_create_pipeline(self, mock_table_env, mock_stream_env):
        """Test Flink pipeline creation."""
        config = StreamConfig(
            platform="flink",
            brokers=["localhost:9092"],
            topic="test_topic"
        )
        
        handler = FlinkStreamHandler(config)
        
        # Mock table environment
        mock_table_instance = Mock()
        handler.table_env = mock_table_instance
        
        # Create mock processor
        processor = Mock(spec=StreamProcessor)
        
        # Create pipeline
        handler.create_pipeline(processor)
        
        # Check that SQL DDL was executed
        self.assertEqual(mock_table_instance.execute_sql.call_count, 2)  # source and sink tables


class TestPulsarStreamHandler(unittest.TestCase):
    """Test Pulsar stream handler."""
    
    @patch('automl_platform.api.streaming.PULSAR_AVAILABLE', True)
    @patch('automl_platform.api.streaming.pulsar.Client')
    def test_initialization(self, mock_client):
        """Test Pulsar handler initialization."""
        config = StreamConfig(
            platform="pulsar",
            brokers=["pulsar://localhost:6650"],
            topic="test_topic"
        )
        
        handler = PulsarStreamHandler(config)
        
        self.assertEqual(handler.config, config)
        mock_client.assert_called_once_with("pulsar://pulsar://localhost:6650")
    
    @patch('automl_platform.api.streaming.PULSAR_AVAILABLE', False)
    def test_pulsar_not_available(self):
        """Test error when Pulsar is not installed."""
        config = StreamConfig(platform="pulsar", brokers=["localhost"], topic="test")
        
        with self.assertRaises(ImportError):
            PulsarStreamHandler(config)
    
    @patch('automl_platform.api.streaming.PULSAR_AVAILABLE', True)
    @patch('automl_platform.api.streaming.pulsar.Client')
    async def test_start_consumer(self, mock_client_class):
        """Test starting Pulsar consumer."""
        config = StreamConfig(
            platform="pulsar",
            brokers=["pulsar://localhost:6650"],
            topic="test_topic"
        )
        
        # Create mock client and consumer
        mock_client = Mock()
        mock_consumer = Mock()
        mock_producer = Mock()
        
        mock_client.subscribe.return_value = mock_consumer
        mock_client.create_producer.return_value = mock_producer
        mock_client_class.return_value = mock_client
        
        handler = PulsarStreamHandler(config)
        
        # Mock processor
        processor = AsyncMock(spec=StreamProcessor)
        
        # Mock receive to raise timeout immediately to exit loop
        mock_consumer.receive.side_effect = Exception("Timeout")
        
        # Start consumer (will exit on exception)
        try:
            await handler.start_consumer(processor, output_topic="output")
        except:
            pass
        
        # Verify consumer was created
        mock_client.subscribe.assert_called_once()


class TestRedisStreamHandler(unittest.TestCase):
    """Test Redis Streams handler."""
    
    @patch('automl_platform.api.streaming.REDIS_AVAILABLE', True)
    @patch('automl_platform.api.streaming.redis.Redis')
    def test_initialization(self, mock_redis):
        """Test Redis handler initialization."""
        config = StreamConfig(
            platform="redis",
            brokers=["localhost:6379"],
            topic="test_stream"
        )
        
        handler = RedisStreamHandler(config)
        
        self.assertEqual(handler.config, config)
        mock_redis.assert_called_once()
    
    @patch('automl_platform.api.streaming.REDIS_AVAILABLE', False)
    def test_redis_not_available(self):
        """Test error when Redis is not installed."""
        config = StreamConfig(platform="redis", brokers=["localhost"], topic="test")
        
        with self.assertRaises(ImportError):
            RedisStreamHandler(config)
    
    @patch('automl_platform.api.streaming.REDIS_AVAILABLE', True)
    @patch('automl_platform.api.streaming.redis.Redis')
    @patch('automl_platform.api.streaming.os.getpid')
    async def test_start_consumer(self, mock_getpid, mock_redis_class):
        """Test starting Redis Streams consumer."""
        config = StreamConfig(
            platform="redis",
            brokers=["localhost:6379"],
            topic="test_stream"
        )
        
        # Mock process ID
        mock_getpid.return_value = 12345
        
        # Create mock Redis client
        mock_redis = Mock()
        mock_redis_class.return_value = mock_redis
        
        handler = RedisStreamHandler(config)
        
        # Mock xreadgroup to return empty list then raise exception to exit
        mock_redis.xreadgroup.side_effect = [[], Exception("Exit")]
        
        # Mock processor
        processor = AsyncMock(spec=StreamProcessor)
        
        # Start consumer (will exit on exception)
        try:
            await handler.start_consumer(processor, output_stream="output")
        except:
            pass
        
        # Verify consumer group was created
        mock_redis.xgroup_create.assert_called_once()


class TestStreamingOrchestrator(unittest.TestCase):
    """Test streaming orchestrator."""
    
    def test_initialization_kafka(self):
        """Test orchestrator initialization with Kafka."""
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="test"
        )
        
        with patch('automl_platform.api.streaming.KAFKA_AVAILABLE', True):
            orchestrator = StreamingOrchestrator(config)
            
            self.assertEqual(orchestrator.config, config)
            self.assertIsNotNone(orchestrator.handler)
    
    def test_initialization_unsupported_platform(self):
        """Test error with unsupported platform."""
        config = StreamConfig(
            platform="unsupported",
            brokers=["localhost"],
            topic="test"
        )
        
        with self.assertRaises(ValueError):
            StreamingOrchestrator(config)
    
    def test_set_processor(self):
        """Test setting stream processor."""
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="test"
        )
        
        with patch('automl_platform.api.streaming.KAFKA_AVAILABLE', True):
            orchestrator = StreamingOrchestrator(config)
            
            processor = Mock(spec=StreamProcessor)
            orchestrator.set_processor(processor)
            
            self.assertEqual(orchestrator.processor, processor)
    
    @patch('automl_platform.api.streaming.KafkaStreamHandler')
    async def test_start_without_processor(self, mock_handler_class):
        """Test error when starting without processor."""
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="test"
        )
        
        with patch('automl_platform.api.streaming.KAFKA_AVAILABLE', True):
            orchestrator = StreamingOrchestrator(config)
            
            with self.assertRaises(ValueError):
                await orchestrator.start()
    
    def test_get_metrics(self):
        """Test getting orchestrator metrics."""
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="test"
        )
        
        with patch('automl_platform.api.streaming.KAFKA_AVAILABLE', True):
            orchestrator = StreamingOrchestrator(config)
            
            # Set mock processor with metrics
            processor = Mock(spec=StreamProcessor)
            processor.get_metrics.return_value = {
                "messages_processed": 1000,
                "throughput_per_sec": 100
            }
            orchestrator.set_processor(processor)
            
            metrics = orchestrator.get_metrics()
            
            self.assertEqual(metrics["platform"], "kafka")
            self.assertEqual(metrics["topic"], "test")
            self.assertEqual(metrics["messages_processed"], 1000)


class TestWindowedAggregator(unittest.TestCase):
    """Test windowed aggregation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = WindowedAggregator(window_size=60, slide_interval=10)
    
    def test_initialization(self):
        """Test aggregator initialization."""
        self.assertEqual(self.aggregator.window_size, 60)
        self.assertEqual(self.aggregator.slide_interval, 10)
        self.assertEqual(len(self.aggregator.windows), 0)
    
    def test_add_values(self):
        """Test adding values to window."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        self.aggregator.add("sensor1", 10.5, timestamp)
        self.aggregator.add("sensor1", 11.0, timestamp)
        self.aggregator.add("sensor1", 10.8, timestamp)
        
        # Check window was created
        window_key = f"sensor1_{timestamp.replace(second=0, microsecond=0).isoformat()}"
        self.assertIn(window_key, self.aggregator.windows)
        self.assertEqual(len(self.aggregator.windows[window_key]["values"]), 3)
    
    def test_get_aggregates(self):
        """Test getting window aggregates."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        # Add numeric values
        values = [10.0, 12.0, 11.0, 13.0, 9.0]
        for val in values:
            self.aggregator.add("metric1", val, timestamp)
        
        # Get aggregates
        aggregates = self.aggregator.get_aggregates("metric1", timestamp)
        
        self.assertEqual(aggregates["count"], 5)
        self.assertEqual(aggregates["sum"], 55.0)
        self.assertEqual(aggregates["mean"], 11.0)
        self.assertEqual(aggregates["min"], 9.0)
        self.assertEqual(aggregates["max"], 13.0)
    
    def test_cleanup_old_windows(self):
        """Test cleaning up old windows."""
        current_time = datetime(2024, 1, 1, 12, 0, 0)
        old_time = current_time - timedelta(seconds=200)
        
        # Add old window
        self.aggregator.add("old_metric", 1.0, old_time)
        
        # Add current window
        self.aggregator.add("current_metric", 2.0, current_time)
        
        # Cleanup old windows
        self.aggregator.cleanup_old_windows(current_time)
        
        # Old window should be removed
        old_key = f"old_metric_{old_time.replace(second=0, microsecond=0).isoformat()}"
        current_key = f"current_metric_{current_time.replace(second=0, microsecond=0).isoformat()}"
        
        self.assertNotIn(old_key, self.aggregator.windows)
        self.assertIn(current_key, self.aggregator.windows)
    
    def test_non_numeric_values(self):
        """Test handling non-numeric values."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        
        # Add mixed values
        self.aggregator.add("mixed", "string_value", timestamp)
        self.aggregator.add("mixed", 10.0, timestamp)
        self.aggregator.add("mixed", None, timestamp)
        
        # Get aggregates
        aggregates = self.aggregator.get_aggregates("mixed", timestamp)
        
        # Should only aggregate numeric values
        self.assertEqual(aggregates["count"], 3)  # Total count
        self.assertEqual(aggregates["sum"], 10.0)  # Only numeric sum


class TestIntegration(unittest.TestCase):
    """Integration tests for streaming components."""
    
    @patch('automl_platform.api.streaming.KAFKA_AVAILABLE', True)
    @patch('automl_platform.api.streaming.KafkaConsumer')
    @patch('automl_platform.api.streaming.KafkaProducer')
    async def test_end_to_end_kafka_streaming(self, mock_producer_class, mock_consumer_class):
        """Test end-to-end Kafka streaming with ML processor."""
        # Configure
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="input_topic",
            batch_size=10
        )
        
        # Create mock model
        model = Mock()
        model.predict = Mock(return_value=np.array([0.9] * 10))
        
        # Create processor
        processor = MLStreamProcessor(config, model=model)
        
        # Create orchestrator
        orchestrator = StreamingOrchestrator(config)
        orchestrator.set_processor(processor)
        
        # Get metrics
        metrics = orchestrator.get_metrics()
        
        self.assertEqual(metrics["platform"], "kafka")
        self.assertEqual(metrics["status"], "running")
    
    def test_multiple_platform_support(self):
        """Test that multiple platforms can be configured."""
        platforms = ["kafka", "pulsar", "redis"]
        
        for platform in platforms:
            config = StreamConfig(
                platform=platform,
                brokers=["localhost"],
                topic="test"
            )
            
            # Mock availability
            with patch(f'automl_platform.api.streaming.{platform.upper()}_AVAILABLE', True):
                if platform == "kafka":
                    with patch('automl_platform.api.streaming.KafkaStreamHandler'):
                        orchestrator = StreamingOrchestrator(config)
                elif platform == "pulsar":
                    with patch('automl_platform.api.streaming.pulsar.Client'):
                        orchestrator = StreamingOrchestrator(config)
                elif platform == "redis":
                    with patch('automl_platform.api.streaming.redis.Redis'):
                        orchestrator = StreamingOrchestrator(config)
                
                self.assertIsNotNone(orchestrator.handler)


if __name__ == "__main__":
    unittest.main()
