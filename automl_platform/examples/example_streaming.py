"""
Streaming and Real-time Processing Example
==========================================
Place in: automl_platform/examples/streaming_realtime_example.py

Demonstrates real-time data streaming with Kafka, model inference,
and performance monitoring with Prometheus metrics.
"""

import json
import asyncio
import logging
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import threading
from prometheus_client import start_http_server, generate_latest

# Import streaming components
from automl_platform.api.streaming import (
    StreamConfig,
    StreamMessage,
    KafkaStreamProcessor,
    PulsarStreamProcessor,
    MLStreamProcessor,
    StreamingOrchestrator,
    create_stream_processor,
    streaming_registry
)

# Import model components
from automl_platform.config import AutoMLConfig
from automl_platform.orchestrator import AutoMLOrchestrator
from automl_platform.mlflow_registry import MLflowRegistry
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingExample:
    """Real-time streaming and inference examples"""
    
    def __init__(self):
        self.config = AutoMLConfig()
        self.registry = MLflowRegistry(self.config)
        self.orchestrator = StreamingOrchestrator()
        self.model = None
        self.model_path = "models/streaming_model.pkl"
    
    def example_1_kafka_basic_streaming(self):
        """Example 1: Basic Kafka streaming setup"""
        print("\n" + "="*80)
        print("EXAMPLE 1: Basic Kafka Streaming")
        print("="*80)
        
        # Configure Kafka stream
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="sensor-data",
            consumer_group="automl-consumer",
            batch_size=100,
            window_size=60,
            checkpoint_interval=30,
            tenant_id="example-tenant"
        )
        
        print(f"\n📡 Kafka Configuration:")
        print(f"  - Brokers: {config.brokers}")
        print(f"  - Topic: {config.topic}")
        print(f"  - Consumer Group: {config.consumer_group}")
        print(f"  - Batch Size: {config.batch_size}")
        
        # Create custom processor
        class SensorDataProcessor(KafkaStreamProcessor):
            """Process sensor data stream"""
            
            async def _process_message_impl(self, message: StreamMessage) -> Optional[StreamMessage]:
                """Process individual sensor reading"""
                
                # Extract sensor data
                sensor_data = message.value
                
                # Data validation
                if not self._validate_sensor_data(sensor_data):
                    logger.warning(f"Invalid sensor data: {sensor_data}")
                    return None
                
                # Feature extraction
                features = self._extract_features(sensor_data)
                
                # Anomaly detection
                is_anomaly = self._detect_anomaly(features)
                
                # Create enriched message
                enriched_data = {
                    **sensor_data,
                    'features': features,
                    'is_anomaly': is_anomaly,
                    'processed_at': datetime.now().isoformat()
                }
                
                return StreamMessage(
                    key=message.key,
                    value=enriched_data,
                    timestamp=datetime.now(),
                    partition=message.partition,
                    offset=message.offset
                )
            
            async def _process_batch_impl(self, messages: List[StreamMessage]) -> List[StreamMessage]:
                """Process batch of sensor readings"""
                
                # Collect all sensor values
                sensor_values = [msg.value for msg in messages]
                df = pd.DataFrame(sensor_values)
                
                # Batch statistics
                batch_stats = {
                    'mean_temperature': df['temperature'].mean() if 'temperature' in df else 0,
                    'max_pressure': df['pressure'].max() if 'pressure' in df else 0,
                    'anomaly_rate': 0,
                    'batch_size': len(messages),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Process each message
                results = []
                anomaly_count = 0
                
                for msg in messages:
                    processed = await self._process_message_impl(msg)
                    if processed:
                        results.append(processed)
                        if processed.value.get('is_anomaly'):
                            anomaly_count += 1
                
                batch_stats['anomaly_rate'] = anomaly_count / len(messages) if messages else 0
                
                # Log batch statistics
                logger.info(f"Batch processed: {batch_stats}")
                
                return results
            
            def _validate_sensor_data(self, data: Dict) -> bool:
                """Validate sensor data"""
                required_fields = ['sensor_id', 'temperature', 'pressure', 'timestamp']
                return all(field in data for field in required_fields)
            
            def _extract_features(self, data: Dict) -> Dict:
                """Extract features from sensor data"""
                return {
                    'temp_pressure_ratio': data['temperature'] / (data['pressure'] + 1),
                    'temp_squared': data['temperature'] ** 2,
                    'pressure_log': np.log1p(data['pressure'])
                }
            
            def _detect_anomaly(self, features: Dict) -> bool:
                """Simple anomaly detection"""
                # Simple threshold-based detection
                if features['temp_pressure_ratio'] > 100:
                    return True
                if features['temp_squared'] > 10000:
                    return True
                return False
        
        # Create processor instance
        processor = SensorDataProcessor(config)
        
        # Add to orchestrator
        self.orchestrator.add_processor("sensor_processor", processor)
        
        print("\n✅ Kafka processor configured and ready")
        
        # Simulate message production
        print("\n📤 Simulating sensor data production...")
        self._produce_sample_messages(config)
        
        return processor
    
    def example_2_ml_streaming_inference(self):
        """Example 2: ML model inference in streaming pipeline"""
        print("\n" + "="*80)
        print("EXAMPLE 2: ML Streaming Inference")
        print("="*80)
        
        # First, train a simple model
        print("\n🎯 Training model for streaming inference...")
        X, y = self._create_sample_data()
        self._train_model(X, y)
        
        # Configure ML stream processor
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="ml-predictions",
            consumer_group="ml-consumer",
            batch_size=50,
            tenant_id="ml-tenant"
        )
        
        # Create ML processor
        ml_processor = MLStreamProcessor(config, self.model_path)
        
        print(f"\n🤖 ML Processor Configuration:")
        print(f"  - Model: {self.model_path}")
        print(f"  - Topic: {config.topic}")
        print(f"  - Batch Size: {config.batch_size}")
        
        # Custom ML processor with advanced features
        class AdvancedMLProcessor(MLStreamProcessor):
            """Advanced ML processor with monitoring"""
            
            def __init__(self, config: StreamConfig, model_path: str):
                super().__init__(config, model_path)
                self.prediction_history = []
                self.confidence_threshold = 0.7
            
            async def _process_message_impl(self, message: StreamMessage) -> Optional[StreamMessage]:
                """Process with confidence scoring"""
                
                if not self.model:
                    return message
                
                try:
                    # Extract features
                    features = self._extract_features(message.value)
                    
                    # Make prediction with probability
                    prediction = self.model.predict([features])[0]
                    
                    # Get prediction probability if available
                    confidence = 1.0
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba([features])[0]
                        confidence = max(probabilities)
                    
                    # Store in history for drift detection
                    self.prediction_history.append({
                        'features': features,
                        'prediction': prediction,
                        'confidence': confidence,
                        'timestamp': datetime.now()
                    })
                    
                    # Detect potential drift
                    drift_detected = self._check_drift()
                    
                    # Create result
                    result_value = message.value.copy()
                    result_value.update({
                        'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                        'confidence': confidence,
                        'high_confidence': confidence >= self.confidence_threshold,
                        'drift_warning': drift_detected,
                        'model_version': getattr(self.model, 'version', '1.0'),
                        'inference_timestamp': datetime.now().isoformat()
                    })
                    
                    return StreamMessage(
                        key=message.key,
                        value=result_value,
                        timestamp=datetime.now(),
                        partition=message.partition,
                        offset=message.offset
                    )
                    
                except Exception as e:
                    logger.error(f"ML inference error: {e}")
                    return None
            
            def _check_drift(self) -> bool:
                """Check for prediction drift"""
                if len(self.prediction_history) < 100:
                    return False
                
                # Simple drift detection: check if recent predictions differ from historical
                recent = self.prediction_history[-20:]
                historical = self.prediction_history[-100:-20]
                
                recent_mean = np.mean([p['confidence'] for p in recent])
                historical_mean = np.mean([p['confidence'] for p in historical])
                
                # Drift if confidence drops significantly
                return (historical_mean - recent_mean) > 0.1
        
        # Create advanced processor
        advanced_processor = AdvancedMLProcessor(config, self.model_path)
        
        # Add to orchestrator
        self.orchestrator.add_processor("ml_processor", advanced_processor)
        
        print("\n✅ ML streaming processor configured")
        
        # Simulate streaming inference
        print("\n🔮 Simulating streaming predictions...")
        asyncio.run(self._simulate_ml_streaming(advanced_processor))
        
        return advanced_processor
    
    def example_3_multi_stream_orchestration(self):
        """Example 3: Multi-stream orchestration with different platforms"""
        print("\n" + "="*80)
        print("EXAMPLE 3: Multi-Stream Orchestration")
        print("="*80)
        
        # Configure multiple streams
        streams = [
            {
                "name": "kafka_stream",
                "config": StreamConfig(
                    platform="kafka",
                    brokers=["localhost:9092"],
                    topic="events",
                    consumer_group="event-consumer",
                    tenant_id="tenant-1"
                )
            },
            {
                "name": "pulsar_stream",
                "config": StreamConfig(
                    platform="pulsar",
                    brokers=["localhost:6650"],
                    topic="metrics",
                    consumer_group="metric-consumer",
                    tenant_id="tenant-2"
                )
            }
        ]
        
        print("\n📡 Configuring multiple streams:")
        
        for stream_info in streams:
            try:
                processor = create_stream_processor(stream_info["config"])
                self.orchestrator.add_processor(stream_info["name"], processor)
                print(f"  ✅ {stream_info['name']}: {stream_info['config'].platform}")
            except Exception as e:
                print(f"  ❌ {stream_info['name']}: {e}")
        
        # Start processors
        print("\n▶️ Starting stream processors...")
        for name in ["kafka_stream"]:  # Only start Kafka for demo
            try:
                self.orchestrator.start_processor(name)
                print(f"  ✅ Started: {name}")
            except Exception as e:
                print(f"  ❌ Failed to start {name}: {e}")
        
        # Monitor metrics
        print("\n📊 Stream Metrics:")
        metrics = self.orchestrator.get_metrics()
        for name, metric in metrics.items():
            print(f"\n  {name}:")
            print(f"    - Processed: {metric['processed']}")
            print(f"    - Errors: {metric['errors']}")
            print(f"    - Throughput: {metric['throughput']:.2f} msg/s")
            print(f"    - Running: {metric['running']}")
        
        return self.orchestrator
    
    def example_4_prometheus_monitoring(self):
        """Example 4: Prometheus metrics monitoring"""
        print("\n" + "="*80)
        print("EXAMPLE 4: Prometheus Metrics Monitoring")
        print("="*80)
        
        # Start Prometheus metrics server
        print("\n📊 Starting Prometheus metrics server on port 8090...")
        start_http_server(8090, registry=streaming_registry)
        
        # Configure stream with monitoring
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="monitored-stream",
            consumer_group="monitoring-consumer",
            tenant_id="monitoring-tenant"
        )
        
        # Create processor
        processor = KafkaStreamProcessor(config)
        
        print("\n📈 Metrics exposed at: http://localhost:8090/metrics")
        print("\nAvailable metrics:")
        print("  - ml_streaming_messages_total: Total messages processed")
        print("  - ml_streaming_throughput: Current throughput")
        print("  - ml_streaming_lag: Consumer lag")
        print("  - ml_streaming_processing_latency_seconds: Processing latency")
        print("  - ml_streaming_batch_size: Batch sizes")
        print("  - ml_streaming_errors_total: Error count")
        print("  - ml_streaming_active_consumers: Active consumers")
        
        # Simulate processing to generate metrics
        print("\n⚡ Simulating stream processing for metrics...")
        asyncio.run(self._simulate_with_metrics(processor))
        
        # Display current metrics
        print("\n📊 Current Metrics Snapshot:")
        metrics_output = generate_latest(streaming_registry).decode('utf-8')
        
        # Parse and display key metrics
        for line in metrics_output.split('\n'):
            if 'ml_streaming_' in line and not line.startswith('#'):
                print(f"  {line}")
        
        return processor
    
    def example_5_windowed_aggregations(self):
        """Example 5: Windowed aggregations and real-time analytics"""
        print("\n" + "="*80)
        print("EXAMPLE 5: Windowed Aggregations & Real-time Analytics")
        print("="*80)
        
        class WindowedAggregationProcessor(KafkaStreamProcessor):
            """Processor with windowed aggregations"""
            
            def __init__(self, config: StreamConfig):
                super().__init__(config)
                self.windows = {
                    '1min': {'duration': 60, 'data': []},
                    '5min': {'duration': 300, 'data': []},
                    '15min': {'duration': 900, 'data': []}
                }
            
            async def _process_message_impl(self, message: StreamMessage) -> Optional[StreamMessage]:
                """Process with windowing"""
                
                current_time = datetime.now()
                
                # Add to windows
                for window_name, window in self.windows.items():
                    # Remove old data
                    cutoff = current_time - timedelta(seconds=window['duration'])
                    window['data'] = [
                        d for d in window['data'] 
                        if d['timestamp'] > cutoff
                    ]
                    
                    # Add new data
                    window['data'].append({
                        'value': message.value,
                        'timestamp': current_time
                    })
                
                # Calculate aggregations
                aggregations = {}
                for window_name, window in self.windows.items():
                    if window['data']:
                        values = [d['value'].get('value', 0) for d in window['data']]
                        aggregations[window_name] = {
                            'count': len(values),
                            'sum': sum(values),
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': min(values),
                            'max': max(values),
                            'p50': np.percentile(values, 50),
                            'p95': np.percentile(values, 95)
                        }
                
                # Create result with aggregations
                result_value = message.value.copy()
                result_value['window_aggregations'] = aggregations
                result_value['aggregation_timestamp'] = current_time.isoformat()
                
                # Detect trends
                if len(self.windows['5min']['data']) > 10:
                    recent_values = [d['value'].get('value', 0) for d in self.windows['5min']['data'][-10:]]
                    trend = 'increasing' if recent_values[-1] > recent_values[0] else 'decreasing'
                    result_value['trend'] = trend
                
                return StreamMessage(
                    key=message.key,
                    value=result_value,
                    timestamp=current_time,
                    partition=message.partition,
                    offset=message.offset
                )
        
        # Configure windowed processor
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="analytics-stream",
            consumer_group="analytics-consumer",
            window_size=60,
            tenant_id="analytics-tenant"
        )
        
        processor = WindowedAggregationProcessor(config)
        
        print(f"\n⏱️ Windowed Aggregation Configuration:")
        print(f"  - Windows: 1min, 5min, 15min")
        print(f"  - Metrics: count, sum, mean, std, min, max, p50, p95")
        print(f"  - Trend Detection: enabled")
        
        # Simulate windowed processing
        print("\n📊 Simulating windowed aggregations...")
        asyncio.run(self._simulate_windowed_processing(processor))
        
        return processor
    
    # Helper methods
    
    def _create_sample_data(self):
        """Create sample data for ML model"""
        np.random.seed(42)
        n_samples = 1000
        
        X = np.random.randn(n_samples, 10)
        y = (X[:, 0] + X[:, 1] * 0.5 - X[:, 2] * 0.3 > 0).astype(int)
        
        return X, y
    
    def _train_model(self, X, y):
        """Train and save a simple model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        print(f"  Model accuracy: {score:.3f}")
        
        # Save model
        import os
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, self.model_path)
        print(f"  Model saved to: {self.model_path}")
        
        self.model = model
    
    def _produce_sample_messages(self, config: StreamConfig):
        """Produce sample messages to Kafka"""
        try:
            from kafka import KafkaProducer
            
            producer = KafkaProducer(
                bootstrap_servers=config.brokers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            # Send sample messages
            for i in range(10):
                message = {
                    'sensor_id': f'sensor_{i % 3}',
                    'temperature': 20 + np.random.randn() * 5,
                    'pressure': 100 + np.random.randn() * 10,
                    'timestamp': datetime.now().isoformat()
                }
                
                producer.send(config.topic, value=message)
                print(f"  Sent: sensor_{i % 3} - temp={message['temperature']:.1f}")
                time.sleep(0.1)
            
            producer.flush()
            print("\n✅ Sample messages sent to Kafka")
            
        except Exception as e:
            print(f"\n⚠️ Could not send messages to Kafka: {e}")
            print("  Make sure Kafka is running on localhost:9092")
    
    async def _simulate_ml_streaming(self, processor):
        """Simulate ML streaming inference"""
        # Generate sample messages
        for i in range(5):
            message = StreamMessage(
                key=f"msg_{i}",
                value={
                    'feature_0': np.random.randn(),
                    'feature_1': np.random.randn(),
                    'feature_2': np.random.randn(),
                    'feature_3': np.random.randn(),
                    'feature_4': np.random.randn(),
                    'feature_5': np.random.randn(),
                    'feature_6': np.random.randn(),
                    'feature_7': np.random.randn(),
                    'feature_8': np.random.randn(),
                    'feature_9': np.random.randn(),
                },
                timestamp=datetime.now(),
                partition=0,
                offset=i
            )
            
            # Process message
            result = await processor.process(message)
            
            if result:
                print(f"\n  Message {i}:")
                print(f"    Prediction: {result.value.get('prediction')}")
                print(f"    Confidence: {result.value.get('confidence', 0):.3f}")
                print(f"    High Confidence: {result.value.get('high_confidence')}")
                print(f"    Drift Warning: {result.value.get('drift_warning')}")
    
    async def _simulate_with_metrics(self, processor):
        """Simulate processing with metrics generation"""
        # Process batches
        for batch_num in range(3):
            messages = []
            
            # Create batch
            for i in range(20):
                messages.append(StreamMessage(
                    key=f"key_{i}",
                    value={'value': np.random.randn() * 100},
                    timestamp=datetime.now(),
                    partition=0,
                    offset=batch_num * 20 + i
                ))
            
            # Process batch
            await processor.process_batch(messages)
            
            print(f"  Batch {batch_num + 1}: Processed {len(messages)} messages")
            
            # Update throughput metric
            processor.update_throughput_metric()
            
            await asyncio.sleep(1)
    
    async def _simulate_windowed_processing(self, processor):
        """Simulate windowed aggregation processing"""
        
        # Generate time series data
        for i in range(30):
            value = 50 + 10 * np.sin(i / 5) + np.random.randn() * 2
            
            message = StreamMessage(
                key=f"ts_{i}",
                value={'value': value, 'timestamp': datetime.now().isoformat()},
                timestamp=datetime.now(),
                partition=0,
                offset=i
            )
            
            # Process message
            result = await processor.process(message)
            
            if result and i % 5 == 0:
                aggregations = result.value.get('window_aggregations', {})
                
                print(f"\n  Time step {i}:")
                print(f"    Current value: {value:.2f}")
                
                for window_name, stats in aggregations.items():
                    if stats:
                        print(f"    {window_name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
                
                if 'trend' in result.value:
                    print(f"    Trend: {result.value['trend']}")
            
            await asyncio.sleep(0.2)


def main():
    """Main execution function"""
    example = StreamingExample()
    
    print("\n" + "="*80)
    print("STREAMING AND REAL-TIME PROCESSING EXAMPLES")
    print("="*80)
    
    # Example 1: Basic Kafka streaming
    kafka_processor = example.example_1_kafka_basic_streaming()
    
    # Example 2: ML streaming inference
    ml_processor = example.example_2_ml_streaming_inference()
    
    # Example 3: Multi-stream orchestration
    orchestrator = example.example_3_multi_stream_orchestration()
    
    # Example 4: Prometheus monitoring
    monitored_processor = example.example_4_prometheus_monitoring()
    
    # Example 5: Windowed aggregations
    windowed_processor = example.example_5_windowed_aggregations()
    
    print("\n" + "="*80)
    print("✅ ALL STREAMING EXAMPLES COMPLETED!")
    print("="*80)
    
    print("\n📊 Summary:")
    print("  - Kafka streaming processor configured")
    print("  - ML inference pipeline established")
    print("  - Multi-stream orchestration demonstrated")
    print("  - Prometheus metrics exposed on port 8090")
    print("  - Windowed aggregations implemented")
    
    print("\n⚠️ Note: Ensure Kafka/Pulsar are running for full functionality")
    print("  docker run -d --name kafka -p 9092:9092 confluentinc/cp-kafka:latest")
    
    # Stop all processors
    print("\n🛑 Stopping all processors...")
    orchestrator.stop_all()


if __name__ == "__main__":
    main()
