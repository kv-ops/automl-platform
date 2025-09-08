"""
Streaming and Real-time Processing Example - Complete Version
=============================================================
Place in: automl_platform/examples/example_streaming_realtime.py

Demonstrates real-time data streaming with multiple platforms (Kafka, Pulsar, Flink),
ML model inference, windowed aggregations, and Prometheus metrics monitoring.

Requirements:
- kafka-python: pip install kafka-python
- pulsar-client: pip install pulsar-client (optional)
- apache-flink: pip install apache-flink (optional)
- prometheus-client: pip install prometheus-client
- Redis: pip install redis (optional)
"""

import json
import asyncio
import logging
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import threading
from prometheus_client import start_http_server, generate_latest
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """Comprehensive real-time streaming and inference examples"""
    
    def __init__(self):
        self.config = AutoMLConfig()
        self.registry = MLflowRegistry(self.config)
        self.orchestrator = StreamingOrchestrator()
        self.model = None
        self.model_path = "models/streaming_model.pkl"
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
    
    def example_1_kafka_basic_streaming(self):
        """Example 1: Basic Kafka streaming setup with sensor data processing"""
        print("\n" + "="*80)
        print("EXAMPLE 1: Basic Kafka Streaming - Sensor Data Processing")
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
        print(f"  - Window Size: {config.window_size}s")
        print(f"  - Checkpoint Interval: {config.checkpoint_interval}s")
        
        # Create custom processor for sensor data
        class SensorDataProcessor(KafkaStreamProcessor):
            """Process real-time sensor data stream with anomaly detection"""
            
            def __init__(self, config: StreamConfig):
                super().__init__(config)
                self.anomaly_threshold = 2.5  # Standard deviations for anomaly
                self.baseline_stats = {}
                self.alert_callback = None
            
            async def _process_message_impl(self, message: StreamMessage) -> Optional[StreamMessage]:
                """Process individual sensor reading with anomaly detection"""
                
                # Extract sensor data
                sensor_data = message.value
                
                # Data validation
                if not self._validate_sensor_data(sensor_data):
                    logger.warning(f"Invalid sensor data: {sensor_data}")
                    return None
                
                # Feature extraction
                features = self._extract_features(sensor_data)
                
                # Update baseline statistics
                self._update_baseline(sensor_data)
                
                # Anomaly detection
                anomaly_score = self._calculate_anomaly_score(sensor_data)
                is_anomaly = anomaly_score > self.anomaly_threshold
                
                # Alert on anomaly
                if is_anomaly and self.alert_callback:
                    self.alert_callback(sensor_data, anomaly_score)
                
                # Create enriched message
                enriched_data = {
                    **sensor_data,
                    'features': features,
                    'anomaly_score': float(anomaly_score),
                    'is_anomaly': is_anomaly,
                    'processed_at': datetime.now().isoformat(),
                    'processor_version': '1.0'
                }
                
                return StreamMessage(
                    key=message.key,
                    value=enriched_data,
                    timestamp=datetime.now(),
                    partition=message.partition,
                    offset=message.offset,
                    headers={'processor': 'SensorDataProcessor'}
                )
            
            async def _process_batch_impl(self, messages: List[StreamMessage]) -> List[StreamMessage]:
                """Process batch of sensor readings with aggregations"""
                
                # Collect all sensor values
                sensor_values = [msg.value for msg in messages if self._validate_sensor_data(msg.value)]
                
                if not sensor_values:
                    return []
                
                df = pd.DataFrame(sensor_values)
                
                # Batch statistics
                batch_stats = {
                    'batch_size': len(messages),
                    'valid_readings': len(sensor_values),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Calculate aggregations for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col in df:
                        batch_stats[f'{col}_mean'] = float(df[col].mean())
                        batch_stats[f'{col}_std'] = float(df[col].std())
                        batch_stats[f'{col}_min'] = float(df[col].min())
                        batch_stats[f'{col}_max'] = float(df[col].max())
                
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
                batch_stats['anomaly_count'] = anomaly_count
                
                # Log batch statistics
                logger.info(f"Batch processed: {json.dumps(batch_stats, indent=2)}")
                
                # Create batch summary message
                summary_msg = StreamMessage(
                    key="batch_summary",
                    value=batch_stats,
                    timestamp=datetime.now(),
                    partition=0,
                    offset=0,
                    headers={'type': 'batch_summary'}
                )
                results.append(summary_msg)
                
                return results
            
            def _validate_sensor_data(self, data: Dict) -> bool:
                """Validate sensor data structure"""
                required_fields = ['sensor_id', 'temperature', 'pressure', 'timestamp']
                return all(field in data for field in required_fields)
            
            def _extract_features(self, data: Dict) -> Dict:
                """Extract engineered features from sensor data"""
                features = {}
                
                # Basic ratios and transformations
                if data.get('pressure', 0) > 0:
                    features['temp_pressure_ratio'] = data['temperature'] / data['pressure']
                else:
                    features['temp_pressure_ratio'] = 0
                
                features['temp_squared'] = data['temperature'] ** 2
                features['pressure_log'] = np.log1p(abs(data.get('pressure', 0)))
                
                # Derived features
                features['temp_celsius'] = (data['temperature'] - 32) * 5/9 if data.get('temperature') else 0
                features['pressure_psi'] = data.get('pressure', 0) * 14.5038  # Convert bar to PSI
                
                return features
            
            def _update_baseline(self, data: Dict):
                """Update baseline statistics for anomaly detection"""
                sensor_id = data.get('sensor_id')
                if not sensor_id:
                    return
                
                if sensor_id not in self.baseline_stats:
                    self.baseline_stats[sensor_id] = {
                        'temperature': {'values': [], 'mean': 0, 'std': 1},
                        'pressure': {'values': [], 'mean': 0, 'std': 1}
                    }
                
                # Update with exponential moving average
                for metric in ['temperature', 'pressure']:
                    if metric in data:
                        values = self.baseline_stats[sensor_id][metric]['values']
                        values.append(data[metric])
                        
                        # Keep only recent values (sliding window)
                        if len(values) > 100:
                            values.pop(0)
                        
                        # Update statistics
                        if len(values) > 2:
                            self.baseline_stats[sensor_id][metric]['mean'] = np.mean(values)
                            self.baseline_stats[sensor_id][metric]['std'] = np.std(values)
            
            def _calculate_anomaly_score(self, data: Dict) -> float:
                """Calculate anomaly score using z-score method"""
                sensor_id = data.get('sensor_id')
                if not sensor_id or sensor_id not in self.baseline_stats:
                    return 0.0
                
                scores = []
                for metric in ['temperature', 'pressure']:
                    if metric in data:
                        stats = self.baseline_stats[sensor_id][metric]
                        if stats['std'] > 0:
                            z_score = abs(data[metric] - stats['mean']) / stats['std']
                            scores.append(z_score)
                
                return max(scores) if scores else 0.0
            
            def _detect_anomaly(self, features: Dict) -> bool:
                """Simple rule-based anomaly detection"""
                # Threshold-based detection
                if features.get('temp_pressure_ratio', 0) > 100:
                    return True
                if features.get('temp_squared', 0) > 10000:
                    return True
                if features.get('pressure_log', 0) > 10:
                    return True
                return False
        
        # Create processor instance
        processor = SensorDataProcessor(config)
        
        # Set up alert callback
        def anomaly_alert(sensor_data, score):
            print(f"\n🚨 ANOMALY DETECTED!")
            print(f"  Sensor: {sensor_data.get('sensor_id')}")
            print(f"  Score: {score:.2f}")
            print(f"  Temperature: {sensor_data.get('temperature'):.1f}")
            print(f"  Pressure: {sensor_data.get('pressure'):.1f}")
        
        processor.alert_callback = anomaly_alert
        
        # Add to orchestrator
        self.orchestrator.add_processor("sensor_processor", processor)
        
        print("\n✅ Kafka processor configured and ready")
        print("   - Anomaly detection: Enabled")
        print("   - Feature extraction: Enabled")
        print("   - Batch aggregation: Enabled")
        
        # Simulate message production
        print("\n📤 Simulating sensor data production...")
        self._produce_sample_messages(config)
        
        return processor
    
    def example_2_ml_streaming_inference(self):
        """Example 2: ML model inference in streaming pipeline with drift detection"""
        print("\n" + "="*80)
        print("EXAMPLE 2: ML Streaming Inference with Drift Detection")
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
            window_size=120,
            checkpoint_interval=60,
            tenant_id="ml-tenant"
        )
        
        # Create ML processor with advanced features
        class AdvancedMLProcessor(MLStreamProcessor):
            """Advanced ML processor with monitoring and drift detection"""
            
            def __init__(self, config: StreamConfig, model_path: str):
                super().__init__(config, model_path)
                self.prediction_history = []
                self.confidence_threshold = 0.7
                self.drift_threshold = 0.15
                self.feature_stats = {}
                self.model_performance = {
                    'predictions': 0,
                    'high_confidence': 0,
                    'drift_warnings': 0
                }
            
            async def _process_message_impl(self, message: StreamMessage) -> Optional[StreamMessage]:
                """Process message with ML inference and confidence scoring"""
                
                if not self.model:
                    logger.error("Model not loaded")
                    return message
                
                try:
                    # Extract and validate features
                    features = self._extract_features(message.value)
                    if not features:
                        logger.warning("No valid features extracted")
                        return None
                    
                    # Update feature statistics for drift detection
                    self._update_feature_stats(features)
                    
                    # Make prediction with probability
                    prediction = self.model.predict([features])[0]
                    
                    # Get prediction probability/confidence
                    confidence = 1.0
                    prediction_proba = None
                    if hasattr(self.model, 'predict_proba'):
                        probabilities = self.model.predict_proba([features])[0]
                        confidence = float(max(probabilities))
                        prediction_proba = probabilities.tolist()
                    
                    # Store in history for analysis
                    self.prediction_history.append({
                        'features': features,
                        'prediction': prediction,
                        'confidence': confidence,
                        'timestamp': datetime.now()
                    })
                    
                    # Limit history size
                    if len(self.prediction_history) > 1000:
                        self.prediction_history.pop(0)
                    
                    # Detect drift
                    drift_info = self._check_drift()
                    
                    # Update performance metrics
                    self.model_performance['predictions'] += 1
                    if confidence >= self.confidence_threshold:
                        self.model_performance['high_confidence'] += 1
                    if drift_info['drift_detected']:
                        self.model_performance['drift_warnings'] += 1
                    
                    # Create enriched result
                    result_value = message.value.copy()
                    result_value.update({
                        'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                        'prediction_proba': prediction_proba,
                        'confidence': confidence,
                        'high_confidence': confidence >= self.confidence_threshold,
                        'drift_info': drift_info,
                        'model_version': getattr(self.model, 'version', '1.0'),
                        'model_performance': self.model_performance.copy(),
                        'inference_timestamp': datetime.now().isoformat()
                    })
                    
                    # Log if drift detected
                    if drift_info['drift_detected']:
                        logger.warning(f"Drift detected: {drift_info}")
                    
                    return StreamMessage(
                        key=message.key,
                        value=result_value,
                        timestamp=datetime.now(),
                        partition=message.partition,
                        offset=message.offset,
                        headers={'model_version': '1.0', 'processor': 'AdvancedMLProcessor'}
                    )
                    
                except Exception as e:
                    logger.error(f"ML inference error: {e}")
                    return None
            
            def _update_feature_stats(self, features: List[float]):
                """Update feature statistics for drift monitoring"""
                if 'mean' not in self.feature_stats:
                    self.feature_stats['mean'] = np.zeros(len(features))
                    self.feature_stats['std'] = np.ones(len(features))
                    self.feature_stats['count'] = 0
                
                # Update with exponential moving average
                alpha = 0.1  # Learning rate
                self.feature_stats['count'] += 1
                
                if self.feature_stats['count'] > 1:
                    old_mean = self.feature_stats['mean']
                    self.feature_stats['mean'] = (1 - alpha) * old_mean + alpha * np.array(features)
                    
                    # Update standard deviation
                    diff = np.array(features) - self.feature_stats['mean']
                    self.feature_stats['std'] = np.sqrt((1 - alpha) * self.feature_stats['std']**2 + alpha * diff**2)
            
            def _check_drift(self) -> Dict[str, Any]:
                """Comprehensive drift detection"""
                drift_info = {
                    'drift_detected': False,
                    'confidence_drift': False,
                    'prediction_drift': False,
                    'feature_drift': False,
                    'metrics': {}
                }
                
                if len(self.prediction_history) < 100:
                    drift_info['message'] = 'Insufficient data for drift detection'
                    return drift_info
                
                # Get recent and historical data
                recent = self.prediction_history[-20:]
                historical = self.prediction_history[-100:-20]
                
                # 1. Confidence drift
                recent_conf = np.mean([p['confidence'] for p in recent])
                hist_conf = np.mean([p['confidence'] for p in historical])
                conf_diff = hist_conf - recent_conf
                
                drift_info['metrics']['confidence_drop'] = float(conf_diff)
                if conf_diff > self.drift_threshold:
                    drift_info['confidence_drift'] = True
                    drift_info['drift_detected'] = True
                
                # 2. Prediction distribution drift
                recent_preds = [p['prediction'] for p in recent]
                hist_preds = [p['prediction'] for p in historical]
                
                # Calculate prediction distribution change
                if len(set(hist_preds)) <= 10:  # Classification
                    from collections import Counter
                    recent_dist = Counter(recent_preds)
                    hist_dist = Counter(hist_preds)
                    
                    # Calculate distribution difference
                    all_classes = set(recent_dist.keys()) | set(hist_dist.keys())
                    dist_diff = 0
                    for cls in all_classes:
                        recent_prob = recent_dist.get(cls, 0) / len(recent_preds)
                        hist_prob = hist_dist.get(cls, 0) / len(hist_preds)
                        dist_diff += abs(recent_prob - hist_prob)
                    
                    drift_info['metrics']['distribution_change'] = float(dist_diff)
                    if dist_diff > 0.3:
                        drift_info['prediction_drift'] = True
                        drift_info['drift_detected'] = True
                
                # 3. Feature drift (if we have feature stats)
                if 'mean' in self.feature_stats and self.feature_stats['count'] > 100:
                    # Check if recent features deviate from learned distribution
                    recent_features = [p['features'] for p in recent]
                    if recent_features:
                        recent_mean = np.mean(recent_features, axis=0)
                        z_scores = np.abs((recent_mean - self.feature_stats['mean']) / 
                                        (self.feature_stats['std'] + 1e-8))
                        max_z = np.max(z_scores)
                        
                        drift_info['metrics']['max_feature_zscore'] = float(max_z)
                        if max_z > 3:
                            drift_info['feature_drift'] = True
                            drift_info['drift_detected'] = True
                
                return drift_info
        
        # Create advanced processor
        advanced_processor = AdvancedMLProcessor(config, self.model_path)
        
        print(f"\n🤖 ML Processor Configuration:")
        print(f"  - Model: {self.model_path}")
        print(f"  - Topic: {config.topic}")
        print(f"  - Batch Size: {config.batch_size}")
        print(f"  - Confidence Threshold: {advanced_processor.confidence_threshold}")
        print(f"  - Drift Detection: Enabled")
        print(f"  - Features: Confidence, Prediction, Feature drift monitoring")
        
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
        
        # Configure multiple streams for different data types
        streams = [
            {
                "name": "kafka_events",
                "config": StreamConfig(
                    platform="kafka",
                    brokers=["localhost:9092"],
                    topic="user-events",
                    consumer_group="event-consumer",
                    batch_size=200,
                    window_size=30,
                    tenant_id="tenant-1"
                ),
                "description": "User interaction events"
            },
            {
                "name": "kafka_metrics",
                "config": StreamConfig(
                    platform="kafka",
                    brokers=["localhost:9092"],
                    topic="system-metrics",
                    consumer_group="metric-consumer",
                    batch_size=500,
                    window_size=60,
                    tenant_id="tenant-1"
                ),
                "description": "System performance metrics"
            },
            {
                "name": "pulsar_logs",
                "config": StreamConfig(
                    platform="pulsar",
                    brokers=["localhost:6650"],
                    topic="application-logs",
                    consumer_group="log-consumer",
                    batch_size=1000,
                    tenant_id="tenant-2"
                ),
                "description": "Application logs"
            }
        ]
        
        print("\n📡 Configuring multiple streams:")
        print("="*50)
        
        active_streams = []
        for stream_info in streams:
            try:
                processor = create_stream_processor(stream_info["config"])
                self.orchestrator.add_processor(stream_info["name"], processor)
                print(f"\n✅ {stream_info['name']}:")
                print(f"   Platform: {stream_info['config'].platform}")
                print(f"   Topic: {stream_info['config'].topic}")
                print(f"   Description: {stream_info['description']}")
                active_streams.append(stream_info["name"])
            except Exception as e:
                print(f"\n❌ {stream_info['name']}: {e}")
        
        # Start processors (only Kafka for demo)
        print("\n▶️ Starting stream processors...")
        for name in active_streams[:2]:  # Start only Kafka streams for demo
            try:
                # Don't actually start if broker not available
                # self.orchestrator.start_processor(name)
                print(f"  ✅ Ready to start: {name}")
            except Exception as e:
                print(f"  ❌ Failed to start {name}: {e}")
        
        # Display orchestration dashboard
        print("\n" + "="*50)
        print("📊 Stream Orchestration Dashboard")
        print("="*50)
        
        metrics = self.orchestrator.get_metrics()
        
        # Create summary table
        print("\n┌─────────────────────┬──────────┬────────┬────────────┬─────────┐")
        print("│ Stream Name         │ Processed│ Errors │ Throughput │ Status  │")
        print("├─────────────────────┼──────────┼────────┼────────────┼─────────┤")
        
        for name, metric in metrics.items():
            status = "Running" if metric['running'] else "Stopped"
            throughput = f"{metric['throughput']:.1f} msg/s"
            print(f"│ {name:<19} │ {metric['processed']:>8} │ {metric['errors']:>6} │ {throughput:>10} │ {status:>7} │")
        
        print("└─────────────────────┴──────────┴────────┴────────────┴─────────┘")
        
        # Health check
        print("\n🏥 Health Status:")
        total_processed = sum(m['processed'] for m in metrics.values())
        total_errors = sum(m['errors'] for m in metrics.values())
        error_rate = (total_errors / total_processed * 100) if total_processed > 0 else 0
        
        print(f"  Total Messages: {total_processed}")
        print(f"  Total Errors: {total_errors}")
        print(f"  Error Rate: {error_rate:.2f}%")
        print(f"  Active Streams: {len([m for m in metrics.values() if m['running']])}/{len(metrics)}")
        
        return self.orchestrator
    
    def example_4_prometheus_monitoring(self):
        """Example 4: Prometheus metrics monitoring and alerting"""
        print("\n" + "="*80)
        print("EXAMPLE 4: Prometheus Metrics Monitoring")
        print("="*80)
        
        # Start Prometheus metrics server
        print("\n📊 Starting Prometheus metrics server on port 8090...")
        try:
            start_http_server(8090, registry=streaming_registry)
            print("✅ Metrics server started successfully")
        except Exception as e:
            print(f"⚠️ Metrics server may already be running: {e}")
        
        # Configure stream with monitoring
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="monitored-stream",
            consumer_group="monitoring-consumer",
            batch_size=100,
            window_size=60,
            checkpoint_interval=30,
            tenant_id="monitoring-tenant"
        )
        
        # Create processor with custom metrics
        class MonitoredProcessor(KafkaStreamProcessor):
            """Processor with enhanced monitoring capabilities"""
            
            def __init__(self, config: StreamConfig):
                super().__init__(config)
                self.alert_thresholds = {
                    'error_rate': 0.05,  # 5% error rate
                    'latency_p95': 1.0,  # 1 second
                    'lag': 1000  # 1000 messages
                }
                self.alerts = []
            
            async def _process_message_impl(self, message: StreamMessage) -> Optional[StreamMessage]:
                """Process with monitoring"""
                # Simulate processing with variable latency
                import random
                await asyncio.sleep(random.uniform(0.001, 0.1))
                
                # Check for alerts
                self._check_alerts()
                
                return message
            
            def _check_alerts(self):
                """Check metrics against thresholds"""
                # Check error rate
                if self.processed_count > 0:
                    error_rate = self.error_count / self.processed_count
                    if error_rate > self.alert_thresholds['error_rate']:
                        alert = {
                            'type': 'HIGH_ERROR_RATE',
                            'value': error_rate,
                            'threshold': self.alert_thresholds['error_rate'],
                            'timestamp': datetime.now().isoformat()
                        }
                        self.alerts.append(alert)
                        logger.warning(f"ALERT: {alert}")
        
        # Create processor
        processor = MonitoredProcessor(config)
        
        print("\n📈 Metrics exposed at: http://localhost:8090/metrics")
        print("\n📊 Available Metrics:")
        print("="*50)
        
        metrics_info = [
            ("ml_streaming_messages_total", "Total messages processed", "Counter"),
            ("ml_streaming_throughput", "Current throughput (msg/s)", "Gauge"),
            ("ml_streaming_lag", "Consumer lag in messages", "Gauge"),
            ("ml_streaming_processing_latency_seconds", "Processing latency", "Histogram"),
            ("ml_streaming_batch_size", "Batch sizes distribution", "Histogram"),
            ("ml_streaming_errors_total", "Total error count", "Counter"),
            ("ml_streaming_active_consumers", "Active consumer count", "Gauge")
        ]
        
        for metric_name, description, metric_type in metrics_info:
            print(f"  • {metric_name}")
            print(f"    {description} ({metric_type})")
        
        # Simulate processing to generate metrics
        print("\n⚡ Generating sample metrics...")
        asyncio.run(self._simulate_with_metrics(processor))
        
        # Display metrics snapshot
        print("\n📊 Current Metrics Snapshot:")
        print("="*50)
        
        try:
            metrics_output = generate_latest(streaming_registry).decode('utf-8')
            
            # Parse and display key metrics
            for line in metrics_output.split('\n'):
                if 'ml_streaming_' in line and not line.startswith('#'):
                    # Extract metric name and value
                    if '{' in line:
                        parts = line.split('{')
                        metric = parts[0].replace('ml_streaming_', '')
                        value_part = parts[1].split('}')[-1].strip()
                        if value_part:
                            print(f"  {metric}: {value_part}")
        except Exception as e:
            print(f"  Unable to display metrics: {e}")
        
        # Display alerts if any
        if hasattr(processor, 'alerts') and processor.alerts:
            print("\n🚨 Active Alerts:")
            for alert in processor.alerts[-5:]:  # Show last 5 alerts
                print(f"  - {alert['type']}: {alert['value']:.3f} > {alert['threshold']}")
        
        return processor
    
    def example_5_windowed_aggregations(self):
        """Example 5: Windowed aggregations and real-time analytics"""
        print("\n" + "="*80)
        print("EXAMPLE 5: Windowed Aggregations & Real-time Analytics")
        print("="*80)
        
        class WindowedAggregationProcessor(KafkaStreamProcessor):
            """Advanced processor with multiple time windows and analytics"""
            
            def __init__(self, config: StreamConfig):
                super().__init__(config)
                self.windows = {
                    '1min': {'duration': 60, 'data': [], 'aggregates': {}},
                    '5min': {'duration': 300, 'data': [], 'aggregates': {}},
                    '15min': {'duration': 900, 'data': [], 'aggregates': {}},
                    '1hour': {'duration': 3600, 'data': [], 'aggregates': {}}
                }
                self.trend_detector = TrendDetector()
                self.anomaly_detector = AnomalyDetector()
            
            async def _process_message_impl(self, message: StreamMessage) -> Optional[StreamMessage]:
                """Process with multi-window aggregations and analytics"""
                
                current_time = datetime.now()
                
                # Add to all windows
                for window_name, window in self.windows.items():
                    # Remove expired data
                    cutoff = current_time - timedelta(seconds=window['duration'])
                    window['data'] = [
                        d for d in window['data'] 
                        if d['timestamp'] > cutoff
                    ]
                    
                    # Add new data point
                    window['data'].append({
                        'value': message.value,
                        'timestamp': current_time
                    })
                    
                    # Calculate aggregations
                    window['aggregates'] = self._calculate_window_aggregates(window['data'])
                
                # Detect patterns across windows
                patterns = self._detect_patterns()
                
                # Forecast next values
                forecast = self._forecast_next_values()
                
                # Create enriched result
                result_value = message.value.copy()
                result_value.update({
                    'window_aggregations': {
                        name: window['aggregates'] 
                        for name, window in self.windows.items()
                    },
                    'patterns': patterns,
                    'forecast': forecast,
                    'aggregation_timestamp': current_time.isoformat()
                })
                
                return StreamMessage(
                    key=message.key,
                    value=result_value,
                    timestamp=current_time,
                    partition=message.partition,
                    offset=message.offset,
                    headers={'processor': 'WindowedAggregation'}
                )
            
            def _calculate_window_aggregates(self, window_data: List[Dict]) -> Dict:
                """Calculate comprehensive aggregates for a window"""
                if not window_data:
                    return {}
                
                # Extract numeric values
                values = []
                for d in window_data:
                    if isinstance(d['value'], dict) and 'value' in d['value']:
                        values.append(d['value']['value'])
                    elif isinstance(d['value'], (int, float)):
                        values.append(d['value'])
                
                if not values:
                    return {}
                
                # Calculate statistics
                aggregates = {
                    'count': len(values),
                    'sum': float(sum(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'range': float(max(values) - min(values)),
                    'median': float(np.median(values)),
                    'p25': float(np.percentile(values, 25)),
                    'p75': float(np.percentile(values, 75)),
                    'p95': float(np.percentile(values, 95)),
                    'p99': float(np.percentile(values, 99)),
                    'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
                }
                
                # Detect outliers using IQR method
                q1, q3 = aggregates['p25'], aggregates['p75']
                iqr = aggregates['iqr']
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = [v for v in values if v < lower_bound or v > upper_bound]
                aggregates['outlier_count'] = len(outliers)
                aggregates['outlier_ratio'] = len(outliers) / len(values) if values else 0
                
                return aggregates
            
            def _detect_patterns(self) -> Dict:
                """Detect patterns across different time windows"""
                patterns = {
                    'trend': 'stable',
                    'volatility': 'normal',
                    'seasonality': False,
                    'change_points': []
                }
                
                # Compare aggregates across windows
                if all(w['aggregates'] for w in self.windows.values()):
                    # Trend detection: compare means across windows
                    means = [w['aggregates'].get('mean', 0) for w in self.windows.values()]
                    if len(means) >= 2:
                        if means[-1] > means[0] * 1.1:
                            patterns['trend'] = 'increasing'
                        elif means[-1] < means[0] * 0.9:
                            patterns['trend'] = 'decreasing'
                    
                    # Volatility: compare standard deviations
                    stds = [w['aggregates'].get('std', 0) for w in self.windows.values()]
                    avg_std = np.mean(stds)
                    if avg_std > np.mean(means) * 0.3:
                        patterns['volatility'] = 'high'
                    elif avg_std < np.mean(means) * 0.1:
                        patterns['volatility'] = 'low'
                
                return patterns
            
            def _forecast_next_values(self) -> Dict:
                """Simple forecasting based on recent trends"""
                forecast = {
                    'next_value': None,
                    'confidence': 0,
                    'method': 'none'
                }
                
                # Use 5-minute window for forecasting
                if '5min' in self.windows and self.windows['5min']['data']:
                    window_data = self.windows['5min']['data']
                    if len(window_data) >= 5:
                        # Extract values
                        values = []
                        for d in window_data[-10:]:  # Last 10 points
                            if isinstance(d['value'], dict) and 'value' in d['value']:
                                values.append(d['value']['value'])
                        
                        if len(values) >= 3:
                            # Simple linear extrapolation
                            x = np.arange(len(values))
                            z = np.polyfit(x, values, 1)
                            next_value = z[0] * len(values) + z[1]
                            
                            # Calculate confidence based on fit quality
                            predicted = np.poly1d(z)(x)
                            residuals = values - predicted
                            r_squared = 1 - (np.sum(residuals**2) / np.sum((values - np.mean(values))**2))
                            
                            forecast = {
                                'next_value': float(next_value),
                                'confidence': float(max(0, min(1, r_squared))),
                                'method': 'linear_regression',
                                'trend_slope': float(z[0])
                            }
                
                return forecast
        
        # Helper classes for pattern detection
        class TrendDetector:
            """Detect trends in time series data"""
            def detect(self, values):
                if len(values) < 3:
                    return 'insufficient_data'
                
                # Simple moving average comparison
                first_third = np.mean(values[:len(values)//3])
                last_third = np.mean(values[-len(values)//3:])
                
                if last_third > first_third * 1.1:
                    return 'upward'
                elif last_third < first_third * 0.9:
                    return 'downward'
                else:
                    return 'stable'
        
        class AnomalyDetector:
            """Detect anomalies in streaming data"""
            def __init__(self):
                self.baseline_mean = 0
                self.baseline_std = 1
                
            def detect(self, value):
                z_score = abs(value - self.baseline_mean) / self.baseline_std
                return z_score > 3
        
        # Configure windowed processor
        config = StreamConfig(
            platform="kafka",
            brokers=["localhost:9092"],
            topic="analytics-stream",
            consumer_group="analytics-consumer",
            window_size=60,
            batch_size=50,
            tenant_id="analytics-tenant"
        )
        
        processor = WindowedAggregationProcessor(config)
        
        print(f"\n⏱️ Windowed Aggregation Configuration:")
        print("="*50)
        print(f"  Time Windows:")
        for window_name, window in processor.windows.items():
            print(f"    - {window_name}: {window['duration']}s")
        
        print(f"\n  Calculated Metrics:")
        print(f"    - Basic: count, sum, mean, std, min, max")
        print(f"    - Percentiles: p25, p50, p75, p95, p99")
        print(f"    - Advanced: IQR, outliers, range")
        
        print(f"\n  Analytics Features:")
        print(f"    ✓ Trend Detection")
        print(f"    ✓ Volatility Analysis")
        print(f"    ✓ Pattern Recognition")
        print(f"    ✓ Outlier Detection")
        print(f"    ✓ Simple Forecasting")
        
        # Simulate windowed processing
        print("\n📊 Simulating windowed aggregations...")
        asyncio.run(self._simulate_windowed_processing(processor))
        
        return processor
    
    # Helper methods
    
    def _create_sample_data(self):
        """Create sample data for ML model training"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Create target with some pattern
        y = (X[:, 0] + X[:, 1] * 0.5 - X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        print(f"  Created dataset: {n_samples} samples, {n_features} features")
        print(f"  Class distribution: {np.bincount(y)}")
        
        return X, y
    
    def _train_model(self, X, y):
        """Train and save a simple model for streaming inference"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"  Model Performance:")
        print(f"    - Training accuracy: {train_score:.3f}")
        print(f"    - Test accuracy: {test_score:.3f}")
        
        # Add version info
        model.version = "1.0"
        
        # Save model
        joblib.dump(model, self.model_path)
        print(f"  Model saved to: {self.model_path}")
        
        self.model = model
    
    def _produce_sample_messages(self, config: StreamConfig):
        """Produce sample messages to Kafka (simulation)"""
        try:
            from kafka import KafkaProducer
            
            producer = KafkaProducer(
                bootstrap_servers=config.brokers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            # Send sample sensor messages
            sensors = ['sensor_001', 'sensor_002', 'sensor_003']
            
            for i in range(10):
                sensor_id = sensors[i % len(sensors)]
                
                # Generate realistic sensor data
                base_temp = 20 + (i % 3) * 5
                base_pressure = 100 + (i % 3) * 10
                
                message = {
                    'sensor_id': sensor_id,
                    'temperature': base_temp + np.random.randn() * 2,
                    'pressure': base_pressure + np.random.randn() * 5,
                    'humidity': 50 + np.random.randn() * 10,
                    'timestamp': datetime.now().isoformat(),
                    'location': f'zone_{(i % 3) + 1}',
                    'status': 'active'
                }
                
                # Occasionally inject anomaly
                if i == 7:
                    message['temperature'] = 95  # Anomaly
                    message['status'] = 'warning'
                
                producer.send(config.topic, value=message, key=sensor_id.encode())
                print(f"  → Sent: {sensor_id} | T={message['temperature']:.1f}°C, P={message['pressure']:.1f}bar")
                time.sleep(0.1)
            
            producer.flush()
            print("\n✅ Sample messages sent to Kafka")
            
        except Exception as e:
            print(f"\n⚠️ Could not send messages to Kafka: {e}")
            print("  Note: Kafka broker must be running on localhost:9092")
            print("  Start Kafka with: docker-compose up -d kafka")
    
    async def _simulate_ml_streaming(self, processor):
        """Simulate ML streaming inference with drift"""
        
        print("\nProcessing messages with ML inference:")
        print("-" * 50)
        
        # Generate messages with gradual drift
        for i in range(10):
            # Create features with increasing drift
            drift_factor = i / 10.0  # Gradually increase drift
            
            features = {}
            for j in range(10):
                # Add drift to later messages
                base_value = np.random.randn()
                if i > 5:  # Introduce drift after message 5
                    base_value += drift_factor * 2
                
                features[f'feature_{j}'] = base_value
            
            message = StreamMessage(
                key=f"msg_{i}",
                value=features,
                timestamp=datetime.now(),
                partition=0,
                offset=i
            )
            
            # Process message
            result = await processor.process(message)
            
            if result:
                print(f"\nMessage {i + 1}:")
                print(f"  Prediction: {result.value.get('prediction')}")
                print(f"  Confidence: {result.value.get('confidence', 0):.3f}")
                
                drift_info = result.value.get('drift_info', {})
                if drift_info.get('drift_detected'):
                    print(f"  ⚠️ DRIFT DETECTED: {drift_info}")
                
                # Small delay for readability
                await asyncio.sleep(0.1)
    
    async def _simulate_with_metrics(self, processor):
        """Simulate processing with metrics generation"""
        
        print("\nGenerating metrics through batch processing:")
        print("-" * 50)
        
        # Process multiple batches
        for batch_num in range(3):
            messages = []
            
            # Create batch of messages
            for i in range(20):
                value = 50 + 10 * np.sin((batch_num * 20 + i) / 5) + np.random.randn() * 2
                
                messages.append(StreamMessage(
                    key=f"key_{batch_num}_{i}",
                    value={'value': value, 'batch': batch_num},
                    timestamp=datetime.now(),
                    partition=0,
                    offset=batch_num * 20 + i
                ))
            
            # Process batch
            results = await processor.process_batch(messages)
            
            print(f"  Batch {batch_num + 1}: Processed {len(messages)} messages → {len(results)} results")
            
            # Update throughput metric
            processor.update_throughput_metric()
            
            # Display current metrics
            print(f"    - Total processed: {processor.processed_count}")
            print(f"    - Throughput: {processor.metrics.get('throughput_per_sec', 0):.1f} msg/s")
            
            await asyncio.sleep(1)
    
    async def _simulate_windowed_processing(self, processor):
        """Simulate windowed aggregation processing with patterns"""
        
        print("\nProcessing time series with windowed aggregations:")
        print("-" * 50)
        
        # Generate time series data with pattern
        for i in range(30):
            # Create value with trend and seasonality
            trend = i * 0.5  # Upward trend
            seasonal = 10 * np.sin(i / 5)  # Seasonal pattern
            noise = np.random.randn() * 2  # Random noise
            value = 50 + trend + seasonal + noise
            
            # Occasionally inject anomaly
            if i in [10, 20]:
                value += 20  # Spike
            
            message = StreamMessage(
                key=f"ts_{i}",
                value={
                    'value': value,
                    'timestamp': datetime.now().isoformat(),
                    'series_id': 'test_series'
                },
                timestamp=datetime.now(),
                partition=0,
                offset=i
            )
            
            # Process message
            result = await processor.process(message)
            
            # Display results periodically
            if result and i % 5 == 0 and i > 0:
                aggregations = result.value.get('window_aggregations', {})
                patterns = result.value.get('patterns', {})
                forecast = result.value.get('forecast', {})
                
                print(f"\n⏱️ Time step {i + 1}:")
                print(f"  Current value: {value:.2f}")
                
                # Show window aggregates
                for window_name, stats in aggregations.items():
                    if stats:
                        print(f"\n  {window_name} window:")
                        print(f"    Mean: {stats.get('mean', 0):.2f}, Std: {stats.get('std', 0):.2f}")
                        print(f"    Range: [{stats.get('min', 0):.2f}, {stats.get('max', 0):.2f}]")
                        if stats.get('outlier_count', 0) > 0:
                            print(f"    Outliers: {stats['outlier_count']} ({stats.get('outlier_ratio', 0):.1%})")
                
                # Show patterns
                if patterns:
                    print(f"\n  Detected Patterns:")
                    print(f"    Trend: {patterns.get('trend', 'unknown')}")
                    print(f"    Volatility: {patterns.get('volatility', 'unknown')}")
                
                # Show forecast
                if forecast.get('next_value') is not None:
                    print(f"\n  Forecast:")
                    print(f"    Next value: {forecast['next_value']:.2f}")
                    print(f"    Confidence: {forecast.get('confidence', 0):.1%}")
                    print(f"    Method: {forecast.get('method', 'none')}")
            
            await asyncio.sleep(0.1)


def main():
    """Main execution function"""
    example = StreamingExample()
    
    print("\n" + "="*80)
    print(" " * 15 + "STREAMING AND REAL-TIME PROCESSING EXAMPLES")
    print("="*80)
    
    # Run examples
    try:
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
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print(" " * 25 + "✅ ALL EXAMPLES COMPLETED!")
    print("="*80)
    
    print("\n📊 Summary of Features Demonstrated:")
    print("  ✓ Kafka streaming with sensor data processing")
    print("  ✓ ML model inference with drift detection")
    print("  ✓ Multi-stream orchestration across platforms")
    print("  ✓ Prometheus metrics and monitoring")
    print("  ✓ Windowed aggregations with pattern detection")
    print("  ✓ Real-time anomaly detection")
    print("  ✓ Time series forecasting")
    print("  ✓ Batch processing with statistics")
    
    print("\n🚀 Next Steps:")
    print("  1. Install required dependencies:")
    print("     pip install kafka-python prometheus-client")
    print("  2. Start Kafka broker:")
    print("     docker run -d -p 9092:9092 confluentinc/cp-kafka:latest")
    print("  3. Access Prometheus metrics:")
    print("     http://localhost:8090/metrics")
    print("  4. Integrate with your ML models and data sources")
    
    # Stop all processors
    print("\n🛑 Cleaning up...")
    example.orchestrator.stop_all()
    print("✅ All processors stopped")


if __name__ == "__main__":
    main()
