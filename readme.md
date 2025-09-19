# AutoML Platform v3.1 - Enterprise MLOps Edition with Expert Mode

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-2.9%2B-0194E2)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)
[![Expert Mode](https://img.shields.io/badge/Expert%20Mode-Available-gold)](docs/expert-mode.md)
[![ONNX](https://img.shields.io/badge/ONNX-1.15%2B-5C5C5C)](https://onnx.ai/)
[![River](https://img.shields.io/badge/River-0.19%2B-00CED1)](https://riverml.xyz/)
[![Test Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen)](https://codecov.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-ready AutoML platform with **Expert Mode** for advanced users, enterprise MLOps capabilities including incremental learning, real-time streaming, advanced scheduling, billing system, and LLM-powered insights.

## 🆕 New in v3.1 - Expert Mode

The platform now features a **dual-mode interface**:

- **🚀 Simplified Mode (Default)**: Streamlined interface with optimized defaults for non-technical users
- **🎓 Expert Mode**: Full access to 30+ algorithms, advanced HPO, distributed computing, and GPU configuration

### Quick Comparison

| Feature | Simplified Mode | Expert Mode |
|---------|----------------|-------------|
| **Algorithms** | 3 reliable (XGBoost, RF, LR) | 30+ including neural networks |
| **HPO** | 20 iterations, automatic | Up to 500 iterations, customizable |
| **Validation** | 3-fold CV | 2-10 folds, multiple strategies |
| **Workers** | 2 (fixed) | 1-32 configurable |
| **GPU** | Disabled | Full GPU support |
| **Ensemble** | Voting only | Stacking, Blending, custom |
| **Feature Engineering** | Automatic | Full control |
| **Time to configure** | < 1 minute | 5-10 minutes |
| **Best for** | Quick prototypes, beginners | Production, research, optimization |

## 🚀 Quick Start with Expert Mode

### Enable Expert Mode

```bash
# Simplified mode (default)
python main.py train --data data.csv --target churn

# Expert mode - access all options
python main.py train --expert --data data.csv --target churn \
    --algorithms XGBoost,LightGBM,CatBoost,TabNet \
    --hpo-iter 200 \
    --n-workers 16 \
    --gpu

# Set globally via environment variable
export AUTOML_EXPERT_MODE=true
python main.py train --data data.csv --target churn
```

### Web Interface

1. Start the dashboard: `streamlit run automl_platform/ui/dashboard.py`
2. Click the "🎓 Mode Expert" checkbox in the sidebar
3. All advanced options become available

## 🚀 Features Overview

### Core AutoML
- **No Data Leakage**: All preprocessing within CV folds using sklearn Pipeline
- **30+ Algorithms**: Including XGBoost, LightGBM, CatBoost, Neural Networks
- **Hyperparameter Optimization**: Optuna, Grid Search, Random Search
- **Ensemble Methods**: Voting, stacking, blending with meta-learners
- **Imbalance Handling**: SMOTE, class weights, focal loss
- **Expert Mode**: Advanced configuration for power users

### Incremental & Streaming ML
- **Online Learning Models**: 
  - SGD-based incremental models (classification/regression)
  - River models (Hoeffding Trees, Adaptive Random Forest)
  - Neural incremental models with experience replay
- **Drift Detection Algorithms**: ADWIN, DDM, EDDM, Page-Hinkley
- **Stream Processors**: MLStreamProcessor for real-time predictions
- **Replay Buffer**: Configurable buffer size and replay frequency
- **Windowed Aggregation**: Time-based feature aggregation

### Real-Time Streaming
- **Multiple Platforms**: Kafka, Apache Flink, Apache Pulsar, Redis Streams
- **Stream Configuration**: Batch size, window size, checkpoint intervals
- **Exactly-Once Semantics**: Support for guaranteed message processing
- **Streaming Orchestrator**: Unified interface for all streaming platforms
- **Performance Metrics**: Throughput tracking, latency monitoring

### MLOps & Production

#### Model Registry (MLflow)
- **Version Management**: Track all model versions with metadata
- **Stage Transitions**: None → Staging → Production → Archived
- **Model Comparison**: Compare metrics between versions
- **Rollback Support**: Quick rollback to previous versions
- **Signature Inference**: Automatic input/output schema capture

#### A/B Testing Framework
- **Statistical Tests**: t-test, Mann-Whitney U, Chi-square
- **Effect Size Calculation**: Cohen's d, rank biserial, Cramér's V
- **Visualization Suite**: ROC curves, PR curves, confusion matrices, residual plots
- **Traffic Routing**: Configurable traffic splits for champion/challenger
- **Auto-promotion**: Promote winner based on statistical significance

#### Model Export Service
- **ONNX Export**: Dynamic/static quantization, INT8 optimization
- **PMML Support**: For traditional ML deployment
- **Edge Deployment**: Automated package creation with inference scripts
- **Quantization Options**: Dynamic INT8, static INT8 with calibration
- **Multi-format Support**: ONNX, PMML, TFLite (planned), CoreML (planned)

### Enterprise Features
- **Multi-Tenant Architecture**: Secure tenant isolation with resource management
- **Billing Middleware**: Real-time quota enforcement and usage tracking at API level
- **Advanced Job Scheduling**:
  - Priority-based queues (GPU_TRAINING, GPU_INFERENCE, CPU_PRIORITY, LLM, BATCH)
  - Plan-based quotas (Free, Trial, Pro, Enterprise)
  - Automatic worker scaling
  - GPU resource management
- **Comprehensive Billing System**:
  - Subscription plans with tiered pricing
  - Usage tracking (API calls, predictions, GPU hours, storage)
  - Payment integration (Stripe, PayPal)
  - Overage charge calculation
  - Invoice generation and auto-payment
- **LLM-Powered Features**:
  - Akkio-style conversational data cleaning
  - Feature engineering suggestions
  - Natural language model explanations
  - RAG-based knowledge queries
  - Code generation for AutoML tasks

### Advanced Monitoring & Alerting
- **Drift Detection**: KS test for numerical, Chi-square for categorical
- **Population Stability Index (PSI)**: Monitor distribution shifts
- **Multi-Channel Alerts**: Slack webhooks, Email (SMTP), generic webhooks
- **Prometheus Metrics**: Complete metrics export for Grafana
- **Data Quality Monitoring**: Schema validation, outlier detection, missing value tracking
- **Billing Metrics**: Cost tracking per tenant/model

## Table of Contents

- [Installation](#installation)
- [Expert Mode Guide](#expert-mode-guide)
- [Quick Start](#quick-start)
- [Incremental Learning](#incremental-learning)
- [Streaming ML](#streaming-ml)
- [Job Scheduling](#job-scheduling)
- [Billing System](#billing-system)
- [Billing Middleware](#billing-middleware)
- [LLM Features](#llm-features)
- [MLOps Workflow](#mlops-workflow)
- [A/B Testing](#ab-testing)
- [Model Export](#model-export)
- [Monitoring & Alerts](#monitoring--alerts)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [UI Dashboards](#ui-dashboards)
- [Testing](#testing)
- [Development](#development)
- [Deployment](#deployment)
- [Performance](#performance)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.8 or higher
- Redis (for caching and workers)
- MLflow (for model registry)
- Optional: Kafka/Pulsar (for streaming), Docker, Airflow/Prefect
- Optional: PostgreSQL/MySQL (for production)
- Optional: CUDA 11.0+ for GPU support

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/your-org/automl-platform.git
cd automl-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Basic installation (recommended for beginners)
pip install -r requirements-minimal.txt

# Or use the installation script
./install_mlops.sh
```

### Installation Options

```bash
# Basic MLOps installation (Simplified Mode)
pip install -r requirements-minimal.txt

# Full installation (includes Expert Mode features)
pip install -r requirements.txt

# GPU support (Expert Mode)
pip install -r requirements-gpu.txt

# Streaming support (Expert Mode)
pip install kafka-python apache-flink pulsar-client redis

# Incremental learning (Expert Mode)
pip install river vowpalwabbit

# LLM features
pip install openai anthropic langchain

# A/B testing and visualization
pip install scipy matplotlib seaborn

# Model export
pip install onnx onnxruntime skl2onnx sklearn2pmml

# Complete installation with script (All features)
./install_mlops.sh --gpu --airflow --streaming --llm --expert
```

## Expert Mode Guide

### Activation Methods

1. **Command Line Flag**
```bash
python main.py train --expert --data data.csv --target churn
```

2. **Environment Variable**
```bash
export AUTOML_EXPERT_MODE=true
python main.py train --data data.csv --target churn
```

3. **Configuration File**
```yaml
# config.yaml
expert_mode: true
environment: production
```

4. **Web Interface**
- Toggle the "🎓 Mode Expert" checkbox in sidebar
- Or use the toggle in the configuration wizard

### Expert Mode CLI Options

When expert mode is enabled, these additional options become available:

```bash
python main.py train --expert \
    --data data.csv \
    --target churn \
    --algorithms XGBoost,LightGBM,CatBoost,TabNet,FTTransformer \
    --exclude SVM,NaiveBayes \
    --cv-folds 10 \
    --hpo-method optuna \
    --hpo-iter 200 \
    --ensemble stacking \
    --n-workers 16 \
    --gpu \
    --gpu-workers 2 \
    --scoring f1_weighted \
    --feature-selection shap \
    --handle-imbalance smote \
    --calibrate-probabilities
```

### Expert Mode Configuration Examples

#### Example 1: Neural Network Training (Expert Only)
```python
from automl_platform.config import AutoMLConfig

config = AutoMLConfig(expert_mode=True)
config.algorithms = ["TabNet", "FTTransformer", "XGBoost"]
config.include_neural_networks = True
config.hpo_n_iter = 100
config.worker.gpu_workers = 2
config.worker.enable_gpu_queue = True
```

#### Example 2: Distributed Training with Ray (Expert Only)
```python
config = AutoMLConfig(expert_mode=True)
config.worker.backend = "ray"
config.worker.max_workers = 32
config.worker.autoscale_enabled = True
config.worker.autoscale_min_workers = 4
config.worker.autoscale_max_workers = 32
```

#### Example 3: Advanced Feature Engineering (Expert Only)
```python
config = AutoMLConfig(expert_mode=True)
config.create_polynomial = True
config.polynomial_degree = 3
config.create_interactions = True
config.feature_selection_method = "boruta"
config.max_features_generated = 100
```

### Simplified Mode Examples

#### Example 1: Quick Training (Default)
```bash
# Just provide data and target - everything else is automatic
python main.py train --data sales.csv --target revenue
```

#### Example 2: Basic Configuration
```bash
python main.py train \
    --data sales.csv \
    --target revenue \
    --scoring accuracy \
    --output ./results
```

### Mode-Specific API Endpoints

The API automatically adapts based on expert mode:

```python
# Simplified Mode API
POST /api/train/simple
{
    "data_path": "data.csv",
    "target": "churn",
    "metric": "accuracy"
}

# Expert Mode API (requires expert flag or header)
POST /api/train/expert
Headers: {"X-Expert-Mode": "true"}
{
    "data_path": "data.csv",
    "target": "churn",
    "algorithms": ["XGBoost", "LightGBM", "TabNet"],
    "hpo_config": {
        "method": "optuna",
        "n_trials": 200,
        "pruning": true
    },
    "distributed": {
        "backend": "ray",
        "n_workers": 16
    }
}
```

## Quick Start

### 1. Start Services

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Start API server (automatically detects expert mode from env)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit Dashboard (with expert mode toggle)
streamlit run automl_platform/ui/dashboard.py

# Start Redis (for caching and job queue)
redis-server

# Expert Mode: Start additional services
# Start Kafka (for streaming - expert only)
docker-compose up -d kafka zookeeper

# Start Ray cluster (for distributed computing - expert only)
ray start --head --dashboard-host 0.0.0.0

# Start Airflow (for workflow orchestration - expert only)
airflow standalone
```

### 2. Basic AutoML Training

#### Simplified Mode (Default)
```python
from automl_platform.orchestrator import AutoMLOrchestrator
from automl_platform.config import AutoMLConfig
import pandas as pd

# Load data
df = pd.read_csv("your_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Simple configuration
config = AutoMLConfig(expert_mode=False)  # Default
orchestrator = AutoMLOrchestrator(config)

# Train with automatic settings
orchestrator.fit(X, y, task="classification")
predictions = orchestrator.predict(X_test)
```

#### Expert Mode
```python
# Enable expert mode for full control
config = AutoMLConfig(expert_mode=True)

# Configure advanced options
config.algorithms = ["XGBoost", "LightGBM", "CatBoost", "TabNet", "FTTransformer"]
config.hpo_method = "optuna"
config.hpo_n_iter = 200
config.ensemble_method = "stacking"
config.worker.backend = "ray"
config.worker.max_workers = 16
config.worker.enable_gpu_queue = True

# Advanced training
orchestrator = AutoMLOrchestrator(config)
orchestrator.fit(
    X, y,
    task="classification",
    register_best_model=True,
    model_name="customer_churn_expert",
    enable_distributed=True,
    gpu_per_trial=0.5
)

# Export for deployment
orchestrator.export_best_model(
    format="onnx",
    quantize=True,
    output_path="models/churn_model.onnx"
)
```

## Billing Middleware

The billing middleware provides request-level quota enforcement and usage tracking for FastAPI applications.

### Setup Billing Middleware

```python
from fastapi import FastAPI
from automl_platform.api.billing import BillingManager
from automl_platform.api.billing_middleware import (
    BillingMiddleware,
    BillingEnforcer,
    InvoiceGenerator
)

app = FastAPI()

# Initialize billing
billing_manager = BillingManager()

# Add middleware
app.add_middleware(BillingMiddleware, billing_manager=billing_manager)

# Create enforcer for decorators
enforcer = BillingEnforcer(billing_manager)

# Example endpoint with quota enforcement
@app.post("/api/train")
@enforcer.require_quota("models", amount=1)
@enforcer.require_plan(PlanType.STARTER)
async def train_model(request: Request):
    """Endpoint that requires quota and minimum plan"""
    # Training logic here
    return {"status": "training started"}

# Expert mode endpoint (higher quotas)
@app.post("/api/train/expert")
@enforcer.require_quota("models", amount=5)
@enforcer.require_plan(PlanType.PROFESSIONAL)
@enforcer.require_expert_mode()  # New decorator
async def train_expert_model(request: Request):
    """Expert training with advanced options"""
    # Expert training logic
    return {"status": "expert training started"}
```

### Usage Limits by Plan (with Expert Mode)

| Feature | Free | Starter | Professional | Enterprise |
|---------|------|---------|--------------|------------|
| **Simplified Mode** | ✅ | ✅ | ✅ | ✅ |
| **Expert Mode** | ❌ | ❌ | ✅ | ✅ |
| Models | 3 | 10 | 50 | Unlimited |
| Algorithms (Simplified) | 3 | 3 | 3 | 3 |
| Algorithms (Expert) | - | - | 30+ | 30+ |
| Workers | 1 | 2 | 8 | 32 |
| GPU Access | ❌ | ❌ | ✅ | ✅ |
| Predictions/month | 1,000 | 10,000 | 100,000 | Unlimited |
| API calls/day | 100 | 1,000 | 10,000 | Unlimited |
| GPU hours/month | 0 | 0 | 10 | 100 |
| Storage (GB) | 1 | 10 | 100 | 1,000 |
| Team members | 1 | 3 | 10 | Unlimited |
| LLM calls/month | 0 | 100 | 1,000 | Unlimited |
| Price/month | $0 | $49 | $299 | $999 |

## Project Structure

```
automl-platform/
├── main.py                          # Main CLI with expert mode support
├── app.py                           # FastAPI application
├── config.yaml                      # Configuration file
├── requirements.txt                 # Full dependencies (expert mode)
├── requirements-minimal.txt         # Minimal dependencies (simplified mode)
├── requirements-gpu.txt            # GPU dependencies (expert mode only)
├── install_mlops.sh               # Installation script with --expert flag
├── .gitignore                     # Git ignore file
├── DEPLOYMENT_GUIDE.md           # Deployment documentation
├── README.md                      # This file
├── docs/
│   ├── expert-mode.md            # Expert mode documentation
│   ├── simplified-mode.md        # Simplified mode guide
│   └── migration-guide.md        # Migration from v3.0 to v3.1
├── automl_platform/              # Main package
│   ├── config.py                 # Config with expert_mode support
│   ├── orchestrator.py           # AutoML orchestrator
│   ├── mlflow_registry.py        # MLflow model registry integration
│   ├── ab_testing.py             # A/B testing with statistical analysis
│   ├── export_service.py         # Model export (ONNX/PMML/Edge)
│   ├── retraining_service.py     # Automated retraining
│   ├── incremental_learning.py   # Online learning module
│   ├── data_prep.py              # Data preprocessing
│   ├── model_selection.py        # Model selection & HPO
│   ├── monitoring.py             # Advanced monitoring with alerts
│   ├── scheduler.py              # Job scheduling with quotas
│   ├── ui/
│   │   ├── dashboard.py          # Streamlit UI with expert mode toggle
│   │   └── streamlit_ab_testing.py # A/B testing dashboard
│   ├── api/                      # API components
│   │   ├── mlops_endpoints.py    # MLOps REST endpoints
│   │   ├── billing.py            # Billing & subscriptions
│   │   ├── billing_middleware.py # Billing middleware for FastAPI
│   │   ├── llm_endpoints.py      # LLM-powered endpoints
│   │   └── streaming.py          # Streaming ML endpoints
│   └── examples/                  # Examples
│       ├── mlops_integration.py  # Complete MLOps workflow
│       ├── expert_mode_demo.py   # Expert mode examples
│       └── simplified_demo.py    # Simplified mode examples
└── tests/                         # Comprehensive test suite
    ├── test_expert_mode.py        # Expert mode tests
    ├── test_simplified_mode.py   # Simplified mode tests
    └── ...                        # Other test files
```

## Testing

### Comprehensive Test Suite

The platform includes a complete test suite with **81% overall coverage** across all modules, including expert mode tests.

```bash
# Run all tests
pytest tests/ -v --cov=automl_platform

# Run expert mode specific tests
pytest tests/test_expert_mode.py -v

# Run simplified mode tests
pytest tests/test_simplified_mode.py -v

# Run with coverage report
pytest tests/ --cov=automl_platform --cov-report=html --cov-report=term
```

### Test Coverage Summary

| Module | Coverage | Tests | Status |
|--------|----------|-------|--------|
| config.py (with expert mode) | ~90% | 20 | ✅ Implemented |
| main.py (with expert mode) | ~85% | 15 | ✅ Implemented |
| ui/dashboard.py | ~80% | 25 | ✅ Implemented |
| mlflow_registry.py | ~85% | 11 | ✅ Implemented |
| ab_testing.py | ~80% | 22 | ✅ Implemented |
| export_service.py | ~75% | 16 | ✅ Implemented |
| incremental_learning.py | ~80% | 45 | ✅ Implemented |
| api/streaming.py | ~75% | 50 | ✅ Implemented |
| scheduler.py | ~85% | 55 | ✅ Implemented |
| api/billing.py | ~80% | 40 | ✅ Implemented |
| monitoring.py | ~85% | 60 | ✅ Implemented |
| **Overall** | **~81%** | **359** | ✅ Complete |

## Performance

### Benchmarks

| Operation | Simplified Mode | Expert Mode | Notes |
|-----------|----------------|-------------|-------|
| Configuration time | <10s | 1-5 min | Expert has more options |
| Training (100K rows) | ~10 min | ~30 min | Expert tests more models |
| HPO iterations | 20 | 100-500 | Configurable in expert |
| Model count | 3 | 30+ | Expert tests all algorithms |
| Accuracy gain | Baseline | +2-5% | Expert optimization |
| Resource usage | Low | High | Expert uses more workers |
| GPU utilization | 0% | 80%+ | Expert only |

### Optimization Tips

#### For Simplified Mode:
1. Use default settings for quick results
2. Focus on data quality over model tuning
3. Suitable for datasets < 1M rows
4. Best for prototyping and POCs

#### For Expert Mode:
1. **GPU Acceleration**:
   - Use `--gpu` flag for 3-5x speedup
   - Configure `gpu_per_trial` for optimal usage
   - Batch predictions for GPU utilization

2. **Distributed Computing**:
   - Use Ray backend for `n_workers > 4`
   - Configure autoscaling for dynamic workloads
   - Monitor worker utilization

3. **HPO Optimization**:
   - Start with 50 iterations, increase if needed
   - Use pruning for early stopping
   - Enable warm start for iterative training

## Development Workflow

### For Simplified Mode Users

1. Start with simplified mode (default)
2. If results are satisfactory, deploy
3. If not, consider expert mode consultation

### For Expert Mode Users

1. Enable expert mode via flag or environment
2. Start with recommended expert defaults
3. Iteratively refine configuration
4. Monitor resource usage and costs
5. Document optimal settings for reproduction

## Migration Guide

### From v3.0 to v3.1

1. **No breaking changes** - All v3.0 code works in v3.1
2. **Default behavior unchanged** - Simplified mode is default
3. **Expert features are opt-in** - Use `--expert` flag
4. **Configuration files compatible** - Add `expert_mode: true` if needed

### Enabling Expert Features in Existing Projects

```python
# Old code (v3.0)
config = AutoMLConfig()
config.hpo_n_iter = 100  # This still works

# New code (v3.1) - Explicit expert mode
config = AutoMLConfig(expert_mode=True)
config.hpo_n_iter = 200  # Access to higher limits
config.algorithms = ["TabNet", "FTTransformer"]  # Expert algorithms
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Consider both modes when implementing features
4. Write tests for both simplified and expert modes
5. Ensure all tests pass (`pytest tests/`)
6. Update documentation for both modes
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open Pull Request

### Priority Areas for Contribution

- **Mode-specific features**: Enhance either simplified or expert mode
- **Mode switching**: Improve the transition between modes
- **Documentation**: Mode-specific tutorials and guides
- **UI/UX**: Better visualization of mode differences
- **Performance**: Optimize for each mode's use case
- **Templates**: Pre-configured settings for common scenarios

## Changelog

### Version 3.1.0 (2024-01)
- 🆕 **Added Expert Mode**: Dual-mode interface for beginners and experts
- ✅ Added `--expert` flag to CLI for advanced options
- ✅ Added expert mode toggle in web interface
- ✅ Separated algorithm lists for each mode
- ✅ Mode-specific HPO configurations
- ✅ Environment variable support (`AUTOML_EXPERT_MODE`)
- ✅ Mode information saved with trained models
- ✅ Plan-based access control for expert mode
- ✅ Comprehensive mode-specific documentation
- ✅ Mode-aware billing and quotas
- 📝 Updated all documentation with mode information
- 🎯 Backward compatible - no breaking changes

### Version 3.0.1 (2024-12)
- ✅ Complete test suite implementation (299 tests, 81% coverage)
- ✅ Added comprehensive tests for all modules
- ✅ Fixed import issues in various modules
- 📝 Updated documentation with test coverage information

### Version 3.0.0 (2024-01)
- Initial v3.0 release with enterprise features
- Added MLflow integration, A/B testing, streaming ML
- Comprehensive billing system
- LLM-powered features

## Support

- **Documentation**: 
  - [Simplified Mode Guide](https://docs.automl-platform.com/simplified)
  - [Expert Mode Guide](https://docs.automl-platform.com/expert)
  - [Full Documentation](https://automl-platform.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-org/automl-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/automl-platform/discussions)
- **Slack Community**: [Join our Slack](https://automl-community.slack.com)
  - Channel: #simplified-mode for beginners
  - Channel: #expert-mode for advanced users
- **Email**: 
  - General: support@automl-platform.com
  - Expert support: expert@automl-platform.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MLflow for model registry
- River for incremental learning
- Apache Kafka/Flink for streaming
- Optuna for hyperparameter optimization
- ONNX for model interoperability
- FastAPI for REST API framework
- Streamlit for dashboards
- Celery/Ray for distributed processing
- All contributors who suggested the dual-mode interface

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{automl_platform,
  title = {AutoML Platform: Enterprise MLOps Edition with Expert Mode},
  author = {Your Organization},
  year = {2024},
  version = {3.1.0},
  url = {https://github.com/your-org/automl-platform},
  note = {Featuring dual-mode interface for all skill levels}
}
```

---

**Built for everyone: From ML beginners to experts**

*Choose your mode: 🚀 Simplified for quick results | 🎓 Expert for full control*

*Version 3.1.0 - Last updated: January 2024*
