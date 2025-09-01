# AutoML Platform v3.0 - Enterprise MLOps Edition

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-2.9%2B-0194E2)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)
[![ONNX](https://img.shields.io/badge/ONNX-1.15%2B-5C5C5C)](https://onnx.ai/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-ready AutoML platform with enterprise MLOps capabilities including model registry, A/B testing, automated retraining, and multi-format model export.

## 🚀 New in v3.0 - MLOps Features

- **MLflow Integration**: Complete model registry with versioning and stages
- **A/B Testing**: Statistical testing for model comparison in production
- **Automated Retraining**: Drift-triggered retraining with Airflow/Prefect
- **Model Export**: ONNX, PMML, and edge deployment packages
- **Advanced Monitoring**: Drift detection, performance tracking, alerting

## Key Features

### Core AutoML
- **No Data Leakage**: All preprocessing within CV folds using sklearn Pipeline
- **30+ Algorithms**: Including XGBoost, LightGBM, CatBoost, Neural Networks
- **Hyperparameter Optimization**: Optuna, Grid Search, Random Search
- **Ensemble Methods**: Voting, stacking, blending with meta-learners
- **Imbalance Handling**: SMOTE, class weights, focal loss

### MLOps & Production
- **Model Registry**: MLflow-based versioning with promotion stages
- **A/B Testing**: Built-in statistical significance testing
- **Automated Retraining**: Schedule-based and drift-triggered
- **Model Export**: ONNX (with quantization), TFLite, CoreML
- **Edge Deployment**: Optimized packages for IoT and mobile

### Enterprise Features
- **Multi-Tenant**: Secure tenant isolation with resource management
- **Authentication**: JWT, API keys, SSO support (Keycloak/Auth0)
- **Real-Time Processing**: Kafka/Pulsar streaming integration
- **Data Connectors**: Snowflake, BigQuery, Databricks, MongoDB
- **LLM Integration**: GPT-4/Claude for insights and explanations

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [MLOps Workflow](#mlops-workflow)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.8 or higher
- Redis (for caching and workers)
- MLflow (for model registry)
- Optional: Docker, Airflow/Prefect

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/your-org/automl-platform.git
cd automl-platform

# Installation rapide (recommandé)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements-minimal.txt

# Or use the installation script
./install_mlops.sh
```

### Installation Options

```bash
# Basic MLOps installation (recommended)
pip install -r requirements-minimal.txt

# Full installation (may have conflicts)
pip install -r requirements.txt

# GPU support
pip install -r requirements-gpu.txt

# With installation script
./install_mlops.sh [--gpu] [--airflow|--prefect]
```

## Quick Start

### 1. Start Services

```bash
# Start MLflow server
./start_mlflow.sh

# Start API server
./start_api.sh

# Start Dashboard
./start_dashboard.sh

# Start Airflow (if installed)
airflow standalone
```

### 2. Train and Register Model

```python
from automl_platform.orchestrator import AutoMLOrchestrator
from automl_platform.config import AutoMLConfig
import pandas as pd

# Configure
config = AutoMLConfig()
config.mlflow_tracking_uri = "http://localhost:5000"

# Train with automatic registration
orchestrator = AutoMLOrchestrator(config)
orchestrator.fit(
    X_train, y_train,
    register_best_model=True,
    model_name="customer_churn"
)

# Export for deployment
orchestrator.export_best_model(
    format="onnx",
    quantize=True
)
```

### 3. A/B Testing

```python
from automl_platform.mlflow_registry import ABTestingService

# Create A/B test
ab_service = ABTestingService(registry)
test_id = ab_service.create_ab_test(
    model_name="customer_churn",
    champion_version=1,
    challenger_version=2,
    traffic_split=0.2
)

# Get results with statistical analysis
results = ab_service.get_test_results(test_id)
print(f"P-value: {results['statistical_significance']['p_value']}")

# Auto-promote winner if significant
ab_service.conclude_test(test_id, promote_winner=True)
```

## MLOps Workflow

### Complete MLOps Pipeline

```bash
# Run the complete MLOps example
python automl_platform/examples/mlops_integration.py
```

This demonstrates:
1. Model training and registration
2. Version management with MLflow
3. A/B testing with statistical significance
4. Automated retraining setup
5. Model export for edge deployment

### Model Lifecycle

```
Development → Staging → A/B Test → Production → Monitoring → Retraining
     ↓           ↓          ↓           ↓            ↓            ↓
  MLflow     Validation  Statistics  Serving    Drift Check   Trigger
```

## Project Structure

```
automl-platform/
├── app.py                          # Main FastAPI application
├── config.yaml                     # Configuration file
├── requirements.txt                # Full dependencies (reference)
├── requirements-minimal.txt       # Minimal dependencies (use this)
├── requirements-gpu.txt           # GPU dependencies
├── install_mlops.sh              # Installation script
├── .gitignore                     # Git ignore file
├── DEPLOYMENT_GUIDE.md           # Deployment documentation
├── automl_platform/              # Main package
│   ├── orchestrator.py          # AutoML orchestrator (enhanced)
│   ├── mlflow_registry.py       # MLflow integration
│   ├── retraining_service.py    # Automated retraining
│   ├── export_service.py        # Model export (ONNX/PMML)
│   ├── data_prep.py             # Data preprocessing
│   ├── model_selection.py       # Model selection & HPO
│   ├── monitoring.py            # Drift detection
│   ├── scheduler.py             # Job scheduling with Celery
│   ├── api/                     # API components
│   │   ├── mlops_endpoints.py  # MLOps REST endpoints
│   │   ├── billing.py          # Billing & subscriptions
│   │   ├── billing-middleware.py # Billing middleware
│   │   └── llm_endpoints.py    # LLM endpoints
│   └── examples/                # Examples
│       └── mlops_integration.py # Complete MLOps workflow
└── tests/                       # Test suite
```

## API Documentation

### MLOps Endpoints

Access interactive API docs at `http://localhost:8000/docs`

#### Model Registry

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/mlops/models/register` | Register new model |
| POST | `/api/v1/mlops/models/promote` | Promote model stage |
| GET | `/api/v1/mlops/models/{name}/versions` | Get version history |
| POST | `/api/v1/mlops/models/{name}/rollback` | Rollback to previous |

#### A/B Testing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/mlops/ab-tests/create` | Create A/B test |
| GET | `/api/v1/mlops/ab-tests/{id}/results` | Get test results |
| POST | `/api/v1/mlops/ab-tests/{id}/conclude` | Conclude and promote |

#### Model Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/mlops/models/export` | Export model (ONNX) |
| GET | `/api/v1/mlops/models/export/{name}/{version}/download` | Download exported |

#### Automated Retraining

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/mlops/retraining/check` | Check if retraining needed |
| POST | `/api/v1/mlops/retraining/trigger/{name}` | Trigger manual retraining |
| POST | `/api/v1/mlops/retraining/schedule` | Create schedule |

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/train` | Start training |
| POST | `/api/v1/predict` | Make predictions |
| GET | `/api/v1/experiments/{id}` | Get experiment status |
| WebSocket | `/ws/experiments/{id}` | Real-time updates |

## Configuration

### MLOps Configuration

```yaml
# MLflow settings
mlflow_tracking_uri: "http://localhost:5000"
enable_ab_testing: true

# Retraining configuration
retraining:
  drift_threshold: 0.3
  performance_degradation_threshold: 0.1
  check_frequency: "daily"
  min_data_points: 1000

# Model export
export:
  enable_onnx: true
  enable_quantization: true
  optimize_for_edge: true

# Workflow orchestration
orchestration:
  backend: "prefect"  # or "airflow"
  dag_directory: "~/airflow/dags"
```

### Full Configuration Example

```yaml
# Environment
environment: production
debug: false

# API
api:
  host: 0.0.0.0
  port: 8000
  enable_auth: true
  enable_rate_limit: true

# Storage
storage:
  backend: minio
  endpoint: localhost:9000
  models_bucket: models
  datasets_bucket: datasets

# Monitoring
monitoring:
  enabled: true
  drift_detection_enabled: true
  prometheus_enabled: true
  alerting_enabled: true

# Billing
billing:
  enabled: true
  plan_type: enterprise
```

## Development

### Setting up Development Environment

```bash
# Install dev dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/ -v --cov=automl_platform

# Format code
black automl_platform/

# Type checking
mypy automl_platform/

# Linting
flake8 automl_platform/
```

### Running MLOps Tests

```bash
# Test MLflow integration
pytest tests/test_mlflow_registry.py -v

# Test A/B testing
pytest tests/test_ab_testing.py -v

# Test model export
pytest tests/test_export_service.py -v

# Integration tests
pytest tests/integration/test_mlops_workflow.py -v
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t automl-platform:latest .

# Run with docker-compose
docker-compose up -d

# Scale workers
docker-compose scale worker=4
```

### Kubernetes Deployment

```bash
# Deploy with Helm
helm install automl-platform ./helm-chart

# Configure autoscaling
kubectl autoscale deployment automl-api --min=2 --max=10 --cpu-percent=70
```

### Cloud Deployment

```bash
# AWS ECS
ecs-cli compose up

# Google Cloud Run
gcloud run deploy automl-platform --image gcr.io/project/automl

# Azure Container Instances
az container create --resource-group automl --name platform
```

## Monitoring

### Metrics

Prometheus metrics available at `/metrics`:

- `model_predictions_total` - Total predictions
- `model_drift_score` - Current drift score
- `model_accuracy` - Model accuracy in production
- `retraining_runs_total` - Number of retrainings
- `ab_test_conversions` - A/B test conversion rates

### Dashboards

- **MLflow UI**: `http://localhost:5000` - Model registry and experiments
- **Airflow UI**: `http://localhost:8080` - Workflow orchestration
- **API Docs**: `http://localhost:8000/docs` - Interactive API
- **Streamlit**: `http://localhost:8501` - Custom dashboard

## Performance

### Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Training (1M rows) | ~30 min | 10 models with HPO |
| Prediction (1K batch) | <100ms | With preprocessing |
| ONNX Export | <5s | With quantization |
| Model Registration | <1s | MLflow backend |
| A/B Test Analysis | <500ms | Chi-square test |

### Optimization Tips

1. **Use GPU for XGBoost/LightGBM**: 3-5x speedup
2. **Enable ONNX quantization**: 75% size reduction
3. **Use batch predictions**: 10x throughput
4. **Cache preprocessing**: 50% latency reduction

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards

- Follow PEP 8
- Add type hints
- Write docstrings
- Include unit tests
- Update documentation

## Support

- **Documentation**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/your-org/automl-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/automl-platform/discussions)
- **Email**: support@automl-platform.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MLflow for model registry
- Optuna for hyperparameter optimization
- ONNX for model interoperability
- FastAPI for REST API framework

---

**Built for enterprise ML workflows with production-ready MLOps capabilities**

*Version 3.0 - Last updated: 2024*
