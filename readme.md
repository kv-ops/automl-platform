# AutoML Platform v2.0

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-ready AutoML platform with advanced features including LLM integration, multi-tenant architecture, real-time streaming, and enterprise-grade security.

## Key Features

- **No Data Leakage**: All preprocessing within CV folds using sklearn Pipeline + ColumnTransformer
- **Advanced AutoML**: 30+ models, hyperparameter optimization, ensemble methods
- **LLM Integration**: AI-powered data cleaning, feature suggestions, and model explanations
- **Multi-Tenant**: Secure tenant isolation with billing and resource management
- **Real-Time**: Streaming data processing with Kafka/Pulsar/Redis support
- **Enterprise Ready**: Authentication, monitoring, deployment, and scaling
- **Data Connectors**: Snowflake, BigQuery, Databricks, PostgreSQL, MongoDB
- **Model Explainability**: SHAP, LIME, and natural language explanations
- **REST API**: FastAPI with WebSocket support for real-time updates

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Features](#features)
- [Architecture](#architecture)
- [Development](#development)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.8 or higher
- Redis (for caching and workers)
- Optional: Docker for containerized services

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-org/automl-platform.git
cd automl-platform

# Create virtual environment
python -m venv automl_env
source automl_env/bin/activate  # Linux/Mac
# or automl_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

For specific features, install additional packages:

```bash
# For LLM integration
pip install openai anthropic

# For cloud connectors
pip install snowflake-connector-python google-cloud-bigquery

# For streaming
pip install kafka-python pulsar-client

# For infrastructure
pip install docker kubernetes
```

## Quick Start

### 1. Basic Configuration

Create or modify `config.yaml`:

```yaml
api:
  host: 0.0.0.0
  port: 8000
  enable_auth: false  # Set to true for production

storage:
  backend: local
  local_base_path: ./ml_storage

monitoring:
  enabled: true

llm:
  enabled: false  # Set to true with API keys
```

### 2. Start the API Server

```bash
# Direct execution
python app.py

# Or with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Using the API

#### Upload Data
```bash
curl -X POST "http://localhost:8000/api/v1/data/upload" \
     -F "file=@your_data.csv" \
     -F "dataset_name=my_dataset"
```

#### Start Training
```bash
curl -X POST "http://localhost:8000/api/v1/train" \
     -H "Content-Type: application/json" \
     -d '{
       "dataset_id": "my_dataset",
       "target_column": "target",
       "experiment_name": "my_experiment",
       "algorithms": ["all"],
       "max_runtime_seconds": 3600
     }'
```

#### Make Predictions
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "model_id": "model_123",
       "data": {"feature1": 1.0, "feature2": "value"}
     }'
```

## Project Structure

```
automl-platform/
├── app.py                      # Main FastAPI application
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── automl_platform/           # Main package
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── orchestrator.py        # AutoML orchestrator
│   ├── data_prep.py          # Data preprocessing
│   ├── model_selection.py    # Model selection & HPO
│   ├── metrics.py            # Metrics calculation
│   ├── feature_engineering.py # Feature engineering
│   ├── ensemble.py           # Ensemble methods
│   ├── inference.py          # Model inference
│   ├── storage.py            # Storage management
│   ├── monitoring.py         # Monitoring & drift
│   ├── llm.py               # LLM integration
│   ├── prompts.py           # LLM prompts
│   ├── data_quality_agent.py # Data quality
│   ├── worker.py            # Background workers
│   └── api/                 # API components
│       ├── __init__.py
│       ├── billing.py       # Billing & subscriptions
│       ├── connectors.py    # Data connectors
│       ├── infrastructure.py # Multi-tenant infrastructure
│       ├── llm_endpoints.py # LLM API endpoints
│       └── streaming.py     # Real-time streaming
└── logs/                    # Log files
```

## Usage

### Python API

```python
from automl_platform import AutoMLOrchestrator, AutoMLConfig
import pandas as pd

# Load data
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Configure AutoML
config = AutoMLConfig(
    cv_folds=5,
    hpo_method="optuna",
    algorithms=["all"],
    handle_imbalance=True
)

# Train models
orchestrator = AutoMLOrchestrator(config)
orchestrator.fit(X, y)

# Get results
leaderboard = orchestrator.get_leaderboard()
print(leaderboard)

# Save best model
orchestrator.save_pipeline("model.pkl")
```

### REST API

Access the interactive API documentation at `http://localhost:8000/docs` after starting the server.

## Configuration

### Core Settings

```yaml
# Basic configuration
environment: development
debug: true
random_state: 42

# API settings
api:
  host: 0.0.0.0
  port: 8000
  enable_auth: false
  max_upload_size_mb: 100

# Storage configuration
storage:
  backend: local  # or s3, gcs, minio
  local_base_path: ./ml_storage

# Monitoring
monitoring:
  enabled: true
  drift_detection_enabled: true
  min_quality_score: 70.0
```

### Advanced Configuration

```yaml
# LLM integration
llm:
  enabled: true
  provider: openai
  api_key: sk-your-key
  enable_rag: true
  enable_data_cleaning: true

# Multi-tenant
enable_multi_tenant: true
billing:
  enabled: true

# Workers
worker:
  enabled: true
  backend: celery
  broker_url: redis://localhost:6379/0
```

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| POST | `/api/v1/data/upload` | Upload dataset |
| POST | `/api/v1/train` | Start training |
| GET | `/api/v1/experiments/{id}` | Get experiment status |
| POST | `/api/v1/predict` | Make prediction |
| GET | `/api/v1/models` | List models |

### Advanced Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/features/engineer` | Auto feature engineering |
| POST | `/api/v1/data/{id}/quality` | Data quality check |
| POST | `/api/v1/llm/chat` | Chat with AI assistant |
| POST | `/api/v1/connectors/query` | Query external databases |
| POST | `/api/v1/tenants` | Create tenant |
| WebSocket | `/ws/experiments/{id}` | Real-time updates |

## Features

### AutoML Capabilities

- **30+ Algorithms**: Linear models, trees, boosting, ensembles
- **Hyperparameter Optimization**: Optuna, Grid Search, Random Search
- **Feature Engineering**: Automatic feature generation and selection
- **Ensemble Methods**: Voting, stacking, blending
- **Cross-Validation**: Stratified, time series aware
- **Imbalance Handling**: SMOTE, class weights, cost-sensitive learning

### Data Preprocessing

- **No Data Leakage**: All transformations in sklearn pipelines
- **Missing Value Handling**: Smart imputation strategies
- **Outlier Detection**: IQR, Isolation Forest, statistical methods
- **Feature Encoding**: One-hot, target, ordinal encoding
- **Scaling**: Robust, standard, min-max scaling
- **Text Processing**: TF-IDF, embeddings, NLP features

### Enterprise Features

- **Multi-Tenant**: Secure tenant isolation
- **Authentication**: JWT, API keys, role-based access
- **Monitoring**: Model drift, performance tracking, alerts
- **Billing**: Usage tracking, subscription management
- **Deployment**: Docker, Kubernetes, cloud-native
- **Streaming**: Real-time data processing
- **Explainability**: SHAP, LIME, natural language explanations

## Architecture

### Core Components

1. **Orchestrator**: Main AutoML engine
2. **Data Preprocessor**: Feature engineering pipeline
3. **Model Selector**: Algorithm selection and HPO
4. **Storage Manager**: Model and data persistence
5. **Monitoring Service**: Performance and drift tracking
6. **LLM Assistant**: AI-powered features

### Design Principles

- **Modular Architecture**: Loosely coupled components
- **Pipeline-First**: No data leakage by design
- **Configuration-Driven**: YAML-based configuration
- **Async-First**: Non-blocking operations
- **Cloud-Native**: Containerized, scalable
- **API-First**: Everything accessible via REST API

## Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest tests/ -v

# Format code
black automl_platform/

# Type checking
mypy automl_platform/

# Linting
flake8 automl_platform/
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=automl_platform

# Specific test file
pytest tests/test_orchestrator.py -v
```

### Docker Development

```bash
# Build image
docker build -t automl-platform .

# Run container
docker run -p 8000:8000 automl-platform

# With docker-compose
docker-compose up -d
```

## Monitoring and Observability

### Metrics

The platform exposes Prometheus metrics at `/metrics`:

- Request counts and latencies
- Model performance metrics
- Resource usage
- Error rates

### Logging

Structured logging to files and stdout:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Health Checks

Multiple health check endpoints:

- `/health` - Basic health
- `/api/v1/status` - Detailed system status

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: Full docs at `/docs` endpoint
- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions
- **Email**: support@automl-platform.com

---

**Built for production ML workflows with enterprise-grade features**
