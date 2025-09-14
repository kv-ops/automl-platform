# AutoML Platform - Optional Dependencies Guide

## Overview

This document describes all optional dependencies (extras) available for the AutoML Platform.
The authoritative source for dependency versions is `pyproject.toml`.

## Installation

### Basic Installation
```bash
pip install automl-platform
```

### With Optional Dependencies
```bash
# Single extra
pip install automl-platform[gpu]

# Multiple extras
pip install automl-platform[gpu,deep,monitoring]

# All dependencies
pip install automl-platform[all]
```

## GPU-Related Extras

### `[gpu]` - Core GPU Support
Essential GPU libraries for acceleration:
- CUDA array processing (cupy, pycuda)
- GPU monitoring (gputil, nvidia-ml-py3)
- Memory profiling (pytorch-memlab)
- Optimized inference (onnxruntime-gpu)

```bash
pip install automl-platform[gpu]
```

### `[deep]` - Deep Learning Frameworks
Major deep learning frameworks with GPU support:
- PyTorch ecosystem (torch, torchvision, torchaudio)
- TensorFlow
- PyTorch Lightning
- Transformers

```bash
pip install automl-platform[deep]
```

### `[distributed-gpu]` - Distributed GPU Training
Advanced distributed training frameworks:
- Horovod - Uber's distributed training framework
- FairScale - Facebook's distributed training library
- DeepSpeed - Microsoft's optimization library

```bash
pip install automl-platform[distributed-gpu]
```

### `[automl-gpu]` - AutoML with GPU
AutoML frameworks with GPU acceleration:
- AutoGluon with PyTorch backend
- NNI (Neural Network Intelligence)

```bash
pip install automl-platform[automl-gpu]
```

### `[serving-gpu]` - GPU Inference Serving
Production inference optimization:
- Triton Inference Server client
- TensorRT (requires manual installation)
- Torch-TensorRT

```bash
pip install automl-platform[serving-gpu]
```

### `[gpu-complete]` - All GPU Features
Installs all GPU-related extras:
```bash
pip install automl-platform[gpu-complete]
```

## Core Extras

### `[auth]` - Authentication
Enhanced authentication support:
- Keycloak integration
- SAML support
- Okta integration
- JOSE/JWT utilities

### `[sso]` - Single Sign-On
SSO provider integrations:
- Keycloak
- SAML
- Okta
- OAuth/OIDC support

### `[api]` - Enhanced API Features
Production API server features:
- Async file handling
- WebSocket support
- Rate limiting
- Production WSGI server (Gunicorn)

### `[monitoring]` - Observability
Monitoring and observability tools:
- OpenTelemetry
- Prometheus metrics
- Sentry error tracking
- Datadog integration
- Evidently for ML monitoring

### `[distributed]` - Distributed Computing
Distributed processing frameworks:
- Ray for distributed ML
- Dask for parallel computing
- Celery for task queues

## ML/Data Science Extras

### `[explain]` - Model Explainability
Model interpretation tools:
- SHAP
- LIME
- ELI5
- InterpretML

### `[timeseries]` - Time Series Analysis
Time series forecasting:
- Prophet
- statsmodels
- pmdarima
- sktime
- darts

### `[nlp]` - Natural Language Processing
NLP libraries:
- Sentence Transformers
- spaCy
- NLTK
- Gensim

### `[vision]` - Computer Vision
Computer vision tools:
- OpenCV
- Pillow
- Albumentations

## Infrastructure Extras

### `[cloud]` - Cloud Providers
Cloud storage and compute:
- AWS (boto3, s3fs)
- Google Cloud (BigQuery, Storage)
- Azure (Blob Storage)
- Snowflake
- Databricks

### `[streaming]` - Stream Processing
Real-time data processing:
- Kafka
- Pulsar
- Redis Streams
- Faust

### `[orchestration]` - Workflow Orchestration
Workflow management:
- Apache Airflow
- Prefect
- Dagster
- Kedro

### `[mlops]` - MLOps Tools
ML lifecycle management:
- DVC
- Weights & Biases
- Neptune
- BentoML
- Great Expectations

## Development Extras

### `[dev]` - Development Tools
Testing and code quality:
- pytest and plugins
- Black, Ruff, mypy
- pre-commit
- Security scanners

### `[docs]` - Documentation
Documentation generation:
- Sphinx
- MkDocs
- Jupyter Book

## Installation Profiles

### Enterprise Deployment
```bash
pip install automl-platform[enterprise]
```
Includes: api, storage, distributed, mlops, monitoring, auth, sso, orchestration, export, streaming, cloud

### Complete Installation
```bash
pip install automl-platform[all]
```
Includes all available extras.

## GPU Setup Requirements

### Prerequisites
1. NVIDIA GPU with CUDA Capability >= 3.5
2. CUDA Toolkit 11.8
3. cuDNN 8.6+
4. NVIDIA Driver >= 450.80.02

### Verification
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

## Version Compatibility Matrix

| Python | CUDA | PyTorch | TensorFlow |
|--------|------|---------|------------|
| 3.9+   | 11.8 | 2.1.x   | 2.15.x     |
| 3.10+  | 12.1 | 2.2.x   | 2.16.x     |

## Troubleshooting

### CUDA Version Mismatch
If you encounter CUDA version issues:
```bash
# Check system CUDA version
nvidia-smi

# Install specific PyTorch version
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### Memory Issues
For large models or datasets:
```python
# Set memory growth for TensorFlow
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Clear PyTorch cache
import torch
torch.cuda.empty_cache()
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/automl-platform/automl-platform/issues
- Documentation: https://docs.automl-platform.com
- Community Slack: https://automl-platform.slack.com