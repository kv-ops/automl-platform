# AutoML Platform - Optional Dependencies Guide

## Overview

This document describes all optional dependencies (extras) available for the AutoML Platform v3.2.1.
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

## NEW: Intelligent Agents Extra

### `[agents]` - AI-Powered Data Cleaning
Intelligent data cleaning with OpenAI GPT-4:
- OpenAI API integration
- Web scraping for validation standards
- Token counting and cost management

```bash
pip install automl-platform[agents]
```

## GPU-Related Extras

### `[gpu]` - Core GPU Support
Essential GPU libraries for acceleration:
- **CUDA array processing**: cupy for NumPy-like operations on GPU
- **CUDA kernel programming**: pycuda for custom CUDA kernels
- **GPU compilation**: numba for JIT compilation to CUDA
- **GPU monitoring**: gputil, nvidia-ml-py3, pynvml, gpustat
- **Memory profiling**: pytorch-memlab, torch-tb-profiler
- **Optimized inference**: onnxruntime-gpu

**Note**: This extra provides GPU infrastructure but NOT deep learning frameworks. For PyTorch/TensorFlow, add `[deep]` or use `[gpu-complete]`.

```bash
pip install automl-platform[gpu]
```

### `[deep]` - Deep Learning Frameworks
Major deep learning frameworks with GPU support:
- PyTorch ecosystem (torch, torchvision, torchaudio)
- TensorFlow
- PyTorch Lightning
- PyTorch TabNet
- Transformers

```bash
pip install automl-platform[deep]
```

### `[gpu,deep]` - Recommended GPU Setup
**Best practice**: Install both for complete GPU-accelerated deep learning:
```bash
pip install automl-platform[gpu,deep]
```
This combination provides:
- All GPU infrastructure and monitoring
- All deep learning frameworks
- Optimal compatibility and performance

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

```bash
pip install automl-platform[serving-gpu]
```

### `[gpu-complete]` - All GPU Features
Installs ALL GPU-related extras (recommended for research/experimentation):
```bash
pip install automl-platform[gpu-complete]
```
Includes: gpu, deep, distributed-gpu, automl-gpu, serving-gpu

## Core Extras

### `[auth]` - Authentication
Enhanced authentication support:
- Keycloak integration
- SAML support
- Okta integration
- Microsoft authentication (MSAL)
- JOSE/JWT utilities
- OAuth utilities

### `[sso]` - Single Sign-On
SSO provider integrations:
- Keycloak
- SAML
- Okta
- OAuth/OIDC support

### `[api]` - Enhanced API Features
Production API server features:
- WebSocket support
- Rate limiting
- Socket.IO
- API versioning

### `[monitoring]` - Observability
Monitoring and observability tools:
- OpenTelemetry
- Prometheus metrics
- Sentry error tracking
- Datadog integration
- Jaeger distributed tracing

### `[distributed]` - Distributed Computing
Distributed processing frameworks:
- Ray for distributed ML
- Dask for parallel computing
- Dramatiq for task queues

## ML/Data Science Extras

### `[explain]` - Model Explainability
Model interpretation tools:
- SHAP
- LIME
- ELI5
- InterpretML
- Alibi
- Captum

### `[timeseries]` - Time Series Analysis
Time series forecasting:
- Prophet
- statsmodels
- pmdarima
- sktime
- tsfresh
- darts
- neuralforecast

### `[nlp]` - Natural Language Processing
NLP libraries:
- Sentence Transformers
- spaCy
- NLTK
- Gensim
- TextBlob
- Language detection

### `[vision]` - Computer Vision
Computer vision tools:
- OpenCV
- Pillow
- Albumentations
- Supervision
- Ultralytics (YOLO)

## Infrastructure Extras

### `[cloud]` - Cloud Providers
Cloud storage and compute:
- AWS (boto3, s3fs)
- Google Cloud (BigQuery, Storage)
- Azure (Blob Storage, Identity)
- Snowflake
- Databricks (databricks-sql-connector for SQL)

### `[connectors]` - Extended Data Connectors
CRM and database connectors:
- HubSpot, Salesforce, Pipedrive, Zoho CRM
- Oracle, MongoDB, Cassandra, Elasticsearch
- InfluxDB, MySQL
- Advanced Excel and Google Sheets
- Databricks (databricks-connect for Spark)
- Snowflake, BigQuery (also available here for data access)

### `[streaming]` - Stream Processing
Real-time data processing:
- Kafka (multiple clients)
- Pulsar
- Redis Streams
- Faust

### `[orchestration]` - Workflow Orchestration
Workflow management:
- Apache Airflow
- Prefect
- Dagster
- Kedro
- Luigi

### `[mlops]` - MLOps Tools
ML lifecycle management:
- DVC
- Weights & Biases
- Neptune
- BentoML
- Great Expectations

### `[production]` - Production Deployment
Production deployment tools:
- Docker
- Kubernetes
- Nginx

## Development Extras

### `[dev]` - Development Tools
Testing and code quality:
- pytest plugins (mock, benchmark)
- Faker, Factory Boy
- pre-commit, Bandit, Safety
- Locust for load testing

### `[docs]` - Documentation
Documentation generation:
- Sphinx with extensions
- MkDocs with Material theme
- Jupyter Book

## Installation Profiles

### `[agents]` - Intelligent Data Cleaning (NEW)
```bash
pip install automl-platform[agents]
```
AI-powered data cleaning with OpenAI GPT-4

### `[nocode]` - No-Code Experience
```bash
pip install automl-platform[nocode]
```
Complete no-code experience with UI, connectors, and reporting

### `[enterprise]` - Enterprise Deployment
```bash
pip install automl-platform[enterprise]
```
Production-ready enterprise deployment with all infrastructure

### `[gpu-complete]` - Complete GPU Stack
```bash
pip install automl-platform[gpu-complete]
```
All GPU-related features and frameworks

### `[all]` - Complete Installation
```bash
pip install automl-platform[all]
```
Everything - all available extras

## Python Version Compatibility

| Python Version | Status | Notes |
|---------------|--------|-------|
| 3.9           | ✅ Fully Supported | Requires `tomli` for TOML parsing |
| 3.10          | ✅ Fully Supported | Requires `tomli` for TOML parsing |
| 3.11          | ✅ Fully Supported | Uses built-in `tomllib` |
| 3.12          | ✅ Fully Supported | Uses built-in `tomllib` |
| 3.13+         | ❌ Not Yet Supported | Pending dependency updates |

## GPU Setup Requirements

### Prerequisites
1. **NVIDIA GPU** with CUDA Capability >= 3.5
2. **CUDA Toolkit** 11.8+ (12.1 for newer features)
3. **cuDNN** 8.6+
4. **NVIDIA Driver** >= 450.80.02

### Installation Options

#### Option 1: Core GPU Support Only
```bash
pip install automl-platform[gpu]
```
**Includes**: CUDA libraries, GPU monitoring, memory optimization  
**Use case**: GPU computation without deep learning

#### Option 2: GPU + Deep Learning (RECOMMENDED)
```bash
pip install automl-platform[gpu,deep]
```
**Includes**: Everything from [gpu] plus PyTorch, TensorFlow, and DL frameworks  
**Use case**: Standard GPU-accelerated machine learning

#### Option 3: Complete GPU Stack
```bash
pip install automl-platform[gpu-complete]
```
**Includes**: All GPU features including distributed training and AutoML  
**Use case**: Research, experimentation, or maximum capability

### Verification
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
# Check GPU memory
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"GPU Memory: {info.free / 1024**3:.1f} GB free / {info.total / 1024**3:.1f} GB total")
```

## Recommended Installation Patterns

### For Data Scientists
```bash
# Standard ML with GPU support
pip install automl-platform[gpu,deep,explain,viz]

# With intelligent agents
pip install automl-platform[gpu,deep,explain,viz,agents]
```

### For ML Engineers
```bash
# Production deployment
pip install automl-platform[gpu,deep,monitoring,mlops,api]

# With distributed training
pip install automl-platform[gpu-complete,monitoring,mlops,api]
```

### For Business Users
```bash
# No-code with intelligent cleaning
pip install automl-platform[nocode,agents]
```

### For Researchers
```bash
# Everything GPU-related
pip install automl-platform[gpu-complete,explain,timeseries,nlp,vision]
```

### For Enterprise
```bash
# Complete enterprise deployment
pip install automl-platform[enterprise]

# Enterprise with GPU
pip install automl-platform[enterprise,gpu-complete]
```

## Version Compatibility Matrix

| Python | CUDA | PyTorch | TensorFlow | cupy | Notes |
|--------|------|---------|------------|------|-------|
| 3.9-3.10 | 11.8 | 2.1.x | 2.15.x | 12.x | Requires `tomli` |
| 3.11-3.12 | 11.8 | 2.1.x | 2.15.x | 12.x | Built-in `tomllib` |
| 3.10-3.12 | 12.1 | 2.2.x | 2.16.x | 13.x | Latest features |

## Troubleshooting

### CUDA Version Mismatch
If you encounter CUDA version issues:
```bash
# Check system CUDA version
nvidia-smi

# Install specific PyTorch version for your CUDA
# CUDA 11.8
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 12.1
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
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

# Set memory fraction for TensorFlow
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
```

### PyCUDA Installation Issues
PyCUDA requires CUDA development headers. If installation fails:
```bash
# Ubuntu/Debian
sudo apt-get install cuda-toolkit-11-8 python3-dev

# CentOS/RHEL
sudo yum install cuda-toolkit-11-8 python3-devel

# Install PyCUDA with specific CUDA path
export CUDA_ROOT=/usr/local/cuda-11.8
export PATH=$CUDA_ROOT/bin:$PATH
pip install pycuda

# Alternative: Use conda for easier installation
conda install -c conda-forge pycuda
```

### TOML Parsing on Python 3.9/3.10
The platform automatically handles TOML parsing compatibility:
- Python 3.11+: Uses built-in `tomllib`
- Python 3.9-3.10: Automatically installs and uses `tomli`

If you encounter issues with `generate_requirements.py`:
```bash
pip install tomli>=2.0.1
```

## Performance Optimization Tips

### GPU Utilization
```python
# Monitor GPU usage during training
import gpustat
gpustat.print_gpustat()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### Multi-GPU Setup
```python
# DataParallel for single node
model = torch.nn.DataParallel(model)

# DistributedDataParallel for multiple nodes
import torch.distributed as dist
dist.init_process_group("nccl")
model = torch.nn.parallel.DistributedDataParallel(model)
```

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/automl-platform/automl-platform/issues
- **Documentation**: https://docs.automl-platform.com
- **Community Slack**: https://automl-platform.slack.com
- **GPU Setup Guide**: https://docs.automl-platform.com/gpu-setup

## Version History

- **v3.2.1** (Current): Added intelligent agents extra, harmonized dependencies, fixed Python 3.9/3.10 compatibility, added PyCUDA to GPU extras
- **v3.2.0**: Extended connectors, no-code UI enhancements
- **v3.1.0**: GPU support, distributed training
- **v3.0.0**: Enterprise features, SSO, GDPR compliance

## Quick Reference

| Task | Recommended Installation |
|------|-------------------------|
| Basic ML | `pip install automl-platform` |
| ML with GPU | `pip install automl-platform[gpu,deep]` |
| No-code UI | `pip install automl-platform[nocode]` |
| Intelligent cleaning | `pip install automl-platform[agents]` |
| Enterprise | `pip install automl-platform[enterprise]` |
| Everything | `pip install automl-platform[all]` |
