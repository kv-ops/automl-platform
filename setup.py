"""
Setup script for AutoML Platform
Version 3.0.0 - Enterprise Edition with MLOps and Distributed Computing
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read version from version file
version_file = this_directory / "automl_platform" / "__version__.py"
version = "3.0.0"
if version_file.exists():
    with open(version_file) as f:
        exec(f.read())

# Core requirements (minimal set for basic functionality)
install_requires = [
    # Core ML libraries
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "joblib>=1.3.0",
    
    # Configuration
    "pyyaml>=6.0.1",
    "python-dotenv>=1.0.0",
    
    # Data validation
    "pydantic>=2.4.0",
    "marshmallow>=3.20.0",
    
    # Utilities
    "tqdm>=4.65.0",
    "click>=8.1.0",
    "rich>=13.5.0",
]

# Optional dependencies organized by feature
extras_require = {
    # Hyperparameter optimization
    "hpo": [
        "optuna>=3.4.0",
        "optuna-dashboard>=0.13.0",
        "hyperopt>=0.2.7",
        "scikit-optimize>=0.9.0",
    ],
    
    # Boosting algorithms
    "boosting": [
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
    ],
    
    # Deep learning
    "deep": [
        "tensorflow>=2.13.0",
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.30.0",
    ],
    
    # Imbalanced learning
    "imbalance": [
        "imbalanced-learn>=0.11.0",
        "smote-variants>=0.4.0",
    ],
    
    # Model explainability
    "explain": [
        "shap>=0.43.0",
        "lime>=0.2.0",
        "eli5>=0.13.0",
        "interpret>=0.4.0",
        "alibi>=0.9.0",
    ],
    
    # Time series
    "timeseries": [
        "statsmodels>=0.14.0",
        "prophet>=1.1.0",
        "pmdarima>=2.0.0",
        "sktime>=0.24.0",
        "tsfresh>=0.20.0",
        "darts>=0.25.0",
    ],
    
    # NLP
    "nlp": [
        "sentence-transformers>=2.2.0",
        "nltk>=3.8.0",
        "spacy>=3.6.0",
        "gensim>=4.3.0",
        "textblob>=0.17.0",
    ],
    
    # Computer Vision
    "vision": [
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "albumentations>=1.3.0",
        "torchvision>=0.15.0",
    ],
    
    # API & Web Framework
    "api": [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "gunicorn>=21.2.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.0",
        "httpx>=0.25.0",
        "websockets>=12.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "slowapi>=0.1.9",
    ],
    
    # Database & Storage
    "storage": [
        "sqlalchemy>=2.0.0",
        "alembic>=1.12.0",
        "psycopg2-binary>=2.9.0",
        "pymongo>=4.5.0",
        "redis>=5.0.0",
        "minio>=7.1.0",
        "boto3>=1.28.0",
        "google-cloud-storage>=2.10.0",
        "azure-storage-blob>=12.19.0",
    ],
    
    # Distributed Computing
    "distributed": [
        "ray[default]>=2.7.0",
        "dask[complete]>=2023.10.0",
        "celery[redis]>=5.3.0",
        "flower>=2.0.0",
        "apache-beam>=2.50.0",
    ],
    
    # MLOps & Model Management
    "mlops": [
        "mlflow>=2.8.0",
        "dvc>=3.30.0",
        "wandb>=0.15.0",
        "neptune-client>=1.8.0",
        "bentoml>=1.1.0",
        "evidently>=0.4.0",
        "great-expectations>=0.17.0",
    ],
    
    # Workflow Orchestration
    "orchestration": [
        "apache-airflow>=2.7.0",
        "prefect>=2.14.0",
        "dagster>=1.5.0",
        "kedro>=0.18.0",
    ],
    
    # Model Export & Serving
    "export": [
        "onnx>=1.15.0",
        "onnxruntime>=1.16.0",
        "sklearn2pmml>=0.100.0",
        "tensorflow-lite>=2.13.0",
        "coremltools>=7.0",
        "tensorrt>=8.6.0",
    ],
    
    # Streaming & Real-time
    "streaming": [
        "kafka-python>=2.0.0",
        "confluent-kafka>=2.2.0",
        "pulsar-client>=3.3.0",
        "redis-py-cluster>=2.1.0",
        "faust-streaming>=0.10.0",
    ],
    
    # Feature Store
    "feature_store": [
        "feast>=0.35.0",
        "tecton>=0.7.0",
        "featuretools>=1.27.0",
    ],
    
    # Data Connectors
    "connectors": [
        "snowflake-connector-python>=3.3.0",
        "google-cloud-bigquery>=3.13.0",
        "databricks-connect>=13.0.0",
        "pyodbc>=5.0.0",
        "cx_Oracle>=8.3.0",
        "cassandra-driver>=3.28.0",
        "elasticsearch>=8.10.0",
    ],
    
    # Monitoring & Observability
    "monitoring": [
        "prometheus-client>=0.18.0",
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-instrumentation-fastapi>=0.41b0",
        "jaeger-client>=4.8.0",
        "sentry-sdk>=1.35.0",
    ],
    
    # Security & Auth
    "security": [
        "python-keycloak>=3.7.0",
        "authlib>=1.2.0",
        "python-saml>=1.15.0",
        "cryptography>=41.0.0",
        "pycryptodome>=3.19.0",
    ],
    
    # LLM Integration
    "llm": [
        "openai>=1.3.0",
        "anthropic>=0.7.0",
        "langchain>=0.0.340",
        "llama-index>=0.9.0",
        "chromadb>=0.4.0",
        "tiktoken>=0.5.0",
    ],
    
    # Visualization
    "viz": [
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "altair>=5.1.0",
        "bokeh>=3.3.0",
        "holoviews>=1.18.0",
        "panel>=1.3.0",
        "streamlit>=1.28.0",
    ],
    
    # Development & Testing
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.12.0",
        "pytest-benchmark>=4.0.0",
        "hypothesis>=6.90.0",
        "faker>=20.0.0",
        "factory-boy>=3.3.0",
        "black>=23.11.0",
        "ruff>=0.1.0",
        "mypy>=1.7.0",
        "isort>=5.12.0",
        "pre-commit>=3.5.0",
        "bandit>=1.7.0",
        "safety>=2.3.0",
    ],
    
    # Documentation
    "docs": [
        "sphinx>=7.2.0",
        "sphinx-rtd-theme>=1.3.0",
        "sphinx-autodoc-typehints>=1.24.0",
        "sphinx-copybutton>=0.5.0",
        "myst-parser>=2.0.0",
        "jupyter-book>=0.15.0",
    ],
    
    # Profiling & Optimization
    "profiling": [
        "memory-profiler>=0.61.0",
        "line-profiler>=4.1.0",
        "py-spy>=0.3.0",
        "scalene>=1.5.0",
        "yappi>=1.4.0",
    ],
}

# Combine all extras for complete installation
all_extras = []
for extra in extras_require.values():
    all_extras.extend(extra)

# Remove duplicates and create 'all' extra
extras_require["all"] = list(set(all_extras))

# Enterprise edition includes production essentials
extras_require["enterprise"] = list(set([
    *extras_require["api"],
    *extras_require["storage"],
    *extras_require["distributed"],
    *extras_require["mlops"],
    *extras_require["monitoring"],
    *extras_require["security"],
    *extras_require["orchestration"],
    *extras_require["export"],
    *extras_require["streaming"],
]))

# Production deployment essentials
extras_require["production"] = [
    "gunicorn>=21.2.0",
    "supervisor>=4.2.0",
    "nginx>=0.0.1",
    "docker>=6.1.0",
    "kubernetes>=28.1.0",
    "helm>=0.0.1",
]

# Setup configuration
setup(
    # Package metadata
    name="automl-platform",
    version=version,
    author="AutoML Platform Team",
    author_email="team@automl-platform.com",
    description="Enterprise AutoML platform with MLOps, distributed training, and production deployment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/automl-platform/automl-platform",
    license="MIT",
    
    # Package configuration
    packages=find_packages(
        include=["automl_platform", "automl_platform.*"],
        exclude=["tests", "tests.*", "docs", "docs.*", "examples", "examples.*"]
    ),
    package_dir={
        "automl_platform": "automl_platform",
    },
    include_package_data=True,
    package_data={
        "automl_platform": [
            "*.yaml",
            "*.yml",
            "*.json",
            "*.toml",
            "templates/**/*",
            "static/**/*",
            "migrations/**/*",
            "configs/**/*",
        ]
    },
    
    # Dependencies
    python_requires=">=3.9,<3.13",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Entry points
    entry_points={
        "console_scripts": [
            # Main CLI
            "automl=automl_platform.cli.main:cli",
            
            # Training & Prediction
            "automl-train=automl_platform.cli.train:train_cli",
            "automl-predict=automl_platform.cli.predict:predict_cli",
            "automl-evaluate=automl_platform.cli.evaluate:evaluate_cli",
            
            # API Server
            "automl-api=automl_platform.api.app:main",
            "automl-worker=automl_platform.worker.celery_app:main",
            
            # MLOps
            "automl-mlflow=automl_platform.mlops.mlflow_server:main",
            "automl-monitor=automl_platform.monitoring.monitor:main",
            "automl-retrain=automl_platform.mlops.retrainer:main",
            
            # Data Management
            "automl-data=automl_platform.cli.data:data_cli",
            "automl-feature=automl_platform.cli.feature:feature_cli",
            
            # Export & Deployment
            "automl-export=automl_platform.cli.export:export_cli",
            "automl-deploy=automl_platform.cli.deploy:deploy_cli",
            "automl-serve=automl_platform.serving.server:main",
            
            # Admin & Management
            "automl-admin=automl_platform.cli.admin:admin_cli",
            "automl-migrate=automl_platform.cli.migrate:migrate_cli",
            "automl-backup=automl_platform.cli.backup:backup_cli",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Framework :: FastAPI",
        "Natural Language :: English",
    ],
    
    # Keywords
    keywords=[
        "automl",
        "machine-learning",
        "deep-learning",
        "data-science",
        "artificial-intelligence",
        "mlops",
        "distributed-computing",
        "feature-engineering",
        "hyperparameter-optimization",
        "model-deployment",
        "ensemble-learning",
        "automated-machine-learning",
        "production-ml",
        "enterprise-ml",
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "https://docs.automl-platform.com",
        "API Reference": "https://api.automl-platform.com/docs",
        "Bug Tracker": "https://github.com/automl-platform/automl-platform/issues",
        "Source Code": "https://github.com/automl-platform/automl-platform",
        "Changelog": "https://github.com/automl-platform/automl-platform/blob/main/CHANGELOG.md",
        "Docker Hub": "https://hub.docker.com/r/automl-platform/automl",
        "Helm Charts": "https://charts.automl-platform.com",
        "Slack Community": "https://automl-platform.slack.com",
        "Commercial Support": "https://automl-platform.com/support",
    },
    
    # Testing
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
        "pytest-mock>=3.12.0",
    ],
    
    # Additional options
    zip_safe=False,
    platforms="any",
    
    # Custom commands
    cmdclass={},
    
    # Extra metadata
    long_description_content_type="text/markdown",
    provides=["automl_platform"],
)
