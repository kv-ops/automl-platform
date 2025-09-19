"""
Setup script for AutoML Platform
Version 3.2.0 - Enterprise Edition with No-Code UI and Extended Connectors
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read version from version file
version = "3.2.1"  # Updated version for extended connectors
version_file = this_directory / "automl_platform" / "__version__.py"
if version_file.exists():
    with open(version_file) as f:
        exec(f.read())

# Core requirements (essential dependencies for basic functionality)
install_requires = [
    # Core ML libraries
    "pandas>=2.0.0,<3.0.0",
    "numpy>=1.24.0,<2.0.0",
    "scikit-learn>=1.3.0,<2.0.0",
    "scipy>=1.11.0,<2.0.0",
    "joblib>=1.3.0",

    # Configuration & Validation
    "pyyaml>=6.0.1",
    "pydantic>=2.5.0,<3.0.0",
    "python-dotenv>=1.0.0",

    # AutoML Core
    "optuna>=3.4.0,<4.0.0",
    "xgboost>=2.0.0,<3.0.0",
    "lightgbm>=4.0.0,<5.0.0",
    "catboost>=1.2.0,<2.0.0",
    "imbalanced-learn>=0.10.0,<1.0.0",

    # API Framework (Essential)
    "fastapi>=0.104.0,<1.0.0",
    "uvicorn[standard]>=0.24.0,<1.0.0",
    "starlette>=0.27.0",

    # HTTP & Core Auth
    "httpx>=0.25.0",
    "authlib>=1.2.0",
    "requests>=2.31.0",

    # Storage (Essential)
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "redis>=5.0.0",

    # Security (Essential)
    "cryptography>=41.0.0",
    "pyjwt>=2.8.0",
    "passlib[bcrypt]>=1.7.4",

    # MLOps (Essential)
    "mlflow>=2.9.0,<3.0.0",

    # UI Framework (Essential for no-code)
    "streamlit>=1.30.0",
    "plotly>=5.18.0",
    "streamlit-option-menu>=0.3.6",

    # Utilities
    "tqdm>=4.66.0",
    "click>=8.1.0",
    "prometheus-client>=0.19.0",

    # Additional utilities
    "python-multipart>=0.0.6",
    "aiofiles>=23.2.0",
    "aiohttp>=3.9.0",
    "tabulate>=0.9.0",
    "tenacity>=8.2.0",
    "psutil>=5.9.6",
    
    # NEW: Essential for extended connectors
    "openpyxl>=3.1.0",  # Pour Excel avancé
    "xlsxwriter>=3.1.0",  # Pour l'export Excel
    "pyarrow>=14.0.0",  # Pour Parquet
]

# Optional dependencies organized by feature
extras_require = {
    # NEW: Extended connectors bundle
    "connectors": [
        # Excel support
        "openpyxl>=3.1.0",
        "xlsxwriter>=3.1.0",
        "xlrd>=2.0.1",  # Pour lire les anciens fichiers Excel
        
        # Google Sheets support
        "gspread>=5.12.0",
        "google-auth>=2.27.0",
        "google-auth-oauthlib>=1.2.0",
        "google-auth-httplib2>=0.2.0",
        "pygsheets>=2.0.6",  # Alternative à gspread
        
        # CRM connectors
        "hubspot-api-client>=8.2.0",  # HubSpot
        "simple-salesforce>=1.12.5",  # Salesforce
        "pipedrive-python-lib>=1.2.0",  # Pipedrive
        "zoho-crm>=0.5.0",  # Zoho CRM
        
        # Database connectors (extended)
        "snowflake-connector-python>=3.7.0",
        "google-cloud-bigquery>=3.15.0",
        "databricks-connect>=14.1.0",
        "pyodbc>=5.0.1",
        "cx_Oracle>=8.3.0",
        "pymongo>=4.6.0",
        "cassandra-driver>=3.29.0",
        "elasticsearch>=8.12.0",
        "influxdb-client>=1.40.0",
        "mysqlclient>=2.2.0",
        "pymysql>=1.1.0",
    ],
    
    # Enhanced UI/Dashboard components
    "ui_advanced": [
        "streamlit-extras>=0.3.6",
        "streamlit-aggrid>=0.3.4",
        "streamlit-authenticator>=0.2.3",
        "streamlit-chat>=0.1.1",
        "streamlit-elements>=0.1.0",
        "streamlit-lottie>=0.0.5",
        "streamlit-drawable-canvas>=0.9.3",
        "streamlit-autorefresh>=1.0.1",
        "streamlit-webrtc>=0.47.0",
        "streamlit-folium>=0.15.0",
        "streamlit-ace>=0.1.1",
        "streamlit-tags>=1.2.8",
        "streamlit-tree-select>=0.0.5",
    ],

    # Report generation
    "reporting": [
        "reportlab>=4.0.0",
        "python-docx>=1.1.0",
        "xlsxwriter>=3.1.0",
        "fpdf2>=2.7.0",
        "jinja2>=3.1.0",
        "weasyprint>=60.0",
        "python-pptx>=0.6.0",
    ],

    # Enhanced Authentication & SSO
    "auth": [
        "python-keycloak>=3.7.0",
        "python-saml>=1.15.0",
        "okta>=2.9.0",
        "python-jose[cryptography]>=3.3.0",
        "msal>=1.26.0",  # Microsoft Authentication
        "google-auth>=2.27.0",
        "oauthlib>=3.2.0",
    ],

    # GPU Computing & Acceleration
    "gpu": [
        "cupy-cuda11x>=12.0.0,<13.0.0",
        "numba[cuda]>=0.58.0",
        "gputil>=1.4.0",
        "nvidia-ml-py3>=7.352.0",
        "pynvml>=11.5.0",
        "gpustat>=1.1.1",
        "onnxruntime-gpu>=1.16.0,<2.0.0",
        "pytorch-memlab>=0.3.0",
        "torch-tb-profiler>=0.4.0",
    ],

    # Advanced distributed GPU training
    "distributed_gpu": [
        "horovod>=0.28.0,<1.0.0",
        "fairscale>=0.4.0,<1.0.0",
        "deepspeed>=0.12.0,<1.0.0",
        "ray[train]>=2.8.0",
    ],

    # Hyperparameter optimization
    "hpo": [
        "optuna-dashboard>=0.13.0",
        "hyperopt>=0.2.7",
        "scikit-optimize>=0.9.0",
        "nevergrad>=0.6.0",
    ],

    # Deep learning
    "deep": [
        "tensorflow>=2.15.0,<3.0.0",
        "torch>=2.1.0,<3.0.0",
        "torchvision>=0.16.0,<1.0.0",
        "torchaudio>=2.1.0,<3.0.0",
        "pytorch-tabnet>=4.1.0",
        "pytorch-lightning>=2.1.0",
        "transformers>=4.36.0",
    ],

    # Model explainability
    "explain": [
        "shap>=0.43.0",
        "lime>=0.2.0",
        "eli5>=0.13.0",
        "interpret>=0.4.0",
        "alibi>=0.9.0",
        "captum>=0.7.0",
    ],

    # Time series
    "timeseries": [
        "statsmodels>=0.14.0",
        "prophet>=1.1.5",
        "pmdarima>=2.0.0",
        "sktime>=0.24.0",
        "tsfresh>=0.20.0",
        "darts>=0.26.0",
        "neuralforecast>=1.6.0",
    ],

    # NLP
    "nlp": [
        "sentence-transformers>=2.2.0",
        "nltk>=3.8.0",
        "spacy>=3.7.0",
        "gensim>=4.3.0",
        "textblob>=0.17.0",
        "langdetect>=1.0.9",
        "textstat>=0.7.3",
    ],

    # Computer Vision
    "vision": [
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "albumentations>=1.3.0",
        "torchvision>=0.16.0",
        "supervision>=0.17.0",
        "ultralytics>=8.1.0",
    ],

    # Enhanced API features
    "api": [
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.0",
        "websockets>=12.0",
        "slowapi>=0.1.9",
        "gunicorn>=21.2.0",
        "python-socketio>=5.11.0",
        "fastapi-limiter>=0.1.5",
        "fastapi-versioning>=0.10.0",
    ],

    # Database & Storage (updated with connectors)
    "storage": [
        "alembic>=1.13.0",
        "pymongo>=4.6.0",
        "minio>=7.2.0",
        "boto3>=1.34.0",
        "google-cloud-storage>=2.10.0",
        "azure-storage-blob>=12.19.0",
        "aioboto3>=12.0.0",
        # Database connectors now included here too
        "snowflake-connector-python>=3.7.0",
        "mysqlclient>=2.2.0",
        "pymysql>=1.1.0",
    ],

    # Distributed Computing
    "distributed": [
        "ray[default,train,tune]>=2.8.0,<3.0.0",
        "dask[complete]>=2023.12.0",
        "celery[redis]>=5.3.0",
        "flower>=2.0.0",
        "dramatiq[redis]>=1.15.0",
    ],

    # MLOps & Model Management
    "mlops": [
        "dvc>=3.48.0",
        "wandb>=0.16.0",
        "neptune-client>=1.8.0",
        "bentoml>=1.1.0",
        "evidently>=0.4.0",
        "great-expectations>=0.18.0",
        "deepchecks>=0.17.0",
    ],

    # Workflow Orchestration
    "orchestration": [
        "apache-airflow>=2.8.0",
        "prefect>=2.14.0",
        "dagster>=1.6.0",
        "kedro>=0.19.0",
        "luigi>=3.5.0",
    ],

    # Model Export & Serving
    "export": [
        "onnx>=1.15.0,<2.0.0",
        "onnxruntime>=1.16.0,<2.0.0",
        "skl2onnx>=1.16.0,<2.0.0",
        "sklearn2pmml>=0.104.0",
        "tensorflow-lite>=2.15.0",
        "coremltools>=7.1",
        "tensorflowjs>=4.17.0",
    ],

    # Streaming & Real-time
    "streaming": [
        "kafka-python>=2.0.2",
        "confluent-kafka>=2.3.0",
        "pulsar-client>=3.4.0",
        "redis-py-cluster>=2.1.0",
        "faust-streaming>=0.10.0",
        "aiokafka>=0.10.0",
    ],

    # Feature Store
    "feature_store": [
        "feast>=0.36.0",
        "featuretools>=1.28.0",
        "featureform>=1.12.0",
    ],

    # Monitoring & Observability
    "monitoring": [
        "opentelemetry-api>=1.22.0",
        "opentelemetry-sdk>=1.22.0",
        "opentelemetry-instrumentation-fastapi>=0.43b0",
        "jaeger-client>=4.8.0",
        "sentry-sdk>=1.40.0",
        "datadog>=0.49.0",
        "prometheus-fastapi-instrumentator>=6.1.0",
    ],

    # LLM Integration
    "llm": [
        "openai>=1.10.0",
        "anthropic>=0.8.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "llama-index>=0.10.0",
        "chromadb>=0.4.22",
        "tiktoken>=0.6.0",
        "instructor>=0.5.0",
    ],

    # Visualization
    "viz": [
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "altair>=5.2.0",
        "bokeh>=3.3.0",
        "holoviews>=1.18.0",
        "panel>=1.3.0",
        "hvplot>=0.9.0",
    ],

    # Development & Testing
    "dev": [
        "pytest>=8.0.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.23.0",
        "pytest-mock>=3.12.0",
        "pytest-benchmark>=4.0.0",
        "hypothesis>=6.98.0",
        "faker>=22.0.0",
        "factory-boy>=3.3.0",
        "black>=24.0.0",
        "ruff>=0.2.0",
        "mypy>=1.8.0",
        "isort>=5.13.0",
        "pre-commit>=3.6.0",
        "bandit>=1.7.0",
        "safety>=3.0.0",
        "locust>=2.20.0",
    ],

    # Documentation
    "docs": [
        "sphinx>=7.2.0",
        "sphinx-rtd-theme>=2.0.0",
        "sphinx-autodoc-typehints>=1.25.0",
        "sphinx-copybutton>=0.5.0",
        "myst-parser>=2.0.0",
        "jupyter-book>=0.15.0",
        "mkdocs>=1.5.0",
        "mkdocs-material>=9.5.0",
    ],

    # Cloud providers
    "cloud": [
        "boto3>=1.34.0",
        "s3fs>=2024.2.0",
        "google-cloud-bigquery>=3.15.0",
        "google-cloud-storage>=2.10.0",
        "azure-storage-blob>=12.19.0",
        "azure-identity>=1.15.0",
        "snowflake-connector-python>=3.7.0",
        "databricks-sql-connector>=2.9.0",
    ],
}

# No-code bundle - Everything needed for non-technical users (updated)
extras_require["nocode"] = list(set([
    *extras_require["connectors"],  # NEW: Include all connectors
    *extras_require["ui_advanced"],
    *extras_require["reporting"],
    *extras_require["viz"],
    "jupyter>=1.0.0",
    "notebook>=7.0.0",
    "voila>=0.5.0",
    "papermill>=2.5.0",
]))

# Enterprise edition includes production essentials (updated)
extras_require["enterprise"] = list(set([
    *extras_require["api"],
    *extras_require["storage"],
    *extras_require["distributed"],
    *extras_require["mlops"],
    *extras_require["monitoring"],
    *extras_require["auth"],
    *extras_require["orchestration"],
    *extras_require["export"],
    *extras_require["streaming"],
    *extras_require["cloud"],
    *extras_require["nocode"],
    *extras_require["connectors"],  # NEW: Include connectors
]))

# Combine all extras for complete installation
all_extras = []
for extra in extras_require.values():
    all_extras.extend(extra)
extras_require["all"] = list(set(all_extras))

# Production deployment essentials
extras_require["production"] = [
    "gunicorn>=21.2.0",
    "supervisor>=4.2.0",
    "docker>=7.0.0",
    "kubernetes>=29.0.0",
    "nginx>=0.2.0",
]

# Setup configuration
setup(
    # Package metadata
    name="automl-platform",
    version=version,
    author="AutoML Platform Team",
    author_email="team@automl-platform.com",
    description="Enterprise AutoML platform with no-code UI, extended connectors, MLOps, distributed training, and production deployment",
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
            "ui/assets/**/*",
            "ui/components/**/*",
            "ui/pages/**/*",
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

            # UI Dashboard (NEW)
            "automl-ui=automl_platform.ui.dashboard:main",
            "automl-dashboard=automl_platform.ui.dashboard:main",

            # Training & Prediction
            "automl-train=automl_platform.cli.train:train_cli",
            "automl-predict=automl_platform.cli.predict:predict_cli",
            "automl-evaluate=automl_platform.cli.evaluate:evaluate_cli",
            "automl-wizard=automl_platform.cli.wizard:wizard_cli",

            # API Server
            "automl-api=automl_platform.api.api:main",
            "automl-worker=automl_platform.worker.celery_app:main",

            # MLOps
            "automl-mlflow=automl_platform.mlops.mlflow_server:main",
            "automl-monitor=automl_platform.monitoring.monitor:main",
            "automl-retrain=automl_platform.mlops.retrainer:main",

            # Data Management
            "automl-data=automl_platform.cli.data:data_cli",
            "automl-feature=automl_platform.cli.feature:feature_cli",
            "automl-connect=automl_platform.cli.connect:connect_cli",

            # Export & Deployment
            "automl-export=automl_platform.cli.export:export_cli",
            "automl-deploy=automl_platform.cli.deploy:deploy_cli",
            "automl-serve=automl_platform.serving.server:main",

            # Admin & Management
            "automl-admin=automl_platform.cli.admin:admin_cli",
            "automl-migrate=automl_platform.cli.migrate:migrate_cli",
            "automl-backup=automl_platform.cli.backup:backup_cli",
            "automl-config=automl_platform.cli.config:config_cli",

            # Reports & Analytics (NEW)
            "automl-report=automl_platform.cli.report:report_cli",
            "automl-analytics=automl_platform.cli.analytics:analytics_cli",
            
            # NEW: Connector utilities
            "automl-excel=automl_platform.cli.connectors:excel_cli",
            "automl-gsheets=automl_platform.cli.connectors:gsheets_cli",
            "automl-crm=automl_platform.cli.connectors:crm_cli",
        ],
    },

    # Classifiers
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
        "Topic :: Office/Business :: Financial :: Spreadsheet",
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
        "Framework :: Jupyter",
        "Natural Language :: English",
        "Natural Language :: French",
    ],

    # Keywords (updated)
    keywords=[
        "automl",
        "machine-learning",
        "deep-learning",
        "no-code",
        "low-code",
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
        "dashboard",
        "streamlit",
        "visualization",
        "reporting",
        "authentication",
        "sso",
        "oauth",
        "oidc",
        "gpu",
        "cuda",
        "excel",
        "google-sheets",
        "crm",
        "hubspot",
        "salesforce",
        "data-connectors",
    ],

    # Project URLs
    project_urls={
        "Documentation": "https://docs.automl-platform.com",
        "API Reference": "https://api.automl-platform.com/docs",
        "Dashboard": "https://dashboard.automl-platform.com",
        "Bug Tracker": "https://github.com/automl-platform/automl-platform/issues",
        "Source Code": "https://github.com/automl-platform/automl-platform",
        "Changelog": "https://github.com/automl-platform/automl-platform/blob/main/CHANGELOG.md",
        "Docker Hub": "https://hub.docker.com/r/automl-platform/automl",
        "Helm Charts": "https://charts.automl-platform.com",
        "Demo": "https://demo.automl-platform.com",
        "Tutorials": "https://tutorials.automl-platform.com",
        "YouTube": "https://youtube.com/@automl-platform",
        "Slack Community": "https://automl-platform.slack.com",
        "Commercial Support": "https://automl-platform.com/support",
        "Enterprise": "https://automl-platform.com/enterprise",
        "Connectors Guide": "https://docs.automl-platform.com/connectors",
    },

    # Testing
    test_suite="tests",
    tests_require=[
        "pytest>=8.0.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.23.0",
        "pytest-mock>=3.12.0",
    ],

    # Additional options
    zip_safe=False,
    platforms="any",

    # Extra metadata
    provides=["automl_platform"],
)
