"""Centralized dependency manifest for AutoML Platform packaging.

This module captures the canonical dependency definitions for the project so
that setup.py, pyproject.toml synchronization utilities, and requirement file
generators can share a single source of truth.  Keeping the lists here avoids
subtle drift between packaging metadata, developer documentation, and the
installation experience.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, List, Mapping, MutableMapping, Sequence, Tuple

# Ordered groups of core dependencies allow generated files to retain helpful
# commentary while providing setup.py with a flattened install_requires list.
CORE_DEPENDENCY_GROUPS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        (
            "Core ML libraries",
            [
                "pandas>=2.0.0,<3.0.0",
                "numpy>=1.24.0,<2.0.0",
                "scikit-learn>=1.3.0,<2.0.0",
                "scipy>=1.11.0,<2.0.0",
                "joblib>=1.3.0",
                "optuna>=3.4.0,<4.0.0",
                "xgboost>=2.0.0,<3.0.0",
                "lightgbm>=4.0.0,<5.0.0",
                "catboost>=1.2.0,<2.0.0",
                "imbalanced-learn>=0.10.0,<1.0.0",
            ],
        ),
        (
            "Configuration & validation",
            [
                "pyyaml>=6.0.1",
                "pydantic>=2.5.0,<3.0.0",
                "python-dotenv>=1.0.0",
            ],
        ),
        (
            "API & backend",
            [
                "fastapi>=0.104.0,<1.0.0",
                "uvicorn[standard]>=0.24.0,<1.0.0",
                "starlette>=0.27.0",
                "aiofiles>=23.2.0",
                "python-multipart>=0.0.6",
                "slowapi>=0.1.9",
            ],
        ),
        (
            "HTTP clients & auth",
            [
                "requests>=2.31.0",
                "httpx>=0.25.0",
                "aiohttp>=3.9.0",
                "authlib>=1.2.0",
            ],
        ),
        (
            "Persistence & tracking",
            [
                "sqlalchemy>=2.0.0",
                "psycopg2-binary>=2.9.0",
                "redis>=5.0.0",
                "alembic>=1.13.0",
                "mlflow>=2.9.0,<3.0.0",
                "pyarrow>=14.0.0",
            ],
        ),
        (
            "Background workers",
            [
                "celery[redis]>=5.3.0",
            ],
        ),
        (
            "Security",
            [
                "cryptography>=41.0.0",
                "pyjwt>=2.8.0",
                "passlib[bcrypt]>=1.7.4",
            ],
        ),
        (
            "Observability & ops",
            [
                "prometheus-client>=0.19.0",
                "psutil>=5.9.6",
            ],
        ),
    ]
)


def get_core_dependencies() -> List[str]:
    """Return the flattened list of core dependencies preserving order."""
    return [pkg for group in CORE_DEPENDENCY_GROUPS.values() for pkg in group]


# Ordered base extras definitions – these map directly to feature-oriented
# install extras and exclude aggregate bundles like ``nocode`` or ``enterprise``.
BASE_EXTRAS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        (
            "agents",
            [
                "openai>=1.10.0",
                "beautifulsoup4>=4.11.0",
                "tiktoken>=0.6.0",
            ],
        ),
        (
            "auth",
            [
                "python-jose[cryptography]>=3.3.0",
                "python-keycloak>=3.0.0",
                "itsdangerous>=2.1.0",
                "okta-jwt-verifier>=0.2.3",
                "msal>=1.25.0",
                "requests-oauthlib>=1.3.1",
            ],
        ),
        (
            "sso",
            [
                "auth0-python>=4.5.0",
                "python3-saml>=1.16.0",
                "okta-jwt-verifier>=0.2.3",
                "msal>=1.25.0",
                "google-auth-oauthlib>=1.0.0",
                "requests-oauthlib>=1.3.1",
            ],
        ),
        (
            "api",
            [
                "fastapi-pagination>=0.12.11",
                "fastapi-versioning>=0.10.0",
                "python-socketio>=5.9.0",
                "starlette-exporter>=0.19.0",
                "slowapi>=0.1.9",
            ],
        ),
        (
            "ui",
            [
                "streamlit>=1.30.0",
                "streamlit-option-menu>=0.3.6",
                "streamlit-authenticator>=0.2.3",
                "streamlit-aggrid>=0.3.4",
                "plotly>=5.18.0",
                "matplotlib>=3.8.0",
                "seaborn>=0.13.0",
            ],
        ),
        (
            "reporting",
            [
                "reportlab>=4.0.0",
                "python-docx>=1.1.0",
                "xlsxwriter>=3.1.0",
                "fpdf2>=2.7.0",
                "jinja2>=3.1.0",
            ],
        ),
        (
            "export",
            [
                "onnx>=1.15.0",
                "onnxruntime>=1.16.0",
                "skl2onnx>=1.16.0",
                "sklearn2pmml>=0.100.2",
                "tensorflow>=2.15.0,<3.0.0",
                "coremltools>=6.3.0",
            ],
        ),
        (
            "connectors",
            [
                "openpyxl>=3.1.0",
                "gspread>=5.7.2",
                "google-auth>=2.14.1",
                "google-auth-oauthlib>=1.0.0",
                "pyarrow>=14.0.0",
                "fastparquet>=2023.10.0",
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
        ),
        (
            "cloud",
            [
                "boto3>=1.34.0",
                "aioboto3>=12.0.0",
                "google-cloud-storage>=2.10.0",
                "azure-storage-blob>=12.19.0",
                "minio>=7.2.0",
            ],
        ),
        (
            "streaming",
            [
                "kafka-python>=2.0.2",
                "aiokafka>=0.8.1",
                "pulsar-client>=3.3.0",
                "pyflink>=1.18.0",
                "faust-streaming>=0.10.8",
                "redis>=5.0.0",
            ],
        ),
        (
            "feature-store",
            [
                "feast>=0.35.0",
                "redis-om>=0.2.2",
                "deltalake>=0.12.0",
                "great-expectations>=0.18.0",
            ],
        ),
        (
            "orchestration",
            [
                "apache-airflow>=2.8.0",
                "prefect>=2.14.0",
                "dagster>=1.5.0",
                "kedro>=0.18.0",
                "luigi>=3.2.0",
            ],
        ),
        (
            "production",
            [
                "docker>=6.1.3",
                "kubernetes>=28.1.0",
                "gunicorn>=21.2.0",
                "uvicorn[standard]>=0.24.0,<1.0.0",
                "supervisor>=4.2.5",
            ],
        ),
        (
            "mlops",
            [
                "mlflow>=2.9.0,<3.0.0",
                "dvc>=3.0.0",
                "wandb>=0.16.0",
                "neptune-client>=1.7.0",
                "bentoml>=1.1.0",
                "great-expectations>=0.18.0",
            ],
        ),
        (
            "quality",
            [
                "deepchecks>=0.17.0",
                "evidently>=0.4.0",
            ],
        ),
        (
            "monitoring",
            [
                "opentelemetry-api>=1.22.0",
                "opentelemetry-sdk>=1.22.0",
                "opentelemetry-instrumentation-fastapi>=0.43b0",
                "jaeger-client>=4.8.0",
                "sentry-sdk>=1.40.0",
                "datadog>=0.49.0",
            ],
        ),
        (
            "explain",
            [
                "shap>=0.43.0",
                "lime>=0.2.0",
                "eli5>=0.13.0",
                "interpret>=0.4.4",
                "alibi>=0.9.5",
                "captum>=0.7.0",
            ],
        ),
        (
            "timeseries",
            [
                "prophet>=1.1.4",
                "statsmodels>=0.14.0",
                "pmdarima>=2.0.3",
                "sktime>=0.23.0",
                "tsfresh>=0.20.0",
                "darts>=0.27.0",
                "neuralforecast>=1.7.0",
            ],
        ),
        (
            "nlp",
            [
                "sentence-transformers>=2.2.0",
                "spacy>=3.7.0",
                "nltk>=3.8.1",
                "gensim>=4.3.0",
                "textblob>=0.17.1",
                "langdetect>=1.0.9",
            ],
        ),
        (
            "vision",
            [
                "opencv-python>=4.8.0",
                "pillow>=10.0.0",
                "albumentations>=1.3.1",
                "supervision>=0.17.0",
                "ultralytics>=8.0.0",
            ],
        ),
        (
            "hpo",
            [
                "optuna>=3.4.0",
                "hyperopt>=0.2.7",
                "scikit-optimize>=0.9.0",
                "nevergrad>=1.0.0",
                "bayesian-optimization>=1.4.3",
            ],
        ),
        (
            "llm",
            [
                "openai>=1.10.0",
                "anthropic>=0.8.0",
                "langchain>=0.1.0",
                "langchain-community>=0.0.20",
                "llama-index>=0.10.0",
                "chromadb>=0.4.22",
                "tiktoken>=0.6.0",
            ],
        ),
        (
            "distributed",
            [
                "ray[default,train,tune]>=2.8.0,<3.0.0",
                "dask[distributed]>=2023.12.0",
            ],
        ),
        (
            "gpu",
            [
                "cupy-cuda11x>=12.0.0,<13.0.0",
                "numba[cuda]>=0.58.0",
                "gputil>=1.4.0",
                "nvidia-ml-py3>=7.352.0",
                "pynvml>=11.5.0",
                "gpustat>=1.1.1",
                "onnxruntime-gpu>=1.16.0,<2.0.0",
            ],
        ),
        (
            "deep",
            [
                "torch>=2.1.0,<3.0.0",
                "torchvision>=0.16.0,<1.0.0",
                "torchaudio>=2.1.0,<3.0.0",
                "tensorflow>=2.15.0,<3.0.0",
                "pytorch-lightning>=2.1.0",
            ],
        ),
        (
            "distributed-gpu",
            [
                "horovod[pytorch]>=0.28.0",
                "deepspeed>=0.12.0",
                "fairscale>=0.4.13",
            ],
        ),
        (
            "automl-gpu",
            [
                "autogluon>=1.0.0",
                "nni>=3.0.0",
            ],
        ),
        (
            "serving-gpu",
            [
                "tritonclient[grpc]>=2.38.0",
                "tensorflow-serving-api>=2.15.0",
            ],
        ),
        (
            "workers",
            [
                "flower>=2.0.0",
            ],
        ),
        (
            "tests",
            [
                "pytest>=8.0.0",
                "pytest-asyncio>=0.23.0",
                "pytest-cov>=4.1.0",
                "pytest-mock>=3.12.0",
                "hypothesis>=6.98.0",
            ],
        ),
        (
            "docs",
            [
                "sphinx>=7.2.0",
                "sphinx-rtd-theme>=2.0.0",
                "myst-parser>=2.0.0",
                "sphinx-autodoc-typehints>=1.25.0",
                "sphinx-copybutton>=0.5.0",
                "mkdocs>=1.5.0",
                "mkdocs-material>=9.5.0",
                "jupyter-book>=0.15.0",
            ],
        ),
        (
            "dev",
            [
                "black>=24.0.0",
                "ruff>=0.2.0",
                "mypy>=1.8.0",
                "isort>=5.13.0",
                "pre-commit>=3.6.0",
                "ipython>=8.12.0",
            ],
        ),
    ]
)

# Aggregated extras compose one or more of the base groups to produce curated
# bundles.  They intentionally omit direct package listings so that the manifest
# can expand them consistently across setup.py, pyproject.toml, and the
# documentation generator.
ENTERPRISE_EXTRAS: Tuple[str, ...] = (
    "nocode",
    "cloud",
    "monitoring",
    "quality",
    "distributed",
    "llm",
    "workers",
    "auth",
    "sso",
    "production",
    "mlops",
    "feature-store",
    "streaming",
    "orchestration",
)

GPU_COMPLETE_EXTRAS: Tuple[str, ...] = (
    "gpu",
    "deep",
    "distributed-gpu",
    "automl-gpu",
    "serving-gpu",
)

ALL_EXTRAS_BUNDLE: Tuple[str, ...] = tuple(BASE_EXTRAS.keys()) + (
    "nocode",
    "enterprise",
    "gpu-complete",
)

AGGREGATED_EXTRAS: "OrderedDict[str, Sequence[str]]" = OrderedDict(
    [
        ("nocode", ("ui", "connectors", "reporting")),
        ("enterprise", ENTERPRISE_EXTRAS),
        ("gpu-complete", GPU_COMPLETE_EXTRAS),
        ("all", ALL_EXTRAS_BUNDLE),
    ]
)

# Presentation groupings for documentation output.  Each entry maps a category
# heading to the extras that should appear under it.
OPTIONAL_DEPENDENCY_CATEGORIES: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("Intelligent Agents", ["agents"]),
        ("Security & Access", ["auth", "sso"]),
        ("API & Integration", ["api", "streaming", "orchestration"]),
        ("Feature Management", ["feature-store"]),
        ("User Experience", ["ui", "nocode"]),
        ("Reporting", ["reporting", "export"]),
        ("Connectors & Data Sources", ["connectors", "cloud"]),
        ("Monitoring & Quality", ["monitoring", "quality", "mlops", "production"]),
        ("LLM Integration", ["llm"]),
        ("Advanced Analytics", ["explain", "timeseries", "nlp", "vision", "hpo"]),
        (
            "Distributed & GPU",
            [
                "distributed",
                "gpu",
                "deep",
                "distributed-gpu",
                "automl-gpu",
                "serving-gpu",
                "gpu-complete",
            ],
        ),
        ("Automation & Workers", ["workers"]),
        ("Developer Tooling", ["tests", "docs", "dev"]),
        ("Enterprise Bundle", ["enterprise", "all"]),
    ]
)

# Default order for extras output that keeps base groups first followed by the
# aggregate bundles.
EXTRA_ORDER: List[str] = list(BASE_EXTRAS.keys()) + list(AGGREGATED_EXTRAS.keys())


def resolve_extra(
    name: str,
    base_extras: Mapping[str, Sequence[str]] | None = None,
    aggregated: Mapping[str, Sequence[str]] | None = None,
    _stack: MutableMapping[str, bool] | None = None,
) -> List[str]:
    """Resolve an extra into a deduplicated list of dependency strings."""

    base_extras = base_extras or BASE_EXTRAS
    aggregated = aggregated or AGGREGATED_EXTRAS
    stack = _stack or {}

    if name in stack:
        raise ValueError(f"Circular extra dependency detected: {' -> '.join(list(stack) + [name])}")
    stack[name] = True

    ordered: List[str] = []

    if name in base_extras:
        ordered.extend(base_extras[name])

    if name in aggregated:
        for child in aggregated[name]:
            ordered.extend(resolve_extra(child, base_extras, aggregated, stack))

    # Deduplicate while preserving order.
    deduped = list(dict.fromkeys(ordered))

    stack.pop(name, None)
    return deduped


def get_base_extras() -> "OrderedDict[str, List[str]]":
    """Return a copy of the base extras mapping."""
    return OrderedDict((name, list(pkgs)) for name, pkgs in BASE_EXTRAS.items())


def get_aggregated_extras() -> "OrderedDict[str, List[str]]":
    """Return the computed aggregated extras mapping."""
    result: "OrderedDict[str, List[str]]" = OrderedDict()
    for name in AGGREGATED_EXTRAS.keys():
        result[name] = resolve_extra(name)
    return result


def get_all_extras() -> "OrderedDict[str, List[str]]":
    """Return a combined mapping of base and aggregated extras."""
    combined = get_base_extras()
    for name, packages in get_aggregated_extras().items():
        combined[name] = packages
    return combined


def iter_extras_in_order() -> Iterable[tuple[str, List[str]]]:
    """Iterate over extras following EXTRA_ORDER."""
    resolved = get_all_extras()
    for name in EXTRA_ORDER:
        if name in resolved:
            yield name, resolved[name]


def validate_manifest() -> None:
    """Basic sanity checks for the manifest definitions."""
    # Ensure aggregated extras reference known extras.
    for name, children in AGGREGATED_EXTRAS.items():
        for child in children:
            if child not in BASE_EXTRAS and child not in AGGREGATED_EXTRAS:
                raise ValueError(f"Aggregated extra '{name}' references unknown extra '{child}'")

    # Ensure duplicates in base extras don't exist (helps avoid surprises when
    # deduplicating aggregated bundles).
    for name, packages in BASE_EXTRAS.items():
        if len(packages) != len(set(packages)):
            raise ValueError(f"Duplicate package detected in base extra '{name}'")


# Execute validations on import to catch manifest drift early.
validate_manifest()
