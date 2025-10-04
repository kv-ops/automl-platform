"""
Setup script for AutoML Platform
Version 3.2.1 - Enterprise Edition with No-Code UI and Extended Connectors
"""

from setuptools import setup, find_packages
from pathlib import Path

from dependency_manifest import (
    get_core_dependencies,
    get_base_extras,
    get_aggregated_extras,
)

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read version from version file
version = "3.2.1"  # Updated version for extended connectors
version_file = this_directory / "automl_platform" / "__version__.py"
if version_file.exists():
    with open(version_file) as f:
        exec(f.read())

install_requires = get_core_dependencies()

# Optional dependencies organized by feature - harmonized with pyproject.toml
extras_require = get_base_extras()
extras_require.update(get_aggregated_extras())

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

            # UI Dashboard
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

            # Reports & Analytics
            "automl-report=automl_platform.cli.report:report_cli",
            "automl-analytics=automl_platform.cli.analytics:analytics_cli",
            
            # Connector utilities
            "automl-excel=automl_platform.cli.connectors:excel_cli",
            "automl-gsheets=automl_platform.cli.connectors:gsheets_cli",
            "automl-crm=automl_platform.cli.connectors:crm_cli",

            # Intelligent cleaning
            "automl-clean=automl_platform.cli.clean:clean_cli",
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

    # Keywords
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
        "intelligent-agents",
        "openai",
        "data-cleaning",
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
        "Agents Guide": "https://docs.automl-platform.com/agents",
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
