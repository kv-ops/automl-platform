"""Setup script for AutoML Platform."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Core requirements (minimal set) - UPDATED FOR PYTHON 3.9+
install_requires = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",  # Updated to match requirements.txt
    "joblib>=1.3.0",
    "pyyaml>=6.0.1",
]

# Optional dependencies
extras_require = {
    # Hyperparameter optimization
    "hpo": [
        "optuna>=3.4.0",
        "optuna-dashboard>=0.8.0",
    ],
    
    # Boosting algorithms
    "boosting": [
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
    ],
    
    # Imbalanced learning
    "imbalance": [
        "imbalanced-learn>=0.10.0",
    ],
    
    # Model explainability
    "explain": [
        "shap>=0.43.0",
        "lime>=0.2.0",
    ],
    
    # Time series
    "timeseries": [
        "statsmodels>=0.13.0",
        "prophet>=1.1.0",
        "pmdarima>=1.8.0",
        "sktime>=0.13.0",
    ],
    
    # NLP
    "nlp": [
        "sentence-transformers>=2.0.0",
        "nltk>=3.6.0",
        "spacy>=3.0.0",
    ],
    
    # API
    "api": [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.0",
    ],
    
    # Visualization
    "viz": [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.18.0",
    ],
    
    # Development
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=2.12.0",
        "pytest-asyncio>=0.21.0",
        "black>=23.11.0",
        "flake8>=6.1.0",
        "mypy>=0.910",
        "isort>=5.9.0",
        "pre-commit>=2.15.0",
        "types-PyYAML>=5.4.0",
        "types-requests>=2.25.0",
    ],
    
    # Documentation
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
        "sphinx-autodoc-typehints>=1.12.0",
    ],
}

# All extras
all_extras = []
for extra in extras_require.values():
    all_extras.extend(extra)

extras_require["all"] = list(set(all_extras))

# Setup configuration
setup(
    # Package metadata
    name="automl-platform",
    version="3.0.0",
    author="AutoML Platform Team",
    author_email="team@automl-platform.com",
    description="Production-ready AutoML platform with no data leakage",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/automl-platform/automl-platform",
    license="MIT",
    
    # Package configuration - EXPLICIT MAPPING
    packages=["automl_platform", "automl_platform.api"],  # Explicit packages with underscore
    package_dir={
        "automl_platform": "automl_platform",  # Map automl_platform package to automl_platform directory
        "automl_platform.api": "automl_platform/api",  # Map api subpackage
    },
    include_package_data=True,
    package_data={
        "automl_platform": [
            "*.yaml",
            "*.yml",
            "*.json",
            "templates/*.html",
            "static/*",
        ]
    },
    
    # Dependencies - UPDATED TO PYTHON 3.9+
    python_requires=">=3.9",  # Changed from >=3.8 to >=3.9
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Entry points
    entry_points={
        "console_scripts": [
            "automl=automl_platform.main:main",
            "automl-train=automl_platform.main:train",
            "automl-predict=automl_platform.main:predict_cmd",
            "automl-api=automl_platform.api.app:main",
        ],
    },
    
    # Classifiers - REMOVED 3.8, ADDED 3.12
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: Web Environment",
    ],
    
    # Keywords
    keywords=[
        "automl",
        "machine-learning",
        "data-science",
        "artificial-intelligence",
        "sklearn",
        "pipeline",
        "hyperparameter-optimization",
        "feature-engineering",
        "model-selection",
        "explainability",
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "https://automl-platform.readthedocs.io",
        "Bug Tracker": "https://github.com/automl-platform/automl-platform/issues",
        "Source Code": "https://github.com/automl-platform/automl-platform",
        "Changelog": "https://github.com/automl-platform/automl-platform/blob/main/CHANGELOG.md",
    },
    
    # Testing
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-cov>=2.12.0",
    ],
    
    # Additional options
    zip_safe=False,
    platforms="any",
)
