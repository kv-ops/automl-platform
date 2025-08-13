from setuptools import setup, find_packages

setup(
    name="automl-platform",
    version="2.0.0",
    author="AutoML Platform Team",
    description="Complete ML automation framework with monitoring, fairness, and explainability",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "scipy>=1.7.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "hpo": ["optuna>=2.10.0"],
        "boosting": ["xgboost>=1.5.0", "lightgbm>=3.2.0", "catboost>=1.0.0"],
        "explain": ["shap>=0.40.0", "lime>=0.2.0"],
        "timeseries": ["prophet>=1.0.0", "pmdarima>=1.8.0"],
        "nlp": ["sentence-transformers>=2.0.0"],
        "dev": ["pytest>=6.0.0", "black>=21.0", "flake8>=3.9.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
