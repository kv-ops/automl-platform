# AutoML Platform v2.0

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Pytest-orange)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-ready AutoML platform with sklearn pipelines, no data leakage, comprehensive model testing, and explainability.

## 🎯 Key Features

✅ **No Data Leakage**: All preprocessing done within CV folds using sklearn Pipeline + ColumnTransformer  
✅ **Exhaustive Model Testing**: Tests 30+ sklearn models + XGBoost, LightGBM, CatBoost  
✅ **Automatic Feature Engineering**: Datetime, text (TF-IDF), lag features, polynomial features  
✅ **Hyperparameter Optimization**: Optuna, RandomizedSearchCV, or GridSearchCV  
✅ **Imbalance Handling**: class_weight, SMOTE, ADASYN (per-fold)  
✅ **Model Explainability**: SHAP, LIME, permutation importance  
✅ **Production Ready**: Proper logging, configuration, testing, CI/CD  
✅ **REST API**: FastAPI with async endpoints, background jobs, model caching  
✅ **Comprehensive Testing**: Unit tests, integration tests, no-leakage verification  

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line](#command-line)
  - [Python API](#python-api)
  - [REST API](#rest-api)
- [Configuration](#configuration)
- [Supported Models](#supported-models)
- [Data Preprocessing](#data-preprocessing)
- [Testing](#testing)
- [Architecture](#architecture)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/automl-platform/automl-platform.git
cd automl-platform

# Install core dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Installation with Specific Extras

```bash
# For hyperparameter optimization
pip install -e ".[hpo]"

# For boosting algorithms
pip install -e ".[boosting]"

# For model explainability
pip install -e ".[explain]"

# For API functionality
pip install -e ".[api]"

# For development
pip install -e ".[dev]"
```

## 🎮 Quick Start

### 1. Using Command Line

```bash
# Basic training
python main.py train --data iris.csv --target species

# Training with configuration
python main.py train \
    --data data.csv \
    --target target_column \
    --config config.yaml \
    --output ./results

# Making predictions
python main.py predict \
    --model ./results/pipeline.joblib \
    --data test.csv \
    --output predictions.csv
```

### 2. Using Python API

```python
from automl_platform import AutoMLConfig, AutoMLOrchestrator
import pandas as pd

# Load data
df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Configure
config = AutoMLConfig(
    cv_folds=5,
    hpo_method="optuna",
    algorithms=["all"],  # Test all available models
    handle_imbalance=True
)

# Train
orchestrator = AutoMLOrchestrator(config)
orchestrator.fit(X, y)

# Get results
leaderboard = orchestrator.get_leaderboard()
print(leaderboard)

# Save best model
orchestrator.save_pipeline("best_model.joblib")

# Make predictions
predictions = orchestrator.predict(new_data)
```

### 3. Using REST API

```bash
# Start API server
python main.py api --host 0.0.0.0 --port 8000

# Or using uvicorn directly
uvicorn automl_platform.api.app:app --reload
```

API endpoints:
- `POST /train` - Train new model
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions
- `GET /models` - List trained models
- `GET /health` - Health check

## ⚙️ Configuration

Create `config.yaml` to customize the platform:

```yaml
# General settings
random_state: 42
n_jobs: -1

# Data preprocessing
scaling_method: robust
outlier_method: iqr
handle_imbalance: true

# Model selection
algorithms:
  - all  # or specify: [RandomForest, XGBoost, LightGBM]
cv_folds: 5

# Hyperparameter optimization
hpo_method: optuna
hpo_n_iter: 20

# Output settings
output_dir: ./automl_output
save_pipeline: true
generate_report: true
```

## 🤖 Supported Models

### Classification (28 models)
- **Linear**: LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
- **SVM**: LinearSVC, SVC, NuSVC  
- **Trees**: DecisionTree, RandomForest, ExtraTrees, GradientBoosting
- **Boosting**: XGBoost, LightGBM, CatBoost
- **Ensemble**: AdaBoost, Bagging, Voting, Stacking
- **Others**: KNN, NaiveBayes, MLP, LDA, QDA

### Regression (30 models)
- **Linear**: LinearRegression, Ridge, Lasso, ElasticNet
- **Robust**: Huber, TheilSen, RANSAC, Quantile
- **Trees**: DecisionTree, RandomForest, ExtraTrees, GradientBoosting
- **Boosting**: XGBoost, LightGBM, CatBoost
- **Others**: SVR, KNN, MLP, GaussianProcess

## 🔄 Data Preprocessing Pipeline

All preprocessing is done **within CV folds** to prevent leakage:

1. **Feature Type Detection**
   - Numeric, categorical, datetime, text auto-detection
   
2. **Missing Values**
   - Numeric: median imputation
   - Categorical: mode imputation
   
3. **Outlier Handling**
   - IQR method (default)
   - Isolation Forest
   - Z-score
   
4. **Feature Encoding**
   - Low cardinality: OneHotEncoder
   - High cardinality: Target encoding (in-fold)
   
5. **Scaling**
   - RobustScaler (default)
   - StandardScaler, MinMaxScaler
   
6. **Feature Engineering**
   - Datetime: year, month, day, weekday, quarter
   - Text: TF-IDF + TruncatedSVD
   - Time series: lag features, rolling statistics

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=automl_platform --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py -v

# Type checking
mypy automl_platform

# Linting
flake8 automl_platform

# Code formatting
black automl_platform
```

## 🏗️ Architecture

```
automl_platform/
├── config.py         # Configuration management
├── data_prep.py      # Data preprocessing pipeline
├── model_selection.py # Model selection and HPO
├── orchestrator.py   # Main AutoML engine
├── metrics.py        # Metrics calculation
├── inference.py      # Inference utilities
├── llm.py           # LLM interface (future)
└── api/
    └── app.py       # FastAPI application
```

### Key Design Principles

1. **No Data Leakage**: All transformations in sklearn Pipeline
2. **Modular Design**: Separated concerns, testable components
3. **Configuration-Driven**: YAML-based configuration
4. **Extensive Testing**: Unit + integration tests
5. **Production Ready**: Logging, error handling, monitoring

## ⚡ Performance

- **Parallel Processing**: Uses all CPU cores by default
- **Efficient HPO**: Optuna with pruning
- **Memory Optimization**: Batch processing for large datasets
- **Caching**: Model and preprocessor caching
- **Early Stopping**: For boosting algorithms

### Benchmarks

| Dataset | Models Tested | Time | Best Model | Score |
|---------|--------------|------|------------|-------|
| Iris (150 samples) | 28 | 45s | RandomForest | 0.98 |
| Wine (178 samples) | 28 | 52s | XGBoost | 0.99 |
| Boston (506 samples) | 30 | 2m | LightGBM | 0.89 |
| Adult (48K samples) | 28 | 15m | CatBoost | 0.87 |

## 📚 Documentation

Full documentation available at: [https://automl-platform.readthedocs.io](https://automl-platform.readthedocs.io)

### Examples

See the `examples/` directory for:
- Binary classification
- Multi-class classification
- Regression
- Time series forecasting
- Text classification
- Custom pipelines

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- scikit-learn for the amazing ML framework
- Optuna for hyperparameter optimization
- XGBoost, LightGBM, CatBoost teams
- FastAPI for the modern API framework
- All contributors and users

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/automl-platform/automl-platform/issues)
- Discussions: [GitHub Discussions](https://github.com/automl-platform/automl-platform/discussions)
- Email: team@automl-platform.com

---

**Built with ❤️ for the ML community**
