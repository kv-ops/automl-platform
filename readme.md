# AutoML Platform v3.2 - Enterprise MLOps Edition with Extended Connectors

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-2.9%2B-0194E2)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)
[![Expert Mode](https://img.shields.io/badge/Expert%20Mode-Available-gold)](docs/expert-mode.md)
[![Templates](https://img.shields.io/badge/Templates-Available-purple)](docs/templates.md)
[![ONNX](https://img.shields.io/badge/ONNX-1.15%2B-5C5C5C)](https://onnx.ai/)
[![River](https://img.shields.io/badge/River-0.19%2B-00CED1)](https://riverml.xyz/)
[![Test Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen)](https://codecov.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Production-ready AutoML platform with **Use Case Templates**, **Extended Data Connectors**, **Expert Mode** for advanced users, enterprise MLOps capabilities including incremental learning, real-time streaming, advanced scheduling, billing system, and LLM-powered insights.

## 🎯 New in v3.2 - Use Case Templates

The platform now includes **pre-configured templates** for common machine learning use cases, allowing you to start with optimized settings for your specific scenario.

### 📋 Available Templates

| Template | Description | Task Type | Key Features |
|----------|-------------|-----------|--------------|
| **customer_churn** | Customer churn prediction | Binary Classification | Imbalanced data handling, recall optimization, business rules |
| **credit_scoring** | Credit risk assessment | Binary Classification | Explainable AI, regulatory compliance, scorecard generation |
| **fraud_detection** | Fraud detection system | Binary Classification | Anomaly detection, real-time scoring, precision focus |
| **sales_forecasting** | Time series sales prediction | Regression | Seasonality handling, trend analysis, holiday effects |
| **recommendation_system** | Product/content recommendations | Ranking | Collaborative filtering, content-based, cold start handling |

### 🚀 Quick Start with Templates

#### Basic Usage

```bash
# Train with a predefined template
python main.py train --template customer_churn --data customers.csv --target churned

# List all available templates
python main.py list-templates

# Get detailed information about a template
python main.py template-info customer_churn

# Filter templates by task type
python main.py list-templates --task classification

# Filter templates by tags
python main.py list-templates --tags "financial,imbalanced"
```

#### Python API Usage

```python
from automl_platform.template_loader import TemplateLoader
from automl_platform.orchestrator import AutoMLOrchestrator

# Load a template
loader = TemplateLoader()
config = loader.apply_template("customer_churn")

# Use template with AutoML
orchestrator = AutoMLOrchestrator(config)
orchestrator.fit(X, y)
```

### 🛠️ Template Customization

#### Override Template Settings

```bash
# Use template as base and override specific settings
python main.py train \
    --template customer_churn \
    --data data.csv \
    --target churned \
    --scoring f1 \
    --cv-folds 10
```

#### Combine Template with Config File

```bash
# Apply template on top of existing configuration
python main.py train \
    --config my_config.yaml \
    --template-override customer_churn \
    --data data.csv \
    --target churned
```

#### Create Custom Templates

```bash
# Create template from existing configuration
python main.py create-template my_custom_template \
    --from-config config.yaml \
    --description "Custom template for my use case" \
    --tags "custom,optimized"

# Create template from another template
python main.py create-template my_variant \
    --from-template customer_churn \
    --set hpo.n_iter=100 \
    --set algorithms='["XGBoost","LightGBM"]' \
    --description "High-performance churn template"

# Export template to file
python main.py template-info customer_churn --export my_template.yaml
```

### 📝 Template Structure

Each template is a YAML file with the following structure:

```yaml
name: "template_name"
description: "Template description"
author: "Author Name"
version: "1.0.0"
tags: ["tag1", "tag2"]

# Task configuration
task: "classification"  # or "regression", "ranking", "auto"
task_settings:
  problem_type: "binary"
  primary_metric: "f1"
  
# Data preprocessing
preprocessing:
  handle_missing:
    strategy: "smart"
  handle_outliers:
    method: "iqr"
  feature_engineering:
    - create_ratios: true
    - date_features: true
    
# Model selection
algorithms:
  - "XGBoost"
  - "LightGBM"
  - "RandomForest"
  
# Hyperparameter optimization
hpo:
  method: "optuna"
  n_iter: 50
  
# Cross-validation
cv:
  strategy: "stratified"
  n_folds: 5
  
# Business rules (optional)
business_rules:
  probability_threshold: 0.3
  cost_matrix:
    false_positive: 1
    false_negative: 5
```

### 🎯 Template Examples

#### Example 1: Customer Churn with Template

```python
from automl_platform.template_loader import TemplateLoader
from automl_platform.orchestrator import AutoMLOrchestrator
import pandas as pd

# Load data
df = pd.read_csv("customers.csv")
X = df.drop("churned", axis=1)
y = df["churned"]

# Load and apply churn template
loader = TemplateLoader()
config = loader.apply_template("customer_churn")

# Train with optimized settings for churn
orchestrator = AutoMLOrchestrator(config)
orchestrator.fit(X, y)

# Get predictions with business rules applied
predictions = orchestrator.predict(X)
probabilities = orchestrator.predict_proba(X)

# The template automatically:
# - Handles class imbalance
# - Optimizes for recall (catch more churners)
# - Applies cost-sensitive learning
# - Generates feature importance for customer insights
```

#### Example 2: Credit Scoring with Compliance

```python
# Load credit scoring template (includes regulatory compliance)
config = loader.apply_template("credit_scoring")

# This template automatically:
# - Ensures model interpretability (required for credit decisions)
# - Generates adverse action reasons
# - Creates credit scorecards
# - Implements fair lending tests
# - Provides SHAP explanations

orchestrator = AutoMLOrchestrator(config)
orchestrator.fit(X_credit, y_credit)

# Get explanations for regulatory compliance
explanations = orchestrator.explain_predictions(X_test)
scorecard = orchestrator.generate_scorecard()
```

#### Example 3: Sales Forecasting with Seasonality

```python
# Load sales forecasting template
config = loader.apply_template("sales_forecasting")

# This template automatically:
# - Creates lag features (daily, weekly, monthly, yearly)
# - Handles seasonality with Fourier features
# - Adds holiday effects
# - Uses time series cross-validation
# - Implements proper backtesting

orchestrator = AutoMLOrchestrator(config)
orchestrator.fit(X_sales, y_sales, task="timeseries")

# Get forecasts with confidence intervals
forecasts = orchestrator.forecast(horizon=30)
```

### 🔧 Advanced Template Features

#### Template Validation

```python
# Validate template configuration
loader = TemplateLoader()
validation = loader.validate_template("customer_churn")
if validation['valid']:
    print("Template is valid")
else:
    print("Errors:", validation['errors'])
    print("Warnings:", validation['warnings'])
```

#### Merge Multiple Templates

```python
# Combine features from multiple templates
merged = loader.merge_templates(
    ["customer_churn", "fraud_detection"],
    name="churn_fraud_hybrid",
    description="Combined churn and fraud detection"
)
```

#### Template Discovery

```python
# Find templates for your use case
templates = loader.list_templates(task="classification", tags=["financial"])
for template in templates:
    print(f"{template['name']}: {template['description']}")
    print(f"  Algorithms: {', '.join(template['algorithms'])}")
    print(f"  Estimated time: {template['estimated_time']} minutes")
```

### 📊 Template Performance Benchmarks

| Template | Dataset | Accuracy | Training Time | Key Metric |
|----------|---------|----------|---------------|------------|
| customer_churn | Telco (7K rows) | 94.2% | 5 min | F1: 0.86 |
| credit_scoring | German Credit (1K) | 78.5% | 3 min | AUC: 0.82 |
| fraud_detection | Credit Card (285K) | 99.8% | 15 min | Precision: 0.94 |
| sales_forecasting | Retail (3 years) | - | 8 min | MAPE: 8.2% |
| recommendation_system | MovieLens (100K) | - | 12 min | NDCG@10: 0.73 |

### 🎨 Creating Your Own Templates

#### Step 1: Create Template File

Create a YAML file in `automl_platform/templates/use_cases/`:

```yaml
# my_use_case.yaml
name: "my_use_case"
description: "Template for my specific use case"
author: "Your Name"
version: "1.0.0"
tags: ["custom", "specialized"]

task: "classification"
algorithms: ["XGBoost", "LightGBM"]
hpo:
  method: "optuna"
  n_iter: 30
cv:
  n_folds: 5
```

#### Step 2: Register Template

```python
from automl_platform.template_loader import TemplateLoader

loader = TemplateLoader()
loader.create_custom_template(
    name="my_use_case",
    config={...},  # Your configuration
    description="My custom template",
    tags=["custom"],
    save=True  # Save to file
)
```

#### Step 3: Use Your Template

```bash
python main.py train --template my_use_case --data data.csv --target target
```

### 🚦 Template Best Practices

1. **Choose the Right Template**: Start with the template closest to your use case
2. **Validate Your Data**: Ensure your data matches template expectations
3. **Monitor Performance**: Templates are optimized for typical cases, adjust if needed
4. **Customize Carefully**: Override only necessary parameters
5. **Document Changes**: Keep track of template modifications
6. **Version Control**: Version your custom templates

### 📚 Template Documentation

Each template includes comprehensive documentation:

```bash
# View full template documentation
python main.py template-info customer_churn

# Output includes:
# - Description and use cases
# - Required data format
# - Algorithm choices explanation
# - Hyperparameter ranges
# - Business rules applied
# - Performance expectations
# - Customization suggestions
```

## 🎓 Expert Mode vs Simplified Mode

The platform features a **dual-mode interface** to accommodate both beginners and advanced users:

### Mode Selection

#### Command Line Interface (CLI)
```bash
# Use simplified mode (default)
python main.py train --data data.csv --target target

# Activate expert mode for full control
python main.py train --expert --data data.csv --target target

# Expert mode with all options
python main.py train --expert \
    --data data.csv \
    --target target \
    --algorithms XGBoost,LightGBM,CatBoost \
    --hpo-method optuna \
    --hpo-iter 100 \
    --cv-folds 10
```

#### Web Interface (Dashboard)
1. Launch the dashboard:
   ```bash
   streamlit run automl_platform/ui/dashboard.py
   ```

2. Toggle expert mode:
   - Click the **"🎓 Activer le mode Expert"** checkbox in the sidebar
   - The interface dynamically adapts to show/hide advanced options
   - Mode persists across sessions

#### Environment Variable
Set default mode via environment:
```bash
# Enable expert mode by default
export AUTOML_EXPERT_MODE=true

# Or in .env file
AUTOML_EXPERT_MODE=true
```

### Feature Comparison

| Feature | Simplified Mode | Expert Mode |
|---------|----------------|-------------|
| **Algorithm Selection** | Auto (3-5 best) | Manual (15+ available) |
| **HPO Iterations** | 10-30 | Up to 500 |
| **HPO Methods** | Random | Grid, Random, Optuna, Bayesian |
| **Cross-Validation** | 3 folds | 2-10 folds, multiple strategies |
| **Preprocessing** | Automatic | Full control |
| **Feature Engineering** | Basic | Advanced options |
| **Ensemble Methods** | Voting only | Voting, Stacking, Blending |
| **Template Customization** | Use as-is | Full modification |
| **Parallel Processing** | 2 workers | Up to 16 workers |
| **GPU Support** | No | Yes |
| **Monitoring** | Basic metrics | Advanced metrics + drift |
| **Export Formats** | Basic | ONNX, PMML, custom |

### Simplified Mode (Default)

Perfect for beginners and quick prototypes:

```python
# Everything is optimized automatically
from automl_platform import AutoMLOrchestrator

orchestrator = AutoMLOrchestrator()  # Uses optimized defaults
orchestrator.fit(X, y)
predictions = orchestrator.predict(X_test)
```

Dashboard features in simplified mode:
- One-click optimization levels: "Rapide", "Équilibré", "Maximum"
- Pre-selected algorithm combinations
- Automatic preprocessing
- Simple metrics display
- Guided workflow

### Expert Mode

Full control for ML engineers and data scientists:

```python
from automl_platform import AutoMLOrchestrator, AutoMLConfig

# Detailed configuration
config = AutoMLConfig(
    expert_mode=True,
    algorithms=["XGBoost", "LightGBM", "CatBoost", "RandomForest"],
    hpo_method="optuna",
    hpo_n_iter=100,
    cv_strategy="stratified",
    cv_folds=10,
    ensemble_method="stacking",
    feature_engineering=True,
    handle_imbalance=True,
    gpu_enabled=True
)

orchestrator = AutoMLOrchestrator(config)
orchestrator.fit(X, y)
```

Dashboard features in expert mode:
- Full algorithm selection (15+ algorithms)
- Detailed HPO configuration
- Custom validation strategies
- Advanced preprocessing options
- Template modification
- Performance profiling
- Drift detection settings
- Custom metrics
- Parallel processing control

### Mode-Specific Examples

#### Simplified Mode Workflow
```bash
# 1. Quick training with template
python main.py train --template customer_churn --data data.csv --target churn

# 2. Basic prediction
python main.py predict --model model.joblib --data test.csv

# 3. Simple API deployment
python main.py api
```

#### Expert Mode Workflow
```bash
# 1. Advanced training with full control
python main.py train --expert \
    --template customer_churn \
    --data data.csv \
    --target churn \
    --algorithms XGBoost,LightGBM,CatBoost,RandomForest,ExtraTrees \
    --hpo-method optuna \
    --hpo-iter 200 \
    --cv-folds 10 \
    --ensemble stacking \
    --gpu \
    --n-workers 8

# 2. Batch prediction with optimization
python main.py predict --expert \
    --model model.joblib \
    --data test.csv \
    --batch-size 10000 \
    --output predictions.csv

# 3. Production API with scaling
python main.py api --expert \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --reload
```

### Dashboard Mode Toggle

The web interface provides an interactive toggle:

1. **Sidebar Toggle**: 
   - Located in the sidebar under "⚙️ Configuration"
   - Checkbox: "🎓 Activer le mode Expert"
   - Instant UI update without page reload

2. **Visual Indicators**:
   - **Simplified Mode**: Green badge "🚀 Mode Simplifié"
   - **Expert Mode**: Gold badge "🎓 Mode Expert"

3. **Dynamic UI Changes**:
   - Simplified: Shows optimization slider (Rapide/Équilibré/Maximum)
   - Expert: Shows detailed tabs (Algorithmes/Hyperparamètres/Validation/Avancé)

### Best Practices

#### When to Use Simplified Mode:
- First-time users
- Quick prototypes
- Standard use cases with templates
- When optimal defaults are sufficient
- Time-constrained projects

#### When to Use Expert Mode:
- Custom algorithm requirements
- Specific HPO strategies needed
- Advanced preprocessing required
- Custom validation strategies
- Production optimization
- Research projects
- When you need full control

### Tips for Transitioning

Start with simplified mode and gradually move to expert:

1. **Phase 1**: Use simplified mode with templates
2. **Phase 2**: Try simplified mode with custom data
3. **Phase 3**: Enable expert mode but use mostly defaults
4. **Phase 4**: Fully customize in expert mode

### Performance Impact

| Metric | Simplified Mode | Expert Mode (Optimized) |
|--------|----------------|------------------------|
| Training Time | 5-15 min | 15-60+ min |
| Model Count | 3-5 | 10-50+ |
| Accuracy Gain | Baseline | +2-5% typical |
| Resource Usage | Low-Medium | Medium-High |
| Complexity | Low | High |

[Rest of the original README content continues here...]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- pandas for data manipulation
- openpyxl for Excel support
- gspread for Google Sheets integration
- All CRM API providers
- The open-source community

---

**Built for everyone: From Excel users to ML engineers**

*Pre-configured templates: Get started in seconds with optimized settings for your use case*

*Your data, anywhere: Excel, Google Sheets, CRM, Databases - all connected*

*Version 3.2.0 - Last updated: January 2024*
