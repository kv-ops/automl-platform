# AutoML Platform - Template System Guide

## Overview

The AutoML Platform template system provides pre-configured machine learning pipelines optimized for specific use cases. Templates encapsulate best practices, proven algorithm combinations, and domain-specific preprocessing steps to accelerate your ML projects.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Available Templates](#available-templates)
3. [Using Templates](#using-templates)
4. [Creating Custom Templates](#creating-custom-templates)
5. [Template Configuration](#template-configuration)
6. [Advanced Usage](#advanced-usage)
7. [Best Practices](#best-practices)

## Quick Start

### List Available Templates

```bash
python main.py list-templates
```

### Train with a Template

```bash
# Use the customer churn template
python main.py train --template customer_churn --data data.csv --target churn

# Use the fraud detection template
python main.py train --template fraud_detection --data transactions.csv --target is_fraud
```

### Get Template Information

```bash
python main.py template-info customer_churn
```

## Available Templates

### 1. Customer Churn Prediction (`customer_churn`)

**Use Case**: Predict which customers are likely to cancel their subscription or stop using your service.

**Key Features**:
- Optimized for imbalanced data (typically 5-20% churn rate)
- Focus on recall to catch potential churners
- Includes customer behavior features
- Cost-sensitive learning with configurable false negative costs

**Best For**: SaaS, Telecom, Banking, Insurance

```bash
python main.py train --template customer_churn --data customers.csv --target churned
```

### 2. Sales Forecasting (`sales_forecasting`)

**Use Case**: Predict future sales volumes with time series analysis.

**Key Features**:
- Time series specific features (lags, rolling statistics)
- Seasonality detection and handling
- Holiday and trend features
- Backtesting with multiple time windows

**Best For**: Retail, E-commerce, Supply Chain

```bash
python main.py train --template sales_forecasting --data sales_history.csv --target sales_amount
```

### 3. Fraud Detection (`fraud_detection`)

**Use Case**: Identify fraudulent transactions in real-time.

**Key Features**:
- Extreme class imbalance handling (0.1-1% fraud rate)
- Velocity and behavioral features
- Anomaly detection integration
- Real-time scoring optimization (<100ms latency)

**Best For**: Banking, Payment Processing, E-commerce

```bash
python main.py train --template fraud_detection --data transactions.csv --target is_fraud
```

### 4. Credit Scoring (`credit_scoring`)

**Use Case**: Assess credit risk for loan applications.

**Key Features**:
- Regulatory compliance focus (FCRA, ECOA)
- Explainable AI with reason codes
- Scorecard development
- Fair lending tests
- Weight of Evidence binning

**Best For**: Banks, Credit Unions, Fintech Lenders

```bash
python main.py train --template credit_scoring --data applications.csv --target default
```

### 5. Recommendation System (`recommendation_system`)

**Use Case**: Build personalized recommendation engines.

**Key Features**:
- Hybrid collaborative and content-based filtering
- Cold start handling
- Real-time personalization
- Diversity and novelty optimization

**Best For**: E-commerce, Media Streaming, Content Platforms

```bash
python main.py train --template recommendation_system --data interactions.csv --target rating
```

## Using Templates

### Basic Usage

Templates provide complete configurations that can be used directly:

```bash
# Step 1: Choose a template
python main.py list-templates

# Step 2: View template details
python main.py template-info fraud_detection

# Step 3: Train with the template
python main.py train \
  --template fraud_detection \
  --data transactions.csv \
  --target is_fraud \
  --output ./fraud_model
```

### Overriding Template Settings

You can override specific template settings via command line:

```bash
python main.py train \
  --template customer_churn \
  --data data.csv \
  --target churn \
  --cv-folds 10 \              # Override CV folds
  --hpo-iter 100 \              # More HPO iterations
  --algorithms XGBoost,LightGBM # Specific algorithms
```

### Combining Templates with Custom Config

Apply a template on top of your existing configuration:

```bash
python main.py train \
  --config my_config.yaml \
  --template-override fraud_detection \
  --data data.csv \
  --target fraud
```

## Creating Custom Templates

### From Command Line

Create a template from an existing configuration:

```bash
# From a config file
python main.py create-template my_template \
  --from-config successful_model/config.yaml \
  --description "Template for product classification" \
  --tags "classification,products,e-commerce"

# From another template with modifications
python main.py create-template my_fraud_template \
  --from-template fraud_detection \
  --set hpo.n_iter=100 \
  --set algorithms='["XGBoost","LightGBM"]' \
  --description "Custom fraud template with more HPO"
```

### Programmatically

```python
from automl_platform.template_loader import TemplateLoader

loader = TemplateLoader()

# Create custom template
config = {
    "task": "classification",
    "algorithms": ["XGBoost", "RandomForest"],
    "hpo": {
        "method": "optuna",
        "n_iter": 50
    },
    "cv": {
        "n_folds": 5,
        "strategy": "stratified"
    }
}

template = loader.create_custom_template(
    name="my_custom_template",
    config=config,
    description="Custom template for specific use case",
    tags=["custom", "classification"]
)
```

### Template YAML Structure

Create a template file in `automl_platform/templates/use_cases/`:

```yaml
# my_template.yaml
name: "my_template"
description: "Template for specific use case"
author: "Your Name"
version: "1.0.0"
tags: ["classification", "custom"]

# Task configuration
task: "classification"
task_settings:
  problem_type: "binary"
  primary_metric: "roc_auc"

# Preprocessing
preprocessing:
  handle_missing:
    strategy: "smart"
  handle_outliers:
    method: "iqr"
  feature_engineering:
    - create_ratios: true
    - interaction_features: true

# Model selection
algorithms:
  - "XGBoost"
  - "LightGBM"
  - "RandomForest"

# HPO configuration
hpo:
  method: "optuna"
  n_iter: 50
  timeout: 1800

# Cross-validation
cv:
  strategy: "stratified"
  n_folds: 5

# Monitoring
monitoring:
  drift_detection: true
  performance_monitoring:
    track_metrics: ["roc_auc", "f1"]
```

## Template Configuration

### Core Sections

#### Task Configuration

```yaml
task: "classification"  # or "regression", "ranking"
task_settings:
  problem_type: "binary"  # or "multiclass"
  primary_metric: "roc_auc"
  class_imbalance_handling: true
```

#### Preprocessing

```yaml
preprocessing:
  handle_missing:
    strategy: "smart"  # or "mean", "median", "mode", "drop"
    threshold: 0.3  # Drop columns with >30% missing
  
  handle_outliers:
    method: "iqr"  # or "isolation_forest", "clip", "winsorize"
    factor: 1.5
  
  feature_engineering:
    - create_ratios: true
    - date_features: true
    - text_features: false
    - interaction_features: true
```

#### Algorithm Selection

```yaml
algorithms:
  - "XGBoost"
  - "LightGBM"
  - "CatBoost"
  - "RandomForest"

exclude_algorithms:
  - "SVM"  # Too slow
  - "KNN"  # Poor with imbalance
```

#### Hyperparameter Optimization

```yaml
hpo:
  method: "optuna"  # or "grid", "random", "none"
  n_iter: 50
  timeout: 1800  # seconds
  
  search_spaces:
    XGBoost:
      max_depth: [3, 10]
      learning_rate: [0.01, 0.3]
      n_estimators: [100, 500]
```

#### Business Rules

```yaml
business_rules:
  probability_threshold: 0.3
  cost_matrix:
    false_positive: 1
    false_negative: 5
  
  hard_rules:
    - condition: "amount > 10000"
      action: "flag_for_review"
```

## Advanced Usage

### Template Inheritance

Merge multiple templates to combine their strengths:

```python
from automl_platform.template_loader import TemplateLoader

loader = TemplateLoader()

# Merge fraud detection with real-time scoring
merged = loader.merge_templates(
    ["fraud_detection", "real_time_scoring"],
    name="fraud_realtime",
    description="Fraud detection optimized for real-time"
)
```

### Dynamic Template Selection

Select template based on data characteristics:

```python
import pandas as pd
from automl_platform.template_loader import TemplateLoader

def select_template(df, target_col):
    """Select best template based on data."""
    loader = TemplateLoader()
    
    # Check target distribution
    target = df[target_col]
    unique_values = target.nunique()
    
    if unique_values == 2:
        # Binary classification
        imbalance_ratio = target.value_counts().min() / len(target)
        
        if imbalance_ratio < 0.01:
            return "fraud_detection"  # Extreme imbalance
        elif imbalance_ratio < 0.2:
            return "customer_churn"  # Moderate imbalance
        else:
            return "quick_start"  # Balanced
    
    elif unique_values > 2 and unique_values < 20:
        return "quick_start"  # Multiclass
    
    else:
        # Check for time column
        if any('date' in col.lower() for col in df.columns):
            return "sales_forecasting"
        else:
            return "quick_start"  # Default regression

# Usage
df = pd.read_csv("data.csv")
template_name = select_template(df, "target")
print(f"Recommended template: {template_name}")
```

### Template Validation

Validate template before using:

```python
from automl_platform.template_loader import TemplateLoader

loader = TemplateLoader()

# Validate template
validation = loader.validate_template("customer_churn")

if validation["valid"]:
    print("Template is valid")
else:
    print("Template has issues:")
    for error in validation["errors"]:
        print(f"  - {error}")
```

### Export and Share Templates

Export templates for sharing with team:

```bash
# Export as YAML
python main.py template-info customer_churn --export churn_template.yaml

# Export as JSON
python main.py template-info fraud_detection --export fraud_template.json
```

## Best Practices

### 1. Choose the Right Template

- **Start with domain-specific templates** when available
- **Use `quick_start` template** for initial exploration
- **Use `production` template** for deployment-ready models

### 2. Validate with Your Data

Always validate template assumptions with your specific data:

```python
# Check class distribution matches template expectations
print(df[target].value_counts(normalize=True))

# Check data size recommendations
print(f"Data size: {len(df)} rows")
```

### 3. Iterative Refinement

1. Start with a template
2. Evaluate results
3. Create custom template with improvements
4. Share successful templates with team

### 4. Template Versioning

Version your custom templates:

```yaml
name: "fraud_detection"
version: "2.1.0"  # Semantic versioning
changelog:
  - "2.1.0: Added real-time features"
  - "2.0.0: Major algorithm update"
  - "1.0.0: Initial version"
```

### 5. Documentation

Document template assumptions and requirements:

```yaml
name: "credit_scoring"
description: "Credit scoring with regulatory compliance"

requirements:
  min_samples: 5000
  required_features: ["income", "credit_score", "debt_ratio"]
  data_quality:
    max_missing: 0.2
    min_target_rate: 0.05
  
assumptions:
  - "Data includes credit bureau information"
  - "Target is binary (good/bad)"
  - "Compliant with FCRA regulations"
```

## Template Examples by Industry

### Financial Services

```bash
# Credit risk assessment
python main.py train --template credit_scoring --data applications.csv --target default

# Fraud detection
python main.py train --template fraud_detection --data transactions.csv --target fraud

# Customer churn in banking
python main.py train --template customer_churn --data customers.csv --target attrition
```

### E-commerce

```bash
# Product recommendations
python main.py train --template recommendation_system --data purchases.csv --target rating

# Sales forecasting
python main.py train --template sales_forecasting --data daily_sales.csv --target revenue

# Customer lifetime value
python main.py train --template customer_ltv --data customers.csv --target ltv
```

### Healthcare

```bash
# Patient readmission (using churn template)
python main.py train --template customer_churn --data patients.csv --target readmitted

# Clinical risk scoring (using credit scoring template)
python main.py train --template credit_scoring --data clinical.csv --target high_risk
```

## Troubleshooting

### Template Not Found

```bash
# List all available templates
python main.py list-templates

# Check template directory
ls automl_platform/templates/use_cases/
```

### Template Validation Errors

```bash
# Validate template
python main.py template-info my_template

# Check for errors in YAML
python -c "import yaml; yaml.safe_load(open('template.yaml'))"
```

### Performance Issues

- Reduce `hpo.n_iter` for faster training
- Limit algorithms to fastest ones (RandomForest, LogisticRegression)
- Reduce `cv.n_folds` to 3 for quick iteration

## Contributing Templates

Share your successful templates with the community:

1. Create a well-documented template
2. Test with multiple datasets
3. Document assumptions and requirements
4. Submit as pull request to the repository

## Support

For template-related questions:

- Check documentation: `python main.py template-info <template>`
- View examples in `automl_platform/templates/use_cases/`
- Open an issue on GitHub for bugs or feature requests
