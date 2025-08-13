# AutoML Platform v2.0

Complete ML automation framework with monitoring, fairness, and explainability.

## Features

- 🚀 Automated model training with hyperparameter optimization
- 📊 Data drift detection and quality monitoring
- 🔍 Model explainability (SHAP/LIME)
- ⚖️ Fairness metrics and bias mitigation
- 🌐 REST API for deployment
- 📈 Comprehensive monitoring and alerting
- 🕒 Time series forecasting support
- 📝 NLP feature extraction

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

Or install with extras:

```bash
pip install -e ".[hpo,boosting,explain]"
```

## Quick Start

### Basic Usage

```python
from automl_platform import load_data, train_cv, save_model
from automl_platform.data import split_features_target

# Load and prepare data
df = load_data("data.csv")
X, y = split_features_target(df, "target")

# Train model with automatic HPO
model, info = train_cv(X, y, task="classification")

# Save model
save_model(model, "model.pkl")

print(f"Best model: {info['algorithm']}")
print(f"CV Score: {info['best_score']:.4f}")
```

### API Deployment

```python
# Start API server
uvicorn automl_platform.api.app:app --host 0.0.0.0 --port 8000

# Or programmatically
from automl_platform.api import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Monitoring

```python
from automl_platform.monitoring import check_drift, check_data_quality

# Check data quality
quality_report = check_data_quality(new_data, reference_data)
print(f"Quality score: {quality_report['overall_quality_score']}")

# Check for drift
drift_report = check_drift(new_data, reference_data, threshold=0.1)
if drift_report["drift_detected"]:
    print("⚠️ Data drift detected!")
```

### Explainability

```python
from automl_platform.explain import explain_global, explain_local

# Global feature importance
global_exp = explain_global(model, X)
print("Top features:", sorted(global_exp["importances"].items(), 
                            key=lambda x: x[1], reverse=True)[:5])

# Local explanation for a specific instance
local_exp = explain_local(model, X, instance_idx=0)
print("Instance explanation:", local_exp["importances"])
```

### Fairness

```python
from automl_platform.fairness import fairness_report

# Generate fairness report
report = fairness_report(y_true, y_pred, protected_attribute)
print(f"Fairness score: {report['fairness_score']:.2f}")

# Use fairness-aware model
from automl_platform.fairness.wrappers import ThresholdOptimizedModel

fair_model = ThresholdOptimizedModel(base_model)
fair_model.fit(X, y, protected=protected_attribute)
fair_predictions = fair_model.predict(X_test, protected=protected_test)
```

## API Endpoints

- `GET /` - Root endpoint with API info
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions
- `POST /check_drift` - Check for data drift
- `GET /metrics` - API metrics
- `POST /reload_model` - Reload model from disk

## Configuration

```python
from automl_platform.config import EnhancedPlatformConfig, set_config

config = EnhancedPlatformConfig(
    n_trials=50,
    cv_folds=5,
    algorithms=["xgboost", "lightgbm", "catboost"],
    time_budget=7200,
    drift_threshold=0.05
)
set_config(config)
```

## Project Structure

```
automl_platform/
├── api/           # REST API
├── config/        # Configuration
├── data/          # Data I/O
├── explain/       # Explainability
├── fairness/      # Fairness metrics
├── features/      # Feature engineering
├── modeling/      # Model training
└── monitoring/    # Monitoring & alerts
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
