# Guide de Déploiement MLOps - Plateforme AutoML

## 📋 Table des Matières
1. [Architecture MLOps](#architecture-mlops)
2. [Installation et Configuration](#installation-et-configuration)
3. [Utilisation des Fonctionnalités](#utilisation-des-fonctionnalités)
4. [API Endpoints](#api-endpoints)
5. [Monitoring et Maintenance](#monitoring-et-maintenance)

---

## 🏗️ Architecture MLOps

### Composants Principaux

```
automl_platform/
├── mlflow_registry.py      # Gestion des versions avec MLflow
├── retraining_service.py   # Retraining automatique (Airflow/Prefect)
├── export_service.py       # Export ONNX/PMML/Edge
├── orchestrator.py         # Orchestrateur principal (mis à jour)
├── api/
│   └── mlops_endpoints.py  # Endpoints REST pour MLOps
└── examples/
    └── mlops_integration.py # Exemple d'intégration complète
```

### Stack Technologique

- **Model Registry**: MLflow (avec fallback local)
- **Workflow Orchestration**: Airflow ou Prefect
- **Model Export**: ONNX, PMML, TensorFlow Lite
- **A/B Testing**: Intégré avec analyse statistique
- **Storage**: MinIO/S3 compatible
- **API**: FastAPI

---

## 🚀 Installation et Configuration

### 1. Installation des Dépendances

```bash
# Core MLOps
pip install mlflow>=2.0.0
pip install onnx onnxruntime skl2onnx
pip install sklearn2pmml

# Orchestration (choisir un)
pip install apache-airflow  # Option 1: Airflow
pip install prefect         # Option 2: Prefect

# Optionnel pour fonctionnalités avancées
pip install tensorflow       # Pour export TFLite
pip install coremltools      # Pour export iOS
```

### 2. Configuration MLflow

```bash
# Démarrer MLflow server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Ou avec PostgreSQL
mlflow server --backend-store-uri postgresql://user:pass@host/db \
              --default-artifact-root s3://bucket/path \
              --host 0.0.0.0
```

### 3. Configuration Airflow

```bash
# Initialiser Airflow
export AIRFLOW_HOME=~/airflow
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com

# Démarrer les services
airflow webserver -p 8080 &
airflow scheduler &
```

### 4. Configuration Prefect (Alternative)

```bash
# Démarrer Prefect server
prefect server start

# Créer work queue
prefect work-queue create ml-retraining-queue
```

### 5. Configuration dans config.yaml

```yaml
# config.yaml
mlflow_tracking_uri: "http://localhost:5000"
enable_ab_testing: true
enable_auto_retraining: true

billing:
  enabled: true
  plan_type: "pro"

monitoring:
  drift_detection_enabled: true
  drift_threshold: 0.3
  
worker:
  backend: "celery"  # ou "prefect"
  max_workers: 4
```

---

## 💡 Utilisation des Fonctionnalités

### 1. Training et Registration de Modèle

```python
from automl_platform.orchestrator import AutoMLOrchestrator
from automl_platform.config import AutoMLConfig

# Configuration
config = AutoMLConfig()
config.mlflow_tracking_uri = "http://localhost:5000"

# Training avec registration automatique
orchestrator = AutoMLOrchestrator(config)
orchestrator.fit(
    X_train, 
    y_train,
    register_best_model=True,
    model_name="my_model"
)
```

### 2. A/B Testing

```python
from automl_platform.mlflow_registry import ABTestingService

# Créer un test A/B
ab_service = ABTestingService(registry)
test_id = ab_service.create_ab_test(
    model_name="my_model",
    champion_version=1,
    challenger_version=2,
    traffic_split=0.2  # 20% au challenger
)

# Analyser les résultats
results = ab_service.get_test_results(test_id)
print(f"Champion: {results['champion']['success_rate']:.2%}")
print(f"Challenger: {results['challenger']['success_rate']:.2%}")

# Conclure et promouvoir le gagnant
ab_service.conclude_test(test_id, promote_winner=True)
```

### 3. Export de Modèle

```python
# Export ONNX avec quantization
result = orchestrator.export_best_model(
    format="onnx",
    sample_data=X_sample
)

# Export pour Edge (multiple formats)
edge_result = orchestrator.export_best_model(
    format="edge",
    sample_data=X_sample
)
```

### 4. Retraining Automatique

```python
from automl_platform.retraining_service import RetrainingService

# Vérifier si retraining nécessaire
should_retrain, reason, metrics = retraining.should_retrain("my_model")

if should_retrain:
    # Déclencher retraining
    result = await retraining.retrain_model(
        "my_model", 
        X_new, 
        y_new,
        reason=reason
    )

# Créer schedule automatique
schedule = retraining.create_retraining_schedule()
```

---

## 🌐 API Endpoints

### Model Registry

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/mlops/models/register` | POST | Enregistrer un nouveau modèle |
| `/api/v1/mlops/models/promote` | POST | Promouvoir un modèle (Staging/Production) |
| `/api/v1/mlops/models/{name}/versions` | GET | Historique des versions |
| `/api/v1/mlops/models/{name}/compare` | GET | Comparer deux versions |
| `/api/v1/mlops/models/{name}/rollback` | POST | Rollback vers version précédente |

### A/B Testing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/mlops/ab-tests/create` | POST | Créer test A/B |
| `/api/v1/mlops/ab-tests/{id}/results` | GET | Résultats du test |
| `/api/v1/mlops/ab-tests/{id}/conclude` | POST | Conclure test et promouvoir |
| `/api/v1/mlops/ab-tests/active` | GET | Tests actifs |

### Model Export

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/mlops/models/export` | POST | Exporter modèle (ONNX/PMML/Edge) |
| `/api/v1/mlops/models/export/{name}/{version}/download` | GET | Télécharger modèle exporté |

### Automated Retraining

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/mlops/retraining/check` | POST | Vérifier besoin de retraining |
| `/api/v1/mlops/retraining/trigger/{name}` | POST | Déclencher retraining manuel |
| `/api/v1/mlops/retraining/schedule` | POST | Créer schedule automatique |
| `/api/v1/mlops/retraining/history` | GET | Historique des retrainings |

### Exemple d'Appel API

```bash
# Enregistrer un modèle
curl -X POST http://localhost:8000/api/v1/mlops/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "customer_churn",
    "description": "XGBoost model v2",
    "metrics": {"accuracy": 0.92, "f1": 0.89}
  }'

# Créer A/B test
curl -X POST http://localhost:8000/api/v1/mlops/ab-tests/create \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "customer_churn",
    "champion_version": 1,
    "challenger_version": 2,
    "traffic_split": 0.1
  }'

# Exporter en ONNX
curl -X POST http://localhost:8000/api/v1/mlops/models/export \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "customer_churn",
    "version": 2,
    "format": "onnx",
    "quantize": true
  }'
```

---

## 📊 Monitoring et Maintenance

### 1. Monitoring des Modèles

```python
# Dashboard de monitoring
from automl_platform.monitoring import ModelMonitor

monitor = ModelMonitor(config)

# Drift detection
drift_score = monitor.get_drift_score("my_model")
if drift_score > 0.3:
    logger.warning(f"High drift detected: {drift_score}")

# Performance tracking
perf_metrics = monitor.get_performance_metrics("my_model")
```

### 2. Métriques Prometheus

Les métriques sont exposées sur `/metrics`:

- `model_predictions_total` - Nombre total de prédictions
- `model_inference_time_seconds` - Temps d'inférence
- `model_drift_score` - Score de drift actuel
- `model_accuracy` - Accuracy en production
- `retraining_runs_total` - Nombre de retrainings

### 3. Alertes

Configuration des alertes dans `config.yaml`:

```yaml
monitoring:
  alerting_enabled: true
  alert_channels: ["log", "email", "slack"]
  slack_webhook_url: "https://hooks.slack.com/..."
  accuracy_alert_threshold: 0.8
  drift_alert_threshold: 0.5
```

### 4. Logs Structurés

```python
import logging
import json

# Configuration des logs JSON
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "model_name": getattr(record, 'model_name', None),
            "version": getattr(record, 'version', None)
        }
        return json.dumps(log_obj)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

---

## 🔧 Troubleshooting

### Problèmes Courants

1. **MLflow connection error**
   ```bash
   # Vérifier que MLflow server est lancé
   curl http://localhost:5000/health
   ```

2. **ONNX export fails**
   ```python
   # Vérifier les dépendances
   import onnx
   import skl2onnx
   print(f"ONNX: {onnx.__version__}")
   print(f"skl2onnx: {skl2onnx.__version__}")
   ```

3. **Airflow DAG not visible**
   ```bash
   # Recharger les DAGs
   airflow dags list
   airflow dags unpause model_retraining
   ```

4. **A/B test not routing correctly**
   ```python
   # Vérifier les tests actifs
   active_tests = ab_testing.get_active_tests()
   print(f"Active tests: {active_tests}")
   ```

---

## 🎯 Best Practices

### 1. Versioning Strategy
- Utiliser semantic versioning pour les modèles
- Taguer les modèles avec métadonnées pertinentes
- Garder historique des 10 dernières versions minimum

### 2. A/B Testing
- Commencer avec 5-10% de trafic pour challenger
- Minimum 1000 échantillons pour significance statistique
- Durée minimum de 7 jours pour patterns hebdomadaires

### 3. Retraining
- Schedule pendant heures creuses (2-4 AM)
- Validation systématique avant promotion
- Garder modèle de fallback en cas d'échec

### 4. Export et Déploiement
- Toujours quantizer pour edge deployment
- Tester inference time sur hardware cible
- Versionner les exports avec le modèle source

---

## 📚 Références

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Prefect Documentation](https://docs.prefect.io/)

---

## 🤝 Support

Pour toute question ou problème:
- Créer une issue sur le repository
- Contacter l'équipe MLOps
- Consulter les logs dans `/logs/mlops/`

---

*Document mis à jour le: 2024*
*Version: 1.0.0*
