# AutoML Platform - Examples Directory

## 📚 Vue d'ensemble

Ce dossier contient des exemples pratiques démontrant l'utilisation des différentes fonctionnalités de la plateforme AutoML. Chaque exemple est conçu pour être autonome et peut être exécuté indépendamment.

## 🗂️ Structure des exemples

### 1. **example_integration.py** ✅
Pipeline MLOps complet avec entraînement, versioning, A/B testing et ré-entraînement automatique.
- **Fonctionnalités**: AutoML, MLflow, A/B Testing, Retraining
- **Prérequis**: MLflow server démarré, Redis actif
- **Commande**: `python example_integration.py`

### 2. **data_preparation_example.py** 
Préparation de données et feature engineering avancé.
- **Fonctionnalités**: DataPreprocessor, AutoFeatureEngineer, DataQualityAssessment
- **Datasets**: Iris, Titanic, données synthétiques
- **Commande**: `python data_preparation_example.py`

### 3. **streaming_realtime_example.py**
Traitement en temps réel avec Kafka et monitoring des performances.
- **Fonctionnalités**: KafkaStreamProcessor, MLStreamProcessor, métriques Prometheus
- **Prérequis**: Kafka démarré, modèle entraîné
- **Commande**: `python streaming_realtime_example.py`

### 4. **sso_rgpd_example.py**
Intégration SSO et conformité RGPD.
- **Fonctionnalités**: SSO (Keycloak/Auth0), gestion des consentements, requêtes RGPD
- **Prérequis**: Serveur SSO configuré, base de données PostgreSQL
- **Commande**: `python sso_rgpd_example.py`

### 5. **ui_walkthrough.py**
Guide d'utilisation de l'interface Streamlit.
- **Fonctionnalités**: Dashboard interactif, chat AI, visualisations
- **Prérequis**: API backend démarrée
- **Commande**: `streamlit run ui_walkthrough.py`

### 6. **monitoring_drift_example.py**
Surveillance de la dérive et alertes en production.
- **Fonctionnalités**: ModelMonitor, détection de dérive, alertes
- **Prérequis**: Modèle en production, données de référence
- **Commande**: `python monitoring_drift_example.py`

### 7. **api_client_example.py**
Utilisation de l'API REST de la plateforme.
- **Fonctionnalités**: Endpoints REST, authentification, opérations CRUD
- **Prérequis**: Serveur API démarré
- **Commande**: `python api_client_example.py`

### 8. **batch_processing_example.py**
Traitement par lots et orchestration avec Airflow.
- **Fonctionnalités**: Batch predictions, scheduling, pipelines
- **Prérequis**: Airflow configuré
- **Commande**: `python batch_processing_example.py`

## 🚀 Démarrage rapide

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Configuration minimale
```bash
# Copier le fichier de configuration exemple
cp config.example.yml config.yml

# Démarrer les services essentiels
docker-compose up -d redis postgres mlflow
```

### Premier exemple
```bash
# Commencer par l'exemple de préparation de données
python data_preparation_example.py

# Puis l'entraînement complet
python example_integration.py
```

## 📊 Datasets d'exemple

Les exemples utilisent plusieurs datasets :
- **Iris** : Classification multi-classe simple
- **Titanic** : Classification binaire avec données manquantes
- **California Housing** : Régression avec features géographiques
- **Synthetic** : Données générées pour tests spécifiques

## 🔧 Variables d'environnement

Créez un fichier `.env` avec :
```env
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Redis
REDIS_URL=redis://localhost:6379

# PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost/automl

# Kafka
KAFKA_BROKERS=localhost:9092

# SSO (optionnel)
KEYCLOAK_URL=http://localhost:8080
AUTH0_DOMAIN=your-domain.auth0.com

# API
API_BASE_URL=http://localhost:8000
API_KEY=your-api-key
```

## 📝 Notes importantes

1. **Ordre recommandé** : data_preparation → example_integration → monitoring_drift
2. **Ressources** : Certains exemples (streaming, batch) nécessitent plus de ressources
3. **Données sensibles** : Les exemples RGPD utilisent des données fictives
4. **Monitoring** : Les métriques Prometheus sont exposées sur le port 9090

## 🐛 Résolution des problèmes

### Erreur de connexion MLflow
```bash
# Vérifier que MLflow est démarré
mlflow server --host 0.0.0.0 --port 5000
```

### Kafka non disponible
```bash
# Démarrer Kafka avec Docker
docker run -d --name kafka -p 9092:9092 confluentinc/cp-kafka:latest
```

### Import errors
```bash
# Installer le package en mode développement
pip install -e .
```

## 📚 Documentation complète

Pour plus de détails, consultez :
- [Documentation API](../docs/api.md)
- [Guide MLOps](../docs/mlops_guide.md)
- [Architecture](../docs/architecture.md)

## 💡 Contribution

Pour ajouter un nouvel exemple :
1. Créer le fichier dans `examples/`
2. Documenter les fonctionnalités utilisées
3. Ajouter une section dans ce README
4. Inclure des commentaires détaillés dans le code

## 📧 Support

Pour toute question : support@automl-platform.com
