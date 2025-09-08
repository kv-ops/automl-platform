# AutoML Platform - Examples Directory

## 📚 Vue d'ensemble

Ce dossier contient des exemples pratiques démontrant l'utilisation des différentes fonctionnalités de la plateforme AutoML. Chaque exemple est conçu pour être autonome et peut être exécuté indépendamment.

## 🗂️ Structure des exemples

### 1. **example_data_preprocessing.py** 🆕 ✅
Démonstration complète du preprocessing, feature engineering et évaluation de la qualité des données.
- **Fonctionnalités**: DataPreprocessor, AutoFeatureEngineer, DataQualityAssessment, AutoMLOrchestrator
- **Datasets**: Iris, Wine, California Housing, Custom synthetic data
- **Points clés**: 
  - Détection automatique des types de features
  - Gestion des valeurs manquantes et outliers
  - Feature engineering automatique (interactions, polynômes, ratios)
  - Évaluation de la qualité avec scoring
  - Comparaison des performances avec/sans preprocessing
- **Commande**: `python example_data_preprocessing.py`

### 2. **example_ui_streamlit.md** 🆕 ✅
Guide complet d'utilisation de l'interface Streamlit interactive.
- **Fonctionnalités**: Interface web complète pour AutoML
- **Sections couvertes**:
  - Upload et analyse de données
  - Configuration et lancement d'entraînements
  - Visualisation du leaderboard
  - Analyse approfondie des modèles
  - Assistant IA intégré
  - Génération de rapports
- **Lancement**: `streamlit run automl_platform/ui/streamlit_app.py`

### 3. **example_mlops_integration.py** ✅
Pipeline MLOps complet avec entraînement, versioning, A/B testing et ré-entraînement automatique.
- **Fonctionnalités**: AutoML, MLflow, A/B Testing, Retraining, Export de modèles
- **Prérequis**: MLflow server démarré, Redis actif
- **Commande**: `python example_mlops_integration.py`

### 4. **example_streaming_realtime.py** ✅
Traitement en temps réel avec Kafka et monitoring des performances.
- **Fonctionnalités**: KafkaStreamProcessor, MLStreamProcessor, métriques Prometheus
- **Datasets**: Données de capteurs, séries temporelles
- **Prérequis**: Kafka démarré (optionnel pour simulation)
- **Commande**: `python example_streaming_realtime.py`

### 5. **example_sso_auth.py** ✅
Authentification SSO avec multiples fournisseurs.
- **Fonctionnalités**: SSO (Keycloak/Auth0/Okta), SAML, sessions, RBAC
- **Prérequis**: Redis, configuration des providers SSO (optionnel)
- **Commande**: `python example_sso_auth.py`

### 6. **example_rgpd_service.py** ✅
Conformité RGPD/GDPR complète.
- **Fonctionnalités**: Gestion des consentements, requêtes RGPD, anonymisation
- **Prérequis**: PostgreSQL, Redis
- **Commande**: `python example_rgpd_service.py`

## 🚀 Démarrage rapide

### Installation des dépendances
```bash
# Dépendances de base
pip install -r requirements.txt

# Dépendances spécifiques aux exemples
pip install kafka-python prometheus-client
pip install pulsar-client  # Optionnel
pip install apache-flink   # Optionnel
pip install python3-saml   # Optionnel pour SAML
pip install streamlit plotly  # Pour l'interface UI
```

### Configuration minimale
```bash
# Copier le fichier de configuration exemple
cp config.example.yml config.yml

# Démarrer les services essentiels
docker-compose up -d redis postgres mlflow
```

### Ordre d'exécution recommandé pour les nouveaux utilisateurs
```bash
# 1. Commencer par comprendre le preprocessing et la qualité des données
python example_data_preprocessing.py

# 2. Explorer l'interface graphique Streamlit
streamlit run automl_platform/ui/streamlit_app.py
# Puis suivre le guide dans example_ui_streamlit.md

# 3. Découvrir le pipeline MLOps complet
python example_mlops_integration.py

# 4. Tester le streaming temps réel
python example_streaming_realtime.py

# 5. Explorer l'authentification SSO
python example_sso_auth.py

# 6. Découvrir la conformité RGPD
python example_rgpd_service.py
```

## 📊 Fonctionnalités par exemple

### **Data Preprocessing** (`example_data_preprocessing.py`) 🆕
- ✅ Chargement de datasets publics (Iris, Wine, California Housing)
- ✅ Création de données synthétiques avec problèmes de qualité
- ✅ Détection automatique des types de colonnes
- ✅ Gestion des valeurs manquantes (multiple stratégies)
- ✅ Détection et traitement des outliers
- ✅ Encodage des variables catégorielles
- ✅ Normalisation/Standardisation des features numériques
- ✅ Feature engineering automatique :
  - Interactions entre features
  - Features polynomiales
  - Ratios et agrégations
  - Features temporelles
  - Features textuelles (TF-IDF)
- ✅ Évaluation de la qualité des données (score 0-100)
- ✅ Détection de dérive (drift)
- ✅ Comparaison des performances :
  - Modèle sur données brutes
  - Modèle sur données preprocessées
  - Modèle sur données avec feature engineering
- ✅ Rapport de qualité détaillé avec recommandations

### **UI Streamlit** (`example_ui_streamlit.md`) 🆕
- ✅ Guide complet d'utilisation de l'interface
- ✅ Upload et visualisation de données
- ✅ Analyse de qualité interactive
- ✅ Configuration d'entraînement via UI
- ✅ Suggestions de features par IA
- ✅ Monitoring en temps réel de l'entraînement
- ✅ Leaderboard interactif avec visualisations
- ✅ Analyse approfondie des modèles :
  - Feature importance
  - SHAP values
  - Matrices de confusion
  - Courbes ROC/PR
- ✅ Assistant IA conversationnel
- ✅ Génération de rapports (PDF, HTML, Markdown)
- ✅ Export de modèles
- ✅ Historique des expériences

### **MLOps Integration** (`example_mlops_integration.py`)
- ✅ Entraînement AutoML avec multiple algorithmes
- ✅ Versioning des modèles avec MLflow
- ✅ A/B testing entre modèles
- ✅ Export ONNX et edge deployment
- ✅ Ré-entraînement automatique
- ✅ Comparaison de versions

### **Streaming Real-time** (`example_streaming_realtime.py`)
- ✅ Traitement de données de capteurs avec Kafka
- ✅ Inférence ML en streaming
- ✅ Détection de dérive (drift)
- ✅ Agrégations fenêtrées multi-temporelles
- ✅ Métriques Prometheus
- ✅ Détection d'anomalies
- ✅ Prévisions simples

### **SSO Authentication** (`example_sso_auth.py`)
- ✅ Multi-provider (Keycloak, Auth0, Okta, Azure AD)
- ✅ Gestion des sessions et tokens
- ✅ RBAC (Role-Based Access Control)
- ✅ SAML 2.0
- ✅ Configuration multi-tenant
- ✅ Sécurité avancée (PKCE, rate limiting)

### **RGPD Compliance** (`example_rgpd_service.py`)
- ✅ Gestion complète des consentements
- ✅ Droits des personnes (Articles 15-17, 20)
- ✅ Anonymisation et pseudonymisation
- ✅ Chiffrement des données sensibles
- ✅ Politiques de rétention
- ✅ Rapports de conformité

## 🎯 Cas d'usage par exemple

### Data Preprocessing
- Nettoyer et préparer des données pour le ML
- Créer automatiquement des features pertinentes
- Évaluer et améliorer la qualité des données
- Comparer l'impact du preprocessing sur les performances
- Générer des rapports de qualité des données

### UI Streamlit
- Interface no-code pour utilisateurs non-techniques
- Exploration interactive des données
- Configuration visuelle des pipelines ML
- Monitoring en temps réel des entraînements
- Collaboration et partage de résultats

### MLOps Integration
- Entraîner et déployer des modèles ML
- Gérer le cycle de vie des modèles
- Effectuer des tests A/B en production
- Automatiser le ré-entraînement

### Streaming Real-time
- Traiter des données IoT en temps réel
- Faire de l'inférence sur des flux de données
- Détecter des anomalies en streaming
- Calculer des agrégations temporelles

### SSO Authentication
- Intégrer l'authentification entreprise
- Gérer des accès multi-tenant
- Implémenter RBAC
- Sécuriser les API

### RGPD Compliance
- Gérer les consentements utilisateurs
- Traiter les demandes RGPD
- Anonymiser les données
- Générer des rapports de conformité

## 🔧 Variables d'environnement

Créez un fichier `.env` avec :
```env
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Redis
REDIS_URL=redis://localhost:6379

# PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost/automl

# Kafka (optionnel)
KAFKA_BROKERS=localhost:9092

# SSO Keycloak (optionnel)
KEYCLOAK_ENABLED=false
KEYCLOAK_CLIENT_ID=your-client-id
KEYCLOAK_CLIENT_SECRET=your-secret
KEYCLOAK_URL=http://localhost:8080
KEYCLOAK_REALM=master

# SSO Auth0 (optionnel)
AUTH0_ENABLED=false
AUTH0_CLIENT_ID=your-client-id
AUTH0_CLIENT_SECRET=your-secret
AUTH0_DOMAIN=your-domain.auth0.com

# RGPD
RGPD_ENCRYPTION_KEY=your-base64-key

# API
API_BASE_URL=http://localhost:8000
API_KEY=your-api-key

# Streamlit (pour l'UI)
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## 🐳 Services Docker

### Docker Compose minimal
```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: automl
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    ports:
      - "5432:5432"
  
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0
```

### Kafka (optionnel)
```bash
# Démarrer Kafka pour le streaming
docker run -d --name kafka \
  -p 9092:9092 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  confluentinc/cp-kafka:latest
```

## 📝 Notes importantes

1. **Services requis** : Tous les exemples peuvent s'exécuter en mode simulation même sans les services externes
2. **Données** : Les exemples utilisent des données synthétiques générées automatiquement
3. **RGPD** : Les données personnelles dans les exemples sont fictives
4. **Monitoring** : Les métriques Prometheus sont exposées sur le port 8090 quand activées
5. **SSO** : Les exemples SSO fonctionnent avec des données mockées si les providers ne sont pas configurés
6. **UI Streamlit** : L'interface nécessite le port 8501 libre par défaut

## 🐛 Résolution des problèmes

### Erreur de connexion MLflow
```bash
# Vérifier que MLflow est démarré
mlflow server --host 0.0.0.0 --port 5000
```

### Kafka non disponible
```bash
# Les exemples fonctionnent en mode simulation
# Pour utiliser Kafka réellement :
docker run -d --name kafka -p 9092:9092 confluentinc/cp-kafka:latest
```

### Import errors
```bash
# Installer le package en mode développement
pip install -e .

# Ou ajouter le path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Redis non disponible
```bash
# Démarrer Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### Streamlit ne démarre pas
```bash
# Vérifier le port
lsof -i :8501

# Utiliser un autre port
streamlit run automl_platform/ui/streamlit_app.py --server.port 8502
```

## 📊 Métriques et monitoring

### Prometheus (streaming example)
Quand l'exemple streaming est exécuté, les métriques sont disponibles :
- **URL**: http://localhost:8090/metrics
- **Métriques**: throughput, latence, erreurs, lag

### MLflow (MLOps example)
- **URL**: http://localhost:5000
- **Tracking**: Expériences, modèles, métriques

### Streamlit (UI example)
- **URL**: http://localhost:8501
- **Interface**: Dashboard interactif complet
- **Métriques**: Toutes les métriques en temps réel

## 💡 Contribution

Pour ajouter un nouvel exemple :
1. Créer le fichier dans `examples/`
2. Documenter les fonctionnalités utilisées
3. Ajouter une section dans ce README
4. Inclure des commentaires détaillés dans le code
5. Fournir des données de test ou génération synthétique

## 📚 Documentation associée

Pour plus de détails sur les composants utilisés :
- [Configuration AutoML](../config.py)
- [Data Preprocessing](../data_prep.py)
- [Feature Engineering](../feature_engineering.py)
- [Data Quality Agent](../data_quality_agent.py)
- [AutoML Orchestrator](../orchestrator.py)
- [Streamlit UI](../ui/streamlit_app.py)
- [Service de streaming](../api/streaming.py)
- [Service SSO](../sso_service.py)
- [Service RGPD](../rgpd_compliance_service.py)
- [Registry MLflow](../mlflow_registry.py)

## 🎓 Parcours d'apprentissage recommandé

### Débutant
1. `example_data_preprocessing.py` - Comprendre les données
2. `example_ui_streamlit.md` - Interface visuelle
3. `example_mlops_integration.py` - Pipeline de base

### Intermédiaire
4. `example_streaming_realtime.py` - Temps réel
5. `example_sso_auth.py` - Sécurité

### Avancé
6. `example_rgpd_service.py` - Conformité
7. Création de pipelines personnalisés
8. Intégration en production

## 📧 Support

Pour toute question sur les exemples :
- Consultez les commentaires dans le code
- Vérifiez les logs d'exécution
- Assurez-vous que les services requis sont démarrés
- Consultez la documentation spécifique de chaque module
