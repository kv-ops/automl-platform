# AutoML Platform - Examples Directory

## 📚 Vue d'ensemble

Ce dossier contient des exemples pratiques démontrant l'utilisation des différentes fonctionnalités de la plateforme AutoML. Chaque exemple est conçu pour être autonome et peut être exécuté indépendamment.

## 🗂️ Structure des exemples

### 1. **example_mlops_integration.py** ✅
Pipeline MLOps complet avec entraînement, versioning, A/B testing et ré-entraînement automatique.
- **Fonctionnalités**: AutoML, MLflow, A/B Testing, Retraining, Export de modèles
- **Prérequis**: MLflow server démarré, Redis actif
- **Commande**: `python example_mlops_integration.py`

### 2. **example_streaming_realtime.py** ✅
Traitement en temps réel avec Kafka et monitoring des performances.
- **Fonctionnalités**: KafkaStreamProcessor, MLStreamProcessor, métriques Prometheus
- **Datasets**: Données de capteurs, séries temporelles
- **Prérequis**: Kafka démarré (optionnel pour simulation)
- **Commande**: `python example_streaming_realtime.py`

### 3. **example_sso_auth.py** ✅
Authentification SSO avec multiples fournisseurs.
- **Fonctionnalités**: SSO (Keycloak/Auth0/Okta), SAML, sessions, RBAC
- **Prérequis**: Redis, configuration des providers SSO (optionnel)
- **Commande**: `python example_sso_auth.py`

### 4. **example_rgpd_service.py** ✅
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
```

### Configuration minimale
```bash
# Copier le fichier de configuration exemple
cp config.example.yml config.yml

# Démarrer les services essentiels
docker-compose up -d redis postgres mlflow
```

### Ordre d'exécution recommandé
```bash
# 1. Commencer par le pipeline MLOps complet
python example_mlops_integration.py

# 2. Tester le streaming temps réel
python example_streaming_realtime.py

# 3. Explorer l'authentification SSO
python example_sso_auth.py

# 4. Découvrir la conformité RGPD
python example_rgpd_service.py
```

## 📊 Fonctionnalités par exemple

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

## 📊 Métriques et monitoring

### Prometheus (streaming example)
Quand l'exemple streaming est exécuté, les métriques sont disponibles :
- **URL**: http://localhost:8090/metrics
- **Métriques**: throughput, latence, erreurs, lag

### MLflow (MLOps example)
- **URL**: http://localhost:5000
- **Tracking**: Expériences, modèles, métriques

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
- [Service de streaming](../api/streaming.py)
- [Service SSO](../sso_service.py)
- [Service RGPD](../rgpd_compliance_service.py)
- [Registry MLflow](../mlflow_registry.py)

## 🎯 Cas d'usage par exemple

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

## 📧 Support

Pour toute question sur les exemples :
- Consultez les commentaires dans le code
- Vérifiez les logs d'exécution
- Assurez-vous que les services requis sont démarrés
