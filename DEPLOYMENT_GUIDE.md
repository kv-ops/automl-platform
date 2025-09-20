# AutoML Platform - Guide de Déploiement SaaS

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis](#prérequis)
3. [Installation rapide](#installation-rapide)
4. [Configuration détaillée](#configuration-détaillée)
5. [Déploiement en production](#déploiement-en-production)
6. [Configuration SSO](#configuration-sso)
7. [Stockage et persistance](#stockage-et-persistance)
8. [Monitoring et supervision](#monitoring-et-supervision)
9. [Intégration MLOps](#intégration-mlops)
10. [Sauvegarde et restauration](#sauvegarde-et-restauration)
11. [Dépannage](#dépannage)

## Vue d'ensemble

L'AutoML Platform en mode SaaS est une solution complète prête à déployer qui comprend :

- **API Backend** : FastAPI avec authentification JWT et SSO (fichier `main.py` avec commande `api`)
- **Interface Web** : Application Streamlit no-code pour les utilisateurs non techniques
- **Workers** : Traitement distribué avec Celery pour l'entraînement des modèles
- **Stockage** : MinIO (S3-compatible) pour les datasets et modèles
- **Base de données** : PostgreSQL avec support multi-bases (automl, keycloak, airflow, metadata)
- **Cache** : Redis pour les sessions et la communication inter-services
- **SSO** : Keycloak pour l'authentification d'entreprise (intégré via `auth.py` et `sso_service.py`)
- **MLOps** : MLflow pour le tracking et registry des modèles
- **Monitoring** : Prometheus + Grafana pour la supervision

## Prérequis

### Configuration système minimale

- **CPU** : 4 cœurs minimum (8 recommandés)
- **RAM** : 8 GB minimum (16 GB recommandés)
- **Stockage** : 50 GB d'espace libre
- **OS** : Linux (Ubuntu 20.04+, CentOS 8+) ou macOS
- **Réseau** : Ports 80, 443, 8000, 8501, 5000, 8080 disponibles

### Logiciels requis

```bash
# Docker (version 20.10+)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose (version 2.0+)
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Git
sudo apt-get update && sudo apt-get install -y git

# OpenSSL pour génération de mots de passe
sudo apt-get install -y openssl
```

### Support GPU (optionnel)

Pour le support GPU avec NVIDIA :

```bash
# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Installation rapide

### 1. Cloner le dépôt

```bash
git clone https://github.com/your-org/automl-platform.git
cd automl-platform
```

### 2. Déploiement automatique

Le script `deploy_saas.sh` automatise complètement le déploiement :

```bash
# Déploiement production standard
./scripts/deploy_saas.sh --env prod

# Déploiement avec monitoring
./scripts/deploy_saas.sh --env prod --monitoring

# Déploiement avec GPU
./scripts/deploy_saas.sh --env prod --gpu

# Déploiement complet avec toutes les options
./scripts/deploy_saas.sh --env prod --monitoring --gpu --scale 4
```

### 3. Accès à la plateforme

Une fois le déploiement terminé (environ 2-3 minutes), accédez à :

- **Interface Web** : http://localhost:8501
- **API** : http://localhost:8000
- **Documentation API** : http://localhost:8000/docs
- **MLflow** : http://localhost:5000
- **Keycloak Admin** : http://localhost:8080/admin
- **MinIO Console** : http://localhost:9001
- **Flower (Celery)** : http://localhost:5555

## Configuration détaillée

### Variables d'environnement

Le script `deploy_saas.sh` crée automatiquement un fichier `.env` à partir de `.env.example` avec :
- Génération automatique de mots de passe sécurisés
- Configuration du mode SaaS (`AUTOML_MODE=saas`)
- Désactivation du mode expert (`AUTOML_EXPERT_MODE=false`)
- Activation du SSO et multi-tenant

Pour personnaliser, modifiez le fichier `.env` généré :

```env
# Mode de déploiement
ENVIRONMENT=production
AUTOML_MODE=saas                    # Mode SaaS activé
AUTOML_EXPERT_MODE=false            # Interface simplifiée pour non-techniques

# SSO Configuration
SSO_ENABLED=true
KEYCLOAK_ENABLED=true
KEYCLOAK_HOSTNAME=localhost         # Changer pour votre domaine
KEYCLOAK_PORT=8080
KEYCLOAK_REALM=automl
KEYCLOAK_CLIENT_ID=automl-platform

# Multi-tenant
MULTI_TENANT_ENABLED=true
DEFAULT_TENANT_PLAN=trial
TRIAL_DURATION_DAYS=14

# Quotas par défaut
DEFAULT_MAX_WORKERS=2
DEFAULT_MAX_CONCURRENT_JOBS=2
DEFAULT_STORAGE_QUOTA_GB=10
```

### Configuration Docker Compose

Le fichier `docker-compose.yml` est préconfiguré avec tous les services. Les services principaux :

- **postgres** : Base de données avec script `init-multi-db.sh` pour créer les bases (automl, keycloak, metadata, airflow)
- **redis** : Cache et broker de messages
- **minio** : Stockage S3-compatible avec buckets préconfigurés
- **keycloak** : SSO avec realm import automatique
- **api** : Backend FastAPI (utilise `main.py` existant)
- **ui** : Interface Streamlit
- **mlflow** : Tracking et registry des modèles
- **worker-cpu** : Workers Celery pour le traitement
- **prometheus/grafana** : Stack de monitoring (optionnel)

## Déploiement en production

### 1. Préparation du serveur

```bash
# Mise à jour système
sudo apt-get update && sudo apt-get upgrade -y

# Configuration firewall
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API
sudo ufw allow 8501/tcp  # UI
sudo ufw allow 5000/tcp  # MLflow
sudo ufw allow 8080/tcp  # Keycloak
sudo ufw enable

# Configuration swap (recommandé)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 2. Configuration HTTPS avec Let's Encrypt

```bash
# Installation Certbot
sudo apt-get install -y certbot

# Génération certificat
sudo certbot certonly --standalone -d yourdomain.com -d api.yourdomain.com

# Mise à jour .env avec les domaines
sed -i 's|PUBLIC_API_URL=.*|PUBLIC_API_URL=https://api.yourdomain.com|' .env
sed -i 's|PUBLIC_UI_URL=.*|PUBLIC_UI_URL=https://yourdomain.com|' .env
sed -i 's|KEYCLOAK_HOSTNAME=.*|KEYCLOAK_HOSTNAME=auth.yourdomain.com|' .env

# Activer Nginx reverse proxy
docker-compose --profile production up -d nginx
```

### 3. Déploiement avec Docker Swarm (haute disponibilité)

```bash
# Initialiser Swarm
docker swarm init --advertise-addr <MANAGER-IP>

# Déployer la stack
docker stack deploy -c docker-compose.yml automl

# Scaler les services
docker service scale automl_worker-cpu=5
docker service scale automl_api=3
```

### 4. Configuration DNS

Configurez vos enregistrements DNS :

```
A    @              -> YOUR_SERVER_IP
A    api            -> YOUR_SERVER_IP
A    app            -> YOUR_SERVER_IP  
A    mlflow         -> YOUR_SERVER_IP
A    auth           -> YOUR_SERVER_IP  # Pour Keycloak
```

### 5. Démarrage des services

Le fichier `dockerfile.saas` contient un script de démarrage intégré qui lance automatiquement :
- L'API sur le port 8000 (utilisant `main.py api`)
- L'interface UI sur le port 8501

```bash
# Les conteneurs démarrent automatiquement avec CMD intégré
docker-compose up -d

# Vérifier les logs
docker-compose logs -f api
docker-compose logs -f ui
```

## Configuration SSO

### Keycloak

L'authentification SSO est gérée par les modules `automl_platform/auth.py` et `automl_platform/sso_service.py` qui supportent :
- Keycloak (par défaut)
- Auth0
- Okta
- Azure AD
- SAML

#### Configuration automatique

Le script `deploy_saas.sh` configure automatiquement Keycloak. Pour une configuration manuelle :

1. **Accéder à la console admin** : http://localhost:8080/admin
   - User: `admin` (défini dans `KEYCLOAK_ADMIN`)
   - Password: Voir votre fichier `.env`

2. **Le realm est créé automatiquement** avec :
   - Nom : `automl`
   - Client : `automl-platform`
   - Redirect URIs configurées

3. **Rôles préconfigurés** (dans `auth.py`) :
   - `admin` : Accès complet
   - `data_scientist` : Création et gestion de projets ML
   - `viewer` : Lecture seule  
   - `trial` : Accès limité (4 workers max, comme DataRobot)

4. **Script d'initialisation via API** :
   ```bash
   # Configuration automatique via l'API Admin de Keycloak
   docker exec -it automl_keycloak /opt/keycloak/bin/kcadm.sh config credentials \
     --server http://localhost:8080 \
     --realm master \
     --user admin \
     --password ${KEYCLOAK_ADMIN_PASSWORD}
   
   # Créer/mettre à jour le realm
   docker exec -it automl_keycloak /opt/keycloak/bin/kcadm.sh create realms \
     -s realm=automl \
     -s enabled=true
   ```

### Multi-tenant et RBAC

Le système multi-tenant est intégré dans `auth.py` avec :
- Isolation par tenant (base de données, buckets MinIO, namespaces)
- Plans de tarification (FREE, TRIAL, PRO, ENTERPRISE)
- Quotas par plan (workers, storage, compute minutes)
- Rate limiting par plan

### Auth0 (Alternative)

Pour utiliser Auth0 au lieu de Keycloak :

1. Définir dans `.env` :
   ```env
   KEYCLOAK_ENABLED=false
   AUTH0_ENABLED=true
   AUTH0_DOMAIN=your-tenant.auth0.com
   AUTH0_CLIENT_ID=your-client-id
   AUTH0_CLIENT_SECRET=your-client-secret
   ```

2. Le module `sso_service.py` gère automatiquement Auth0

## Stockage et persistance

### Configuration MinIO

MinIO est configuré automatiquement avec les buckets :
- `models` : Modèles MLflow
- `datasets` : Datasets uploadés
- `artifacts` : Artefacts de training
- `reports` : Rapports générés (public)
- `backups` : Sauvegardes
- `user-uploads` : Fichiers utilisateurs

Accès console : http://localhost:9001
- User: Voir `MINIO_ACCESS_KEY` dans `.env`
- Password: Voir `MINIO_SECRET_KEY` dans `.env`

### Base de données PostgreSQL

Le script `init-multi-db.sh` crée automatiquement :
- **automl** : Base principale avec schéma et extensions
- **keycloak** : Base pour SSO
- **metadata** : Métadonnées des modèles avec tables préconfigurées
- **airflow** : Orchestration des workflows (si activé)

Optimisations appliquées :
- Extensions : uuid-ossp, pgcrypto, pg_stat_statements
- Utilisateur monitoring en lecture seule
- Paramètres de performance optimisés
- Autovacuum configuré

## Monitoring et supervision

### Stack de monitoring

Activée avec `--monitoring` lors du déploiement :

```bash
./scripts/deploy_saas.sh --env prod --monitoring
```

### Prometheus

- URL : http://localhost:9090
- Métriques collectées depuis tous les services
- Configuration dans `monitoring/prometheus.yml`

### Grafana  

- URL : http://localhost:3000
- SSO intégré avec Keycloak
- Dashboards préconfigurés pour AutoML

### Métriques exposées

L'API expose des métriques sur `/metrics` :
- `model_predictions_total` : Nombre de prédictions
- `model_inference_time_seconds` : Temps d'inférence
- `model_drift_score` : Score de drift
- `auth_attempts_total` : Tentatives d'authentification
- `api_calls_total` : Appels API par endpoint

## Intégration MLOps

### MLflow

MLflow est intégré pour le tracking et le model registry :

- URL : http://localhost:5000
- Backend : PostgreSQL
- Artifacts : MinIO (S3-compatible)
- Authentification basique activée

### Utilisation depuis le code

```python
# Le module orchestrator.py utilise automatiquement MLflow
from automl_platform.orchestrator import AutoMLOrchestrator
from automl_platform.config import AutoMLConfig

config = AutoMLConfig()
config.mlflow_tracking_uri = "http://localhost:5000"

orchestrator = AutoMLOrchestrator(config)
orchestrator.fit(X_train, y_train, register_best_model=True)
```

### A/B Testing et Retraining

Voir `DEPLOYMENT_GUIDE.md` pour les détails sur :
- A/B Testing avec analyse statistique
- Retraining automatique avec Airflow/Prefect
- Export de modèles (ONNX, PMML, TFLite)

## Sauvegarde et restauration

### Sauvegarde automatique

```bash
# Créer une sauvegarde avant mise à jour
./scripts/deploy_saas.sh --backup

# Les sauvegardes sont stockées dans ./backups/
```

### Sauvegarde manuelle

```bash
# Sauvegarde complète des volumes
docker run --rm \
  -v automl_postgres_data:/postgres \
  -v automl_minio_data:/minio \
  -v automl_redis_data:/redis \
  -v ./backups:/backup \
  alpine tar czf /backup/backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  /postgres /minio /redis
```

### Restauration

```bash
# Restaurer depuis une sauvegarde
./scripts/deploy_saas.sh --restore backup_20240101_120000.tar.gz
```

## Dépannage

### Problèmes courants

#### 1. Services ne démarrent pas

```bash
# Vérifier les logs
docker-compose logs -f api
docker-compose logs -f ui

# Vérifier l'état des conteneurs
docker-compose ps

# Redémarrer un service spécifique
docker-compose restart api
```

#### 2. Erreurs de connexion à la base de données

```bash
# Vérifier PostgreSQL
docker exec -it automl_postgres psql -U automl -d automl -c "\l"

# Réinitialiser si nécessaire
docker-compose down -v
docker-compose up -d postgres
# Attendre 30 secondes
docker-compose up -d
```

#### 3. Problèmes SSO/Keycloak

```bash
# Vérifier le statut de Keycloak
curl http://localhost:8080/health/ready

# Réinitialiser Keycloak
docker-compose stop keycloak
docker volume rm automl_keycloak_data
docker-compose up -d keycloak

# Vérifier la configuration du realm
curl http://localhost:8080/realms/automl/.well-known/openid-configuration
```

#### 4. Espace disque insuffisant

```bash
# Nettoyer Docker
docker system prune -a -f
docker volume prune -f

# Vérifier l'utilisation
docker system df
df -h
```

#### 5. Problèmes de performance

```bash
# Augmenter les workers
docker-compose up -d --scale worker-cpu=4

# Vérifier l'utilisation des ressources
docker stats

# Ajuster les limites dans docker-compose.yml
```

### Logs et debugging

```bash
# Tous les logs en temps réel
docker-compose logs -f

# Logs d'un service spécifique
docker-compose logs -f api --tail=100

# Activer le mode debug
sed -i 's/LOG_LEVEL=info/LOG_LEVEL=debug/' .env
docker-compose restart api
```

### Validation de l'installation

Le script `deploy_saas.sh` effectue automatiquement des health checks. Pour vérifier manuellement :

```bash
# API
curl http://localhost:8000/health

# UI  
curl http://localhost:8501/_stcore/health

# MLflow
curl http://localhost:5000/health

# Keycloak
curl http://localhost:8080/health/ready

# MinIO
curl http://localhost:9000/minio/health/live
```

### Architecture complète

```
┌─────────────────────────────────────────────────────────────┐
│                   Nginx (Reverse Proxy)                      │
└────────────┬────────────────────────────────┬────────────────┘
             │                                │
             ▼                                ▼
┌─────────────────────┐          ┌─────────────────────┐
│    UI (Streamlit)   │          │   API (FastAPI)     │
│    Port: 8501       │          │    Port: 8000      │
│  (Démarrage auto)   │          │  (main.py api)      │
└──────────┬──────────┘          └──────────┬──────────┘
           │                                 │
           └────────────┬────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Keycloak SSO + Auth Service │
         │  (auth.py + sso_service.py)  │
         └──────────┬───────────────────┘
                    │
         ┌──────────────────────────────┐
         │      Message Broker          │
         │         (Redis)              │
         └──────────┬───────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐      ┌───────────────┐
│  CPU Workers  │      │  GPU Workers  │
│   (Celery)    │      │  (Optional)   │
└───────────────┘      └───────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
    ┌───────────────────────────────────┐
    │         Storage Layer             │
    ├───────────────────────────────────┤
    │  PostgreSQL │ MinIO │ MLflow      │
    │  (Multi-DB) │ (S3)  │ (Registry)  │
    └───────────────────────────────────┘
```

**Note** : L'API et l'UI démarrent automatiquement via le script intégré dans le Dockerfile. L'authentification SSO est gérée par les modules `auth.py` et `sso_service.py` existants.

### Checklist de production

- [ ] Variables d'environnement sécurisées (générées automatiquement)
- [ ] HTTPS configuré avec certificats valides
- [ ] Sauvegardes automatiques activées
- [ ] Monitoring en place (Prometheus + Grafana)
- [ ] Logs centralisés configurés
- [ ] SSO configuré et testé
- [ ] Firewall configuré
- [ ] Limites de ressources définies dans docker-compose.yml
- [ ] Health checks validés
- [ ] Documentation à jour
- [ ] Plan de reprise d'activité testé

## Support

Pour toute question ou problème :
- Consulter `DEPLOYMENT_GUIDE.md` pour les détails MLOps
- Créer une issue sur le repository
- Consulter les logs dans `/app/logs/` ou via `docker-compose logs`

---

*Document mis à jour pour la version SaaS*
*Dernière modification : 2024*
