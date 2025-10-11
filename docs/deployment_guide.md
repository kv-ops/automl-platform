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
9. [Sauvegarde et restauration](#sauvegarde-et-restauration)
10. [Dépannage](#dépannage)

## Vue d'ensemble

L'AutoML Platform en mode SaaS est une solution complète prête à déployer qui comprend :

- **API Backend** : FastAPI avec authentification JWT et SSO
- **Interface Web** : Application Streamlit no-code pour les utilisateurs non techniques
- **Workers** : Traitement distribué avec Celery pour l'entraînement des modèles
- **Stockage** : MinIO (S3-compatible) pour les datasets et modèles
- **Base de données** : PostgreSQL pour les métadonnées
- **Cache** : Redis pour les sessions et la communication inter-services
- **SSO** : Keycloak pour l'authentification d'entreprise
- **Monitoring** : Prometheus + Grafana pour la supervision

## Prérequis

### Configuration système minimale

- **CPU** : 4 cœurs minimum (8 recommandés)
- **RAM** : 8 GB minimum (16 GB recommandés)
- **Stockage** : 50 GB d'espace libre
- **OS** : Linux (Ubuntu 20.04+, CentOS 8+) ou macOS
- **Réseau** : Ports 80, 443, 8000, 8501 disponibles

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

# Make et outils de build (optionnel)
sudo apt-get install -y make build-essential
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

Utilisez le script de déploiement pour une installation automatisée :

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

Une fois le déploiement terminé, accédez à :

- **Interface Web** : http://localhost:8501
- **API** : http://localhost:8000
- **Documentation API** : http://localhost:8000/docs
- **MLflow** : http://localhost:5000
- **Keycloak Admin** : http://localhost:8080/admin

## Configuration détaillée

### Variables d'environnement

Créez un fichier `.env` à la racine du projet :

```env
# =============================================================================
# Configuration principale
# =============================================================================

# Mode de déploiement
ENVIRONMENT=production
AUTOML_MODE=saas
AUTOML_EXPERT_MODE=false  # false pour les utilisateurs non techniques

# =============================================================================
# Base de données
# =============================================================================

POSTGRES_USER=automl
POSTGRES_PASSWORD=your-secure-password-here
POSTGRES_DB=automl
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# =============================================================================
# Redis
# =============================================================================

REDIS_PASSWORD=your-redis-password-here
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/0  # Utilisé par le middleware d'authentification

# =============================================================================
# Stockage MinIO (S3-compatible)
# =============================================================================

MINIO_ACCESS_KEY=your-minio-access-key
MINIO_SECRET_KEY=your-minio-secret-key
MINIO_ENDPOINT=minio:9000
MINIO_SECURE=false

# Buckets
S3_BUCKET_MODELS=models
S3_BUCKET_DATASETS=datasets
S3_BUCKET_ARTIFACTS=artifacts
S3_BUCKET_REPORTS=reports

# =============================================================================
# API Configuration
# =============================================================================

API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
JWT_SECRET_KEY=your-very-long-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:8501,https://yourdomain.com

# Rate limiting
RATE_LIMIT_ENABLED=true
DEFAULT_RATE_LIMIT=100  # requests per minute

# =============================================================================
# Interface Web (UI)
# =============================================================================

UI_PORT=8501
UI_FRAMEWORK=streamlit
UI_TITLE=AutoML Platform
UI_THEME=light
MAX_UPLOAD_SIZE=1000  # MB

# URLs publiques
PUBLIC_API_URL=https://api.yourdomain.com
PUBLIC_UI_URL=https://app.yourdomain.com

# Fonctionnalités
ENABLE_CHAT_ASSISTANT=true
ENABLE_AUTO_ML=true
ENABLE_EXPERT_MODE=false
ENABLE_COLLABORATION=true
DEFAULT_LANGUAGE=fr

# =============================================================================
# SSO - Keycloak
# =============================================================================

SSO_ENABLED=true
KEYCLOAK_ENABLED=true
KEYCLOAK_URL=http://keycloak:8080
KEYCLOAK_REALM=automl
KEYCLOAK_CLIENT_ID=automl-platform
KEYCLOAK_CLIENT_SECRET=your-keycloak-secret
KEYCLOAK_ADMIN=admin
KEYCLOAK_ADMIN_PASSWORD=your-admin-password

# Alternative: Auth0
AUTH0_ENABLED=false
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your-client-id
AUTH0_CLIENT_SECRET=your-client-secret

# Alternative: Okta
OKTA_ENABLED=false
OKTA_DOMAIN=your-org.okta.com
OKTA_CLIENT_ID=your-client-id
OKTA_CLIENT_SECRET=your-client-secret

# =============================================================================
# MLflow
# =============================================================================

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_USER=mlflow
MLFLOW_PASSWORD=mlflow-password

# =============================================================================
# Workers (Celery)
# =============================================================================

CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis:6379/0
CPU_WORKER_REPLICAS=2
CPU_WORKER_CONCURRENCY=4
WORKER_MAX_MEMORY_GB=8

# GPU Workers (optionnel)
GPU_WORKER_REPLICAS=0
CUDA_VISIBLE_DEVICES=0

# =============================================================================
# LLM APIs (optionnel)
# =============================================================================

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
ENABLE_AI_ASSISTANT=true

# =============================================================================
# Multi-tenant
# =============================================================================

MULTI_TENANT_ENABLED=true
DEFAULT_TENANT_PLAN=trial
TRIAL_DURATION_DAYS=14

# Quotas par défaut
DEFAULT_MAX_WORKERS=2
DEFAULT_MAX_CONCURRENT_JOBS=2
DEFAULT_STORAGE_QUOTA_GB=10
DEFAULT_MONTHLY_COMPUTE_MINUTES=1000

# =============================================================================
# Monitoring
# =============================================================================

PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_USER=admin
GRAFANA_PASSWORD=grafana-password

# =============================================================================
# Backup
# =============================================================================

BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # 2h du matin chaque jour
BACKUP_RETENTION_DAYS=30
BACKUP_S3_BUCKET=backups
```

### Génération sécurisée des secrets

Les services refuseront désormais de démarrer si des secrets critiques sont
absents ou contiennent les valeurs d'exemple historiques (`minioadmin`,
`change-this-secret-key`, etc.). Générez systématiquement vos secrets via une
source cryptographiquement sécurisée :

```bash
# Clé secrète de plateforme (AUTOML_SECRET_KEY)
python - <<'PY'
import secrets
print(secrets.token_urlsafe(64))
PY

# Identifiants MinIO compatibles avec la validation
export MINIO_ACCESS_KEY="$(openssl rand -hex 16)"
export MINIO_SECRET_KEY="$(openssl rand -base64 32)"

# Secret JWT robuste
openssl rand -base64 48
```

Le script `scripts/deploy_saas.sh` applique automatiquement ces commandes via
`openssl rand`. Pour les environnements gérés (Vault, AWS Secrets Manager,
Kubernetes Secrets, etc.), stockez ces valeurs hors du dépôt Git et référencez
les uniquement via des variables d'environnement.

### Migration depuis les anciennes valeurs par défaut

Lors d'une mise à niveau, la plateforme échoue volontairement au démarrage si
les anciens identifiants `minioadmin` ou les secrets JWT de démonstration sont
toujours présents. Procédez comme suit pour migrer en conservant vos données :

1. **Générez les nouveaux secrets** (voir ci-dessus) et enregistrez-les dans
   votre gestionnaire de secrets.
2. **Mettez à jour** le `.env`, les fichiers `docker-compose.override.yml` ou
   les manifests Kubernetes avec `AUTOML_SECRET_KEY`, `MINIO_ACCESS_KEY` et
   `MINIO_SECRET_KEY` fraichement générés.
3. **Rotation MinIO** : si vous utilisiez l'utilisateur racine `minioadmin`,
   créez un nouvel utilisateur avec les nouveaux identifiants, migrez les
   politiques et désactivez l'ancien compte avant redémarrage. Définissez au
   préalable la valeur de l'ancien secret (`export MINIO_OLD_SECRET="<mot de passe actuel>"`) puis exécutez :

   ```bash
   docker compose exec minio sh -c '
     mc alias set local http://localhost:9000 minioadmin ${MINIO_OLD_SECRET};
     mc admin user add local "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY";
     mc admin policy attach local readwrite --user "$MINIO_ACCESS_KEY";
     mc admin user disable local minioadmin
   '
   ```

   Les buckets existants restent accessibles ; seule l'identité utilisée par la
   plateforme change.
4. **Redémarrez** la stack (`docker compose up -d --force-recreate`) et
   vérifiez les journaux pour confirmer l'utilisation des nouveaux secrets.

> 💡 **Astuce sécurité** : d'autres secrets (PostgreSQL, Redis, Grafana, Flower,
> etc.) ne disposent pas encore de validation centralisée. Appliquez le même
> processus de génération et de rotation pour maintenir un niveau de sécurité
> homogène.

### Configuration Docker Compose

Le fichier `docker-compose.yml` est déjà configuré. Pour personnaliser :

```yaml
# Modifier les limites de ressources
services:
  worker-cpu:
    deploy:
      resources:
        limits:
          cpus: "8"      # Augmenter les CPU
          memory: 16G    # Augmenter la RAM
```

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

# Configuration Nginx (optionnel)
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
```

### 4. Configuration DNS

Configurez vos enregistrements DNS :

```
A    @              -> YOUR_SERVER_IP
A    api            -> YOUR_SERVER_IP
A    app            -> YOUR_SERVER_IP
A    mlflow         -> YOUR_SERVER_IP
A    keycloak       -> YOUR_SERVER_IP
```

## Configuration SSO

### Keycloak

1. **Accéder à la console admin** : http://localhost:8080/admin

2. **Créer un realm** :
   - Nom : `automl`
   - Display name : `AutoML Platform`

3. **Configurer le client** :
   ```json
   {
     "clientId": "automl-platform",
     "rootUrl": "http://localhost:8501",
     "redirectUris": ["http://localhost:8501/*"],
     "webOrigins": ["http://localhost:8501"],
     "publicClient": false,
     "protocol": "openid-connect",
     "standardFlowEnabled": true,
     "implicitFlowEnabled": false,
     "directAccessGrantsEnabled": true
   }
   ```

4. **Créer des rôles** :
   - `admin` : Accès complet
   - `data_scientist` : Création et gestion de projets
   - `viewer` : Lecture seule
   - `trial` : Accès limité

5. **Mapper les attributs** :
   - email → email
   - given_name → firstName
   - family_name → lastName
   - groups → groups

### Auth0 (Alternative)

1. **Créer une application** :
   - Type : Regular Web Application
   - Allowed Callback URLs : `http://localhost:8501/callback`
   - Allowed Logout URLs : `http://localhost:8501`

2. **Configurer les règles** :
   ```javascript
   function addAppMetadata(user, context, callback) {
     user.app_metadata = user.app_metadata || {};
     user.app_metadata.roles = user.app_metadata.roles || [];
     user.app_metadata.plan = user.app_metadata.plan || 'trial';
     
     context.idToken['https://automl/roles'] = user.app_metadata.roles;
     context.idToken['https://automl/plan'] = user.app_metadata.plan;
     
     callback(null, user, context);
   }
   ```

### Azure AD (Alternative)

1. **Enregistrer l'application** :
   - Redirect URI : `http://localhost:8501/callback`
   - Supported account types : Multitenant

2. **Configurer les permissions API** :
   - Microsoft Graph : User.Read
   - Custom scopes : automl.read, automl.write

## Stockage et persistance

### Configuration MinIO

1. **Générer des identifiants sécurisés** :
   ```bash
   export MINIO_ACCESS_KEY="$(openssl rand -hex 16)"
   export MINIO_SECRET_KEY="$(openssl rand -base64 32)"
   ```
   Conservez ces valeurs dans votre gestionnaire de secrets (Vault, AWS Secrets Manager, etc.).

2. **Accéder à la console** : http://localhost:9001 (authentification avec les secrets générés)

3. **Créer les buckets** :
   ```bash
   # Via MC CLI
   docker exec -it automl_minio mc alias set local http://localhost:9000 "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
   docker exec -it automl_minio mc mb local/models
   docker exec -it automl_minio mc mb local/datasets
   docker exec -it automl_minio mc mb local/artifacts
   ```

4. **Configurer les politiques** :
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {"AWS": ["*"]},
         "Action": ["s3:GetObject"],
         "Resource": ["arn:aws:s3:::reports/*"]
       }
     ]
   }
   ```

> ℹ️ **Rotation** : planifiez la rotation des identifiants MinIO via votre gestionnaire de secrets.
> Les services refuseront de démarrer tant que les nouvelles valeurs ne sont pas propagées.

### Volumes Docker

Les données sont persistées dans des volumes Docker :

- `postgres_data` : Base de données
- `redis_data` : Cache et sessions
- `minio_data` : Fichiers et modèles
- `shared_models` : Modèles partagés entre services

### Désactiver la persistance

Pour des environnements éphémères (CI, notebooks temporaires, démonstrations),
il est possible de désactiver complètement la persistance des artefacts en
configurant `storage.backend: none`. Dans ce mode, la plateforme instancie un
``StorageManager`` spécial qui lève immédiatement une erreur explicite si une
opération de sauvegarde ou de chargement est invoquée. Cela permet de détecter
rapidement les usages qui supposent une persistance tout en évitant de créer
des dossiers temporaires non désirés.

### Configuration Google Cloud Storage

L'option `storage.backend: gcs` permet de déléguer le stockage des modèles et des jeux de données à Google Cloud Storage. Pour un déploiement sécurisé :

1. **Création des buckets** : créez manuellement (ou via IaC) les buckets `models`, `datasets` et `artifacts` dans votre projet GCP. Assignez-leur une stratégie de rétention adaptée à vos contraintes de conformité.
2. **Authentification** :
   - En développement, utilisez un compte de service dédié (rôle minimal `Storage Object Admin`) et exposez son chemin via la clé `storage.credentials_path` ou la variable d'environnement `GOOGLE_APPLICATION_CREDENTIALS`.
   - En production sur GKE, privilégiez **Workload Identity** afin d'éviter la distribution de fichiers de clés.
3. **Sécurité des secrets** : ne validez jamais les fichiers de crédential dans Git. Stockez-les dans un coffre-fort (Secret Manager, Vault…) et montez-les au runtime.
4. **Validation** : exécutez `pytest tests/test_storage.py::TestStorageManager::test_storage_manager_gcs_backend` pour vérifier que la configuration GCS est fonctionnelle.

> 💡 Lorsque `storage.credentials_path` est défini, la validation de configuration s'assure désormais que le fichier existe réellement afin d'éviter les déploiements incomplets.

## Monitoring et supervision

### Prometheus

Configuration dans `monitoring/prometheus.yml` :

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
    
  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow:5000']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### Grafana

1. **Accéder** : http://localhost:3000

2. **Importer les dashboards** :
   - AutoML Overview (ID: 15001)
   - ML Training Metrics (ID: 15002)
   - API Performance (ID: 15003)

3. **Configurer les alertes** :
   ```yaml
   groups:
     - name: automl_alerts
       rules:
         - alert: HighAPILatency
           expr: http_request_duration_seconds > 1
           for: 5m
           annotations:
             summary: "API latency is high"
   ```

### Logs centralisés

Pour ELK Stack (optionnel) :

```yaml
# Ajouter dans docker-compose.yml
elasticsearch:
  image: elasticsearch:8.11.0
  environment:
    - discovery.type=single-node
    - ES_JAVA_OPTS=-Xms512m -Xmx512m

kibana:
  image: kibana:8.11.0
  depends_on:
    - elasticsearch
  ports:
    - "5601:5601"
```

## Sauvegarde et restauration

### Sauvegarde automatique

Le script de backup est programmé via Celery Beat :

```python
# automl_platform/tasks/backup.py
from celery import shared_task
import subprocess

@shared_task
def backup_database():
    """Sauvegarde quotidienne de la base de données"""
    subprocess.run([
        "pg_dump",
        "-h", "postgres",
        "-U", "automl",
        "-d", "automl",
        "-f", f"/backups/db_{datetime.now()}.sql"
    ])
```

### Sauvegarde manuelle

```bash
# Sauvegarde complète
./scripts/deploy_saas.sh --backup

# Sauvegarde spécifique
docker exec automl_postgres pg_dump -U automl automl > backup.sql
docker exec automl_redis redis-cli --auth $REDIS_PASSWORD BGSAVE
```

### Restauration

```bash
# Restauration depuis une sauvegarde
./scripts/deploy_saas.sh --restore backup_20240101_120000.tar.gz

# Restauration manuelle de la base
docker exec -i automl_postgres psql -U automl automl < backup.sql
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

# Redémarrer les services
docker-compose restart api ui
```

#### 2. Erreurs de connexion à la base de données

```bash
# Vérifier PostgreSQL
docker exec -it automl_postgres psql -U automl -d automl -c "\l"

# Recréer la base si nécessaire
docker-compose down -v
docker-compose up -d postgres
docker-compose up -d
```

#### 3. Problèmes SSO/Keycloak

```bash
# Réinitialiser Keycloak
docker-compose stop keycloak
docker volume rm automl_keycloak_data
docker-compose up -d keycloak

# Vérifier la configuration
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

### Logs et debugging

```bash
# Tous les logs
docker-compose logs -f

# Logs spécifiques
docker-compose logs -f api --tail=100
docker-compose logs -f worker-cpu --tail=100

# Mode debug
export LOG_LEVEL=debug
docker-compose up
```

### Support

Pour obtenir de l'aide :

1. Consultez la [documentation complète](https://docs.automl-platform.com)
2. Ouvrez une issue sur [GitHub](https://github.com/your-org/automl-platform/issues)
3. Contactez le support : support@automl-platform.com

## Annexes

### Architecture complète

```
┌─────────────────────────────────────────────────────────────┐
│                         Nginx (Reverse Proxy)                │
└────────────┬────────────────────────────────┬────────────────┘
             │                                │
             ▼                                ▼
┌─────────────────────┐          ┌─────────────────────┐
│    UI (Streamlit)   │          │   API (FastAPI)     │
└──────────┬──────────┘          └──────────┬──────────┘
           │                                 │
           └────────────┬────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │      Message Broker          │
         │         (Redis)              │
         └──────────┬───────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐      ┌───────────────┐
│  CPU Workers  │      │  GPU Workers  │
└───────────────┘      └───────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
    ┌───────────────────────────────────┐
    │         Storage Layer             │
    ├───────────────────────────────────┤
    │  PostgreSQL │ MinIO │ MLflow      │
    └───────────────────────────────────┘
```

### Checklist de production

- [ ] Variables d'environnement sécurisées
- [ ] HTTPS configuré
- [ ] Sauvegardes automatiques activées
- [ ] Monitoring en place
- [ ] Logs centralisés
- [ ] SSO configuré et testé
- [ ] Firewall configuré
- [ ] Limites de ressources définies
- [ ] Health checks configurés
- [ ] Documentation à jour
- [ ] Plan de reprise d'activité testé
