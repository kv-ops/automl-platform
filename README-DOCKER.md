# AutoML Platform – Docker Operations Guide

This document describes how to provision, operate, and troubleshoot the AutoML Platform using the Docker assets that live in this repository.
Toutes les ressources Docker citées (compose, scripts, configuration) se trouvent directement à la racine du dépôt afin de simplifier la découverte—aucun préfixe `/v2` n'est requis.

## 1. Prérequis

| Outil | Version recommandée | Commande de vérification |
|-------|---------------------|---------------------------|
| Docker Engine | ≥ 24.0 | `docker version`
| Docker Compose V2 | ≥ 2.20 | `docker compose version`
| OpenSSL | ≥ 1.1 | `openssl version`
| GNU Make | ≥ 4.2 | `make --version`

Assurez-vous également que les ports suivants sont libres sur votre machine : `5432`, `6379`, `8000`, `8501`, `9000-9001`, `5000`, `8080`, `3000`, `5555`.

## 2. Quick start (3 commandes)

```bash
make secrets          # Génère .env.production, .env.staging et .env.development
make prod             # Démarre la stack complète de démonstration
make logs             # Suit les logs agrégés pour vérifier l’état
```

Pour un environnement de développement allégé :

```bash
make dev              # Utilise docker-compose.dev.yml avec hot reload
```

## 3. Environnements supportés

| Commande | Description | Fichier .env |
|----------|-------------|--------------|
| `make dev` | Stack minimale (PostgreSQL, Redis, API, UI) | `.env.development` (auto-créé si absent) |
| `make prod` | Stack complète (API, UI, MinIO, MLflow, Keycloak, workers, monitoring, Nginx) | `.env.production` |
| `make secrets` | Génération de secrets pour production, staging et dev | produit `.env.production`, `.env.staging`, `.env.development` |

Le fichier `.env.example` reste la référence fonctionnelle. Les scripts n’insèrent aucun secret dans Git – ils sont générés localement via OpenSSL.

## 4. Architecture des services

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Nginx   │───▶│   UI     │───▶│  Browser │
└────┬─────┘    └────┬─────┘    └──────────┘
     │               │
     ▼               ▼
┌──────────┐    ┌──────────┐         ┌──────────┐
│  API     │───▶│  Worker  │────────▶│ Redis    │
└────┬─────┘    └──────────┘         └──────────┘
     │                                ▲
     ▼                                │
┌──────────┐    ┌──────────┐          │
│ Postgres │    │  MinIO   │◀─────────┘
└────┬─────┘    └──────────┘
     │
     ▼
┌──────────┐    ┌──────────┐
│ Keycloak │    │ MLflow   │
└──────────┘    └──────────┘
```

Les services Prometheus, Grafana, exporters Redis/Postgres et le reverse proxy Nginx se branchent sur ce cœur. Flower expose la file d’attente Celery. Tous les services communiquent via le réseau Docker `automl_network` (production) ou `automl_dev` (développement).

## 5. Génération et gestion des secrets

- `./scripts/generate-secrets.sh` génère des fichiers `.env.production`, `.env.staging` et `.env.development` avec des valeurs fortes (`openssl rand`).
- Chaque exécution crée une sauvegarde datée (`.bak`) avant régénération, sauf si `--force` est passé via `make secrets`.
- Les rôles PostgreSQL (`monitoring`, `backup_user`) et les identifiants MinIO, Keycloak, Grafana, Flower, Stripe sont générés.
- Les API keys externes (OpenAI, Anthropic, Hugging Face) restent à saisir manuellement.

## 6. Commandes utiles (Makefile)

| Commande | Description |
|----------|-------------|
| `make secrets` | Génère / régénère les fichiers `.env.*` avec des secrets cryptographiques |
| `make dev` | Démarre l’environnement de développement (docker-compose.dev.yml) |
| `make prod` | Démarre la stack complète production-ready |
| `make stop` | Arrête les containers lancés (utilise `.env.production`) |
| `make logs` | Suivi en direct de l’ensemble des logs Docker Compose |
| `make clean` | Supprime les containers mais conserve les volumes |
| `make backup` | Lance le script `scripts/backup.sh` pour PostgreSQL + MinIO |
| `make status` | Affiche l’état des services et les URLs d’accès |

## 7. Accès aux services (production)

| Service | URL | Identifiants par défaut |
|---------|-----|------------------------|
| UI Streamlit | https://localhost (via Nginx) ou http://localhost:8501 | Utilisateurs Keycloak (`admin/admin123` temporaire) |
| API FastAPI | https://localhost/api/docs ou http://localhost:8000/docs | Jeton JWT à obtenir via Keycloak |
| MLflow | https://localhost/mlflow ou http://localhost:5000 | `mlflow` / mot de passe `.env.production` |
| Keycloak | https://localhost/auth ou http://localhost:8080 | `admin` / valeur générée (`KEYCLOAK_ADMIN_PASSWORD`) |
| MinIO Console | https://localhost:9001 | `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` |
| Grafana | https://localhost:3000 | `GRAFANA_USER` / `GRAFANA_PASSWORD` |
| Flower (Celery) | http://localhost:5555 | `FLOWER_USER` / `FLOWER_PASSWORD` |
| Prometheus | http://localhost:9090 | (pas d’auth) |

⚠️ Les mots de passe générés sont stockés uniquement dans vos fichiers `.env.*` locaux.

## 8. Scripts opérationnels

- `scripts/init-multi-db.sh` : crée les bases `automl`, `keycloak`, `airflow`, `metadata` et provisionne les rôles `monitoring` / `backup_user` avec privilèges minimaux.
- `scripts/wait-for-it.sh` : utilitaire générique pour attendre qu’un service TCP réponde.
- `scripts/generate-secrets.sh` : génération des secrets (`make secrets`).
- `scripts/backup.sh` : sauvegarde PostgreSQL (`pg_dumpall`) + mirroring MinIO + archivage des configs (Nginx/Keycloak/monitoring).

## 9. Procédure de démarrage manuelle

```bash
# Étape 1 : Générer les secrets
./scripts/generate-secrets.sh

# Étape 2 : Construire les images
DOCKER_BUILDKIT=1 docker compose --env-file .env.production build

# Étape 3 : Lancer les services
DOCKER_BUILDKIT=1 docker compose --env-file .env.production up -d

# Étape 4 : Vérifier la santé
docker compose --env-file .env.production ps
```

## 10. Troubleshooting

| Problème | Cause probable | Solution |
|----------|----------------|----------|
| `permission denied` sur scripts | Droits d’exécution manquants | `chmod +x scripts/*.sh` |
| Keycloak ne démarre pas | Fichier `keycloak/realm-export.json` absent | Vérifier le montage et régénérer via Git |
| MinIO errors « credential required » | Variables `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` non définies | Re-générer les secrets ou vérifier `.env.production` |
| API en erreur 503 au boot | Dépendances non prêtes | `wait-for-it` intégré, relancer `make prod` ou vérifier santé `docker compose ps` |
| Grafana login échoue | Mot de passe par défaut non mis à jour | Utiliser la valeur `GRAFANA_PASSWORD` du `.env.production` |
| Sauvegarde MinIO échoue | Service `minio` non démarré ou credentials invalides | Vérifier que le service est healthy avant d’exécuter `scripts/backup.sh` |

## 11. Commandes de validation rapide

```bash
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
open http://localhost:8000/docs        # ou xdg-open sous Linux
open http://localhost:3000             # Grafana
```

## 12. Checklist Démo

1. `make secrets`
2. `make prod`
3. `make logs`
4. Vérifier `docker compose --env-file .env.production ps`
5. Accéder aux URLs :
   - UI : https://localhost (ou http://localhost:8501)
   - API : https://localhost/api/docs
   - MLflow : https://localhost/mlflow
   - Keycloak : https://localhost/auth
   - Grafana : http://localhost:3000
   - Flower : http://localhost:5555

## 13. Sauvegardes & restauration

- Lancer un backup : `./scripts/backup.sh --env-file .env.production`
- Les dumps sont déposés dans `backups/<horodatage>/`.
- Pour restaurer PostgreSQL : `docker compose exec postgres psql -U $POSTGRES_USER -f /backups/postgres_dump.sql` (monter le dossier au préalable).
- Pour restaurer MinIO : `docker compose run --rm minio-init mc mirror /backup myminio`.

## 14. Notes complémentaires

- Le reverse proxy Nginx termine TLS en lisant les certificats montés dans `nginx/ssl/` (`tls.crt` et `tls.key`). Utiliser mkcert ou Let’s Encrypt pour la démo.
- La cible `make prod` génère automatiquement un certificat autosigné si aucun n'est présent dans `nginx/ssl/` (via `make ssl-cert`).
- Les dashboards Grafana sont provisionnés automatiquement depuis `monitoring/grafana/dashboards/`.
- Les exporter Prometheus (`postgres-exporter`, `redis-exporter`) sont activés par défaut et nécessitent les rôles générés via `scripts/init-multi-db.sh`.
- Le fichier `.dockerignore` exclut les artefacts de build et les jeux de données volumineux pour accélérer les builds.

Bonnes démos !
