# Alembic Database Migrations - AutoML Platform

## 📋 Architecture Multi-Base

Ce projet utilise **2 bases PostgreSQL principales** pour isolation et conformité :

```
┌─────────────────────────────────────────────┐
│         PostgreSQL Server (automl_postgres)  │
├─────────────────────────────────────────────┤
│  📦 automl        → MLflow tracking         │
│                     (géré par MLflow)        │
│                                              │
│  📦 automl_app    → Auth + Application      │
│                     (schémas `public` +      │
│                      `audit`, gérés par      │
│                      Alembic)                │
└─────────────────────────────────────────────┘
```

## 🗂️ Structure des fichiers

```
alembic/
├── README-alembic.md          # Ce fichier
├── env.py                     # Configuration Alembic pour automl_app
├── script.py.mako             # Template de migration
└── versions/                  # Fichiers de migration
    └── XXXXX_initial.py       # Migration initiale
```

## 🔧 Configuration

### alembic.ini

La configuration principale pointe vers la base **automl_app** via :

```ini
[alembic]
script_location = %(here)s/alembic
# L'URL est récupérée depuis AUTOML_DATABASE_URL
```

### Variables d'environnement

```bash
# Base application (Auth, Users, Tenants, Projects)
AUTOML_DATABASE_URL=postgresql://user:pass@postgres:5432/automl_app

# Base audit (Logs, RGPD, conformité) - partage la base automl_app (schéma `audit`)
AUTOML_AUDIT_DATABASE_URL=postgresql://user:pass@postgres:5432/automl_app

# Base MLflow (tracking - géré par MLflow, pas Alembic)
MLFLOW_DATABASE_URL=postgresql://user:pass@postgres:5432/automl
```

## 🚀 Commandes principales

### Générer une nouvelle migration

```bash
# Depuis le conteneur Docker
docker exec -it automl_api_dev alembic revision --autogenerate -m "add user table"

# Ou en local
alembic revision --autogenerate -m "add user table"
```

### Appliquer les migrations

```bash
# Depuis le conteneur Docker
docker exec -it automl_api_dev alembic upgrade head

# Ou en local
alembic upgrade head
```

### Voir l'état des migrations

```bash
# Version actuelle
alembic current

# Historique complet
alembic history --verbose
```

### Revenir en arrière

```bash
# Revenir d'une migration
alembic downgrade -1

# Revenir à une version spécifique
alembic downgrade <revision_id>
```

## 🛠️ Script utilitaire

Un script Bash facilite les opérations multi-base :

```bash
# Appliquer toutes les migrations
./scripts/run_migrations.sh migrate

# Générer une nouvelle migration
./scripts/run_migrations.sh generate-app "add user table"

# Voir le statut
./scripts/run_migrations.sh status

# Aide complète
./scripts/run_migrations.sh
```

## 📦 Modèles inclus

### Base automl_app

Les migrations gèrent les tables suivantes :

- **tenants** : Locataires multi-tenant (unifié auth + infrastructure)
- **users** : Utilisateurs avec SSO
- **roles** : Rôles RBAC
- **permissions** : Permissions granulaires
- **api_keys** : Clés API pour authentification machine
- **audit_logs** : Logs d'audit de base
- **projects** : Projets des utilisateurs
- **user_roles, role_permissions, project_users** : Tables de liaison

### Schéma audit (dans automl_app, optionnel)

Si configuré via `alembic_audit.ini`, gère :

- **audit_events** : Événements d'audit détaillés
- **gdpr_requests** : Requêtes RGPD/GDPR
- **consent_records** : Consentements utilisateurs
- **data_retention** : Politiques de rétention

### Base automl (MLflow)

**Non géré par Alembic** - MLflow crée et maintient ses propres tables :
- experiments, runs, metrics, params, tags
- registered_models, model_versions
- datasets, inputs

## ⚠️ Bonnes pratiques

### Avant de générer une migration

1. **Vérifier les modèles** :
   ```bash
   # S'assurer que tous les modèles sont importés dans alembic/env.py
   grep "from automl_platform" alembic/env.py
   ```

2. **Tester en local** :
   ```bash
   # Base de test
   alembic upgrade head
   alembic downgrade base
   ```

3. **Review le code généré** :
   ```python
   # Vérifier alembic/versions/XXXXX_*.py
   # Ajuster si nécessaire (indexes, contraintes, etc.)
   ```

### En développement

```bash
# Reset complet si nécessaire
docker exec -it automl_postgres_dev psql -U automl -d automl_app -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"

# Regénérer tout
rm alembic/versions/*.py
alembic revision --autogenerate -m "initial"
alembic upgrade head
```

### En production

```bash
# TOUJOURS sauvegarder avant migration
pg_dump automl_app > backup_$(date +%Y%m%d_%H%M%S).sql

# Tester sur staging d'abord
alembic upgrade head --sql > migration.sql  # Générer SQL
# Review du SQL, puis apply

# En production, avec downtime
alembic upgrade head
```

## 🔍 Debugging

### Migration échoue

```bash
# Voir les détails
alembic upgrade head --verbose

# Vérifier l'état
alembic current
alembic history

# Forcer stamp si table alembic_version corrompue
alembic stamp head
```

### Conflit de schéma

```bash
# Comparer DB vs modèles
alembic revision --autogenerate -m "check_diff"
# Voir le fichier généré dans versions/
```

### Connexion DB échoue

```bash
# Vérifier variables d'environnement
echo $AUTOML_DATABASE_URL

# Test connexion directe
psql $AUTOML_DATABASE_URL -c "SELECT version();"
```

## 📚 Ressources

- [Documentation Alembic](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Models](https://docs.sqlalchemy.org/en/20/orm/)
- [Migration Best Practices](https://alembic.sqlalchemy.org/en/latest/tutorial.html)

## 🏗️ Workflow de développement

1. **Modifier un modèle SQLAlchemy**
   ```python
   # Exemple: automl_platform/models/tenant.py
   class Tenant(Base):
       new_field = Column(String(100))
   ```

2. **Générer la migration**
   ```bash
   docker exec -it automl_api_dev alembic revision --autogenerate -m "add tenant new_field"
   ```

3. **Review le fichier généré**
   ```bash
   # Éditer alembic/versions/XXXXX_add_tenant_new_field.py
   # Vérifier upgrade() et downgrade()
   ```

4. **Appliquer en dev**
   ```bash
   docker exec -it automl_api_dev alembic upgrade head
   ```

5. **Tester l'application**
   ```bash
   # Vérifier que tout fonctionne
   docker logs automl_api_dev -f
   ```

6. **Commit**
   ```bash
   git add alembic/versions/XXXXX_add_tenant_new_field.py
   git commit -m "feat: add tenant new_field"
   ```

---

**Note** : Ce README est maintenu manuellement. Pour toute question, consulter l'équipe DevOps.
