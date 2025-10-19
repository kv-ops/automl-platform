#!/bin/bash
# Script pour exécuter les migrations Alembic multi-base
# Usage: ./scripts/run_migrations.sh {migrate|generate-app|generate-audit|status}
#
# Architecture:
#   - automl       : MLflow tracking (géré par MLflow, pas Alembic)
#   - automl_app   : Auth + Application (migrations via alembic.ini)
#   - automl_audit : Audit + RGPD (migrations via alembic_audit.ini si créé)

set -e

echo "========================================="
echo "AutoML Platform - Database Migrations"
echo "========================================="

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérifier que les variables d'environnement sont définies
check_env() {
    if [ -z "$AUTOML_DATABASE_URL" ]; then
        error "AUTOML_DATABASE_URL not set"
        exit 1
    fi
    
    if [ -z "$AUDIT_DATABASE_URL" ]; then
        error "AUDIT_DATABASE_URL not set"
        exit 1
    fi
    
    info "Environment variables validated"
}

# Migration pour automl_app (utilise alembic.ini par défaut)
migrate_app() {
    info "Migrating automl_app database..."
    
    if [ ! -f "alembic.ini" ]; then
        error "alembic.ini not found"
        exit 1
    fi
    
    # Créer le dossier versions si nécessaire
    mkdir -p alembic/versions
    
    # Appliquer les migrations
    alembic upgrade head
    
    if [ $? -eq 0 ]; then
        info "automl_app migrations completed ✓"
    else
        error "automl_app migrations failed"
        exit 1
    fi
}

# Migration pour automl_audit (optionnel si alembic_audit.ini existe)
migrate_audit() {
    # Vérifier si une configuration séparée existe pour audit
    if [ ! -f "alembic_audit.ini" ]; then
        warn "alembic_audit.ini not found - skipping audit migrations"
        warn "Audit tables can be managed via main alembic.ini if needed"
        return 0
    fi
    
    info "Migrating automl_audit database..."
    
    # Créer le dossier versions si nécessaire
    mkdir -p alembic/versions/audit
    
    # Appliquer les migrations
    alembic -c alembic_audit.ini upgrade head
    
    if [ $? -eq 0 ]; then
        info "automl_audit migrations completed ✓"
    else
        error "automl_audit migrations failed"
        exit 1
    fi
}

# Générer une nouvelle migration pour app
generate_app_migration() {
    local message=$1
    if [ -z "$message" ]; then
        error "Migration message required"
        echo "Usage: $0 generate-app 'migration message'"
        exit 1
    fi
    
    info "Generating migration for automl_app: $message"
    alembic revision --autogenerate -m "$message"
}

# Générer une nouvelle migration pour audit
generate_audit_migration() {
    local message=$1
    if [ -z "$message" ]; then
        error "Migration message required"
        echo "Usage: $0 generate-audit 'migration message'"
        exit 1
    fi
    
    info "Generating migration for automl_audit: $message"
    alembic -c alembic_audit.ini revision --autogenerate -m "$message"
}

# Afficher l'état des migrations
show_status() {
    info "Migration status for automl_app:"
    alembic current
    echo ""
    alembic history --verbose | head -20
    
    if [ -f "alembic_audit.ini" ]; then
        echo ""
        info "Migration status for automl_audit:"
        alembic -c alembic_audit.ini current
        echo ""
        alembic -c alembic_audit.ini history --verbose | head -20
    else
        echo ""
        warn "No separate audit configuration found (alembic_audit.ini)"
    fi
}

# Menu principal
case "$1" in
    migrate)
        check_env
        migrate_app
        migrate_audit
        info "All migrations completed successfully ✓"
        ;;
    generate-app)
        generate_app_migration "$2"
        ;;
    generate-audit)
        if [ ! -f "alembic_audit.ini" ]; then
            error "alembic_audit.ini not found. Create it first or use main alembic.ini"
            exit 1
        fi
        generate_audit_migration "$2"
        ;;
    status)
        show_status
        ;;
    init)
        info "Initializing Alembic for automl_app..."
        if [ ! -d "alembic" ]; then
            alembic init alembic
            info "Alembic initialized. Now edit alembic/env.py to import your models."
        else
            warn "alembic/ directory already exists"
        fi
        ;;
    *)
        echo "AutoML Platform - Database Migration Tool"
        echo ""
        echo "Usage: $0 {migrate|generate-app|generate-audit|status|init} [message]"
        echo ""
        echo "Commands:"
        echo "  migrate              - Run all pending migrations"
        echo "  generate-app MSG     - Generate new app migration (alembic.ini)"
        echo "  generate-audit MSG   - Generate new audit migration (alembic_audit.ini)"
        echo "  status               - Show migration status"
        echo "  init                 - Initialize Alembic structure"
        echo ""
        echo "Examples:"
        echo "  $0 migrate"
        echo "  $0 generate-app 'add user table'"
        echo "  $0 generate-audit 'add gdpr consent'"
        echo "  $0 status"
        echo ""
        echo "Notes:"
        echo "  - automl       : MLflow database (managed by MLflow, not Alembic)"
        echo "  - automl_app   : Application database (uses alembic.ini)"
        echo "  - automl_audit : Audit database (uses alembic_audit.ini if exists)"
        exit 1
        ;;
esac
