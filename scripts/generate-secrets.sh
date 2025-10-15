#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'USAGE'
Usage: ./scripts/generate-secrets.sh [--environment ENV] [--force]

Generate secure .env files with cryptographically strong secrets. By default,
production, staging, and development files are created in the project root:
  - .env.production
  - .env.staging
  - .env.development

Options:
  -e, --environment ENV   Generate secrets for a single environment
                          (production|staging|development).
  -f, --force             Overwrite existing environment files without creating
                          a timestamped backup.
  -h, --help              Show this help message.
USAGE
}

require_openssl() {
    if ! command -v openssl >/dev/null 2>&1; then
        echo "[ERROR] OpenSSL is required to generate secrets." >&2
        exit 1
    fi
}

rand_base64() {
    local length="$1"
    openssl rand -base64 "$length" | tr -d '\n'
}

rand_hex() {
    local length="$1"
    openssl rand -hex "$length" | tr -d '\n'
}

backup_file() {
    local file="$1"
    if [[ -f "$file" ]]; then
        local timestamp
        timestamp="$(date +%Y%m%d_%H%M%S)"
        cp "$file" "${file}.${timestamp}.bak"
    fi
}

write_env_file() {
    local environment="$1"
    local output_file="$2"
    local force="$3"

    if [[ -f "$output_file" && "$force" != "true" ]]; then
        backup_file "$output_file"
    fi

    local AUTOML_SECRET_KEY JWT_SECRET_KEY SESSION_SECRET_KEY
    local POSTGRES_PASSWORD POSTGRES_MONITORING_PASSWORD POSTGRES_BACKUP_PASSWORD
    local REDIS_PASSWORD MINIO_ACCESS_KEY MINIO_SECRET_KEY MINIO_SERVICE_PASSWORD
    local MLFLOW_PASSWORD KEYCLOAK_CLIENT_SECRET KEYCLOAK_ADMIN_PASSWORD
    local FLOWER_PASSWORD GRAFANA_PASSWORD STRIPE_API_KEY STRIPE_WEBHOOK_SECRET

    AUTOML_SECRET_KEY="$(rand_base64 48)"
    JWT_SECRET_KEY="$(rand_base64 48)"
    SESSION_SECRET_KEY="$(rand_base64 32)"
    POSTGRES_PASSWORD="$(rand_base64 32)"
    POSTGRES_MONITORING_PASSWORD="$(rand_base64 24)"
    POSTGRES_BACKUP_PASSWORD="$(rand_base64 24)"
    REDIS_PASSWORD="$(rand_base64 32)"
    MINIO_ACCESS_KEY="$(rand_hex 16)"
    MINIO_SECRET_KEY="$(rand_base64 32)"
    MINIO_SERVICE_PASSWORD="$(rand_base64 24)"
    MLFLOW_PASSWORD="$(rand_base64 24)"
    KEYCLOAK_CLIENT_SECRET="$(rand_base64 32)"
    KEYCLOAK_ADMIN_PASSWORD="$(rand_base64 24)"
    FLOWER_PASSWORD="$(rand_base64 24)"
    GRAFANA_PASSWORD="$(rand_base64 24)"
    STRIPE_API_KEY="$(rand_base64 32)"
    STRIPE_WEBHOOK_SECRET="$(rand_base64 32)"

    cat > "$output_file" <<ENVFILE
# ============================================================================
# AutoML Platform Secrets - ${environment^}
# Generated on $(date -u +"%Y-%m-%dT%H:%M:%SZ") using scripts/generate-secrets.sh
# ============================================================================
ENVIRONMENT=${environment}

# =========================== CORE APPLICATION ================================
AUTOML_SECRET_KEY=${AUTOML_SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
SESSION_SECRET_KEY=${SESSION_SECRET_KEY}

# =============================== DATABASE ====================================
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=automl
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=automl
POSTGRES_MULTIPLE_DATABASES=automl,keycloak,airflow,metadata
POSTGRES_MONITORING_USER=monitoring
POSTGRES_MONITORING_PASSWORD=${POSTGRES_MONITORING_PASSWORD}
POSTGRES_BACKUP_USER=backup_user
POSTGRES_BACKUP_PASSWORD=${POSTGRES_BACKUP_PASSWORD}

# ================================= REDIS =====================================
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PASSWORD}

# ================================= MINIO =====================================
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
MINIO_SERVICE_PASSWORD=${MINIO_SERVICE_PASSWORD}

# ================================= MLFLOW ====================================
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_USER=mlflow
MLFLOW_PASSWORD=${MLFLOW_PASSWORD}

# ================================= KEYCLOAK ==================================
KEYCLOAK_HOSTNAME=keycloak
KEYCLOAK_PORT=8080
KEYCLOAK_REALM=automl
KEYCLOAK_CLIENT_ID=automl-platform
KEYCLOAK_CLIENT_SECRET=${KEYCLOAK_CLIENT_SECRET}
KEYCLOAK_ADMIN=admin
KEYCLOAK_ADMIN_PASSWORD=${KEYCLOAK_ADMIN_PASSWORD}

# ================================= WORKERS ===================================
FLOWER_USER=admin
FLOWER_PASSWORD=${FLOWER_PASSWORD}

# ================================= MONITORING ================================
GRAFANA_USER=admin
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
PROMETHEUS_RETENTION=30d

# ============================ BILLING & WEBHOOKS =============================
STRIPE_API_KEY=${STRIPE_API_KEY}
STRIPE_WEBHOOK_SECRET=${STRIPE_WEBHOOK_SECRET}

# ============================= OPTIONAL PROVIDERS ============================
# Provide vendor issued credentials when enabling external integrations.
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HUGGINGFACEHUB_API_TOKEN=

# ============================== OPTIONAL OVERRIDES ===========================
# Uncomment and override as needed for ${environment} deployments.
# CORS_ORIGINS=http://localhost:3000,http://localhost:8501
# ENABLE_AI_ASSISTANT=true
# ENABLE_AUTO_ML=true
ENVFILE

    echo "[INFO] Wrote ${output_file}"
}

main() {
    local target="all"
    local force="false"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -e|--environment)
                target="$2"
                shift 2
                ;;
            -f|--force)
                force="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "[ERROR] Unknown option: $1" >&2
                show_help
                exit 1
                ;;
        esac
    done

    require_openssl

    case "$target" in
        production|staging|development)
            local file=".env.${target}"
            write_env_file "$target" "$file" "$force"
            ;;
        all)
            write_env_file production .env.production "$force"
            write_env_file staging .env.staging "$force"
            write_env_file development .env.development "$force"
            ;;
        *)
            echo "[ERROR] Invalid environment: $target" >&2
            show_help
            exit 1
            ;;
    esac
}

main "$@"
