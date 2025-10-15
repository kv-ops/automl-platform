#!/usr/bin/env bash
# =============================================================================
# Backup automation for the AutoML platform.
# Creates PostgreSQL dumps, MinIO object snapshots, and archives configuration
# files into the backups/ directory.
# =============================================================================
set -euo pipefail

BACKUP_ROOT="backups"
ENV_FILE=".env.production"
RETENTION_DAYS=7
DOCKER_COMPOSE="docker compose"

usage() {
    cat <<USAGE
Usage: ./scripts/backup.sh [options]
  -e, --env-file FILE     Environment file to load (default: .env.production)
  -o, --output DIR        Backup directory (default: backups)
  -r, --retention DAYS    Retain backups for DAYS (default: 7)
  -h, --help              Show this help message
USAGE
}

require_tools() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "[ERROR] Docker is required to run backups." >&2
        exit 1
    fi
    if ! docker compose version >/dev/null 2>&1; then
        echo "[ERROR] Docker Compose v2 is required." >&2
        exit 1
    fi
}

load_env() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "[ERROR] Environment file '$file' not found." >&2
        exit 1
    fi
    set -a
    # shellcheck source=/dev/null
    source "$file"
    set +a
}

cleanup_old_backups() {
    find "$BACKUP_ROOT" -maxdepth 1 -type d -mtime +"$RETENTION_DAYS" -exec rm -rf {} + 2>/dev/null || true
}

backup_postgres() {
    local output_dir="$1"
    local filename="$output_dir/postgres_dump.sql"
    echo "[INFO] Dumping PostgreSQL databases to $filename"
    env PGPASSWORD="$POSTGRES_PASSWORD" \
        $DOCKER_COMPOSE --env-file "$ENV_FILE" exec -T postgres \
        pg_dumpall -U "$POSTGRES_USER" --clean --if-exists > "$filename"
}

backup_minio() {
    local output_dir="$1/minio"
    mkdir -p "$output_dir"
    echo "[INFO] Mirroring MinIO buckets to $output_dir"
    $DOCKER_COMPOSE --env-file "$ENV_FILE" run --rm \
        -v "$output_dir:/backup" \
        minio-init /bin/sh -c "\
            mc alias set myminio http://minio:9000 '${MINIO_ACCESS_KEY}' '${MINIO_SECRET_KEY}' >/dev/null && \
            mc mirror --quiet myminio /backup"
}

backup_configs() {
    local output_dir="$1/configs"
    mkdir -p "$output_dir"
    echo "[INFO] Archiving configuration files"
    tar -czf "$output_dir/nginx.tgz" nginx || true
    tar -czf "$output_dir/keycloak.tgz" keycloak || true
    tar -czf "$output_dir/monitoring.tgz" monitoring || true
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -e|--env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            -o|--output)
                BACKUP_ROOT="$2"
                shift 2
                ;;
            -r|--retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "[ERROR] Unknown option: $1" >&2
                usage
                exit 1
                ;;
        esac
    done

    require_tools
    load_env "$ENV_FILE"

    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local destination="$BACKUP_ROOT/$timestamp"
    mkdir -p "$destination"

    backup_postgres "$destination"
    backup_minio "$destination"
    backup_configs "$destination"
    cleanup_old_backups

    echo "[INFO] Backup completed: $destination"
}

main "$@"
