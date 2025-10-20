#!/usr/bin/env bash
# =============================================================================
# PostgreSQL Multi-Database Initialization Script
# Creates the databases and service accounts required by the AutoML platform.
# This script is executed automatically by the postgres container entrypoint.
# =============================================================================
set -euo pipefail

log() {
    echo "[init-multi-db] $*"
}

PRIMARY_USER="${POSTGRES_USER:?POSTGRES_USER must be set}"
PRIMARY_DB="${POSTGRES_DB:-$PRIMARY_USER}"
MONITORING_USER="${POSTGRES_MONITORING_USER:?POSTGRES_MONITORING_USER must be set}"
MONITORING_PASSWORD="${POSTGRES_MONITORING_PASSWORD:?POSTGRES_MONITORING_PASSWORD must be set}"
BACKUP_USER="${POSTGRES_BACKUP_USER:?POSTGRES_BACKUP_USER must be set}"
BACKUP_PASSWORD="${POSTGRES_BACKUP_PASSWORD:?POSTGRES_BACKUP_PASSWORD must be set}"
DOLLAR_SIGN='$'

create_database() {
    local database="$1"
    log "Ensuring database '${database}' exists"
    psql -v ON_ERROR_STOP=1 --username "$PRIMARY_USER" <<EOSQL
        SELECT 'CREATE DATABASE "${database}"'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${database}')\gexec
        GRANT ALL PRIVILEGES ON DATABASE "${database}" TO "${PRIMARY_USER}";
EOSQL
}

configure_main_database() {
    log "Configuring primary database '${PRIMARY_DB}'"
    psql -v ON_ERROR_STOP=1 --username "$PRIMARY_USER" --dbname "$PRIMARY_DB" <<EOSQL
        CREATE SCHEMA IF NOT EXISTS automl AUTHORIZATION "${PRIMARY_USER}";
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";
        CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
        ALTER DATABASE "${PRIMARY_DB}" SET search_path TO automl,public;
        ALTER DEFAULT PRIVILEGES IN SCHEMA automl
            GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "${PRIMARY_USER}";
EOSQL
}

create_role_if_missing() {
    local role_name="$1"
    local role_password="$2"
    local role_comment="$3"
    log "Ensuring role '${role_name}' exists (${role_comment})"
    psql -v ON_ERROR_STOP=1 --username "$PRIMARY_USER" <<EOSQL
        DO
        ${DOLLAR_SIGN}${DOLLAR_SIGN}
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${role_name}') THEN
                EXECUTE format('CREATE ROLE %I WITH LOGIN PASSWORD %L;', '${role_name}', '${role_password}');
            END IF;
        END
        ${DOLLAR_SIGN}${DOLLAR_SIGN}
EOSQL
}

configure_monitoring_role() {
    create_role_if_missing "$MONITORING_USER" "$MONITORING_PASSWORD" "monitoring user"
    psql -v ON_ERROR_STOP=1 --username "$PRIMARY_USER" --dbname "$PRIMARY_DB" <<EOSQL
        GRANT CONNECT ON DATABASE "${PRIMARY_DB}" TO "${MONITORING_USER}";
        GRANT USAGE ON SCHEMA automl TO "${MONITORING_USER}";
        GRANT SELECT ON ALL TABLES IN SCHEMA automl TO "${MONITORING_USER}";
        ALTER DEFAULT PRIVILEGES IN SCHEMA automl
            GRANT SELECT ON TABLES TO "${MONITORING_USER}";
EOSQL
}

configure_backup_role() {
    create_role_if_missing "$BACKUP_USER" "$BACKUP_PASSWORD" "backup user"
    for database in ${POSTGRES_MULTIPLE_DATABASES//,/ }; do
        psql -v ON_ERROR_STOP=1 --username "$PRIMARY_USER" --dbname "$database" <<EOSQL
            GRANT CONNECT ON DATABASE "${database}" TO "${BACKUP_USER}";
            GRANT USAGE ON SCHEMA public TO "${BACKUP_USER}";
            GRANT SELECT ON ALL TABLES IN SCHEMA public TO "${BACKUP_USER}";
            ALTER DEFAULT PRIVILEGES IN SCHEMA public
                GRANT SELECT ON TABLES TO "${BACKUP_USER}";
            DO
            $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.schemata WHERE schema_name = 'audit'
                ) THEN
                    EXECUTE format('GRANT USAGE ON SCHEMA %I TO %I;', 'audit', '${BACKUP_USER}');
                    EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO %I;', 'audit', '${BACKUP_USER}');
                    EXECUTE format('ALTER DEFAULT PRIVILEGES IN SCHEMA %I GRANT SELECT ON TABLES TO %I;', 'audit', '${BACKUP_USER}');
                END IF;
            END
            $$;
EOSQL
    done
    psql -v ON_ERROR_STOP=1 --username "$PRIMARY_USER" <<EOSQL
        ALTER ROLE "${BACKUP_USER}" WITH REPLICATION;
EOSQL
}

configure_additional_databases() {
    if [[ -z "${POSTGRES_MULTIPLE_DATABASES:-}" ]]; then
        log "No additional databases requested"
        return
    fi

    log "Creating additional databases: ${POSTGRES_MULTIPLE_DATABASES}"
    for database in ${POSTGRES_MULTIPLE_DATABASES//,/ }; do
        [[ "$database" == "$PRIMARY_DB" ]] && continue
        create_database "$database"

        case "$database" in
            automl_app)
                psql -v ON_ERROR_STOP=1 --username "$PRIMARY_USER" --dbname "$database" <<EOSQL
                    DO
                    $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_namespace WHERE nspname = 'public'
                        ) THEN
                            EXECUTE format('CREATE SCHEMA %I AUTHORIZATION %I;', 'public', '${PRIMARY_USER}');
                        ELSE
                            EXECUTE format('ALTER SCHEMA %I OWNER TO %I;', 'public', '${PRIMARY_USER}');
                        END IF;
                    END
                    $$;

                    CREATE SCHEMA IF NOT EXISTS audit AUTHORIZATION "${PRIMARY_USER}";
                    ALTER SCHEMA audit OWNER TO "${PRIMARY_USER}";
                    ALTER DATABASE "${database}" SET search_path TO public,audit;

                    GRANT USAGE ON SCHEMA audit TO "${PRIMARY_USER}";
                    ALTER DEFAULT PRIVILEGES IN SCHEMA public
                        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "${PRIMARY_USER}";
                    ALTER DEFAULT PRIVILEGES IN SCHEMA public
                        GRANT USAGE, SELECT ON SEQUENCES TO "${PRIMARY_USER}";
                    ALTER DEFAULT PRIVILEGES IN SCHEMA audit
                        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO "${PRIMARY_USER}";
                    ALTER DEFAULT PRIVILEGES IN SCHEMA audit
                        GRANT USAGE, SELECT ON SEQUENCES TO "${PRIMARY_USER}";

                    GRANT USAGE ON SCHEMA public TO "${BACKUP_USER}";
                    GRANT SELECT ON ALL TABLES IN SCHEMA public TO "${BACKUP_USER}";
                    GRANT USAGE ON SCHEMA audit TO "${BACKUP_USER}";
                    GRANT SELECT ON ALL TABLES IN SCHEMA audit TO "${BACKUP_USER}";
                    ALTER DEFAULT PRIVILEGES IN SCHEMA public
                        GRANT SELECT ON TABLES TO "${BACKUP_USER}";
                    ALTER DEFAULT PRIVILEGES IN SCHEMA audit
                        GRANT SELECT ON TABLES TO "${BACKUP_USER}";
                EOSQL
                ;;
            keycloak)
                psql -v ON_ERROR_STOP=1 --username "$PRIMARY_USER" --dbname "$database" <<EOSQL
                    ALTER DATABASE "${database}" SET log_statement = 'all';
                    ALTER DATABASE "${database}" SET log_min_duration_statement = 250;
EOSQL
                ;;
            airflow)
                psql -v ON_ERROR_STOP=1 --username "$PRIMARY_USER" --dbname "$database" <<EOSQL
                    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
                    ALTER DATABASE "${database}" SET timezone TO 'UTC';
EOSQL
                ;;
            metadata)
                psql -v ON_ERROR_STOP=1 --username "$PRIMARY_USER" --dbname "$database" <<EOSQL
                    CREATE SCHEMA IF NOT EXISTS models;
                    CREATE SCHEMA IF NOT EXISTS experiments;
                    CREATE SCHEMA IF NOT EXISTS datasets;
EOSQL
                ;;
        esac
    done
}

apply_cluster_tuning() {
    log "Applying PostgreSQL performance settings"
    psql -v ON_ERROR_STOP=1 --username "$PRIMARY_USER" <<EOSQL
        ALTER SYSTEM SET shared_buffers = '256MB';
        ALTER SYSTEM SET work_mem = '4MB';
        ALTER SYSTEM SET maintenance_work_mem = '64MB';
        ALTER SYSTEM SET effective_cache_size = '1GB';
        ALTER SYSTEM SET checkpoint_completion_target = 0.9;
        ALTER SYSTEM SET wal_buffers = '16MB';
        ALTER SYSTEM SET max_connections = 200;
        ALTER SYSTEM SET log_statement = 'mod';
        ALTER SYSTEM SET log_min_duration_statement = 500;
        SELECT pg_reload_conf();
EOSQL
}

main() {
    configure_main_database
    configure_additional_databases
    configure_monitoring_role
    configure_backup_role
    apply_cluster_tuning
    log "PostgreSQL initialization completed"
}

main "$@"
