#!/bin/bash
# =============================================================================
# PostgreSQL Multi-Database Initialization Script
# =============================================================================
# This script creates multiple databases in PostgreSQL for different services
# Place this in scripts/init-multi-db.sh

set -e
set -u

# Function to create a database
function create_database() {
    local database=$1
    echo "Creating database '$database'"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        SELECT 'CREATE DATABASE $database'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
        GRANT ALL PRIVILEGES ON DATABASE $database TO $POSTGRES_USER;
EOSQL
}

# Create main database (already created by default)
echo "Configuring main database '$POSTGRES_DB'"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create schema for better organization
    CREATE SCHEMA IF NOT EXISTS automl;
    
    -- Create extensions
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";
    CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
    
    -- Performance configurations
    ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
    ALTER SYSTEM SET pg_stat_statements.track = 'all';
    ALTER SYSTEM SET pg_stat_statements.max = 10000;
    
    -- Create read-only user for monitoring
    DO
    \$\$
    BEGIN
        IF NOT EXISTS (
            SELECT FROM pg_catalog.pg_roles
            WHERE rolname = 'monitoring'
        ) THEN
            CREATE ROLE monitoring WITH LOGIN PASSWORD 'monitoring_password';
            GRANT CONNECT ON DATABASE $POSTGRES_DB TO monitoring;
            GRANT USAGE ON SCHEMA automl TO monitoring;
            GRANT SELECT ON ALL TABLES IN SCHEMA automl TO monitoring;
            ALTER DEFAULT PRIVILEGES IN SCHEMA automl GRANT SELECT ON TABLES TO monitoring;
        END IF;
    END
    \$\$;
EOSQL

# Create additional databases from POSTGRES_MULTIPLE_DATABASES environment variable
if [ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
    echo "Creating multiple databases: $POSTGRES_MULTIPLE_DATABASES"
    for db in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
        if [ "$db" != "$POSTGRES_DB" ]; then
            create_database $db
            
            # Special configuration for specific databases
            case $db in
                "keycloak")
                    echo "Configuring Keycloak database"
                    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$db" <<-EOSQL
                        -- Keycloak specific settings
                        ALTER DATABASE keycloak SET log_statement = 'all';
                        ALTER DATABASE keycloak SET log_duration = on;
EOSQL
                    ;;
                    
                "airflow")
                    echo "Configuring Airflow database"
                    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$db" <<-EOSQL
                        -- Airflow specific settings
                        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
                        ALTER DATABASE airflow SET max_connections = 200;
EOSQL
                    ;;
                    
                "metadata")
                    echo "Configuring Metadata database"
                    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$db" <<-EOSQL
                        -- Metadata store for ML models
                        CREATE SCHEMA IF NOT EXISTS models;
                        CREATE SCHEMA IF NOT EXISTS experiments;
                        CREATE SCHEMA IF NOT EXISTS datasets;
                        
                        -- Create tables for metadata
                        CREATE TABLE IF NOT EXISTS models.registry (
                            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                            name VARCHAR(255) NOT NULL,
                            version VARCHAR(50) NOT NULL,
                            algorithm VARCHAR(100),
                            metrics JSONB,
                            parameters JSONB,
                            tags JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255),
                            status VARCHAR(50) DEFAULT 'staging',
                            UNIQUE(name, version)
                        );
                        
                        CREATE TABLE IF NOT EXISTS experiments.runs (
                            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                            experiment_id UUID NOT NULL,
                            name VARCHAR(255),
                            status VARCHAR(50),
                            start_time TIMESTAMP,
                            end_time TIMESTAMP,
                            metrics JSONB,
                            params JSONB,
                            tags JSONB,
                            user_id VARCHAR(255),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        
                        CREATE TABLE IF NOT EXISTS datasets.catalog (
                            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                            name VARCHAR(255) NOT NULL UNIQUE,
                            description TEXT,
                            source VARCHAR(500),
                            format VARCHAR(50),
                            size_bytes BIGINT,
                            row_count BIGINT,
                            column_count INT,
                            schema JSONB,
                            statistics JSONB,
                            tags JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            created_by VARCHAR(255)
                        );
                        
                        -- Create indexes
                        CREATE INDEX idx_models_name ON models.registry(name);
                        CREATE INDEX idx_models_status ON models.registry(status);
                        CREATE INDEX idx_models_tags ON models.registry USING gin(tags);
                        CREATE INDEX idx_experiments_user ON experiments.runs(user_id);
                        CREATE INDEX idx_datasets_tags ON datasets.catalog USING gin(tags);
EOSQL
                    ;;
            esac
        fi
    done
fi

# Create backup user
echo "Creating backup user"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    DO
    \$\$
    BEGIN
        IF NOT EXISTS (
            SELECT FROM pg_catalog.pg_roles
            WHERE rolname = 'backup_user'
        ) THEN
            CREATE ROLE backup_user WITH LOGIN PASSWORD 'backup_password' REPLICATION;
            GRANT CONNECT ON DATABASE $POSTGRES_DB TO backup_user;
            GRANT CONNECT ON DATABASE keycloak TO backup_user;
            GRANT CONNECT ON DATABASE airflow TO backup_user;
            GRANT CONNECT ON DATABASE metadata TO backup_user;
        END IF;
    END
    \$\$;
EOSQL

# Optimize PostgreSQL settings
echo "Applying performance optimizations"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    -- Memory settings
    ALTER SYSTEM SET shared_buffers = '256MB';
    ALTER SYSTEM SET effective_cache_size = '1GB';
    ALTER SYSTEM SET maintenance_work_mem = '64MB';
    ALTER SYSTEM SET work_mem = '4MB';
    
    -- Checkpoint settings
    ALTER SYSTEM SET checkpoint_completion_target = 0.9;
    ALTER SYSTEM SET wal_buffers = '16MB';
    ALTER SYSTEM SET default_statistics_target = 100;
    ALTER SYSTEM SET random_page_cost = 1.1;
    
    -- Connection settings
    ALTER SYSTEM SET max_connections = 200;
    ALTER SYSTEM SET max_prepared_transactions = 100;
    
    -- Logging
    ALTER SYSTEM SET log_statement = 'mod';
    ALTER SYSTEM SET log_duration = off;
    ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';
    ALTER SYSTEM SET log_checkpoints = on;
    ALTER SYSTEM SET log_connections = on;
    ALTER SYSTEM SET log_disconnections = on;
    ALTER SYSTEM SET log_lock_waits = on;
    ALTER SYSTEM SET log_temp_files = 0;
    
    -- Autovacuum settings
    ALTER SYSTEM SET autovacuum = on;
    ALTER SYSTEM SET autovacuum_max_workers = 4;
    ALTER SYSTEM SET autovacuum_naptime = '30s';
    
    SELECT pg_reload_conf();
EOSQL

echo "PostgreSQL initialization completed successfully!"
