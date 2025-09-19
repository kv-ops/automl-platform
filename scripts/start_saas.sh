#!/bin/bash
# =============================================================================
# AutoML Platform SaaS - Startup Script
# =============================================================================
# This script starts all services for the SaaS platform
# It can run API, UI, and Workers based on the SERVICE_MODE environment variable

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------

# Default service mode: all services
SERVICE_MODE=${SERVICE_MODE:-all}

# Service flags
START_API=${START_API:-true}
START_UI=${START_UI:-true}
START_WORKER=${START_WORKER:-false}
START_SCHEDULER=${START_SCHEDULER:-false}

# Determine what to start based on SERVICE_MODE
case "$SERVICE_MODE" in
    all)
        START_API=true
        START_UI=true
        START_WORKER=false
        START_SCHEDULER=false
        ;;
    api)
        START_API=true
        START_UI=false
        START_WORKER=false
        START_SCHEDULER=false
        ;;
    ui)
        START_API=false
        START_UI=true
        START_WORKER=false
        START_SCHEDULER=false
        ;;
    worker)
        START_API=false
        START_UI=false
        START_WORKER=true
        START_SCHEDULER=false
        ;;
    scheduler)
        START_API=false
        START_UI=false
        START_WORKER=false
        START_SCHEDULER=true
        ;;
esac

# -----------------------------------------------------------------------------
# Database Initialization
# -----------------------------------------------------------------------------

wait_for_service() {
    local service_name=$1
    local service_url=$2
    local max_attempts=30
    local attempt=1

    log_info "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$service_url" > /dev/null 2>&1; then
            log_info "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "$service_name failed to start after $max_attempts attempts"
    return 1
}

# Wait for database if API is starting
if [ "$START_API" = "true" ] || [ "$START_WORKER" = "true" ]; then
    log_info "Checking database connection..."
    
    # Extract database host from DATABASE_URL
    DB_HOST=$(echo $DATABASE_URL | sed -n 's/.*@\([^:\/]*\).*/\1/p')
    DB_PORT=$(echo $DATABASE_URL | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    DB_PORT=${DB_PORT:-5432}
    
    # Wait for PostgreSQL
    max_attempts=30
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if pg_isready -h "$DB_HOST" -p "$DB_PORT" > /dev/null 2>&1; then
            log_info "Database is ready!"
            break
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    # Run database migrations
    log_info "Running database migrations..."
    cd /app
    python -m automl_platform.migrations.run_migrations || log_warn "Migrations may have already been applied"
fi

# -----------------------------------------------------------------------------
# Service Initialization
# -----------------------------------------------------------------------------

# Function to create necessary directories
init_directories() {
    local dirs=("/app/data" "/app/logs" "/app/uploads" "/app/models" "/app/reports" "/app/cache")
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            log_info "Creating directory: $dir"
            mkdir -p "$dir"
        fi
    done
}

# Initialize directories
init_directories

# -----------------------------------------------------------------------------
# Start Services
# -----------------------------------------------------------------------------

# PID tracking for graceful shutdown
PIDS=()

# Trap signals for graceful shutdown
trap 'cleanup' EXIT INT TERM

cleanup() {
    log_info "Shutting down services..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    wait
    log_info "All services stopped"
    exit 0
}

# Start API Service
if [ "$START_API" = "true" ]; then
    log_info "Starting API service on port ${API_PORT:-8000}..."
    
    # Start with gunicorn for production
    if [ "$ENVIRONMENT" = "production" ]; then
        gunicorn automl_platform.api.main:app \
            --bind "${API_HOST:-0.0.0.0}:${API_PORT:-8000}" \
            --workers "${API_WORKERS:-4}" \
            --worker-class uvicorn.workers.UvicornWorker \
            --timeout 120 \
            --keep-alive 5 \
            --max-requests 1000 \
            --max-requests-jitter 50 \
            --access-logfile /app/logs/api_access.log \
            --error-logfile /app/logs/api_error.log \
            --log-level "${LOG_LEVEL:-info}" &
        PIDS+=($!)
    else
        # Development mode with auto-reload
        uvicorn automl_platform.api.main:app \
            --host "${API_HOST:-0.0.0.0}" \
            --port "${API_PORT:-8000}" \
            --reload \
            --log-level "${LOG_LEVEL:-info}" &
        PIDS+=($!)
    fi
    
    # Wait for API to be ready
    sleep 5
    wait_for_service "API" "http://localhost:${API_PORT:-8000}/health"
fi

# Start UI Service
if [ "$START_UI" = "true" ]; then
    log_info "Starting UI service on port ${UI_PORT:-8501}..."
    
    case "$UI_FRAMEWORK" in
        streamlit)
            streamlit run /app/ui/app.py \
                --server.port "${UI_PORT:-8501}" \
                --server.address "${UI_HOST:-0.0.0.0}" \
                --server.maxUploadSize "${MAX_UPLOAD_SIZE:-1000}" \
                --server.enableCORS false \
                --server.enableXsrfProtection true \
                --browser.serverAddress "${PUBLIC_UI_URL:-localhost}" \
                --browser.serverPort "${UI_PORT:-8501}" \
                --theme.primaryColor "#1E88E5" \
                --theme.backgroundColor "#FFFFFF" \
                --theme.secondaryBackgroundColor "#F0F2F6" \
                --theme.font "sans serif" &
            PIDS+=($!)
            ;;
            
        gradio)
            python /app/ui/gradio_app.py &
            PIDS+=($!)
            ;;
            
        *)
            log_error "Unknown UI framework: $UI_FRAMEWORK"
            exit 1
            ;;
    esac
    
    # Wait for UI to be ready
    sleep 10
    wait_for_service "UI" "http://localhost:${UI_PORT:-8501}/_stcore/health" || \
    wait_for_service "UI" "http://localhost:${UI_PORT:-7860}/api/health"
fi

# Start Worker Service
if [ "$START_WORKER" = "true" ]; then
    log_info "Starting Celery worker..."
    
    celery -A automl_platform.worker.celery_app worker \
        --loglevel="${LOG_LEVEL:-info}" \
        --concurrency="${WORKER_CONCURRENCY:-4}" \
        --queues="${WORKER_QUEUES:-default,cpu,training,prediction}" \
        --hostname="worker-${HOSTNAME}@%h" \
        --max-tasks-per-child=100 \
        --max-memory-per-child=2000000 \
        --logfile=/app/logs/worker.log &
    PIDS+=($!)
fi

# Start Scheduler Service (Celery Beat)
if [ "$START_SCHEDULER" = "true" ]; then
    log_info "Starting Celery Beat scheduler..."
    
    celery -A automl_platform.worker.celery_app beat \
        --loglevel="${LOG_LEVEL:-info}" \
        --pidfile=/app/celerybeat.pid \
        --schedule=/app/celerybeat-schedule &
    PIDS+=($!)
fi

# -----------------------------------------------------------------------------
# Health Monitoring
# -----------------------------------------------------------------------------

monitor_services() {
    while true; do
        sleep 60
        
        # Check API health
        if [ "$START_API" = "true" ]; then
            if ! curl -s -f "http://localhost:${API_PORT:-8000}/health" > /dev/null 2>&1; then
                log_warn "API health check failed"
            fi
        fi
        
        # Check UI health
        if [ "$START_UI" = "true" ]; then
            if ! curl -s -f "http://localhost:${UI_PORT:-8501}/_stcore/health" > /dev/null 2>&1; then
                log_warn "UI health check failed"
            fi
        fi
        
        # Log memory usage
        if command -v free > /dev/null; then
            free -h | grep Mem | awk '{print "[INFO] Memory usage: " $3 " / " $2}'
        fi
    done
}

# Start monitoring in background
monitor_services &
MONITOR_PID=$!

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------

log_info "========================================="
log_info "AutoML Platform SaaS - Started"
log_info "========================================="
log_info "Mode: ${SERVICE_MODE}"
log_info "Environment: ${ENVIRONMENT}"
log_info "Log Level: ${LOG_LEVEL}"

if [ "$START_API" = "true" ]; then
    log_info "API: http://localhost:${API_PORT:-8000}"
    log_info "API Docs: http://localhost:${API_PORT:-8000}/docs"
fi

if [ "$START_UI" = "true" ]; then
    log_info "UI: http://localhost:${UI_PORT:-8501}"
fi

if [ "$SSO_ENABLED" = "true" ]; then
    log_info "SSO: Enabled via ${KEYCLOAK_URL:-Keycloak}"
fi

log_info "========================================="
log_info "Press Ctrl+C to stop all services"
log_info "========================================="

# Wait for all background processes
wait "${PIDS[@]}"
