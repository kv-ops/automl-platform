#!/bin/bash
# =============================================================================
# AutoML Platform - SaaS Deployment Script
# =============================================================================
# This script automates the deployment of the AutoML Platform in SaaS mode
# Usage: ./scripts/deploy_saas.sh [options]
#
# Options:
#   --env [dev|staging|prod]  : Environment to deploy (default: prod)
#   --build                   : Build images locally instead of pulling
#   --gpu                     : Enable GPU support
#   --monitoring              : Enable monitoring stack (Prometheus/Grafana)
#   --backup                  : Backup existing data before deployment
#   --restore [file]          : Restore from backup file
#   --scale [n]               : Number of worker replicas (default: 2)
#   --help                    : Show this help message

set -e

# =============================================================================
# Configuration
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="prod"
BUILD_IMAGES=false
ENABLE_GPU=false
ENABLE_MONITORING=false
BACKUP_DATA=false
RESTORE_FILE=""
WORKER_SCALE=2
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Deployment directory
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_section() {
    echo -e "\n${BLUE}=========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================${NC}\n"
}

show_help() {
    cat << EOF
AutoML Platform - SaaS Deployment Script

Usage: $0 [options]

Options:
    --env [dev|staging|prod]  Environment to deploy (default: prod)
    --build                   Build images locally instead of pulling
    --gpu                     Enable GPU support
    --monitoring              Enable monitoring stack
    --backup                  Backup existing data before deployment
    --restore [file]          Restore from backup file
    --scale [n]               Number of worker replicas (default: 2)
    --help                    Show this help message

Examples:
    # Production deployment with monitoring
    $0 --env prod --monitoring

    # Development deployment with local build
    $0 --env dev --build

    # Production with GPU support and 4 workers
    $0 --env prod --gpu --scale 4

    # Restore from backup
    $0 --restore backup_20240101_120000.tar.gz
EOF
    exit 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --build)
            BUILD_IMAGES=true
            shift
            ;;
        --gpu)
            ENABLE_GPU=true
            shift
            ;;
        --monitoring)
            ENABLE_MONITORING=true
            shift
            ;;
        --backup)
            BACKUP_DATA=true
            shift
            ;;
        --restore)
            RESTORE_FILE="$2"
            shift 2
            ;;
        --scale)
            WORKER_SCALE="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# =============================================================================
# Pre-flight Checks
# =============================================================================

log_section "Pre-flight Checks"

# Check Docker
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed. Please install Docker first."
    exit 1
fi
log_info "Docker version: $(docker --version)"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
log_info "Docker Compose found"

# Check if we're in the right directory
if [ ! -f "$DEPLOY_DIR/docker-compose.yml" ]; then
    log_error "docker-compose.yml not found. Please run this script from the project root."
    exit 1
fi

# Check GPU support if requested
if [ "$ENABLE_GPU" = true ]; then
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi not found. GPU support may not be available."
        read -p "Continue without GPU? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        ENABLE_GPU=false
    else
        log_info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    fi
fi

# =============================================================================
# Environment Configuration
# =============================================================================

log_section "Environment Configuration"

# Create .env file if it doesn't exist
if [ ! -f "$DEPLOY_DIR/$ENV_FILE" ]; then
    log_info "Creating .env file from .env.example..."
    
    # Copy template
    cp "$DEPLOY_DIR/.env.example" "$DEPLOY_DIR/$ENV_FILE"
    
    # Add SaaS-specific configurations
    cat >> "$DEPLOY_DIR/$ENV_FILE" << EOF

# =============================================================================
# SaaS Mode Configuration (Auto-generated)
# =============================================================================

# Enable SaaS mode
AUTOML_MODE=saas
AUTOML_EXPERT_MODE=false

# SSO Configuration
SSO_ENABLED=true
KEYCLOAK_ENABLED=true
KEYCLOAK_CLIENT_SECRET=$(openssl rand -base64 32)

# Multi-tenant
MULTI_TENANT_ENABLED=true
DEFAULT_TENANT_PLAN=trial

# Secure passwords (auto-generated)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
MINIO_ROOT_PASSWORD=$(openssl rand -base64 32)
JWT_SECRET_KEY=$(openssl rand -base64 48)
# =============================================================================
# AutoML Platform - Environment Configuration
# Generated: $(date)
# Environment: ${ENVIRONMENT}
# =============================================================================

# Environment
ENVIRONMENT=${ENVIRONMENT}
AUTOML_MODE=saas
AUTOML_EXPERT_MODE=false

# Database
POSTGRES_USER=automl
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_PORT=5432

# Redis
REDIS_PASSWORD=$(openssl rand -base64 32)
REDIS_PORT=6379

# MinIO (Object Storage)
MINIO_ACCESS_KEY=$(openssl rand -hex 16)
MINIO_SECRET_KEY=$(openssl rand -base64 32)
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001

# API Configuration
API_PORT=8000
API_MAX_WORKERS=4
JWT_SECRET_KEY=$(openssl rand -base64 48)

# UI Configuration
UI_PORT=8501
UI_FRAMEWORK=streamlit
MAX_UPLOAD_SIZE=1000
PUBLIC_API_URL=http://localhost:8000
PUBLIC_UI_URL=http://localhost:8501

# SSO Configuration
SSO_ENABLED=true
KEYCLOAK_ENABLED=true
KEYCLOAK_HOSTNAME=localhost
KEYCLOAK_PORT=8080
KEYCLOAK_REALM=automl
KEYCLOAK_CLIENT_ID=automl-platform
KEYCLOAK_CLIENT_SECRET=$(openssl rand -base64 32)
KEYCLOAK_ADMIN=admin
KEYCLOAK_ADMIN_PASSWORD=$(openssl rand -base64 24)

# MLflow
MLFLOW_PORT=5000
MLFLOW_USER=mlflow
MLFLOW_PASSWORD=$(openssl rand -base64 24)

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=$(openssl rand -base64 24)
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Worker Configuration
CPU_WORKER_REPLICAS=${WORKER_SCALE}
CPU_WORKER_CONCURRENCY=4
CPU_WORKER_CORES=4
CPU_WORKER_MEMORY=8

# Flower (Celery Monitoring)
FLOWER_PORT=5555
FLOWER_USER=admin
FLOWER_PASSWORD=$(openssl rand -base64 24)

# LLM APIs (Optional)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
ENABLE_AI_ASSISTANT=false

# Billing (Optional)
BILLING_ENABLED=false
STRIPE_API_KEY=
STRIPE_WEBHOOK_SECRET=

# Multi-tenant
MULTI_TENANT_ENABLED=true
DEFAULT_TENANT_PLAN=trial

# Logging
LOG_LEVEL=info

# Backup
BACKUP_ENABLED=true
BACKUP_RETENTION_DAYS=30
EOF
    log_info ".env file created with secure random passwords"
else
    log_info "Using existing .env file"
fi

# Load environment variables
source "$DEPLOY_DIR/$ENV_FILE"

# =============================================================================
# Backup Existing Data
# =============================================================================

if [ "$BACKUP_DATA" = true ] || [ -n "$RESTORE_FILE" ]; then
    log_section "Data Management"
    
    BACKUP_DIR="$DEPLOY_DIR/backups"
    mkdir -p "$BACKUP_DIR"
    
    if [ "$BACKUP_DATA" = true ]; then
        log_info "Creating backup of existing data..."
        
        BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).tar.gz"
        
        # Stop services before backup
        log_info "Stopping services..."
        docker-compose -f "$DEPLOY_DIR/$COMPOSE_FILE" stop
        
        # Backup volumes
        docker run --rm \
            -v automl_postgres_data:/postgres \
            -v automl_minio_data:/minio \
            -v automl_redis_data:/redis \
            -v "$BACKUP_DIR":/backup \
            alpine tar czf "/backup/$BACKUP_FILE" \
            /postgres /minio /redis 2>/dev/null || true
        
        log_info "Backup saved to: $BACKUP_DIR/$BACKUP_FILE"
    fi
    
    if [ -n "$RESTORE_FILE" ]; then
        if [ ! -f "$BACKUP_DIR/$RESTORE_FILE" ]; then
            log_error "Restore file not found: $BACKUP_DIR/$RESTORE_FILE"
            exit 1
        fi
        
        log_info "Restoring from backup: $RESTORE_FILE"
        
        # Stop services before restore
        docker-compose -f "$DEPLOY_DIR/$COMPOSE_FILE" down
        
        # Restore volumes
        docker run --rm \
            -v automl_postgres_data:/postgres \
            -v automl_minio_data:/minio \
            -v automl_redis_data:/redis \
            -v "$BACKUP_DIR":/backup \
            alpine tar xzf "/backup/$RESTORE_FILE" -C /
        
        log_info "Restore completed"
    fi
fi

# =============================================================================
# Build or Pull Images
# =============================================================================

log_section "Container Images"

cd "$DEPLOY_DIR"

if [ "$BUILD_IMAGES" = true ]; then
    log_info "Building Docker images locally..."
    
    # Build main SaaS image if dockerfile.saas exists
    if [ -f "dockerfile.saas" ]; then
        log_info "Building SaaS all-in-one image..."
        docker build -f dockerfile.saas -t automl-platform:saas .
    fi
    
    # Build using docker-compose
    COMPOSE_COMMAND="docker-compose -f $COMPOSE_FILE"
    
    if [ "$ENABLE_GPU" = true ]; then
        COMPOSE_COMMAND="$COMPOSE_COMMAND --profile gpu"
    fi
    
    if [ "$ENABLE_MONITORING" = true ]; then
        COMPOSE_COMMAND="$COMPOSE_COMMAND --profile monitoring"
    fi
    
    $COMPOSE_COMMAND build --parallel
else
    log_info "Pulling Docker images..."
    docker-compose -f "$COMPOSE_FILE" pull
fi

# =============================================================================
# Deploy Services
# =============================================================================

log_section "Deploying Services"

# Prepare docker-compose command
COMPOSE_COMMAND="docker-compose -f $COMPOSE_FILE"

# Add profiles if enabled
PROFILES=""
if [ "$ENABLE_GPU" = true ]; then
    PROFILES="$PROFILES --profile gpu"
fi

if [ "$ENABLE_MONITORING" = true ]; then
    PROFILES="$PROFILES --profile monitoring"
fi

# Always enable production profile in prod environment
if [ "$ENVIRONMENT" = "prod" ]; then
    PROFILES="$PROFILES --profile production"
fi

# Start services
log_info "Starting services..."
$COMPOSE_COMMAND $PROFILES up -d

# Wait for services to be healthy
log_info "Waiting for services to be healthy..."

wait_for_service() {
    local service=$1
    local port=$2
    local path=${3:-/}
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "http://localhost:$port$path" > /dev/null 2>&1; then
            log_info "$service is ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_warn "$service may not be fully ready"
    return 1
}

# Wait for core services
wait_for_service "PostgreSQL" 5432 || true
wait_for_service "Redis" 6379 || true
wait_for_service "MinIO" 9000 "/minio/health/live"
wait_for_service "Keycloak" 8080 "/health/ready"
wait_for_service "API" 8000 "/health"
wait_for_service "UI" 8501 "/_stcore/health"
wait_for_service "MLflow" 5000 "/health"

if [ "$ENABLE_MONITORING" = true ]; then
    wait_for_service "Prometheus" 9090 "/-/healthy"
    wait_for_service "Grafana" 3000 "/api/health"
fi

# =============================================================================
# Post-Deployment Configuration
# =============================================================================

log_section "Post-Deployment Configuration"

# Initialize Keycloak realm if needed
if [ "$SSO_ENABLED" = "true" ] && [ "$KEYCLOAK_ENABLED" = "true" ]; then
    log_info "Configuring Keycloak..."
    
    # Wait a bit more for Keycloak to fully initialize
    sleep 10
    
    # Check if realm exists or needs to be created
    if ! curl -s -f "http://localhost:${KEYCLOAK_PORT}/realms/${KEYCLOAK_REALM}" > /dev/null 2>&1; then
        log_info "Creating Keycloak realm..."
        # Add realm creation logic here if needed
    fi
fi

# Scale workers
if [ "$WORKER_SCALE" -gt 1 ]; then
    log_info "Scaling workers to $WORKER_SCALE replicas..."
    docker-compose -f "$COMPOSE_FILE" up -d --scale worker-cpu=$WORKER_SCALE
fi

# =============================================================================
# Display Summary
# =============================================================================

log_section "Deployment Complete!"

echo -e "${GREEN}AutoML Platform SaaS has been successfully deployed!${NC}\n"

echo "Access URLs:"
echo "==========================================="
echo -e "${BLUE}Web UI:${NC}          http://localhost:${UI_PORT}"
echo -e "${BLUE}API:${NC}             http://localhost:${API_PORT}"
echo -e "${BLUE}API Docs:${NC}        http://localhost:${API_PORT}/docs"
echo -e "${BLUE}MLflow:${NC}          http://localhost:${MLFLOW_PORT}"
echo -e "${BLUE}MinIO Console:${NC}   http://localhost:${MINIO_CONSOLE_PORT}"
echo -e "${BLUE}Flower:${NC}          http://localhost:${FLOWER_PORT}"

if [ "$SSO_ENABLED" = "true" ]; then
    echo -e "${BLUE}Keycloak:${NC}        http://localhost:${KEYCLOAK_PORT}"
    echo -e "  Admin Console:  http://localhost:${KEYCLOAK_PORT}/admin"
    echo -e "  Admin User:     ${KEYCLOAK_ADMIN}"
    echo -e "  Admin Password: Check .env file"
fi

if [ "$ENABLE_MONITORING" = true ]; then
    echo -e "${BLUE}Grafana:${NC}         http://localhost:${GRAFANA_PORT}"
    echo -e "  User:          ${GRAFANA_USER}"
    echo -e "  Password:      Check .env file"
    echo -e "${BLUE}Prometheus:${NC}      http://localhost:${PROMETHEUS_PORT}"
fi

echo ""
echo "Default Credentials:"
echo "==========================================="
echo "All passwords have been generated and stored in: $DEPLOY_DIR/.env"
echo ""
echo "To view logs:"
echo "  docker-compose -f $COMPOSE_FILE logs -f [service_name]"
echo ""
echo "To stop services:"
echo "  docker-compose -f $COMPOSE_FILE stop"
echo ""
echo "To remove everything:"
echo "  docker-compose -f $COMPOSE_FILE down -v"
echo ""

# =============================================================================
# Health Check Summary
# =============================================================================

log_info "Running final health checks..."

HEALTH_CHECK_FAILED=false

check_service_health() {
    local service=$1
    local url=$2
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} $service"
    else
        echo -e "  ${RED}✗${NC} $service"
        HEALTH_CHECK_FAILED=true
    fi
}

echo ""
echo "Service Health Status:"
echo "==========================================="
check_service_health "API" "http://localhost:${API_PORT}/health"
check_service_health "UI" "http://localhost:${UI_PORT}/_stcore/health"
check_service_health "MLflow" "http://localhost:${MLFLOW_PORT}/health"
check_service_health "MinIO" "http://localhost:${MINIO_API_PORT}/minio/health/live"
check_service_health "Keycloak" "http://localhost:${KEYCLOAK_PORT}/health/ready"

if [ "$ENABLE_MONITORING" = true ]; then
    check_service_health "Prometheus" "http://localhost:${PROMETHEUS_PORT}/-/healthy"
    check_service_health "Grafana" "http://localhost:${GRAFANA_PORT}/api/health"
fi

echo ""

if [ "$HEALTH_CHECK_FAILED" = true ]; then
    log_warn "Some services may not be fully operational. Check logs for details."
else
    log_info "All services are healthy and running!"
fi

# =============================================================================
# Save Deployment Info
# =============================================================================

DEPLOYMENT_INFO="$DEPLOY_DIR/deployment_info.json"
cat > "$DEPLOYMENT_INFO" << EOF
{
  "deployment_date": "$(date -Iseconds)",
  "environment": "${ENVIRONMENT}",
  "version": "$(git describe --tags --always 2>/dev/null || echo 'unknown')",
  "services": {
    "api": "http://localhost:${API_PORT}",
    "ui": "http://localhost:${UI_PORT}",
    "mlflow": "http://localhost:${MLFLOW_PORT}",
    "keycloak": "http://localhost:${KEYCLOAK_PORT}",
    "minio": "http://localhost:${MINIO_CONSOLE_PORT}"
  },
  "features": {
    "gpu_enabled": ${ENABLE_GPU},
    "monitoring_enabled": ${ENABLE_MONITORING},
    "sso_enabled": ${SSO_ENABLED},
    "multi_tenant": ${MULTI_TENANT_ENABLED}
  },
  "workers": ${WORKER_SCALE}
}
EOF

log_info "Deployment info saved to: $DEPLOYMENT_INFO"

exit 0
