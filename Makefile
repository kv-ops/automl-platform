# ============================================================================
# AutoML Platform - Makefile
# Simplified commands for Docker operations and development workflow
# ============================================================================

.PHONY: help install setup up down restart logs clean test deploy status backup restore \
prod dev stop secrets ssl-cert

# Variables
DOCKER_COMPOSE = docker compose
ENV_FILE = .env
BACKUP_DIR = backups
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
PROD_ENV_FILE ?= .env.production
DEV_ENV_FILE ?= .env.development
DEV_COMPOSE_FILE ?= docker-compose.dev.yml

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
help:
	@echo "Available commands:"
	@echo "  make secrets   - Generate .env.production/.env.staging/.env.development"
	@echo "  make dev       - Launch the lightweight development stack"
	@echo "  make prod      - Launch the full production stack"
	@echo "  make stop      - Stop running containers"
	@echo "  make logs      - Tail Docker Compose logs"
	@echo "  make clean     - Remove containers (keeps data volumes)"
	@echo ""
	@echo "Extended commands remain available for advanced workflows."
	@echo "Run 'make status' to inspect container health or consult README-DOCKER.md for a full guide."

# ============================================================================
# Installation & Setup
# ============================================================================

install: check-requirements setup-env init-dirs pull build migrate up-min
	@echo "$(GREEN)✅ Installation complete!$(NC)"
	@echo "$(BLUE)Access the platform at:$(NC)"
	@echo "  - Dashboard: http://localhost:8501"
	@echo "  - API Docs:  http://localhost:8000/docs"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. make ui        # Open dashboard"
	@echo "  2. make status    # Check services"
	@echo "  3. make help      # See all commands"

check-requirements:
	@echo "$(BLUE)Checking requirements...$(NC)"
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)Docker is required but not installed.$(NC)" >&2; exit 1; }
	@command -v docker compose version >/dev/null 2>&1 || { echo "$(RED)Docker Compose v2 is required but not installed.$(NC)" >&2; exit 1; }
	@echo "$(GREEN)✓ Docker and Docker Compose found$(NC)"

setup-env:
	@echo "$(BLUE)Setting up environment...$(NC)"
	@if [ ! -f $(ENV_FILE) ]; then \
		cp .env.example $(ENV_FILE); \
		echo "$(GREEN)✓ Created .env file from template$(NC)"; \
		echo "$(YELLOW)⚠️  Please edit .env with your configuration$(NC)"; \
	else \
		echo "$(GREEN)✓ .env file already exists$(NC)"; \
	fi

init-dirs:
	@echo "$(BLUE)Creating directories...$(NC)"
	@mkdir -p $(BACKUP_DIR) logs data models uploads reports
	@mkdir -p monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources
	@mkdir -p dags airflow/logs airflow/plugins
	@echo "$(GREEN)✓ Directories created$(NC)"

# ============================================================================
# Service Management
# ============================================================================

# Minimal setup (for development)
up-min:
	@echo "$(BLUE)Starting minimal services...$(NC)"
	@$(DOCKER_COMPOSE) up -d postgres redis minio mlflow api ui worker-cpu
	@echo "$(GREEN)✓ Minimal services started$(NC)"
	@$(MAKE) wait-healthy
	@$(MAKE) status

# Standard setup
up:
	@echo "$(BLUE)Starting standard services...$(NC)"
	@$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@$(MAKE) wait-healthy
	@$(MAKE) status

# Full setup with monitoring
up-full:
	@echo "$(BLUE)Starting full platform with monitoring...$(NC)"
	@$(DOCKER_COMPOSE) --profile monitoring up -d
	@echo "$(GREEN)✓ Full platform started$(NC)"
	@$(MAKE) wait-healthy
	@$(MAKE) status

# With GPU support
up-gpu:
	@echo "$(BLUE)Starting with GPU support...$(NC)"
	@$(DOCKER_COMPOSE) --profile gpu --profile monitoring up -d
	@echo "$(GREEN)✓ GPU services started$(NC)"
	@$(MAKE) wait-healthy
	@$(MAKE) status

# Production setup
up-prod:
	@echo "$(BLUE)Starting production environment...$(NC)"
	@$(DOCKER_COMPOSE) --profile production --profile monitoring --profile auth --profile premium up -d
	@echo "$(GREEN)✓ Production environment started$(NC)"
	@$(MAKE) wait-healthy
	@$(MAKE) status

# Stop all services
down:
	@echo "$(BLUE)Stopping all services...$(NC)"
	@$(DOCKER_COMPOSE) --profile production --profile monitoring --profile auth --profile gpu --profile premium down
	@echo "$(GREEN)✓ All services stopped$(NC)"

# Restart services
restart:
	@echo "$(BLUE)Restarting services...$(NC)"
	@$(MAKE) down
	@sleep 2
	@$(MAKE) up
	@echo "$(GREEN)✓ Services restarted$(NC)"

# Wait for services to be healthy
wait-healthy:
	@echo "$(BLUE)Waiting for services to be healthy...$(NC)"
	@sleep 5
	@timeout 60 bash -c 'until docker exec automl_postgres pg_isready; do sleep 2; done' || true
	@timeout 60 bash -c 'until curl -f http://localhost:8000/health > /dev/null 2>&1; do sleep 2; done' || true
	@timeout 60 bash -c 'until curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; do sleep 2; done' || true

# ============================================================================
# Development
# ============================================================================

# Development mode with hot reload
dev:
	@echo "$(BLUE)Starting lightweight development stack...$(NC)"
	@if [ ! -f $(DEV_COMPOSE_FILE) ]; then \
		echo "$(RED)Missing $(DEV_COMPOSE_FILE).$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f $(DEV_ENV_FILE) ]; then \
		echo "$(YELLOW)$(DEV_ENV_FILE) not found. Falling back to .env example.$(NC)"; \
		cp .env.example $(DEV_ENV_FILE); \
	fi
	@DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) -f $(DEV_COMPOSE_FILE) --env-file $(DEV_ENV_FILE) up -d --build
	@echo "$(GREEN)Development stack is running$(NC)"

prod:
	@echo "$(BLUE)Starting production stack...$(NC)"
	@if [ ! -f $(PROD_ENV_FILE) ]; then \
		echo "$(RED)Missing $(PROD_ENV_FILE). Run 'make secrets' first.$(NC)"; \
		exit 1; \
	fi
	@$(MAKE) ssl-cert
	@DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) --env-file $(PROD_ENV_FILE) --profile production up -d
	@echo "$(GREEN)✓ Production stack is running$(NC)"

stop:
	@echo "$(BLUE)Stopping containers...$(NC)"
	@if [ ! -f $(PROD_ENV_FILE) ]; then \
		echo "$(RED)Missing $(PROD_ENV_FILE). Run 'make secrets' first.$(NC)"; \
		exit 1; \
	fi
	@$(DOCKER_COMPOSE) --env-file $(PROD_ENV_FILE) --profile production down
	@echo "$(GREEN)✓ Containers stopped$(NC)"

secrets:
	@echo "$(BLUE)Generating environment secrets...$(NC)"
	@./scripts/generate-secrets.sh --force
	@echo "$(GREEN)✓ Secrets generated in .env.production, .env.staging, and .env.development$(NC)"

# Run tests
test:
	@echo "$(BLUE)Running all tests...$(NC)"
	@docker exec automl_api pytest tests/ -v --cov=automl_platform --cov-report=term-missing

test-ui:
	@echo "$(BLUE)Running UI tests...$(NC)"
	@docker exec automl_ui pytest tests/ui/ -v

test-api:
	@echo "$(BLUE)Running API tests...$(NC)"
	@docker exec automl_api pytest tests/api/ -v

# Code quality
lint:
	@echo "$(BLUE)Running linters...$(NC)"
	@docker exec automl_api ruff check automl_platform/
	@docker exec automl_api mypy automl_platform/

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	@docker exec automl_api black automl_platform/
	@docker exec automl_api isort automl_platform/

# ============================================================================
# Monitoring & Logs
# ============================================================================

logs:
	@$(DOCKER_COMPOSE) --env-file $(PROD_ENV_FILE) logs -f --tail=100

logs-ui:
	@$(DOCKER_COMPOSE) logs -f ui --tail=100

logs-api:
	@$(DOCKER_COMPOSE) logs -f api --tail=100

logs-worker:
	@$(DOCKER_COMPOSE) logs -f worker-cpu --tail=100

logs-error:
	@$(DOCKER_COMPOSE) logs --tail=1000 | grep -E "ERROR|CRITICAL|Exception|Failed"

status:
	@echo "$(BLUE)╔═══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║                     Service Status                           ║$(NC)"
	@echo "$(BLUE)╚═══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep automl || true
	@echo ""
	@echo "$(GREEN)Access URLs:$(NC)"
	@echo "  📊 Dashboard:  http://localhost:8501"
	@echo "  🔌 API Docs:   http://localhost:8000/docs"
	@echo "  📈 MLflow:     http://localhost:5000"
	@echo "  🌸 Flower:     http://localhost:5555"
	@echo "  💾 MinIO:      http://localhost:9001"
	@if [ "$$(docker ps -q -f name=automl_grafana)" ]; then \
		echo "  📊 Grafana:    http://localhost:3000"; \
	fi

monitor:
	@echo "$(BLUE)Opening monitoring dashboards...$(NC)"
	@open http://localhost:3000 || xdg-open http://localhost:3000 || echo "Grafana: http://localhost:3000"
	@open http://localhost:9090 || xdg-open http://localhost:9090 || echo "Prometheus: http://localhost:9090"
	@open http://localhost:5555 || xdg-open http://localhost:5555 || echo "Flower: http://localhost:5555"

# ============================================================================
# Quick Access
# ============================================================================

ui:
	@echo "$(BLUE)Opening UI Dashboard...$(NC)"
	@open http://localhost:8501 || xdg-open http://localhost:8501 || echo "Dashboard: http://localhost:8501"

api:
	@echo "$(BLUE)Opening API Documentation...$(NC)"
	@open http://localhost:8000/docs || xdg-open http://localhost:8000/docs || echo "API: http://localhost:8000/docs"

mlflow:
	@echo "$(BLUE)Opening MLflow...$(NC)"
	@open http://localhost:5000 || xdg-open http://localhost:5000 || echo "MLflow: http://localhost:5000"

flower:
	@echo "$(BLUE)Opening Flower...$(NC)"
	@open http://localhost:5555 || xdg-open http://localhost:5555 || echo "Flower: http://localhost:5555"

grafana:
	@echo "$(BLUE)Opening Grafana...$(NC)"
	@open http://localhost:3000 || xdg-open http://localhost:3000 || echo "Grafana: http://localhost:3000"

minio:
	@echo "$(BLUE)Opening MinIO Console...$(NC)"
	@open http://localhost:9001 || xdg-open http://localhost:9001 || echo "MinIO: http://localhost:9001"

# ============================================================================
# Shell Access
# ============================================================================

shell-ui:
	@docker exec -it automl_ui /bin/bash

shell-api:
	@docker exec -it automl_api /bin/bash

shell-worker:
	@docker exec -it automl_worker_cpu /bin/bash

db-shell:
	@docker exec -it automl_postgres psql -U automl -d automl

redis-cli:
	@docker exec -it automl_redis redis-cli -a redis_secret

# ============================================================================
# Database Management
# ============================================================================

migrate:
	@echo "$(BLUE)Running database migrations...$(NC)"
	@docker exec automl_api alembic upgrade head
	@echo "$(GREEN)✓ Migrations completed$(NC)"

migrate-create name=change:
	@echo "$(BLUE)Creating new migration: $(name)$(NC)"
	@docker exec automl_api alembic revision --autogenerate -m "$(name)"

db-reset:
	@echo "$(YELLOW)⚠️  This will reset the database. Are you sure? [y/N]$(NC)"
	@read -r confirm && [ "$$confirm" = "y" ] && \
		docker exec automl_api alembic downgrade base && \
		docker exec automl_api alembic upgrade head && \
		echo "$(GREEN)✓ Database reset completed$(NC)"

# ============================================================================
# Data Management
# ============================================================================

backup:
	@echo "$(BLUE)Creating backup...$(NC)"
	@mkdir -p $(BACKUP_DIR)
	@docker exec automl_postgres pg_dump -U automl automl > $(BACKUP_DIR)/db_$(TIMESTAMP).sql
	@docker cp automl_api:/app/models $(BACKUP_DIR)/models_$(TIMESTAMP) 2>/dev/null || true
	@docker cp automl_api:/app/data $(BACKUP_DIR)/data_$(TIMESTAMP) 2>/dev/null || true
	@echo "$(GREEN)✓ Backup created in $(BACKUP_DIR)/$(NC)"
	@ls -lh $(BACKUP_DIR)/*_$(TIMESTAMP)*

restore:
	@echo "$(YELLOW)Available backups:$(NC)"
	@ls -1 $(BACKUP_DIR)/*.sql | tail -5
	@echo ""
	@read -p "Enter backup filename (without path): " backup_file; \
	if [ -f "$(BACKUP_DIR)/$$backup_file" ]; then \
		docker cp $(BACKUP_DIR)/$$backup_file automl_postgres:/tmp/restore.sql && \
		docker exec automl_postgres psql -U automl automl < /tmp/restore.sql && \
		echo "$(GREEN)✓ Database restored from $$backup_file$(NC)"; \
	else \
		echo "$(RED)✗ Backup file not found$(NC)"; \
	fi

# ============================================================================
# Building & Deployment
# ============================================================================

build:
	@echo "$(BLUE)Building all images...$(NC)"
	@DOCKER_BUILDKIT=1 $(DOCKER_COMPOSE) build --parallel
	@echo "$(GREEN)✓ Build completed$(NC)"

build-ui:
	@echo "$(BLUE)Building UI image...$(NC)"
	@DOCKER_BUILDKIT=1 docker build --target ui -t automl-ui:latest .
	@echo "$(GREEN)✓ UI image built$(NC)"

build-api:
	@echo "$(BLUE)Building API image...$(NC)"
	@DOCKER_BUILDKIT=1 docker build --target api -t automl-api:latest .
	@echo "$(GREEN)✓ API image built$(NC)"

push:
	@echo "$(BLUE)Pushing images to registry...$(NC)"
	@docker tag automl-ui:latest $(REGISTRY)/automl-ui:latest
	@docker tag automl-api:latest $(REGISTRY)/automl-api:latest
	@docker push $(REGISTRY)/automl-ui:latest
	@docker push $(REGISTRY)/automl-api:latest
	@echo "$(GREEN)✓ Images pushed$(NC)"

deploy:
	@echo "$(BLUE)Deploying to production...$(NC)"
	@$(MAKE) build
	@$(MAKE) push
	@echo "$(GREEN)✓ Deployment completed$(NC)"

scale: n ?= 2
scale:
	@echo "$(BLUE)Scaling workers to $(n) instances...$(NC)"
	@$(DOCKER_COMPOSE) up -d --scale worker-cpu=$(n)
	@echo "$(GREEN)✓ Scaled to $(n) workers$(NC)"

ssl-cert:
	@if [ ! -f nginx/ssl/tls.crt ] || [ ! -f nginx/ssl/tls.key ]; then \
		echo "$(YELLOW)Generating self-signed certificate for nginx...$(NC)"; \
		mkdir -p nginx/ssl; \
		openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
		  -keyout nginx/ssl/tls.key \
		  -out nginx/ssl/tls.crt \
		  -subj "/CN=localhost"; \
		chmod 600 nginx/ssl/tls.key; \
		echo "$(GREEN)✓ Self-signed certificate generated$(NC)"; \
	else \
		echo "$(GREEN)✓ SSL certificate already present$(NC)"; \
	fi

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "$(YELLOW)Stopping and removing containers...$(NC)"
	@$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Containers removed$(NC)"

clean-data:
	@echo "$(RED)⚠️  This will delete all data! Continue? [y/N]$(NC)"
	@read -r confirm && [ "$$confirm" = "y" ] && \
		$(DOCKER_COMPOSE) down -v && \
		echo "$(GREEN)✓ All data removed$(NC)"

clean-all: clean-data
	@echo "$(YELLOW)Removing images...$(NC)"
	@docker images | grep automl | awk '{print $$3}' | xargs -r docker rmi -f
	@echo "$(GREEN)✓ Complete cleanup done$(NC)"

prune:
	@echo "$(BLUE)Pruning Docker resources...$(NC)"
	@docker system prune -af --volumes
	@echo "$(GREEN)✓ Docker pruned$(NC)"

# ============================================================================
# Utility
# ============================================================================

update:
	@echo "$(BLUE)Updating dependencies...$(NC)"
	@git pull
	@$(MAKE) build
	@echo "$(GREEN)✓ Update completed$(NC)"

version:
	@echo "$(BLUE)Version Information:$(NC)"
	@echo "Docker:         $$(docker --version)"
	@echo "Docker Compose: $$(docker compose version)"
	@echo "Platform:       $$(cat automl_platform/__version__.py | grep version | cut -d'"' -f2)"

.DEFAULT_GOAL := help
