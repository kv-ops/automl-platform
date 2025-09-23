# Multi-stage Dockerfile for AutoML Platform
# Supports both API backend and UI dashboard
# Version 3.2.1 - With integrated connectors support

# ============================================================================
# Stage 1: Base dependencies
# ============================================================================
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    gcc \
    g++ \
    git \
    curl \
    wget \
    build-essential \
    pkg-config \
    # PostgreSQL client
    libpq-dev \
    postgresql-client \
    # OpenMP for LightGBM
    libgomp1 \
    # Graphics libraries for visualization
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf-2.0-dev \
    # PDF generation
    libffi-dev \
    libssl-dev \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # Scientific computing
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    # Monitoring tools
    htop \
    vim \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================================================
# Stage 2: Python dependencies builder
# ============================================================================
FROM base as builder

# Copy requirements files if they exist (fallback to setup.py)
COPY requirements*.txt ./
COPY setup.py ./
COPY README.md ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install all requirements
# First try requirements.txt if it exists, otherwise use setup.py
RUN if [ -f "requirements.txt" ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir .; \
    fi

# ============================================================================
# Stage 3: Runtime environment
# ============================================================================
FROM base as runtime

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . /app

# The .streamlit folder is already at the root level from COPY . /app
# Ensure it's also in the home directory for Streamlit to find it
RUN cp -r /app/.streamlit ~/.streamlit 2>/dev/null || true

# Install the package in development mode
# This will use setup.py which now includes all connectors in install_requires
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data \
    /app/models \
    /app/logs \
    /app/uploads \
    /app/reports \
    /app/cache \
    /app/mlflow \
    /app/artifacts

# Set consistent environment variables for Streamlit
ENV STREAMLIT_SERVER_ENABLE_CORS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash automl && \
    chown -R automl:automl /app && \
    chown -R automl:automl ~/.streamlit

# ============================================================================
# Stage 4: Development environment (optional)
# ============================================================================
FROM runtime as development

# Install additional development tools
RUN pip install --no-cache-dir \
    pytest>=8.0.0 \
    pytest-cov>=4.1.0 \
    pytest-asyncio>=0.23.0 \
    black>=24.0.0 \
    ruff>=0.2.0 \
    mypy>=1.8.0 \
    ipython>=8.0.0 \
    jupyter>=1.0.0

# Enable hot reload for development
ENV STREAMLIT_DEVELOPMENT_MODE=true \
    PYTHONDONTWRITEBYTECODE=0

# ============================================================================
# Stage 5a: API Server
# ============================================================================
FROM runtime as api

# Switch to non-root user
USER automl

# Expose API port
EXPOSE 8000

# Set environment variables with defaults
ENV PORT=8000 \
    API_BASE_URL=${API_BASE_URL:-http://localhost:8000} \
    MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000}

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default command for API
CMD ["sh", "-c", "uvicorn automl_platform.api.api:app \
     --host 0.0.0.0 \
     --port ${PORT} \
     --workers 4 \
     --loop uvloop \
     --access-log \
     --log-level info"]

# ============================================================================
# Stage 5b: UI Dashboard (Streamlit) - FIXED CONFIG
# ============================================================================
FROM runtime as ui

# Switch to non-root user
USER automl

# Expose Streamlit port
EXPOSE 8501

# Set environment variables for configuration - ALIGNED WITH XSRF
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    API_BASE_URL=${API_BASE_URL:-http://localhost:8000} \
    MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000} \
    AUTOML_EXPERT_MODE=${AUTOML_EXPERT_MODE:-false}

# Health check for UI
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command for UI - REMOVED conflicting CLI flags
CMD ["streamlit", "run", \
     "automl_platform/ui/dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]

# ============================================================================
# Stage 5c: Worker (Celery)
# ============================================================================
FROM runtime as worker

# Switch to non-root user
USER automl

# Set environment variables
ENV C_FORCE_ROOT=false \
    CELERY_BROKER_URL=${CELERY_BROKER_URL:-redis://localhost:6379/0} \
    CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND:-redis://localhost:6379/0}

# Default command for worker
CMD ["celery", "-A", "automl_platform.worker.celery_app", \
     "worker", \
     "--loglevel=info", \
     "--concurrency=4", \
     "--max-tasks-per-child=1000", \
     "--time-limit=3600", \
     "--soft-time-limit=3300"]

# ============================================================================
# Stage 5d: Scheduler (Celery Beat)
# ============================================================================
FROM runtime as scheduler

# Switch to non-root user
USER automl

# Set environment variables
ENV CELERY_BROKER_URL=${CELERY_BROKER_URL:-redis://localhost:6379/0} \
    CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND:-redis://localhost:6379/0}

# Default command for scheduler
CMD ["celery", "-A", "automl_platform.worker.celery_app", \
     "beat", \
     "--loglevel=info", \
     "--scheduler=django_celery_beat.schedulers:DatabaseScheduler"]

# ============================================================================
# Stage 5e: Flower (Celery monitoring)
# ============================================================================
FROM runtime as flower

# Switch to non-root user
USER automl

# Expose Flower port
EXPOSE 5555

# Set environment variables
ENV CELERY_BROKER_URL=${CELERY_BROKER_URL:-redis://localhost:6379/0} \
    FLOWER_PORT=5555

# Health check for Flower
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5555/healthcheck || exit 1

# Default command for Flower
CMD ["celery", "-A", "automl_platform.worker.celery_app", \
     "flower", \
     "--port=5555", \
     "--broker_api=${CELERY_BROKER_URL}"]

# ============================================================================
# Stage 5f: All-in-one (Development/Demo) - FIXED SUPERVISOR CONFIG
# ============================================================================
FROM runtime as all-in-one

# Install supervisor for process management
USER root
RUN apt-get update && apt-get install -y supervisor && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for all services - ALIGNED
ENV API_BASE_URL=${API_BASE_URL:-http://localhost:8000} \
    MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000} \
    AUTOML_EXPERT_MODE=${AUTOML_EXPERT_MODE:-false} \
    CELERY_BROKER_URL=${CELERY_BROKER_URL:-redis://localhost:6379/0} \
    CELERY_RESULT_BACKEND=${CELERY_RESULT_BACKEND:-redis://localhost:6379/0} \
    STREAMLIT_SERVER_ENABLE_CORS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Create supervisor configuration with proper environment passing - FIXED UI COMMAND
RUN echo '\
[supervisord]\n\
nodaemon=true\n\
user=root\n\
logfile=/app/logs/supervisord.log\n\
pidfile=/var/run/supervisord.pid\n\
\n\
[unix_http_server]\n\
file=/var/run/supervisor.sock\n\
chmod=0700\n\
\n\
[rpcinterface:supervisor]\n\
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface\n\
\n\
[supervisorctl]\n\
serverurl=unix:///var/run/supervisor.sock\n\
\n\
[program:api]\n\
command=/opt/venv/bin/uvicorn automl_platform.api.api:app --host 0.0.0.0 --port 8000 --workers 2\n\
directory=/app\n\
user=automl\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/app/logs/api.out.log\n\
stderr_logfile=/app/logs/api.err.log\n\
environment=PATH="/opt/venv/bin:%(ENV_PATH)s",API_BASE_URL="%(ENV_API_BASE_URL)s",MLFLOW_TRACKING_URI="%(ENV_MLFLOW_TRACKING_URI)s"\n\
priority=10\n\
\n\
[program:ui]\n\
command=/opt/venv/bin/streamlit run automl_platform/ui/dashboard.py --server.port=8501 --server.address=0.0.0.0\n\
directory=/app\n\
user=automl\n\
autostart=true\n\
autorestart=true\n\
stdout_logfile=/app/logs/ui.out.log\n\
stderr_logfile=/app/logs/ui.err.log\n\
environment=PATH="/opt/venv/bin:%(ENV_PATH)s",API_BASE_URL="%(ENV_API_BASE_URL)s",MLFLOW_TRACKING_URI="%(ENV_MLFLOW_TRACKING_URI)s",AUTOML_EXPERT_MODE="%(ENV_AUTOML_EXPERT_MODE)s",STREAMLIT_SERVER_ENABLE_CORS="true",STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION="true"\n\
priority=20\n\
\n\
[program:worker]\n\
command=/opt/venv/bin/celery -A automl_platform.worker.celery_app worker --loglevel=info --concurrency=2\n\
directory=/app\n\
user=automl\n\
autostart=false\n\
autorestart=true\n\
stdout_logfile=/app/logs/worker.out.log\n\
stderr_logfile=/app/logs/worker.err.log\n\
environment=PATH="/opt/venv/bin:%(ENV_PATH)s",CELERY_BROKER_URL="%(ENV_CELERY_BROKER_URL)s",CELERY_RESULT_BACKEND="%(ENV_CELERY_RESULT_BACKEND)s"\n\
priority=30\n\
\n\
[program:flower]\n\
command=/opt/venv/bin/celery -A automl_platform.worker.celery_app flower --port=5555\n\
directory=/app\n\
user=automl\n\
autostart=false\n\
autorestart=true\n\
stdout_logfile=/app/logs/flower.out.log\n\
stderr_logfile=/app/logs/flower.err.log\n\
environment=PATH="/opt/venv/bin:%(ENV_PATH)s",CELERY_BROKER_URL="%(ENV_CELERY_BROKER_URL)s"\n\
priority=40\n\
\n\
[group:automl]\n\
programs=api,ui\n\
priority=999\n\
' > /etc/supervisor/conf.d/automl.conf

# Expose all ports
EXPOSE 8000 8501 5555

# Health check for all-in-one
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health && curl -f http://localhost:8000/health || exit 1

# Run supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]

# ============================================================================
# Stage 6: Production optimized
# ============================================================================
FROM runtime as production

# Install production server
RUN pip install --no-cache-dir gunicorn>=21.2.0 uvloop>=0.19.0

# Switch to non-root user
USER automl

# Copy and setup entrypoint script
USER root
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Wait for database if needed\n\
if [ -n "$DATABASE_URL" ]; then\n\
    echo "Waiting for database..."\n\
    python -c "import time; time.sleep(5)"\n\
fi\n\
\n\
# Run migrations if needed\n\
if [ "$RUN_MIGRATIONS" = "true" ]; then\n\
    echo "Running migrations..."\n\
    python -m automl_platform.cli.migrate upgrade\n\
fi\n\
\n\
# Start the service\n\
exec "$@"\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

USER automl

ENTRYPOINT ["/entrypoint.sh"]

# Default to UI service - FIXED: No conflicting flags
CMD ["streamlit", "run", "automl_platform/ui/dashboard.py"]

# ============================================================================
# Default stage selection: UI for Render deployment
# ============================================================================
FROM ui as default
