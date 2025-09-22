# Multi-stage Dockerfile for AutoML Platform
# Supports both API backend and UI dashboard

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

# Copy requirements files
COPY requirements.txt ./

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install all requirements from single file
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 3: Runtime environment
# ============================================================================
FROM base as runtime

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . /app

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/data \
    /app/models \
    /app/logs \
    /app/uploads \
    /app/reports \
    /app/cache \
    /app/mlflow \
    /app/artifacts \
    ~/.streamlit

# Configure Streamlit
RUN echo '\
[general]\n\
email = ""\n\
\n\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
maxUploadSize = 1000\n\
enableXsrfProtection = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
serverAddress = "localhost"\n\
serverPort = 8501\n\
\n\
[theme]\n\
primaryColor = "#1E88E5"\n\
backgroundColor = "#FFFFFF"\n\
secondaryBackgroundColor = "#F0F2F6"\n\
textColor = "#262730"\n\
font = "sans serif"\n\
\n\
[runner]\n\
magicEnabled = true\n\
installTracer = false\n\
fixMatplotlib = true\n\
' > ~/.streamlit/config.toml

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
    mypy>=1.8.0

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

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for API
CMD ["uvicorn", "automl_platform.api.api:app", \
     "--host", "0.0.0.0", \
     "--port", "${PORT:-8000}", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--access-log", \
     "--log-level", "info"]

# ============================================================================
# Stage 5b: UI Dashboard
# ============================================================================
FROM runtime as ui

# Switch to non-root user
USER automl

# Expose Streamlit port
EXPOSE 8501

# Health check for UI
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command for UI
CMD ["streamlit", "run", \
     "automl_platform/ui/dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]

# ============================================================================
# Stage 5c: Worker (Celery)
# ============================================================================
FROM runtime as worker

# Switch to non-root user
USER automl

# Default command for worker
CMD ["celery", "-A", "automl_platform.worker.celery_app", \
     "worker", \
     "--loglevel=info", \
     "--concurrency=4", \
     "--max-tasks-per-child=1000"]

# ============================================================================
# Stage 5d: All-in-one (Development/Demo)
# ============================================================================
FROM runtime as all-in-one

# Install supervisor for process management
USER root
RUN apt-get update && apt-get install -y supervisor && \
    rm -rf /var/lib/apt/lists/*

# Create supervisor configuration
RUN echo '\
[supervisord]\n\
nodaemon=true\n\
user=root\n\
\n\
[program:api]\n\
command=/opt/venv/bin/uvicorn automl_platform.api.api:app --host 0.0.0.0 --port 8000\n\
directory=/app\n\
user=automl\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/app/logs/api.err.log\n\
stdout_logfile=/app/logs/api.out.log\n\
\n\
[program:ui]\n\
command=/opt/venv/bin/streamlit run automl_platform/ui/dashboard.py --server.port=8501 --server.address=0.0.0.0\n\
directory=/app\n\
user=automl\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/app/logs/ui.err.log\n\
stdout_logfile=/app/logs/ui.out.log\n\
\n\
[program:worker]\n\
command=/opt/venv/bin/celery -A automl_platform.worker.celery_app worker --loglevel=info\n\
directory=/app\n\
user=automl\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/app/logs/worker.err.log\n\
stdout_logfile=/app/logs/worker.out.log\n\
\n\
[program:flower]\n\
command=/opt/venv/bin/celery -A automl_platform.worker.celery_app flower --port=5555\n\
directory=/app\n\
user=automl\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/app/logs/flower.err.log\n\
stdout_logfile=/app/logs/flower.out.log\n\
' > /etc/supervisor/conf.d/automl.conf

# Expose all ports
EXPOSE 8000 8501 5555

# Run supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]

# ============================================================================
# Default stage selection
# ============================================================================
FROM ui as default
