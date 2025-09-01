#!/bin/bash

# ==============================================================================
# AutoML Platform - MLOps Installation Script
# ==============================================================================
# This script installs all required dependencies for the MLOps features
# Usage: ./install_mlops.sh [--gpu] [--airflow|--prefect] [--full]
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
INSTALL_GPU=false
INSTALL_AIRFLOW=false
INSTALL_PREFECT=false
INSTALL_FULL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            INSTALL_GPU=true
            shift
            ;;
        --airflow)
            INSTALL_AIRFLOW=true
            shift
            ;;
        --prefect)
            INSTALL_PREFECT=true
            shift
            ;;
        --full)
            INSTALL_FULL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpu] [--airflow|--prefect] [--full]"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}AutoML Platform MLOps Installation${NC}"
echo -e "${GREEN}========================================${NC}"

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.8+ is required. Current version: $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install core requirements
echo -e "\n${YELLOW}Installing core MLOps dependencies...${NC}"
pip install --no-cache-dir \
    mlflow>=2.9.0 \
    onnx>=1.15.0 \
    onnxruntime>=1.16.0 \
    skl2onnx>=1.16.0 \
    sklearn2pmml>=0.100.0

echo -e "${GREEN}✓ Core MLOps dependencies installed${NC}"

# Install workflow orchestration
if [ "$INSTALL_AIRFLOW" = true ]; then
    echo -e "\n${YELLOW}Installing Apache Airflow...${NC}"
    AIRFLOW_VERSION=2.8.0
    PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
    CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
    
    pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
    pip install apache-airflow-providers-celery>=3.5.0
    
    # Initialize Airflow
    export AIRFLOW_HOME=${AIRFLOW_HOME:-~/airflow}
    airflow db init
    
    echo -e "${GREEN}✓ Airflow installed and initialized${NC}"
    echo -e "${YELLOW}  Run 'airflow standalone' to start Airflow${NC}"
    
elif [ "$INSTALL_PREFECT" = true ]; then
    echo -e "\n${YELLOW}Installing Prefect...${NC}"
    pip install --no-cache-dir \
        prefect>=2.14.0 \
        prefect-aws>=0.4.0
    
    echo -e "${GREEN}✓ Prefect installed${NC}"
    echo -e "${YELLOW}  Run 'prefect server start' to start Prefect${NC}"
fi

# Install GPU dependencies if requested
if [ "$INSTALL_GPU" = true ]; then
    echo -e "\n${YELLOW}Installing GPU dependencies...${NC}"
    pip install --no-cache-dir \
        torch==2.1.0+cu118 \
        torchvision==0.16.0+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118
    
    pip install --no-cache-dir \
        onnxruntime-gpu>=1.16.0 \
        cupy-cuda11x>=12.3.0
    
    echo -e "${GREEN}✓ GPU dependencies installed${NC}"
fi

# Install full requirements if requested
if [ "$INSTALL_FULL" = true ]; then
    echo -e "\n${YELLOW}Installing full requirements...${NC}"
    pip install --no-cache-dir -r requirements.txt
    echo -e "${GREEN}✓ All requirements installed${NC}"
fi

# Setup MLflow
echo -e "\n${YELLOW}Setting up MLflow...${NC}"
mkdir -p mlruns
mkdir -p mlflow_artifacts

# Create MLflow configuration
cat > mlflow_config.py << 'EOF'
import mlflow
import os

# Set tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Create default experiment
experiment_name = "automl_default"
try:
    mlflow.create_experiment(experiment_name)
    print(f"Created experiment: {experiment_name}")
except:
    print(f"Experiment {experiment_name} already exists")

print("MLflow setup complete!")
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
EOF

python mlflow_config.py
rm mlflow_config.py

# Create start scripts
echo -e "\n${YELLOW}Creating start scripts...${NC}"

# MLflow start script
cat > start_mlflow.sh << 'EOF'
#!/bin/bash
echo "Starting MLflow server..."
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlflow_artifacts \
    --serve-artifacts
EOF
chmod +x start_mlflow.sh

# API start script
cat > start_api.sh << 'EOF'
#!/bin/bash
echo "Starting AutoML API..."
source venv/bin/activate
uvicorn automl_platform.main:app --reload --host 0.0.0.0 --port 8000
EOF
chmod +x start_api.sh

# Dashboard start script
cat > start_dashboard.sh << 'EOF'
#!/bin/bash
echo "Starting Streamlit Dashboard..."
source venv/bin/activate
streamlit run automl_platform/ui/dashboard.py --server.port 8501
EOF
chmod +x start_dashboard.sh

echo -e "${GREEN}✓ Start scripts created${NC}"

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"

python << 'EOF'
import sys
errors = []

# Check core MLOps packages
packages = {
    'mlflow': 'MLflow',
    'onnx': 'ONNX',
    'onnxruntime': 'ONNX Runtime',
    'skl2onnx': 'sklearn to ONNX converter',
}

for package, name in packages.items():
    try:
        __import__(package)
        print(f"✓ {name} installed")
    except ImportError:
        errors.append(name)
        print(f"✗ {name} not found")

if errors:
    print(f"\n⚠ Missing packages: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n✓ All core packages verified")
EOF

# Final instructions
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Quick Start Guide:${NC}"
echo -e "1. Start MLflow:     ${GREEN}./start_mlflow.sh${NC}"
echo -e "2. Start API:        ${GREEN}./start_api.sh${NC}"
echo -e "3. Start Dashboard:  ${GREEN}./start_dashboard.sh${NC}"

if [ "$INSTALL_AIRFLOW" = true ]; then
    echo -e "4. Start Airflow:    ${GREEN}airflow standalone${NC}"
elif [ "$INSTALL_PREFECT" = true ]; then
    echo -e "4. Start Prefect:    ${GREEN}prefect server start${NC}"
fi

echo -e "\n${YELLOW}Access Points:${NC}"
echo -e "- MLflow UI:     ${GREEN}http://localhost:5000${NC}"
echo -e "- API Docs:      ${GREEN}http://localhost:8000/docs${NC}"
echo -e "- Dashboard:     ${GREEN}http://localhost:8501${NC}"

if [ "$INSTALL_AIRFLOW" = true ]; then
    echo -e "- Airflow UI:    ${GREEN}http://localhost:8080${NC}"
elif [ "$INSTALL_PREFECT" = true ]; then
    echo -e "- Prefect UI:    ${GREEN}http://localhost:4200${NC}"
fi

echo -e "\n${YELLOW}Test the installation:${NC}"
echo -e "${GREEN}python automl_platform/examples/mlops_integration.py${NC}"

echo -e "\n${GREEN}Happy MLOps! 🚀${NC}"
