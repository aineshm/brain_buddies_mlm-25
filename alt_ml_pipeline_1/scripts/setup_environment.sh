#!/bin/bash
# Alt ML Pipeline 1 - Environment Setup Script
# Usage: bash scripts/setup_environment.sh

set -e  # Exit on error

echo "=========================================="
echo "Alt ML Pipeline 1 - Environment Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR="$HOME/venvs/alt_pipeline_1"
PYTHON_VERSION="3.11"
PYTHON_CMD="python3.11"  # Specific Python version to use

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Python version
echo "Checking Python version..."

# First, try to find Python 3.11 specifically
if command -v $PYTHON_CMD &> /dev/null; then
    CURRENT_PYTHON=$($PYTHON_CMD --version | cut -d' ' -f2)
    print_success "Found Python $CURRENT_PYTHON at $(which $PYTHON_CMD)"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
    CURRENT_PYTHON=$(python3.11 --version | cut -d' ' -f2)
    print_success "Found Python $CURRENT_PYTHON at $(which python3.11)"
else
    print_error "Python 3.11 not found!"
    echo ""
    echo "Please install Python 3.11 first:"
    echo ""
    echo "On macOS with Homebrew:"
    echo "  brew install python@3.11"
    echo ""
    echo "On Ubuntu/Debian:"
    echo "  sudo apt install python3.11 python3.11-venv"
    echo ""
    echo "Using pyenv (recommended for managing multiple Python versions):"
    echo "  pyenv install 3.11.7"
    echo "  pyenv local 3.11.7"
    echo ""
    exit 1
fi

# Verify it's at least Python 3.11
PYTHON_MAJOR=$(echo $CURRENT_PYTHON | cut -d'.' -f1)
PYTHON_MINOR=$(echo $CURRENT_PYTHON | cut -d'.' -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 11 ]]; then
    print_warning "Python $CURRENT_PYTHON found, but 3.11+ recommended for ML libraries"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment at: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists. Remove it? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        rm -rf "$VENV_DIR"
        print_success "Removed existing virtual environment"
    else
        print_warning "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    print_success "Created virtual environment with $PYTHON_CMD"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel --quiet
print_success "Package managers upgraded"

# Install requirements
echo ""
echo "Installing requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Requirements installed"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Create output directories
echo ""
echo "Creating output directories..."
OUTPUT_DIR="$HOME/mlm_outputs/alt_pipeline_1"
mkdir -p "$OUTPUT_DIR"/{data/{processed,synthetic,splits},models/{ssl_pretrained,individual,ensemble},visualizations/{data_quality,training_progress,predictions,confusion_matrices},results/{predictions,analysis},experiments/mlflow}
print_success "Output directories created at: $OUTPUT_DIR"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}')" && print_success "PyTorch installed"
python3 -c "import ultralytics; print(f'Ultralytics {ultralytics.__version__}')" && print_success "Ultralytics installed"
python3 -c "import mlflow; print(f'MLflow {mlflow.__version__}')" && print_success "MLflow installed"

# Check for GPU
echo ""
echo "Checking GPU availability..."
GPU_CHECK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
if [ "$GPU_CHECK" == "True" ]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    print_success "GPU available: $GPU_NAME"
elif python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
    print_success "Apple Silicon GPU (MPS) available"
else
    print_warning "No GPU detected. Training will be slower on CPU."
fi

# Create activation helper script
echo ""
echo "Creating activation helper script..."
cat > activate_alt_pipeline.sh << 'EOF'
#!/bin/bash
# Activate Alt ML Pipeline 1 environment
source ~/venvs/alt_pipeline_1/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)/alt_ml_pipeline_1"
export ALT_PIPELINE_OUTPUT="$HOME/mlm_outputs/alt_pipeline_1"
echo "Alt ML Pipeline 1 environment activated"
echo "Output directory: $ALT_PIPELINE_OUTPUT"
EOF
chmod +x activate_alt_pipeline.sh
print_success "Created activation helper: ./activate_alt_pipeline.sh"

# Success message
echo ""
echo "=========================================="
print_success "Environment setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source $VENV_DIR/bin/activate"
echo "     Or use: source ./activate_alt_pipeline.sh"
echo "  2. Run pipeline: python scripts/run_pipeline.py"
echo "  3. View experiments: mlflow ui --backend-store-uri ~/mlm_outputs/alt_pipeline_1/experiments/mlflow"
echo ""
echo "To deactivate: deactivate"
echo ""
