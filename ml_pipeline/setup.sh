#!/bin/bash
# ML Pipeline Setup Script
# Run this to set up the complete pipeline

set -e  # Exit on error

echo "================================================================"
echo "Candida Albicans ML Pipeline - Setup Script"
echo "================================================================"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "  ❌ ERROR: Python 3.8+ required"
    exit 1
fi
echo "  ✓ Python version OK"
echo ""

# Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "  ⚠️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Virtual environment activated"
echo ""

# Install dependencies
echo "[4/6] Installing dependencies..."
echo "  This may take 5-10 minutes..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt

# Verify key packages
echo ""
echo "  Verifying installations:"
python3 -c "import torch; print(f'    ✓ PyTorch {torch.__version__}')"
python3 -c "import torch; print(f'    ✓ CUDA available: {torch.cuda.is_available()}')"
python3 -c "from ultralytics import YOLO; print('    ✓ Ultralytics YOLOv8')"
python3 -c "import mlflow; print(f'    ✓ MLflow {mlflow.__version__}')"
python3 -c "import cv2; print(f'    ✓ OpenCV {cv2.__version__}')"
echo ""

# Create directory structure (if not exists)
echo "[5/6] Setting up directory structure..."
mkdir -p data/raw data/processed
mkdir -p results/candida_segmentation
mkdir -p checkpoints
mkdir -p notebooks
echo "  ✓ Directories created"
echo ""

# Download YOLOv8 pretrained weights
echo "[6/6] Downloading YOLOv8 pretrained weights..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')" > /dev/null 2>&1
python3 -c "from ultralytics import YOLO; YOLO('yolov8s-seg.pt')" > /dev/null 2>&1
echo "  ✓ Pretrained weights downloaded"
echo ""

# Summary
echo "================================================================"
echo "SETUP COMPLETE!"
echo "================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Convert data:"
echo "   cd scripts/data_prep"
echo "   python convert_all_annotations.py \\"
echo "       --data-dir /Users/aineshmohan/Documents/mlm \\"
echo "       --output-dir ../../data/processed/yolo_dataset"
echo ""
echo "3. Start training:"
echo "   cd ../training"
echo "   python train_yolo_segmentation.py \\"
echo "       --data ../../data/processed/yolo_dataset/dataset.yaml \\"
echo "       --model n \\"
echo "       --epochs 100 \\"
echo "       --device 0"
echo ""
echo "See QUICKSTART.md for more details!"
echo "================================================================"
