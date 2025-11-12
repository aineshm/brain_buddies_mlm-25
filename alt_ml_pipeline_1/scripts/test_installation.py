"""
Test Installation Script
Verifies that all dependencies are installed correctly
"""

import sys
from pathlib import Path

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color


def print_test(name, passed, message=""):
    """Print test result"""
    if passed:
        print(f"{GREEN}✓{NC} {name}")
    else:
        print(f"{RED}✗{NC} {name}")
        if message:
            print(f"  {message}")


def test_imports():
    """Test that all required packages can be imported"""
    print("\n" + "=" * 60)
    print("Testing Package Imports")
    print("=" * 60 + "\n")

    tests = []

    # Core packages
    try:
        import torch
        tests.append(("PyTorch", True, torch.__version__))
    except ImportError as e:
        tests.append(("PyTorch", False, str(e)))

    try:
        import torchvision
        tests.append(("TorchVision", True, torchvision.__version__))
    except ImportError as e:
        tests.append(("TorchVision", False, str(e)))

    try:
        import ultralytics
        tests.append(("Ultralytics (YOLO)", True, ultralytics.__version__))
    except ImportError as e:
        tests.append(("Ultralytics (YOLO)", False, str(e)))

    try:
        import cv2
        tests.append(("OpenCV", True, cv2.__version__))
    except ImportError as e:
        tests.append(("OpenCV", False, str(e)))

    try:
        import numpy
        tests.append(("NumPy", True, numpy.__version__))
    except ImportError as e:
        tests.append(("NumPy", False, str(e)))

    try:
        import pandas
        tests.append(("Pandas", True, pandas.__version__))
    except ImportError as e:
        tests.append(("Pandas", False, str(e)))

    try:
        import matplotlib
        tests.append(("Matplotlib", True, matplotlib.__version__))
    except ImportError as e:
        tests.append(("Matplotlib", False, str(e)))

    try:
        import mlflow
        tests.append(("MLflow", True, mlflow.__version__))
    except ImportError as e:
        tests.append(("MLflow", False, str(e)))

    try:
        import yaml
        tests.append(("PyYAML", True, "OK"))
    except ImportError as e:
        tests.append(("PyYAML", False, str(e)))

    try:
        import plotly
        tests.append(("Plotly", True, plotly.__version__))
    except ImportError as e:
        tests.append(("Plotly", False, str(e)))

    # Print results
    for name, passed, version in tests:
        if passed:
            print(f"{GREEN}✓{NC} {name:25s} {version}")
        else:
            print(f"{RED}✗{NC} {name:25s} {version}")

    all_passed = all(t[1] for t in tests)
    return all_passed


def test_gpu():
    """Test GPU availability"""
    print("\n" + "=" * 60)
    print("Testing GPU Availability")
    print("=" * 60 + "\n")

    import torch

    # CUDA (NVIDIA)
    cuda_available = torch.cuda.is_available()
    print_test("CUDA (NVIDIA GPU)", cuda_available)
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"  Device: {device_name}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # MPS (Apple Silicon)
    mps_available = torch.backends.mps.is_available()
    print_test("MPS (Apple Silicon GPU)", mps_available)
    if mps_available:
        print(f"  Device: Apple Silicon GPU")

    if not cuda_available and not mps_available:
        print(f"\n{YELLOW}⚠{NC} No GPU detected. Training will use CPU (slower).")
        print("  CPU training is functional but will take 4-6x longer.")

    return cuda_available or mps_available


def test_directories():
    """Test that required directories exist"""
    print("\n" + "=" * 60)
    print("Testing Directory Structure")
    print("=" * 60 + "\n")

    import os

    output_dir = Path(os.path.expanduser("~/mlm_outputs/alt_pipeline_1"))

    required_dirs = [
        output_dir / "data" / "processed",
        output_dir / "data" / "synthetic",
        output_dir / "data" / "splits",
        output_dir / "models" / "ssl_pretrained",
        output_dir / "models" / "individual",
        output_dir / "models" / "ensemble",
        output_dir / "visualizations" / "data_quality",
        output_dir / "visualizations" / "training_progress",
        output_dir / "visualizations" / "predictions",
        output_dir / "visualizations" / "confusion_matrices",
        output_dir / "results",
        output_dir / "experiments" / "mlflow"
    ]

    all_exist = True
    for dir_path in required_dirs:
        exists = dir_path.exists()
        print_test(str(dir_path.relative_to(output_dir.parent)), exists)
        if not exists:
            all_exist = False

    return all_exist


def test_config():
    """Test that config file is valid"""
    print("\n" + "=" * 60)
    print("Testing Configuration")
    print("=" * 60 + "\n")

    import yaml

    config_path = Path("configs/config.yaml")

    if not config_path.exists():
        print_test("Config file exists", False, f"{config_path} not found")
        return False

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print_test("Config file exists", True)
        print_test("Config is valid YAML", True)

        # Check required keys
        required_keys = ['project', 'data', 'training', 'evaluation']
        for key in required_keys:
            if key in config:
                print_test(f"Config has '{key}' section", True)
            else:
                print_test(f"Config has '{key}' section", False)

        return True

    except Exception as e:
        print_test("Config is valid YAML", False, str(e))
        return False


def test_data():
    """Test that YOLO dataset exists"""
    print("\n" + "=" * 60)
    print("Testing Data Availability")
    print("=" * 60 + "\n")

    yolo_dataset_path = Path("../ml_pipeline/data/processed/yolo_dataset")

    if not yolo_dataset_path.exists():
        print_test("YOLO dataset exists", False, f"{yolo_dataset_path} not found")
        print(f"\n{YELLOW}⚠{NC} Original YOLO dataset not found.")
        print("  This is OK if you plan to use different data.")
        print("  Otherwise, run data preparation in ml_pipeline first.")
        return False

    # Check for images and labels
    train_images = yolo_dataset_path / "images" / "train"
    train_labels = yolo_dataset_path / "labels" / "train"

    train_images_exist = train_images.exists()
    train_labels_exist = train_labels.exists()

    print_test("Training images directory exists", train_images_exist)
    print_test("Training labels directory exists", train_labels_exist)

    if train_images_exist:
        n_images = len(list(train_images.glob('*.tif')) + list(train_images.glob('*.tiff')))
        print(f"  Found {n_images} training images")

    if train_labels_exist:
        n_labels = len(list(train_labels.glob('*.txt')))
        print(f"  Found {n_labels} training labels")

    return train_images_exist and train_labels_exist


def test_modules():
    """Test that custom modules can be imported"""
    print("\n" + "=" * 60)
    print("Testing Custom Modules")
    print("=" * 60 + "\n")

    sys.path.insert(0, str(Path.cwd()))

    modules = [
        ("src.data.data_loader", "YOLODataLoader"),
        ("src.evaluation.visualize", "Visualizer"),
        ("src.training.train_baseline", "BaselineTrainer"),
    ]

    all_passed = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print_test(f"{module_name}.{class_name}", True)
        except Exception as e:
            print_test(f"{module_name}.{class_name}", False, str(e))
            all_passed = False

    return all_passed


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Alt ML Pipeline 1 - Installation Test")
    print("=" * 60)

    results = {
        "imports": test_imports(),
        "gpu": test_gpu(),
        "directories": test_directories(),
        "config": test_config(),
        "data": test_data(),
        "modules": test_modules()
    }

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60 + "\n")

    for test_name, passed in results.items():
        print_test(test_name.capitalize(), passed)

    # Overall result
    all_passed = all(results.values())
    data_warning = not results["data"]

    print("\n" + "=" * 60)
    if all_passed:
        print(f"{GREEN}✓ All tests passed!{NC}")
        print("=" * 60)
        print("\nYou're ready to start training!")
        print("\nNext steps:")
        print("  1. python src/data/data_loader.py")
        print("  2. python scripts/run_pipeline.py --phase foundation --fold 0 --visualize")
    elif data_warning and sum(results.values()) == len(results) - 1:
        print(f"{YELLOW}⚠ Installation OK, but data not found{NC}")
        print("=" * 60)
        print("\nCore installation is working, but the YOLO dataset is missing.")
        print("\nOptions:")
        print("  1. Run data preparation in ml_pipeline/")
        print("  2. Provide your own data")
        print("  3. Continue anyway (some tests will fail)")
    else:
        print(f"{RED}✗ Some tests failed{NC}")
        print("=" * 60)
        print("\nPlease fix the failed tests before continuing.")
        print("\nCommon fixes:")
        print("  - Ensure virtual environment is activated")
        print("  - Run: pip install -r requirements.txt")
        print("  - Run: bash scripts/setup_environment.sh")

    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
