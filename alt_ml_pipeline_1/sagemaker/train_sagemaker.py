"""
SageMaker Training Script for Alt ML Pipeline 1
This script is executed inside the SageMaker training container
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Disable WandB before any imports
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

# SageMaker directories
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_OUTPUT_DATA_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')

from ultralytics import YOLO, settings
import torch

# Disable WandB in Ultralytics
settings['wandb'] = False


def train_yolo_sagemaker(args):
    """
    Train YOLO model on SageMaker

    Args:
        args: Command line arguments
    """
    print("=" * 80)
    print("YOLO Training on SageMaker")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")
    print(f"Device: {args.device}")
    print()

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ No GPU detected, using CPU (slower)")
    print()

    # Find dataset.yaml in training channel
    data_yaml = Path(SM_CHANNEL_TRAINING) / 'data.yaml'
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found in {SM_CHANNEL_TRAINING}")

    print(f"Dataset config: {data_yaml}")
    print()

    # Initialize model
    model = YOLO(args.model)

    # Training parameters
    train_params = {
        'data': str(data_yaml),
        'epochs': args.epochs,
        'imgsz': args.image_size,
        'batch': args.batch_size,
        'device': args.device,
        'patience': args.patience,
        'save': True,
        'project': SM_MODEL_DIR,
        'name': 'training',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': args.optimizer,
        'lr0': args.learning_rate,
        'weight_decay': args.weight_decay,
        'plots': True,
        'save_period': 10,
        'val': True,
        'workers': args.workers
    }

    print("Starting training...")
    print("-" * 80)

    # Train
    results = model.train(**train_params)

    print()
    print("-" * 80)
    print("Training complete!")
    print()

    # Evaluate
    print("Evaluating on validation set...")
    metrics = model.val()

    # Extract metrics
    results_dict = {
        'mAP50': float(metrics.box.map50) if hasattr(metrics, 'box') else 0.0,
        'mAP50-95': float(metrics.box.map) if hasattr(metrics, 'box') else 0.0,
        'precision': float(metrics.box.mp) if hasattr(metrics, 'box') else 0.0,
        'recall': float(metrics.box.mr) if hasattr(metrics, 'box') else 0.0,
    }

    # Calculate F1
    if results_dict['precision'] + results_dict['recall'] > 0:
        results_dict['f1'] = 2 * (results_dict['precision'] * results_dict['recall']) / \
                             (results_dict['precision'] + results_dict['recall'])
    else:
        results_dict['f1'] = 0.0

    print()
    print("=" * 80)
    print("Final Results:")
    print("=" * 80)
    for key, value in results_dict.items():
        print(f"{key:15s}: {value:.4f}")
    print("=" * 80)
    print()

    # Save metrics to output
    metrics_file = Path(SM_OUTPUT_DATA_DIR) / 'metrics.json'
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Metrics saved to: {metrics_file}")

    # Copy best model to model directory
    best_model = Path(SM_MODEL_DIR) / 'training' / 'weights' / 'best.pt'
    if best_model.exists():
        import shutil
        shutil.copy(best_model, Path(SM_MODEL_DIR) / 'best.pt')
        print(f"Best model copied to: {Path(SM_MODEL_DIR) / 'best.pt'}")

    print()
    print("✓ SageMaker training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO on SageMaker')

    # Model parameters
    parser.add_argument('--model', type=str, default='yolov8n-seg.pt',
                       help='YOLO model to use')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (can be larger on GPU)')
    parser.add_argument('--image-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')

    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, default='AdamW',
                       help='Optimizer (AdamW, SGD, Adam)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Weight decay')

    # Hardware
    parser.add_argument('--device', type=str, default='0',
                       help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loading workers')

    args = parser.parse_args()

    # Auto-detect device
    if args.device == 'auto':
        args.device = '0' if torch.cuda.is_available() else 'cpu'

    train_yolo_sagemaker(args)
