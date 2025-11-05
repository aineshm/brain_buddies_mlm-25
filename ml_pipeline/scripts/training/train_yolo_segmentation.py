"""
YOLOv8 Instance Segmentation Training Script

Train YOLOv8 for Candida albicans cell morphology segmentation and classification.
"""

import sys
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import mlflow
from datetime import datetime
import argparse
from typing import Optional


class YOLOTrainer:
    """Trainer for YOLOv8 instance segmentation."""

    def __init__(
        self,
        data_yaml: str,
        model_size: str = 'n',
        img_size: int = 640,
        batch_size: int = 16,
        epochs: int = 100,
        patience: int = 50,
        device: str = 'auto',
        project_name: str = 'candida_segmentation',
        experiment_name: Optional[str] = None,
        resume: bool = False,
        pretrained: bool = True,
        use_mlflow: bool = True
    ):
        """
        Initialize trainer.

        Args:
            data_yaml: Path to dataset YAML configuration
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            img_size: Input image size
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            patience: Early stopping patience
            device: Device to use ('auto', 'cpu', '0', '0,1', etc.)
            project_name: Project name for saving results
            experiment_name: Experiment name (default: timestamp)
            resume: Resume from last checkpoint
            pretrained: Start from pretrained weights
            use_mlflow: Use MLflow for experiment tracking
        """
        self.data_yaml = Path(data_yaml)
        self.model_size = model_size
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.project_name = project_name
        self.resume = resume
        self.pretrained = pretrained
        self.use_mlflow = use_mlflow

        # Generate experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_name = f'yolov8{model_size}_{timestamp}'
        else:
            self.experiment_name = experiment_name

        # Setup paths
        self.output_dir = Path('ml_pipeline/results') / self.project_name / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = None

        print("="*80)
        print("YOLOv8 INSTANCE SEGMENTATION TRAINER")
        print("="*80)
        print(f"Model: YOLOv8{model_size}-seg")
        print(f"Dataset: {self.data_yaml}")
        print(f"Image size: {img_size}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Device: {device}")
        print(f"Output: {self.output_dir}")
        print("="*80 + "\n")

    def _load_model(self):
        """Load YOLOv8 segmentation model."""
        if self.resume:
            # Resume from last checkpoint
            checkpoint = self.output_dir / 'weights' / 'last.pt'
            if checkpoint.exists():
                print(f"Resuming from checkpoint: {checkpoint}")
                self.model = YOLO(str(checkpoint))
            else:
                print("⚠️  No checkpoint found for resume, starting from scratch")
                self._load_pretrained()
        else:
            self._load_pretrained()

    def _load_pretrained(self):
        """Load pretrained model or create new one."""
        if self.pretrained:
            # Load pretrained COCO weights
            model_name = f'yolov8{self.model_size}-seg.pt'
            print(f"Loading pretrained model: {model_name}")
            self.model = YOLO(model_name)
        else:
            # Create model from scratch
            model_yaml = f'yolov8{self.model_size}-seg.yaml'
            print(f"Creating model from scratch: {model_yaml}")
            self.model = YOLO(model_yaml)

    def get_training_config(self) -> dict:
        """
        Get training configuration with microscopy-specific augmentations.

        Returns:
            Dictionary of training hyperparameters
        """
        return {
            # Dataset
            'data': str(self.data_yaml),

            # Training
            'epochs': self.epochs,
            'patience': self.patience,
            'batch': self.batch_size,
            'imgsz': self.img_size,
            'device': self.device,
            'workers': 8,
            'project': str(Path('ml_pipeline/results') / self.project_name),
            'name': self.experiment_name,
            'exist_ok': True,
            'pretrained': self.pretrained,
            'optimizer': 'AdamW',  # Better for small datasets
            'verbose': True,
            'seed': 42,
            'deterministic': False,
            'single_cls': False,
            'rect': False,  # Rectangular training (faster but less augmentation)
            'cos_lr': True,  # Cosine learning rate scheduler
            'close_mosaic': 10,  # Disable mosaic in last N epochs

            # Hyperparameters
            'lr0': 0.01,  # Initial learning rate
            'lrf': 0.01,  # Final learning rate fraction
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,  # Box loss gain
            'cls': 0.5,  # Class loss gain (lower for small dataset)
            'dfl': 1.5,  # Distribution focal loss gain

            # Augmentation (adapted for microscopy)
            'hsv_h': 0.0,  # Hue (disabled for grayscale)
            'hsv_s': 0.0,  # Saturation (disabled)
            'hsv_v': 0.4,  # Value/brightness
            'degrees': 180.0,  # Rotation (full rotation for cells)
            'translate': 0.1,
            'scale': 0.5,
            'shear': 5.0,
            'perspective': 0.0,  # Disabled for microscopy
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.1,
            'copy_paste': 0.1,  # Copy-paste augmentation
            'erasing': 0.2,  # Random erasing

            # Validation
            'val': True,
            'plots': True,
            'save': True,
            'save_period': -1,  # Save every N epochs (-1 = only save final)

            # Class weights (handle imbalance)
            # Will be computed from dataset if not specified
        }

    def compute_class_weights(self) -> dict:
        """
        Compute class weights based on frequency for handling class imbalance.

        Returns:
            Dictionary mapping class names to weights
        """
        # Load dataset configuration
        with open(self.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        # Count instances per class (would need to parse label files)
        # For now, use manual counts from analysis
        class_counts = {
            'single dispersed cell': 1488,
            'clump dispersed cell': 979,
            'planktonic': 2643,
            'yeast form': 702,
            'psuedohyphae': 155,  # Rare
            'hyphae': 1306,
            'biofilm': 189
        }

        total = sum(class_counts.values())

        # Compute inverse frequency weights
        weights = {}
        for cls_name, count in class_counts.items():
            if count > 0:
                weight = total / (len(class_counts) * count)
                weights[cls_name] = weight
            else:
                weights[cls_name] = 1.0

        # Normalize weights
        max_weight = max(weights.values())
        weights = {k: v / max_weight for k, v in weights.items()}

        print("\nClass weights (for handling imbalance):")
        for cls_name, weight in weights.items():
            print(f"  {cls_name}: {weight:.3f}")

        return weights

    def train(self):
        """Run training."""
        # Load model
        self._load_model()

        # Get training config
        config = self.get_training_config()

        # Compute class weights
        class_weights = self.compute_class_weights()

        # Setup MLflow
        if self.use_mlflow:
            mlflow.set_experiment(self.project_name)
            mlflow.start_run(run_name=self.experiment_name)

            # Log parameters
            mlflow.log_params({
                'model_size': self.model_size,
                'img_size': self.img_size,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'optimizer': config['optimizer'],
                'lr0': config['lr0'],
                'pretrained': self.pretrained
            })

            # Log class weights
            for cls_name, weight in class_weights.items():
                mlflow.log_param(f'weight_{cls_name}', weight)

        # Train model
        print("\nStarting training...\n")
        results = self.model.train(**config)

        # Log metrics to MLflow
        if self.use_mlflow:
            # Get final metrics
            metrics_file = Path(config['project']) / self.experiment_name / 'results.csv'
            if metrics_file.exists():
                import pandas as pd
                metrics_df = pd.read_csv(metrics_file)

                # Log final metrics
                final_metrics = metrics_df.iloc[-1].to_dict()
                for key, value in final_metrics.items():
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        mlflow.log_metric(f'final_{key}', value)

            # Log artifacts
            mlflow.log_artifacts(str(Path(config['project']) / self.experiment_name))

            mlflow.end_run()

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Results saved to: {Path(config['project']) / self.experiment_name}")
        print(f"Best weights: {Path(config['project']) / self.experiment_name / 'weights' / 'best.pt'}")
        print("="*80 + "\n")

        return results

    def validate(self, weights_path: Optional[str] = None):
        """
        Run validation on best model.

        Args:
            weights_path: Path to weights file (default: use best.pt from training)
        """
        if weights_path is None:
            weights_path = self.output_dir / 'weights' / 'best.pt'

        print(f"\nValidating model: {weights_path}\n")

        model = YOLO(str(weights_path))
        metrics = model.val(data=str(self.data_yaml), imgsz=self.img_size)

        print("\nValidation Metrics:")
        print("-" * 60)
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")

        if hasattr(metrics, 'seg'):
            print(f"\nSegmentation Metrics:")
            print(f"mAP50 (mask): {metrics.seg.map50:.4f}")
            print(f"mAP50-95 (mask): {metrics.seg.map:.4f}")

        print("-" * 60)

        return metrics


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 instance segmentation for cell morphology'
    )

    # Dataset
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset YAML file'
    )

    # Model
    parser.add_argument(
        '--model',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLOv8 model size'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=True,
        help='Use pretrained weights'
    )

    # Training
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (auto, cpu, 0, 0,1, etc.)'
    )

    # Experiment tracking
    parser.add_argument(
        '--project',
        type=str,
        default='candida_segmentation',
        help='Project name'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking'
    )

    # Actions
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only run validation on existing model'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to weights for validation'
    )

    args = parser.parse_args()

    # Create trainer
    trainer = YOLOTrainer(
        data_yaml=args.data,
        model_size=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        project_name=args.project,
        experiment_name=args.name,
        resume=args.resume,
        pretrained=args.pretrained,
        use_mlflow=not args.no_mlflow
    )

    # Run training or validation
    if args.validate_only:
        trainer.validate(args.weights)
    else:
        trainer.train()

        # Run validation after training
        print("\nRunning final validation...\n")
        trainer.validate()


if __name__ == '__main__':
    main()
