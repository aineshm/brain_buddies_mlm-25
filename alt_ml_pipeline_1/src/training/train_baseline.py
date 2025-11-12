"""
Baseline Training Script for Alt ML Pipeline 1
Train a simple YOLOv8 model to establish baseline performance
"""

import os
import sys
from pathlib import Path
import yaml
import json
from datetime import datetime
import numpy as np

# IMPORTANT: Disable WandB BEFORE importing ultralytics
# This prevents WandB from trying to initialize with invalid project names
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO, settings
import torch
import mlflow
import mlflow.pytorch

from src.data.data_loader import YOLODataLoader
from src.evaluation.visualize import Visualizer

# Disable WandB and built-in MLflow in Ultralytics settings
# We use our own MLflow implementation for better control
settings['wandb'] = False
settings['mlflow'] = False


class BaselineTrainer:
    """
    Train baseline YOLO model and establish performance benchmark
    """

    def __init__(self, config_path: str, fold_idx: int = 0):
        """
        Initialize baseline trainer

        Args:
            config_path: Path to config.yaml
            fold_idx: Which fold to use for training (0-4 for 5-fold CV)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.fold_idx = fold_idx
        self.output_dir = Path(os.path.expandvars(self.config['project']['output_dir']))
        self.experiment_name = f"baseline_fold_{fold_idx}"

        # Initialize data loader
        self.data_loader = YOLODataLoader(config_path)

        # Initialize visualizer
        self.visualizer = Visualizer(self.output_dir, self.config['data']['classes'])

        # Setup MLflow
        mlflow.set_tracking_uri(str(self.output_dir / 'experiments' / 'mlflow'))
        mlflow.set_experiment(self.config['experiment']['experiment_name'])

        # Device
        if self.config['hardware']['device'] == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = self.config['hardware']['device']

        print(f"Using device: {self.device}")

    def prepare_data(self):
        """
        Prepare data split for this fold
        """
        print(f"\n{'=' * 60}")
        print(f"Preparing data for fold {self.fold_idx}")
        print(f"{'=' * 60}\n")

        # Create splits
        splits = self.data_loader.create_leave_one_sequence_out_splits()

        # Prepare this fold
        self.dataset_yaml_path = self.data_loader.prepare_fold(self.fold_idx, splits)

        split = splits[self.fold_idx]
        print(f"Validation sequence: {split['val_sequence']}")
        print(f"Training images: {len(split['train_images'])}")
        print(f"Validation images: {len(split['val_images'])}")

        return split

    def train(self):
        """
        Train baseline YOLO model
        """
        print(f"\n{'=' * 60}")
        print(f"Training Baseline Model - Fold {self.fold_idx}")
        print(f"{'=' * 60}\n")

        # Get training config
        train_config = self.config['training']['baseline']

        # Initialize model
        model = YOLO(train_config['model'])

        # Training parameters
        train_params = {
            'data': self.dataset_yaml_path,
            'epochs': train_config['epochs'],
            'imgsz': train_config['image_size'],
            'batch': train_config['batch_size'],
            'device': self.device,
            'patience': train_config['patience'],
            'save': True,
            'project': str(self.output_dir / 'models' / 'individual'),
            'name': self.experiment_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': self.config['training']['optimizer']['name'],
            'lr0': self.config['training']['optimizer']['lr'],
            'weight_decay': self.config['training']['optimizer']['weight_decay'],
            'plots': True,
            'save_period': 10,
            'val': True
        }

        # Start MLflow run
        with mlflow.start_run(run_name=self.experiment_name):
            # Log parameters
            mlflow.log_params({
                'fold': self.fold_idx,
                'model': train_config['model'],
                'epochs': train_config['epochs'],
                'batch_size': train_config['batch_size'],
                'image_size': train_config['image_size'],
                'device': self.device,
                'optimizer': self.config['training']['optimizer']['name'],
                'learning_rate': self.config['training']['optimizer']['lr']
            })

            # Train
            print("Starting training...")
            results = model.train(**train_params)

            # Get best model path
            best_model_path = Path(train_params['project']) / self.experiment_name / 'weights' / 'best.pt'

            # Evaluate on validation set
            print("\nEvaluating on validation set...")
            metrics = model.val()

            # Extract metrics
            results_dict = {
                'fold': self.fold_idx,
                'mAP50': float(metrics.box.map50) if hasattr(metrics, 'box') else 0.0,
                'mAP50-95': float(metrics.box.map) if hasattr(metrics, 'box') else 0.0,
                'precision': float(metrics.box.mp) if hasattr(metrics, 'box') else 0.0,
                'recall': float(metrics.box.mr) if hasattr(metrics, 'box') else 0.0,
                'f1': 0.0  # Will calculate
            }

            # Calculate F1
            if results_dict['precision'] + results_dict['recall'] > 0:
                results_dict['f1'] = 2 * (results_dict['precision'] * results_dict['recall']) / \
                                     (results_dict['precision'] + results_dict['recall'])

            # Log metrics
            mlflow.log_metrics(results_dict)

            # Save metrics
            metrics_path = self.output_dir / 'results' / f'baseline_fold_{self.fold_idx}_metrics.json'
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(results_dict, f, indent=2)

            # Log model
            mlflow.pytorch.log_model(model, "model")

            print(f"\n{'=' * 60}")
            print("Training Complete - Results:")
            print(f"{'=' * 60}")
            for key, value in results_dict.items():
                if key != 'fold':
                    print(f"{key:15s}: {value:.4f}")
            print(f"{'=' * 60}\n")

            # Return results
            return results_dict, best_model_path

    def visualize_results(self, model_path: Path):
        """
        Create visualizations of model predictions

        Args:
            model_path: Path to trained model
        """
        print("\nCreating visualizations...")

        # Load model
        model = YOLO(str(model_path))

        # Get validation images
        with open(self.dataset_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)

        val_images_dir = Path(dataset_config['path']) / dataset_config['val']
        val_images = sorted(list(val_images_dir.glob('*.tif')) + list(val_images_dir.glob('*.tiff')))[:16]

        # Run predictions
        images = []
        ground_truths = []
        predictions = []

        for img_path in val_images:
            # Load image
            import cv2
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

            # Load ground truth
            label_path = img_path.parent.parent.parent / 'labels' / 'val' / (img_path.stem + '.txt')
            gt_boxes = []
            gt_labels = []

            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])

                            # Convert YOLO format to pixel coordinates
                            h, w = img.shape[:2]
                            x1 = int((x_center - width / 2) * w)
                            y1 = int((y_center - height / 2) * h)
                            x2 = int((x_center + width / 2) * w)
                            y2 = int((y_center + height / 2) * h)

                            gt_boxes.append([x1, y1, x2, y2])
                            gt_labels.append(class_id)

            ground_truths.append({'boxes': gt_boxes, 'labels': gt_labels})

            # Run prediction
            results = model(img_path, verbose=False)

            pred_boxes = []
            pred_labels = []
            pred_scores = []

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                scores = results[0].boxes.conf.cpu().numpy()

                pred_boxes = boxes.tolist()
                pred_labels = classes.tolist()
                pred_scores = scores.tolist()

            predictions.append({
                'boxes': pred_boxes,
                'labels': pred_labels,
                'scores': pred_scores
            })

        # Create visualization
        self.visualizer.visualize_predictions(
            images, ground_truths, predictions,
            save_name=f'baseline_fold_{self.fold_idx}_predictions.png',
            n_samples=16
        )

        print("Visualizations created successfully!")


def main():
    """
    Main entry point
    """
    import argparse

    parser = argparse.ArgumentParser(description='Train baseline YOLO model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--fold', type=int, default=0,
                       help='Fold index (0-4)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations after training')

    args = parser.parse_args()

    # Initialize trainer
    trainer = BaselineTrainer(args.config, args.fold)

    # Print dataset summary
    trainer.data_loader.print_dataset_summary()

    # Prepare data
    split = trainer.prepare_data()

    # Train
    results, model_path = trainer.train()

    # Visualize
    if args.visualize:
        trainer.visualize_results(model_path)

    print(f"\nBaseline training complete!")
    print(f"Model saved to: {model_path}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"mAP50: {results['mAP50']:.4f}")


if __name__ == "__main__":
    main()
