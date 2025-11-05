"""
SageMaker-compatible training script.
Handles SageMaker-specific paths and environment.
"""

import os
import sys
import argparse
from pathlib import Path

# SageMaker environment variables
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
SM_OUTPUT_DATA_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
SM_NUM_GPUS = int(os.environ.get('SM_NUM_GPUS', 1))


def train():
    """Run training on SageMaker."""
    parser = argparse.ArgumentParser()

    # SageMaker parameters
    parser.add_argument('--model-dir', type=str, default=SM_MODEL_DIR)
    parser.add_argument('--data-dir', type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument('--output-dir', type=str, default=SM_OUTPUT_DATA_DIR)

    # Training parameters (passed as hyperparameters)
    parser.add_argument('--model', type=str, default='s')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--patience', type=int, default=50)

    args = parser.parse_args()

    print("="*80)
    print("TRAINING ON AWS SAGEMAKER")
    print("="*80)
    print(f"Model directory: {args.model_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of GPUs: {SM_NUM_GPUS}")
    print(f"Model size: YOLOv8{args.model}-seg")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print("="*80 + "\n")

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO

    # Find dataset YAML
    dataset_yaml = None
    possible_paths = [
        Path(args.data_dir) / 'yolo_dataset' / 'dataset.yaml',
        Path(args.data_dir) / 'dataset.yaml',
        Path(args.data_dir) / 'processed' / 'yolo_dataset' / 'dataset.yaml',
    ]

    for path in possible_paths:
        if path.exists():
            dataset_yaml = path
            break

    if dataset_yaml is None:
        print("ERROR: Could not find dataset.yaml")
        print("Searched in:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nDirectory contents:")
        for root, dirs, files in os.walk(args.data_dir):
            print(f"\n{root}:")
            for file in files:
                print(f"  {file}")
        sys.exit(1)

    print(f"Found dataset YAML: {dataset_yaml}\n")

    # Update dataset YAML paths to be absolute for SageMaker
    import yaml
    with open(dataset_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    # Update paths to be relative to data_dir
    data_config['path'] = str(Path(args.data_dir) / 'yolo_dataset')
    data_config['train'] = 'images/train'
    data_config['val'] = 'images/val'

    # Save updated config
    updated_yaml = Path(args.data_dir) / 'dataset_sagemaker.yaml'
    with open(updated_yaml, 'w') as f:
        yaml.dump(data_config, f)

    print(f"Updated dataset config saved to: {updated_yaml}\n")

    # Initialize model
    model_name = f'yolov8{args.model}-seg.pt'
    print(f"Loading pretrained model: {model_name}")
    model = YOLO(model_name)

    # Training configuration
    print("\nStarting training...\n")
    results = model.train(
        data=str(updated_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.model_dir,
        name='training',
        exist_ok=True,
        verbose=True,
        patience=args.patience,
        save=True,
        plots=True,
        # Optimizer
        optimizer='AdamW',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Augmentation
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.4,
        degrees=180.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Validation
        val=True,
        save_period=-1
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    # Copy best weights to model directory root for easy access
    import shutil
    best_weights = Path(args.model_dir) / 'training' / 'weights' / 'best.pt'
    last_weights = Path(args.model_dir) / 'training' / 'weights' / 'last.pt'

    if best_weights.exists():
        shutil.copy(best_weights, Path(args.model_dir) / 'best.pt')
        print(f"✓ Best model saved to: {Path(args.model_dir) / 'best.pt'}")

    if last_weights.exists():
        shutil.copy(last_weights, Path(args.model_dir) / 'last.pt')
        print(f"✓ Last model saved to: {Path(args.model_dir) / 'last.pt'}")

    # Copy results and plots
    results_csv = Path(args.model_dir) / 'training' / 'results.csv'
    if results_csv.exists():
        shutil.copy(results_csv, Path(args.model_dir) / 'results.csv')
        print(f"✓ Results saved to: {Path(args.model_dir) / 'results.csv'}")

    # Copy validation predictions
    val_plots = list((Path(args.model_dir) / 'training').glob('*.png'))
    if val_plots:
        plots_dir = Path(args.model_dir) / 'plots'
        plots_dir.mkdir(exist_ok=True)
        for plot in val_plots:
            shutil.copy(plot, plots_dir / plot.name)
        print(f"✓ Plots saved to: {plots_dir}/")

    print("="*80)

    # Print final metrics
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        if len(df) > 0:
            final_metrics = df.iloc[-1]
            print("\nFinal Metrics:")
            print("-" * 60)
            for col in df.columns:
                if col.strip() and not pd.isna(final_metrics[col]):
                    print(f"  {col}: {final_metrics[col]:.4f}")
            print("-" * 60)

    return results


if __name__ == '__main__':
    train()
