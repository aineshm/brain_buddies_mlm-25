"""
Visualize and save preprocessing steps for YOLO training.

This script shows what transformations are applied to images during training
and optionally saves examples to S3 for inspection.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import cv2
from PIL import Image
import yaml
from typing import List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from ultralytics import YOLO
    from ultralytics.data import YOLODataset
    from ultralytics.data.augment import Compose, LetterBox, RandomHSV, RandomFlip, Mosaic, MixUp
except ImportError:
    print("ERROR: ultralytics not installed")
    print("Run: pip install ultralytics")
    sys.exit(1)


class PreprocessingVisualizer:
    """Visualize YOLO preprocessing pipeline."""

    def __init__(
        self,
        dataset_yaml: str,
        output_dir: str,
        num_samples: int = 10,
        img_size: int = 640
    ):
        """
        Initialize visualizer.

        Args:
            dataset_yaml: Path to dataset.yaml
            output_dir: Where to save visualization images
            num_samples: Number of samples to visualize
            img_size: Image size for preprocessing
        """
        self.dataset_yaml = dataset_yaml
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.img_size = img_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset_config(self) -> dict:
        """Load dataset configuration."""
        with open(self.dataset_yaml, 'r') as f:
            return yaml.safe_load(f)

    def create_preprocessing_examples(self):
        """
        Create side-by-side comparisons of original vs preprocessed images.
        """
        print("="*80)
        print("PREPROCESSING VISUALIZATION")
        print("="*80)
        print(f"Dataset: {self.dataset_yaml}")
        print(f"Output: {self.output_dir}")
        print(f"Samples: {self.num_samples}")
        print()

        # Load dataset config
        config = self.load_dataset_config()
        data_path = Path(config['path'])
        train_path = data_path / config['train']

        # Get list of training images
        image_files = list(train_path.glob('*.tif')) + list(train_path.glob('*.tiff'))
        if not image_files:
            image_files = list(train_path.glob('*.png')) + list(train_path.glob('*.jpg'))

        if not image_files:
            print(f"ERROR: No images found in {train_path}")
            return

        print(f"Found {len(image_files)} training images")
        print()

        # Sample random images
        np.random.seed(42)
        sampled_images = np.random.choice(image_files, min(self.num_samples, len(image_files)), replace=False)

        # Document preprocessing steps
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write("# Preprocessing Steps Applied During Training\n\n")
            f.write("## Overview\n\n")
            f.write("YOLO applies the following preprocessing and augmentation steps:\n\n")
            f.write("### 1. Resize & Padding\n")
            f.write(f"- Target size: {self.img_size}×{self.img_size}\n")
            f.write("- LetterBox: Maintains aspect ratio with padding\n")
            f.write("- Padding color: Black (0)\n\n")
            f.write("### 2. HSV Augmentation (Training only)\n")
            f.write("- Hue shift: 0.0 (disabled for grayscale)\n")
            f.write("- Saturation shift: 0.0 (disabled)\n")
            f.write("- Value (brightness): ±40%\n\n")
            f.write("### 3. Geometric Transforms (Training only)\n")
            f.write("- Rotation: ±180° (full rotation for microscopy)\n")
            f.write("- Translation: ±10%\n")
            f.write("- Scale: 0.5×-1.5×\n")
            f.write("- Shear: ±5°\n")
            f.write("- Horizontal flip: 50%\n")
            f.write("- Vertical flip: 50%\n\n")
            f.write("### 4. Advanced Augmentation (Training only)\n")
            f.write("- **Mosaic** (100%): Combines 4 images into one\n")
            f.write("- **MixUp** (10%): Blends 2 images together\n")
            f.write("- **Copy-Paste** (10%): Pastes objects from other images\n\n")
            f.write("### 5. Normalization\n")
            f.write("- Pixel values: [0, 255] for uint8 or [0, 65535] for uint16\n")
            f.write("- Converted to float32 and divided by 255\n")
            f.write("- Result: [0.0, 1.0] range\n\n")
            f.write("## Example Images\n\n")
            f.write("The images below show:\n")
            f.write("- **Left**: Original image\n")
            f.write("- **Right**: After resize + letterbox (basic preprocessing)\n\n")
            f.write("Note: Augmentations (rotation, flip, mosaic, etc.) are applied randomly during training,\n")
            f.write("so each epoch sees different variations of the images.\n\n")

        # Process each sampled image
        for idx, img_path in enumerate(sampled_images, 1):
            print(f"[{idx}/{len(sampled_images)}] Processing {img_path.name}...")

            # Load original image
            if str(img_path).lower().endswith(('.tif', '.tiff')):
                # Handle TIFF files (may be 16-bit)
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img.dtype == np.uint16:
                    # Convert 16-bit to 8-bit for visualization
                    img = (img / 256).astype(np.uint8)
            else:
                img = cv2.imread(str(img_path))

            if img is None:
                print(f"  WARNING: Could not load {img_path}")
                continue

            # Convert to RGB if grayscale
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            original_shape = img.shape[:2]

            # Apply LetterBox (YOLO's resize with padding)
            letterbox = LetterBox(new_shape=(self.img_size, self.img_size), auto=False, scaleFill=False)
            img_processed = letterbox(image=img)

            # Create side-by-side comparison
            # Resize original to match height for visualization
            h_orig, w_orig = img.shape[:2]
            h_proc, w_proc = img_processed.shape[:2]

            # Scale original to same height as processed
            scale = h_proc / h_orig
            new_w = int(w_orig * scale)
            img_orig_resized = cv2.resize(img, (new_w, h_proc))

            # Create side-by-side
            side_by_side = np.hstack([img_orig_resized, img_processed])

            # Add labels
            label_height = 30
            label_img = np.ones((label_height, side_by_side.shape[1], 3), dtype=np.uint8) * 255
            cv2.putText(label_img, f"Original ({original_shape[1]}x{original_shape[0]})",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(label_img, f"Preprocessed ({self.img_size}x{self.img_size})",
                       (new_w + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Combine labels and images
            final_img = np.vstack([label_img, side_by_side])

            # Save
            output_path = self.output_dir / f"preprocessing_example_{idx:02d}_{img_path.stem}.jpg"
            cv2.imwrite(str(output_path), final_img)
            print(f"  Saved: {output_path.name}")

        print()
        print("="*80)
        print("PREPROCESSING VISUALIZATION COMPLETE")
        print("="*80)
        print(f"\nSaved {len(sampled_images)} examples to: {self.output_dir}")
        print(f"README: {readme_path}")
        print()
        print("To upload to S3:")
        print(f"  aws s3 sync {self.output_dir} s3://YOUR-BUCKET/preprocessing-examples/")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize YOLO preprocessing steps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize local dataset
  python3 visualize_preprocessing.py \\
      --dataset ../../data/processed/yolo_dataset/dataset.yaml \\
      --output ../../results/preprocessing_examples

  # For SageMaker (after training)
  python3 visualize_preprocessing.py \\
      --dataset /opt/ml/input/data/training/yolo_dataset/dataset.yaml \\
      --output /opt/ml/output/preprocessing_examples \\
      --num-samples 20
        """
    )

    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to dataset.yaml'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for visualization images'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of sample images to visualize (default: 10)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Image size for preprocessing (default: 640)'
    )

    args = parser.parse_args()

    visualizer = PreprocessingVisualizer(
        dataset_yaml=args.dataset,
        output_dir=args.output,
        num_samples=args.num_samples,
        img_size=args.img_size
    )

    visualizer.create_preprocessing_examples()


if __name__ == '__main__':
    main()
