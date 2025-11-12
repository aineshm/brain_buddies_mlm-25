"""
Data Loader for Alt ML Pipeline 1
Loads existing YOLO dataset and prepares for training
"""

import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import KFold
from collections import Counter
import json


class YOLODataLoader:
    """
    Load and prepare YOLO format dataset for training
    """

    def __init__(self, config_path: str):
        """
        Initialize data loader

        Args:
            config_path: Path to config.yaml
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.project_root = Path(self.config['project']['root_dir'])
        self.output_dir = Path(os.path.expandvars(self.config['project']['output_dir']))
        self.yolo_dataset_path = self.project_root / self.config['data']['existing_yolo_dataset']

        self.classes = self.config['data']['classes']
        self.num_classes = self.config['data']['num_classes']

        # Paths
        self.train_images = self.yolo_dataset_path / 'images' / 'train'
        self.train_labels = self.yolo_dataset_path / 'labels' / 'train'
        self.val_images = self.yolo_dataset_path / 'images' / 'val'
        self.val_labels = self.yolo_dataset_path / 'labels' / 'val'

        self.processed_dir = self.output_dir / 'data' / 'processed'
        self.splits_dir = self.output_dir / 'data' / 'splits'

    def load_dataset_info(self) -> Dict:
        """
        Load and analyze the existing YOLO dataset

        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'train_images': [],
            'train_labels': [],
            'val_images': [],
            'val_labels': [],
            'class_distribution': Counter(),
            'total_annotations': 0,
            'images_per_sequence': {}
        }

        # Load training data
        if self.train_images.exists():
            info['train_images'] = sorted(list(self.train_images.glob('*.tif')) + list(self.train_images.glob('*.tiff')))
            info['train_labels'] = sorted(list(self.train_labels.glob('*.txt')))

        # Load validation data
        if self.val_images.exists():
            info['val_images'] = sorted(list(self.val_images.glob('*.tif')) + list(self.val_images.glob('*.tiff')))
            info['val_labels'] = sorted(list(self.val_labels.glob('*.txt')))

        # Analyze labels for class distribution
        all_labels = info['train_labels'] + info['val_labels']
        for label_file in all_labels:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        info['class_distribution'][class_id] += 1
                        info['total_annotations'] += 1

        # Group images by sequence
        all_images = info['train_images'] + info['val_images']
        for img_path in all_images:
            # Extract sequence name (assumes format: SequenceName_frame_XXXX.tif)
            filename = img_path.stem
            parts = filename.split('_frame_')
            if len(parts) == 2:
                sequence_name = parts[0]
                if sequence_name not in info['images_per_sequence']:
                    info['images_per_sequence'][sequence_name] = []
                info['images_per_sequence'][sequence_name].append(img_path)

        return info

    def create_leave_one_sequence_out_splits(self) -> List[Dict]:
        """
        Create 5-fold cross-validation splits by leaving one sequence out

        Returns:
            List of split dictionaries with train/val image paths
        """
        info = self.load_dataset_info()
        sequences = list(info['images_per_sequence'].keys())

        splits = []
        for i, val_sequence in enumerate(sequences):
            split = {
                'fold': i,
                'val_sequence': val_sequence,
                'train_images': [],
                'val_images': info['images_per_sequence'][val_sequence]
            }

            # Add all other sequences to training
            for seq in sequences:
                if seq != val_sequence:
                    split['train_images'].extend(info['images_per_sequence'][seq])

            splits.append(split)

        return splits

    def prepare_fold(self, fold_idx: int, splits: List[Dict]) -> str:
        """
        Prepare a specific fold by copying images/labels to output directory

        Args:
            fold_idx: Index of the fold to prepare
            splits: List of split dictionaries

        Returns:
            Path to the prepared dataset YAML file
        """
        split = splits[fold_idx]
        fold_dir = self.splits_dir / f'fold_{fold_idx}'

        # Create directories
        (fold_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (fold_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (fold_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (fold_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

        # Copy training images and labels
        for img_path in split['train_images']:
            # Copy image
            dest_img = fold_dir / 'images' / 'train' / img_path.name
            if not dest_img.exists():
                shutil.copy(img_path, dest_img)

            # Copy corresponding label
            label_name = img_path.stem + '.txt'
            src_label = img_path.parent.parent.parent / 'labels' / img_path.parent.name / label_name
            dest_label = fold_dir / 'labels' / 'train' / label_name
            if src_label.exists() and not dest_label.exists():
                shutil.copy(src_label, dest_label)

        # Copy validation images and labels
        for img_path in split['val_images']:
            # Copy image
            dest_img = fold_dir / 'images' / 'val' / img_path.name
            if not dest_img.exists():
                shutil.copy(img_path, dest_img)

            # Copy corresponding label
            label_name = img_path.stem + '.txt'
            src_label = img_path.parent.parent.parent / 'labels' / img_path.parent.name / label_name
            dest_label = fold_dir / 'labels' / 'val' / label_name
            if src_label.exists() and not dest_label.exists():
                shutil.copy(src_label, dest_label)

        # Create dataset YAML for YOLO
        dataset_yaml = {
            'path': str(fold_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': self.num_classes,
            'names': self.classes
        }

        yaml_path = fold_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f)

        return str(yaml_path)

    def visualize_sample_images(self, n_samples: int = 5, output_dir: Optional[Path] = None) -> None:
        """
        Visualize sample images from each class for sanity check

        Args:
            n_samples: Number of samples per class to visualize
            output_dir: Directory to save visualizations (defaults to output_dir/visualizations/data_quality)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if output_dir is None:
            output_dir = self.output_dir / 'visualizations' / 'data_quality'
        output_dir.mkdir(parents=True, exist_ok=True)

        info = self.load_dataset_info()

        # Find samples for each class
        class_samples = {i: [] for i in range(self.num_classes)}

        # Scan ALL training labels (not just first 50) to ensure we find samples
        for label_file in info['train_labels']:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    # Get classes present in this image
                    classes_present = set([int(line.split()[0]) for line in lines if line.strip()])

                    # Add to samples if we need more of this class
                    for class_id in classes_present:
                        if len(class_samples[class_id]) < n_samples:
                            # Find corresponding image
                            img_name = label_file.stem + '.tif'
                            img_path = self.train_images / img_name
                            if not img_path.exists():
                                img_path = self.train_images / (label_file.stem + '.tiff')

                            if img_path.exists():
                                class_samples[class_id].append((img_path, label_file))

            # Early exit if we have enough samples for all classes
            if all(len(samples) >= n_samples for samples in class_samples.values()):
                break

        # Check if we have any samples at all
        max_samples_found = max([len(samples) for samples in class_samples.values()])
        if max_samples_found == 0:
            print("⚠ Warning: No samples found in the first 50 training images")
            print("  Try increasing the search range or check your data")
            return

        # Determine number of columns (at least 1, at most n_samples)
        n_cols = min(n_samples, max_samples_found)
        if n_cols == 0:
            n_cols = 1

        # Create visualization
        fig, axes = plt.subplots(self.num_classes, n_cols,
                                  figsize=(4 * n_cols, 4 * self.num_classes))

        # Ensure axes is 2D array
        if self.num_classes == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for class_id in range(self.num_classes):
            samples = class_samples[class_id][:n_samples]

            for col_idx in range(n_cols):
                ax = axes[class_id, col_idx]

                if col_idx < len(samples):
                    img_path, label_file = samples[col_idx]

                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Load labels
                        with open(label_file, 'r') as f:
                            lines = f.readlines()

                        # Plot
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(f'{self.classes[class_id]}', fontsize=10)

                        # Draw bounding boxes for this class
                        h, w = img.shape[:2]
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) > 4:
                                cls = int(parts[0])
                                if cls == class_id:
                                    # Convert YOLO format to pixel coordinates
                                    x_center, y_center, width, height = map(float, parts[1:5])
                                    x1 = int((x_center - width / 2) * w)
                                    y1 = int((y_center - height / 2) * h)
                                    x2 = int((x_center + width / 2) * w)
                                    y2 = int((y_center + height / 2) * h)

                                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                              linewidth=2, edgecolor='red', facecolor='none')
                                    ax.add_patch(rect)
                else:
                    # No sample for this class in this column - hide axis
                    ax.axis('off')
                    if col_idx == 0:
                        ax.text(0.5, 0.5, f'{self.classes[class_id]}\n(no samples)',
                               ha='center', va='center', fontsize=10, color='gray')

        plt.tight_layout()
        output_path = output_dir / 'sample_images_by_class.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Sample images visualization saved to: {output_path}")

    def print_dataset_summary(self) -> None:
        """
        Print comprehensive dataset summary
        """
        info = self.load_dataset_info()

        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"\nTotal Images: {len(info['train_images']) + len(info['val_images'])}")
        print(f"  Training: {len(info['train_images'])}")
        print(f"  Validation: {len(info['val_images'])}")
        print(f"\nTotal Annotations: {info['total_annotations']}")

        print(f"\nSequences: {len(info['images_per_sequence'])}")
        for seq_name, images in info['images_per_sequence'].items():
            print(f"  {seq_name}: {len(images)} frames")

        print("\nClass Distribution:")
        total = sum(info['class_distribution'].values())
        for class_id in range(self.num_classes):
            count = info['class_distribution'].get(class_id, 0)
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {class_id}. {self.classes[class_id]:20s}: {count:5d} ({percentage:5.1f}%)")

        print("\n" + "=" * 60 + "\n")

        # Save to JSON
        summary_path = self.output_dir / 'data' / 'dataset_summary.json'
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            'total_images': len(info['train_images']) + len(info['val_images']),
            'train_images': len(info['train_images']),
            'val_images': len(info['val_images']),
            'total_annotations': info['total_annotations'],
            'sequences': {name: len(imgs) for name, imgs in info['images_per_sequence'].items()},
            'class_distribution': {self.classes[i]: info['class_distribution'].get(i, 0) for i in range(self.num_classes)}
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Dataset summary saved to: {summary_path}")


if __name__ == "__main__":
    # Test data loader
    import sys

    config_path = "configs/config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    print(f"Loading data with config: {config_path}")
    loader = YOLODataLoader(config_path)

    # Print summary
    loader.print_dataset_summary()

    # Create splits
    print("\nCreating leave-one-sequence-out splits...")
    splits = loader.create_leave_one_sequence_out_splits()
    print(f"Created {len(splits)} folds")

    for split in splits:
        print(f"\nFold {split['fold']}:")
        print(f"  Validation sequence: {split['val_sequence']}")
        print(f"  Training images: {len(split['train_images'])}")
        print(f"  Validation images: {len(split['val_images'])}")

    # Visualize samples (raw)
    print("\nVisualizing sample images (raw data)...")
    loader.visualize_sample_images(n_samples=3)

    # Also show preprocessing comparison
    print("\nGenerating preprocessing visualizations (raw vs enhanced)...")
    from preprocessor import visualize_preprocessing_examples

    output_dir = loader.output_dir / 'visualizations' / 'preprocessing'
    visualize_preprocessing_examples(
        data_dir=loader.yolo_dataset_path,
        output_dir=output_dir,
        n_samples=5,
        class_names=loader.classes
    )

    print(f"\n{'='*80}")
    print("✓ Data exploration complete!")
    print(f"{'='*80}")
    print("\nGenerated visualizations:")
    print(f"  1. Raw samples by class: {loader.output_dir / 'visualizations' / 'data_quality' / 'sample_images_by_class.png'}")
    print(f"  2. Preprocessing comparison: {output_dir}/")
    print("\nThe preprocessing visualizations show:")
    print("  - Left: Raw image (dark & low contrast)")
    print("  - Right: After CLAHE enhancement + letterbox resize")
    print("  - This is what YOLO sees during training!")
    print()
