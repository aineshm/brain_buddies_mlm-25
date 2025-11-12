"""
Preprocessing Module for Alt ML Pipeline 1
Shows what images look like AFTER preprocessing (like what YOLO sees)
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ImagePreprocessor:
    """
    Preprocess microscopy images for visualization and training
    """

    def __init__(self, target_size: int = 640):
        """
        Initialize preprocessor

        Args:
            target_size: Target image size (YOLO default is 640)
        """
        self.target_size = target_size

    def load_image(self, img_path: Path) -> Optional[np.ndarray]:
        """
        Load image with proper handling for microscopy formats

        Args:
            img_path: Path to image file

        Returns:
            Image as numpy array (RGB, 8-bit) or None if loading fails
        """
        try:
            # Load image (handles 16-bit TIFF)
            if str(img_path).lower().endswith(('.tif', '.tiff')):
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    return None

                # Convert 16-bit to 8-bit
                if img.dtype == np.uint16:
                    # Normalize to 8-bit with proper scaling
                    img = (img / 256).astype(np.uint8)
            else:
                img = cv2.imread(str(img_path))

            if img is None:
                return None

            # Convert grayscale to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:  # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None

    def enhance_contrast(self, img: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """
        Enhance contrast for better visibility

        Args:
            img: Input image (RGB)
            method: Enhancement method ('clahe', 'hist_eq', 'normalize')

        Returns:
            Contrast-enhanced image
        """
        if method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Works well for microscopy images
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return enhanced

        elif method == 'hist_eq':
            # Simple histogram equalization
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            return enhanced

        elif method == 'normalize':
            # Simple normalization (stretch to full range)
            normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            return normalized.astype(np.uint8)

        else:
            return img

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640),
                  color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
        """
        Resize image with aspect ratio preserved (letterbox padding)
        This matches what YOLO does during training

        Args:
            img: Input image
            new_shape: Target shape (height, width)
            color: Padding color

        Returns:
            Letterboxed image
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        # Divide padding into 2 sides
        dw /= 2
        dh /= 2

        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Add border
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img

    def preprocess_for_training(self, img: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Full preprocessing pipeline (what images look like during training)

        Args:
            img: Input image
            enhance: Whether to apply contrast enhancement

        Returns:
            Preprocessed image
        """
        # Enhance contrast (makes dark images visible)
        if enhance:
            img = self.enhance_contrast(img, method='clahe')

        # Letterbox resize (YOLO preprocessing)
        img = self.letterbox(img, (self.target_size, self.target_size))

        return img

    def visualize_preprocessing_comparison(self, img_path: Path, label_path: Optional[Path] = None,
                                          output_path: Optional[Path] = None, class_names: list = None):
        """
        Create side-by-side visualization of raw vs preprocessed image

        Args:
            img_path: Path to image
            label_path: Optional path to YOLO label file
            output_path: Where to save visualization
            class_names: List of class names for labels
        """
        # Load raw image
        raw_img = self.load_image(img_path)
        if raw_img is None:
            print(f"Failed to load {img_path}")
            return

        # Preprocess image
        processed_img = self.preprocess_for_training(raw_img.copy(), enhance=True)

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot raw image
        axes[0].imshow(raw_img)
        axes[0].set_title('Raw Image (Dark & Low Contrast)', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Plot preprocessed image
        axes[1].imshow(processed_img)
        axes[1].set_title('After Preprocessing (CLAHE + Letterbox)', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Add labels if provided
        if label_path is not None and label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # Draw on processed image (scaled)
            h, w = processed_img.shape[:2]
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)

                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            linewidth=2, edgecolor='lime', facecolor='none')
                    axes[1].add_patch(rect)

                    # Add label
                    if class_names and class_id < len(class_names):
                        label_text = class_names[class_id]
                        axes[1].text(x1, y1 - 5, label_text, color='lime',
                                    fontsize=10, fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        # Add metadata
        fig.text(0.5, 0.02, f'Image: {img_path.name} | Original size: {raw_img.shape[:2]} | Target size: {self.target_size}×{self.target_size}',
                ha='center', fontsize=10, style='italic')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {output_path}")

        plt.close()


def visualize_preprocessing_examples(data_dir: Path, output_dir: Path, n_samples: int = 10,
                                     class_names: list = None):
    """
    Create preprocessing comparison for multiple images

    Args:
        data_dir: Directory containing images and labels
        output_dir: Where to save visualizations
        n_samples: Number of samples to visualize
        class_names: List of class names
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = ImagePreprocessor(target_size=640)

    # Find images
    images_dir = data_dir / 'images' / 'train'
    labels_dir = data_dir / 'labels' / 'train'

    image_files = sorted(list(images_dir.glob('*.tif')) + list(images_dir.glob('*.tiff')))
    image_files = image_files[:n_samples]

    print(f"\n{'='*80}")
    print(f"PREPROCESSING VISUALIZATION")
    print(f"{'='*80}\n")
    print(f"Found {len(image_files)} images")
    print(f"Output directory: {output_dir}\n")

    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing {img_path.name}...")

        # Find corresponding label
        label_path = labels_dir / (img_path.stem + '.txt')

        # Create output path
        output_path = output_dir / f'preprocessing_comparison_{idx:02d}_{img_path.stem}.png'

        # Visualize
        preprocessor.visualize_preprocessing_comparison(
            img_path, label_path, output_path, class_names
        )

    print(f"\n✓ Created {len(image_files)} preprocessing visualizations")
    print(f"  View them in: {output_dir}\n")


if __name__ == "__main__":
    import sys
    import yaml

    # Load config
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Paths
    project_root = Path(config['project']['root_dir'])
    yolo_dataset = project_root / config['data']['existing_yolo_dataset']
    output_dir = Path.home() / 'mlm_outputs' / 'alt_pipeline_1' / 'visualizations' / 'preprocessing'

    # Create visualizations
    visualize_preprocessing_examples(
        data_dir=yolo_dataset,
        output_dir=output_dir,
        n_samples=10,
        class_names=config['data']['classes']
    )
