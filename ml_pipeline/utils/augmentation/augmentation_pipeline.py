"""
Data augmentation pipeline for microscopy images.

Critical for training with small dataset (205 frames).
Uses Albumentations for advanced augmentation with segmentation mask support.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from typing import Dict, Any, Optional


class MicroscopyAugmentation:
    """Augmentation pipeline for microscopy cell images."""

    def __init__(
        self,
        image_size: int = 640,
        mode: str = 'train',
        normalize: bool = True
    ):
        """
        Initialize augmentation pipeline.

        Args:
            image_size: Target image size for model input
            mode: 'train', 'val', or 'test'
            normalize: Whether to normalize images
        """
        self.image_size = image_size
        self.mode = mode
        self.normalize = normalize

        self.transform = self._build_pipeline()

    def _build_pipeline(self) -> A.Compose:
        """Build augmentation pipeline based on mode."""

        if self.mode == 'train':
            return self._train_pipeline()
        elif self.mode == 'val':
            return self._val_pipeline()
        else:  # test
            return self._test_pipeline()

    def _train_pipeline(self) -> A.Compose:
        """
        Training augmentation pipeline.

        Aggressive augmentation to combat small dataset size.
        All transforms preserve segmentation masks.
        """
        transforms = [
            # Resize (may need to handle different input resolutions)
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),

            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                shear=(-10, 10),
                p=0.3
            ),

            # Elastic transform (simulates biological variation)
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                alpha_affine=5,
                p=0.2
            ),

            # Intensity transforms (preserve relative intensities for microscopy)
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),

            # Blur (simulates focus variations)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.3),

            # Noise (simulates imaging noise)
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.2),

            # CLAHE (enhances local contrast, common in microscopy)
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

            # Grid distortion (simulates optical aberrations)
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.2
            ),

            # Coarse dropout (simulates occlusions/imaging artifacts)
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.2
            ),
        ]

        # Add normalization if requested
        if self.normalize:
            transforms.append(
                A.Normalize(
                    mean=[0.485],  # Single channel (grayscale)
                    std=[0.229],
                    max_pixel_value=65535.0  # 16-bit images
                )
            )

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3  # Keep boxes with >30% visibility
            )
        )

    def _val_pipeline(self) -> A.Compose:
        """
        Validation pipeline - minimal transforms.

        Only resize and normalize, no augmentation.
        """
        transforms = [
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0
            ),
        ]

        if self.normalize:
            transforms.append(
                A.Normalize(
                    mean=[0.485],
                    std=[0.229],
                    max_pixel_value=65535.0
                )
            )

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            )
        )

    def _test_pipeline(self) -> A.Compose:
        """Test pipeline - identical to validation."""
        return self._val_pipeline()

    def __call__(
        self,
        image: np.ndarray,
        masks: Optional[np.ndarray] = None,
        bboxes: Optional[np.ndarray] = None,
        class_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Apply augmentation pipeline.

        Args:
            image: Input image (H, W) or (H, W, C)
            masks: Segmentation masks (H, W, N) where N is number of objects
            bboxes: Bounding boxes in YOLO format (N, 4)
            class_labels: Class labels (N,)

        Returns:
            Dictionary with augmented image and annotations
        """
        # Ensure image is grayscale
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # Add channel dimension

        # Prepare data
        data = {'image': image}

        if masks is not None:
            data['masks'] = masks

        if bboxes is not None and class_labels is not None:
            data['bboxes'] = bboxes
            data['class_labels'] = class_labels

        # Apply transforms
        augmented = self.transform(**data)

        return augmented


class YOLOAugmentation:
    """
    Simplified augmentation for YOLO training.

    YOLO has built-in augmentation, but we can add microscopy-specific transforms.
    """

    @staticmethod
    def get_yolo_augmentation_config() -> Dict[str, Any]:
        """
        Get augmentation configuration for YOLO training.

        Returns YOLO hyperparameters for augmentation.
        """
        return {
            # Mosaic augmentation (combine 4 images)
            'mosaic': 1.0,

            # MixUp augmentation (blend 2 images)
            'mixup': 0.1,

            # Augmentation hyperparameters
            'degrees': 15.0,        # Rotation degrees
            'translate': 0.1,       # Translation fraction
            'scale': 0.5,           # Scale gain
            'shear': 5.0,           # Shear degrees
            'perspective': 0.0,     # Perspective gain (0 = disabled for microscopy)
            'flipud': 0.5,          # Vertical flip probability
            'fliplr': 0.5,          # Horizontal flip probability

            # HSV augmentation (adapted for grayscale)
            'hsv_h': 0.0,           # Hue (disabled for grayscale)
            'hsv_s': 0.0,           # Saturation (disabled for grayscale)
            'hsv_v': 0.4,           # Value (brightness)

            # Additional augmentation
            'blur': 0.01,           # Blur probability
            'erasing': 0.2,         # Random erasing probability
            'crop_fraction': 1.0,   # Crop fraction (1.0 = no crop)
        }


def visualize_augmentation(
    image: np.ndarray,
    masks: Optional[np.ndarray] = None,
    num_samples: int = 4
):
    """
    Visualize augmentation effects.

    Args:
        image: Input image
        masks: Optional segmentation masks
        num_samples: Number of augmented samples to generate
    """
    import matplotlib.pyplot as plt

    aug_train = MicroscopyAugmentation(mode='train', normalize=False)

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 4, 8))

    # Original
    axes[0, 0].imshow(image.squeeze(), cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    if masks is not None:
        axes[1, 0].imshow(masks.max(axis=2), cmap='nipy_spectral')
        axes[1, 0].set_title('Original Masks')
        axes[1, 0].axis('off')

    # Augmented samples
    for i in range(1, num_samples):
        data = {'image': image}
        if masks is not None:
            data['masks'] = masks

        augmented = aug_train(**data)

        axes[0, i].imshow(augmented['image'].squeeze(), cmap='gray')
        axes[0, i].set_title(f'Augmented {i}')
        axes[0, i].axis('off')

        if masks is not None and 'masks' in augmented:
            axes[1, i].imshow(augmented['masks'].max(axis=2), cmap='nipy_spectral')
            axes[1, i].set_title(f'Augmented Masks {i}')
            axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_samples.png', dpi=150, bbox_inches='tight')
    print("Augmentation samples saved to augmentation_samples.png")


if __name__ == '__main__':
    # Example usage
    print("Microscopy Augmentation Pipeline")
    print("=" * 60)

    # Create pipelines
    train_aug = MicroscopyAugmentation(mode='train', image_size=640)
    val_aug = MicroscopyAugmentation(mode='val', image_size=640)

    print("\nTraining augmentations:")
    print(train_aug.transform)

    print("\nValidation augmentations:")
    print(val_aug.transform)

    print("\nYOLO augmentation config:")
    yolo_config = YOLOAugmentation.get_yolo_augmentation_config()
    for key, value in yolo_config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
