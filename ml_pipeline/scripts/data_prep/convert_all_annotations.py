"""
Batch convert all CVAT XML annotations to YOLO format.

This script processes all annotated datasets and organizes them for YOLOv8 training.
"""

import sys
from pathlib import Path
import shutil
import zipfile
import tifffile
from xml_to_yolo import XMLToYOLOConverter

# Dataset configurations
DATASETS = [
    {
        'name': 'training',
        'tif': 'training.tif',
        'zip': 'trainingannotations.zip',
        'xml': 'trainingannotations.xml',
        'width': 1392,
        'height': 1040,
        'split': 'train'
    },
    {
        'name': 'MattLines1',
        'tif': 'MattLines1.tif',
        'zip': 'MattLines1annotations.zip',
        'xml': 'MattLines1annotations.xml',
        'width': 244,
        'height': 242,
        'split': 'train'
    },
    {
        'name': 'MattLines7',
        'tif': 'MattLines7.tif',
        'zip': 'MattLines7annotations.zip',
        'xml': 'MattLines7annotations.xml',
        'width': 244,
        'height': 242,
        'split': 'train'
    },
    {
        'name': 'MattLines27',
        'tif': 'MattLines27.tif',
        'zip': 'MattLines27annotations.zip',
        'xml': 'MattLines27annotations.xml',
        'width': 1392,
        'height': 1040,
        'split': 'val'  # Most comprehensive for validation
    },
    {
        'name': 'LizaMutant38',
        'tif': 'LizaMutant38.tif',
        'zip': 'LizaMutant38annotations.zip',
        'xml': 'LizaMutant38annotations.xml',
        'width': 1392,
        'height': 1040,
        'split': 'train'
    }
]


def unzip_annotations(data_dir: Path, dataset: dict, temp_dir: Path):
    """
    Unzip annotation files to temporary directory.

    Args:
        data_dir: Root data directory
        dataset: Dataset configuration dict
        temp_dir: Temporary directory for extraction

    Returns:
        Path to extracted XML file
    """
    zip_path = data_dir / 'Annotated Data' / dataset['zip']

    if not zip_path.exists():
        # Try root directory
        zip_path = data_dir / dataset['zip']

    if not zip_path.exists():
        print(f"  ⚠️  WARNING: {zip_path} not found, skipping...")
        return None

    # Extract
    extract_dir = temp_dir / dataset['name']
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Find XML file
    xml_files = list(extract_dir.glob('*.xml'))
    if not xml_files:
        print(f"  ⚠️  WARNING: No XML file found in {zip_path.name}")
        return None

    return xml_files[0]


def extract_tiff_frames(tif_path: Path, output_dir: Path, base_name: str):
    """
    Extract individual frames from multi-frame TIFF file.

    Args:
        tif_path: Path to TIFF file
        output_dir: Output directory for frames
        base_name: Base name for frame files

    Returns:
        Number of frames extracted
    """
    if not tif_path.exists():
        print(f"  ⚠️  WARNING: {tif_path} not found")
        return 0

    print(f"  Extracting frames from {tif_path.name}...")

    # Load TIFF
    tif_data = tifffile.imread(tif_path)

    # Handle both single and multi-frame TIFFs
    if tif_data.ndim == 2:
        frames = [tif_data]
    else:
        frames = tif_data

    # Save individual frames
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame_path = output_dir / f'{base_name}_frame_{i:04d}.tif'
        tifffile.imwrite(frame_path, frame)

    print(f"  Extracted {len(frames)} frames")
    return len(frames)


def create_yolo_dataset_yaml(output_dir: Path, train_count: int, val_count: int):
    """
    Create YOLO dataset configuration file.

    Args:
        output_dir: Root output directory
        train_count: Number of training images
        val_count: Number of validation images
    """
    yaml_path = output_dir / 'dataset.yaml'

    yaml_content = f"""# Candida albicans Morphology Dataset
# Generated for YOLOv8 instance segmentation training

# Paths (relative to this file)
path: {output_dir.absolute()}
train: images/train
val: images/val

# Number of classes
nc: 7

# Class names
names:
  0: single dispersed cell
  1: clump dispersed cell
  2: planktonic
  3: yeast form
  4: psuedohyphae
  5: hyphae
  6: biofilm

# Dataset statistics
# Training images: {train_count}
# Validation images: {val_count}
# Total images: {train_count + val_count}

# Cell morphology categories:
# - Dispersed cells: single, clump, planktonic
# - Biofilm-associated: yeast form, pseudohyphae, hyphae
# - Biofilm regions: biofilm polygons
"""

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ Dataset YAML created: {yaml_path}")


def main():
    """Main conversion pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert all CVAT annotations to YOLO format and organize dataset'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/Users/aineshmohan/Documents/mlm',
        help='Root data directory containing Annotated Data folder'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/processed/yolo_dataset',
        help='Output directory for YOLO dataset'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("="*80)
    print("CVAT XML → YOLO FORMAT CONVERTER")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")

    # Create output structure
    temp_dir = output_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count = 0

    # Process each dataset
    for i, dataset in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}] Processing {dataset['name']}...")
        print("-" * 80)

        # 1. Unzip annotations
        xml_path = unzip_annotations(data_dir, dataset, temp_dir)
        if xml_path is None:
            continue

        # 2. Convert XML to YOLO format
        labels_output = temp_dir / dataset['name'] / 'labels'
        converter = XMLToYOLOConverter(
            str(xml_path),
            str(labels_output),
            dataset['width'],
            dataset['height']
        )

        frame_counts = converter.convert_all_frames(dataset['name'])

        # 3. Extract TIFF frames
        tif_path = data_dir / 'Annotated Data' / dataset['tif']
        if not tif_path.exists():
            tif_path = data_dir / dataset['tif']

        frames_output = temp_dir / dataset['name'] / 'frames'
        num_frames = extract_tiff_frames(tif_path, frames_output, dataset['name'])

        # 4. Copy to final structure
        split = dataset['split']
        for frame_num in frame_counts.keys():
            # Copy image
            src_img = frames_output / f"{dataset['name']}_frame_{frame_num:04d}.tif"
            dst_img = output_dir / 'images' / split / f"{dataset['name']}_frame_{frame_num:04d}.tif"
            if src_img.exists():
                shutil.copy(src_img, dst_img)

            # Copy label
            src_label = labels_output / f"{dataset['name']}_frame_{frame_num:04d}.txt"
            dst_label = output_dir / 'labels' / split / f"{dataset['name']}_frame_{frame_num:04d}.txt"
            if src_label.exists():
                shutil.copy(src_label, dst_label)

        # Update counts
        if split == 'train':
            train_count += len(frame_counts)
        else:
            val_count += len(frame_counts)

        converter.print_statistics()

    # 5. Create dataset YAML
    create_yolo_dataset_yaml(output_dir, train_count, val_count)

    # 6. Cleanup
    print("\n" + "="*80)
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    # 7. Summary
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Total images: {train_count + val_count}")
    print(f"\nDataset ready for YOLOv8 training:")
    print(f"  {output_dir}/dataset.yaml")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
