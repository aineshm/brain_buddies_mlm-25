"""
Convert CVAT XML annotations to YOLO format for YOLOv8 instance segmentation training.

YOLO format per line:
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

Where coordinates are normalized (0-1) polygon vertices.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import json
from collections import defaultdict


class XMLToYOLOConverter:
    """Convert CVAT XML annotations to YOLO segmentation format."""

    # Class mapping - matches your 7 categories
    CLASS_MAPPING = {
        'single dispersed cell': 0,
        'clump dispersed cell': 1,
        'planktonic': 2,
        'yeast form': 3,
        'psuedohyphae': 4,  # Note: keeping original typo for consistency
        'hyphae': 5,
        'biofilm': 6
    }

    def __init__(self, xml_path: str, output_dir: str, image_width: int, image_height: int):
        """
        Initialize converter.

        Args:
            xml_path: Path to CVAT XML annotation file
            output_dir: Directory to save YOLO format annotations
            image_width: Width of images in pixels
            image_height: Height of images in pixels
        """
        self.xml_path = Path(xml_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_width = image_width
        self.image_height = image_height

        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()

        # Statistics
        self.stats = defaultdict(int)

    def ellipse_to_polygon(self, cx: float, cy: float, rx: float, ry: float,
                           num_points: int = 16) -> List[Tuple[float, float]]:
        """
        Convert ellipse to polygon approximation.

        Args:
            cx, cy: Center coordinates
            rx, ry: Radii
            num_points: Number of points to approximate ellipse

        Returns:
            List of (x, y) coordinates forming polygon
        """
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points = []
        for angle in angles:
            x = cx + rx * np.cos(angle)
            y = cy + ry * np.sin(angle)
            points.append((x, y))
        return points

    def polyline_to_polygon(self, points: List[Tuple[float, float]],
                           width: float = 3.0) -> List[Tuple[float, float]]:
        """
        Convert polyline (hyphae) to polygon by adding width.

        Args:
            points: List of polyline points
            width: Width to add around polyline

        Returns:
            Polygon points
        """
        if len(points) < 2:
            return points

        # Simple approach: create perpendicular offsets
        polygon = []

        # Add offset on one side
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            # Perpendicular vector
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx / length, dy / length
                perp_x, perp_y = -dy * width, dx * width
                polygon.append((x1 + perp_x, y1 + perp_y))

        # Add last point
        polygon.append((points[-1][0] + perp_x, points[-1][1] + perp_y))

        # Add offset on other side (reverse)
        for i in range(len(points) - 1, 0, -1):
            x1, y1 = points[i]
            x2, y2 = points[i - 1]

            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx / length, dy / length
                perp_x, perp_y = -dy * width, dx * width
                polygon.append((x1 + perp_x, y1 + perp_y))

        # Close polygon
        polygon.append((points[0][0] + perp_x, points[0][1] + perp_y))

        return polygon

    def normalize_coordinates(self, points: List[Tuple[float, float]]) -> List[float]:
        """
        Normalize coordinates to 0-1 range.

        Args:
            points: List of (x, y) coordinates

        Returns:
            Flattened list of normalized coordinates
        """
        normalized = []
        for x, y in points:
            # Clamp to image bounds
            x = max(0, min(x, self.image_width))
            y = max(0, min(y, self.image_height))

            # Normalize
            norm_x = x / self.image_width
            norm_y = y / self.image_height

            normalized.extend([norm_x, norm_y])

        return normalized

    def parse_ellipse(self, shape_elem) -> Tuple[int, List[float]]:
        """Parse ellipse annotation and convert to polygon."""
        label = shape_elem.get('label')
        if label not in self.CLASS_MAPPING:
            return None, None

        class_id = self.CLASS_MAPPING[label]

        cx = float(shape_elem.get('cx'))
        cy = float(shape_elem.get('cy'))
        rx = float(shape_elem.get('rx'))
        ry = float(shape_elem.get('ry'))

        # Convert ellipse to polygon
        polygon_points = self.ellipse_to_polygon(cx, cy, rx, ry)
        normalized = self.normalize_coordinates(polygon_points)

        self.stats[f'ellipse_{label}'] += 1

        return class_id, normalized

    def parse_polygon(self, shape_elem) -> Tuple[int, List[float]]:
        """Parse polygon annotation."""
        label = shape_elem.get('label')
        if label not in self.CLASS_MAPPING:
            return None, None

        class_id = self.CLASS_MAPPING[label]

        points_str = shape_elem.get('points')
        points = []
        for point_pair in points_str.split(';'):
            if point_pair.strip():
                x, y = map(float, point_pair.split(','))
                points.append((x, y))

        normalized = self.normalize_coordinates(points)

        self.stats[f'polygon_{label}'] += 1

        return class_id, normalized

    def parse_polyline(self, shape_elem) -> Tuple[int, List[float]]:
        """Parse polyline annotation and convert to polygon."""
        label = shape_elem.get('label')
        if label not in self.CLASS_MAPPING:
            return None, None

        class_id = self.CLASS_MAPPING[label]

        points_str = shape_elem.get('points')
        points = []
        for point_pair in points_str.split(';'):
            if point_pair.strip():
                x, y = map(float, point_pair.split(','))
                points.append((x, y))

        # Convert polyline to polygon with width
        polygon_points = self.polyline_to_polygon(points, width=3.0)
        normalized = self.normalize_coordinates(polygon_points)

        self.stats[f'polyline_{label}'] += 1

        return class_id, normalized

    def convert_frame(self, frame_num: int, output_name: str) -> int:
        """
        Convert annotations for a specific frame.

        Args:
            frame_num: Frame number to convert
            output_name: Output filename (without extension)

        Returns:
            Number of annotations written
        """
        annotations = []

        # Parse all tracks
        for track in self.root.findall('.//track'):
            label = track.get('label')

            # Find shapes for this frame
            for shape_elem in track:
                shape_frame = int(shape_elem.get('frame', -1))
                if shape_frame != frame_num:
                    continue

                # Skip if outside frame or occluded
                if shape_elem.get('outside') == '1':
                    continue

                # Parse based on shape type
                class_id, normalized_coords = None, None

                if shape_elem.tag == 'ellipse':
                    shape_elem.set('label', label)  # Add label for parsing
                    class_id, normalized_coords = self.parse_ellipse(shape_elem)
                elif shape_elem.tag == 'polygon':
                    shape_elem.set('label', label)
                    class_id, normalized_coords = self.parse_polygon(shape_elem)
                elif shape_elem.tag == 'polyline':
                    shape_elem.set('label', label)
                    class_id, normalized_coords = self.parse_polyline(shape_elem)

                if class_id is not None and normalized_coords:
                    # Format: <class_id> <x1> <y1> <x2> <y2> ...
                    coords_str = ' '.join([f'{c:.6f}' for c in normalized_coords])
                    annotations.append(f'{class_id} {coords_str}')

        # Write to file
        if annotations:
            output_path = self.output_dir / f'{output_name}.txt'
            with open(output_path, 'w') as f:
                f.write('\n'.join(annotations))

        return len(annotations)

    def convert_all_frames(self, base_name: str) -> Dict[int, int]:
        """
        Convert all frames in the annotation file.

        Args:
            base_name: Base name for output files (e.g., 'training')

        Returns:
            Dictionary mapping frame number to annotation count
        """
        frame_counts = {}

        # Get all unique frame numbers
        frames = set()
        for track in self.root.findall('.//track'):
            for shape_elem in track:
                frame_num = int(shape_elem.get('frame', -1))
                if frame_num >= 0:
                    frames.add(frame_num)

        print(f"Converting {len(frames)} frames from {self.xml_path.name}...")

        for frame_num in sorted(frames):
            output_name = f'{base_name}_frame_{frame_num:04d}'
            count = self.convert_frame(frame_num, output_name)
            frame_counts[frame_num] = count

            if count > 0:
                print(f"  Frame {frame_num:04d}: {count} annotations")

        return frame_counts

    def save_class_mapping(self):
        """Save class mapping to YAML file for YOLO training."""
        yaml_path = self.output_dir / 'classes.yaml'

        # Create reverse mapping (id -> name)
        id_to_name = {v: k for k, v in self.CLASS_MAPPING.items()}

        yaml_content = f"""# Candida albicans cell morphology classes
# Generated from CVAT XML annotations

names:
"""
        for i in range(len(self.CLASS_MAPPING)):
            if i in id_to_name:
                yaml_content += f"  {i}: {id_to_name[i]}\n"

        yaml_content += f"\nnc: {len(self.CLASS_MAPPING)}  # number of classes\n"

        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"\nClass mapping saved to {yaml_path}")

    def print_statistics(self):
        """Print conversion statistics."""
        print("\n" + "="*60)
        print("CONVERSION STATISTICS")
        print("="*60)

        total = sum(self.stats.values())
        print(f"Total annotations: {total}\n")

        print("By type and class:")
        for key, count in sorted(self.stats.items()):
            print(f"  {key}: {count}")

        print("="*60 + "\n")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert CVAT XML annotations to YOLO format'
    )
    parser.add_argument('xml_path', help='Path to XML annotation file')
    parser.add_argument('output_dir', help='Output directory for YOLO annotations')
    parser.add_argument('--width', type=int, required=True, help='Image width')
    parser.add_argument('--height', type=int, required=True, help='Image height')
    parser.add_argument('--base-name', default='image', help='Base name for output files')

    args = parser.parse_args()

    # Convert annotations
    converter = XMLToYOLOConverter(
        args.xml_path,
        args.output_dir,
        args.width,
        args.height
    )

    frame_counts = converter.convert_all_frames(args.base_name)
    converter.save_class_mapping()
    converter.print_statistics()

    print(f"\nConverted {len(frame_counts)} frames")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
