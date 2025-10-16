# XML Shape Parser Documentation

This package provides comprehensive tools for parsing and analyzing XML annotation files containing shape data for image frames.

## Overview

The XML annotation format contains several types of shapes:
- **Ellipses**: Defined by center coordinates (cx, cy) and radii (rx, ry)
- **Boxes**: Defined by top-left (xtl, ytl) and bottom-right (xbr, ybr) coordinates  
- **Polygons**: Defined by a series of points forming a closed shape
- **Polylines**: Defined by a series of points forming an open line

## Quick Start

```python
from xml_shape_parser import XMLShapeParser

# Initialize parser
parser = XMLShapeParser("path/to/annotations.xml")

# Get all shapes organized by frame
all_shapes = parser.parse_all_shapes()

# Get summary statistics
stats = parser.get_summary_stats()
print(f"Total shapes: {stats['total_shapes']}")
print(f"Shape types: {stats['shapes_by_type']}")

# Export to JSON
parser.export_to_json("output.json")
```

## Features

### 1. Parse Individual Shape Types
```python
# Get only ellipses
ellipses = parser.parse_ellipses()

# Get only boxes  
boxes = parser.parse_boxes()

# Get only polygons
polygons = parser.parse_polygons()

# Get only polylines
polylines = parser.parse_polylines()
```

### 2. Frame-based Analysis
```python
# Get shapes for specific frame
frame_5_shapes = all_shapes[5] if 5 in all_shapes else []

# Count shapes per frame
for frame_num, shapes in all_shapes.items():
    print(f"Frame {frame_num}: {len(shapes)} shapes")
```

### 3. Label-based Filtering
```python
# Filter shapes by label
planktonic_shapes = []
for frame_shapes in all_shapes.values():
    for shape in frame_shapes:
        if shape['label'] == 'planktonic':
            planktonic_shapes.append(shape)
```

### 4. Shape Property Access
Each shape contains these common properties:
- `shape_type`: "ellipse", "box", "polygon", or "polyline"
- `label`: The annotation label/class
- `frame`: Frame number
- `keyframe`: Boolean indicating if this is a keyframe
- `outside`: Boolean indicating if shape is outside frame
- `occluded`: Boolean indicating if shape is occluded
- `z_order`: Layer order for overlapping shapes

#### Ellipse-specific properties:
- `cx`, `cy`: Center coordinates
- `rx`, `ry`: X and Y radii

#### Box-specific properties:
- `xtl`, `ytl`: Top-left coordinates
- `xbr`, `ybr`: Bottom-right coordinates

#### Polygon/Polyline-specific properties:
- `points`: List of (x, y) coordinate tuples

## Example Analysis Tasks

### Count shapes by type and label:
```python
stats = parser.get_summary_stats()
print("Shapes by type:", stats['shapes_by_type'])
print("Shapes by label:", stats['shapes_by_label'])
```

### Find largest ellipses:
```python
ellipses = parser.parse_ellipses()
large_ellipses = []

for frame_ellipses in ellipses.values():
    for ellipse in frame_ellipses:
        area = 3.14159 * ellipse['rx'] * ellipse['ry']
        if area > 50:  # Threshold
            large_ellipses.append((ellipse, area))

# Sort by area
large_ellipses.sort(key=lambda x: x[1], reverse=True)
```

### Track shape movement across frames:
```python
# Group shapes by track ID (if available in XML structure)
# This would require additional parsing of track elements
```

## Output Formats

### JSON Export
The `export_to_json()` method creates a structured JSON file with:
- Metadata (source file, statistics)
- All shapes organized by frame
- Complete shape properties

### Custom Text Export
Use the example functions in `shape_analysis_examples.py` to create custom text exports for specific frames or filtered shape sets.

## Data Statistics from Sample File

From the MattLines1annotations.xml file:
- **Total shapes**: 1,798
- **Frames**: 29 (0-40, with some gaps)
- **Shape types**: 
  - Ellipses: 1,132 (63%)
  - Polylines: 544 (30%) 
  - Polygons: 122 (7%)
- **Labels**: planktonic, single dispersed cell, hyphae, biofilm, clump dispersed cell, yeast form, psuedohyphae
- **Average shapes per frame**: 62
- **Frame with most shapes**: Frame 26 (190 shapes)

## Notes

- Shapes can span multiple frames (tracks)
- Some shapes may be marked as "outside" (not visible in current frame)
- Z-order determines layering when shapes overlap
- Keyframes indicate manually annotated positions vs interpolated