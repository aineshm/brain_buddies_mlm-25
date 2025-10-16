# TIF Annotation Visualization Tools

This package provides comprehensive tools for visualizing XML shape annotations overlaid on TIF image frames. The tools parse CVAT-format XML annotation files and display various shape types (ellipses, boxes, polygons, polylines) on their corresponding TIF frames.

## 🎯 Quick Start

```bash
# View available frames
python quick_frame_viewer.py list

# View a specific frame
python quick_frame_viewer.py 26

# Run comprehensive analysis
python tif_annotation_visualizer.py
```

## 📁 Generated Files

After running the visualization tools, you'll have:

### Overview Files
- `frames_overview.png` - Multi-frame overview showing 6 frames side-by-side
- `shape_summary.txt` - Detailed text report with statistics

### Individual Frame Visualizations
- `frame_X_detailed.png` - Single frame with annotations and legend
- `frame_X_comparison.png` - Side-by-side original vs annotated view
- `exported_frames/` - Directory with individual annotated frames

### Data Files
- `parsed_shapes.json` - All shape data in structured JSON format

## 🔧 Tool Overview

### 1. `xml_shape_parser.py`
**Core parsing functionality**
- Extracts all shape types from XML annotations
- Organizes data by frame number
- Provides summary statistics
- Exports to JSON format

### 2. `tif_annotation_visualizer.py` 
**Main visualization engine**
- Loads TIF files (handles NumPy 2.0 compatibility)
- Overlays all annotation types with color coding
- Creates detailed visualizations with legends
- Batch exports multiple frames

### 3. `interactive_frame_explorer.py`
**Side-by-side comparisons**
- Shows original vs annotated frame views
- Provides detailed statistics per frame
- Creates comparison visualizations

### 4. `quick_frame_viewer.py`
**Command-line frame viewer**
- Quick access to any frame
- Lists all available frames
- Minimal setup for rapid exploration

### 5. `shape_analysis_examples.py`
**Analysis and filtering examples**
- Demonstrates various analysis techniques
- Shows how to filter by label or frame
- Provides size analysis functions

## 📊 Data Structure

### Shape Types Supported
- **Ellipses**: Center coordinates (cx, cy) + radii (rx, ry)
- **Boxes**: Top-left (xtl, ytl) + bottom-right (xbr, ybr)
- **Polygons**: Series of points forming closed shapes
- **Polylines**: Series of points forming open lines

### Label Types in Dataset
- `planktonic` - Single-celled organisms (red)
- `single dispersed cell` - Individual cells (teal)
- `clump dispersed cell` - Cell clusters (blue)
- `hyphae` - Fungal filaments (green)
- `biofilm` - Microbial communities (yellow)
- `yeast form` - Yeast cells (plum)
- `psuedohyphae` - Elongated yeast (mint)

## 📈 Dataset Statistics

From the MattLines1 dataset:
- **Total annotations**: 1,798 shapes
- **Frames with data**: 29 frames (0-40, with gaps)
- **Image size**: 242×244 pixels, 41 frames total
- **Shape distribution**: 63% ellipses, 30% polylines, 7% polygons
- **Most active frame**: Frame 26 (95 shapes)

### Frame Progression
- **Early frames (0-3)**: Primarily planktonic cells (13-15 ellipses)
- **Middle frames (4-13)**: Introduction of hyphae (polylines)
- **Later frames (14-26)**: Complex scenes with all shape types
- **Peak activity**: Frames 24-26 (80-95 shapes each)

## 🎨 Visualization Features

### Color Coding
Each label type has a distinct color for easy identification:
- Consistent colors across all visualizations
- Semi-transparent shapes to see underlying image
- Bold text labels for shape identification

### Shape Rendering
- **Ellipses**: Outline with center point marker
- **Boxes**: Rectangle outline with center label
- **Polygons**: Filled with transparency for area visualization
- **Polylines**: Bold lines for clear path tracking

### Legends and Statistics
- Automatic legend generation for active labels
- Shape count summaries in titles
- Frame-by-frame statistics
- Export-ready high-resolution outputs

## 🔍 Advanced Usage

### Custom Analysis
```python
from xml_shape_parser import XMLShapeParser

parser = XMLShapeParser("annotations.xml")
all_shapes = parser.parse_all_shapes()

# Find all large ellipses
large_ellipses = []
for frame_shapes in all_shapes.values():
    for shape in frame_shapes:
        if shape['shape_type'] == 'ellipse':
            area = 3.14159 * shape['rx'] * shape['ry']
            if area > 50:
                large_ellipses.append(shape)
```

### Custom Visualization
```python
from tif_annotation_visualizer import TIFAnnotationVisualizer

viz = TIFAnnotationVisualizer("image.tif", "annotations.xml")
fig = viz.visualize_frame(frame_num=5, save_path="custom.png")
```

### Batch Processing
```python
# Export frames 10-20 with annotations
viz.export_all_frames("output_dir", frame_range=(10, 20))
```

## 🛠️ Technical Notes

### NumPy 2.0 Compatibility
The tools automatically handle NumPy 2.0 compatibility issues with tifffile by falling back to PIL for TIFF loading when needed.

### Memory Management
- Figures are automatically closed after export to free memory
- Large batch operations use efficient frame-by-frame processing
- JSON exports handle large datasets without memory issues

### File Format Support
- **Input**: TIFF files (single or multi-frame), XML annotations
- **Output**: PNG images, JSON data, text reports
- **Compatibility**: CVAT annotation format

## 📝 Example Workflow

1. **Parse annotations**: `python xml_shape_parser.py`
2. **Generate overview**: `python tif_annotation_visualizer.py`
3. **Explore frames**: `python quick_frame_viewer.py list`
4. **View specific frame**: `python quick_frame_viewer.py 26`
5. **Create comparisons**: `python interactive_frame_explorer.py`

## 🎉 Results

The tools successfully visualize all 1,798 annotations across 29 frames, providing:
- Clear shape overlays on original microscopy images
- Statistical analysis of annotation distribution
- Export-ready visualizations for presentations
- Structured data for further analysis

Perfect for biological image analysis, annotation validation, and dataset exploration!