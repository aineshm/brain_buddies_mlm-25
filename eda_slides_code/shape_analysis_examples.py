#!/usr/bin/env python3
"""
Example usage of the XML Shape Parser for annotation analysis.
This script demonstrates various ways to extract and analyze shape data.
"""

from xml_shape_parser import XMLShapeParser
import matplotlib.pyplot as plt
import numpy as np

def analyze_shapes_by_frame(parser):
    """Analyze how many shapes appear in each frame."""
    all_shapes = parser.parse_all_shapes()
    
    frames = sorted(all_shapes.keys())
    shape_counts = [len(all_shapes[frame]) for frame in frames]
    
    print(f"Frame analysis:")
    print(f"- Frame range: {min(frames)} to {max(frames)}")
    print(f"- Average shapes per frame: {np.mean(shape_counts):.1f}")
    print(f"- Max shapes in frame: {max(shape_counts)} (frame {frames[np.argmax(shape_counts)]})")
    print(f"- Min shapes in frame: {min(shape_counts)} (frame {frames[np.argmin(shape_counts)]})")
    
    return frames, shape_counts

def analyze_shape_sizes(parser):
    """Analyze the sizes of different shape types."""
    all_shapes = parser.parse_all_shapes()
    
    ellipse_areas = []
    box_areas = []
    polygon_point_counts = []
    
    for frame_shapes in all_shapes.values():
        for shape in frame_shapes:
            if shape['shape_type'] == 'ellipse':
                # Approximate area of ellipse
                area = np.pi * shape['rx'] * shape['ry']
                ellipse_areas.append(area)
            elif shape['shape_type'] == 'box':
                # Area of rectangle
                width = shape['xbr'] - shape['xtl']
                height = shape['ybr'] - shape['ytl']
                area = width * height
                box_areas.append(area)
            elif shape['shape_type'] == 'polygon':
                polygon_point_counts.append(len(shape['points']))
    
    print(f"\nShape size analysis:")
    if ellipse_areas:
        print(f"- Ellipses: {len(ellipse_areas)} total, avg area: {np.mean(ellipse_areas):.1f}")
    if box_areas:
        print(f"- Boxes: {len(box_areas)} total, avg area: {np.mean(box_areas):.1f}")
    if polygon_point_counts:
        print(f"- Polygons: {len(polygon_point_counts)} total, avg points: {np.mean(polygon_point_counts):.1f}")

def get_shapes_for_specific_frame(parser, frame_num):
    """Get all shapes for a specific frame."""
    all_shapes = parser.parse_all_shapes()
    
    if frame_num in all_shapes:
        shapes = all_shapes[frame_num]
        print(f"\nFrame {frame_num} contains {len(shapes)} shapes:")
        
        for i, shape in enumerate(shapes):
            print(f"  {i+1}. {shape['shape_type']} - {shape['label']}")
            if shape['shape_type'] == 'ellipse':
                print(f"     Center: ({shape['cx']:.1f}, {shape['cy']:.1f}), Size: {shape['rx']:.1f}x{shape['ry']:.1f}")
            elif shape['shape_type'] == 'box':
                print(f"     Box: ({shape['xtl']:.1f}, {shape['ytl']:.1f}) to ({shape['xbr']:.1f}, {shape['ybr']:.1f})")
            elif shape['shape_type'] in ['polygon', 'polyline']:
                print(f"     {len(shape['points'])} points")
        
        return shapes
    else:
        print(f"Frame {frame_num} not found in annotations")
        return []

def filter_shapes_by_label(parser, target_label):
    """Get all shapes with a specific label across all frames."""
    all_shapes = parser.parse_all_shapes()
    
    filtered_shapes = []
    for frame_num, frame_shapes in all_shapes.items():
        for shape in frame_shapes:
            if shape['label'] == target_label:
                filtered_shapes.append(shape)
    
    print(f"\nFound {len(filtered_shapes)} shapes with label '{target_label}':")
    
    # Group by frame
    frames_with_label = {}
    for shape in filtered_shapes:
        frame = shape['frame']
        if frame not in frames_with_label:
            frames_with_label[frame] = []
        frames_with_label[frame].append(shape)
    
    for frame in sorted(frames_with_label.keys())[:5]:  # Show first 5 frames
        count = len(frames_with_label[frame])
        print(f"  Frame {frame}: {count} {target_label} shapes")
    
    return filtered_shapes

def export_shapes_for_frame(parser, frame_num, output_file):
    """Export shapes from a specific frame to a simple text format."""
    shapes = get_shapes_for_specific_frame(parser, frame_num)
    
    with open(output_file, 'w') as f:
        f.write(f"Shapes for Frame {frame_num}\n")
        f.write("=" * 30 + "\n\n")
        
        for i, shape in enumerate(shapes):
            f.write(f"Shape {i+1}: {shape['shape_type']} - {shape['label']}\n")
            
            if shape['shape_type'] == 'ellipse':
                f.write(f"  Center: ({shape['cx']}, {shape['cy']})\n")
                f.write(f"  Radii: ({shape['rx']}, {shape['ry']})\n")
            elif shape['shape_type'] == 'box':
                f.write(f"  Top-left: ({shape['xtl']}, {shape['ytl']})\n")
                f.write(f"  Bottom-right: ({shape['xbr']}, {shape['ybr']})\n")
            elif shape['shape_type'] in ['polygon', 'polyline']:
                f.write(f"  Points ({len(shape['points'])}):\n")
                for j, (x, y) in enumerate(shape['points'][:5]):  # Show first 5 points
                    f.write(f"    {j+1}: ({x}, {y})\n")
                if len(shape['points']) > 5:
                    f.write(f"    ... and {len(shape['points']) - 5} more points\n")
            
            f.write(f"  Occluded: {shape['occluded']}, Outside: {shape['outside']}\n")
            f.write("\n")
    
    print(f"Frame {frame_num} shapes exported to {output_file}")

def main():
    """Demonstrate various analysis functions."""
    xml_file = "annotated_data_1001/MattLines1annotations.xml"
    
    print("=== XML Shape Analysis Examples ===")
    print(f"Analyzing: {xml_file}\n")
    
    # Initialize parser
    parser = XMLShapeParser(xml_file)
    
    # Basic statistics
    stats = parser.get_summary_stats()
    print(f"Dataset overview:")
    print(f"- Total shapes: {stats['total_shapes']}")
    print(f"- Unique labels: {', '.join(stats['unique_labels'])}")
    print(f"- Shape types: {', '.join(stats['shapes_by_type'].keys())}")
    
    # Analyze by frame
    frames, shape_counts = analyze_shapes_by_frame(parser)
    
    # Analyze shape sizes
    analyze_shape_sizes(parser)
    
    # Look at a specific frame
    get_shapes_for_specific_frame(parser, 0)
    
    # Filter by label
    filter_shapes_by_label(parser, "planktonic")
    
    # Export specific frame
    export_shapes_for_frame(parser, 0, "frame_0_shapes.txt")
    
    print(f"\n=== Analysis complete! ===")

if __name__ == "__main__":
    main()