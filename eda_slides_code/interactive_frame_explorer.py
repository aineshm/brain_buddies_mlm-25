#!/usr/bin/env python3
"""
Interactive TIF Frame Explorer
A simplified version for exploring individual frames with annotations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Polygon
from tif_annotation_visualizer import TIFAnnotationVisualizer
import os

def explore_frame_interactive(tif_file: str, xml_file: str, frame_num: int = 0):
    """Interactively explore a single frame with detailed annotations."""
    
    if not os.path.exists(tif_file) or not os.path.exists(xml_file):
        print(f"File not found: {tif_file} or {xml_file}")
        return
    
    # Initialize visualizer
    visualizer = TIFAnnotationVisualizer(tif_file, xml_file)
    
    # Get available frames
    available_frames = sorted(visualizer.all_shapes.keys())
    print(f"Available frames: {available_frames}")
    
    if frame_num not in available_frames:
        print(f"Frame {frame_num} not available. Using frame {available_frames[0]}")
        frame_num = available_frames[0]
    
    # Get frame data
    frame_image = visualizer.get_frame_image(frame_num)
    shapes = visualizer.all_shapes.get(frame_num, [])
    
    if frame_image is None:
        print(f"Could not load frame {frame_num}")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Original image
    ax1.imshow(frame_image, cmap='gray')
    ax1.set_title(f'Frame {frame_num} - Original Image\nSize: {frame_image.shape}', fontsize=12)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    
    # Right plot: Image with annotations
    ax2.imshow(frame_image, cmap='gray', alpha=0.8)
    
    # Draw annotations and collect statistics
    shape_stats = {}
    label_stats = {}
    
    for shape in shapes:
        if not shape.get('outside', False):  # Skip shapes marked as outside
            shape_type = shape['shape_type']
            label = shape['label']
            
            # Update statistics
            shape_stats[shape_type] = shape_stats.get(shape_type, 0) + 1
            label_stats[label] = label_stats.get(label, 0) + 1
            
            # Draw shape
            color = visualizer.get_label_color(label)
            
            if shape_type == 'ellipse':
                ellipse = Ellipse(
                    (shape['cx'], shape['cy']),
                    2 * shape['rx'], 2 * shape['ry'],
                    color=color, fill=False, linewidth=2, alpha=0.8
                )
                ax2.add_patch(ellipse)
                
                # Add small center point
                ax2.plot(shape['cx'], shape['cy'], 'o', color=color, markersize=3)
                
            elif shape_type == 'box':
                width = shape['xbr'] - shape['xtl']
                height = shape['ybr'] - shape['ytl']
                rectangle = patches.Rectangle(
                    (shape['xtl'], shape['ytl']), width, height,
                    color=color, fill=False, linewidth=2, alpha=0.8
                )
                ax2.add_patch(rectangle)
                
            elif shape_type == 'polygon':
                if len(shape['points']) >= 3:
                    polygon = Polygon(
                        shape['points'], color=color, 
                        fill=True, alpha=0.3, linewidth=2
                    )
                    ax2.add_patch(polygon)
                    
            elif shape_type == 'polyline':
                if len(shape['points']) >= 2:
                    points = np.array(shape['points'])
                    ax2.plot(points[:, 0], points[:, 1], 
                           color=color, linewidth=2, alpha=0.8)
    
    # Set title with statistics
    total_visible = sum(shape_stats.values())
    shape_summary = ", ".join([f"{count} {stype}" for stype, count in shape_stats.items()])
    ax2.set_title(f'Frame {frame_num} - With Annotations\n{total_visible} shapes: {shape_summary}', fontsize=12)
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    
    # Create legend for labels
    legend_elements = []
    for label, color in visualizer.label_colors.items():
        if label in label_stats:
            count = label_stats[label]
            legend_elements.append(
                plt.Line2D([0], [0], color=color, lw=3, 
                          label=f'{label} ({count})')
            )
    
    if legend_elements:
        ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
    
    plt.tight_layout()
    
    # Print detailed statistics
    print(f"\n=== Frame {frame_num} Analysis ===")
    print(f"Image dimensions: {frame_image.shape}")
    print(f"Total visible shapes: {total_visible}")
    print(f"Shape breakdown: {shape_stats}")
    print(f"Label breakdown: {label_stats}")
    
    # Save the comparison
    output_file = f"frame_{frame_num}_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to: {output_file}")
    
    plt.show()
    
    return fig

def create_shape_summary_report(tif_file: str, xml_file: str, output_file: str = "shape_summary.txt"):
    """Create a text report summarizing all shapes across all frames."""
    
    visualizer = TIFAnnotationVisualizer(tif_file, xml_file)
    
    with open(output_file, 'w') as f:
        f.write("TIF Annotation Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        stats = visualizer.parser.get_summary_stats()
        f.write(f"Dataset Overview:\n")
        f.write(f"- Source TIF: {tif_file}\n")
        f.write(f"- Source XML: {xml_file}\n")
        f.write(f"- TIF dimensions: {visualizer.tif_data.shape}\n")
        f.write(f"- Total shapes: {stats['total_shapes']}\n")
        f.write(f"- Total frames with annotations: {stats['total_frames']}\n")
        f.write(f"- Shape types: {', '.join(stats['shapes_by_type'].keys())}\n")
        f.write(f"- Unique labels: {', '.join(stats['unique_labels'])}\n\n")
        
        # Shape type breakdown
        f.write("Shape Type Breakdown:\n")
        for shape_type, count in stats['shapes_by_type'].items():
            percentage = (count / stats['total_shapes']) * 100
            f.write(f"- {shape_type}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Label breakdown
        f.write("Label Breakdown:\n")
        for label, count in stats['shapes_by_label'].items():
            percentage = (count / stats['total_shapes']) * 100
            f.write(f"- {label}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        # Frame-by-frame breakdown
        f.write("Frame-by-Frame Analysis:\n")
        all_shapes = visualizer.all_shapes
        
        for frame_num in sorted(all_shapes.keys())[:10]:  # First 10 frames
            shapes = all_shapes[frame_num]
            visible_shapes = [s for s in shapes if not s.get('outside', False)]
            
            f.write(f"\nFrame {frame_num}: {len(visible_shapes)} visible shapes\n")
            
            # Count by type and label for this frame
            frame_types = {}
            frame_labels = {}
            
            for shape in visible_shapes:
                shape_type = shape['shape_type']
                label = shape['label']
                frame_types[shape_type] = frame_types.get(shape_type, 0) + 1
                frame_labels[label] = frame_labels.get(label, 0) + 1
            
            f.write(f"  Types: {dict(frame_types)}\n")
            f.write(f"  Labels: {dict(frame_labels)}\n")
        
        if len(all_shapes) > 10:
            f.write(f"\n... and {len(all_shapes) - 10} more frames\n")
    
    print(f"Summary report saved to: {output_file}")

def main():
    """Interactive frame exploration."""
    tif_file = "annotated_data_1001/MattLines1.tif"
    xml_file = "annotated_data_1001/MattLines1annotations.xml"
    
    print("=== Interactive TIF Frame Explorer ===")
    
    # Create summary report
    create_shape_summary_report(tif_file, xml_file)
    
    # Explore specific frames
    frames_to_explore = [0, 1, 5, 26]  # Frame 26 has the most shapes
    
    for frame_num in frames_to_explore:
        print(f"\n--- Exploring Frame {frame_num} ---")
        try:
            fig = explore_frame_interactive(tif_file, xml_file, frame_num)
            if fig:
                plt.close(fig)  # Close to free memory
        except Exception as e:
            print(f"Error exploring frame {frame_num}: {e}")
    
    print("\n=== Exploration complete! ===")

if __name__ == "__main__":
    main()