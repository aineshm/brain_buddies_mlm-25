#!/usr/bin/env python3
"""
Quick Frame Viewer
Simple script to quickly view any frame with annotations.
Usage: python quick_frame_viewer.py [frame_number]
"""

import sys
import os
from tif_annotation_visualizer import TIFAnnotationVisualizer
import matplotlib.pyplot as plt

def quick_view_frame(frame_num: int, tif_file: str = None, xml_file: str = None):
    """Quickly view a specific frame with annotations."""
    
    # Default file paths
    if tif_file is None:
        tif_file = "annotated_data_1001/MattLines1.tif"
    if xml_file is None:
        xml_file = "annotated_data_1001/MattLines1annotations.xml"
    
    # Check if files exist
    if not os.path.exists(tif_file):
        print(f"TIF file not found: {tif_file}")
        return False
    
    if not os.path.exists(xml_file):
        print(f"XML file not found: {xml_file}")
        return False
    
    try:
        # Initialize visualizer
        print(f"Loading frame {frame_num}...")
        visualizer = TIFAnnotationVisualizer(tif_file, xml_file)
        
        # Check if frame exists
        available_frames = sorted(visualizer.all_shapes.keys())
        if frame_num not in available_frames:
            print(f"Frame {frame_num} not available.")
            print(f"Available frames: {available_frames}")
            return False
        
        # Visualize the frame
        output_file = f"quick_view_frame_{frame_num}.png"
        fig = visualizer.visualize_frame(frame_num, save_path=output_file, show_plot=True)
        
        # Print frame statistics
        shapes = visualizer.all_shapes.get(frame_num, [])
        visible_shapes = [s for s in shapes if not s.get('outside', False)]
        
        print(f"\nFrame {frame_num} Statistics:")
        print(f"- Total visible shapes: {len(visible_shapes)}")
        
        # Count by type
        type_counts = {}
        label_counts = {}
        for shape in visible_shapes:
            shape_type = shape['shape_type']
            label = shape['label']
            type_counts[shape_type] = type_counts.get(shape_type, 0) + 1
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print(f"- Shape types: {dict(type_counts)}")
        print(f"- Labels: {dict(label_counts)}")
        print(f"- Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error viewing frame {frame_num}: {e}")
        return False

def list_available_frames(tif_file: str = None, xml_file: str = None):
    """List all available frames with basic statistics."""
    
    # Default file paths
    if tif_file is None:
        tif_file = "annotated_data_1001/MattLines1.tif"
    if xml_file is None:
        xml_file = "annotated_data_1001/MattLines1annotations.xml"
    
    try:
        visualizer = TIFAnnotationVisualizer(tif_file, xml_file)
        available_frames = sorted(visualizer.all_shapes.keys())
        
        print(f"Available frames: {len(available_frames)} total")
        print("Frame | Shapes | Types")
        print("-" * 25)
        
        for frame_num in available_frames:
            shapes = visualizer.all_shapes.get(frame_num, [])
            visible_shapes = [s for s in shapes if not s.get('outside', False)]
            
            type_counts = {}
            for shape in visible_shapes:
                shape_type = shape['shape_type']
                type_counts[shape_type] = type_counts.get(shape_type, 0) + 1
            
            types_str = ", ".join([f"{count}{t[0]}" for t, count in type_counts.items()])
            print(f"{frame_num:5d} | {len(visible_shapes):6d} | {types_str}")
        
        return available_frames
        
    except Exception as e:
        print(f"Error listing frames: {e}")
        return []

def main():
    """Main function for command line usage."""
    
    if len(sys.argv) == 1:
        print("=== Quick Frame Viewer ===")
        print("Usage:")
        print("  python quick_frame_viewer.py [frame_number]")
        print("  python quick_frame_viewer.py list")
        print()
        
        # Show available frames
        available_frames = list_available_frames()
        
        if available_frames:
            print(f"\nExample: python quick_frame_viewer.py {available_frames[0]}")
            print(f"Example: python quick_frame_viewer.py {available_frames[-1]}")
    
    elif sys.argv[1].lower() == "list":
        list_available_frames()
    
    else:
        try:
            frame_num = int(sys.argv[1])
            success = quick_view_frame(frame_num)
            if not success:
                print(f"Failed to view frame {frame_num}")
        except ValueError:
            print(f"Invalid frame number: {sys.argv[1]}")
            print("Frame number must be an integer")

if __name__ == "__main__":
    main()