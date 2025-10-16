#!/usr/bin/env python3
"""
TIF Annotation Visualizer
Overlays parsed XML shape annotations onto TIF file frames.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Polygon
from PIL import Image
import tifffile
from xml_shape_parser import XMLShapeParser
import os
from typing import Dict, List, Any, Tuple

class TIFAnnotationVisualizer:
    """Visualizes XML shape annotations overlaid on TIF file frames."""
    
    def __init__(self, tif_path: str, xml_path: str):
        """Initialize with TIF and XML file paths."""
        self.tif_path = tif_path
        self.xml_path = xml_path
        
        # Load TIF file
        print(f"Loading TIF file: {tif_path}")
        try:
            self.tif_data = tifffile.imread(tif_path)
        except AttributeError as e:
            if "newbyteorder" in str(e):
                print("NumPy 2.0 compatibility issue detected, trying alternative method...")
                # Use PIL as fallback for multi-frame TIFF
                from PIL import Image
                img = Image.open(tif_path)
                frames = []
                try:
                    while True:
                        frames.append(np.array(img))
                        img.seek(img.tell() + 1)
                except EOFError:
                    pass
                self.tif_data = np.array(frames) if len(frames) > 1 else frames[0]
            else:
                raise e
        print(f"TIF shape: {self.tif_data.shape}")
        
        # Parse XML annotations
        print(f"Parsing XML annotations: {xml_path}")
        self.parser = XMLShapeParser(xml_path)
        self.all_shapes = self.parser.parse_all_shapes()
        print(f"Found annotations for {len(self.all_shapes)} frames")
        
        # Define colors for different labels
        self.label_colors = {
            'planktonic': '#FF6B6B',       # Red
            'single dispersed cell': '#4ECDC4',  # Teal
            'clump dispersed cell': '#45B7D1',   # Blue
            'hyphae': '#96CEB4',           # Green
            'biofilm': '#FFEAA7',          # Yellow
            'yeast form': '#DDA0DD',       # Plum
            'psuedohyphae': '#98D8C8',     # Mint
            'unknown': '#95A5A6'           # Gray
        }
        
        # Shape visualization settings
        self.shape_settings = {
            'ellipse': {'linewidth': 1.5, 'alpha': 0.7, 'fill': False},
            'box': {'linewidth': 1.5, 'alpha': 0.7, 'fill': False},
            'polygon': {'linewidth': 1.5, 'alpha': 0.3, 'fill': True},
            'polyline': {'linewidth': 2.0, 'alpha': 0.8, 'fill': False}
        }
    
    def get_frame_image(self, frame_num: int) -> np.ndarray:
        """Extract a specific frame from the TIF data."""
        if len(self.tif_data.shape) == 3:
            # Multi-frame TIF
            if frame_num < self.tif_data.shape[0]:
                return self.tif_data[frame_num]
            else:
                print(f"Warning: Frame {frame_num} not available in TIF (max: {self.tif_data.shape[0]-1})")
                return None
        elif len(self.tif_data.shape) == 2:
            # Single frame TIF
            if frame_num == 0:
                return self.tif_data
            else:
                print(f"Warning: Only single frame available, requested frame {frame_num}")
                return None
        else:
            print(f"Unexpected TIF shape: {self.tif_data.shape}")
            return None
    
    def get_label_color(self, label: str) -> str:
        """Get color for a specific label."""
        return self.label_colors.get(label, self.label_colors['unknown'])
    
    def draw_ellipse(self, ax: plt.Axes, shape: Dict[str, Any]):
        """Draw an ellipse shape on the axes."""
        color = self.get_label_color(shape['label'])
        settings = self.shape_settings['ellipse']
        
        ellipse = Ellipse(
            (shape['cx'], shape['cy']),
            2 * shape['rx'],  # width = 2 * radius_x
            2 * shape['ry'],  # height = 2 * radius_y
            color=color,
            **settings
        )
        ax.add_patch(ellipse)
        
        # Add label text
        ax.text(shape['cx'], shape['cy'], shape['label'][:4], 
                fontsize=8, ha='center', va='center', 
                color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    def draw_box(self, ax: plt.Axes, shape: Dict[str, Any]):
        """Draw a box shape on the axes."""
        color = self.get_label_color(shape['label'])
        settings = self.shape_settings['box']
        
        width = shape['xbr'] - shape['xtl']
        height = shape['ybr'] - shape['ytl']
        
        rectangle = patches.Rectangle(
            (shape['xtl'], shape['ytl']),
            width, height,
            color=color,
            **settings
        )
        ax.add_patch(rectangle)
        
        # Add label text at center of box
        center_x = shape['xtl'] + width / 2
        center_y = shape['ytl'] + height / 2
        ax.text(center_x, center_y, shape['label'][:4], 
                fontsize=8, ha='center', va='center', 
                color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    def draw_polygon(self, ax: plt.Axes, shape: Dict[str, Any]):
        """Draw a polygon shape on the axes."""
        color = self.get_label_color(shape['label'])
        settings = self.shape_settings['polygon']
        
        if len(shape['points']) >= 3:
            polygon = Polygon(
                shape['points'],
                color=color,
                **settings
            )
            ax.add_patch(polygon)
            
            # Add label text at centroid
            points = np.array(shape['points'])
            centroid_x = np.mean(points[:, 0])
            centroid_y = np.mean(points[:, 1])
            ax.text(centroid_x, centroid_y, shape['label'][:4], 
                    fontsize=8, ha='center', va='center', 
                    color='white', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    def draw_polyline(self, ax: plt.Axes, shape: Dict[str, Any]):
        """Draw a polyline shape on the axes."""
        color = self.get_label_color(shape['label'])
        settings = self.shape_settings['polyline'].copy()
        # Remove 'fill' parameter as it's not valid for plot()
        settings.pop('fill', None)
        
        if len(shape['points']) >= 2:
            points = np.array(shape['points'])
            ax.plot(points[:, 0], points[:, 1], 
                   color=color, **settings)
            
            # Add label text at midpoint
            mid_idx = len(points) // 2
            mid_x, mid_y = points[mid_idx]
            ax.text(mid_x, mid_y, shape['label'][:4], 
                    fontsize=8, ha='center', va='center', 
                    color='white', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    def visualize_frame(self, frame_num: int, save_path: str = None, show_plot: bool = True) -> plt.Figure:
        """Visualize a single frame with its annotations."""
        # Get frame image
        frame_image = self.get_frame_image(frame_num)
        if frame_image is None:
            return None
        
        # Get shapes for this frame
        shapes = self.all_shapes.get(frame_num, [])
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display the image
        ax.imshow(frame_image, cmap='gray', alpha=0.9)
        
        # Draw all shapes
        shape_counts = {'ellipse': 0, 'box': 0, 'polygon': 0, 'polyline': 0}
        
        for shape in shapes:
            if not shape.get('outside', False):  # Skip shapes marked as outside
                shape_type = shape['shape_type']
                shape_counts[shape_type] += 1
                
                if shape_type == 'ellipse':
                    self.draw_ellipse(ax, shape)
                elif shape_type == 'box':
                    self.draw_box(ax, shape)
                elif shape_type == 'polygon':
                    self.draw_polygon(ax, shape)
                elif shape_type == 'polyline':
                    self.draw_polyline(ax, shape)
        
        # Set title and labels
        total_shapes = sum(shape_counts.values())
        shape_summary = ", ".join([f"{count} {stype}" for stype, count in shape_counts.items() if count > 0])
        
        ax.set_title(f'Frame {frame_num} - {total_shapes} annotations\n{shape_summary}', 
                    fontsize=14, pad=20)
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)
        
        # Create legend
        legend_elements = []
        for label, color in self.label_colors.items():
            # Check if this label exists in current frame
            if any(shape['label'] == label for shape in shapes):
                legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, label=label))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Frame {frame_num} saved to {save_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        return fig
    
    def create_frame_overview(self, max_frames: int = 9, save_path: str = None) -> plt.Figure:
        """Create an overview showing multiple frames with annotations."""
        available_frames = sorted(self.all_shapes.keys())[:max_frames]
        
        # Calculate grid layout
        n_frames = len(available_frames)
        n_cols = min(3, n_frames)
        n_rows = (n_frames + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_frames == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, frame_num in enumerate(available_frames):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Get and display frame
            frame_image = self.get_frame_image(frame_num)
            if frame_image is not None:
                ax.imshow(frame_image, cmap='gray', alpha=0.9)
                
                # Draw shapes (simplified for overview)
                shapes = self.all_shapes.get(frame_num, [])
                visible_shapes = [s for s in shapes if not s.get('outside', False)]
                
                for shape in visible_shapes:
                    color = self.get_label_color(shape['label'])
                    
                    if shape['shape_type'] == 'ellipse':
                        ellipse = Ellipse(
                            (shape['cx'], shape['cy']),
                            2 * shape['rx'], 2 * shape['ry'],
                            color=color, fill=False, linewidth=1, alpha=0.8
                        )
                        ax.add_patch(ellipse)
                    elif shape['shape_type'] == 'box':
                        width = shape['xbr'] - shape['xtl']
                        height = shape['ybr'] - shape['ytl']
                        rectangle = patches.Rectangle(
                            (shape['xtl'], shape['ytl']), width, height,
                            color=color, fill=False, linewidth=1, alpha=0.8
                        )
                        ax.add_patch(rectangle)
                
                ax.set_title(f'Frame {frame_num}\n{len(visible_shapes)} shapes')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide empty subplots
        for i in range(n_frames, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Overview saved to {save_path}")
        
        return fig
    
    def export_all_frames(self, output_dir: str, frame_range: Tuple[int, int] = None):
        """Export all frames with annotations to individual image files."""
        os.makedirs(output_dir, exist_ok=True)
        
        available_frames = sorted(self.all_shapes.keys())
        if frame_range:
            available_frames = [f for f in available_frames if frame_range[0] <= f <= frame_range[1]]
        
        print(f"Exporting {len(available_frames)} frames to {output_dir}")
        
        for frame_num in available_frames:
            output_path = os.path.join(output_dir, f"frame_{frame_num:03d}_annotated.png")
            fig = self.visualize_frame(frame_num, save_path=output_path, show_plot=False)
            if fig:
                plt.close(fig)  # Free memory
        
        print(f"Export complete: {len(available_frames)} frames saved")

def main():
    """Example usage of the TIF annotation visualizer."""
    # File paths
    tif_file = "annotated_data_1001/MattLines1.tif"
    xml_file = "annotated_data_1001/MattLines1annotations.xml"
    
    if not os.path.exists(tif_file):
        print(f"TIF file not found: {tif_file}")
        return
    
    if not os.path.exists(xml_file):
        print(f"XML file not found: {xml_file}")
        return
    
    print("=== TIF Annotation Visualizer ===")
    
    # Initialize visualizer
    visualizer = TIFAnnotationVisualizer(tif_file, xml_file)
    
    # Show overview of multiple frames
    print("\nCreating overview of first 6 frames...")
    overview_fig = visualizer.create_frame_overview(max_frames=6, save_path="frames_overview.png")
    
    # Visualize specific frames
    print("\nVisualizing individual frames...")
    
    # Frame 0
    fig0 = visualizer.visualize_frame(0, save_path="frame_0_detailed.png", show_plot=False)
    
    # Frame with most annotations (from previous analysis: frame 26)
    if 26 in visualizer.all_shapes:
        fig26 = visualizer.visualize_frame(26, save_path="frame_26_detailed.png", show_plot=False)
    
    # Export first 5 frames
    print("\nExporting first 5 frames...")
    visualizer.export_all_frames("exported_frames", frame_range=(0, 4))
    
    print("\n=== Visualization complete! ===")
    print("Generated files:")
    print("- frames_overview.png: Overview of multiple frames")
    print("- frame_0_detailed.png: Detailed view of frame 0")
    print("- frame_26_detailed.png: Detailed view of frame 26 (if available)")
    print("- exported_frames/: Directory with individual annotated frames")

if __name__ == "__main__":
    main()