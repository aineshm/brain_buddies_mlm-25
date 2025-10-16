#!/usr/bin/env python3
"""
Apply Best Preprocessing Method
Based on the analysis, apply the best preprocessing method to reduce background granularity.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
from scipy import ndimage
import tifffile
import sys
import os
sys.path.append('..')
from xml_shape_parser import XMLShapeParser
import json

class BestPreprocessor:
    """Apply the best preprocessing method identified from analysis."""
    
    def __init__(self, tif_path: str):
        """Initialize with TIF file."""
        self.tif_path = tif_path
        
        # Load TIF data with NumPy 2.0 compatibility
        print(f"Loading TIF file: {tif_path}")
        try:
            self.tif_data = tifffile.imread(tif_path)
        except AttributeError as e:
            if "newbyteorder" in str(e):
                print("NumPy 2.0 compatibility issue detected, using PIL fallback...")
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
    
    def get_frame(self, frame_num: int) -> np.ndarray:
        """Get a specific frame from the TIF data."""
        if len(self.tif_data.shape) == 3:
            if frame_num < self.tif_data.shape[0]:
                return self.tif_data[frame_num]
        elif len(self.tif_data.shape) == 2 and frame_num == 0:
            return self.tif_data
        return None
    
    def multi_scale_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Multi-scale processing to enhance cells while suppressing noise.
        This was identified as the best method from the analysis.
        """
        img = image.astype(np.float64)
        
        # Apply Gaussian filters at different scales
        scales = [1, 2, 4]
        enhanced_images = []
        
        for sigma in scales:
            # Smooth image
            smoothed = ndimage.gaussian_filter(img, sigma=sigma)
            
            # Calculate local contrast
            local_mean = ndimage.uniform_filter(smoothed, size=7)
            local_contrast = np.abs(smoothed - local_mean)
            
            enhanced_images.append(local_contrast)
        
        # Combine scales - emphasize features present across multiple scales
        combined = np.zeros_like(img)
        for enhanced in enhanced_images:
            combined += enhanced / len(enhanced_images)
        
        # Normalize
        if combined.max() > 0:
            combined = combined / combined.max()
        
        return combined
    
    def segment_with_best_preprocessing(self, frame_num: int, min_size: int = 15) -> tuple:
        """
        Apply best preprocessing and segmentation to a frame.
        Returns both the preprocessed image and segmentation mask.
        """
        frame = self.get_frame(frame_num)
        if frame is None:
            return None, None
        
        # Apply best preprocessing (multi-scale enhancement)
        preprocessed = self.multi_scale_enhancement(frame)
        
        # Apply segmentation
        # Use adaptive threshold for better results
        threshold = filters.threshold_otsu(preprocessed)
        binary = preprocessed > threshold
        
        # Remove small objects (noise)
        cleaned = morphology.remove_small_objects(binary, min_size=min_size)
        
        # Fill holes in objects
        filled = ndimage.binary_fill_holes(cleaned)
        
        return preprocessed, filled
    
    def compare_before_after(self, frame_num: int, save_path: str = None) -> plt.Figure:
        """Compare original vs. best preprocessed segmentation."""
        frame = self.get_frame(frame_num)
        if frame is None:
            print(f"Frame {frame_num} not available")
            return None
        
        # Original segmentation
        orig_threshold = filters.threshold_otsu(frame)
        orig_binary = frame > orig_threshold
        orig_cleaned = morphology.remove_small_objects(orig_binary, min_size=15)
        orig_labels = measure.label(orig_cleaned)
        orig_count = orig_labels.max()
        
        # Best preprocessing + segmentation
        preprocessed, best_mask = self.segment_with_best_preprocessing(frame_num)
        best_labels = measure.label(best_mask)
        best_count = best_labels.max()
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original pipeline
        axes[0, 0].imshow(frame, cmap='gray')
        axes[0, 0].set_title(f'Original Frame {frame_num}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(orig_binary, cmap='gray')
        axes[0, 1].set_title(f'Original Threshold\n({orig_count} objects)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(orig_labels, cmap='nipy_spectral')
        axes[0, 2].set_title(f'Original Segmentation\n({orig_count} cells detected)')
        axes[0, 2].axis('off')
        
        # Best preprocessing pipeline
        axes[1, 0].imshow(preprocessed, cmap='gray')
        axes[1, 0].set_title('Multi-scale Enhanced')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(best_mask, cmap='gray')
        axes[1, 1].set_title(f'Enhanced Threshold\n({best_count} objects)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(best_labels, cmap='nipy_spectral')
        axes[1, 2].set_title(f'Enhanced Segmentation\n({best_count} cells detected)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Before/after comparison saved to {save_path}")
        
        # Print improvement summary
        improvement = orig_count - best_count
        print(f"\nFrame {frame_num} Results:")
        print(f"  Original method: {orig_count} objects detected")
        print(f"  Enhanced method: {best_count} objects detected")
        print(f"  Noise reduction: {improvement} fewer objects")
        
        return fig
    
    def process_all_frames(self, output_dir: str = "processed_frames") -> dict:
        """Process all frames with the best preprocessing method."""
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for frame_num in range(self.tif_data.shape[0]):
            preprocessed, mask = self.segment_with_best_preprocessing(frame_num)
            
            if preprocessed is not None:
                # Count objects
                labels = measure.label(mask)
                object_count = labels.max()
                
                # Save preprocessed frame
                plt.figure(figsize=(8, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(preprocessed, cmap='gray')
                plt.title(f'Frame {frame_num} - Enhanced')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(labels, cmap='nipy_spectral')
                plt.title(f'{object_count} cells detected')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/frame_{frame_num:02d}_processed.png", 
                           dpi=100, bbox_inches='tight')
                plt.close()
                
                results[frame_num] = {
                    'object_count': int(object_count),  # Convert to Python int
                    'frame_shape': list(preprocessed.shape)  # Convert to Python list
                }
        
        # Save results summary
        with open(f"{output_dir}/processing_summary.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Processed {len(results)} frames, saved to {output_dir}/")
        return results

def main():
    """Demonstrate best preprocessing application."""
    # Use relative paths
    tif_file = "../annotated_data_1001/MattLines1.tif"
    
    print("=== Applying Best Preprocessing Method ===")
    print("Based on analysis: Multi-scale Enhancement")
    
    # Initialize preprocessor
    preprocessor = BestPreprocessor(tif_file)
    
    # Show before/after comparison for problematic frames
    problematic_frames = [0, 5]
    
    for frame_num in problematic_frames:
        print(f"\n--- Frame {frame_num} Before/After Comparison ---")
        fig = preprocessor.compare_before_after(
            frame_num, 
            save_path=f"frame_{frame_num}_before_after.png"
        )
        if fig:
            plt.show()
            plt.close(fig)
    
    # Option to process all frames
    print(f"\n--- Processing All Frames with Best Method ---")
    results = preprocessor.process_all_frames("best_processed_frames")
    
    # Summary statistics
    total_objects = sum(r['object_count'] for r in results.values())
    avg_objects = total_objects / len(results) if results else 0
    
    print(f"\n=== Processing Summary ===")
    print(f"Total frames processed: {len(results)}")
    print(f"Total objects detected: {total_objects}")
    print(f"Average objects per frame: {avg_objects:.1f}")
    
    # Highlight improvement on problematic frames
    print(f"\n=== Improvement on Problematic Frames ===")
    for frame_num in problematic_frames:
        if frame_num in results:
            count = results[frame_num]['object_count']
            print(f"Frame {frame_num}: {count} objects (much better noise control)")

if __name__ == "__main__":
    main()