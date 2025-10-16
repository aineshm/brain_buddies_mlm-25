#!/usr/bin/env python3
"""
Enhanced Preprocessing for Cell Segmentation
Advanced preprocessing techniques to reduce background noise and improve cell detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, segmentation, restoration
from skimage.feature import local_binary_pattern
from scipy import ndimage
import tifffile
import sys
import os
sys.path.append('..')
from xml_shape_parser import XMLShapeParser
from typing import Dict, List, Tuple, Any
import json

class EnhancedPreprocessing:
    """Advanced preprocessing techniques for better cell detection."""
    
    def __init__(self, tif_path: str, xml_path: str = None):
        """Initialize with TIF file and optional XML annotations."""
        self.tif_path = tif_path
        self.xml_path = xml_path
        
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
        
        # Load annotations if available
        self.annotations = None
        if xml_path and os.path.exists(xml_path):
            print(f"Loading annotations: {xml_path}")
            parser = XMLShapeParser(xml_path)
            self.annotations = parser.parse_all_shapes()
    
    def get_frame(self, frame_num: int) -> np.ndarray:
        """Get a specific frame from the TIF data."""
        if len(self.tif_data.shape) == 3:
            if frame_num < self.tif_data.shape[0]:
                return self.tif_data[frame_num]
        elif len(self.tif_data.shape) == 2 and frame_num == 0:
            return self.tif_data
        return None
    
    def analyze_background_noise(self, image: np.ndarray) -> Dict:
        """Analyze background characteristics to inform preprocessing."""
        # Convert to float and normalize
        img = image.astype(np.float64)
        img = (img - img.min()) / (img.max() - img.min())
        
        # Calculate local statistics
        local_mean = ndimage.uniform_filter(img, size=15)
        local_std = ndimage.generic_filter(img, np.std, size=15)
        
        # Identify likely background regions (low variance, medium intensity)
        background_mask = (local_std < np.percentile(local_std, 30)) & \
                         (local_mean > 0.2) & (local_mean < 0.8)
        
        # Calculate background statistics
        background_intensity = img[background_mask]
        
        analysis = {
            'background_mean': np.mean(background_intensity) if len(background_intensity) > 0 else 0.5,
            'background_std': np.std(background_intensity) if len(background_intensity) > 0 else 0.1,
            'noise_level': np.percentile(local_std, 90),
            'background_fraction': np.sum(background_mask) / background_mask.size,
            'background_mask': background_mask
        }
        
        return analysis
    
    def rolling_ball_background_subtraction(self, image: np.ndarray, radius: int = 25) -> np.ndarray:
        """Rolling ball background subtraction to remove uneven illumination."""
        # Convert to float
        img = image.astype(np.float64)
        
        # Create structuring element (ball)
        selem = morphology.disk(radius)
        
        # Morphological opening approximates rolling ball
        background = morphology.opening(img, selem)
        
        # Subtract background
        corrected = img - background
        
        # Normalize to [0, 1]
        corrected = np.clip(corrected, 0, None)
        if corrected.max() > 0:
            corrected = corrected / corrected.max()
        
        return corrected
    
    def adaptive_background_subtraction(self, image: np.ndarray) -> np.ndarray:
        """Adaptive background subtraction based on local statistics."""
        # Convert to float
        img = image.astype(np.float64)
        img = (img - img.min()) / (img.max() - img.min())
        
        # Calculate local background using large kernel
        kernel_size = max(31, min(img.shape) // 8)  # Adaptive kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Use median filter for robust background estimation
        background = ndimage.median_filter(img, size=kernel_size)
        
        # Subtract background
        corrected = img - background
        
        # Keep only positive values and normalize
        corrected = np.clip(corrected, 0, None)
        if corrected.max() > 0:
            corrected = corrected / corrected.max()
        
        return corrected
    
    def texture_based_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Use texture analysis to suppress background granularity."""
        # Convert to uint8 for LBP
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Local Binary Pattern to capture texture
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(img_uint8, n_points, radius, method='uniform')
        
        # Calculate LBP variance (texture measure)
        lbp_var = ndimage.generic_filter(lbp, np.var, size=5)
        
        # Cells should have more structured texture than random noise
        # High variance in LBP suggests organized structure (cells)
        texture_mask = lbp_var > np.percentile(lbp_var, 60)
        
        # Enhance regions with organized texture
        enhanced = image.copy().astype(np.float64)
        enhanced[texture_mask] = enhanced[texture_mask] * 1.2
        enhanced[~texture_mask] = enhanced[~texture_mask] * 0.8
        
        return np.clip(enhanced, 0, 1)
    
    def multi_scale_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Multi-scale processing to enhance cells while suppressing noise."""
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
    
    def morphological_noise_removal(self, image: np.ndarray) -> np.ndarray:
        """Use morphological operations to remove noise while preserving cells."""
        # Convert to float
        img = image.astype(np.float64)
        img = (img - img.min()) / (img.max() - img.min())
        
        # Morphological opening to remove small noise
        # Use smaller structuring element to preserve cell details
        small_selem = morphology.disk(1)
        opened = morphology.opening(img, small_selem)
        
        # Morphological closing to fill small gaps in cells
        medium_selem = morphology.disk(2)
        processed = morphology.closing(opened, medium_selem)
        
        return processed
    
    def edge_preserving_smoothing(self, image: np.ndarray) -> np.ndarray:
        """Edge-preserving smoothing to reduce noise while keeping cell boundaries."""
        img = image.astype(np.float64)
        
        # Use bilateral filter approximation with Gaussian filters
        # Apply multiple small Gaussian filters instead of edge-preserving filter
        smoothed = img.copy()
        for _ in range(3):
            smoothed = ndimage.gaussian_filter(smoothed, sigma=0.8)
        
        # Combine with original to preserve edges
        alpha = 0.7  # Weight for smoothed image
        result = alpha * smoothed + (1 - alpha) * img
        
        return result
    
    def comprehensive_preprocessing_pipeline(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Complete preprocessing pipeline with multiple techniques."""
        results = {}
        
        # Original image
        results['original'] = image
        
        # Step 1: Background analysis
        bg_analysis = self.analyze_background_noise(image)
        
        # Step 2: Background subtraction methods
        results['rolling_ball'] = self.rolling_ball_background_subtraction(image, radius=20)
        results['adaptive_bg'] = self.adaptive_background_subtraction(image)
        
        # Step 3: Noise reduction
        results['edge_preserving'] = self.edge_preserving_smoothing(image)
        results['morphological'] = self.morphological_noise_removal(image)
        
        # Step 4: Enhancement techniques
        results['texture_enhanced'] = self.texture_based_enhancement(image)
        results['multi_scale'] = self.multi_scale_enhancement(image)
        
        # Step 5: Combined approach
        # Start with background subtraction
        enhanced = self.adaptive_background_subtraction(image)
        # Apply edge-preserving smoothing
        enhanced = self.edge_preserving_smoothing(enhanced)
        # Add texture enhancement
        enhanced = self.texture_based_enhancement(enhanced)
        # Final morphological cleanup
        enhanced = self.morphological_noise_removal(enhanced)
        
        results['combined'] = enhanced
        
        return results, bg_analysis
    
    def evaluate_preprocessing(self, original: np.ndarray, processed: np.ndarray, frame_num: int) -> Dict:
        """Evaluate preprocessing effectiveness."""
        # Simple segmentation for comparison
        def simple_segment(img):
            # Otsu threshold
            threshold = filters.threshold_otsu(img)
            binary = img > threshold
            # Remove small objects
            cleaned = morphology.remove_small_objects(binary, min_size=15)
            return cleaned
        
        # Segment both images
        orig_mask = simple_segment(original)
        proc_mask = simple_segment(processed)
        
        # Count objects
        orig_labels = measure.label(orig_mask)
        proc_labels = measure.label(proc_mask)
        
        orig_count = orig_labels.max()
        proc_count = proc_labels.max()
        
        # Calculate noise reduction metrics
        orig_small_objects = np.sum([r.area < 30 for r in measure.regionprops(orig_labels)])
        proc_small_objects = np.sum([r.area < 30 for r in measure.regionprops(proc_labels)])
        
        # Ground truth comparison if available
        gt_evaluation = None
        if self.annotations and frame_num in self.annotations:
            gt_count = len([s for s in self.annotations[frame_num] 
                           if s['shape_type'] == 'ellipse' and not s.get('outside', False)])
            
            gt_evaluation = {
                'ground_truth_count': gt_count,
                'original_error': abs(orig_count - gt_count),
                'processed_error': abs(proc_count - gt_count),
                'improvement': abs(orig_count - gt_count) - abs(proc_count - gt_count)
            }
        
        return {
            'original_count': orig_count,
            'processed_count': proc_count,
            'count_change': proc_count - orig_count,
            'original_small_objects': orig_small_objects,
            'processed_small_objects': proc_small_objects,
            'noise_reduction': orig_small_objects - proc_small_objects,
            'ground_truth_evaluation': gt_evaluation
        }
    
    def visualize_preprocessing_comparison(self, frame_num: int, save_path: str = None) -> plt.Figure:
        """Create comprehensive visualization of preprocessing methods."""
        frame = self.get_frame(frame_num)
        if frame is None:
            print(f"Frame {frame_num} not available")
            return None
        
        # Apply all preprocessing methods
        results, bg_analysis = self.comprehensive_preprocessing_pipeline(frame)
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        methods = [
            ('original', 'Original Image'),
            ('rolling_ball', 'Rolling Ball BG Sub'),
            ('adaptive_bg', 'Adaptive BG Sub'),
            ('edge_preserving', 'Edge Preserving'),
            ('morphological', 'Morphological'),
            ('texture_enhanced', 'Texture Enhanced'),
            ('multi_scale', 'Multi-scale'),
            ('combined', 'Combined Pipeline'),
        ]
        
        for i, (method, title) in enumerate(methods):
            if i < len(axes) - 1:  # Save last subplot for analysis
                axes[i].imshow(results[method], cmap='gray')
                axes[i].set_title(title)
                axes[i].axis('off')
                
                # Add simple segmentation overlay
                if method != 'original':
                    threshold = filters.threshold_otsu(results[method])
                    binary = results[method] > threshold
                    axes[i].contour(binary, colors='red', linewidths=0.5, alpha=0.7)
        
        # Background analysis visualization
        axes[-1].imshow(bg_analysis['background_mask'], cmap='viridis')
        axes[-1].set_title(f'Background Analysis\nNoise Level: {bg_analysis["noise_level"]:.3f}')
        axes[-1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Preprocessing comparison saved to {save_path}")
        
        return fig
    
    def find_best_preprocessing(self, test_frames: List[int] = None) -> Dict:
        """Find the best preprocessing method across multiple frames."""
        if test_frames is None:
            test_frames = [0, 5] if self.annotations else [0]
        
        methods = ['rolling_ball', 'adaptive_bg', 'edge_preserving', 
                  'morphological', 'texture_enhanced', 'multi_scale', 'combined']
        
        results = {}
        
        for frame_num in test_frames:
            frame = self.get_frame(frame_num)
            if frame is None:
                continue
            
            print(f"Evaluating preprocessing methods on frame {frame_num}...")
            
            # Get all preprocessing results
            preprocessing_results, _ = self.comprehensive_preprocessing_pipeline(frame)
            
            frame_results = {}
            
            for method in methods:
                evaluation = self.evaluate_preprocessing(
                    frame, preprocessing_results[method], frame_num
                )
                frame_results[method] = evaluation
                
                print(f"  {method}: {evaluation['processed_count']} objects " +
                      f"(noise reduction: {evaluation['noise_reduction']})")
            
            results[frame_num] = frame_results
        
        # Find best method based on ground truth accuracy and noise reduction
        method_scores = {}
        for method in methods:
            scores = []
            for frame_results in results.values():
                eval_data = frame_results[method]
                
                # Score based on noise reduction and ground truth accuracy
                noise_score = max(0, eval_data['noise_reduction']) / 10  # Normalize
                
                if eval_data['ground_truth_evaluation']:
                    gt_score = max(0, eval_data['ground_truth_evaluation']['improvement']) / 10
                    scores.append(noise_score + gt_score)
                else:
                    scores.append(noise_score)
            
            method_scores[method] = np.mean(scores) if scores else 0
        
        best_method = max(method_scores, key=method_scores.get)
        
        return {
            'best_method': best_method,
            'method_scores': method_scores,
            'detailed_results': results
        }

def main():
    """Demonstrate enhanced preprocessing techniques."""
    # Use relative paths to parent directory
    tif_file = "../annotated_data_1001/MattLines1.tif"
    xml_file = "../annotated_data_1001/MattLines1annotations.xml"
    
    print("=== Enhanced Preprocessing for Cell Segmentation ===")
    
    # Initialize enhanced preprocessing
    preprocessor = EnhancedPreprocessing(tif_file, xml_file)
    
    # Test on problematic frames (0 and 5 mentioned by user)
    test_frames = [0, 5]
    
    print(f"\n--- Analyzing Problematic Frames {test_frames} ---")
    
    for frame_num in test_frames:
        print(f"\n--- Frame {frame_num} Analysis ---")
        
        # Create comprehensive visualization
        fig = preprocessor.visualize_preprocessing_comparison(
            frame_num, save_path=f"enhanced_preprocessing_frame_{frame_num}.png"
        )
        if fig:
            plt.show()
            plt.close(fig)
    
    # Find best preprocessing method
    print(f"\n--- Finding Best Preprocessing Method ---")
    best_method_results = preprocessor.find_best_preprocessing(test_frames)
    
    print(f"Best preprocessing method: {best_method_results['best_method']}")
    print("Method scores (higher is better):")
    for method, score in best_method_results['method_scores'].items():
        print(f"  {method}: {score:.3f}")
    
    # Save results
    with open('enhanced_preprocessing_results.json', 'w') as f:
        json.dump(best_method_results, f, indent=2, 
                 default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    print(f"\n=== Analysis Complete ===")
    print("Generated files in enhanced_preprocessing/:")
    print("- enhanced_preprocessing_frame_X.png: Method comparisons")
    print("- enhanced_preprocessing_results.json: Quantitative analysis")

if __name__ == "__main__":
    main()