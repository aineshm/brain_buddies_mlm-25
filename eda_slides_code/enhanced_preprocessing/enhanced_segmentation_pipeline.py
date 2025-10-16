#!/usr/bin/env python3
"""
Enhanced Segmentation Pipeline with Improved Preprocessing
Combines adaptive background subtraction and multi-scale enhancement with baseline models.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure, segmentation, feature
from scipy import ndimage
import tifffile
import sys
import os
import json
import time
from datetime import datetime
sys.path.append('..')
from xml_shape_parser import XMLShapeParser
from typing import Dict, List, Tuple, Any

class EnhancedSegmentationPipeline:
    """Enhanced segmentation pipeline with improved preprocessing."""
    
    def __init__(self, tif_path: str, xml_path: str = None):
        """Initialize with TIF file and optional XML annotations."""
        self.tif_path = tif_path
        self.xml_path = xml_path
        self.log_entries = []
        
        # Load TIF data with NumPy 2.0 compatibility
        self.log("Loading TIF file", f"Path: {tif_path}")
        try:
            self.tif_data = tifffile.imread(tif_path)
        except AttributeError as e:
            if "newbyteorder" in str(e):
                self.log("NumPy compatibility issue", "Using PIL fallback for TIF loading")
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
        
        self.log("TIF loaded successfully", f"Shape: {self.tif_data.shape}")
        
        # Load annotations if available
        self.annotations = None
        if xml_path and os.path.exists(xml_path):
            self.log("Loading annotations", f"Path: {xml_path}")
            parser = XMLShapeParser(xml_path)
            self.annotations = parser.parse_all_shapes()
            frame_count = len(self.annotations)
            shape_count = sum(len(shapes) for shapes in self.annotations.values())
            self.log("Annotations loaded", f"{frame_count} frames, {shape_count} total shapes")
    
    def log(self, action: str, details: str = "", result: str = ""):
        """Add entry to log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "timestamp": timestamp,
            "action": action,
            "details": details,
            "result": result
        }
        self.log_entries.append(entry)
        print(f"[{timestamp}] {action}: {details} {result}")
    
    def get_frame(self, frame_num: int) -> np.ndarray:
        """Get a specific frame from the TIF data."""
        if len(self.tif_data.shape) == 3:
            if frame_num < self.tif_data.shape[0]:
                return self.tif_data[frame_num]
        elif len(self.tif_data.shape) == 2 and frame_num == 0:
            return self.tif_data
        return None
    
    def adaptive_background_subtraction(self, image: np.ndarray) -> np.ndarray:
        """Adaptive background subtraction - best method for noise reduction."""
        self.log("Applying adaptive background subtraction", f"Image shape: {image.shape}")
        
        # Convert to float and normalize
        img = image.astype(np.float64)
        img = (img - img.min()) / (img.max() - img.min())
        
        # Calculate adaptive kernel size
        kernel_size = max(31, min(img.shape) // 8)
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
        
        self.log("Adaptive background subtraction complete", f"Kernel size: {kernel_size}")
        return corrected
    
    def multi_scale_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Multi-scale enhancement for feature preservation."""
        self.log("Applying multi-scale enhancement", f"Image shape: {image.shape}")
        
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
        
        self.log("Multi-scale enhancement complete", f"Scales used: {scales}")
        return combined
    
    def combined_preprocessing(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply both preprocessing methods and return results."""
        self.log("Starting combined preprocessing", f"Original image shape: {image.shape}")
        
        # Original normalized image
        original = image.astype(np.float64)
        original = (original - original.min()) / (original.max() - original.min())
        
        # Apply both preprocessing methods
        adaptive_bg = self.adaptive_background_subtraction(original)
        multi_scale = self.multi_scale_enhancement(original)
        
        # Create combined approach: adaptive BG first, then multi-scale
        combined = self.multi_scale_enhancement(adaptive_bg)
        
        results = {
            'original': original,
            'adaptive_bg': adaptive_bg,
            'multi_scale': multi_scale,
            'combined': combined
        }
        
        self.log("Combined preprocessing complete", f"Generated {len(results)} processed versions")
        return results
    
    # Baseline segmentation methods (from original baseline)
    def threshold_based_segmentation(self, image: np.ndarray, min_size: int = 15) -> np.ndarray:
        """Threshold-based segmentation."""
        self.log("Applying threshold-based segmentation", f"Min size: {min_size}")
        
        # Otsu threshold
        threshold = filters.threshold_otsu(image)
        binary = image > threshold
        
        # Remove small objects
        cleaned = morphology.remove_small_objects(binary, min_size=min_size)
        
        # Fill holes
        filled = ndimage.binary_fill_holes(cleaned)
        
        return filled
    
    def adaptive_threshold_segmentation(self, image: np.ndarray, min_size: int = 15) -> np.ndarray:
        """Adaptive threshold segmentation."""
        self.log("Applying adaptive threshold segmentation", f"Min size: {min_size}")
        
        # Convert to uint8
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Apply adaptive threshold
        binary = filters.threshold_local(image, block_size=35, offset=0.01)
        binary_mask = image > binary
        
        # Clean up
        cleaned = morphology.remove_small_objects(binary_mask, min_size=min_size)
        filled = ndimage.binary_fill_holes(cleaned)
        
        return filled
    
    def edge_based_segmentation(self, image: np.ndarray, min_size: int = 15) -> np.ndarray:
        """Edge-based segmentation."""
        self.log("Applying edge-based segmentation", f"Min size: {min_size}")
        
        # Detect edges using Canny
        edges = feature.canny(image, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
        
        # Fill edge regions
        filled = ndimage.binary_fill_holes(edges)
        
        # Remove small objects
        cleaned = morphology.remove_small_objects(filled, min_size=min_size)
        
        return cleaned
    
    def watershed_segmentation(self, image: np.ndarray, min_size: int = 15) -> np.ndarray:
        """Watershed segmentation."""
        self.log("Applying watershed segmentation", f"Min size: {min_size}")
        
        # Find local maxima as markers
        local_maxima = feature.peak_local_maxima(image, min_distance=10, threshold_abs=0.1)
        markers = np.zeros_like(image, dtype=bool)
        if len(local_maxima[0]) > 0:
            markers[local_maxima] = True
        
        # Create markers for watershed
        markers_labeled = measure.label(markers)
        
        # Apply watershed
        if markers_labeled.max() > 0:
            # Create elevation map (inverted image)
            elevation = -image
            segmented = segmentation.watershed(elevation, markers_labeled)
            
            # Convert to binary
            binary = segmented > 0
        else:
            # Fallback to threshold if no markers found
            binary = image > filters.threshold_otsu(image)
        
        # Clean up
        cleaned = morphology.remove_small_objects(binary, min_size=min_size)
        
        return cleaned
    
    def morphological_segmentation(self, image: np.ndarray, min_size: int = 15) -> np.ndarray:
        """Morphological segmentation."""
        self.log("Applying morphological segmentation", f"Min size: {min_size}")
        
        # Threshold
        binary = image > filters.threshold_otsu(image)
        
        # Morphological operations
        # Opening to remove noise
        opened = morphology.opening(binary, morphology.disk(2))
        
        # Closing to fill gaps
        closed = morphology.closing(opened, morphology.disk(3))
        
        # Remove small objects
        cleaned = morphology.remove_small_objects(closed, min_size=min_size)
        
        return cleaned
    
    def adaptive_morphological_segmentation(self, image: np.ndarray, min_size: int = 15) -> np.ndarray:
        """Adaptive morphological segmentation."""
        self.log("Applying adaptive morphological segmentation", f"Min size: {min_size}")
        
        # Use adaptive threshold
        threshold = filters.threshold_local(image, block_size=35)
        binary = image > threshold
        
        # Adaptive morphological operations based on image characteristics
        # Determine structure element size based on image content
        selem_size = max(1, min(image.shape) // 50)
        selem = morphology.disk(selem_size)
        
        # Opening and closing
        opened = morphology.opening(binary, selem)
        closed = morphology.closing(opened, morphology.disk(selem_size + 1))
        
        # Remove small objects
        cleaned = morphology.remove_small_objects(closed, min_size=min_size)
        
        return cleaned
    
    def run_all_segmentation_methods(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run all baseline segmentation methods on preprocessed image."""
        self.log("Running all segmentation methods", f"Image shape: {image.shape}")
        
        methods = {
            'threshold': self.threshold_based_segmentation,
            'adaptive_threshold': self.adaptive_threshold_segmentation,
            'edge_based': self.edge_based_segmentation,
            'watershed': self.watershed_segmentation,
            'morphological': self.morphological_segmentation,
            'adaptive_morphological': self.adaptive_morphological_segmentation
        }
        
        results = {}
        for method_name, method_func in methods.items():
            try:
                start_time = time.time()
                result = method_func(image)
                end_time = time.time()
                results[method_name] = result
                
                # Count objects
                labels = measure.label(result)
                object_count = labels.max()
                
                self.log(f"Segmentation method: {method_name}", 
                        f"Objects: {object_count}, Time: {end_time-start_time:.3f}s")
            except Exception as e:
                self.log(f"Error in {method_name}", str(e))
                results[method_name] = np.zeros_like(image, dtype=bool)
        
        return results
    
    def evaluate_segmentation(self, predicted_mask: np.ndarray, frame_num: int) -> Dict:
        """Evaluate segmentation against ground truth."""
        if not self.annotations or frame_num not in self.annotations:
            return None
        
        # Count predicted objects
        pred_labels = measure.label(predicted_mask)
        pred_count = pred_labels.max()
        
        # Count ground truth objects (ellipses only, not outside)
        gt_shapes = [s for s in self.annotations[frame_num] 
                    if s['shape_type'] == 'ellipse' and not s.get('outside', False)]
        gt_count = len(gt_shapes)
        
        # Calculate basic metrics
        error = abs(pred_count - gt_count)
        relative_error = error / gt_count if gt_count > 0 else float('inf')
        
        evaluation = {
            'predicted_count': pred_count,
            'ground_truth_count': gt_count,
            'absolute_error': error,
            'relative_error': relative_error,
            'accuracy': 1 - min(relative_error, 1.0)  # Capped at 0
        }
        
        return evaluation
    
    def comprehensive_analysis(self, frame_nums: List[int] = None) -> Dict:
        """Run comprehensive analysis on specified frames."""
        if frame_nums is None:
            frame_nums = [0, 5, 26]  # Default test frames
        
        self.log("Starting comprehensive analysis", f"Testing frames: {frame_nums}")
        
        results = {}
        
        for frame_num in frame_nums:
            frame = self.get_frame(frame_num)
            if frame is None:
                self.log(f"Frame {frame_num} not available", "Skipping")
                continue
            
            self.log(f"Analyzing frame {frame_num}", f"Frame shape: {frame.shape}")
            
            # Apply preprocessing
            preprocessed_images = self.combined_preprocessing(frame)
            
            frame_results = {}
            
            # Test each preprocessing method with all segmentation methods
            for prep_name, prep_image in preprocessed_images.items():
                self.log(f"Processing {prep_name} version", f"Frame {frame_num}")
                
                # Run all segmentation methods
                seg_results = self.run_all_segmentation_methods(prep_image)
                
                # Evaluate each segmentation result
                prep_results = {}
                for seg_name, seg_mask in seg_results.items():
                    evaluation = self.evaluate_segmentation(seg_mask, frame_num)
                    
                    prep_results[seg_name] = {
                        'mask': seg_mask,
                        'evaluation': evaluation
                    }
                
                frame_results[prep_name] = prep_results
            
            results[frame_num] = frame_results
            
            self.log(f"Frame {frame_num} analysis complete", 
                    f"Tested {len(preprocessed_images)} preprocessing × {len(seg_results)} segmentation methods")
        
        self.log("Comprehensive analysis complete", f"Analyzed {len(results)} frames")
        return results
    
    def find_best_combinations(self, results: Dict) -> Dict:
        """Find best preprocessing + segmentation combinations."""
        self.log("Finding best method combinations", "Analyzing all results")
        
        # Collect all accuracy scores
        combinations = []
        
        for frame_num, frame_results in results.items():
            for prep_name, prep_results in frame_results.items():
                for seg_name, seg_data in prep_results.items():
                    if seg_data['evaluation']:
                        combination = {
                            'frame': frame_num,
                            'preprocessing': prep_name,
                            'segmentation': seg_name,
                            'accuracy': seg_data['evaluation']['accuracy'],
                            'error': seg_data['evaluation']['absolute_error'],
                            'predicted': seg_data['evaluation']['predicted_count'],
                            'ground_truth': seg_data['evaluation']['ground_truth_count']
                        }
                        combinations.append(combination)
        
        # Sort by accuracy (descending)
        combinations.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Find best combination overall
        best_overall = combinations[0] if combinations else None
        
        # Find best combination per frame
        best_per_frame = {}
        for frame_num in results.keys():
            frame_combinations = [c for c in combinations if c['frame'] == frame_num]
            if frame_combinations:
                best_per_frame[frame_num] = frame_combinations[0]
        
        # Find best preprocessing method overall
        prep_scores = {}
        for combination in combinations:
            prep = combination['preprocessing']
            if prep not in prep_scores:
                prep_scores[prep] = []
            prep_scores[prep].append(combination['accuracy'])
        
        prep_averages = {prep: np.mean(scores) for prep, scores in prep_scores.items()}
        best_preprocessing = max(prep_averages, key=prep_averages.get) if prep_averages else None
        
        # Find best segmentation method overall
        seg_scores = {}
        for combination in combinations:
            seg = combination['segmentation']
            if seg not in seg_scores:
                seg_scores[seg] = []
            seg_scores[seg].append(combination['accuracy'])
        
        seg_averages = {seg: np.mean(scores) for seg, scores in seg_scores.items()}
        best_segmentation = max(seg_averages, key=seg_averages.get) if seg_averages else None
        
        best_combinations = {
            'best_overall': best_overall,
            'best_per_frame': best_per_frame,
            'best_preprocessing': best_preprocessing,
            'best_segmentation': best_segmentation,
            'preprocessing_scores': prep_averages,
            'segmentation_scores': seg_averages,
            'all_combinations': combinations[:10]  # Top 10
        }
        
        self.log("Best combinations identified", 
                f"Best overall: {best_preprocessing} + {best_segmentation}")
        
        return best_combinations
    
    def create_visualization(self, results: Dict, frame_num: int, save_path: str = None) -> plt.Figure:
        """Create comprehensive visualization for a frame."""
        if frame_num not in results:
            return None
        
        frame_results = results[frame_num]
        prep_methods = list(frame_results.keys())
        seg_methods = list(frame_results[prep_methods[0]].keys())
        
        # Create large subplot grid
        fig, axes = plt.subplots(len(prep_methods), len(seg_methods), 
                                figsize=(20, 15))
        
        if len(prep_methods) == 1:
            axes = axes.reshape(1, -1)
        if len(seg_methods) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, prep_name in enumerate(prep_methods):
            for j, seg_name in enumerate(seg_methods):
                ax = axes[i, j]
                
                seg_data = frame_results[prep_name][seg_name]
                mask = seg_data['mask']
                evaluation = seg_data['evaluation']
                
                # Show segmentation result
                labels = measure.label(mask)
                ax.imshow(labels, cmap='nipy_spectral')
                
                # Title with metrics
                if evaluation:
                    title = f"{prep_name} + {seg_name}\n"
                    title += f"Pred: {evaluation['predicted_count']}, "
                    title += f"GT: {evaluation['ground_truth_count']}, "
                    title += f"Acc: {evaluation['accuracy']:.3f}"
                else:
                    title = f"{prep_name} + {seg_name}\nNo GT available"
                
                ax.set_title(title, fontsize=8)
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log("Visualization saved", save_path)
        
        return fig
    
    def save_log(self, filename: str = "enhanced_segmentation_log.json"):
        """Save comprehensive log of all operations."""
        log_data = {
            "session_info": {
                "start_time": self.log_entries[0]["timestamp"] if self.log_entries else None,
                "end_time": self.log_entries[-1]["timestamp"] if self.log_entries else None,
                "tif_file": self.tif_path,
                "xml_file": self.xml_path,
                "tif_shape": list(self.tif_data.shape) if hasattr(self, 'tif_data') else None
            },
            "log_entries": self.log_entries
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.log("Log saved", filename)
        return filename

def main():
    """Run enhanced segmentation pipeline with comprehensive analysis."""
    # Use relative paths
    tif_file = "../annotated_data_1001/MattLines1.tif"
    xml_file = "../annotated_data_1001/MattLines1annotations.xml"
    
    print("=" * 80)
    print("ENHANCED SEGMENTATION PIPELINE WITH IMPROVED PREPROCESSING")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = EnhancedSegmentationPipeline(tif_file, xml_file)
    
    # Run comprehensive analysis
    print("\n" + "=" * 50)
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    results = pipeline.comprehensive_analysis([0, 5, 26])
    
    # Find best combinations
    print("\n" + "=" * 50)
    print("FINDING BEST METHOD COMBINATIONS")
    print("=" * 50)
    
    best_combinations = pipeline.find_best_combinations(results)
    
    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"Best Preprocessing Method: {best_combinations['best_preprocessing']}")
    print(f"Best Segmentation Method: {best_combinations['best_segmentation']}")
    
    if best_combinations['best_overall']:
        best = best_combinations['best_overall']
        print(f"Best Overall Combination:")
        print(f"  Frame {best['frame']}: {best['preprocessing']} + {best['segmentation']}")
        print(f"  Accuracy: {best['accuracy']:.3f}")
        print(f"  Predicted: {best['predicted']}, Ground Truth: {best['ground_truth']}")
    
    print("\nPreprocessing Method Scores:")
    for method, score in best_combinations['preprocessing_scores'].items():
        print(f"  {method}: {score:.3f}")
    
    print("\nSegmentation Method Scores:")
    for method, score in best_combinations['segmentation_scores'].items():
        print(f"  {method}: {score:.3f}")
    
    # Create visualizations for each test frame
    print("\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    for frame_num in [0, 5, 26]:
        if frame_num in results:
            fig = pipeline.create_visualization(
                results, frame_num, 
                save_path=f"enhanced_pipeline_frame_{frame_num}.png"
            )
            if fig:
                plt.close(fig)
    
    # Save comprehensive results
    results_filename = "enhanced_pipeline_results.json"
    with open(results_filename, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for frame_num, frame_results in results.items():
            json_results[str(frame_num)] = {}
            for prep_name, prep_results in frame_results.items():
                json_results[str(frame_num)][prep_name] = {}
                for seg_name, seg_data in prep_results.items():
                    json_results[str(frame_num)][prep_name][seg_name] = {
                        'evaluation': seg_data['evaluation']
                    }
        
        combined_data = {
            'results': json_results,
            'best_combinations': best_combinations
        }
        
        json.dump(combined_data, f, indent=2, default=str)
    
    pipeline.log("Results saved", results_filename)
    
    # Save log
    log_filename = pipeline.save_log("enhanced_segmentation_pipeline_log.json")
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {results_filename}")
    print(f"Log saved to: {log_filename}")
    print(f"Visualizations: enhanced_pipeline_frame_X.png")
    
    return results, best_combinations

if __name__ == "__main__":
    main()