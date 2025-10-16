#!/usr/bin/env python3
"""
Cell Segmentation Baseline Models
Implements various segmentation techniques for cell identification in microscopy images.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import filters, morphology, measure, segmentation
from skimage.segmentation import watershed
from scipy import ndimage
import tifffile
from xml_shape_parser import XMLShapeParser
import os
from typing import Dict, List, Tuple, Any
import json

class CellSegmentationBaseline:
    """Baseline segmentation models for cell identification."""
    
    def __init__(self, tif_path: str, xml_path: str = None):
        """Initialize with TIF file and optional XML annotations for evaluation."""
        self.tif_path = tif_path
        self.xml_path = xml_path
        
        # Load TIF data
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
            print(f"Found annotations for {len(self.annotations)} frames")
    
    def get_frame(self, frame_num: int) -> np.ndarray:
        """Get a specific frame from the TIF data."""
        if len(self.tif_data.shape) == 3:
            if frame_num < self.tif_data.shape[0]:
                return self.tif_data[frame_num]
        elif len(self.tif_data.shape) == 2 and frame_num == 0:
            return self.tif_data
        return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to enhance cell visibility."""
        # Convert to float and normalize
        img = image.astype(np.float64)
        img = (img - img.min()) / (img.max() - img.min())
        
        # Apply Gaussian blur to reduce noise
        img_smooth = filters.gaussian(img, sigma=1.0)
        
        # Enhance contrast using CLAHE
        img_uint8 = (img_smooth * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_uint8)
        
        return img_enhanced.astype(np.float64) / 255.0
    
    def threshold_based_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Simple threshold-based segmentation."""
        # Preprocess
        img_proc = self.preprocess_image(image)
        
        # Apply Otsu's threshold
        threshold = filters.threshold_otsu(img_proc)
        binary = img_proc > threshold
        
        # Morphological operations to clean up
        binary = morphology.remove_small_objects(binary, min_size=20)
        binary = morphology.remove_small_holes(binary, area_threshold=50)
        
        # Apply opening to separate connected objects
        selem = morphology.disk(2)
        binary = morphology.opening(binary, selem)
        
        return binary
    
    def adaptive_threshold_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Adaptive threshold segmentation for varying illumination."""
        # Preprocess
        img_proc = self.preprocess_image(image)
        img_uint8 = (img_proc * 255).astype(np.uint8)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        binary = binary > 0
        
        # Morphological cleanup
        binary = morphology.remove_small_objects(binary, min_size=15)
        binary = morphology.remove_small_holes(binary, area_threshold=30)
        
        return binary
    
    def watershed_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Watershed segmentation to separate touching cells."""
        # Preprocess
        img_proc = self.preprocess_image(image)
        
        # Get initial binary mask
        threshold = filters.threshold_otsu(img_proc)
        binary = img_proc > threshold
        binary = morphology.remove_small_objects(binary, min_size=20)
        
        # Distance transform
        distance = ndimage.distance_transform_edt(binary)
        
        # Find local maxima as markers using a simple approach
        # Apply maximum filter and compare with original
        max_filtered = ndimage.maximum_filter(distance, size=5)
        local_maxima = (distance == max_filtered) & (distance > 0.3 * distance.max())
        markers = measure.label(local_maxima)
        
        # Apply watershed
        labels = watershed(-distance, markers, mask=binary)
        
        return labels, binary
    
    def edge_based_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Edge-based segmentation using Canny edge detection."""
        # Preprocess
        img_proc = self.preprocess_image(image)
        img_uint8 = (img_proc * 255).astype(np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(img_uint8, 50, 150)
        
        # Close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Fill enclosed regions
        binary = ndimage.binary_fill_holes(edges_closed)
        
        # Clean up small objects
        binary = morphology.remove_small_objects(binary, min_size=25)
        
        return binary
    
    def contour_based_segmentation(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Contour-based segmentation to find cell boundaries."""
        # Preprocess
        img_proc = self.preprocess_image(image)
        img_uint8 = (img_proc * 255).astype(np.uint8)
        
        # Apply threshold
        _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area, max_area = 20, 1000
        filtered_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        
        # Create mask from filtered contours
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, filtered_contours, -1, 255, -1)
        
        return filtered_contours, mask > 0
    
    def region_growing_segmentation(self, image: np.ndarray, seed_threshold: float = 0.7) -> np.ndarray:
        """Region growing segmentation starting from bright seed points."""
        # Preprocess
        img_proc = self.preprocess_image(image)
        
        # Find seed points (bright spots)
        seeds = img_proc > (seed_threshold * img_proc.max())
        seeds = morphology.remove_small_objects(seeds, min_size=3)
        
        # Label seeds
        seed_labels = measure.label(seeds)
        
        # Region growing using watershed with seeds
        binary_rough = img_proc > filters.threshold_otsu(img_proc)
        labels = watershed(-img_proc, seed_labels, mask=binary_rough)
        
        return labels > 0
    
    def extract_cell_properties(self, labeled_image: np.ndarray, original_image: np.ndarray) -> List[Dict]:
        """Extract properties of segmented cells."""
        properties = []
        
        regions = measure.regionprops(labeled_image, intensity_image=original_image)
        
        for region in regions:
            props = {
                'area': region.area,
                'centroid': region.centroid,
                'bbox': region.bbox,  # (min_row, min_col, max_row, max_col)
                'eccentricity': region.eccentricity,
                'major_axis_length': region.major_axis_length,
                'minor_axis_length': region.minor_axis_length,
                'mean_intensity': region.mean_intensity,
                'max_intensity': region.max_intensity,
                'perimeter': region.perimeter,
                'solidity': region.solidity,
                'extent': region.extent
            }
            properties.append(props)
        
        return properties
    
    def evaluate_segmentation(self, predicted_mask: np.ndarray, frame_num: int) -> Dict:
        """Evaluate segmentation against ground truth annotations if available."""
        if self.annotations is None or frame_num not in self.annotations:
            return {"error": "No ground truth available"}
        
        # Create ground truth mask from annotations
        frame_shape = predicted_mask.shape
        gt_mask = np.zeros(frame_shape, dtype=bool)
        
        annotations = self.annotations[frame_num]
        cell_count_gt = 0
        
        for shape in annotations:
            if shape.get('outside', False):
                continue
                
            if shape['shape_type'] == 'ellipse':
                # Create ellipse mask
                cy, cx = int(shape['cy']), int(shape['cx'])
                ry, rx = int(shape['ry']), int(shape['rx'])
                
                yy, xx = np.ogrid[:frame_shape[0], :frame_shape[1]]
                ellipse_mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
                gt_mask |= ellipse_mask
                cell_count_gt += 1
        
        # Calculate metrics
        intersection = np.logical_and(predicted_mask, gt_mask)
        union = np.logical_or(predicted_mask, gt_mask)
        
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        precision = np.sum(intersection) / np.sum(predicted_mask) if np.sum(predicted_mask) > 0 else 0
        recall = np.sum(intersection) / np.sum(gt_mask) if np.sum(gt_mask) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Count detected objects
        predicted_labels = measure.label(predicted_mask)
        cell_count_pred = predicted_labels.max()
        
        return {
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'ground_truth_cells': cell_count_gt,
            'predicted_cells': cell_count_pred,
            'count_accuracy': abs(cell_count_pred - cell_count_gt) / max(cell_count_gt, 1)
        }
    
    def compare_methods(self, frame_num: int, save_results: bool = True) -> Dict:
        """Compare all segmentation methods on a single frame."""
        frame = self.get_frame(frame_num)
        if frame is None:
            print(f"Frame {frame_num} not available")
            return {}
        
        print(f"Comparing segmentation methods on frame {frame_num}")
        
        methods = {
            'threshold': self.threshold_based_segmentation,
            'adaptive_threshold': self.adaptive_threshold_segmentation,
            'edge_based': self.edge_based_segmentation,
            'region_growing': self.region_growing_segmentation
        }
        
        results = {}
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Original image
        axes[0].imshow(frame, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Apply each method
        for i, (method_name, method_func) in enumerate(methods.items(), 1):
            try:
                if method_name == 'watershed':
                    labels, binary = method_func(frame)
                    mask = binary
                else:
                    mask = method_func(frame)
                
                # Visualize
                axes[i].imshow(frame, cmap='gray', alpha=0.7)
                axes[i].contour(mask, colors='red', linewidths=1)
                axes[i].set_title(f'{method_name.replace("_", " ").title()}')
                axes[i].axis('off')
                
                # Extract properties
                labeled_mask = measure.label(mask)
                cell_props = self.extract_cell_properties(labeled_mask, frame)
                
                # Evaluate if ground truth available
                evaluation = self.evaluate_segmentation(mask, frame_num)
                
                results[method_name] = {
                    'mask': mask,
                    'cell_count': len(cell_props),
                    'cell_properties': cell_props,
                    'evaluation': evaluation
                }
                
                print(f"{method_name}: {len(cell_props)} cells detected")
                if 'f1_score' in evaluation:
                    print(f"  F1 Score: {evaluation['f1_score']:.3f}")
                
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                results[method_name] = {'error': str(e)}
        
        # Add ground truth if available
        if self.annotations and frame_num in self.annotations:
            gt_mask = np.zeros(frame.shape, dtype=bool)
            annotations = self.annotations[frame_num]
            
            for shape in annotations:
                if shape.get('outside', False):
                    continue
                if shape['shape_type'] == 'ellipse':
                    cy, cx = int(shape['cy']), int(shape['cx'])
                    ry, rx = int(shape['ry']), int(shape['rx'])
                    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
                    ellipse_mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
                    gt_mask |= ellipse_mask
            
            axes[5].imshow(frame, cmap='gray', alpha=0.7)
            axes[5].contour(gt_mask, colors='green', linewidths=1)
            axes[5].set_title('Ground Truth')
            axes[5].axis('off')
        
        plt.tight_layout()
        
        if save_results:
            output_file = f"segmentation_comparison_frame_{frame_num}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to {output_file}")
        
        plt.show()
        
        return results
    
    def batch_evaluate(self, frame_range: Tuple[int, int] = None, method: str = 'threshold') -> Dict:
        """Evaluate a segmentation method across multiple frames."""
        if self.annotations is None:
            print("No annotations available for evaluation")
            return {}
        
        available_frames = sorted(self.annotations.keys())
        if frame_range:
            available_frames = [f for f in available_frames if frame_range[0] <= f <= frame_range[1]]
        
        method_func = getattr(self, f"{method}_based_segmentation")
        
        results = []
        print(f"Evaluating {method} method on {len(available_frames)} frames...")
        
        for frame_num in available_frames:
            frame = self.get_frame(frame_num)
            if frame is None:
                continue
            
            try:
                mask = method_func(frame)
                evaluation = self.evaluate_segmentation(mask, frame_num)
                evaluation['frame'] = frame_num
                results.append(evaluation)
                
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
        
        # Calculate average metrics
        if results:
            avg_metrics = {}
            for metric in ['iou', 'precision', 'recall', 'f1_score', 'count_accuracy']:
                values = [r[metric] for r in results if metric in r and not np.isnan(r[metric])]
                avg_metrics[f'avg_{metric}'] = np.mean(values) if values else 0
            
            return {
                'method': method,
                'frame_results': results,
                'average_metrics': avg_metrics,
                'total_frames': len(results)
            }
        
        return {}

def main():
    """Demonstrate cell segmentation baseline methods."""
    tif_file = "annotated_data_1001/MattLines1.tif"
    xml_file = "annotated_data_1001/MattLines1annotations.xml"
    
    print("=== Cell Segmentation Baseline Models ===")
    
    # Initialize segmentation pipeline
    segmenter = CellSegmentationBaseline(tif_file, xml_file)
    
    # Compare methods on a few representative frames
    test_frames = [0, 5, 26]  # Simple, medium, complex
    
    all_results = {}
    
    for frame_num in test_frames:
        print(f"\n--- Frame {frame_num} Analysis ---")
        frame_results = segmenter.compare_methods(frame_num)
        all_results[frame_num] = frame_results
    
    # Batch evaluation of best method
    print(f"\n--- Batch Evaluation ---")
    batch_results = segmenter.batch_evaluate(frame_range=(0, 10), method='threshold')
    
    if batch_results:
        print(f"Threshold method results across {batch_results['total_frames']} frames:")
        for metric, value in batch_results['average_metrics'].items():
            print(f"  {metric}: {value:.3f}")
    
    # Save comprehensive results
    output_data = {
        'frame_comparisons': all_results,
        'batch_evaluation': batch_results,
        'timestamp': '2025-10-01'
    }
    
    with open('segmentation_baseline_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        import json
        json.dump(output_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    print(f"\n=== Analysis Complete ===")
    print("Generated files:")
    print("- segmentation_comparison_frame_X.png: Visual comparisons")
    print("- segmentation_baseline_results.json: Quantitative results")

if __name__ == "__main__":
    main()