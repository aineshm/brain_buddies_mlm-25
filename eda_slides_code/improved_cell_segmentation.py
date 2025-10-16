#!/usr/bin/env python3
"""
Improved Cell Segmentation Models
Enhanced segmentation techniques with parameter tuning and advanced methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import filters, morphology, measure, segmentation
from skimage.segmentation import watershed, chan_vese, morphological_chan_vese
from scipy import ndimage
import tifffile
from xml_shape_parser import XMLShapeParser
import os
from typing import Dict, List, Tuple, Any
import json

class ImprovedCellSegmentation:
    """Improved segmentation models with better parameter tuning."""
    
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
    
    def enhanced_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing pipeline."""
        # Convert to float
        img = image.astype(np.float64)
        img = (img - img.min()) / (img.max() - img.min())
        
        # Apply bilateral filter to preserve edges while reducing noise
        img_uint8 = (img * 255).astype(np.uint8)
        bilateral = cv2.bilateralFilter(img_uint8, 9, 75, 75)
        
        # Convert back and apply CLAHE
        img_filtered = bilateral.astype(np.float64) / 255.0
        img_uint8_filtered = (img_filtered * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_uint8_filtered)
        
        return img_enhanced.astype(np.float64) / 255.0
    
    def multi_scale_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Multi-scale approach combining different techniques."""
        # Enhanced preprocessing
        img_proc = self.enhanced_preprocessing(image)
        
        # Multi-threshold approach
        thresh_otsu = filters.threshold_otsu(img_proc)
        thresh_local = filters.threshold_local(img_proc, block_size=25, method='gaussian')
        
        # Combine thresholds
        binary1 = img_proc > thresh_otsu
        binary2 = img_proc > thresh_local
        combined_binary = binary1 | binary2
        
        # Morphological operations
        selem = morphology.disk(1)
        cleaned = morphology.opening(combined_binary, selem)
        cleaned = morphology.remove_small_objects(cleaned, min_size=10)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=25)
        
        return cleaned
    
    def advanced_watershed(self, image: np.ndarray) -> np.ndarray:
        """Advanced watershed with better seed detection."""
        # Enhanced preprocessing
        img_proc = self.enhanced_preprocessing(image)
        
        # Initial binary mask
        thresh = filters.threshold_otsu(img_proc)
        binary = img_proc > thresh
        binary = morphology.remove_small_objects(binary, min_size=15)
        
        # Distance transform
        distance = ndimage.distance_transform_edt(binary)
        
        # Better seed detection using morphological operations
        # Create a more restrictive mask for seeds
        seed_threshold = 0.6 * distance.max()
        seeds = distance > seed_threshold
        
        # Apply morphological operations to get better seeds
        selem = morphology.disk(2)
        seeds = morphology.opening(seeds, selem)
        seeds = morphology.remove_small_objects(seeds, min_size=3)
        
        # Label seeds
        markers = measure.label(seeds)
        
        # Apply watershed
        labels = watershed(-distance, markers, mask=binary)
        
        return labels > 0
    
    def contour_refinement_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Contour-based segmentation with refinement."""
        # Enhanced preprocessing
        img_proc = self.enhanced_preprocessing(image)
        img_uint8 = (img_proc * 255).astype(np.uint8)
        
        # Multiple threshold levels
        thresholds = [
            cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0],
            filters.threshold_otsu(img_proc) * 255,
            np.percentile(img_uint8, 85)
        ]
        
        all_contours = []
        
        for thresh_val in thresholds:
            _, binary = cv2.threshold(img_uint8, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and shape
            for contour in contours:
                area = cv2.contourArea(contour)
                if 15 < area < 500:  # Reasonable cell sizes
                    # Check if contour is roughly circular (for cells)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.3:  # Not too elongated
                            all_contours.append(contour)
        
        # Create mask from filtered contours
        mask = np.zeros_like(img_uint8)
        cv2.drawContours(mask, all_contours, -1, 255, -1)
        
        return mask > 0
    
    def blob_detection_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Blob detection for circular cell-like objects."""
        # Enhanced preprocessing
        img_proc = self.enhanced_preprocessing(image)
        img_uint8 = (img_proc * 255).astype(np.uint8)
        
        # Set up SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by Area
        params.filterByArea = True
        params.minArea = 15
        params.maxArea = 500
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.3
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.3
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(255 - img_uint8)  # Invert for dark blobs
        
        # Create mask from detected blobs
        mask = np.zeros_like(img_uint8, dtype=bool)
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            cv2.circle(mask.astype(np.uint8), (x, y), radius, 1, -1)
        
        return mask
    
    def adaptive_morphological_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Adaptive morphological segmentation."""
        # Enhanced preprocessing
        img_proc = self.enhanced_preprocessing(image)
        
        # Adaptive threshold with different block sizes
        img_uint8 = (img_proc * 255).astype(np.uint8)
        
        # Try different block sizes and combine results
        masks = []
        for block_size in [15, 25, 35]:
            thresh_local = cv2.adaptiveThreshold(
                img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, block_size, 2
            )
            masks.append(thresh_local > 0)
        
        # Combine masks using majority voting
        combined = sum(masks) >= 2  # At least 2 out of 3 agree
        
        # Morphological cleaning with adaptive structuring elements
        # Use smaller elements for detailed areas, larger for smooth areas
        texture = ndimage.gaussian_filter(np.gradient(img_proc)[0]**2 + np.gradient(img_proc)[1]**2, sigma=1)
        
        # Adaptive morphological operations
        cleaned = combined.copy()
        
        # Small structuring element for detailed areas
        small_selem = morphology.disk(1)
        cleaned = morphology.opening(cleaned, small_selem)
        
        # Medium structuring element for general cleanup
        medium_selem = morphology.disk(2)
        cleaned = morphology.closing(cleaned, medium_selem)
        
        # Remove small objects and holes
        cleaned = morphology.remove_small_objects(cleaned, min_size=12)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=30)
        
        return cleaned
    
    def evaluate_segmentation(self, predicted_mask: np.ndarray, frame_num: int) -> Dict:
        """Evaluate segmentation against ground truth annotations."""
        if self.annotations is None or frame_num not in self.annotations:
            return {"error": "No ground truth available"}
        
        # Create ground truth mask from ellipse annotations
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
                ellipse_mask = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1
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
            'count_accuracy': 1 - abs(cell_count_pred - cell_count_gt) / max(cell_count_gt, 1)
        }
    
    def compare_improved_methods(self, frame_num: int, save_results: bool = True) -> Dict:
        """Compare improved segmentation methods."""
        frame = self.get_frame(frame_num)
        if frame is None:
            print(f"Frame {frame_num} not available")
            return {}
        
        print(f"Comparing improved segmentation methods on frame {frame_num}")
        
        methods = {
            'multi_scale': self.multi_scale_segmentation,
            'advanced_watershed': self.advanced_watershed,
            'contour_refinement': self.contour_refinement_segmentation,
            'blob_detection': self.blob_detection_segmentation,
            'adaptive_morphological': self.adaptive_morphological_segmentation
        }
        
        results = {}
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        # Original image
        axes[0].imshow(frame, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Apply each method
        for i, (method_name, method_func) in enumerate(methods.items(), 1):
            try:
                mask = method_func(frame)
                
                # Visualize
                axes[i].imshow(frame, cmap='gray', alpha=0.7)
                axes[i].contour(mask, colors='red', linewidths=1.5)
                axes[i].set_title(f'{method_name.replace("_", " ").title()}')
                axes[i].axis('off')
                
                # Extract properties
                labeled_mask = measure.label(mask)
                cell_count = labeled_mask.max()
                
                # Evaluate
                evaluation = self.evaluate_segmentation(mask, frame_num)
                
                results[method_name] = {
                    'mask': mask,
                    'cell_count': cell_count,
                    'evaluation': evaluation
                }
                
                # Add count and F1 score to title
                f1_score = evaluation.get('f1_score', 0)
                axes[i].set_title(f'{method_name.replace("_", " ").title()}\n{cell_count} cells, F1: {f1_score:.3f}')
                
                print(f"{method_name}: {cell_count} cells detected, F1: {f1_score:.3f}")
                
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                results[method_name] = {'error': str(e)}
        
        plt.tight_layout()
        
        if save_results:
            output_file = f"improved_segmentation_frame_{frame_num}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to {output_file}")
        
        plt.show()
        
        return results
    
    def find_best_method(self, test_frames: List[int] = None) -> Dict:
        """Find the best performing method across multiple frames."""
        if test_frames is None:
            test_frames = [0, 5, 26] if self.annotations else [0]
        
        methods = ['multi_scale', 'advanced_watershed', 'contour_refinement', 
                  'blob_detection', 'adaptive_morphological']
        
        method_scores = {method: [] for method in methods}
        
        for frame_num in test_frames:
            frame = self.get_frame(frame_num)
            if frame is None:
                continue
            
            print(f"Evaluating frame {frame_num}...")
            
            for method in methods:
                try:
                    method_func = getattr(self, f"{method}_segmentation")
                    mask = method_func(frame)
                    evaluation = self.evaluate_segmentation(mask, frame_num)
                    
                    if 'f1_score' in evaluation:
                        method_scores[method].append(evaluation['f1_score'])
                
                except Exception as e:
                    print(f"Error with {method} on frame {frame_num}: {e}")
        
        # Calculate average scores
        avg_scores = {}
        for method, scores in method_scores.items():
            avg_scores[method] = np.mean(scores) if scores else 0
        
        # Find best method
        best_method = max(avg_scores, key=avg_scores.get)
        
        return {
            'best_method': best_method,
            'average_scores': avg_scores,
            'detailed_scores': method_scores
        }

def main():
    """Demonstrate improved segmentation methods."""
    tif_file = "annotated_data_1001/MattLines1.tif"
    xml_file = "annotated_data_1001/MattLines1annotations.xml"
    
    print("=== Improved Cell Segmentation Models ===")
    
    # Initialize improved segmentation
    segmenter = ImprovedCellSegmentation(tif_file, xml_file)
    
    # Test on representative frames
    test_frames = [0, 5, 26]
    
    all_results = {}
    
    for frame_num in test_frames:
        print(f"\n--- Frame {frame_num} Analysis ---")
        frame_results = segmenter.compare_improved_methods(frame_num)
        all_results[frame_num] = frame_results
    
    # Find best method
    print(f"\n--- Finding Best Method ---")
    best_method_results = segmenter.find_best_method(test_frames)
    
    print(f"Best method: {best_method_results['best_method']}")
    print("Average F1 scores:")
    for method, score in best_method_results['average_scores'].items():
        print(f"  {method}: {score:.3f}")
    
    # Save results
    output_data = {
        'frame_results': all_results,
        'best_method_analysis': best_method_results,
        'timestamp': '2025-10-01'
    }
    
    with open('improved_segmentation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    
    print(f"\n=== Analysis Complete ===")
    print("Generated files:")
    print("- improved_segmentation_frame_X.png: Visual comparisons")
    print("- improved_segmentation_results.json: Quantitative results")

if __name__ == "__main__":
    main()