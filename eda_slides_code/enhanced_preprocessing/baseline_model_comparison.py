#!/usr/bin/env python3
"""
Baseline Model Comparison: Original vs Enhanced Preprocessing
Compare baseline segmentation performance on original frames vs enhanced preprocessing with ground truth.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, measure
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

class BaselineModelComparison:
    """Compare baseline model performance on original vs enhanced preprocessing."""
    
    def __init__(self, tif_path: str, xml_path: str = None):
        """Initialize with TIF file and XML annotations."""
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
        
        # Load annotations
        self.annotations = None
        if xml_path and os.path.exists(xml_path):
            print(f"Loading annotations: {xml_path}")
            parser = XMLShapeParser(xml_path)
            self.annotations = parser.parse_all_shapes()
            frame_count = len(self.annotations)
            shape_count = sum(len(shapes) for shapes in self.annotations.values())
            print(f"Loaded annotations for {frame_count} frames, {shape_count} total shapes")
    
    def get_frame(self, frame_num: int) -> np.ndarray:
        """Get a specific frame from the TIF data."""
        if len(self.tif_data.shape) == 3:
            if frame_num < self.tif_data.shape[0]:
                return self.tif_data[frame_num]
        elif len(self.tif_data.shape) == 2 and frame_num == 0:
            return self.tif_data
        return None
    
    def get_ground_truth_count(self, frame_num: int) -> int:
        """Get ground truth cell count for a frame."""
        if not self.annotations or frame_num not in self.annotations:
            return 0
        
        # Count ellipses that are not marked as outside
        gt_shapes = [s for s in self.annotations[frame_num] 
                    if s['shape_type'] == 'ellipse' and not s.get('outside', False)]
        return len(gt_shapes)
    
    def create_ground_truth_mask(self, frame_num: int, frame_shape: tuple) -> np.ndarray:
        """Create ground truth mask from annotations."""
        if not self.annotations or frame_num not in self.annotations:
            return np.zeros(frame_shape, dtype=bool)
        
        mask = np.zeros(frame_shape, dtype=bool)
        
        # Draw ellipses from annotations
        for shape in self.annotations[frame_num]:
            if shape['shape_type'] == 'ellipse' and not shape.get('outside', False):
                # Get ellipse parameters from annotation structure
                center_x = shape.get('cx', 0)
                center_y = shape.get('cy', 0)
                rx = shape.get('rx', 5)  # Default radius if not found
                ry = shape.get('ry', 5)
                
                # Create elliptical mask
                y, x = np.ogrid[:frame_shape[0], :frame_shape[1]]
                ellipse_mask = ((x - center_x) / rx)**2 + ((y - center_y) / ry)**2 <= 1
                mask = mask | ellipse_mask
        
        return mask
    
    def adaptive_background_subtraction(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive background subtraction (best method identified)."""
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
        
        return corrected
    
    def baseline_segmentation(self, image: np.ndarray, min_size: int = 15) -> np.ndarray:
        """Apply baseline threshold segmentation (best segmentation method identified)."""
        # Otsu threshold
        threshold = filters.threshold_otsu(image)
        binary = image > threshold
        
        # Remove small objects
        cleaned = morphology.remove_small_objects(binary, min_size=min_size)
        
        # Fill holes
        filled = ndimage.binary_fill_holes(cleaned)
        
        return filled
    
    def calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between prediction and ground truth."""
        intersection = np.logical_and(pred_mask, gt_mask)
        union = np.logical_or(pred_mask, gt_mask)
        
        if np.sum(union) == 0:
            return 1.0 if np.sum(intersection) == 0 else 0.0
        
        return np.sum(intersection) / np.sum(union)
    
    def calculate_precision_recall(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple:
        """Calculate precision and recall."""
        true_positive = np.sum(np.logical_and(pred_mask, gt_mask))
        false_positive = np.sum(np.logical_and(pred_mask, ~gt_mask))
        false_negative = np.sum(np.logical_and(~pred_mask, gt_mask))
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        
        return precision, recall
    
    def analyze_frame_comparison(self, frame_num: int) -> Dict:
        """Comprehensive analysis comparing original vs enhanced preprocessing for a frame."""
        frame = self.get_frame(frame_num)
        if frame is None:
            return None
        
        print(f"\n=== Analyzing Frame {frame_num} ===")
        
        # Normalize original frame
        original = frame.astype(np.float64)
        original = (original - original.min()) / (original.max() - original.min())
        
        # Apply enhanced preprocessing
        enhanced = self.adaptive_background_subtraction(frame)
        
        # Apply baseline segmentation to both
        original_mask = self.baseline_segmentation(original)
        enhanced_mask = self.baseline_segmentation(enhanced)
        
        # Get ground truth
        gt_count = self.get_ground_truth_count(frame_num)
        gt_mask = self.create_ground_truth_mask(frame_num, frame.shape)
        
        # Count objects
        original_labels = measure.label(original_mask)
        enhanced_labels = measure.label(enhanced_mask)
        gt_labels = measure.label(gt_mask)
        
        original_count = original_labels.max()
        enhanced_count = enhanced_labels.max()
        gt_mask_count = gt_labels.max()
        
        # Calculate metrics
        original_iou = self.calculate_iou(original_mask, gt_mask)
        enhanced_iou = self.calculate_iou(enhanced_mask, gt_mask)
        
        original_precision, original_recall = self.calculate_precision_recall(original_mask, gt_mask)
        enhanced_precision, enhanced_recall = self.calculate_precision_recall(enhanced_mask, gt_mask)
        
        # Calculate F1 scores
        original_f1 = 2 * (original_precision * original_recall) / (original_precision + original_recall) if (original_precision + original_recall) > 0 else 0
        enhanced_f1 = 2 * (enhanced_precision * enhanced_recall) / (enhanced_precision + enhanced_recall) if (enhanced_precision + enhanced_recall) > 0 else 0
        
        # Calculate errors
        original_error = abs(original_count - gt_count)
        enhanced_error = abs(enhanced_count - gt_count)
        
        results = {
            'frame_num': frame_num,
            'original_frame': original,
            'enhanced_frame': enhanced,
            'original_mask': original_mask,
            'enhanced_mask': enhanced_mask,
            'gt_mask': gt_mask,
            'original_labels': original_labels,
            'enhanced_labels': enhanced_labels,
            'ground_truth': {
                'count': gt_count,
                'mask_count': gt_mask_count
            },
            'original_method': {
                'count': original_count,
                'error': original_error,
                'iou': original_iou,
                'precision': original_precision,
                'recall': original_recall,
                'f1_score': original_f1
            },
            'enhanced_method': {
                'count': enhanced_count,
                'error': enhanced_error,
                'iou': enhanced_iou,
                'precision': enhanced_precision,
                'recall': enhanced_recall,
                'f1_score': enhanced_f1
            },
            'improvement': {
                'count_error_reduction': original_error - enhanced_error,
                'iou_improvement': enhanced_iou - original_iou,
                'precision_improvement': enhanced_precision - original_precision,
                'recall_improvement': enhanced_recall - original_recall,
                'f1_improvement': enhanced_f1 - original_f1
            }
        }
        
        # Print summary
        print(f"Ground Truth: {gt_count} cells")
        print(f"Original Method: {original_count} objects (error: {original_error})")
        print(f"Enhanced Method: {enhanced_count} objects (error: {enhanced_error})")
        print(f"Error Reduction: {results['improvement']['count_error_reduction']}")
        print(f"IoU - Original: {original_iou:.3f}, Enhanced: {enhanced_iou:.3f}")
        print(f"F1 - Original: {original_f1:.3f}, Enhanced: {enhanced_f1:.3f}")
        
        return results
    
    def create_comprehensive_visualization(self, results: Dict, save_path: str = None) -> plt.Figure:
        """Create comprehensive visualization showing original vs enhanced vs ground truth."""
        frame_num = results['frame_num']
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Row 1: Original processing
        axes[0, 0].imshow(results['original_frame'], cmap='gray')
        axes[0, 0].set_title(f'Original Frame {frame_num}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(results['original_mask'], cmap='gray')
        axes[0, 1].set_title(f'Original Segmentation\n{results["original_method"]["count"]} objects')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(results['original_labels'], cmap='nipy_spectral')
        axes[0, 2].set_title(f'Original Labels\nError: {results["original_method"]["error"]}')
        axes[0, 2].axis('off')
        
        axes[0, 3].text(0.1, 0.9, f'Original Method Metrics:', fontsize=12, fontweight='bold', transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.8, f'Objects: {results["original_method"]["count"]}', fontsize=10, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.7, f'Error: {results["original_method"]["error"]}', fontsize=10, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.6, f'IoU: {results["original_method"]["iou"]:.3f}', fontsize=10, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.5, f'Precision: {results["original_method"]["precision"]:.3f}', fontsize=10, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.4, f'Recall: {results["original_method"]["recall"]:.3f}', fontsize=10, transform=axes[0, 3].transAxes)
        axes[0, 3].text(0.1, 0.3, f'F1-Score: {results["original_method"]["f1_score"]:.3f}', fontsize=10, transform=axes[0, 3].transAxes)
        axes[0, 3].axis('off')
        
        # Row 2: Enhanced processing
        axes[1, 0].imshow(results['enhanced_frame'], cmap='gray')
        axes[1, 0].set_title(f'Enhanced Frame {frame_num}\n(Adaptive Background Sub.)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(results['enhanced_mask'], cmap='gray')
        axes[1, 1].set_title(f'Enhanced Segmentation\n{results["enhanced_method"]["count"]} objects')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(results['enhanced_labels'], cmap='nipy_spectral')
        axes[1, 2].set_title(f'Enhanced Labels\nError: {results["enhanced_method"]["error"]}')
        axes[1, 2].axis('off')
        
        axes[1, 3].text(0.1, 0.9, f'Enhanced Method Metrics:', fontsize=12, fontweight='bold', transform=axes[1, 3].transAxes)
        axes[1, 3].text(0.1, 0.8, f'Objects: {results["enhanced_method"]["count"]}', fontsize=10, transform=axes[1, 3].transAxes)
        axes[1, 3].text(0.1, 0.7, f'Error: {results["enhanced_method"]["error"]}', fontsize=10, transform=axes[1, 3].transAxes)
        axes[1, 3].text(0.1, 0.6, f'IoU: {results["enhanced_method"]["iou"]:.3f}', fontsize=10, transform=axes[1, 3].transAxes)
        axes[1, 3].text(0.1, 0.5, f'Precision: {results["enhanced_method"]["precision"]:.3f}', fontsize=10, transform=axes[1, 3].transAxes)
        axes[1, 3].text(0.1, 0.4, f'Recall: {results["enhanced_method"]["recall"]:.3f}', fontsize=10, transform=axes[1, 3].transAxes)
        axes[1, 3].text(0.1, 0.3, f'F1-Score: {results["enhanced_method"]["f1_score"]:.3f}', fontsize=10, transform=axes[1, 3].transAxes)
        axes[1, 3].axis('off')
        
        # Row 3: Ground truth and comparison
        axes[2, 0].imshow(results['gt_mask'], cmap='gray')
        axes[2, 0].set_title(f'Ground Truth Mask\n{results["ground_truth"]["count"]} cells')
        axes[2, 0].axis('off')
        
        # Overlay comparison
        overlay = np.zeros((*results['original_frame'].shape, 3))
        overlay[results['gt_mask'], 0] = 1.0  # Ground truth in red
        overlay[results['enhanced_mask'], 1] = 1.0  # Enhanced prediction in green
        overlay[results['original_mask'], 2] = 0.5  # Original prediction in blue
        
        axes[2, 1].imshow(overlay)
        axes[2, 1].set_title('Overlay Comparison\nRed: GT, Green: Enhanced, Blue: Original')
        axes[2, 1].axis('off')
        
        # Side-by-side comparison
        comparison = np.hstack([
            results['original_labels'] / results['original_labels'].max() if results['original_labels'].max() > 0 else results['original_labels'],
            results['enhanced_labels'] / results['enhanced_labels'].max() if results['enhanced_labels'].max() > 0 else results['enhanced_labels']
        ])
        axes[2, 2].imshow(comparison, cmap='nipy_spectral')
        axes[2, 2].set_title('Side-by-Side\nLeft: Original | Right: Enhanced')
        axes[2, 2].axis('off')
        
        # Improvement metrics
        axes[2, 3].text(0.1, 0.9, f'Improvement Analysis:', fontsize=12, fontweight='bold', transform=axes[2, 3].transAxes)
        axes[2, 3].text(0.1, 0.8, f'Error Reduction: {results["improvement"]["count_error_reduction"]}', fontsize=10, transform=axes[2, 3].transAxes)
        axes[2, 3].text(0.1, 0.7, f'IoU Improvement: {results["improvement"]["iou_improvement"]:.3f}', fontsize=10, transform=axes[2, 3].transAxes)
        axes[2, 3].text(0.1, 0.6, f'Precision Gain: {results["improvement"]["precision_improvement"]:.3f}', fontsize=10, transform=axes[2, 3].transAxes)
        axes[2, 3].text(0.1, 0.5, f'Recall Gain: {results["improvement"]["recall_improvement"]:.3f}', fontsize=10, transform=axes[2, 3].transAxes)
        axes[2, 3].text(0.1, 0.4, f'F1 Improvement: {results["improvement"]["f1_improvement"]:.3f}', fontsize=10, transform=axes[2, 3].transAxes)
        
        # Color code improvements
        if results["improvement"]["count_error_reduction"] > 0:
            axes[2, 3].text(0.1, 0.2, '✅ Better object count', fontsize=10, color='green', transform=axes[2, 3].transAxes)
        elif results["improvement"]["count_error_reduction"] < 0:
            axes[2, 3].text(0.1, 0.2, '❌ Worse object count', fontsize=10, color='red', transform=axes[2, 3].transAxes)
        else:
            axes[2, 3].text(0.1, 0.2, '➖ Same object count', fontsize=10, color='orange', transform=axes[2, 3].transAxes)
        
        axes[2, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comprehensive comparison saved: {save_path}")
        
        return fig
    
    def run_comprehensive_comparison(self, frame_nums: List[int] = None) -> Dict:
        """Run comprehensive comparison across multiple frames."""
        if frame_nums is None:
            frame_nums = [0, 5, 26]  # Focus on problematic and test frames
        
        print("=" * 80)
        print("BASELINE MODEL COMPARISON: ORIGINAL vs ENHANCED PREPROCESSING")
        print("=" * 80)
        
        all_results = {}
        summary_stats = {
            'total_frames': 0,
            'original_total_error': 0,
            'enhanced_total_error': 0,
            'original_avg_iou': 0,
            'enhanced_avg_iou': 0,
            'original_avg_f1': 0,
            'enhanced_avg_f1': 0,
            'frames_improved': 0
        }
        
        for frame_num in frame_nums:
            if frame_num >= self.tif_data.shape[0]:
                print(f"Frame {frame_num} not available (max: {self.tif_data.shape[0]-1})")
                continue
            
            # Analyze frame
            results = self.analyze_frame_comparison(frame_num)
            if results:
                all_results[frame_num] = results
                
                # Update summary stats
                summary_stats['total_frames'] += 1
                summary_stats['original_total_error'] += results['original_method']['error']
                summary_stats['enhanced_total_error'] += results['enhanced_method']['error']
                summary_stats['original_avg_iou'] += results['original_method']['iou']
                summary_stats['enhanced_avg_iou'] += results['enhanced_method']['iou']
                summary_stats['original_avg_f1'] += results['original_method']['f1_score']
                summary_stats['enhanced_avg_f1'] += results['enhanced_method']['f1_score']
                
                if results['improvement']['count_error_reduction'] > 0:
                    summary_stats['frames_improved'] += 1
                
                # Create visualization
                fig = self.create_comprehensive_visualization(
                    results, 
                    save_path=f"baseline_comparison_frame_{frame_num}.png"
                )
                if fig:
                    plt.show()
                    plt.close(fig)
        
        # Calculate averages
        if summary_stats['total_frames'] > 0:
            summary_stats['original_avg_iou'] /= summary_stats['total_frames']
            summary_stats['enhanced_avg_iou'] /= summary_stats['total_frames']
            summary_stats['original_avg_f1'] /= summary_stats['total_frames']
            summary_stats['enhanced_avg_f1'] /= summary_stats['total_frames']
        
        # Print final summary
        print("\n" + "=" * 80)
        print("FINAL COMPARISON SUMMARY")
        print("=" * 80)
        
        print(f"Frames Analyzed: {summary_stats['total_frames']}")
        print(f"Frames Improved: {summary_stats['frames_improved']} ({summary_stats['frames_improved']/summary_stats['total_frames']*100:.1f}%)")
        
        print(f"\nTotal Count Errors:")
        print(f"  Original Method: {summary_stats['original_total_error']}")
        print(f"  Enhanced Method: {summary_stats['enhanced_total_error']}")
        print(f"  Total Error Reduction: {summary_stats['original_total_error'] - summary_stats['enhanced_total_error']}")
        
        print(f"\nAverage IoU Scores:")
        print(f"  Original Method: {summary_stats['original_avg_iou']:.3f}")
        print(f"  Enhanced Method: {summary_stats['enhanced_avg_iou']:.3f}")
        print(f"  IoU Improvement: {summary_stats['enhanced_avg_iou'] - summary_stats['original_avg_iou']:.3f}")
        
        print(f"\nAverage F1 Scores:")
        print(f"  Original Method: {summary_stats['original_avg_f1']:.3f}")
        print(f"  Enhanced Method: {summary_stats['enhanced_avg_f1']:.3f}")
        print(f"  F1 Improvement: {summary_stats['enhanced_avg_f1'] - summary_stats['original_avg_f1']:.3f}")
        
        # Save comprehensive results
        results_data = {
            'summary_statistics': summary_stats,
            'frame_results': {}
        }
        
        for frame_num, frame_results in all_results.items():
            # Convert numpy arrays to lists for JSON serialization
            results_data['frame_results'][str(frame_num)] = {
                'ground_truth': frame_results['ground_truth'],
                'original_method': frame_results['original_method'],
                'enhanced_method': frame_results['enhanced_method'],
                'improvement': frame_results['improvement']
            }
        
        with open('baseline_comparison_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\nResults saved to: baseline_comparison_results.json")
        print(f"Visualizations saved as: baseline_comparison_frame_X.png")
        
        return all_results, summary_stats

def main():
    """Run baseline model comparison between original and enhanced preprocessing."""
    # Use relative paths
    tif_file = "../annotated_data_1001/MattLines1.tif"
    xml_file = "../annotated_data_1001/MattLines1annotations.xml"
    
    # Initialize comparison
    comparator = BaselineModelComparison(tif_file, xml_file)
    
    # Run comprehensive comparison on problematic frames and test frame
    test_frames = [0, 5, 26]
    
    results, summary = comparator.run_comprehensive_comparison(test_frames)
    
    print("\n" + "=" * 80)
    print("BASELINE MODEL COMPARISON COMPLETE")
    print("=" * 80)
    
    return results, summary

if __name__ == "__main__":
    main()