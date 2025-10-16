#!/usr/bin/env python3
"""
Final Cell Segmentation Pipeline
Best-performing segmentation pipeline with comprehensive evaluation and easy usage.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import filters, morphology, measure, segmentation
from scipy import ndimage
import tifffile
from xml_shape_parser import XMLShapeParser
from improved_cell_segmentation import ImprovedCellSegmentation
import os
import json
from typing import Dict, List, Tuple, Any

class FinalCellSegmentationPipeline:
    """Production-ready cell segmentation pipeline."""
    
    def __init__(self, tif_path: str, xml_path: str = None):
        """Initialize the segmentation pipeline."""
        # Inherit from improved segmentation
        self.improved_segmenter = ImprovedCellSegmentation(tif_path, xml_path)
        self.tif_path = tif_path
        self.xml_path = xml_path
        self.tif_data = self.improved_segmenter.tif_data
        self.annotations = self.improved_segmenter.annotations
        
    def optimized_segmentation(self, image: np.ndarray, method: str = 'best') -> np.ndarray:
        """
        Optimized segmentation using the best-performing method.
        
        Args:
            image: Input grayscale image
            method: Segmentation method ('best', 'adaptive', 'edge', 'hybrid')
        
        Returns:
            Binary mask of segmented cells
        """
        if method == 'best' or method == 'adaptive':
            return self.improved_segmenter.adaptive_morphological_segmentation(image)
        elif method == 'edge':
            return self._enhanced_edge_segmentation(image)
        elif method == 'hybrid':
            return self._hybrid_segmentation(image)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _enhanced_edge_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Enhanced edge-based segmentation (performed well in baseline)."""
        # Enhanced preprocessing
        img_proc = self.improved_segmenter.enhanced_preprocessing(image)
        img_uint8 = (img_proc * 255).astype(np.uint8)
        
        # Multiple edge detection approaches
        # Canny with different parameters
        edges1 = cv2.Canny(img_uint8, 30, 100)
        edges2 = cv2.Canny(img_uint8, 50, 150)
        edges3 = cv2.Canny(img_uint8, 70, 200)
        
        # Combine edges
        combined_edges = edges1 | edges2 | edges3
        
        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Fill holes
        filled = ndimage.binary_fill_holes(closed)
        
        # Clean up
        cleaned = morphology.remove_small_objects(filled, min_size=20)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=50)
        
        return cleaned
    
    def _hybrid_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Hybrid approach combining multiple methods."""
        # Get results from different methods
        adaptive_result = self.improved_segmenter.adaptive_morphological_segmentation(image)
        edge_result = self._enhanced_edge_segmentation(image)
        
        # Simple voting: if both methods agree, include the pixel
        consensus = adaptive_result & edge_result
        
        # Also include pixels where adaptive method has high confidence
        # (areas with good local contrast)
        img_proc = self.improved_segmenter.enhanced_preprocessing(image)
        local_std = ndimage.generic_filter(img_proc, np.std, size=5)
        high_contrast = local_std > np.percentile(local_std, 70)
        
        # Combine consensus with high-confidence adaptive regions
        result = consensus | (adaptive_result & high_contrast)
        
        # Final cleanup
        result = morphology.remove_small_objects(result, min_size=15)
        result = morphology.remove_small_holes(result, area_threshold=30)
        
        return result
    
    def segment_frame(self, frame_num: int, method: str = 'best', visualize: bool = False) -> Dict:
        """
        Segment a specific frame and return results.
        
        Args:
            frame_num: Frame number to segment
            method: Segmentation method to use
            visualize: Whether to create visualization
        
        Returns:
            Dictionary with segmentation results and metrics
        """
        frame = self.improved_segmenter.get_frame(frame_num)
        if frame is None:
            return {'error': f'Frame {frame_num} not available'}
        
        # Perform segmentation
        mask = self.optimized_segmentation(frame, method=method)
        
        # Extract cell properties
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask, intensity_image=frame)
        
        cell_properties = []
        for region in regions:
            props = {
                'id': region.label,
                'area': region.area,
                'centroid': region.centroid,
                'bbox': region.bbox,
                'major_axis': region.major_axis_length,
                'minor_axis': region.minor_axis_length,
                'eccentricity': region.eccentricity,
                'mean_intensity': region.mean_intensity,
                'perimeter': region.perimeter,
                'circularity': 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0
            }
            cell_properties.append(props)
        
        # Evaluate against ground truth if available
        evaluation = self.improved_segmenter.evaluate_segmentation(mask, frame_num)
        
        # Create visualization if requested
        visualization_path = None
        if visualize:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original image
            ax1.imshow(frame, cmap='gray')
            ax1.set_title(f'Frame {frame_num} - Original')
            ax1.axis('off')
            
            # Segmentation result
            ax2.imshow(frame, cmap='gray', alpha=0.7)
            ax2.contour(mask, colors='red', linewidths=1.5)
            
            # Add cell centers
            for props in cell_properties:
                y, x = props['centroid']
                ax2.plot(x, y, 'bo', markersize=3)
                ax2.text(x+2, y+2, str(props['id']), fontsize=8, color='blue')
            
            f1_score = evaluation.get('f1_score', 0)
            ax2.set_title(f'Segmentation - {len(cell_properties)} cells\nF1 Score: {f1_score:.3f}')
            ax2.axis('off')
            
            plt.tight_layout()
            visualization_path = f'final_segmentation_frame_{frame_num}.png'
            plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
            plt.show()
        
        return {
            'frame_number': frame_num,
            'method_used': method,
            'cell_count': len(cell_properties),
            'cell_properties': cell_properties,
            'segmentation_mask': mask,
            'evaluation_metrics': evaluation,
            'visualization_path': visualization_path
        }
    
    def batch_segment(self, frame_range: Tuple[int, int] = None, method: str = 'best') -> Dict:
        """
        Segment multiple frames and return comprehensive results.
        
        Args:
            frame_range: Tuple of (start_frame, end_frame) or None for all
            method: Segmentation method to use
        
        Returns:
            Dictionary with batch results
        """
        if self.annotations:
            available_frames = sorted(self.annotations.keys())
        else:
            # Assume sequential frames from 0
            available_frames = list(range(self.tif_data.shape[0] if len(self.tif_data.shape) == 3 else 1))
        
        if frame_range:
            available_frames = [f for f in available_frames if frame_range[0] <= f <= frame_range[1]]
        
        print(f"Processing {len(available_frames)} frames with {method} method...")
        
        frame_results = []
        total_cells = 0
        evaluation_metrics = []
        
        for frame_num in available_frames:
            result = self.segment_frame(frame_num, method=method, visualize=False)
            
            if 'error' not in result:
                frame_results.append(result)
                total_cells += result['cell_count']
                
                if 'f1_score' in result['evaluation_metrics']:
                    evaluation_metrics.append(result['evaluation_metrics'])
                
                print(f"Frame {frame_num}: {result['cell_count']} cells detected")
        
        # Calculate summary statistics
        summary_stats = {
            'total_frames_processed': len(frame_results),
            'total_cells_detected': total_cells,
            'average_cells_per_frame': total_cells / len(frame_results) if frame_results else 0,
            'method_used': method
        }
        
        if evaluation_metrics:
            for metric in ['iou', 'precision', 'recall', 'f1_score']:
                values = [eval_m[metric] for eval_m in evaluation_metrics if metric in eval_m]
                if values:
                    summary_stats[f'average_{metric}'] = np.mean(values)
                    summary_stats[f'std_{metric}'] = np.std(values)
        
        return {
            'summary_statistics': summary_stats,
            'frame_results': frame_results,
            'processing_info': {
                'frames_requested': len(available_frames),
                'frames_processed': len(frame_results),
                'method': method
            }
        }
    
    def export_results(self, results: Dict, output_prefix: str = 'segmentation_results'):
        """Export results to various formats."""
        # JSON export
        json_file = f'{output_prefix}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
        
        # CSV export for cell properties
        if 'frame_results' in results:
            import csv
            csv_file = f'{output_prefix}_cells.csv'
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                header = ['frame', 'cell_id', 'area', 'centroid_y', 'centroid_x', 
                         'major_axis', 'minor_axis', 'eccentricity', 'mean_intensity', 
                         'perimeter', 'circularity']
                writer.writerow(header)
                
                # Data
                for frame_result in results['frame_results']:
                    frame_num = frame_result['frame_number']
                    for cell in frame_result['cell_properties']:
                        row = [
                            frame_num, cell['id'], cell['area'],
                            cell['centroid'][0], cell['centroid'][1],
                            cell['major_axis'], cell['minor_axis'],
                            cell['eccentricity'], cell['mean_intensity'],
                            cell['perimeter'], cell['circularity']
                        ]
                        writer.writerow(row)
        
        print(f"Results exported to:")
        print(f"- {json_file}: Complete results in JSON format")
        if 'frame_results' in results:
            print(f"- {csv_file}: Cell properties in CSV format")
    
    def create_summary_report(self, results: Dict, output_file: str = 'segmentation_report.txt'):
        """Create a human-readable summary report."""
        with open(output_file, 'w') as f:
            f.write("Cell Segmentation Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            if 'summary_statistics' in results:
                stats = results['summary_statistics']
                f.write("Summary Statistics:\n")
                f.write(f"- Method used: {stats.get('method_used', 'Unknown')}\n")
                f.write(f"- Frames processed: {stats.get('total_frames_processed', 0)}\n")
                f.write(f"- Total cells detected: {stats.get('total_cells_detected', 0)}\n")
                f.write(f"- Average cells per frame: {stats.get('average_cells_per_frame', 0):.1f}\n")
                
                if 'average_f1_score' in stats:
                    f.write(f"- Average F1 score: {stats['average_f1_score']:.3f} ± {stats.get('std_f1_score', 0):.3f}\n")
                    f.write(f"- Average precision: {stats.get('average_precision', 0):.3f}\n")
                    f.write(f"- Average recall: {stats.get('average_recall', 0):.3f}\n")
                f.write("\n")
            
            # Frame-by-frame breakdown
            if 'frame_results' in results:
                f.write("Frame-by-Frame Results:\n")
                f.write("-" * 30 + "\n")
                
                for frame_result in results['frame_results'][:10]:  # Show first 10
                    frame_num = frame_result['frame_number']
                    cell_count = frame_result['cell_count']
                    f1_score = frame_result.get('evaluation_metrics', {}).get('f1_score', 'N/A')
                    
                    f.write(f"Frame {frame_num}: {cell_count} cells")
                    if f1_score != 'N/A':
                        f.write(f", F1: {f1_score:.3f}")
                    f.write("\n")
                
                if len(results['frame_results']) > 10:
                    f.write(f"... and {len(results['frame_results']) - 10} more frames\n")
        
        print(f"Summary report saved to {output_file}")

def main():
    """Demonstrate the final segmentation pipeline."""
    tif_file = "annotated_data_1001/MattLines1.tif"
    xml_file = "annotated_data_1001/MattLines1annotations.xml"
    
    print("=== Final Cell Segmentation Pipeline ===")
    
    # Initialize pipeline
    pipeline = FinalCellSegmentationPipeline(tif_file, xml_file)
    
    # Test on individual frames with visualization
    test_frames = [0, 5, 26]
    print("\n--- Individual Frame Analysis ---")
    
    for frame_num in test_frames:
        print(f"\nProcessing frame {frame_num}...")
        result = pipeline.segment_frame(frame_num, method='best', visualize=True)
        
        if 'error' not in result:
            print(f"Detected {result['cell_count']} cells")
            if 'f1_score' in result['evaluation_metrics']:
                print(f"F1 Score: {result['evaluation_metrics']['f1_score']:.3f}")
    
    # Batch processing
    print(f"\n--- Batch Processing ---")
    batch_results = pipeline.batch_segment(frame_range=(0, 10), method='best')
    
    # Export results
    pipeline.export_results(batch_results, 'final_segmentation_results')
    pipeline.create_summary_report(batch_results)
    
    print(f"\n=== Pipeline Complete ===")
    print("Generated files:")
    print("- final_segmentation_frame_X.png: Individual frame visualizations")
    print("- final_segmentation_results.json: Complete results")
    print("- final_segmentation_results_cells.csv: Cell properties table")
    print("- segmentation_report.txt: Human-readable summary")

if __name__ == "__main__":
    main()