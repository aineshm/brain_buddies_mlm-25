#!/usr/bin/env python3
"""
Simple Cell Segmentation Tool
Easy-to-use interface for cell segmentation in microscopy images.

Usage:
    python simple_segmentation.py --frame 5 --visualize
    python simple_segmentation.py --batch 0-10 --method best
"""

import argparse
import sys
import os
from final_cell_segmentation import FinalCellSegmentationPipeline

def segment_single_frame(tif_file, xml_file, frame_num, method='best', visualize=False):
    """Segment a single frame."""
    pipeline = FinalCellSegmentationPipeline(tif_file, xml_file)
    
    print(f"Segmenting frame {frame_num} using {method} method...")
    result = pipeline.segment_frame(frame_num, method=method, visualize=visualize)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return False
    
    print(f"✅ Detected {result['cell_count']} cells")
    
    if 'f1_score' in result['evaluation_metrics']:
        f1 = result['evaluation_metrics']['f1_score']
        precision = result['evaluation_metrics']['precision']
        recall = result['evaluation_metrics']['recall']
        print(f"📊 F1 Score: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
    
    if visualize and result['visualization_path']:
        print(f"🖼️  Visualization saved: {result['visualization_path']}")
    
    return True

def segment_batch(tif_file, xml_file, frame_range, method='best'):
    """Segment multiple frames."""
    pipeline = FinalCellSegmentationPipeline(tif_file, xml_file)
    
    print(f"Batch segmenting frames {frame_range[0]}-{frame_range[1]} using {method} method...")
    results = pipeline.batch_segment(frame_range=frame_range, method=method)
    
    if not results['frame_results']:
        print("❌ No frames processed successfully")
        return False
    
    stats = results['summary_statistics']
    print(f"✅ Processed {stats['total_frames_processed']} frames")
    print(f"📊 Total cells detected: {stats['total_cells_detected']}")
    print(f"📈 Average cells per frame: {stats['average_cells_per_frame']:.1f}")
    
    if 'average_f1_score' in stats:
        print(f"🎯 Average F1 Score: {stats['average_f1_score']:.3f}")
    
    # Export results
    pipeline.export_results(results, 'batch_segmentation_results')
    pipeline.create_summary_report(results, 'batch_segmentation_report.txt')
    
    print(f"📁 Results exported:")
    print(f"   - batch_segmentation_results.json")
    print(f"   - batch_segmentation_results_cells.csv") 
    print(f"   - batch_segmentation_report.txt")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Simple Cell Segmentation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_segmentation.py --frame 5 --visualize
  python simple_segmentation.py --batch 0-10 --method best
  python simple_segmentation.py --frame 26 --method adaptive --no-viz
        """
    )
    
    # File arguments
    parser.add_argument('--tif', default='annotated_data_1001/MattLines1.tif',
                       help='Path to TIF file (default: annotated_data_1001/MattLines1.tif)')
    parser.add_argument('--xml', default='annotated_data_1001/MattLines1annotations.xml',
                       help='Path to XML annotations (default: annotated_data_1001/MattLines1annotations.xml)')
    
    # Processing options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--frame', type=int, help='Segment single frame number')
    group.add_argument('--batch', type=str, help='Segment frame range (e.g., "0-10")')
    
    # Method selection
    parser.add_argument('--method', choices=['best', 'adaptive', 'edge', 'hybrid'], 
                       default='best', help='Segmentation method (default: best)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualizations (for single frame)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualizations')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.tif):
        print(f"❌ TIF file not found: {args.tif}")
        return 1
    
    if args.xml and not os.path.exists(args.xml):
        print(f"⚠️  XML file not found: {args.xml} (continuing without ground truth)")
        args.xml = None
    
    # Process arguments
    try:
        if args.frame is not None:
            # Single frame
            visualize = args.visualize and not args.no_viz
            success = segment_single_frame(args.tif, args.xml, args.frame, 
                                         args.method, visualize)
        
        elif args.batch:
            # Batch processing
            if '-' not in args.batch:
                print("❌ Batch format should be 'start-end' (e.g., '0-10')")
                return 1
            
            start, end = map(int, args.batch.split('-'))
            success = segment_batch(args.tif, args.xml, (start, end), args.method)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())