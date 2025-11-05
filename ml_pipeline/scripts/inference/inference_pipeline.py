"""
Complete inference pipeline for Candida albicans analysis.

Combines YOLOv8 detection, ByteTrack tracking, and morphological analysis
to produce all required deliverables.
"""

import sys
from pathlib import Path
import numpy as np
import tifffile
import cv2
from typing import List, Dict, Tuple, Optional
import json
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.inference.cell_tracker import (
    Detection, Track, ByteTrackTracker, CellTrackingAnalyzer, visualize_tracks
)


class CandidaAnalysisPipeline:
    """Complete analysis pipeline from TIFF to deliverables."""

    CLASS_NAMES = {
        0: 'single dispersed cell',
        1: 'clump dispersed cell',
        2: 'planktonic',
        3: 'yeast form',
        4: 'psuedohyphae',
        5: 'hyphae',
        6: 'biofilm'
    }

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = 'auto'
    ):
        """
        Initialize pipeline.

        Args:
            model_path: Path to trained YOLOv8 model weights
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device for inference
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = YOLO(str(self.model_path))

        # Initialize tracker
        self.tracker = ByteTrackTracker(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8
        )

        # Results storage
        self.all_tracks: List[Track] = []
        self.frame_data: Dict[int, any] = {}

    def load_tiff_sequence(self, tiff_path: str) -> np.ndarray:
        """
        Load multi-frame TIFF file.

        Args:
            tiff_path: Path to TIFF file

        Returns:
            Array of shape (num_frames, height, width)
        """
        print(f"Loading TIFF: {tiff_path}")
        tif_data = tifffile.imread(tiff_path)

        if tif_data.ndim == 2:
            tif_data = tif_data[np.newaxis, ...]

        print(f"  Shape: {tif_data.shape}")
        print(f"  Dtype: {tif_data.dtype}")
        print(f"  Frames: {tif_data.shape[0]}")

        return tif_data

    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on single frame.

        Args:
            frame: Input frame

        Returns:
            List of detections
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]

        detections = []

        # Parse results
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            # Get masks if available
            masks = None
            if hasattr(results, 'masks') and results.masks is not None:
                masks = results.masks.data.cpu().numpy()

            # Create detection objects
            for i in range(len(boxes)):
                mask = masks[i] if masks is not None else None

                det = Detection(
                    bbox=boxes[i],
                    score=scores[i],
                    class_id=class_ids[i],
                    mask=mask
                )
                detections.append(det)

        return detections

    def process_sequence(
        self,
        tiff_path: str,
        output_dir: Optional[str] = None,
        save_visualizations: bool = True
    ) -> Dict[str, any]:
        """
        Process entire TIFF sequence.

        Args:
            tiff_path: Path to input TIFF file
            output_dir: Directory to save outputs
            save_visualizations: Whether to save visualization frames

        Returns:
            Dictionary with all analysis results
        """
        # Setup output directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)

        # Load sequence
        frames = self.load_tiff_sequence(tiff_path)
        num_frames = frames.shape[0]

        print(f"\nProcessing {num_frames} frames...")
        print("-" * 60)

        # Process each frame
        for frame_idx in range(num_frames):
            print(f"Frame {frame_idx + 1}/{num_frames}...", end='\r')

            frame = frames[frame_idx]

            # Normalize frame for display (16-bit → 8-bit)
            frame_norm = self._normalize_frame(frame)

            # Detect objects
            detections = self.detect_frame(frame)

            # Update tracker
            active_tracks = self.tracker.update(detections, frame_idx)

            # Store frame data
            self.frame_data[frame_idx] = {
                'detections': detections,
                'tracks': active_tracks,
                'num_detections': len(detections)
            }

            # Save visualization
            if save_visualizations and output_dir is not None:
                vis_frame = visualize_tracks(
                    frame_norm,
                    active_tracks,
                    frame_idx,
                    self.CLASS_NAMES,
                    show_trajectory=True
                )
                vis_path = vis_dir / f'frame_{frame_idx:04d}.png'
                cv2.imwrite(str(vis_path), vis_frame)

        print("\n" + "-" * 60)

        # Get all tracks
        self.all_tracks = self.tracker.get_all_tracks()

        # Analyze results
        results = self._analyze_results()

        # Save results
        if output_dir is not None:
            self._save_results(results, output_dir)

        return results

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize 16-bit frame to 8-bit for visualization."""
        frame_min = frame.min()
        frame_max = frame.max()

        if frame_max > frame_min:
            frame_norm = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
        else:
            frame_norm = np.zeros_like(frame, dtype=np.uint8)

        return frame_norm

    def _analyze_results(self) -> Dict[str, any]:
        """Analyze tracking results to produce deliverables."""
        print("\nAnalyzing results...")

        analyzer = CellTrackingAnalyzer(self.all_tracks, self.CLASS_NAMES)

        # 1. Cell counts per frame
        cell_counts = analyzer.get_cell_count_per_frame()

        # 2. Biofilm area per frame (growth curve)
        biofilm_areas = analyzer.get_biofilm_area_per_frame()

        # 3. Dispersed cell counts
        dispersed_counts = analyzer.get_dispersed_cell_count()

        # 4. Dispersal initiation frame
        dispersal_frame = analyzer.detect_dispersal_initiation(threshold=10)

        # 5. Track statistics
        track_stats = analyzer.get_track_statistics()

        # 6. Per-class counts over time
        class_counts_over_time = {}
        for frame_idx in sorted(cell_counts.keys()):
            class_counts = analyzer.get_cell_count_by_class(frame_idx)
            class_counts_over_time[frame_idx] = class_counts

        results = {
            'total_tracks': len(self.all_tracks),
            'total_frames': len(self.frame_data),
            'dispersal_initiation_frame': dispersal_frame,
            'cell_counts_per_frame': cell_counts,
            'biofilm_areas_per_frame': biofilm_areas,
            'dispersed_counts_per_frame': dispersed_counts,
            'class_counts_over_time': class_counts_over_time,
            'track_statistics': track_stats
        }

        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Total tracks: {results['total_tracks']}")
        print(f"Total frames: {results['total_frames']}")
        print(f"Dispersal initiation: Frame {dispersal_frame}" if dispersal_frame else "No dispersal detected")
        print(f"\nTrack statistics:")
        for key, value in track_stats.items():
            if key != 'tracks_by_class':
                print(f"  {key}: {value}")
        print(f"\nTracks by class:")
        for cls_name, count in track_stats['tracks_by_class'].items():
            print(f"  {cls_name}: {count}")
        print("=" * 60)

        return results

    def _save_results(self, results: Dict[str, any], output_dir: Path):
        """Save results to files."""
        print(f"\nSaving results to {output_dir}...")

        # 1. Save JSON
        json_path = output_dir / 'results.json'
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2)
        print(f"  ✓ {json_path}")

        # 2. Save CSV for time-series data
        # Cell counts
        df_cells = pd.DataFrame({
            'frame': list(results['cell_counts_per_frame'].keys()),
            'total_cells': list(results['cell_counts_per_frame'].values())
        })
        df_cells.to_csv(output_dir / 'cell_counts.csv', index=False)
        print(f"  ✓ {output_dir / 'cell_counts.csv'}")

        # Biofilm areas
        if results['biofilm_areas_per_frame']:
            df_biofilm = pd.DataFrame({
                'frame': list(results['biofilm_areas_per_frame'].keys()),
                'biofilm_area': list(results['biofilm_areas_per_frame'].values())
            })
            df_biofilm.to_csv(output_dir / 'biofilm_growth.csv', index=False)
            print(f"  ✓ {output_dir / 'biofilm_growth.csv'}")

        # Dispersed cells
        df_dispersed = pd.DataFrame({
            'frame': list(results['dispersed_counts_per_frame'].keys()),
            'dispersed_cells': list(results['dispersed_counts_per_frame'].values())
        })
        df_dispersed.to_csv(output_dir / 'dispersed_cells.csv', index=False)
        print(f"  ✓ {output_dir / 'dispersed_cells.csv'}")

        # 3. Generate plots
        self._generate_plots(results, output_dir)

    def _convert_for_json(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _generate_plots(self, results: Dict[str, any], output_dir: Path):
        """Generate analysis plots."""
        print(f"\nGenerating plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Total cell count over time
        frames = list(results['cell_counts_per_frame'].keys())
        counts = list(results['cell_counts_per_frame'].values())
        axes[0, 0].plot(frames, counts, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Frame Number')
        axes[0, 0].set_ylabel('Total Cell Count')
        axes[0, 0].set_title('Total Cell Count Over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # Mark dispersal initiation
        if results['dispersal_initiation_frame'] is not None:
            axes[0, 0].axvline(
                results['dispersal_initiation_frame'],
                color='r', linestyle='--', label='Dispersal Initiation'
            )
            axes[0, 0].legend()

        # 2. Biofilm growth curve
        if results['biofilm_areas_per_frame']:
            bio_frames = list(results['biofilm_areas_per_frame'].keys())
            bio_areas = list(results['biofilm_areas_per_frame'].values())
            axes[0, 1].plot(bio_frames, bio_areas, 'g-', linewidth=2)
            axes[0, 1].set_xlabel('Frame Number')
            axes[0, 1].set_ylabel('Biofilm Area (pixels)')
            axes[0, 1].set_title('Biofilm Growth Curve')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Dispersed cell count
        disp_frames = list(results['dispersed_counts_per_frame'].keys())
        disp_counts = list(results['dispersed_counts_per_frame'].values())
        axes[1, 0].plot(disp_frames, disp_counts, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Frame Number')
        axes[1, 0].set_ylabel('Dispersed Cell Count')
        axes[1, 0].set_title('Dispersed Cells Over Time')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Class distribution
        track_stats = results['track_statistics']
        if 'tracks_by_class' in track_stats:
            classes = list(track_stats['tracks_by_class'].keys())
            class_counts = list(track_stats['tracks_by_class'].values())
            axes[1, 1].bar(range(len(classes)), class_counts)
            axes[1, 1].set_xticks(range(len(classes)))
            axes[1, 1].set_xticklabels(classes, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Track Count')
            axes[1, 1].set_title('Tracks by Cell Type')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = output_dir / 'analysis_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ {plot_path}")
        plt.close()


def main():
    """Main inference script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run complete Candida albicans analysis pipeline'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained YOLOv8 model weights'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input TIFF file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '--no-vis',
        action='store_true',
        help='Disable visualization saving'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device (auto, cpu, 0, etc.)'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = CandidaAnalysisPipeline(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )

    # Run analysis
    results = pipeline.process_sequence(
        tiff_path=args.input,
        output_dir=args.output,
        save_visualizations=not args.no_vis
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
