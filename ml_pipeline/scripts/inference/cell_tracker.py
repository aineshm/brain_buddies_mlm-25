"""
Cell tracking across time-series frames using ByteTrack algorithm.

Tracks individual cells through 41 frames (20 hours, 30-min intervals).
Enables growth analysis, dispersal detection, and trajectory visualization.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import cv2


@dataclass
class Detection:
    """Single object detection."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    class_id: int
    mask: Optional[np.ndarray] = None  # Segmentation mask

    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center."""
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        return (cx, cy)

    @property
    def area(self) -> float:
        """Get bounding box area."""
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return w * h


@dataclass
class Track:
    """Track for a single cell across frames."""
    track_id: int
    class_id: int
    detections: List[Tuple[int, Detection]]  # [(frame_num, detection), ...]
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    def update(self, frame_num: int, detection: Detection):
        """Update track with new detection."""
        self.detections.append((frame_num, detection))
        self.hits += 1
        self.time_since_update = 0
        self.age += 1

    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.time_since_update += 1
        self.age += 1

    @property
    def last_detection(self) -> Optional[Tuple[int, Detection]]:
        """Get last detection."""
        if self.detections:
            return self.detections[-1]
        return None

    @property
    def trajectory(self) -> np.ndarray:
        """Get trajectory as array of centers."""
        centers = []
        for frame_num, det in self.detections:
            centers.append(det.center)
        return np.array(centers)

    def get_detection_at_frame(self, frame_num: int) -> Optional[Detection]:
        """Get detection at specific frame."""
        for fn, det in self.detections:
            if fn == frame_num:
                return det
        return None


class ByteTrackTracker:
    """
    ByteTrack implementation for cell tracking.

    ByteTrack: https://arxiv.org/abs/2110.06864
    Uses all detections (high and low confidence) for robust tracking.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30
    ):
        """
        Initialize tracker.

        Args:
            track_thresh: Detection confidence threshold for first association
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching
            frame_rate: Frame rate (for temporal consistency)
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate

        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []

        self.frame_id = 0
        self.track_id_counter = 0

    def update(self, detections: List[Detection], frame_num: int) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections for current frame
            frame_num: Current frame number

        Returns:
            List of active tracks
        """
        self.frame_id = frame_num

        # Separate detections by confidence
        high_dets = [d for d in detections if d.score >= self.track_thresh]
        low_dets = [d for d in detections if d.score < self.track_thresh]

        # First association: high-confidence detections
        unmatched_tracks, unmatched_dets = self._match_detections(
            self.tracks, high_dets
        )

        # Second association: low-confidence detections with unmatched tracks
        if len(low_dets) > 0 and len(unmatched_tracks) > 0:
            unmatched_tracks, _ = self._match_detections(
                unmatched_tracks, low_dets
            )

        # Update lost tracks
        for track in unmatched_tracks:
            track.mark_missed()
            if track.time_since_update <= self.track_buffer:
                self.lost_tracks.append(track)
            else:
                self.removed_tracks.append(track)

        # Initialize new tracks for unmatched high-confidence detections
        for det in unmatched_dets:
            if det.score >= self.track_thresh:
                new_track = Track(
                    track_id=self._next_id(),
                    class_id=det.class_id,
                    detections=[(frame_num, det)],
                    age=1,
                    hits=1,
                    time_since_update=0
                )
                self.tracks.append(new_track)

        # Move active tracks from lost
        self.tracks = [t for t in self.tracks if t.time_since_update == 0]

        return self.tracks.copy()

    def _match_detections(
        self,
        tracks: List[Track],
        detections: List[Detection]
    ) -> Tuple[List[Track], List[Detection]]:
        """
        Match detections to tracks using IoU.

        Args:
            tracks: List of tracks
            detections: List of detections

        Returns:
            (unmatched_tracks, unmatched_detections)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return tracks, detections

        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            last_det = track.last_detection
            if last_det is None:
                continue
            _, last_detection = last_det

            for j, det in enumerate(detections):
                iou = self._compute_iou(last_detection.bbox, det.bbox)
                iou_matrix[i, j] = iou

        # Hungarian algorithm for matching (simplified greedy version)
        matched_indices = []
        unmatched_track_indices = list(range(len(tracks)))
        unmatched_det_indices = list(range(len(detections)))

        # Greedy matching
        while len(unmatched_track_indices) > 0 and len(unmatched_det_indices) > 0:
            # Find best match
            max_iou = 0
            best_track_idx = -1
            best_det_idx = -1

            for i in unmatched_track_indices:
                for j in unmatched_det_indices:
                    if iou_matrix[i, j] > max_iou and iou_matrix[i, j] > self.match_thresh:
                        max_iou = iou_matrix[i, j]
                        best_track_idx = i
                        best_det_idx = j

            if best_track_idx == -1:
                break

            # Match found
            matched_indices.append((best_track_idx, best_det_idx))
            unmatched_track_indices.remove(best_track_idx)
            unmatched_det_indices.remove(best_det_idx)

        # Update matched tracks
        for track_idx, det_idx in matched_indices:
            tracks[track_idx].update(self.frame_id, detections[det_idx])

        # Get unmatched
        unmatched_tracks = [tracks[i] for i in unmatched_track_indices]
        unmatched_dets = [detections[i] for i in unmatched_det_indices]

        return unmatched_tracks, unmatched_dets

    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute IoU between two bounding boxes.

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]

        Returns:
            IoU score
        """
        # Intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _next_id(self) -> int:
        """Generate next track ID."""
        track_id = self.track_id_counter
        self.track_id_counter += 1
        return track_id

    def get_all_tracks(self) -> List[Track]:
        """Get all tracks (active, lost, and removed)."""
        return self.tracks + self.lost_tracks + self.removed_tracks

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self.get_all_tracks():
            if track.track_id == track_id:
                return track
        return None


class CellTrackingAnalyzer:
    """Analyze tracking results for biological insights."""

    def __init__(self, tracks: List[Track], class_names: Dict[int, str]):
        """
        Initialize analyzer.

        Args:
            tracks: List of all tracks
            class_names: Mapping of class IDs to names
        """
        self.tracks = tracks
        self.class_names = class_names

    def get_cell_count_per_frame(self) -> Dict[int, int]:
        """Get number of cells detected per frame."""
        frame_counts = defaultdict(int)

        for track in self.tracks:
            for frame_num, _ in track.detections:
                frame_counts[frame_num] += 1

        return dict(sorted(frame_counts.items()))

    def get_cell_count_by_class(self, frame_num: int) -> Dict[str, int]:
        """Get cell counts by class for a specific frame."""
        class_counts = defaultdict(int)

        for track in self.tracks:
            det = track.get_detection_at_frame(frame_num)
            if det is not None:
                class_name = self.class_names.get(track.class_id, f'class_{track.class_id}')
                class_counts[class_name] += 1

        return dict(class_counts)

    def get_biofilm_area_per_frame(self, biofilm_class_id: int = 6) -> Dict[int, float]:
        """
        Get total biofilm area per frame.

        Args:
            biofilm_class_id: Class ID for biofilm

        Returns:
            Dictionary mapping frame number to total biofilm area
        """
        biofilm_areas = defaultdict(float)

        for track in self.tracks:
            if track.class_id == biofilm_class_id:
                for frame_num, det in track.detections:
                    if det.mask is not None:
                        # Use mask area if available
                        area = np.sum(det.mask > 0)
                    else:
                        # Use bbox area as approximation
                        area = det.area
                    biofilm_areas[frame_num] += area

        return dict(sorted(biofilm_areas.items()))

    def get_dispersed_cell_count(self) -> Dict[int, int]:
        """Get dispersed cell count per frame (single + clump + planktonic)."""
        dispersed_classes = [0, 1, 2]  # single, clump, planktonic
        dispersed_counts = defaultdict(int)

        for track in self.tracks:
            if track.class_id in dispersed_classes:
                for frame_num, _ in track.detections:
                    dispersed_counts[frame_num] += 1

        return dict(sorted(dispersed_counts.items()))

    def detect_dispersal_initiation(self, threshold: int = 10) -> Optional[int]:
        """
        Detect frame where dispersal begins.

        Args:
            threshold: Minimum number of dispersed cells to consider dispersal started

        Returns:
            Frame number where dispersal initiates, or None
        """
        dispersed_counts = self.get_dispersed_cell_count()

        for frame_num in sorted(dispersed_counts.keys()):
            if dispersed_counts[frame_num] >= threshold:
                return frame_num

        return None

    def get_track_statistics(self) -> Dict[str, any]:
        """Get overall tracking statistics."""
        stats = {
            'total_tracks': len(self.tracks),
            'avg_track_length': np.mean([len(t.detections) for t in self.tracks]),
            'max_track_length': max([len(t.detections) for t in self.tracks]) if self.tracks else 0,
            'tracks_by_class': defaultdict(int)
        }

        for track in self.tracks:
            class_name = self.class_names.get(track.class_id, f'class_{track.class_id}')
            stats['tracks_by_class'][class_name] += 1

        stats['tracks_by_class'] = dict(stats['tracks_by_class'])

        return stats


def visualize_tracks(
    image: np.ndarray,
    tracks: List[Track],
    frame_num: int,
    class_names: Dict[int, str],
    show_trajectory: bool = True,
    trajectory_length: int = 10
) -> np.ndarray:
    """
    Visualize tracks on image.

    Args:
        image: Input image
        tracks: List of tracks
        frame_num: Current frame number
        class_names: Class ID to name mapping
        show_trajectory: Whether to show trajectory trails
        trajectory_length: Number of past frames to show in trajectory

    Returns:
        Annotated image
    """
    # Convert to RGB if grayscale
    if image.ndim == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    # Color map for different classes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
    ]

    for track in tracks:
        det = track.get_detection_at_frame(frame_num)
        if det is None:
            continue

        color = colors[track.class_id % len(colors)]

        # Draw bounding box
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Draw mask if available
        if det.mask is not None:
            mask_overlay = np.zeros_like(vis_image)
            mask_overlay[det.mask > 0] = color
            vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)

        # Draw track ID and class
        class_name = class_names.get(track.class_id, f'{track.class_id}')
        label = f'ID:{track.track_id} {class_name}'
        cv2.putText(
            vis_image, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

        # Draw trajectory
        if show_trajectory and len(track.detections) > 1:
            # Get recent detections
            recent_dets = [
                (fn, d) for fn, d in track.detections
                if frame_num - trajectory_length <= fn <= frame_num
            ]

            if len(recent_dets) > 1:
                points = np.array([d.center for _, d in recent_dets], dtype=np.int32)
                cv2.polylines(vis_image, [points], False, color, 2)

    return vis_image


if __name__ == '__main__':
    print("Cell Tracking Module")
    print("=" * 60)
    print("ByteTrack implementation for Candida albicans tracking")
    print("=" * 60)
