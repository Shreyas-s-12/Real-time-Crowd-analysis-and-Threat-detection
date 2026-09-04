"""
Object Tracking Module for Real-Time Crowd Analysis and Threat Detection
Includes rolling motion history buffers for velocity and acceleration analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque # Keep collections imports
import time # Keep time import
from real_time_crowd_analysis.utils.config import config # Absolute import
from real_time_crowd_analysis.utils.logger import setup_logger # Absolute import

logger = setup_logger("tracker")

class PersonTracker:
    """Track persons across frames using simple centroid tracking"""
    
    def __init__(self, max_disappeared: int = 30, max_distance: int = 50):
        self.next_object_id = 0
        self.objects = dict()           # object_id -> centroid (x, y)
        self.disappeared = dict()       # object_id -> disappeared count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        hist = config.MOTION_HISTORY_SIZE

        # Rolling position history — (x, y, timestamp)
        self.centroid_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=hist)
        )
        # Derived velocity vectors — (vx, vy) pixels/sec
        self.velocity_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=hist)
        )
        # Speed scalars (|v|) for acceleration computation
        self.speed_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=hist)
        )

        # Legacy traces (kept for draw_tracks compatibility)
        self.traces = defaultdict(lambda: deque(maxlen=hist))

        # Crowd analytics
        self.total_count = 0
        self.entry_line_position = None
        self.exit_line_position = None
        self.counted_ids = set()
        
    def register(self, centroid: Tuple[int, int]):
        """Register a new object."""
        oid = self.next_object_id
        self.objects[oid] = centroid
        self.disappeared[oid] = 0
        self.traces[oid].append(centroid)
        self.centroid_history[oid].append((centroid[0], centroid[1], time.time()))
        self.next_object_id += 1
    
    def deregister(self, object_id: int):
        """Deregister an object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.traces[object_id]
        self.centroid_history.pop(object_id, None)
        self.velocity_history.pop(object_id, None)
        self.speed_history.pop(object_id, None)
    
    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int]]:
        """
        Update object positions with new detections
        
        Args:
            rects: List of bounding boxes (x, y, w, h)
            
        Returns:
            Dictionary mapping object_id to centroid
        """
        # If no detections, mark all existing objects as disappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        # Convert rectangles to centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids[i] = (cX, cY)
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(tuple(input_centroids[i]))
        else:
            # Get current object centroids
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            # Compute distances between existing objects and new detections
            from scipy.spatial import distance as dist
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Find minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Track which rows and columns we've already examined
            used_rows = set()
            used_cols = set()
            
            # Update existing objects
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                new_centroid = tuple(input_centroids[col])
                self.objects[object_id] = new_centroid
                self.disappeared[object_id] = 0
                self.traces[object_id].append(new_centroid)
                self._update_motion(object_id, new_centroid)

                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unused rows (objects that disappeared)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            # Handle unused columns (new detections)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            
            # If we have more objects than detections, check for disappearances
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # If we have more detections than objects, register new ones
                for col in unused_cols:
                    self.register(tuple(input_centroids[col]))
        
        return self.objects

    # ------------------------------------------------------------------
    # Motion history helpers
    # ------------------------------------------------------------------

    def _update_motion(self, object_id: int, centroid: Tuple[int, int]):
        """Append centroid with timestamp and compute velocity/speed."""
        now = time.time()
        self.centroid_history[object_id].append((centroid[0], centroid[1], now))

        hist = self.centroid_history[object_id]
        if len(hist) >= 2:
            x1, y1, t1 = hist[-2]
            x2, y2, t2 = hist[-1]
            dt = t2 - t1
            if dt > 0:
                vx = (x2 - x1) / dt
                vy = (y2 - y1) / dt
                speed = np.hypot(vx, vy)
                self.velocity_history[object_id].append((vx, vy))
                self.speed_history[object_id].append((speed, now))

    def get_object_traces(self) -> Dict[int, List[Tuple[int, int]]]:
        """Get movement traces for all tracked objects."""
        return {oid: list(trace) for oid, trace in self.traces.items()}

    def get_velocities(self) -> Dict[int, Tuple[float, float]]:
        """Return latest velocity vector (vx, vy) per tracked object."""
        result = {}
        for oid, vh in self.velocity_history.items():
            if oid in self.objects and vh:
                result[oid] = vh[-1]
        return result

    def get_acceleration_vectors(self) -> Dict[int, float]:
        """
        Return per-object acceleration magnitude (px/s²).
        Computed as speed delta between the last two speed readings.
        """
        result = {}
        for oid, sh in self.speed_history.items():
            if oid not in self.objects or len(sh) < 2:
                continue
            s1, t1 = sh[-2]
            s2, t2 = sh[-1]
            dt = t2 - t1
            if dt > 0:
                result[oid] = abs(s2 - s1) / dt
        return result

    def get_speed_stats(self) -> Dict[str, float]:
        """Aggregate speed stats across all currently tracked objects."""
        speeds = []
        for oid, sh in self.speed_history.items():
            if oid in self.objects and sh:
                speeds.append(sh[-1][0])  # latest speed
        if not speeds:
            return {'mean': 0.0, 'max': 0.0, 'std': 0.0}
        return {
            'mean': float(np.mean(speeds)),
            'max':  float(np.max(speeds)),
            'std':  float(np.std(speeds))
        }
    
    def get_object_count(self) -> int:
        """Get current number of tracked objects"""
        return len(self.objects)
    
    def draw_tracks(self, frame: np.ndarray, traces: Dict[int, List[Tuple[int, int]]] = None) -> np.ndarray:
        """Draw tracking lines on frame"""
        if traces is None:
            traces = self.get_object_traces()
        
        for object_id, points in traces.items():
            if len(points) < 2:
                continue
            
            # Draw trace lines
            for i in range(1, len(points)):
                thickness = int(np.sqrt(64 / float(i + 1)))
                cv2.line(frame, points[i - 1], points[i], (0, 255, 0), thickness)
            
            # Draw object ID
            if points:
                cv2.putText(frame, f"ID: {object_id}", 
                           (points[-1][0] + 10, points[-1][1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

# Global tracker instance
person_tracker = PersonTracker()

# Convenience function
def track_persons(detections: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int]]:
    """Track persons in frame"""
    return person_tracker.update(detections)