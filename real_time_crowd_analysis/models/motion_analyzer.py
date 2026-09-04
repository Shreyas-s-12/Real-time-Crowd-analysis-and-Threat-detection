"""
Motion Analysis Module — True Real-Time Behavioral Analysis
Uses Farneback dense optical flow for per-pixel motion analysis.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque # Keep deque import
import time # Keep time import
from real_time_crowd_analysis.utils.config import config # Absolute import
from real_time_crowd_analysis.utils.logger import setup_logger # Absolute import

logger = setup_logger("motion_analyzer")


class MotionAnalyzer:
    """
    Analyzes dense optical flow per frame.

    Outputs:
        - motion_magnitude   : mean magnitude of the full flow field (pixels/frame)
        - motion_direction   : mean angle of flow vectors (radians)
        - dir_variance       : circular variance [0..1] — high = chaotic directions
        - accel_score        : normalised acceleration (magnitude delta frame-over-frame)
        - flow_rgb           : HSV-colourised flow for debug overlay
        - motion_heatmap     : single-channel intensity map

    All values are computed from real frame data — no hard-coded constants.
    """

    def __init__(self, history_size: int = None):
        self.history_size = history_size or config.MOTION_HISTORY_SIZE
        self.scale = config.OPTICAL_FLOW_SCALE  # resize before flow for speed

        # Farneback parameters (tuned for real-time at 0.5 scale)
        self.fb_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # State
        self.prev_gray_small: Optional[np.ndarray] = None

        # Low light adaptation state
        self._last_is_low_light: bool = False
        self.low_light_mode: str = "Auto" # "Auto", "On", or "Off" manual override
        self._clahe = cv2.createCLAHE(clipLimit=config.LOW_LIGHT_CLAHE_LIMIT, tileGridSize=(8, 8))

        # Rolling history — each entry: {magnitude, dir_variance, accel, timestamp}
        self.motion_history: deque = deque(maxlen=self.history_size)

        # Keep last flow magnitude for acceleration computation
        self._last_magnitude: float = 0.0

        # Accumulated flow for heatmap persistence
        self._heatmap_acc: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_motion(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Process one frame and return motion analytics.

        Returns a dict with keys:
            motion_magnitude, dir_variance, accel_score,
            motion_score (0-1 normalised), flow_rgb, motion_heatmap,
            motion_vectors (list of (x1,y1,x2,y2) for overlay drawing)
        """
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # ---- Low Light Adaptation ----
        mean_brightness = float(np.mean(gray_small))
        
        # Handle Manual Override
        if self.low_light_mode == "On":
            is_low_light = True
        elif self.low_light_mode == "Off":
            is_low_light = False
        else: # Default: Auto
            is_low_light = mean_brightness < config.LOW_LIGHT_THRESHOLD
        
        # Reset buffer if we toggle modes to avoid massive motion spikes from CLAHE intensity shift
        if is_low_light != self._last_is_low_light:
            self.prev_gray_small = None
            self._last_is_low_light = is_low_light

        current_fb_params = self.fb_params.copy()
        if is_low_light:
            # Enhance contrast for better flow detection in dark scenes
            gray_small = self._clahe.apply(gray_small)
            # Increase window size to filter out sensor noise prevalent in low light
            current_fb_params['winsize'] = 25
            current_fb_params['poly_sigma'] = 1.5

        result = {
            'motion_magnitude': 0.0,
            'motion_direction': 0.0,
            'dir_variance': 0.0,
            'accel_score': 0.0,
            'motion_score': 0.0,
            'flow_rgb': None,
            'motion_heatmap': np.zeros((h, w), dtype=np.float32),
            'motion_vectors': [],
            'is_low_light': is_low_light
        }

        if self.prev_gray_small is None:
            self.prev_gray_small = gray_small
            return result

        # ---- Farneback dense optical flow ----
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray_small, gray_small, None, **current_fb_params
        )
        self.prev_gray_small = gray_small

        # ---- Derive magnitude / angle ----
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mean_mag = float(np.mean(mag))
        max_mag  = float(np.max(mag))

        # ---- Directional variance (circular) ----
        dir_variance = self._circular_variance(ang)

        # ---- Acceleration: frame-over-frame magnitude delta ----
        raw_accel = abs(mean_mag - self._last_magnitude)
        self._last_magnitude = mean_mag

        # Normalise accel against spike threshold → [0..1]
        accel_score = min(raw_accel / max(config.ACCEL_SPIKE_THRESHOLD, 0.01), 1.0)

        # Task 1: Increase sensitivity (cap at 10 px/frame instead of 20)
        motion_score = min(mean_mag / 10.0, 1.0)

        # ---- Build debug visuals at original resolution ----
        flow_rgb = self._flow_to_rgb(flow, mag, ang)
        # upscale to original frame size
        flow_rgb = cv2.resize(flow_rgb, (w, h))

        heatmap = self._build_heatmap(mag, (h, w))

        # ---- Motion vectors for overlay (sampled grid) ----
        vectors = self._sample_vectors(flow, mag, (h, w))

        # ---- Store in rolling history ----
        entry = {
            'magnitude': mean_mag,
            'dir_variance': dir_variance,
            'accel_score': accel_score,
            'motion_score': motion_score,
            'timestamp': time.time()
        }
        self.motion_history.append(entry)

        result.update({
            'motion_magnitude': mean_mag,
            'motion_direction': float(np.mean(ang)),
            'dir_variance': dir_variance,
            'accel_score': accel_score,
            'motion_score': motion_score,
            'flow_rgb': flow_rgb,
            'motion_heatmap': heatmap,
            'motion_vectors': vectors,
            'max_magnitude': max_mag,
            'is_low_light': is_low_light
        })
        return result

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------

    def get_motion_trend(self) -> Dict[str, object]:
        """Trend over the last N frames in rolling history."""
        if len(self.motion_history) < 2:
            return {'trend': 'STABLE', 'magnitude_trend': 0.0, 'direction_stability': 1.0}

        recent = list(self.motion_history)[-10:]
        mags = [e['magnitude'] for e in recent]

        x = np.arange(len(mags), dtype=float)
        slope = float(np.polyfit(x, mags, 1)[0]) if len(mags) >= 2 else 0.0

        if slope > 0.3:
            trend = 'INCREASING'
        elif slope < -0.3:
            trend = 'DECREASING'
        else:
            trend = 'STABLE'

        dir_stab = 1.0 - float(np.mean([e['dir_variance'] for e in recent]))

        return {
            'trend': trend,
            'magnitude_trend': slope,
            'direction_stability': dir_stab
        }

    def get_rolling_stats(self) -> Dict[str, float]:
        """Aggregate stats across the entire rolling window."""
        if not self.motion_history:
            return {'mean_magnitude': 0.0, 'mean_accel': 0.0, 'mean_dir_variance': 0.0}

        hist = list(self.motion_history)
        return {
            'mean_magnitude': float(np.mean([e['magnitude'] for e in hist])),
            'mean_accel': float(np.mean([e['accel_score'] for e in hist])),
            'mean_dir_variance': float(np.mean([e['dir_variance'] for e in hist]))
        }

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------

    def draw_debug_overlay(self, frame: np.ndarray,
                           motion_data: Dict[str, object],
                           show_vectors: bool = True,
                           show_heatmap: bool = True) -> np.ndarray:
        """
        Draw motion debug visualisation on *frame* (in-place copy returned).
        Green arrows = motion vectors, colour overlay = intensity heatmap.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # ---- Heatmap blend ----
        if show_heatmap:
            hm = motion_data.get('motion_heatmap')
            if hm is not None and hm.shape[:2] == (h, w):
                hm_norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
                vis = cv2.addWeighted(vis, 0.7, hm_color, 0.3, 0)

        # ---- Motion vectors ----
        if show_vectors:
            for (x1, y1, x2, y2) in motion_data.get('motion_vectors', []):
                cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 80), 1, tipLength=0.3)

        # ---- Stats HUD (bottom-left) ----
        mag = motion_data.get('motion_magnitude', 0.0)
        acc = motion_data.get('accel_score', 0.0)
        dv  = motion_data.get('dir_variance', 0.0)
        ms  = motion_data.get('motion_score', 0.0)

        hud_lines = [
            f"Motion Mag : {mag:.2f} px/f",
            f"Motion Score: {ms*100:.1f}%",
            f"Accel Score : {acc*100:.1f}%",
            f"Dir Variance: {dv:.3f}",
        ]
        y0 = h - 10 - len(hud_lines) * 18
        for i, line in enumerate(hud_lines):
            cv2.putText(vis, line, (10, y0 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        return vis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _circular_variance(ang: np.ndarray) -> float:
        """
        Circular variance of angle field.
        0 = all vectors point the same direction (no chaos).
        1 = completely random directions (maximum chaos).
        """
        sin_mean = float(np.mean(np.sin(ang)))
        cos_mean = float(np.mean(np.cos(ang)))
        r = np.sqrt(sin_mean**2 + cos_mean**2)
        return float(1.0 - r)

    @staticmethod
    def _flow_to_rgb(flow: np.ndarray, mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
        """Visualise flow as HSV image (hue = direction, value = magnitude)."""
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        hsv[..., 0] = (ang * 180.0 / np.pi / 2.0).astype(np.uint8)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _build_heatmap(self, mag_small: np.ndarray, full_size: Tuple[int, int]) -> np.ndarray:
        """
        Build a persistent, exponentially-decaying heatmap at original resolution.
        """
        h, w = full_size
        # upscale magnitude to full frame size
        mag_full = cv2.resize(mag_small, (w, h))

        if self._heatmap_acc is None:
            self._heatmap_acc = np.zeros((h, w), dtype=np.float32)

        # Decay + accumulate
        self._heatmap_acc *= 0.85
        self._heatmap_acc += mag_full.astype(np.float32)

        return self._heatmap_acc.copy()

    def _sample_vectors(self, flow: np.ndarray, mag: np.ndarray,
                        full_size: Tuple[int, int],
                        step: int = 20,
                        min_mag: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Sample a grid of motion vectors for overlay drawing.
        Only vectors with magnitude > min_mag are included.
        Returns list of (x1, y1, x2, y2) tuples in ORIGINAL frame coordinates.
        """
        sh, sw = flow.shape[:2]
        fh, fw = full_size
        scale_x = fw / sw
        scale_y = fh / sh
        # scale arrows to match full-res display
        arrow_scale = 3.0 / self.scale  # compensate for downscaling

        vectors = []
        for y in range(0, sh, step):
            for x in range(0, sw, step):
                if mag[y, x] < min_mag:
                    continue
                dx = flow[y, x, 0]
                dy = flow[y, x, 1]
                x1 = int(x * scale_x)
                y1 = int(y * scale_y)
                x2 = int(x1 + dx * arrow_scale)
                y2 = int(y1 + dy * arrow_scale)
                vectors.append((x1, y1, x2, y2))

        return vectors

    def reset(self):
        """Reset analyzer state (call when switching cameras)."""
        self.prev_gray_small = None
        self.motion_history.clear()
        self._last_magnitude = 0.0
        self._heatmap_acc = None


# ---------------------------------------------------------------------------
# Module-level singleton + convenience wrapper
# ---------------------------------------------------------------------------
motion_analyzer = MotionAnalyzer()


def analyze_motion(frame: np.ndarray) -> Dict[str, object]:
    """Analyse motion in frame and return metrics dict."""
    return motion_analyzer.analyze_motion(frame)