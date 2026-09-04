"""
Threat Detection Module — True Real-Time Behavioral Analysis
Replaces all static confidence values with a dynamic weighted formula.
Adds: crowd validation, threat confirmation window, duplicate suppression.
Adds: Fight detection using YOLO Person Detection and MediaPipe Pose.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time
import hashlib

from ..utils.config import config
from ..utils.logger import setup_logger
from ..utils.alerts import trigger_alert

logger = setup_logger("threat_detector")


# ---------------------------------------------------------------------------
# Threat level tiers
# ---------------------------------------------------------------------------
THREAT_TIERS = [
    (85, "CRITICAL"),    # Risk Score >= 85
    (65, "HIGH RISK"),   # Risk Score >= 65
    (40, "SUSPICIOUS"),  # Risk Score >= 40
    (20, "LOW RISK"),    # Risk Score >= 20
    (0,  "NORMAL"),      # Risk Score >= 0
]


# Module-level helper function for threat description
def _describe_threat(
        motion_score: float,
        combined_accel: float,
        dir_variance: float,
        legacy_density_score: float,
        crowd_count: int,
        weapon_detections: List[Dict],
        has_weapon: bool,
        weapon_persistence_counter: int,
        behavior_score_component: float,
        weapon_score_component: float,
        historical_events_score_component: float, # From historical events
        fight_score_component: float, # New parameter
        is_stampede: bool # New parameter
) -> str:
    """Generate a human-readable threat type string from sub-scores."""
    parts = []
    
    # Prioritize weapon detection in description
    if has_weapon:
        # Get highest confidence of detected weapons
        max_weapon_conf = max([d['conf'] for d in weapon_detections]) if weapon_detections else 0.0
        # Add weapon score if significant
        if weapon_score_component > 0.5:
            parts.append(f"Weapon Score: {weapon_score_component*100:.0f}%")
        
        # If weapon is the primary threat, other factors are secondary
        # But still useful to describe if they contribute significantly

    # Describe crowd density based on new thresholds
    if crowd_count > 80:
        parts.append("HIGH CROWD DENSITY")
    elif crowd_count > 40:
        parts.append("MEDIUM CROWD DENSITY")
    elif crowd_count > 15:
        parts.append("LOW CROWD DENSITY")

    # Describe behavior based on behavior_score_component
    if behavior_score_component > 0.7:
        parts.append("EXTREME BEHAVIOR")
    elif behavior_score_component > 0.5:
        parts.append("ABNORMAL BEHAVIOR")
    elif behavior_score_component > 0.3:
        parts.append("INCREASED ACTIVITY")

    # More granular descriptions for specific motion features if they are high
    if motion_score > 0.6:
        parts.append("RAPID MOVEMENT")
    if combined_accel > 0.7:
        parts.append("ACCELERATION SPIKE")
    if dir_variance > config.DIR_CHAOS_THRESHOLD + 0.1: # Higher threshold for description
        parts.append("DIRECTIONAL CHAOS")
    
    # Add historical context if relevant
    if historical_events_score_component > 0.5:
        parts.append("RECENT ALERT HISTORY")
    
    # Add fight detection if significant
    if fight_score_component > 0.5:
        parts.append("FIGHT DETECTED")
    
    # Specific scenarios
    if motion_score > 0.5 and dir_variance > 0.5 and crowd_count >= config.MIN_CROWD_FOR_PANIC:
        if "PANIC MOVEMENT" not in parts: # Avoid redundancy
            parts.append("PANIC MOVEMENT")
    
    # Add stampede detection
    if is_stampede:
        parts.append("POSSIBLE STAMPEDE")
    
    if not parts:
        parts.append("NORMAL ACTIVITY")
    
    return " + ".join(parts)


# ---------------------------------------------------------------------------
# Threat Confirmation Window
# ---------------------------------------------------------------------------
class ThreatConfirmationWindow:
    """
    Counts consecutive frames where an anomaly condition is True.
    Only emits 'confirmed' once the streak reaches the required minimum.
    Resets on first normal frame.
    """

    def __init__(self, required_frames: int = None):
        self.required = required_frames or config.THREAT_CONFIRMATION_FRAMES
        self._streak: int = 0
        self._confirmed: bool = False

    def update(self, is_anomalous: bool) -> bool:
        """Feed one frame result; return True if threat is now confirmed."""
        if is_anomalous:
            self._streak += 1
            if self._streak >= self.required:
                self._confirmed = True
        else:
            self._streak = 0
            self._confirmed = False
        return self._confirmed

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    def reset(self):
        self._streak = 0
        self._confirmed = False


def _classify_threat_level(risk_score: float) -> str:
    """Map risk score [0..100] to a human-readable threat tier."""
    for threshold, label in THREAT_TIERS:
        if risk_score >= threshold:
            return label
    return "NORMAL" # Should not be reached if 0 is included



# ---------------------------------------------------------------------------
# Main Threat Detector
# ---------------------------------------------------------------------------
class ThreatDetector:
    """
    Detects abnormal crowd behaviour using:
      - Farneback optical flow metrics (from MotionAnalyzer)
      - Per-person tracker velocity / acceleration (from PersonTracker)
      - Crowd density
      - Rolling frame history
      - Threat confirmation window (no single-frame false positives)
      - Duplicate / cooldown suppression
      - Fight detection using YOLO Person Detection and MediaPipe Pose
    """

    def __init__(self, history_length: int = None):
        self.history_length = history_length or config.MOTION_HISTORY_SIZE
        self.weapon_grace_frames = 25  # Keep weapon threat active for ~0.8s if object is lost
        self.weapon_cooldown_counter = 0 # Counts down frames until weapon threat is fully cleared
        self.weapon_persistence_counter = 0 # Counts consecutive frames a weapon is detected
        self.weapon_persistence_threshold = config.WEAPON_PERSISTENCE_FRAMES # Frames needed to confirm weapon

        # Per-object rolling history (populated from tracker data)
        self.speed_history:  Dict[int, deque] = {}
        self.accel_history:  Dict[int, deque] = {}
        self.dir_history:    Dict[int, deque] = {}

        # Crowd-level rolling history
        self.crowd_density_history = deque(maxlen=self.history_length)
        self.avg_speed_history     = deque(maxlen=self.history_length)
        self.dir_variance_history = deque(maxlen=self.history_length) # For stampede detection
        self.optical_flow_magnitude_history = deque(maxlen=self.history_length) # For stampede detection


        # Threat confirmation
        self._confirmation = ThreatConfirmationWindow()

        # Cooldown / duplicate suppression
        self._last_alert_time:  float = 0.0
        self._last_alert_hash:  str   = ""
        self._cooldown: float = config.ALERT_COOLDOWN_SECONDS

        # Expose last analysis result
        self.last_analysis: Dict = {}

        # History for smoothing the final risk score
        self.risk_score_history = deque(maxlen=config.FPS * 2) # Smooth over last 2 seconds

        # Fight detection components
        self.pose_history: Dict[int, deque] = {} # track_id -> deque of pose landmarks for fight detection
        self.fight_persistence_counter = 0 # Counter for fight detection persistence
        try:
            self.mp_pose = __import__('mediapipe').solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_available = True
        except ImportError:
            logger.warning("MediaPipe not available. Fight detection disabled.")
            self.pose_available = False
            self.pose = None
        
        try:
            # Use the global person_detector instance to avoid re-initializing YOLO for fight detection
            from .detector import person_detector as global_person_detector
            self.yolo_person = global_person_detector.model # Access the YOLO model directly
            self.yolo_available = global_person_detector.use_yolo and (self.yolo_person is not None)
            # Only warn if pose is available but YOLO isn't, as YOLO is needed for person boxes
            if not self.yolo_available and self.pose_available:
                logger.warning("YOLO person detection not available via global detector. Fight detection might be limited.")
        except Exception as e:
            logger.warning(f"Could not access global person detector for fight detection: {e}. Fight detection might be limited.")
            self.yolo_available = False
            self.yolo_person = None

        # Thresholds for fight detection (normalized by torso height per frame)
        self.WRIST_SPEED_THRESHOLD = 0.1   # wrist movement > 10% of torso height per frame suggests punching
        self.ANKLE_SPEED_THRESHOLD = 0.1   # ankle movement > 10% of torso height per frame suggests kicking
        self.FIGHT_SCORE_THRESHOLD = 0.5   # fight_score_component above this is considered significant
        self.person_box_association_distance = 50 # Max distance to associate a person box with a tracked object

        # Persistence counters for alert conditions
        self._weapon_alert_persistence = 0
        self._fight_alert_persistence = 0
        self._stampede_alert_persistence = 0
        self._high_risk_alert_persistence = 0

        # Required frames for each condition (from config, with defaults)
        self._weapon_required_frames = getattr(config, 'WEAPON_ALERT_PERSISTENCE_FRAMES', 1) # Alert immediately for weapon (or after 1 frame)
        self._fight_required_frames = getattr(config, 'FIGHT_ALERT_PERSISTENCE_FRAMES', config.FPS * 2) # 2 seconds for fight detection
        self._stampede_required_frames = getattr(config, 'STAMPEDE_ALERT_PERSISTENCE_FRAMES', config.FPS * 3) # 3 seconds for stampede detection
        self._high_risk_required_frames = getattr(config, 'HIGH_RISK_ALERT_PERSISTENCE_FRAMES', config.FPS * 3) # 3 seconds for general high risk

    # ------------------------------------------------------------------
    # Public: main analysis entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        frame: np.ndarray,
        tracked_objects: Dict[int, Tuple[int, int]],
        frame_shape: Tuple[int, int, int],
        motion_data: Dict,          # output from MotionAnalyzer.analyze_motion()
        tracker_accel: Dict[int, float] = None,  # from PersonTracker.get_acceleration_vectors()
        tracker_speeds: Dict[str, float] = None,  # from PersonTracker.get_speed_stats()
        weapon_detections: List[Dict] = None,
        person_boxes: List[List[int]] = None
    ) -> Dict:
        """
        Analyze one frame using behavior-based risk assessment.

        Returns dict with:
            risk_score        (float 0-100)
            risk_score_pct    (str e.g. "63.7%")
            threat_level      (str tier label)
            threat_type       (str description)
            motion_score      (float 0-1)
            combined_accel    (float 0-1)
            dir_variance      (float 0-1)
            density_score     (float 0-1)
            crowd_count       (int)
            confirmation_streak (int frames so far)
            is_confirmed      (bool — True only after enough consecutive frames)
            should_alert      (bool)
            fight_score       (float 0-1)  -- new component
        """
        if frame is None:
            # If no frame is provided, we cannot do fight detection for this frame
            fight_score_component = 0.0
        else:
            fight_score_component = self._compute_fight_score(frame, tracked_objects, frame_shape, person_boxes)

        crowd_count = len(tracked_objects) if tracked_objects else 0
        h, w = frame_shape[:2]
        frame_area = max(w * h, 1)
        
        # ---- Update history for stampede detection ----
        optical_flow_magnitude = float(motion_data.get('motion_magnitude', 0.0))
        avg_speed_from_tracker = tracker_speeds.get('mean', 0.0) if tracker_speeds else 0.0
        self.avg_speed_history.append(avg_speed_from_tracker)
        dir_variance_from_motion = float(motion_data.get('dir_variance', 0.0))
        self.dir_variance_history.append(dir_variance_from_motion)
        self.optical_flow_magnitude_history.append(optical_flow_magnitude)

        # ---- Detect Stampede ----
        stampede_condition = self._detect_stampede(
            avg_speed_from_tracker,
            dir_variance_from_motion,
            optical_flow_magnitude,
            crowd_count
        )


        # ---- Base scores from motion analysis ----
        motion_score  = float(motion_data.get('motion_score',  0.0))
        accel_score   = float(motion_data.get('accel_score',   0.0))
        dir_variance  = float(motion_data.get('dir_variance',  0.0))

        # ---- Sub-score from tracker acceleration ----
        if tracker_accel and crowd_count >= config.MIN_CROWD_FOR_PANIC:
            accels = list(tracker_accel.values())
            # Normalise: cap at ACCEL_SPIKE_THRESHOLD px/s²
            norm_accels = [min(a / max(config.ACCEL_SPIKE_THRESHOLD * 30, 1.0), 1.0)
                           for a in accels]
            tracker_accel_score = float(np.mean(norm_accels)) if norm_accels else 0.0
        else:
            tracker_accel_score = 0.0

        # Blend optical-flow accel with tracker accel
        combined_accel = max(accel_score, tracker_accel_score)

        # ---- Weapon Persistence & Cooldown Logic ----
        weapon_detected_this_frame = bool(weapon_detections)
        
        # Update weapon persistence counter
        if weapon_detected_this_frame:
            self.weapon_persistence_counter = min(self.weapon_persistence_counter + 1, self.weapon_persistence_threshold + 5) # Cap to avoid overflow
            self.weapon_cooldown_counter = self.weapon_grace_frames
        else:
            self.weapon_persistence_counter = max(self.weapon_persistence_counter - 1, 0)
            # Only start cooldown if weapon is no longer persistently detected
            if self.weapon_persistence_counter == 0 and self.weapon_cooldown_counter > 0:
                self.weapon_cooldown_counter -= 1
         
        # A weapon is considered "present" if it's persistently detected OR in its cooldown period
        has_weapon = (self.weapon_persistence_counter >= self.weapon_persistence_threshold) or (self.weapon_cooldown_counter > 0)

        # ---- Speed stats (from tracker) ----
        if tracker_speeds and crowd_count >= config.MIN_CROWD_FOR_PANIC:
            avg_speed = tracker_speeds.get('mean', 0.0)
            max_speed = tracker_speeds.get('max', 0.0)
            # Normalize speed score, capping at a reasonable threshold
            speed_score = min(max_speed / max(config.THREAT_SPEED_THRESHOLD, 0.01), 1.0)
        else:
            avg_speed = 0.0
            max_speed = 0.0
            speed_score = 0.0

        # ---- NEW: Behavior-based Risk Assessment Components ----

        # 1. Crowd Density Score (0.15 weight) - based on crowd count thresholds
        # Crowd count alone should never trigger HIGH RISK. Max contribution 15/100.
        crowd_density_score_component = 0.0
        if crowd_count <= 15:
            crowd_density_score_component = 0.0 # NORMAL crowd density
        elif crowd_count <= 40:
            crowd_density_score_component = 0.15 # LOW RISK crowd density
        elif crowd_count <= 80:
            crowd_density_score_component = 0.30 # MEDIUM RISK crowd density
        else: # > 80 people
            crowd_density_score_component = 0.45 # HIGH CROWD DENSITY
        
        # For backward compatibility with old density_score field, normalize to 0-1
        # This is not directly used in the new risk formula but might be displayed.
        density_ratio = crowd_count / (frame_area / 10_000.0)
        legacy_density_score = min(density_ratio / max(config.THREAT_DENSITY_THRESHOLD / 100.0, 0.01), 1.0)

        # 2. Behavior Score (0.15 weight)
        # Combines motion, acceleration, directional chaos, and speed
        behavior_raw_score = (
            0.3 * motion_score +
            0.3 * combined_accel +
            0.2 * dir_variance +
            0.2 * speed_score
        )
        behavior_score_component = float(np.clip(behavior_raw_score, 0.0, 1.0))

        # 3. Weapon Score (0.30 weight)
        weapon_score_component = 0.0
        if weapon_detected_this_frame:
            max_weapon_conf = max([d['conf'] for d in weapon_detections]) if weapon_detections else 0.0
            weapon_score_component = max_weapon_conf
        elif has_weapon: # Weapon in cooldown period
            # Assign a base score if weapon was recently seen but not in current frame
            weapon_score_component = config.WEAPON_THREAT_CONFIDENCE # e.g., 0.85
        weapon_score_component = float(np.clip(weapon_score_component, 0.0, 1.0))

        # 4. Motion Score (0.05 weight) - direct motion_score from optical flow
        motion_score_component = motion_score # Already 0-1

        # 5. Historical Events (0.05 weight)
        historical_events_score_component = 0.0
        alert_decay_duration = config.ALERT_COOLDOWN_SECONDS * 2 # e.g., 16 seconds
        if self._last_alert_time > 0 and (time.time() - self._last_alert_time < alert_decay_duration):
            historical_events_score_component = max(0.0, 1.0 - (time.time() - self._last_alert_time) / alert_decay_duration)
        
        # 6. Fight Score (0.30 weight)
        # fight_score_component already computed above (0-1)

        # ---- Calculate Raw Risk Score (0-1) ----
        raw_risk_score = (
            0.15 * crowd_density_score_component +
            0.15 * behavior_score_component +
            0.30 * weapon_score_component +
            0.05 * motion_score_component +
            0.05 * historical_events_score_component +
            0.30 * fight_score_component
        )
        
        # Scale to 0-100 and clip
        final_risk_score = float(np.clip(raw_risk_score * 100, 0.0, 100.0))

        # ---- Smooth the score over the last `config.FPS * 2` frames (2 seconds) ----
        self.risk_score_history.append(final_risk_score)
        smoothed_risk_score = float(np.mean(self.risk_score_history))

        # ---- Crowd guard: suppress panic flags for tiny groups (if no weapon) ----
        # The new crowd_density_score_component inherently handles this by giving low weight.
        # We can still cap the overall risk score if the crowd is too small AND no weapon, fight, or stampede.
        fight_cond_active = fight_score_component > self.FIGHT_SCORE_THRESHOLD
        if crowd_count < config.MIN_CROWD_FOR_PANIC and not has_weapon and not fight_cond_active and not stampede_condition:
            smoothed_risk_score = min(smoothed_risk_score, 39.9) # Cap at just below SUSPICIOUS

        # ---- Threat level label ----
        threat_level = _classify_threat_level(smoothed_risk_score)

        # ---- Generate human-readable threat description ----
        threat_type = _describe_threat(
            motion_score,
            combined_accel,
            dir_variance,
            legacy_density_score,
            crowd_count,
            weapon_detections,
            has_weapon,
            self.weapon_persistence_counter,
            behavior_score_component,
            weapon_score_component,
            historical_events_score_component,
            fight_score_component,
            is_stampede=stampede_condition
        )

        # ---- Custom alert conditions with persistence ----
        
        # Suspicious behavior persists for more than 2 seconds (60 frames at 30fps)
        suspicious_cond = smoothed_risk_score >= THREAT_TIERS[2][0] # SUSPICIOUS threshold
        high_risk_cond = smoothed_risk_score >= THREAT_TIERS[1][0] # HIGH RISK threshold

        # Stampede detection persists for at least 3 seconds (90 frames)
        # stampede_condition is already a boolean indicating persistence

        # Update persistence counters for each specific alert condition
        # These counters are for triggering the `should_alert` flag based on persistence requirements.
        # The `_confirmation` window is for general anomalous behavior.
        if weapon_detected_this_frame and (max([d['conf'] for d in weapon_detections]) > 0.75 if weapon_detections else False):
            # Weapon alerts are often critical and might need faster triggering
            # The _weapon_required_frames is typically 1, meaning almost immediate alert
            self._weapon_alert_persistence = min(self._weapon_alert_persistence + 1, self._weapon_required_frames)
        else:
            self._weapon_alert_persistence = 0

        if fight_cond_active:
            self._fight_alert_persistence = min(self._fight_alert_persistence + 1, self._fight_required_frames)
        else:
            self._fight_alert_persistence = 0

        if stampede_condition:
            self._stampede_alert_persistence = min(self._stampede_alert_persistence + 1, self._stampede_required_frames)
        else:
            self._stampede_alert_persistence = 0

        if high_risk_cond:
            self._high_risk_alert_persistence = min(self._high_risk_alert_persistence + 1, self._high_risk_required_frames)
        else:
            self._high_risk_alert_persistence = 0

        # Overall alert condition met if any of the persistences have reached their required frames
        alert_conditions_met = (
            (self._weapon_alert_persistence >= self._weapon_required_frames) or
            (self._fight_alert_persistence >= self._fight_required_frames) or
            (self._stampede_alert_persistence >= self._stampede_required_frames) or
            (self._high_risk_alert_persistence >= self._high_risk_required_frames)
        )

        is_anomalous = alert_conditions_met
        is_confirmed = self._confirmation.update(is_anomalous)
        streak = self._confirmation.streak

        # ---- Should we fire an alert? ----
        should_alert = self._check_alert(
            is_confirmed, smoothed_risk_score, threat_level, threat_type, crowd_count, has_weapon
        )

        # ---- Store result ----
        result = {
            'risk_score':           smoothed_risk_score,
            'risk_score_pct':       f"{smoothed_risk_score:.1f}%",
            'threat_level':         threat_level,
            'threat_type':          threat_type,
            'motion_score':         motion_score,
            'accel_score':          combined_accel, # Renamed from accel_score to combined_accel for clarity
            'dir_variance':         dir_variance,
            'density_score':        legacy_density_score, # Keep for backward compatibility/display
            'speed_score':          speed_score,
            'crowd_count':          crowd_count,
            'confirmation_streak':  streak,
            'is_confirmed':         is_confirmed,
            'should_alert':         should_alert,
            'weapon_detected_raw':  weapon_detected_this_frame,
            'weapon_persistent':    (self.weapon_persistence_counter >= self.weapon_persistence_threshold),
            'weapon_cooldown_active': (self.weapon_cooldown_counter > 0),
            'weapon_persistence_frames': self.weapon_persistence_counter,
            'weapon_cooldown_frames_left': self.weapon_cooldown_counter,
            # New components for dashboard display if needed
            'crowd_density_component': crowd_density_score_component,
            'behavior_component': behavior_score_component,
            'weapon_component': weapon_score_component,
'historical_component': historical_events_score_component, # New component
             'fight_component': fight_score_component, # New component
             'fight_score': fight_score_component, # Redundant but kept for clarity
             'fit_score': fight_score_component, # Alias for fight_score
             'fight_cond': fight_cond_active, # Convenience alias
             'stampede_condition': stampede_condition, # Explicit stampede flag
             'is_stampede': stampede_condition # New field
        }
        self.last_analysis = result
        return result

    # ------------------------------------------------------------------
    # Stampede Detection Logic
    # ------------------------------------------------------------------
    def _detect_stampede(
        self,
        current_avg_speed: float,
        current_dir_variance: float,
        current_optical_flow_magnitude: float,
        current_crowd_count: int
    ) -> bool:
        """
        Detects stampede conditions based on multiple factors.
        Returns True if stampede conditions are met and persist.
        """
        if current_crowd_count < config.MIN_CROWD_FOR_PANIC:
            return False

        # 1. Sudden increase in average speed
        # Compare current speed to average over a short history (e.g., last 10 frames)
        if len(self.avg_speed_history) > config.FPS: # At least 1 second of history
            historical_avg_speed = np.mean(list(self.avg_speed_history)[-config.FPS:]) # Average over last 1 second
            # Speed increase factor (e.g., 1.5x faster than recent average)
            speed_increase_factor = current_avg_speed / max(historical_avg_speed, 0.1) # Avoid division by zero
            if speed_increase_factor < 1.5: # Speed hasn't increased significantly
                return False
        else:
            # Not enough history to determine sudden speed increase
            return False

        # 2. Same-direction crowd movement (low directional chaos)
        # A low dir_variance indicates consistent movement direction
        if current_dir_variance > config.DIR_CHAOS_THRESHOLD: # Use config threshold for "same direction"
            return False

        # 3. Large optical flow magnitude
        # Optical flow magnitude should be high, indicating overall rapid movement
        if current_optical_flow_magnitude < 10.0: # Threshold for "large magnitude" (pixels/frame)
            return False

        # 4. Rapid crowd density change (optional, but can indicate people rushing in/out)
        # For now, let's focus on speed, direction, and overall motion.
        # If all conditions are met, it's a potential stampede
        return True

    # ------------------------------------------------------------------
    # Fight detection helper
    # ------------------------------------------------------------------

    def _compute_fight_score(
        self,
        frame: np.ndarray,
        tracked_objects: Dict[int, Tuple[int, int]],
        frame_shape: Tuple[int, int, int], # (h, w, c)
        person_boxes: Optional[List[List[int]]] = None
    ) -> float:
        """
        Compute fight score based on pose estimation.
        Returns a value between 0.0 and 1.0.
        """
        if not self.pose_available or frame is None:
            return 0.0

        h, w = frame_shape[:2]
        
        if person_boxes is None:
            if not self.yolo_available or self.yolo_person is None:
                return 0.0 # Cannot detect persons without YOLO
            # Run YOLO person detection with improved small object detection
            try:
                # Ensure frame is in correct format (BGR) for YOLO if it expects it, or RGB
                # YOLO models typically expect RGB, but OpenCV reads BGR.
                # Assuming the input `frame` is BGR from OpenCV.
                yolo_input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.yolo_person(yolo_input_frame, classes=[0], conf=0.25, iou=0.45, augment=True, verbose=False)
            except Exception as e:
                logger.warning(f"YOLO person detection failed: {e}")
                return 0.0

            person_boxes = []
            for result in results:
                boxes = result.boxes
                for box in boxes: # Iterate through detected boxes
                    if int(box.cls[0]) == 0:  # person
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        person_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        # else person_boxes is already a list of [x1,y1,x2,y2] (ints)

        if not person_boxes:
            return 0.0

        # Associate detected persons with tracked objects by center distance
        associations = []  # list of (track_id, box)
        used_boxes_indices = set() # Keep track of which person_boxes have been associated
        
        for track_id, center in tracked_objects.items():
            min_dist = float('inf')
            best_box_idx = -1
            
            # Calculate centroid for the tracked object (center is already centroid)
            tracked_centroid = center

            for i, box in enumerate(person_boxes):
                if i in used_boxes_indices:
                    continue
                
                # Calculate centroid for the detected person box
                box_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                
                dist = np.sqrt((tracked_centroid[0] - box_centroid[0])**2 + (tracked_centroid[1] - box_centroid[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_box_idx = i
            
            # If a suitable box is found within a reasonable distance, associate it
            if best_box_idx != -1 and min_dist < 50:  # threshold of 50 pixels for association
                associations.append((track_id, person_boxes[best_box_idx]))
                used_boxes_indices.add(best_box_idx)

        max_fight_score = 0.0
        for track_id, box in associations:
            x1, y1, x2, y2 = box
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, w-1))
            x2 = max(0, min(x2, w-1))
            y1 = max(0, min(y1, h-1))
            y2 = max(0, min(y2, h-1))
            if x2 <= x1 or y2 <= y1:
                continue
            
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue
            
            # Convert BGR to RGB for MediaPipe
            try:
                image_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"Failed to convert person ROI to RGB for MediaPipe: {e}")
                continue
            
            # Run pose estimation
            try:
                results = self.pose.process(image_rgb)
            except Exception as e:
                logger.warning(f"MediaPipe pose processing failed: {e}")
                continue
            
            if not results.pose_landmarks:
                # No pose detected for this person
                if track_id in self.pose_history:
                    del self.pose_history[track_id]
                continue
            
            landmarkes = results.pose_landmarks.landmark
            
            # Initialize history for this track if not present
            if track_id not in self.pose_history:
                self.pose_history[track_id] = deque(maxlen=30)  # ~1 second at 30 fps
            
            self.pose_history[track_id].append(landmarkes)
            
            # Compute fight score for this person based on pose history
            person_fight_score = self._compute_person_fight_score_single(self.pose_history[track_id])
            if person_fight_score > max_fight_score:
                max_fight_score = person_fight_score
        
        return max_fight_score

    def _compute_person_fight_score_single(self, pose_history: deque) -> float:
        """
        Compute fight score for a single person based on pose history.
        Returns a value between 0.0 and 1.0.
        """
        if len(pose_history) < 2:
            return 0.0
        
        # Use the last two frames to compute velocities
        landmarks_prev = list(pose_history)[-2]
        landmarks_curr = list(pose_history)[-1]
        
        # Helper to get landmark by enum value
        def get_landmark(landmarks, enum_val):
            lm = landmarks[enum_val.value]
            return np.array([lm.x, lm.y])
        
        # Compute torso height (shoulder to hip midpoint) from previous frame for normalization
        left_shoulder_prev = get_landmark(landmarks_prev, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder_prev = get_landmark(landmarks_prev, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip_prev = get_landmark(landmarks_prev, self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip_prev = get_landmark(landmarks_prev, self.mp_pose.PoseLandmark.RIGHT_HIP)
        
        shoulder_prev = (left_shoulder_prev + right_shoulder_prev) / 2.0
        hip_prev = (left_hip_prev + right_hip_prev) / 2.0
        torso_height_prev = np.linalg.norm(shoulder_prev - hip_prev)
        
        if torso_height_prev < 1e-6:
            # Avoid division by zero
            torso_height_prev = 1e-6
        
        # Compute wrist speeds (left and right)
        left_wrist_prev = get_landmark(landmarks_prev, self.mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist_prev = get_landmark(landmarks_prev, self.mp_pose.PoseLandmark.RIGHT_WRIST)
        left_wrist_curr = get_landmark(landmarks_curr, self.mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist_curr = get_landmark(landmarks_curr, self.mp_pose.PoseLandmark.RIGHT_WRIST)
        
        wrist_speed_left = np.linalg.norm(left_wrist_curr - left_wrist_prev) / torso_height_prev
        wrist_speed_right = np.linalg.norm(right_wrist_curr - right_wrist_prev) / torso_height_prev
        max_wrist_speed = max(wrist_speed_left, wrist_speed_right)
        
        # Compute ankle speeds (left and right)
        left_ankle_prev = get_landmark(landmarks_prev, self.mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle_prev = get_landmark(landmarks_prev, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        left_ankle_curr = get_landmark(landmarks_curr, self.mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle_curr = get_landmark(landmarks_curr, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        ankle_speed_left = np.linalg.norm(left_ankle_curr - left_ankle_prev) / torso_height_prev
        ankle_speed_right = np.linalg.norm(right_ankle_curr - right_ankle_prev) / torso_height_prev
        max_ankle_speed = max(ankle_speed_left, ankle_speed_right)
        
        # Compute torso angle (falling detection) from current frame
        left_shoulder_curr = get_landmark(landmarks_curr, self.mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder_curr = get_landmark(landmarks_curr, self.mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_hip_curr = get_landmark(landmarks_curr, self.mp_pose.PoseLandmark.LEFT_HIP)
        right_hip_curr = get_landmark(landmarks_curr, self.mp_pose.PoseLandmark.RIGHT_HIP)
        
        shoulder_curr = (left_shoulder_curr + right_shoulder_curr) / 2.0
        hip_curr = (left_hip_curr + right_hip_curr) / 2.0
        torso_vector = hip_curr - shoulder_curr  # vector from shoulder to hip
        
        # Horizontal vector (1, 0)
        horizontal = np.array([1.0, 0.0])
        
        # Compute angle between torso vector and horizontal
        dot_product = np.dot(torso_vector, horizontal)
        norm_torso = np.linalg.norm(torso_vector)
        norm_horizontal = np.linalg.norm(horizontal)  # =1.0
        
        if norm_torso < 1e-6:
            angle = 0.0
        else:
            # Avoid numerical errors
            cos_angle = np.clip(dot_product / (norm_torso * norm_horizontal), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            # We want the acute angle (0 to pi/2) because we care about deviation from horizontal
            angle = min(angle, np.pi - angle)
        
        # Normalize angle to [0,1] where 0 is horizontal (lying down) and 1 is vertical (standing)
        # We want a high score when the person is lying down (angle near 0)
        fall_score = 1.0 - (angle / (np.pi / 2))  # when angle=0 -> score=1; angle=pi/2 -> score=0
        fall_score = max(0.0, min(fall_score, 1.0))  # Clamp to [0, 1]
        
        # Compute scores for punching and kicking
        punching_score = min(max_wrist_speed / self.WRIST_SPEED_THRESHOLD, 1.0)
        kicking_score = min(max_ankle_speed / self.ANKLE_SPEED_THRESHOLD, 1.0)
        
        # Combine: we take the maximum of the three components
        fight_score = max(punching_score, kicking_score, fall_score)

        return fight_score

    # ------------------------------------------------------------------
    # Legacy compatibility shim
    # ------------------------------------------------------------------

    def analyze_crowd_behavior(
        self,
        tracked_objects: Dict[int, Tuple[int, int]],
        frame_shape: Tuple
    ) -> Dict:
        """
        Backward-compatible wrapper used by old dashboard code.
        Returns the same dict keys the old code expected, plus the new ones.
        """
        # Create a dummy frame (black image) to maintain interface
        h, w = frame_shape[:2] if len(frame_shape) >= 2 else (480, 640)
        dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Build minimal motion_data stub so old callers don't break
        motion_stub = {
            'motion_score': 0.0, 'accel_score': 0.0, 'dir_variance': 0.0, 'motion_magnitude': 0.0, 'motion_direction': 0.0
        }
        result = self.analyze(dummy_frame, tracked_objects, frame_shape, motion_stub)
        # Map new keys to old keys for backward compat
        result['threat_level'] = self._map_level_old(result['threat_level'])
        result['confidence']   = result['risk_score'] / 100.0   # Map 0-100 risk_score back to 0-1 confidence
        return result

    @staticmethod
    def _map_level_old(level: str) -> str:
        mapping = {
            'CRITICAL':   'CRITICAL',
            'HIGH RISK':  'HIGH',
            'SUSPICIOUS': 'MEDIUM',
            'LOW RISK':   'LOW', # New mapping for LOW RISK
            'NORMAL':     'NONE',
        }
        return mapping.get(level, 'NONE')

    # ------------------------------------------------------------------
    # Alert gating
    # ------------------------------------------------------------------

    def should_trigger_alert(self, threat_analysis: Dict) -> bool:
        """Legacy method — returns pre-computed should_alert field."""
        return threat_analysis.get('should_alert', False)

    def _check_alert(
        self, is_confirmed: bool, risk_score: float, # Changed confidence to risk_score
        threat_level: str, threat_type: str, crowd_count: int,
        has_weapon: bool = False # New parameter
    ) -> bool:
        """
        Gate alert on:
          1. Threat confirmation window passed (from analyze method's persistence logic)
          2. Not in cooldown period
          3. Not a duplicate of the last alert
        """
        if not is_confirmed:
            return False

        now = time.time()
        # Use a longer cooldown for weapon alerts to prevent rapid re-alerts
        cooldown_period = self._cooldown
        if has_weapon:
            cooldown_period = getattr(config, 'WEAPON_COOLDOWN_SECONDS', self._cooldown)

        if now - self._last_alert_time < cooldown_period:
            return False

        # Duplicate check: hash of type + crowd bucket (and weapon status)
        crowd_bucket = crowd_count // 3
        alert_sig = f"{threat_type}|{crowd_bucket}|{has_weapon}"
        sig_hash = hashlib.md5(alert_sig.encode()).hexdigest()[:8]
        # Allow re-alerting if weapon status changes or after a longer period
        if sig_hash == self._last_alert_hash and (now - self._last_alert_time) < cooldown_period * 2:
            return False

        self._last_alert_time = now
        self._last_alert_hash = sig_hash
        return True

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def reset(self):
        self._confirmation.reset()
        self._last_alert_time = 0.0
        self._last_alert_hash = ""
        self.last_analysis = {}
        self.risk_score_history.clear()
        self.weapon_cooldown_counter = 0
        self.weapon_persistence_counter = 0
        # Clear pose history
        self.pose_history.clear()
        # Reset persistence counters for alert conditions
        self._weapon_alert_persistence = 0
        self._fight_alert_persistence = 0
        self._stampede_alert_persistence = 0
        self._high_risk_alert_persistence = 0
        # Clear crowd-level history
        self.crowd_density_history.clear()
        self.avg_speed_history.clear()
        self.optical_flow_magnitude_history.clear()
        self.dir_variance_history.clear()

    def get_threat_status(self) -> Dict:
        return {
            'threat_detected': self.last_analysis.get('is_confirmed', False),
            'threat_type':     self.last_analysis.get('threat_type', None),
            'risk_score':      self.last_analysis.get('risk_score', 0.0), # Changed confidence to risk_score
            'last_alert_time': self._last_alert_time,
            'weapon_detected_raw': self.last_analysis.get('weapon_detected_raw', False),
            'weapon_persistent': self.last_analysis.get('weapon_persistent', False),
            'weapon_cooldown_active': self.last_analysis.get('weapon_cooldown_active', False)
        }

# ---------------------------------------------------------------------------
# Module-level singleton + convenience wrappers
# ---------------------------------------------------------------------------
threat_detector = ThreatDetector()


def analyze_threat(
    frame: np.ndarray,
    tracked_objects: Dict[int, Tuple[int, int]],
    frame_shape: Tuple,
    motion_data: Dict = None,
    tracker_accel: Dict[int, float] = None,
    tracker_speeds: Dict[str, float] = None,
    weapon_detections: List[Dict] = None,
    person_boxes: List[List[int]] = None
) -> Dict:
    """Full analysis entry point — preferred over analyze_crowd_behavior."""
    if motion_data is None:
        motion_data = {'motion_score': 0.0, 'accel_score': 0.0, 'dir_variance': 0.0, 'motion_magnitude': 0.0}
    return threat_detector.analyze(
        frame,
        tracked_objects,
        frame_shape,
        motion_data,
        tracker_accel=tracker_accel,
        tracker_speeds=tracker_speeds,
        weapon_detections=weapon_detections,
        person_boxes=person_boxes
    )


# For backward compatibility
def analyze_crowd_behavior(
    tracked_objects: Dict[int, Tuple[int, int]],
    frame_shape: Tuple
) -> Dict:
    """Legacy wrapper — use analyze_threat if possible."""
    return threat_detector.analyze_crowd_behavior(tracked_objects, frame_shape)
