"""
Configuration Management for Real-Time Crowd Analysis and Threat Detection
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Main configuration class"""
    
    # Application
    APP_NAME: str = "Real-Time Crowd Analysis and Threat Detection"
    VERSION: str = "1.0.0"
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASETS_DIR: str = os.path.join(BASE_DIR, "datasets")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    RECORDINGS_DIR: str = os.path.join(DATASETS_DIR, "recordings")
    SCREENSHOTS_DIR: str = os.path.join(BASE_DIR, "screenshots")
    
    # Database
    DB_PATH: str = os.path.join(BASE_DIR, "database", "analytics.db")
    
    # YOLO Model
    MODEL_NAME: str = "yolov8m.pt"
    YOLO_MODEL: str = "yolov8m.pt"  # medium version for higher accuracy
    YOLO_MODEL_FALLBACK: str = "yolov8n.pt" # nano version for speed/compatibility
    YOLO_PERSON_CONFIDENCE: float = 0.35 # Confidence threshold for person detections (balanced for real-world webcam)
    YOLO_WEAPON_CONFIDENCE: float = 0.25 # Lower confidence threshold for weapon detections
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_IMG_SIZE: int = 960 # Higher resolution for small object detection
    YOLO_IMG_SIZE_FALLBACK: int = 640 # Standard resolution fallback
    YOLO_FPS_LOW_THRESHOLD: float = 15.0 # Threshold for resolution fallback or frame skipping
    
    # Detection
    CROWD_HIGH_THRESHOLD: int = 20
    CROWD_MEDIUM_THRESHOLD: int = 10
    CROWD_LOW_THRESHOLD: int = 5
    
    # Threat Detection
    THREAT_MOTION_THRESHELD: float = 0.5
    THREAT_DENSITY_THRESHOLD: int = 30
    THREAT_SPEED_THRESHOLD: float = 15.0
    PANIC_DISTANCE_THRESHOLD: float = 50.0

    WEAPON_PERSISTENCE_FRAMES: int = 3 # Number of consecutive frames a weapon must be seen to confirm
    WEAPON_THREAT_CONFIDENCE: float = 0.85 # Base confidence when a weapon is detected
    WEAPON_COOLDOWN_SECONDS: float = 10.0 # How long to keep weapon threat active after disappearance
    # Behavioral Analysis (True Real-Time)
    MOTION_HISTORY_SIZE: int = 30               # rolling buffer length (frames)
    THREAT_CONFIRMATION_FRAMES: int = 12        # consecutive anomalous frames before alert
    ALERT_COOLDOWN_SECONDS: float = 8.0         # min seconds between identical alerts
    MIN_CROWD_FOR_PANIC: int = 3                # crowd panic analysis requires at least N persons
    CONFIDENCE_THRESHOLD: float = 0.35          # minimum confidence to escalate threat level
    OPTICAL_FLOW_SCALE: float = 0.5             # resize factor applied before Farneback flow
    ACCEL_SPIKE_THRESHOLD: float = 2.0          # pixels/frame² — marks acceleration spike
    DIR_CHAOS_THRESHOLD: float = 0.65           # circular variance (0–1) marking directional chaos
    LOW_LIGHT_THRESHOLD: float = 45.0           # mean brightness below which low-light mode activates
    LOW_LIGHT_CLAHE_LIMIT: float = 3.0          # contrast limit for CLAHE enhancement
    
    # New Alert Persistence Frames (at 30 FPS)
    WEAPON_ALERT_PERSISTENCE_FRAMES: int = 1    # Alert immediately for weapon (or after 1 frame)
    FIGHT_ALERT_PERSISTENCE_FRAMES: int = 60    # 2 seconds for fight detection
    STAMPEDE_ALERT_PERSISTENCE_FRAMES: int = 90 # 3 seconds for stampede detection
    HIGH_RISK_ALERT_PERSISTENCE_FRAMES: int = 90 # 3 seconds for general high risk
    # Face Detection
    FACE_CONFIDENCE: float = 0.6
    
    # Network
    NETWORK_TIMEOUT: int = 5
    RECONNECT_ATTEMPTS: int = 3
    RECONNECT_DELAY: int = 2
    
    # Video
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    FPS: int = 30
    
    # UI
    UI_THEME: str = "dark"
    PRIMARY_COLOR: str = "#00A0E9"  # Blue
    WARNING_COLOR: str = "#FF0000"  # Red
    SUCCESS_COLOR: str = "#00FF00"  # Green
    
    # Analytics Graphs
    GRAPH_BG_COLOR: str = "#151515"
    GRAPH_THREAT_COLOR: str = "#FF00FF"
    GRAPH_CONFIDENCE_COLOR: str = "#00FFFF"
    GRAPH_MOTION_COLOR: str = "#00FF00"
    
    # Audio
    ALERT_SOUND_PATH: str = os.path.join(BASE_DIR, "assets", "alert.wav")
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = os.path.join(DATASETS_DIR, "app.log")
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        dirs = [
            cls.DATASETS_DIR,
            cls.RECORDINGS_DIR,
            cls.SCREENSHOTS_DIR,
            os.path.dirname(cls.DB_PATH), # Ensure database directory exists
            os.path.dirname(cls.LOG_FILE) # Ensure log directory exists
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)

# Global config instance
config = Config()