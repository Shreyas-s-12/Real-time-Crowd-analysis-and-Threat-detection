"""
Models Package for Real-Time Crowd Analysis and Threat Detection
"""

# Import all model classes for easy access
from real_time_crowd_analysis.models.detector import PersonDetector, detect_persons, person_detector
from real_time_crowd_analysis.models.tracker import PersonTracker, track_persons
from real_time_crowd_analysis.models.threat_detector import ThreatDetector, analyze_threat
from real_time_crowd_analysis.models.motion_analyzer import MotionAnalyzer, analyze_motion
from real_time_crowd_analysis.models.face_detection import FaceDetector, detect_faces
from real_time_crowd_analysis.models.network_validator import NetworkValidator, validate_network

# Export all classes and functions
__all__ = [
    'PersonDetector',
    'detect_persons',
    'person_detector',
    'PersonTracker', 
    'track_persons',
    'ThreatDetector',
    'analyze_threat',
    'MotionAnalyzer',
    'analyze_motion',
    'FaceDetector',
    'detect_faces',
    'NetworkValidator',
    'validate_network'
]
