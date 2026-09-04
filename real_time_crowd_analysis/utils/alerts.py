"""
Alert System for Real-Time Crowd Analysis and Threat Detection
"""

import os
import threading
import time # Keep time import
from datetime import datetime # Keep datetime import
import cv2 # Keep cv2 import
from real_time_crowd_analysis.utils.config import config # Absolute import
from real_time_crowd_analysis.utils.logger import setup_logger # Absolute import

logger = setup_logger("alerts")

class AlertManager:
    """Manages alert generation and playback"""
    
    def __init__(self):
        self.alert_sound_path = config.ALERT_SOUND_PATH
        self.is_alert_playing = False
        self.alert_thread = None
        
        # Ensure alert sound exists or create a simple beep
        self.ensure_alert_sound()
    
    def ensure_alert_sound(self):
        """Ensure alert sound file exists"""
        alert_dir = os.path.dirname(self.alert_sound_path)
        if alert_dir and not os.path.exists(alert_dir):
            os.makedirs(alert_dir, exist_ok=True)
        
        # If alert sound doesn't exist, we'll use system beep or create a simple one
        if not os.path.exists(self.alert_sound_path):
            logger.warning(f"Alert sound not found at {self.alert_sound_path}")
            # We'll handle this by using system beep in play_alert_sound
    
    def play_alert_sound(self):
        """Play alert sound in a separate thread"""
        if self.is_alert_playing:
            return
        
        def _play_sound():
            self.is_alert_playing = True
            try:
                # Try to play the alert sound file
                if os.path.exists(self.alert_sound_path):
                    # For Windows, we can use winsound
                    try:
                        import winsound
                        winsound.PlaySound(self.alert_sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    except ImportError:
                        # Fallback to system beep
                        print('\a')  # System beep
                else:
                    # System beep as fallback
                    print('\a')  # System beep
                    
                # Keep alert playing for a short duration
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error playing alert sound: {e}")
                # Fallback to system beep
                print('\a')
            finally:
                self.is_alert_playing = False
        
        self.alert_thread = threading.Thread(target=_play_sound, daemon=True)
        self.alert_thread.start()
    
    def trigger_visual_alert(self, frame, threat_detected: bool = False, message: str = "WARNING: POSSIBLE THREAT DETECTED"):
        """Deprecated: UI handles visual alerts directly in dashboard.py to prevent overlapping text."""
        return frame
    
    def trigger_alert(self, threat_type: str = "UNKNOWN", confidence: float = 0.0):
        """Trigger complete alert sequence"""
        logger.warning(f"ALERT TRIGGERED: {threat_type} (confidence: {confidence:.2f})")
        
        # Play sound
        self.play_alert_sound()
        
        # Visual alert will be handled by the calling function
        return True
    
    def stop_alert(self):
        """Stop any ongoing alert"""
        self.is_alert_playing = False
        if self.alert_thread and self.alert_thread.is_alive():
            # Thread will finish naturally
            pass

# Global alert manager instance
alert_manager = AlertManager()

# Convenience functions
def trigger_alert(threat_type: str = "UNKNOWN", confidence: float = 0.0):
    """Trigger an alert"""
    return alert_manager.trigger_alert(threat_type, confidence)

def play_alert_sound():
    """Play alert sound"""
    alert_manager.play_alert_sound()

def trigger_visual_alert(frame, threat_detected: bool = False, message: str = "WARNING: POSSIBLE THREAT DETECTED"):
    """Add visual alert to frame"""
    return alert_manager.trigger_visual_alert(frame, threat_detected, message)