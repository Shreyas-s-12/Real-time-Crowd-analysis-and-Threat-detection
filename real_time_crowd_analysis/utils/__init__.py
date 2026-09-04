"""
Utilities Package for Real-Time Crowd Analysis and Threat Detection
"""

# Import all utility modules for easy access
from real_time_crowd_analysis.utils.logger import setup_logger
from real_time_crowd_analysis.utils.database import DatabaseManager
from .alerts import trigger_alert, play_alert_sound, trigger_visual_alert
from .csv_manager import CSVManager
from .config import config
from .network_utils import NetworkValidator
from .helpers import (
    calculate_distance,
    calculate_speed,
    draw_text_with_background,
    resize_frame,
    safe_divide,
    format_timestamp,
    validate_coordinates
)

# Export all classes and functions
__all__ = [
    'setup_logger',
    'DatabaseManager',
    'trigger_alert',
    'play_alert_sound',
    'trigger_visual_alert',
    'CSVManager',
    'config',
    'NetworkValidator',
    'calculate_distance',
    'calculate_speed',
    'draw_text_with_background',
    'resize_frame',
    'safe_divide',
    'format_timestamp',
    'validate_coordinates'
]