"""
UI Package for Real-Time Crowd Analysis and Threat Detection
"""

# Import all UI components for easy access
from real_time_crowd_analysis.ui.dashboard import DashboardWindow
from real_time_crowd_analysis.ui.alerts_panel import AlertsPanel
from real_time_crowd_analysis.ui.camera_panel import CameraPanel
from real_time_crowd_analysis.ui.splash_screen import show_splash_screen, SplashScreen
from real_time_crowd_analysis.ui.theme import get_stylesheet, get_theme_color, set_theme

# Export all classes and functions
__all__ = [
    'DashboardWindow',
    'AlertsPanel',
    'CameraPanel',
    'show_splash_screen',
    'SplashScreen',
    'get_stylesheet',
    'get_theme_color',
    'set_theme'
]