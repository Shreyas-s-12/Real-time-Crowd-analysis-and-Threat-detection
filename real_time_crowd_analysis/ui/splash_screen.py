"""
Splash Screen Module for Real-Time Crowd Analysis and Threat Detection
"""

import sys
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QLabel, QProgressBar, QFrame) # Keep QtWidgets imports
from PyQt6.QtCore import Qt, QTimer # Keep QtCore imports
from PyQt6.QtGui import QFont, QLinearGradient # Keep QtGui imports
from ..utils.config import config
from ..utils.logger import setup_logger
from ..ui.theme import get_stylesheet

logger = setup_logger("splash_screen")

class SplashScreen(QMainWindow):
    """Animated splash screen for the application"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.setFixedSize(800, 600)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | 
                           Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Center the window
        self.center_window()
        
        # Setup UI
        self.setup_ui()
        
        # Animation variables
        self.progress_value = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        
        logger.info("Splash screen initialized")
    
    def center_window(self):
        """Center the window on screen"""
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
    
    def setup_ui(self):
        """Setup the splash screen UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        # Title label
        title_label = QLabel(config.APP_NAME)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("Arial", 24, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"""
            color: {config.PRIMARY_COLOR};
            margin-bottom: 20px;
        """)
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("AI-Powered Crowd Intelligence System")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_font = QFont("Arial", 14)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setStyleSheet(f"""
            color: #B0B0B0;
            margin-bottom: 30px;
        """)
        layout.addWidget(subtitle_label)
        
        # Progress bar container
        progress_frame = QFrame()
        progress_frame.setFixedHeight(30)
        progress_frame.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 15px;
            }}
        """)
        
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(5, 5, 5, 5)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: rgba(0, 0, 0, 0.2);
                border-radius: 10px;
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {config.PRIMARY_COLOR},
                    stop:1 #00FFFF
                );
                border-radius: 10px;
            }}
        """)
        
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_frame)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(f"""
            color: #B0B0B0;
            font-size: 12px;
            margin-top: 10px;
        """)
        layout.addWidget(self.status_label)
        
        # Version label
        version_label = QLabel(f"Version {config.VERSION}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet(f"""
            color: #666666;
            font-size: 10px;
            margin-top: 20px;
        """)
        layout.addWidget(version_label)
        
        # Apply theme
        central_widget.setStyleSheet(get_stylesheet())
    
    def start_animation(self):
        """Start the splash screen animation"""
        self.progress_value = 0
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing...")
        self.animation_timer.start(30)  # Update every 30ms
        logger.info("Splash screen animation started")
    
    def update_animation(self):
        """Update the animation progress"""
        self.progress_value += 1
        
        if self.progress_value <= 100:
            self.progress_bar.setValue(self.progress_value)
            
            # Update status text based on progress
            if self.progress_value < 20:
                self.status_label.setText("Loading configuration...")
            elif self.progress_value < 40:
                self.status_label.setText("Initializing AI models...")
            elif self.progress_value < 60:
                self.status_label.setText("Setting up camera interfaces...")
            elif self.progress_value < 80:
                self.status_label.setText("Preparing user interface...")
            else:
                self.status_label.setText("Almost ready...")
        else:
            # Animation complete
            self.animation_timer.stop()
            logger.info("Splash screen animation completed")
    
    def close_splash(self):
        """Close the splash screen"""
        self.animation_timer.stop()
        self.close()
        logger.info("Splash screen closed")

# Convenience function
def show_splash_screen() -> SplashScreen:
    """Create and show splash screen"""
    splash = SplashScreen()
    splash.show()
    splash.start_animation()
    return splash