"""
Camera Panel Module for Real-Time Crowd Analysis and Threat Detection
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QHBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
import numpy as np # Keep numpy import
import traceback # Keep traceback import
from real_time_crowd_analysis.ui.theme import get_stylesheet # Absolute import
from real_time_crowd_analysis.utils.logger import setup_logger # Absolute import

logger = setup_logger("camera_panel")


class CameraPanel(QWidget):
    """Panel for displaying live camera feed with overlays"""

    # Signals
    frame_updated = pyqtSignal(np.ndarray)

    def __init__(self):
        logger.info("Initializing camera panel")
        super().__init__()
        self.current_frame = None
        
        # Setup UI
        self.setup_ui()
        
        # Timer for updating display
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start(30)  # Update display ~33 FPS
        
        logger.info("Camera panel initialized")

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title bar
        title_layout = QHBoxLayout()
        
        title_label = QLabel("Live Camera Feed")
        title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #00FFFF;
        """)
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        # Status indicators
        self.status_label = QLabel("Status: Stopped")
        self.status_label.setStyleSheet("""
            font-size: 12px;
            color: #FFA500;
        """)
        title_layout.addWidget(self.status_label)
        
        layout.addLayout(title_layout)
        
        # Video display frame
        self.video_frame = QFrame()
        self.video_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        self.video_frame.setLineWidth(2)
        self.video_frame.setMinimumSize(320, 240)
        self.video_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        video_layout = QVBoxLayout(self.video_frame)
        video_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Placeholder label for when no video
        self.placeholder_label = QLabel("No Camera Feed Available\n\nConnect a camera to begin monitoring")
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_label.setStyleSheet("""
            font-size: 14px;
            color: #888888;
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            padding: 20px;
        """)
        self.placeholder_label.setWordWrap(True)
        self.placeholder_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        video_layout.addWidget(self.placeholder_label)
        
        # Actual video label (hidden by default)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setScaledContents(False)
        self.video_label.setMinimumSize(1, 1)
        self.video_label.hide()
        
        video_layout.addWidget(self.video_label)
        
        layout.addWidget(self.video_frame)
        
        # Info overlay
        info_layout = QHBoxLayout()
        
        self.resolution_label = QLabel("Resolution: --")
        self.resolution_label.setStyleSheet("font-size: 11px; color: #CCCCCC;")
        info_layout.addWidget(self.resolution_label)
        
        info_layout.addStretch()
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-size: 11px; color: #CCCCCC;")
        info_layout.addWidget(self.fps_label)
        
        layout.addLayout(info_layout)

    def update_frame(self, frame: np.ndarray):
        """Update the panel with a new frame"""
        try:
            if frame is not None and frame.size > 0:
                self.current_frame = frame.copy()
                height, width = frame.shape[:2]
                self.resolution_label.setText(f"Resolution: {width}x{height}")
                self.status_label.setText("Status: Running")
                self.status_label.setStyleSheet("""
                    font-size: 12px;
                    color: #00FF00;
                """)
            else:
                self.current_frame = None
                self.resolution_label.setText("Resolution: --")
                self.status_label.setText("Status: No Frame")
                self.status_label.setStyleSheet("""
                    font-size: 12px;
                    color: #FFA500;
                """)
                
        except Exception as e:
            logger.error(f"Error updating frame: {e}")
            traceback.print_exc()

    def update_display(self):
        """Update the displayed image"""
        try:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
                
                # Determine target dimensions from parent frame container
                container_w = self.video_frame.width()
                container_h = self.video_frame.height()
                
                if container_w < 10 or container_h < 10:
                    return

                # Ensure video label is visible and placeholder is hidden
                if not self.video_label.isVisible():
                    self.placeholder_label.hide()
                    self.video_label.show()

                target_w = max(self.video_label.width(), container_w - 10)
                target_h = max(self.video_label.height(), container_h - 10)

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get dimensions
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                
                # Convert to QImage with explicit memory copy for thread safety
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                
                # Scale to fit display while maintaining aspect ratio
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    target_w, 
                    target_h, 
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                # Update display
                self.video_label.setPixmap(scaled_pixmap)
                
            else:
                # Show placeholder, hide video
                if self.video_label.isVisible():
                    self.video_label.hide()
                    self.placeholder_label.show()
                    
        except Exception as e:
            logger.error(f"Error updating display: {e}")
            traceback.print_exc()
            # Show error state
            self.placeholder_label.setText("Camera Error\n\nCheck camera connection")
            self.placeholder_label.show()
            self.video_label.hide()

    def set_source(self, source_type: str, source_info: dict = None):
        """Set the camera source type (for display purposes)"""
        source_names = {
            'webcam': 'Webcam',
            'rtsp': 'RTSP Stream',
            'wired': 'Wired Camera',
            'wireless': 'Wireless Camera',
            'mobile': 'Mobile Camera',
            'file': 'Video File'
        }
        name = source_names.get(source_type, source_type)
        # Could update title or add indicator here
        logger.info(f"Camera source set to: {name}")

    def clear_display(self):
        """Clear the display and show placeholder"""
        self.current_frame = None
        self.video_label.clear()
        self.video_label.hide()
        self.placeholder_label.setText("No Camera Feed Available\n\nConnect a camera to begin monitoring")
        self.placeholder_label.show()
        self.status_label.setText("Status: Stopped")
        self.status_label.setStyleSheet("""
            font-size: 12px;
            color: #FFA500;
        """)
        self.resolution_label.setText("Resolution: --")
        self.fps_label.setText("FPS: --")

# Import cv2 at the end to avoid circular imports
import cv2