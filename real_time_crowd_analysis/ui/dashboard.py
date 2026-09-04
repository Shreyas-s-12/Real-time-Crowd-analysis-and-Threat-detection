"""
Dashboard Module for Real-Time Crowd Analysis and Threat Detection
Wires MotionAnalyzer + PersonTracker + ThreatDetector into the UI.
"""

import traceback
import math
from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QScrollArea, QGridLayout, QProgressBar,
    QTabWidget, QLabel, QFrame, QSplitter, QCheckBox, QPushButton, QLineEdit, QComboBox, QStackedWidget, QFormLayout, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import time
from collections import deque
import numpy as np
import cv2 # Keep cv2 import here for _draw_overlays
from ..models.motion_analyzer import motion_analyzer

from ..utils.config import config

# Task: Safely handle pyqtgraph dependency
try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
    pg.setConfigOptions(antialias=False)
except Exception as e:
    print(f"PyQtGraph unavailable: {e}")
    PYQTGRAPH_AVAILABLE = False

from real_time_crowd_analysis.ui.theme import get_stylesheet
from real_time_crowd_analysis.ui.alerts_panel import AlertsPanel
from real_time_crowd_analysis.ui.camera_panel import CameraPanel
from real_time_crowd_analysis.ui.history_panel import HistoryPanel
from real_time_crowd_analysis.ui.camera_manager import CameraManagerPanel

from real_time_crowd_analysis.utils.logger import setup_logger
from real_time_crowd_analysis.models.detector import person_detector, detect_persons
from real_time_crowd_analysis.models.tracker import track_persons, person_tracker
from real_time_crowd_analysis.models.motion_analyzer import motion_analyzer
from real_time_crowd_analysis.models.threat_detector import threat_detector, analyze_threat
from real_time_crowd_analysis.utils.alerts import trigger_visual_alert
from real_time_crowd_analysis.input_sources.webcam import WebcamSource
from real_time_crowd_analysis.input_sources.mobile_camera import MobileCameraSource
from real_time_crowd_analysis.utils.database import db_manager
from real_time_crowd_analysis.utils.csv_manager import csv_manager

logger = setup_logger("dashboard")

# -----------------------------------------------------------------------
# Colour helpers
# -----------------------------------------------------------------------
_LEVEL_COLORS = {
    'NORMAL':    '#00FF00',
    'SUSPICIOUS':'#FFFF00',
    'HIGH RISK': '#FF6600',
    'CRITICAL':  '#FF00FF',
    'WEAPON':    '#FF0000', # Dedicated color for weapon alerts
    # legacy compat
    'NONE':      '#00FF00',
    'LOW':       '#FFFF00',
    'MEDIUM':    '#FFA500',
    'HIGH':      '#FF0000',
}


class DashboardWindow(QMainWindow):
    def __init__(self):
        """Main dashboard window for the application"""
        try:
            logger.info("Loading dashboard")
            super().__init__()
            self.setWindowTitle("Real-Time Crowd Analysis and Threat Detection")
            self.setGeometry(100, 100, 1400, 900)
            self.setStyleSheet(get_stylesheet())

            self.alert_overlay_active = False # Flag for visual alert overlay
            self.is_fullscreen = False # Track fullscreen state
            # Camera source
            self.camera_source = None

            # Task 5: Temporal Smoothing State (EMA)
            self.ema_alpha = 0.3 # Smoothing factor (Task 5)
            self.telemetry_smoothed = {
                "threat_score": 0.0, 
                "motion_score": 0.0,
                "accel_score": 0.0, # This is combined_accel from threat_detector
                "risk_score": 0.0, # Changed from confidence to risk_score
                "confidence": 0.0 # Added for compatibility
            }

            # FPS tracking
            # self._fps_times: list = [] # Not used, can be removed or used for actual FPS calculation (commented out as per previous diff)
            self._last_fps: float = 0.0

            # Debug overlay toggle
            self._show_debug: bool = False

            # Processing timer
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.is_monitoring = False
            
            # Analytics Telemetry & History (Tasks 1, 8, 9)
            self.telemetry = {
                "threat_score": 0.0, "motion_score": 0.0, "accel_score": 0.0, # accel_score is combined_accel
                "confidence": 0.0, "risk_score": 0.0, "fps": 0.0, "person_count": 0, # confidence kept for compatibility
                "weapon_score": 0.0, "dir_variance": 0.0 # Changed weapon_confidence to weapon_score
            }
            self.telemetry_history = {
                'threat': deque(maxlen=100), 'motion': deque(maxlen=100),
                'accel': deque(maxlen=100), 'risk': deque(maxlen=100), # Changed 'conf' to 'risk'
                'fps': deque(maxlen=100), 'count': deque(maxlen=100)
            }
            self.analytics_timer = QTimer(self)
            self.analytics_timer.timeout.connect(self._refresh_analytics_ui)
            self.analytics_timer.start(350) # Update every 350ms to prevent UI freezing (Task 7)

            self.motion_history = deque(maxlen=100)
            self.risk_history = deque(maxlen=100) # Changed conf_history to risk_history
            self.threat_history = deque(maxlen=100)

            self.graph_timer = QTimer(self)
            self.graph_timer.timeout.connect(self.update_graphs)
            self.graph_timer.start(350)

            # Monitoring state for change detection
            self.prev_crowd_count = 0
            self.prev_threat_level = "NORMAL"
            self.monitoring_location = "Unknown"
            self.monitoring_area = "Unknown"
            self.monitoring_camera_id = "0"
            self.last_move_log_time = 0
            self.current_network = "Local" # Default network name
            self.current_camera_type = "Webcam" # Default camera type
            self.current_ip = "127.0.0.1" # Default IP address
            self._monitoring_start_time = 0.0 # To track session duration
            self.last_analytics_log_time = 0
            self.analytics_history = []
            self._prev_mobile_status = "DISCONNECTED"

            # Build UI
            try:
                self.setup_ui()
            except Exception as e:
                logger.error(f"UI setup failed: {e}")

            # Start camera after UI loads to ensure stability
            # QTimer.singleShot(500, self.start_camera)

            try:
                self.showMaximized()
            except Exception as e:
                logger.warning(f"Initial window maximize failed: {e}")

            logger.info("Dashboard loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            logger.error(traceback.format_exc())
            raise

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def keyPressEvent(self, event):
        """Handle global keyboard shortcuts like F11 for Fullscreen"""
        try:
            if event.key() == Qt.Key.Key_F11:
                self.toggle_fullscreen()
            else:
                super().keyPressEvent(event)
        except Exception as e:
            logger.error(f"Error handling key press: {e}")

    def toggle_fullscreen(self):
        """Toggle between fullscreen and normal window state safely"""
        try:
            if not self.is_fullscreen:
                self.showFullScreen()
                self.is_fullscreen = True
                self.statusBar().showMessage("Fullscreen Mode Enabled (F11 to exit)", 3000)
                logger.info("Dashboard entered fullscreen mode")
            else:
                self.showNormal()
                self.is_fullscreen = False
                self.statusBar().showMessage("Normal Window Mode", 3000)
                logger.info("Dashboard exited fullscreen mode")
        except Exception as e:
            logger.error(f"Fullscreen transition failure: {e}")

    def setup_ui(self):
        """Setup the user interface"""
        if self.centralWidget() is not None:
            return
            
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Root layout for sidebar + main content
        self.root_layout = QHBoxLayout(central_widget)
        self.root_layout.setSpacing(0)
        self.root_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Left Sidebar
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet("background-color: #121212; border-right: 1px solid #333;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(15)

        logo_label = QLabel("AI SURVEILLANCE\nCOMMAND CENTER")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("color: #00FFFF; font-size: 16px; font-weight: bold; border: none; margin-bottom: 20px;")
        sidebar_layout.addWidget(logo_label)

        def create_nav_btn(text):
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent; color: #CCCCCC; text-align: left;
                    padding: 12px 15px; border: none; border-radius: 4px; font-weight: bold; font-size: 14px;
                }
                QPushButton:hover { background-color: rgba(255, 255, 255, 0.1); color: #FFFFFF; }
                QPushButton:checked { background-color: rgba(0, 255, 255, 0.2); color: #00FFFF; }
            """)
            btn.setCheckable(True)
            return btn

        self.btn_nav_dashboard = create_nav_btn("DASHBOARD")
        self.btn_nav_cameras = create_nav_btn("CAMERA SOURCES")
        self.btn_nav_history = create_nav_btn("HISTORY")
        
        from PyQt6.QtWidgets import QButtonGroup
        self.nav_group = QButtonGroup(self)
        self.nav_group.addButton(self.btn_nav_dashboard, 0)
        self.nav_group.addButton(self.btn_nav_cameras, 1)
        self.nav_group.addButton(self.btn_nav_history, 2)
        
        self.btn_nav_dashboard.setChecked(True)
        self.nav_group.idClicked.connect(lambda id: self.main_stack.setCurrentIndex(id))
        
        sidebar_layout.addWidget(self.btn_nav_dashboard)
        sidebar_layout.addWidget(self.btn_nav_cameras)
        sidebar_layout.addWidget(self.btn_nav_history)
        sidebar_layout.addStretch()

        self.root_layout.addWidget(self.sidebar)


        # 2. Main Content Stack
        self.main_stack = QStackedWidget()
        self.root_layout.addWidget(self.main_stack, 1)

        # --- PAGE 0: DASHBOARD ---
        self.dashboard_page = QWidget()
        self.main_stack.addWidget(self.dashboard_page)
        
        main_layout = QVBoxLayout(self.dashboard_page)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Title bar
        title_row = QHBoxLayout()
        title_label = QLabel("Real-Time Crowd Analysis and Threat Detection")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #00FFFF; margin: 4px;"
        )
        title_row.addWidget(title_label)
        title_row.addStretch()
        
        # Camera source dropdown
        camera_source_label = QLabel("Camera Source:")
        camera_source_label.setStyleSheet("color: #AAAAAA; font-size: 12px;")
        title_row.addWidget(camera_source_label)
        
        self.camera_source_combo = QComboBox()
        self.camera_source_combo.addItems(["Webcam", "Wired Camera", "Wireless Camera", "RTSP Camera", "Mobile Camera"])
        self.camera_source_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 120px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #E0E0E0;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(0, 0, 0, 0.8);
                color: #E0E0E0;
                selection-background-color: #00AA00;
                border: 1px solid #333;
            }
        """)
        self.camera_source_combo.currentTextChanged.connect(self._on_camera_source_changed)
        title_row.addWidget(self.camera_source_combo)

        # Debug overlay checkbox
        self.debug_cb = QCheckBox("Debug Overlay")
        self.debug_cb.setStyleSheet("color: #AAAAAA; font-size: 12px;")
        self.debug_cb.stateChanged.connect(self._toggle_debug)
        title_row.addWidget(self.debug_cb)

        # Night mode manual override toggle
        ll_mode_label = QLabel("Night Mode:")
        ll_mode_label.setStyleSheet("color: #AAAAAA; font-size: 12px; margin-left: 10px;")
        title_row.addWidget(ll_mode_label)

        self.low_light_combo = QComboBox()
        self.low_light_combo.addItems(["Auto", "Force On", "Force Off"])
        self.low_light_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #E0E0E0;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: rgba(0, 0, 0, 0.8);
                color: #E0E0E0;
                selection-background-color: #00AA00;
                border: 1px solid #333;
            }
        """)
        self.low_light_combo.currentTextChanged.connect(self._on_low_light_mode_changed)
        title_row.addWidget(self.low_light_combo)

        # Low light indicator
        self.low_light_label = QLabel("🌙 LOW LIGHT MODE")
        self.low_light_label.setStyleSheet("color: #FFA500; font-weight: bold; font-size: 13px; margin-left: 15px;")
        self.low_light_label.hide()
        title_row.addWidget(self.low_light_label)

        main_layout.addLayout(title_row)

        # Camera configuration panels
        self.camera_config_stack = QStackedWidget()
        self.camera_config_stack.setStyleSheet("""
            QStackedWidget {
                background-color: rgba(0, 0, 0, 0.1);
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        self.camera_config_stack.setMaximumHeight(180)
        
        # Webcam configuration panel
        webcam_panel = QWidget()
        webcam_layout = QFormLayout(webcam_panel)
        webcam_layout.setContentsMargins(10, 5, 10, 5)
        webcam_layout.setSpacing(5)
        
        self.webcam_index_input = QLineEdit()
        self.webcam_index_input.setPlaceholderText("0 (default)")
        self.webcam_index_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        webcam_layout.addRow("Camera Index:", self.webcam_index_input)

        # Status labels for webcam
        self.webcam_status_label = QLabel("Status: Stopped")
        self.webcam_status_label.setStyleSheet("color: #AAAAAA;")
        webcam_layout.addRow(self.webcam_status_label)

        self.webcam_resolution_label = QLabel("Resolution: --")
        self.webcam_resolution_label.setStyleSheet("color: #AAAAAA;")
        webcam_layout.addRow(self.webcam_resolution_label)

        self.webcam_fps_label = QLabel("FPS: --")
        self.webcam_fps_label.setStyleSheet("color: #AAAAAA;")
        webcam_layout.addRow(self.webcam_fps_label)
        
        self.connect_webcam_btn = QPushButton("Connect Webcam")
        self.connect_webcam_btn.setStyleSheet("""
            QPushButton {
                background-color: #0066CC;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0077DD;
            }
            QPushButton:pressed {
                background-color: #0055AA;
            }
        """)
        self.connect_webcam_btn.clicked.connect(self._on_connect_webcam)
        webcam_layout.addRow("", self.connect_webcam_btn)
        
        self.camera_config_stack.addWidget(webcam_panel)
        
        # Wired camera configuration panel
        wired_panel = QWidget()
        wired_layout = QFormLayout(wired_panel)
        wired_layout.setContentsMargins(10, 5, 10, 5)
        wired_layout.setSpacing(5)
        
        self.wired_ip_input = QLineEdit()
        self.wired_ip_input.setPlaceholderText("192.168.1.100")
        self.wired_ip_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wired_layout.addRow("Camera IP Address:", self.wired_ip_input)
        
        self.wired_port_input = QLineEdit()
        self.wired_port_input.setPlaceholderText("80")
        self.wired_port_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wired_layout.addRow("Port:", self.wired_port_input)
        
        self.wired_username_input = QLineEdit()
        self.wired_username_input.setPlaceholderText("admin")
        self.wired_username_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wired_layout.addRow("Username:", self.wired_username_input)
        
        self.wired_password_input = QLineEdit()
        self.wired_password_input.setPlaceholderText("••••••")
        self.wired_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.wired_password_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wired_layout.addRow("Password:", self.wired_password_input)
        
        self.wired_rtsp_url_input = QLineEdit()
        self.wired_rtsp_url_input.setPlaceholderText("rtsp://192.168.1.100/live")
        self.wired_rtsp_url_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wired_layout.addRow("RTSP URL:", self.wired_rtsp_url_input)
        
        self.wired_camera_id_input = QLineEdit()
        self.wired_camera_id_input.setPlaceholderText("CAM001")
        self.wired_camera_id_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wired_layout.addRow("Camera ID:", self.wired_camera_id_input)
        
        button_layout = QHBoxLayout()
        
        self.validate_wired_btn = QPushButton("Validate Network")
        self.validate_wired_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6600;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FF7711;
            }
            QPushButton:pressed {
                background-color: #DD5500;
            }
        """)
        self.validate_wired_btn.clicked.connect(self._on_validate_wired)
        button_layout.addWidget(self.validate_wired_btn)
        
        self.connect_wired_btn = QPushButton("Connect Wired Camera")
        self.connect_wired_btn.setStyleSheet("""
            QPushButton {
                background-color: #00AA00;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00CC00;
            }
            QPushButton:pressed {
                background-color: #008800;
            }
        """)
        self.connect_wired_btn.clicked.connect(self._on_connect_wired)
        button_layout.addWidget(self.connect_wired_btn)
        
        wired_layout.addRow("", button_layout)
        
        self.camera_config_stack.addWidget(wired_panel)
        
        # Wireless camera configuration panel
        wireless_panel = QWidget()
        wireless_layout = QFormLayout(wireless_panel)
        wireless_layout.setContentsMargins(10, 5, 10, 5)
        wireless_layout.setSpacing(5)
        
        self.wireless_ssid_input = QLineEdit()
        self.wireless_ssid_input.setPlaceholderText("MyWiFi_Network")
        self.wireless_ssid_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wireless_layout.addRow("SSID/Network Name:", self.wireless_ssid_input)
        
        self.wireless_ip_input = QLineEdit()
        self.wireless_ip_input.setPlaceholderText("192.168.1.105")
        self.wireless_ip_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wireless_layout.addRow("Camera IP:", self.wireless_ip_input)
        
        self.wireless_url_input = QLineEdit()
        self.wireless_url_input.setPlaceholderText("http://192.168.1.105:8080/video")
        self.wireless_url_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wireless_layout.addRow("Camera URL:", self.wireless_url_input)
        
        self.wireless_username_input = QLineEdit()
        self.wireless_username_input.setPlaceholderText("admin")
        self.wireless_username_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wireless_layout.addRow("Username:", self.wireless_username_input)
        
        self.wireless_password_input = QLineEdit()
        self.wireless_password_input.setPlaceholderText("••••••")
        self.wireless_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.wireless_password_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wireless_layout.addRow("Password:", self.wireless_password_input)
        
        self.wireless_camera_id_input = QLineEdit()
        self.wireless_camera_id_input.setPlaceholderText("WLC001")
        self.wireless_camera_id_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        wireless_layout.addRow("Camera ID:", self.wireless_camera_id_input)
        
        self.connect_wireless_btn = QPushButton("Connect Wireless Camera")
        self.connect_wireless_btn.setStyleSheet("""
            QPushButton {
                background-color: #00AA00;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00CC00;
            }
            QPushButton:pressed {
                background-color: #008800;
            }
        """)
        self.connect_wireless_btn.clicked.connect(self._on_connect_wireless)
        wireless_layout.addRow("", self.connect_wireless_btn)
        
        self.camera_config_stack.addWidget(wireless_panel)
        
        # RTSP Stream configuration panel
        rtsp_panel = QWidget()
        rtsp_layout = QFormLayout(rtsp_panel)
        rtsp_layout.setContentsMargins(10, 5, 10, 5)
        rtsp_layout.setSpacing(5)
        
        self.rtsp_url_input = QLineEdit()
        self.rtsp_url_input.setPlaceholderText("rtsp://192.168.1.10/live")
        self.rtsp_url_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        rtsp_layout.addRow("RTSP URL:", self.rtsp_url_input)
        
        self.rtsp_username_input = QLineEdit()
        self.rtsp_username_input.setPlaceholderText("admin")
        self.rtsp_username_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        rtsp_layout.addRow("Username:", self.rtsp_username_input)
        
        self.rtsp_password_input = QLineEdit()
        self.rtsp_password_input.setPlaceholderText("••••••")
        self.rtsp_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.rtsp_password_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        rtsp_layout.addRow("Password:", self.rtsp_password_input)
        
        self.rtsp_stream_name_input = QLineEdit()
        self.rtsp_stream_name_input.setPlaceholderText("Main Stream")
        self.rtsp_stream_name_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        rtsp_layout.addRow("Stream Name:", self.rtsp_stream_name_input)
        
        self.connect_rtsp_btn = QPushButton("Connect RTSP Stream")
        self.connect_rtsp_btn.setStyleSheet("""
            QPushButton {
                background-color: #00AA00;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00CC00;
            }
            QPushButton:pressed {
                background-color: #008800;
            }
        """)
        self.connect_rtsp_btn.clicked.connect(self._on_connect_rtsp)
        rtsp_layout.addRow("", self.connect_rtsp_btn)
        
        self.camera_config_stack.addWidget(rtsp_panel)
        
        # Mobile camera configuration panel
        mobile_panel = QWidget()
        mobile_layout = QFormLayout(mobile_panel)
        mobile_layout.setContentsMargins(10, 5, 10, 5)
        mobile_layout.setSpacing(5)
        
        self.mobile_ip_input = QLineEdit()
        self.mobile_ip_input.setPlaceholderText("e.g., 192.168.1.5")
        self.mobile_ip_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        
        self.mobile_port_input = QLineEdit()
        self.mobile_port_input.setPlaceholderText("e.g., 8080")
        self.mobile_port_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.3);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        
        self.mobile_url_preview = QLineEdit()
        self.mobile_url_preview.setReadOnly(True)
        self.mobile_url_preview.setPlaceholderText("http://192.168.1.5:8080/video")
        self.mobile_url_preview.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0, 0, 0, 0.1);
                color: #888888;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        
        # Connect text change signals to automatically update the url preview
        self.mobile_ip_input.textChanged.connect(self._update_mobile_url_preview)
        self.mobile_port_input.textChanged.connect(self._update_mobile_url_preview)
        
        mobile_layout.addRow("Mobile IP Address:", self.mobile_ip_input)
        mobile_layout.addRow("Port:", self.mobile_port_input)
        mobile_layout.addRow("Stream URL Preview:", self.mobile_url_preview)
        
        mobile_button_layout = QHBoxLayout()
        
        self.connect_mobile_btn = QPushButton("Connect Mobile Camera")
        self.connect_mobile_btn.setStyleSheet("""
            QPushButton {
                background-color: #00AA00;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00CC00;
            }
            QPushButton:pressed {
                background-color: #008800;
            }
        """)
        self.connect_mobile_btn.clicked.connect(self._on_connect_mobile)
        mobile_button_layout.addWidget(self.connect_mobile_btn)
        
        self.disconnect_mobile_btn = QPushButton("Disconnect")
        self.disconnect_mobile_btn.setEnabled(False)
        self.disconnect_mobile_btn.setStyleSheet("""
            QPushButton {
                background-color: #8B0000;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #A52A2A;
            }
            QPushButton:pressed {
                background-color: #5D0000;
            }
            QPushButton:disabled {
                background-color: #333333;
                color: #888888;
            }
        """)
        self.disconnect_mobile_btn.clicked.connect(self._on_disconnect_mobile)
        mobile_button_layout.addWidget(self.disconnect_mobile_btn)
        
        # Status Label
        self.mobile_status_label = QLabel("DISCONNECTED")
        self.mobile_status_label.setStyleSheet("color: #AAAAAA; font-weight: bold; font-size: 12px; margin-left: 10px;")
        mobile_button_layout.addWidget(self.mobile_status_label)
        
        mobile_layout.addRow("", mobile_button_layout)
        
        self.camera_config_stack.addWidget(mobile_panel)
        
        main_layout.addWidget(self.camera_config_stack)

        # Input fields for monitoring parameters
        input_row = QHBoxLayout()
        input_row.setSpacing(10)

        # Location Name
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("Location Name")
        self.location_input.setMinimumWidth(200)
        input_row.addWidget(self.location_input)

        # Area / Zone
        self.area_input = QLineEdit()
        self.area_input.setPlaceholderText("Area / Zone")
        self.area_input.setMinimumWidth(150)
        input_row.addWidget(self.area_input)

        # Camera ID
        self.camera_id_input = QLineEdit()
        self.camera_id_input.setPlaceholderText("Camera ID")
        self.camera_id_input.setMinimumWidth(100)
        input_row.addWidget(self.camera_id_input)

        # Start Monitoring button
        self.start_button = QPushButton("Start Monitoring")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #00AA00;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00CC00;
            }
            QPushButton:pressed {
                background-color: #008800;
            }
        """)
        self.start_button.clicked.connect(self._on_start_monitoring)
        input_row.addWidget(self.start_button)

        # Stop Monitoring button
        self.stop_button = QPushButton("Stop Monitoring")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #8B0000;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #A52A2A;
            }
            QPushButton:pressed {
                background-color: #5D0000;
            }
        """)
        self.stop_button.clicked.connect(self._on_stop_monitoring)
        input_row.addWidget(self.stop_button)

        input_row.addStretch()
        main_layout.addLayout(input_row)

        # Main splitter: camera (left) | metrics+alerts (right)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        main_layout.addWidget(splitter)
        
        # Assign stretch factors to prioritize camera visibility
        main_layout.setStretch(0, 0) # Title
        main_layout.setStretch(1, 0) # Config Stack
        main_layout.setStretch(2, 0) # Input Controls
        main_layout.setStretch(3, 1) # Camera & Metrics
        
        # Left: camera panel
        self.camera_panel = CameraPanel()
        splitter.addWidget(self.camera_panel)

        # Right: tabs
        right_panel = QWidget()
        right_panel.setMinimumWidth(320)
        right_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        try:
            self.setup_overview_tab()
        except Exception as e:
            logger.error(f"Overview tab setup failed: {e}")

        try:
            self.setup_analytics_tab()
        except Exception as e:
            logger.error(f"Analytics tab setup failed: {e}")

        self.alerts_panel = AlertsPanel()
        self.tabs.addTab(self.alerts_panel, "Alerts")

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)  # Camera takes 75%
        splitter.setStretchFactor(1, 1)  # Metrics take 25%

        # --- PAGE 1: CAMERAS ---
        self.cameras_page = CameraManagerPanel(parent_dashboard=self)
        self.main_stack.addWidget(self.cameras_page)

        # --- PAGE 2: HISTORY ---
        self.history_page = HistoryPanel()
        self.main_stack.addWidget(self.history_page)


        self.statusBar().showMessage("System ready")


    # ------------------------------------------------------------------

    def setup_overview_tab(self):
        """Overview tab — live metrics grid"""
        tab = QWidget()
        # Main layout for the tab that supports scrolling
        main_tab_layout = QVBoxLayout(tab)
        main_tab_layout.setContentsMargins(0, 0, 0, 0)

        # QScrollArea prevents UI elements from collapsing when window height is reduced
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # --- Metrics frame ---
        metrics_frame = QFrame()
        metrics_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        metrics_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        mf_layout = QVBoxLayout(metrics_frame)
        mf_layout.setContentsMargins(10, 15, 10, 15)
        mf_layout.setSpacing(15) 

        def _create_metric_label() -> QLabel:
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background: transparent; border: none;")
            lbl.setMinimumHeight(45)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return lbl
        
        title = QLabel("Live Metrics") # Relocated: Was outside method, now correctly inside setup_overview_tab
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00FFFF; margin: 5px;")
        mf_layout.addWidget(title)
        
        self.crowd_count_label  = _create_metric_label()
        self.fps_label          = _create_metric_label()
        self.threat_level_label = _create_metric_label()
        self.confidence_label   = _create_metric_label()
        self.motion_score_label = _create_metric_label()
        self.accel_score_label  = _create_metric_label()
        self.dir_var_label      = _create_metric_label()
        self.streak_label       = _create_metric_label()
        # --- NEW LABELS FOR MOTION ANALYSIS ---
        self.avg_speed_label    = _create_metric_label()
        self.crowd_flow_label   = _create_metric_label()
        self.motion_status_label = _create_metric_label()
        self.abnormal_motion_label = _create_metric_label()
        # --- NEW LABELS FOR DENSITY, BEHAVIOUR, RISK ---
        self.density_label      = _create_metric_label()
        self.behaviour_label    = _create_metric_label()
        self.risk_score_label   = _create_metric_label()

        for lbl in (
            self.crowd_count_label, self.fps_label, self.threat_level_label,
            self.confidence_label, self.motion_score_label,
            self.accel_score_label, self.dir_var_label, self.streak_label,
            self.avg_speed_label, self.crowd_flow_label,
            self.motion_status_label, self.abnormal_motion_label,
            self.density_label, self.behaviour_label, self.risk_score_label
        ):
            mf_layout.addWidget(lbl)
            
        # Initial placeholder values
        self._update_metrics({}, 0, 0.0, {})

        layout.addWidget(metrics_frame)

        # --- Activity log ---
        activity_title = QLabel("Recent Activity:")
        activity_title.setStyleSheet("font-weight: bold; margin-top: 8px; color: #CCCCCC;")
        activity_title.setMinimumHeight(25)
        layout.addWidget(activity_title)

        self.activity_text = QLabel("System initialized…")
        self.activity_text.setWordWrap(True)
        self.activity_text.setMinimumHeight(120)
        self.activity_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.activity_text.setStyleSheet(
            "background-color: rgba(0,0,0,0.3); padding: 8px; border-radius: 5px; color: #AAAAAA;"
        )
        layout.addWidget(self.activity_text)
        
        layout.addStretch()
        
        scroll.setWidget(container)
        main_tab_layout.addWidget(scroll)
        
        self.tabs.addTab(tab, "Overview")

    # ------------------------------------------------------------------

    def setup_analytics_tab(self):
        """Analytics tab — real-time statistics and event history"""
        tab = QWidget()
        main_tab_layout = QVBoxLayout(tab)
        main_tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        container = QWidget()
        analytics_layout = QVBoxLayout(container)
        analytics_layout.setContentsMargins(10, 10, 10, 10)
        analytics_layout.setSpacing(15)
        
        analytics_layout.setStretch(0, 2)
        analytics_layout.setStretch(1, 1)

        # 1. Summary Cards (Grid)
        summary_title = QLabel("LIVE ANALYTICS SUMMARY")
        summary_title.setStyleSheet("font-weight: bold; color: #00FFFF; font-size: 14px;")
        analytics_layout.addWidget(summary_title)

        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)

        self.ana_crowd_card, self.ana_crowd_val = self._create_analytics_card("CROWD COUNT")
        self.ana_motion_card, self.ana_motion_val = self._create_analytics_card("MOTION SCORE")
        self.ana_threat_card, self.ana_threat_val = self._create_analytics_card("THREAT LEVEL")
        self.ana_risk_card, self.ana_risk_val = self._create_analytics_card("RISK SCORE") # Changed from ana_conf_card/val
        self.ana_accel_card, self.ana_accel_val = self._create_analytics_card("ACCEL SCORE")
        self.ana_dir_card, self.ana_dir_val = self._create_analytics_card("DIR VARIANCE")

        grid.addWidget(self.ana_crowd_card, 0, 0)
        grid.addWidget(self.ana_motion_card, 0, 1)
        grid.addWidget(self.ana_threat_card, 1, 0)
        grid.addWidget(self.ana_risk_card, 1, 1) # Changed from ana_conf_card
        grid.addWidget(self.ana_accel_card, 2, 0)
        grid.addWidget(self.ana_dir_card, 2, 1)
        
        analytics_layout.addWidget(grid_widget)

        # 2. Intensity Bars (Simple Graphs)
        intensity_title = QLabel("INTENSITY INDICATORS")
        intensity_title.setStyleSheet("font-weight: bold; color: #00FFFF; font-size: 11px; margin-top: 5px;")
        analytics_layout.addWidget(intensity_title)

        intensity_frame = QFrame()
        intensity_frame.setStyleSheet("background-color: rgba(0, 0, 0, 0.2); border-radius: 4px; padding: 10px;")
        intensity_layout = QVBoxLayout(intensity_frame)
        
        def add_bar(label):
            l = QLabel(label)
            l.setStyleSheet("color: #AAA; font-size: 10px;")
            pb = QProgressBar()
            pb.setStyleSheet("""
                QProgressBar { border: 1px solid #333; border-radius: 2px; text-align: center; height: 12px; background: #050505; }
                QProgressBar::chunk { background-color: #00AA00; }
            """)
            pb.setTextVisible(False)
            intensity_layout.addWidget(l)
            intensity_layout.addWidget(pb)
            return pb

        self.bar_motion = add_bar("MOTION INTENSITY")
        self.bar_risk   = add_bar("RISK LEVEL") # Changed from bar_conf
        self.bar_accel  = add_bar("ACCELERATION SPIKE")
        
        analytics_layout.addWidget(intensity_frame)

        # 2.5. Real-Time Trend Graph (Tasks 7, 8, 9)
        trend_title = QLabel("TELEMETRY TREND (LAST 100 FRAMES)")
        trend_title.setStyleSheet("font-weight: bold; color: #00FFFF; font-size: 11px; margin-top: 10px;")
        analytics_layout.addWidget(trend_title)

        try:
            if PYQTGRAPH_AVAILABLE:
                # Task 7 & 8: Optimized, Resizable Graph
                self.telemetry_plot = pg.PlotWidget()
                self.telemetry_plot.setBackground('#050505')
                self.telemetry_plot.setYRange(0, 100)
                self.telemetry_plot.showGrid(x=True, y=True, alpha=0.1)
                self.telemetry_plot.setLabel('left', 'Intensity', units='%')
                self.telemetry_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                self.telemetry_plot.setMinimumHeight(220)
                self.telemetry_plot.setMaximumHeight(260)
                
                # Add Legend
                legend = self.telemetry_plot.addLegend(offset=(5, 5))
                if hasattr(legend, 'setLabelTextSize'):
                    legend.setLabelTextSize('8pt')
                
                # Create Curves with optimized line settings
                self.threat_curve = self.telemetry_plot.plot(
                    pen=pg.mkPen(color=config.GRAPH_THREAT_COLOR, width=2), 
                    name="Threat Score"
                )
                self.risk_curve = self.telemetry_plot.plot( # Changed from conf_curve
                    pen=pg.mkPen(color=config.GRAPH_CONFIDENCE_COLOR, width=2), # Keep color for consistency
                    name="AI Confidence"
                )
                self.motion_curve = self.telemetry_plot.plot(
                    pen=pg.mkPen(color=config.GRAPH_MOTION_COLOR, width=1), 
                    name="Motion"
                )
                analytics_layout.addWidget(self.telemetry_plot)
            else:
                raise ImportError("PyQtGraph not available")
        except Exception as e:
            logger.warning(f"Graph init error: {e}. Trend visualization disabled.")
            self.fallback_label = QLabel("Graph module (pyqtgraph) unavailable.\nTrend visualization is disabled.")
            self.fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.fallback_label.setStyleSheet("""
                background-color: rgba(40, 0, 0, 0.3);
                color: #FF6B6B;
                border: 1px dashed #FF6B6B;
                padding: 20px;
                border-radius: 4px;
            """)
            analytics_layout.addWidget(self.fallback_label)

        # 3. History Log
        history_title = QLabel("ANALYTICS EVENT HISTORY")
        history_title.setStyleSheet("font-weight: bold; color: #00FFFF; font-size: 11px; margin-top: 10px;")
        analytics_layout.addWidget(history_title)

        self.analytics_history_text = QLabel("Awaiting data stream...")
        self.analytics_history_text.setWordWrap(True)
        self.analytics_history_text.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.analytics_history_text.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.3);
            color: #00FF00;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 11px;
            padding: 8px;
            border-radius: 4px;
        """)
        self.analytics_history_text.setMinimumHeight(200)
        analytics_layout.addWidget(self.analytics_history_text)

        analytics_layout.addStretch()
        scroll.setWidget(container)
        main_tab_layout.addWidget(scroll)
        self.tabs.addTab(tab, "Analytics")

    # -----------------------------------------------------------------------
    # Camera
    # -----------------------------------------------------------------------

    def start_camera(self):
        """Safely start the camera source and processing timer"""
        logger.info("Starting camera thread")
        try:
            # Prevent double initialization if already running
            is_active = False
            if self.camera_source and hasattr(self.camera_source, 'is_running'):
                is_active = self.camera_source.is_running # Access as property
            
            if is_active:
                logger.info("Camera already running, skipping start")
                return True

            # Validate camera index from input
            cam_idx = 0
            if hasattr(self, 'monitoring_camera_id') and str(self.monitoring_camera_id).isdigit():
                cam_idx = int(self.monitoring_camera_id)

            self.camera_source = WebcamSource(camera_id=cam_idx)
            if self.camera_source.start():
                if not self.timer.isActive():
                    self.timer.start(30)   # ~33 FPS
                self.log_event("Camera Connected", "System hardware initialized", current_crowd_count=0, current_threat_level="NORMAL")
                return True
            else:
                logger.error("Failed to start camera capture")
                self.log_event("Connection Failed", "No camera hardware detected", current_crowd_count=0, current_threat_level="NORMAL")
                self.statusBar().showMessage("Error: No Camera Feed Available", 5000)
                if self.camera_panel:
                    self.camera_panel.clear_display()
                return False
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            traceback.print_exc()
            self.update_activity(f"Camera error: {e}")
            return False

    # -----------------------------------------------------------------------
# Threaded Pipeline Implementation
    # -----------------------------------------------------------------------
    
    def start_pipeline_thread(self):
        print("start_pipeline_thread called")
        if hasattr(self, 'pipeline_thread') and self.pipeline_thread.is_alive():
            print("Pipeline thread already running")
            return
            
        import threading
        from queue import Queue, Empty
        
        print("Creating pipeline queue and thread")
        self.pipeline_queue = Queue(maxsize=1)
        self.pipeline_running = True
        
        def pipeline_worker():
            print("Pipeline worker function defined")
            import time
            import cv2
            import numpy as np
            
            fps_times = []
            frame_counter = 0
            adaptive_skip_interval = 2
            processing_res = (640, 360)
            
            last_detections = []
            last_weapons = []
            last_tracked_objects = []
            last_crowd_count = 0
            last_threat = {}
            
            print(f"Pipeline worker starting, camera_source={self.camera_source}")
            
            while self.pipeline_running:
                print("Pipeline worker iteration")
                try:
                    pipeline_start = time.perf_counter()
                    
                    # 1. Capture Time
                    t0 = time.perf_counter()
                    if not self.camera_source:
                        print("Camera source is None in pipeline")
                        time.sleep(0.01)
                        continue
                        
                    frame = self.camera_source.get_frame()
                    if frame is None:
                        print("Frame received: None")
                        time.sleep(0.01)
                        continue
                    print(f"Frame received: True")
                    capture_time = (time.perf_counter() - t0) * 1000
                    
                    now = time.perf_counter()
                    fps_times.append(now)
                    fps_times = [t for t in fps_times if now - t < 2.0]
                    fps = len(fps_times) / 2.0
                    
                    if fps > 0.1:
                        if fps < 12.0:
                            processing_res = (640, 360)
                            adaptive_skip_interval = 3
                        elif fps > 22.0:
                            processing_res = (960, 540)
                            adaptive_skip_interval = 2
                            
                    frame_resized = cv2.resize(frame, processing_res)
                    
                    # 2. Analytics Time (Motion)
                    t1 = time.perf_counter()
                    motion_data = motion_analyzer.analyze_motion(frame_resized)
                    analytics_time = (time.perf_counter() - t1) * 1000
                    
                    frame_counter += 1
                    inference_time = 0.0
                    tracking_time = 0.0
                    
                    if frame_counter % adaptive_skip_interval == 0:
                        # 3. Inference Time
                        t2 = time.perf_counter()
                        all_detections = person_detector.detect_all(frame_resized)
                        boxes = [d['box'] for d in all_detections if d['class'] == 0]
                        weapons = [d for d in all_detections if d['class'] != 0]
                        inference_time = (time.perf_counter() - t2) * 1000
                        
                        # 4. Tracking Time
                        t3 = time.perf_counter()
                        tracked_objects = track_persons(boxes)
                        crowd_count = len(tracked_objects) if tracked_objects else 0
                        
                        tracker_accel = person_tracker.get_acceleration_vectors()
                        tracker_speeds = person_tracker.get_speed_stats()
                        tracking_time = (time.perf_counter() - t3) * 1000
                        
                        # Add Threat Analysis to Analytics Time
                        t4 = time.perf_counter()
                        threat = analyze_threat(
                            frame=frame_resized,
                            tracked_objects=tracked_objects,
                            frame_shape=frame_resized.shape,
                            motion_data=motion_data,
                            tracker_accel=tracker_accel,
                            tracker_speeds=tracker_speeds,
                            weapon_detections=weapons,
                            person_boxes=boxes
                        )
                        analytics_time += (time.perf_counter() - t4) * 1000
                        
                        last_detections = all_detections
                        last_weapons = weapons
                        last_tracked_objects = tracked_objects
                        last_crowd_count = crowd_count
                        last_threat = threat
                    else:
                        all_detections = last_detections
                        weapons = last_weapons
                        tracked_objects = last_tracked_objects
                        crowd_count = last_crowd_count
                        threat = last_threat
                        
                    # 5. Render Time (Overlays)
                    t5 = time.perf_counter()
                    
                    # Check for critical alerts
                    alert_msg = None
                    if weapons and (threat.get('weapon_persistent') or threat.get('weapon_detected_raw')):
                        # Use a set to remove duplicates if multiple of same weapon detected
                        w_names = " & ".join(list(set([w['label'] for w in weapons])))
                        alert_msg = f"CRITICAL ALERT: WEAPON DETECTED ({w_names})"
                    elif threat.get('threat_level') in ['CRITICAL', 'HIGH RISK']:
                        alert_msg = f"WARNING: {threat.get('threat_level')} THREAT LEVEL"
                        
                    # We do not pass profiling directly to _draw_overlays here because we construct it later,
                    # but we will handle the overlay cleanups later. We'll pass alert_msg to draw_overlays.
                    annotated_frame = self._draw_overlays(frame_resized.copy(), all_detections, tracked_objects, threat, motion_data, alert_msg)
                    
                    render_time = (time.perf_counter() - t5) * 1000
                    
                    total_pipeline_time = (time.perf_counter() - pipeline_start) * 1000
                    
                    # Optional: Print profiling data strictly for debugging identifying bottleneck
                    # print(f"Profiling (ms): Capture={capture_time:.1f}, Inference={inference_time:.1f}, Tracking={tracking_time:.1f}, Analytics={analytics_time:.1f}, Render={render_time:.1f}, Total={total_pipeline_time:.1f}")
                    
                    data = {
                        'frame': annotated_frame,
                        'fps': fps,
                        'motion_data': motion_data,
                        'all_detections': all_detections,
                        'weapons': weapons,
                        'tracked_objects': tracked_objects,
                        'crowd_count': crowd_count,
                        'threat': threat,
                        'profiling': {
                            'capture_time': capture_time,
                            'inference_time': inference_time,
                            'tracking_time': tracking_time,
                            'analytics_time': analytics_time,
                            'render_time': render_time,
                            'total_pipeline_time': total_pipeline_time
                        }
                    }
                    
                    if self.pipeline_queue.full():
                        try:
                            self.pipeline_queue.get_nowait() # Drop oldest
                        except Empty:
                            pass
                    self.pipeline_queue.put(data)
                    
                    # Yield CPU
                    time.sleep(0.001)

                except Exception as e:
                    print(f"Pipeline worker error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.01)
        
        self.pipeline_thread = threading.Thread(target=pipeline_worker, daemon=True)
        print("Starting pipeline thread")
        self.pipeline_thread.start()
        print(f"Pipeline thread started: {self.pipeline_thread.is_alive()}")

    def update_frame(self):
        from queue import Empty
        import time
        import numpy as np
        
        try:
            # Monitor camera source status changes (keep in UI thread)
            if self.camera_source and hasattr(self.camera_source, 'get_connection_status'):
                status = self.camera_source.get_connection_status()
                if not hasattr(self, '_prev_mobile_status') or self._prev_mobile_status != status:
                    self.update_activity(f"Mobile Camera status: {status}")
                    self.log_event("Mobile Camera Update", f"Connection status changed to {status}")
                    if hasattr(self, 'mobile_status_label') and self.mobile_status_label:
                        self.mobile_status_label.setText(status)
                        color_map = {
                            "CONNECTED": "#00FF00", "DISCONNECTED": "#AAAAAA",
                            "RECONNECTING": "#FFA500", "FAILED": "#FF0000",
                            "STREAM_ERROR": "#FF0000", "ERROR": "#FF0000"
                        }
                        self.mobile_status_label.setStyleSheet(f"color: {color_map.get(status, '#FFF')}; font-weight: bold; font-size: 12px; margin-left: 10px;")
                    self._prev_mobile_status = status

            # Update live FPS display
            if self.camera_source and self.camera_source.is_running:
                fps = self.camera_source.get_fps()
                if hasattr(self, 'webcam_fps_label') and self.webcam_fps_label:
                    self.webcam_fps_label.setText(f"FPS: {fps:.1f}")

            # If monitoring is NOT active, update live feed with raw camera frame
            if not getattr(self, 'is_monitoring', False):
                if self.camera_source and self.camera_source.is_running:
                    raw_frame = self.camera_source.get_frame()
                    if raw_frame is not None:
                        self.camera_panel.update_frame(raw_frame)
                return

            # When monitoring IS active:
            if not hasattr(self, 'pipeline_queue'):
                self.start_pipeline_thread()
                return

            try:
                # Get the latest frame from the pipeline
                data = self.pipeline_queue.get_nowait()
            except Empty:
                return
                
            frame_resized = data['frame']
            fps = data['fps']
            motion_data = data['motion_data']
            weapons = data['weapons']
            crowd_count = data['crowd_count']
            threat = data['threat']
            
            # Update low light indicator visibility
            self.low_light_label.setVisible(motion_data.get('is_low_light', False))

            # ---- 6. Telemetry Pipeline ----
            # threat.get('risk_score') is already smoothed and 0-100
            current_risk_score = threat.get('risk_score', 0.0)
            
            # The threat_score here in dashboard was a custom calculation,
            # now we directly use the smoothed_risk_score from threat_detector
            # For consistency, let's map the threat_detector's risk_score to dashboard's threat_score
            dashboard_threat_score = current_risk_score / 100.0 # Normalize to 0-1 for dashboard's internal telemetry

            # Extract components from threat_detector's result for telemetry
            # Note: threat_detector.py now returns 'combined_accel'
            raw_accel = float(threat.get('accel_score', 0.0)) # This is combined_accel
            processed_accel = 0.0 if raw_accel < 0.12 else raw_accel
            processed_motion = float(motion_data.get('motion_score', 0.0))
            weapon_score_component = threat.get('weapon_component', 0.0) # New field from threat_detector

            self.telemetry_smoothed["threat_score"] = (self.ema_alpha * dashboard_threat_score) + ((1 - self.ema_alpha) * self.telemetry_smoothed["threat_score"])
            self.telemetry_smoothed["motion_score"] = (self.ema_alpha * processed_motion) + ((1 - self.ema_alpha) * self.telemetry_smoothed["motion_score"])
            self.telemetry_smoothed["accel_score"]  = (self.ema_alpha * processed_accel) + ((1 - self.ema_alpha) * self.telemetry_smoothed["accel_score"])
            self.telemetry_smoothed["risk_score"]   = (self.ema_alpha * current_risk_score) + ((1 - self.ema_alpha) * self.telemetry_smoothed["risk_score"]) # Use 0-100 score
            self.telemetry_smoothed["confidence"]   = (self.ema_alpha * dashboard_threat_score) + ((1 - self.ema_alpha) * self.telemetry_smoothed["confidence"]) # For compatibility

            self.telemetry.update({
                "threat_score": self.telemetry_smoothed["threat_score"],
                "motion_score": self.telemetry_smoothed["motion_score"],
                "accel_score":  self.telemetry_smoothed["accel_score"],
                "confidence":   current_risk_score / 100.0, # Normalized 0-1 for compatibility
                "risk_score":   self.telemetry_smoothed["risk_score"], # Use 0-100 score
                "fps": fps,
                "person_count": crowd_count,
                "weapon_score": weapon_score_component, # Use the component score
                "dir_variance": threat.get('dir_variance', 0.0)
            })

            # Append 0-100 risk score to history
            self.telemetry_history['threat'].append(current_risk_score) # Use actual risk score for graph
            self.telemetry_history['motion'].append(self.telemetry['motion_score'])
            self.telemetry_history['accel'].append(self.telemetry['accel_score'])
            self.telemetry_history['risk'].append(self.telemetry['risk_score']) # Changed 'conf' to 'risk'
            self.telemetry_history['fps'].append(fps)
            self.telemetry_history['count'].append(crowd_count)

            now_ms = time.perf_counter()
            self._last_analytics_ui_update_time = getattr(self, '_last_analytics_ui_update_time', 0.0)
            if now_ms - self._last_analytics_ui_update_time > 0.35:
                self._update_analytics_tab(threat, motion_data, crowd_count, fps)
                self._update_metrics(threat, crowd_count, fps, motion_data)
                self._last_analytics_ui_update_time = now_ms

            current_level = threat.get('threat_level', 'NORMAL')

            if crowd_count != self.prev_crowd_count:
                self.log_event("Crowd Count Changed", f"Count: {self.prev_crowd_count} -> {crowd_count}", current_crowd_count=crowd_count, current_threat_level=current_level)
                self.prev_crowd_count = crowd_count

            if current_level != self.prev_threat_level:
                self.log_event("Threat Level Changed", f"Level: {self.prev_threat_level} -> {current_level}", current_crowd_count=crowd_count, current_threat_level=current_level)
                self.prev_threat_level = current_level

            now = time.perf_counter()
            if motion_data.get('motion_score', 0) > 0.05 and (now - self.last_move_log_time > 5.0) and not threat.get('weapon_persistent'):
                self.log_event("Movement Detected", f"Motion: {motion_data['motion_score']*100:.1f}% | Conf: {threat.get('confidence_pct', '0%')}")
                self.last_move_log_time = now

            self.alert_overlay_active = False

            if weapons:
                w_names = ", ".join([w['label'] for w in weapons])
                if threat.get('weapon_persistent') or threat.get('weapon_detected_raw'):
                    self.log_event("WEAPON DETECTED", f"Detected: {w_names} (Risk: {threat.get('risk_score_pct', '0%')})", current_crowd_count=crowd_count, current_threat_level="CRITICAL", weapon_details={'type': w_names, 'risk_score': threat.get('risk_score', 0.0), 'duration': threat.get('weapon_persistence_frames', 0)})
                    self.alert_overlay_active = True
            elif threat.get('should_alert'):
                self._fire_alert(threat)

            # Draw Performance metrics directly on frame via modular function
            profiling = data.get('profiling', {})
            if profiling:
                frame_resized = self._draw_performance_panel(frame_resized, profiling)

            self.camera_panel.update_frame(frame_resized)

        except Exception as e:
            logger.error(f"Error in update_frame: {e}")
            import traceback
            traceback.print_exc()
            self.update_activity(f"Processing error: {e}")

    # -----------------------------------------------------------------------
    # Overlay drawing
    # -----------------------------------------------------------------------

    def _draw_overlays(self, frame, detections, tracked_objects, threat, motion_data, alert_msg=None, profiling=None):
        """Orchestrate professional UI drawing to prevent overlapping text."""
        
        # 1. Tracking trails
        if tracked_objects:
            traces = person_tracker.get_object_traces()
            frame = person_tracker.draw_tracks(frame, traces)

        # 2. Optional debug: motion vectors + heatmap (drawn under everything else)
        if self._show_debug:
            frame = motion_analyzer.draw_debug_overlay(
                frame, motion_data,
                show_vectors=True,
                show_heatmap=True
            )

        # 3. Draw clean bounding boxes (Persons and Weapons)
        frame = self._draw_bounding_boxes(frame, detections)
        
        # 4. Draw fixed Threat Panel on the left
        frame = self._draw_threat_panel(frame, threat)
        
        # 5. Draw Performance Panel on the right
        if profiling:
            frame = self._draw_performance_panel(frame, profiling)
        
        # 6. Draw Alert Banner if active (top center)
        if alert_msg:
            frame = self._draw_alert_banner(frame, alert_msg)
            
        return frame

    def _refresh_analytics_ui(self):
        """Centralized Telemetry Update Logic for Graphs and Summary (Tasks 1-7, 10)"""
        if not hasattr(self, 'telemetry') or not self.telemetry:
            return
            
        try:
            # Task 4 & 9: Real-time reflection with stable ranges
            # Risk Score Bar (was called Confidence/Threat)
            conf_val = int(np.clip(self.telemetry.get('confidence', 0) * 100, 0, 100))
            if hasattr(self, 'bar_risk'):
                self.bar_risk.setValue(conf_val)
                self._apply_bar_style(self.bar_risk, conf_val)

            # Motion Intensity Bar
            if hasattr(self, 'bar_motion'):
                motion_val = int(np.clip(self.telemetry.get('motion_score', 0) * 100, 0, 100))
                self.bar_motion.setValue(motion_val)
                self._apply_bar_style(self.bar_motion, motion_val)

            # Acceleration Spike Bar
            if hasattr(self, 'bar_accel'):
                accel_val = int(np.clip(self.telemetry.get('accel_score', 0) * 100, 0, 100))
                self.bar_accel.setValue(accel_val)
                self._apply_bar_style(self.bar_accel, accel_val)

            # 2. Update Analytics Summary Labels (Tasks 1, 6)
            if hasattr(self, 'ana_crowd_val'):
                self.ana_crowd_val.setText(str(self.telemetry.get('person_count', 0)))
            if hasattr(self, 'ana_motion_val'):
                self.ana_motion_val.setText(f"{self.telemetry.get('motion_score', 0)*100:.1f}%")
            if hasattr(self, 'ana_risk_val'):
                self.ana_risk_val.setText(f"{self.telemetry.get('confidence', 0)*100:.1f}%")
            if hasattr(self, 'ana_accel_val'):
                self.ana_accel_val.setText(f"{self.telemetry.get('accel_score', 0)*100:.1f}%")
            if hasattr(self, 'ana_dir_val'):
                self.ana_dir_val.setText(f"{self.telemetry.get('dir_variance', 0):.3f}")
            
            # Update Overview Confidence Label
            if hasattr(self, 'confidence_label'):
                level = self.prev_threat_level
                color = _LEVEL_COLORS.get(level, '#00FFFF')
                self.confidence_label.setText(self._format_metric_html("CONFIDENCE", f"{self.telemetry.get('confidence', 0) * 100:.1f}%", color))
                
        except Exception as e:
            logger.error(f"Error refreshing analytics UI: {e}")

    def _apply_bar_style(self, bar, val):
        """Task 6: Professional Tiered Colors (Green -> Yellow -> Orange -> Red)"""
        if val > 80: color = "#FF0000"     # Red (High Threat)
        elif val > 65: color = "#FF6600"   # Orange (Suspicious)
        elif val > 35: color = "#FFFF00"   # Yellow/Cyan (Normal Active)
        else: color = "#00FF00"            # Green (Normal Static)
        bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
        
    def update_graphs(self):
        try:
            if not PYQTGRAPH_AVAILABLE or not hasattr(self, 'telemetry_plot'):
                return
                
            # Tie graph values directly to real telemetry
            motion = float(self.telemetry.get("motion_score", 0.0)) * 100
            risk = float(self.telemetry.get("risk_score", 0.0)) # Already 0-100
            threat = float(self.telemetry.get("threat_score", 0.0)) * 100 # This is the 0-1 normalized threat_score

            self.motion_history.append(motion)
            self.risk_history.append(risk) # Changed conf_history to risk_history
            self.threat_history.append(threat)
            
            x = list(range(len(self.motion_history))) # Relocated: Was outside method
            
            self.motion_curve.setData(x, list(self.motion_history))
            self.risk_curve.setData(x, list(self.risk_history)) # Changed conf_curve to risk_curve
            self.threat_curve.setData(x, list(self.threat_history))
        except Exception as e:
            print(f"Graph update error: {e}")
    
    # -----------------------------------------------------------------------
    # Custom Overlay Rendering Methods
    # -----------------------------------------------------------------------
    
    def _draw_alert_banner(self, frame: np.ndarray, msg: str) -> np.ndarray:
        """Draws a high-priority, full-width alert banner at the top of the frame."""
        h, w = frame.shape[:2]
        scale_factor = max(0.6, h / 720.0)
        banner_height = int(50 * scale_factor)
        
        # Semi-transparent red banner background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 150), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Bottom border for the banner
        cv2.line(frame, (0, banner_height), (w, banner_height), (0, 0, 255), max(1, int(2 * scale_factor)))
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX # Relocated: Was outside method
        font_scale = 0.8 * scale_factor
        thickness = max(1, int(2 * scale_factor))
        (text_w, text_h), _ = cv2.getTextSize(msg, font, font_scale, thickness)
        
        text_x = (w - text_w) // 2
        text_y = int(35 * scale_factor)
        
        # Add drop shadow for readability
        cv2.putText(frame, msg, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, msg, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        return frame

    def _draw_threat_panel(self, frame: np.ndarray, threat: dict) -> np.ndarray:
        """Draws a professional, dynamically sized side-panel for threat status."""
        h, w = frame.shape[:2]
        scale_factor = max(0.6, h / 720.0)
        
        level      = threat.get('threat_level', 'NORMAL')
        risk_pct   = threat.get('risk_score_pct', '0.0%') # Changed from confidence_pct
        threat_type = threat.get('threat_type', 'None')
        crowd      = threat.get('crowd_count', 0)
        weapon_persistent = threat.get('weapon_persistent', False)
        
        level_colors_bgr = {
            'NORMAL':    (0, 200, 0),
            'SUSPICIOUS':(0, 255, 255),
            'HIGH RISK': (0, 128, 255),
            'CRITICAL':  (255, 0, 255),
            'WEAPON':    (0, 0, 255),
        }
        
        is_active = weapon_persistent or level in ['CRITICAL', 'HIGH RISK']
        status_color = level_colors_bgr.get('WEAPON', (0, 0, 255)) if weapon_persistent else level_colors_bgr.get(level, (0, 200, 0))
        
        # 1. Build row metrics dynamically # Relocated: Was outside method
        metrics = [
            ("Threat", level, status_color),
            ("Risk Score", risk_pct, (0, 220, 255)), # Changed from Confidence
            ("Persons", str(crowd), (200, 200, 200)),
            ("Status", "ACTIVE" if is_active else "MONITORING", status_color if is_active else (0, 200, 0))
        ]
        
        # Only add weapon info once and only if present
        if weapon_persistent or (threat_type and threat_type != 'None' and threat_type != ''):
            metrics.insert(1, ("Weapon", str(threat_type).replace('WEAPON: ', '').strip(), (220, 220, 220)))

        # 2. Layout calculations # Relocated: Was outside method
        font_scale = 0.55 * scale_factor
        line_height = int(35 * scale_factor)
        padding = int(15 * scale_factor)
        panel_width = int(260 * scale_factor)
        
        header_height = int(45 * scale_factor)
        panel_height = header_height + (len(metrics) * line_height) + padding
        
        start_x = int(15 * scale_factor)
        start_y = int(70 * scale_factor) # Below alert banner
        
        # 3. Draw Panel Background # Relocated: Was outside method
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + panel_width, start_y + panel_height), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # 4. Draw Accent Line
        cv2.line(frame, (start_x, start_y), (start_x, start_y + panel_height), status_color, max(2, int(4 * scale_factor)))
        
        # 5. Draw Header # Relocated: Was outside method
        title_font_scale = 0.65 * scale_factor
        cv2.putText(frame, "THREAT PANEL", (start_x + padding, start_y + int(28 * scale_factor)), 
                    cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (220, 220, 220), max(1, int(2 * scale_factor)), cv2.LINE_AA)
        cv2.line(frame, (start_x + padding, start_y + header_height - 5), (start_x + panel_width - padding, start_y + header_height - 5), (80, 80, 80), 1)

        # 6. Draw Rows
        current_y = start_y + header_height + int(20 * scale_factor)

        for label, val, color in metrics:
            # Left column: Label
            cv2.putText(frame, f"{label}", (start_x + padding, current_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (160, 160, 160), 1, cv2.LINE_AA)
            
            # Right column: Value (right-aligned)
            (val_w, _), _ = cv2.getTextSize(str(val), cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, int(2 * scale_factor)))
            cv2.putText(frame, str(val), (start_x + panel_width - padding - val_w, current_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, int(1.5 * scale_factor)), cv2.LINE_AA)
            
            current_y += line_height
            
        return frame

    def _draw_performance_panel(self, frame: np.ndarray, profiling: dict) -> np.ndarray:
        """Draws the performance metrics in the top right corner."""
        h, w = frame.shape[:2]
        scale_factor = max(0.6, h / 720.0) # Relocated: Was outside method
        
        cap_t = profiling.get('capture_time', 0) # Indentation fix
        inf_t = profiling.get('inference_time', 0)
        ren_t = profiling.get('render_time', 0)
        tot_t = profiling.get('total_pipeline_time', 0)
        
        metrics = [
            ("Capture Delay", f"{cap_t:.1f}ms", (0, 255, 0) if cap_t < 50 else (0, 165, 255)),
            ("Inference Time", f"{inf_t:.1f}ms", (0, 255, 0) if inf_t < 100 else (0, 165, 255)),
            ("Render Time", f"{ren_t:.1f}ms", (0, 255, 0) if ren_t < 50 else (0, 165, 255)),
            ("Total Latency", f"{tot_t:.1f}ms", (0, 0, 255) if tot_t > 300 else ((0, 255, 0) if tot_t < 200 else (0, 165, 255)))
        ]
        
        font_scale = 0.5 * scale_factor # Relocated: Was outside method
        line_height = int(25 * scale_factor)
        padding = int(15 * scale_factor)
        panel_width = int(240 * scale_factor)
        
        header_height = int(40 * scale_factor)
        panel_height = header_height + (len(metrics) * line_height) + padding
        
        start_x = w - panel_width - int(15 * scale_factor)
        start_y = int(70 * scale_factor) # Align with Threat panel
        
        # Draw Panel Background # Relocated: Was outside method
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (start_x + panel_width, start_y + panel_height), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Header
        title_font_scale = 0.55 * scale_factor
        cv2.putText(frame, "PERFORMANCE", (start_x + padding, start_y + int(25 * scale_factor)), 
                    cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (200, 200, 200), max(1, int(1.5 * scale_factor)), cv2.LINE_AA) # Indentation fix
        cv2.line(frame, (start_x + padding, start_y + header_height - 5), (start_x + panel_width - padding, start_y + header_height - 5), (80, 80, 80), 1)

        # Rows
        current_y = start_y + header_height + int(15 * scale_factor)
        
        for label, val, color in metrics:
            cv2.putText(frame, label, (start_x + padding, current_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (150, 150, 150), 1, cv2.LINE_AA)
            
            (val_w, _), _ = cv2.getTextSize(str(val), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.putText(frame, str(val), (start_x + panel_width - padding - val_w, current_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, max(1, int(1.2 * scale_factor)), cv2.LINE_AA)
            
            current_y += line_height
            
        return frame

    def _draw_bounding_boxes(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw clean bounding boxes without overlapping with the UI panels."""
        h, w = frame.shape[:2]
        scale_factor = max(0.6, h / 720.0) # Indentation fix
        font_scale = 0.5 * scale_factor
        
        for det in detections:
            x, y, bw, bh = det['box']
            is_weapon = det['class'] != 0
            
            box_color = (0, 0, 255) if is_weapon else (0, 255, 0)
            box_thickness = max(1, int(2 * scale_factor))
            
            # Pulse effect for weapons
            if is_weapon and int(time.time() * 5) % 2:
                cv2.rectangle(frame, (x-4, y-4), (x+bw+4, y+bh+4), (0, 0, 255), 1)

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, box_thickness)
            
            label = f"{det['label']} {det['conf']:.2f}"
            
            # Prevent text from drawing over the top alert banner
            text_y = max(y - int(10 * scale_factor), int(60 * scale_factor))
            
            # Text background for readability
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, int(1.5 * scale_factor)))
            cv2.rectangle(frame, (x, text_y - text_h - int(6 * scale_factor)), (x + text_w, text_y + int(4 * scale_factor)), (0, 0, 0), -1)
            cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, max(1, int(1.5 * scale_factor)), cv2.LINE_AA)
            
        return frame

    # -----------------------------------------------------------------------
    # UI update helpers
    # -----------------------------------------------------------------------

    def _update_metrics(self, threat: dict, crowd_count: int, fps: float, motion_data: dict):
        if not hasattr(self, 'crowd_count_label') or self.crowd_count_label is None:
            return

        level      = threat.get('threat_level', 'NORMAL')
        ms         = threat.get('motion_score', 0.0)
        ac         = threat.get('accel_score', 0.0)
        dv         = threat.get('dir_variance', 0.0)
        streak     = threat.get('confirmation_streak', 0)
        req        = config.THREAT_CONFIRMATION_FRAMES

        color = _LEVEL_COLORS.get(level, '#FFFFFF')

        # Existing labels with color
        if hasattr(self, 'crowd_count_label') and self.crowd_count_label:
            self.crowd_count_label.setText(self._format_metric_html("PERSONS", crowd_count, color))
        if hasattr(self, 'fps_label') and self.fps_label:
            self.fps_label.setText(self._format_metric_html("FPS", f"{fps:.1f}", color))
        if hasattr(self, 'threat_level_label') and self.threat_level_label:
            self.threat_level_label.setText(self._format_metric_html("THREAT LEVEL", level, color))
        if hasattr(self, 'confidence_label') and self.confidence_label:
            self.confidence_label.setText(self._format_metric_html("CONFIDENCE", f"{self.telemetry.get('confidence', 0) * 100:.1f}%", color))
        if hasattr(self, 'motion_score_label') and self.motion_score_label:
            self.motion_score_label.setText(self._format_metric_html("MOTION SCORE", f"{ms*100:.1f}%", color))
        if hasattr(self, 'accel_score_label') and self.accel_score_label:
            self.accel_score_label.setText(self._format_metric_html("ACCEL SCORE", f"{ac*100:.1f}%", color))
        if hasattr(self, 'dir_var_label') and self.dir_var_label:
            self.dir_var_label.setText(self._format_metric_html("DIR VARIANCE", f"{dv:.3f}", color))
        if hasattr(self, 'streak_label') and self.streak_label:
            self.streak_label.setText(self._format_metric_html("CONFIRM STREAK", f"{streak} / {req}", color))
        
        # New metrics for motion analysis # Relocated: Was outside method
        # Average Speed: convert motion_magnitude (in scaled frame) to original frame pixels per frame
        avg_speed = motion_data.get('motion_magnitude', 0) / motion_analyzer.scale
        # Crowd Flow: direction in degrees
        direction_rad = motion_data.get('motion_direction', 0)
        direction_deg = (direction_rad * 180 / math.pi) % 360
        # Motion Status: based on motion_score (which is ms, 0-1)
        if ms < 0.3:
            motion_status = "Low Activity"
        elif ms < 0.6:
            motion_status = "Moderate Activity"
        else:
            motion_status = "High Activity"
        # Abnormal Motion: based on dir_variance (dv)
        if dv > 0.6:
            abnormal_motion = "High (Chaotic)"
        elif dv > 0.3:
            abnormal_motion = "Medium (Somewhat Chaotic)"
        else:
            abnormal_motion = "Low (Ordered)"
        
        if hasattr(self, 'avg_speed_label') and self.avg_speed_label:
            self.avg_speed_label.setText(self._format_metric_html("AVG SPEED", f"{avg_speed:.1f} px/f", color))
        if hasattr(self, 'crowd_flow_label') and self.crowd_flow_label:
            self.crowd_flow_label.setText(self._format_metric_html("CROWD FLOW", f"{direction_deg:.0f}°", color))
        if hasattr(self, 'motion_status_label') and self.motion_status_label:
            self.motion_status_label.setText(self._format_metric_html("MOTION STATUS", motion_status, color))
        if hasattr(self, 'abnormal_motion_label') and self.abnormal_motion_label:
            self.abnormal_motion_label.setText(self._format_metric_html("ABNORMAL MOTION", abnormal_motion, color)) # Relocated: Was outside method
        # Relocated: Was outside method
        # New metrics for density, behaviour, risk score
        density_score = threat.get('density_score', 0.0)  # 0-1 normalized
        behaviour_score = threat.get('behavior_component', 0.0)  # 0-1
        risk_score = threat.get('risk_score', 0.0)  # 0-100
        
        if hasattr(self, 'density_label') and self.density_label:
            self.density_label.setText(self._format_metric_html("DENSITY", f"{density_score*100:.1f}%", color))
        if hasattr(self, 'behaviour_label') and self.behaviour_label:
            self.behaviour_label.setText(self._format_metric_html("BEHAVIOUR", f"{behaviour_score*100:.1f}%", color))
        if hasattr(self, 'risk_score_label') and self.risk_score_label:
            self.risk_score_label.setText(self._format_metric_html("RISK SCORE", f"{risk_score:.1f}%", color))

    def _format_metric_html(self, label, val, color="#00FFFF"):
        """Helper to generate HTML for metric labels"""
        return (f"<div style='text-align: center; line-height: 1.1;'>"
                f"<span style='color: #888888; font-size: 10px; font-weight: normal;'>{label}</span><br/>"
                f"<span style='color: {color}; font-size: 20px; font-weight: bold;'>{val}</span>"
                f"</div>")

    def _fire_alert(self, threat: dict):
        level = threat.get('threat_level', 'NORMAL')
        msg   = threat.get('threat_type', 'UNKNOWN THREAT')

        metrics = {
            'risk_score_pct': threat.get('risk_score_pct', '0.0%'), # Changed from confidence_pct
            'motion_score':   threat.get('motion_score', 0.0),
            'accel_score':    threat.get('accel_score', 0.0),
            'dir_variance':   threat.get('dir_variance', 0.0),
            'crowd_count':    threat.get('crowd_count', 0),
        }
        self.alerts_panel.add_alert(msg, level, metrics=metrics)
        self.update_activity(f"Alert: [{level}] {msg} | Risk: {metrics['risk_score_pct']}") # Changed Conf to Risk

    def log_event(self, event_type: str, details: str = "", current_crowd_count: Optional[int] = None, current_threat_level: Optional[str] = None, weapon_details: Optional[dict] = None):
        """Structured logging to UI, Database, and CSV"""
        try:
            from datetime import datetime
            now = datetime.now()
            ts_display = now.strftime("%H:%M:%S")
            
            source = self.camera_source_combo.currentText()
            camera_type = self.current_camera_type

            # Add weapon details to log entry if present
            weapon_info_str = ""
            if weapon_details:
                weapon_info_str = f"\nWeapon: {weapon_details.get('type', 'N/A')} (Risk: {weapon_details.get('risk_score', 0.0):.2f}, Duration: {weapon_details.get('duration', 0)}F)"
            
            # 1. Update Recent Activity Panel (Structured Format)
            log_entry = (
                f"[{ts_display}]\n"
                f"Area: {self.monitoring_area}\n"
                f"Event: {event_type}\n"
                f"Detail: {details}\n"
                f"---"
            )
            
            current_text = self.activity_text.text()
            # Keep last 15 lines for readability
            lines = f"{log_entry}\n{current_text}".split('\n')[:20]
            self.activity_text.setText('\n'.join(lines))

            # 2. Persistent Storage (Database)
            crowd_to_log = current_crowd_count if current_crowd_count is not None else self.prev_crowd_count
            threat_to_log = current_threat_level if current_threat_level is not None else self.prev_threat_level

            # Legacy crowd analytics log
            db_data = {
                'location_name':     self.monitoring_location,
                'area_name':         self.monitoring_area,
                'camera_id':         self.monitoring_camera_id,
                'camera_type':       source,
                'network_name':      self.current_network,
                'ip_address':        self.current_ip,
                'crowd_count':       crowd_to_log,
                'threat_status':     threat_to_log,
                'density_level':     getattr(self, '_last_density_score', 0.0),
                'session_duration':  (time.time() - self._monitoring_start_time) if self.is_monitoring else 0.0,
                'connection_status': 'CONNECTED' if self.is_monitoring else 'DISCONNECTED',
                'reconnect_attempts': 0
            }
            db_manager.log_analytics_async(db_data)
            
            # New persistent event history (Requirement 1 & 2)
            event_history_data = { # Relocated: Was outside method
                'location':      self.monitoring_location,
                'area':          self.monitoring_area,
                'camera_source': source,
                'camera_id':     self.monitoring_camera_id,
                'event_type':    event_type,
                'threat_level':  threat_to_log,
                'risk_score':    self.telemetry.get('risk_score', 0.0), # Changed confidence to risk_score
                'motion_score':  self.telemetry.get('motion_score', 0.0),
                'accel_score':   self.telemetry.get('accel_score', 0.0),
                'notes':         details
            }
            db_manager.log_event_async(event_history_data) # Relocated: Was outside method
            

            # Log threat event specifically if it's a weapon detection
            if event_type == "WEAPON DETECTED" and weapon_details:
                threat_db_data = {
                    'location_name': self.monitoring_location,
                    'area_name': self.monitoring_area,
                    'camera_id': self.monitoring_camera_id,
                    'threat_type': f"WEAPON: {weapon_details.get('type', 'N/A')}", # This is fine
                    'risk_score': weapon_details.get('risk_score', 0.0), # Changed confidence to risk_score
                    'frame_saved': '', 
                    'description': f"{details} (Duration: {weapon_details.get('duration', 0)}F)"
                }
                db_manager.insert_threat_event(threat_db_data) # Relocated: Was outside method

            # 3. Persistent Storage (CSV)
            csv_data = db_data.copy()
            csv_data['description'] = f"{event_type}: {details}{weapon_info_str}" 
            csv_data['event_type'] = event_type
            csv_manager.log_crowd_data(csv_data)
            
            logger.info(f"Event Logged: {event_type} - {details}")


        except Exception as e:
            logger.error(f"Failed to record log entry: {e}")

    def update_activity(self, message: str):
        """Legacy wrapper for simple activity messages"""
        self.log_event("System Update", message, current_crowd_count=self.prev_crowd_count, current_threat_level=self.prev_threat_level)

    # -----------------------------------------------------------------------
    # Debug toggle
    # -----------------------------------------------------------------------

    def _toggle_debug(self, state):
        self._show_debug = bool(state)
        logger.info(f"Debug overlay {'enabled' if self._show_debug else 'disabled'}")

    def _on_low_light_mode_changed(self, mode: str):
        """Handle manual override of low light mode from UI"""
        mapping = {
            "Auto": "Auto",
            "Force On": "On",
            "Force Off": "Off"
        }
        internal_mode = mapping.get(mode, "Auto")
        motion_analyzer.low_light_mode = internal_mode
        logger.info(f"Manual Low Light Mode set to: {internal_mode}")
        self.update_activity(f"Low light mode override: {internal_mode}")

    def _safe_stop_camera_source(self):
        """Release and stop previous camera capture safely to prevent duplicate capture objects and webcam locking"""
        if hasattr(self, 'pipeline_running'):
            self.pipeline_running = False
        if self.camera_source:
            try:
                self.camera_source.stop()
            except Exception as e:
                logger.error(f"Error stopping camera source: {e}")
            self.camera_source = None
        if hasattr(self, 'camera_panel') and self.camera_panel:
            self.camera_panel.clear_display()

    def _on_camera_source_changed(self, source: str):
        """Handle camera source dropdown selection change"""
        print(f"Camera source selected: {source}")
        # Update current camera source label (we could store this in a variable)
        self.current_camera_source = source
        # Switch to the appropriate configuration panel
        self.current_camera_type = source # Update camera type for logging
        index_map = {
            "Webcam": 0,
            "Wired Camera": 1,
            "Wireless Camera": 2,
            "RTSP Stream": 3,
            "RTSP Camera": 3,
            "Mobile Camera": 4
        }
        if source in index_map:
            self.camera_config_stack.setCurrentIndex(index_map[source])
        # Update current network/IP based on source type
        self.current_network = source
        self.current_ip = "127.0.0.1" if source == "Webcam" else "Unknown" # Reset or set default

        # Update activity log
        self.update_activity(f"Camera source changed to: {source}")

    def _on_connect_webcam(self):
        """Handle Connect Webcam button click"""
        logger.info("Connect Webcam clicked")
        selected_source = self.camera_source_combo.currentText()
        camera_index_str = self.webcam_index_input.text().strip()
        if not camera_index_str:
            camera_index_str = "0"
            self.webcam_index_input.setText("0")

        try:
            index = int(camera_index_str)
            
            # Stop existing camera safely before creating a new one
            self._safe_stop_camera_source()
            
            self.current_network = "Local"
            self.current_camera_type = "Webcam"
            self.current_ip = "127.0.0.1"
            
            self.webcam_status_label.setText("Status: Connecting...")
            self.webcam_status_label.setStyleSheet("color: #FFA500;") # Orange
            
            # Instantiate new webcam source
            self.camera_source = WebcamSource(camera_id=index)
            
            if self.camera_source.start():
                actual_index = self.camera_source.camera_id
                self.webcam_index_input.setText(str(actual_index))
                
                if actual_index != index:
                    status_text = f"Status: Connected (Cam {actual_index} Fallback)"
                    self.update_activity(f"Camera index {index} failed; auto-connected to working camera index {actual_index}")
                else:
                    status_text = f"Status: Connected (Cam {actual_index})"
                    self.update_activity(f"Webcam index {actual_index} connected successfully")
                
                self.webcam_status_label.setText(status_text)
                self.webcam_status_label.setStyleSheet("color: #00FF00;") # Green

                resolution = self.camera_source.get_resolution()
                self.webcam_resolution_label.setText(f"Resolution: {resolution[0]}x{resolution[1]}")
                
                # Retrieve and display initial frame immediately
                first_frame = self.camera_source.get_frame()
                if first_frame is not None:
                    self.camera_panel.update_frame(first_frame)
                
                # Start display update timer for live feed preview
                if not self.timer.isActive():
                    self.timer.start(30)
            else:
                logger.error(f"Failed to connect to webcam index {index}")
                self.webcam_status_label.setText("Status: Failed")
                self.webcam_status_label.setStyleSheet("color: #FF0000;") # Red
                self.webcam_resolution_label.setText("Resolution: --")
                self.webcam_fps_label.setText("FPS: --")
                if self.camera_panel:
                    self.camera_panel.clear_display()
                QMessageBox.warning(self, "Camera Connection Error", 
                    f"Unable to open webcam (index {index}).\n\n"
                    "• Check if the camera is physically connected.\n"
                    "• Ensure camera access permissions are enabled in Windows settings.\n"
                    "• Verify no other app (e.g. Zoom, Skype, Teams) is using the camera.")
        except ValueError:
            self.update_activity("Invalid camera index")
            logger.error("Invalid camera index entered")
            QMessageBox.warning(self, "Input Error", "Camera index must be a number.")
            self.webcam_status_label.setText("Status: Invalid Index")
            self.webcam_status_label.setStyleSheet("color: #FF0000;") # Red

    def _on_validate_wired(self):
        """Handle Validate Network button click"""
        ip = self.wired_ip_input.text().strip()
        port = self.wired_port_input.text().strip()
        print(f"Validating network connection to {ip}:{port}")
        # In a real implementation, this would perform network validation
        self.update_activity(f"Network validation initiated for {ip}:{port}")
        logger.info(f"Network validation initiated for {ip}:{port}")

    def _on_connect_wired(self):
        """Handle Connect Wired Camera button click"""
        ip = self.wired_ip_input.text().strip()
        port = self.wired_port_input.text().strip()
        username = self.wired_username_input.text().strip()
        password = self.wired_password_input.text()
        rtsp_url = self.wired_rtsp_url_input.text().strip()
        camera_id = self.wired_camera_id_input.text().strip()
        
        print(f"Connecting to wired camera: {ip}:{port}")
        self.update_activity(f"Connecting to wired camera {ip}:{port}")
        logger.info(f"Connecting to wired camera {ip}:{port}")
        self.current_camera_type = "Wired Camera"
        self.current_network = "Wired"
        self.current_ip = ip
        # In a real implementation, this would establish the connection

    def _on_connect_wireless(self):
        """Handle Connect Wireless Camera button click"""
        ssid = self.wireless_ssid_input.text().strip()
        ip = self.wireless_ip_input.text().strip()
        url = self.wireless_url_input.text().strip()
        username = self.wireless_username_input.text().strip()
        password = self.wireless_password_input.text()
        camera_id = self.wireless_camera_id_input.text().strip()
        
        print(f"Connecting to wireless camera: {ip}")
        self.update_activity(f"Connecting to wireless camera {ip}")
        logger.info(f"Connecting to wireless camera {ip}")
        self.current_camera_type = "Wireless Camera"
        self.current_network = ssid or "Wireless"
        self.current_ip = ip
        # In a real implementation, this would establish the connection

    def _on_connect_rtsp(self):
        """Handle Connect RTSP Stream button click"""
        url = self.rtsp_url_input.text().strip()
        username = self.rtsp_username_input.text().strip()
        password = self.rtsp_password_input.text()
        stream_name = self.rtsp_stream_name_input.text().strip()
        
        print(f"Connecting to RTSP stream: {url}")
        self.update_activity(f"Connecting to RTSP stream {url}")
        logger.info(f"Connecting to RTSP stream {url}")
        self.current_camera_type = "RTSP Stream"
        self.current_network = "RTSP"
        self.current_ip = url.split('//')[-1].split('/')[0].split(':')[0] # Extract IP from URL
        # In a real implementation, this would establish the RTSP connection

    def _update_mobile_url_preview(self):
        """Update stream URL preview dynamically based on input"""
        ip = self.mobile_ip_input.text().strip() or "IP"
        port = self.mobile_port_input.text().strip() or "PORT"
        self.mobile_url_preview.setText(f"http://{ip}:{port}/video")

    def _on_connect_mobile(self):
        """Handle Connect Mobile Camera button click"""
        ip = self.mobile_ip_input.text().strip()
        port = self.mobile_port_input.text().strip()
        
        if not ip:
            QMessageBox.warning(self, "Validation Error", "Mobile IP Address is required.")
            return
        if not port:
            port = "8080"
            self.mobile_port_input.setText(port)
            
        url = f"http://{ip}:{port}/video"
        print(f"Connecting to mobile camera: {url}")
        self.update_activity(f"Connecting to mobile camera {url}")
        logger.info(f"Connecting to mobile camera {url}")
        
        self.mobile_status_label.setText("RECONNECTING")
        self.mobile_status_label.setStyleSheet("color: #FFA500; font-weight: bold; font-size: 12px; margin-left: 10px;")
        
        from ..utils.network_utils import validate_camera_connection
        self.update_activity(f"Validating connection to {ip}...")
        
        conn_res = validate_camera_connection(ip, "wireless")
        if conn_res['connection_status'] == 'INVALID_IP':
            self.mobile_status_label.setText("STREAM ERROR")
            self.mobile_status_label.setStyleSheet("color: #FF0000; font-weight: bold; font-size: 12px; margin-left: 10px;")
            QMessageBox.warning(self, "Invalid IP Address", "Please enter a valid IP address.")
            self.log_event("Connection Failed", "Invalid IP Address format entered", current_crowd_count=0, current_threat_level="NORMAL")
            return
            
        # Stop existing camera safely
        self._safe_stop_camera_source()
            
        self.current_camera_type = "Mobile Camera"
        self.current_network = "Wireless"
        self.current_ip = ip
        
        # Instantiate Mobile Camera Source
        self.camera_source = MobileCameraSource(ip_address=ip, port=int(port))
        
        if self.camera_source.start():
            self.timer.start(30)   # ~33 FPS
            self.mobile_status_label.setText("CONNECTED")
            self.mobile_status_label.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 12px; margin-left: 10px;")
            self.connect_mobile_btn.setEnabled(False)
            self.disconnect_mobile_btn.setEnabled(True)
            self.update_activity(f"Mobile Camera connected successfully")
            logger.info(f"Mobile Camera connected successfully")
            self.log_event("Mobile Camera Connected", f"Mobile stream connected to {url}", current_crowd_count=0, current_threat_level="NORMAL")
        else:
            self.mobile_status_label.setText("STREAM ERROR")
            self.mobile_status_label.setStyleSheet("color: #FF0000; font-weight: bold; font-size: 12px; margin-left: 10px;")
            self.update_activity(f"Failed to connect to Mobile Camera at {url}")
            logger.error(f"Failed to connect to Mobile Camera at {url}")
            self.log_event("Connection Failed", f"Mobile stream unreachable at {url}", current_crowd_count=0, current_threat_level="NORMAL")
            QMessageBox.warning(self, "Connection Error", f"Failed to connect to mobile camera at {url}.\nMake sure the app is running and your phone and PC are on the same Wi-Fi network.")

    def _on_disconnect_mobile(self):
        """Handle Disconnect Mobile Camera button click"""
        self._safe_stop_camera_source()
        self.mobile_status_label.setText("DISCONNECTED")
        self.mobile_status_label.setStyleSheet("color: #AAAAAA; font-weight: bold; font-size: 12px; margin-left: 10px;")
        self.connect_mobile_btn.setEnabled(True)
        self.disconnect_mobile_btn.setEnabled(False)
        self.update_activity("Mobile Camera disconnected")
        logger.info("Mobile Camera disconnected")
        self.log_event("Mobile Camera Disconnected", "User disconnected mobile stream", current_crowd_count=0, current_threat_level="NORMAL")
        if self.camera_panel:
            self.camera_panel.clear_display()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def closeEvent(self, event):
        if self.camera_source:
            self.camera_source.stop()
        if self.timer:
            self.timer.stop()
        event.accept()

    def _on_start_monitoring(self):
        """Handle Start Monitoring button click"""
        logger.info("Start Monitoring workflow initiated")
        try:
            # 1. Prevent Duplicate Threads
            if self.is_monitoring:
                logger.info("Monitoring already active")
                self.update_activity("Monitoring already active")
                return

            # 2. Validate Inputs with safe defaults
            location = self.location_input.text().strip() or "Default Site"
            area = self.area_input.text().strip() or "Main Zone"
            camera_id = self.camera_id_input.text().strip() or "0"

            self.monitoring_location = location
            self.monitoring_area = area
            self.monitoring_camera_id = camera_id
            
            # 3. Check for a valid camera source
            is_running = False
            if self.camera_source and hasattr(self.camera_source, 'is_running'):
                is_running = self.camera_source.is_running
            
            # Auto-connect webcam if camera source selected is Webcam and not connected yet
            if not is_running and self.camera_source_combo.currentText() == "Webcam":
                logger.info("Auto-connecting webcam before starting monitoring...")
                self._on_connect_webcam()
                if self.camera_source and hasattr(self.camera_source, 'is_running'):
                    is_running = self.camera_source.is_running

            if not is_running:
                logger.error("No active camera source. Please connect a camera first.")
                QMessageBox.warning(self, "Camera Not Ready", "Please connect a working camera source before starting monitoring.")
                return

            # 4. Start Pipeline and Timers
            self.start_pipeline_thread()
            if not self.timer.isActive():
                self.timer.start(30) # ~33 FPS

            self.is_monitoring = True
            self._monitoring_start_time = time.time()
            
            # 5. Update UI State
            self.start_button.setEnabled(False)
            self.start_button.setText("Monitoring Active")
            self.stop_button.setEnabled(True)
            
            logger.info("Monitoring started successfully")
            self.statusBar().showMessage("Monitoring Started Successfully", 5000)
            self.log_event("Monitoring Started",
                f"Active session at {location} ({area}) using Camera {camera_id}", current_crowd_count=0, current_threat_level="NORMAL")

        except Exception as e:
            logger.error(f"Critical error in _on_start_monitoring: {e}")
            traceback.print_exc()
            self.start_button.setEnabled(True)
            self.start_button.setText("Start Monitoring")
            self.stop_button.setEnabled(False)
            self.update_activity(f"Startup error: {str(e)}")

    def _on_stop_monitoring(self):
        """Handle Stop Monitoring button click"""
        logger.info("Stop Monitoring clicked")
        try:
            # Stop background pipeline thread
            if hasattr(self, 'pipeline_running'):
                self.pipeline_running = False

            self.is_monitoring = False
            
            # Reset UI state
            self.start_button.setEnabled(True)
            self.start_button.setText("Start Monitoring")
            self.stop_button.setEnabled(False)
            
            self.statusBar().showMessage("Monitoring Stopped", 5000)
            self.log_event("Monitoring Stopped", "User terminated monitoring session", current_crowd_count=0, current_threat_level="NORMAL")
            
            self._monitoring_start_time = 0.0
            
            # Reset metrics display
            self._update_metrics({}, 0, 0.0)
            
        except Exception as e:
            logger.error(f"Error in _on_stop_monitoring: {e}")
            traceback.print_exc()
        
        # Reset Analytics Tab visuals
        if hasattr(self, 'analytics_history'):
            self._reset_analytics_display()

    def _create_analytics_card(self, title):
        card = QFrame()
        card.setStyleSheet("background-color: #151515; border: 1px solid #333; border-radius: 5px;")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)
        t_label = QLabel(title)
        t_label.setStyleSheet("color: #888; font-size: 9px; font-weight: bold;")
        t_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v_label = QLabel("--")
        v_label.setStyleSheet("color: #FFFFFF; font-size: 18px; font-weight: bold;")
        v_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(t_label)
        layout.addWidget(v_label)
        return card, v_label

    def _add_analytics_event(self, event_type, details):
        ts = time.strftime("%H:%M:%S")
        entry = f"<span style='color: #00FFFF;'>[{ts}]</span> <b>{event_type}</b>: {details}"
        self.analytics_history.insert(0, entry)
        if len(self.analytics_history) > 40: self.analytics_history.pop()
        self.analytics_history_text.setText("<br>".join(self.analytics_history))

    def _reset_analytics_display(self):
        """Resets the analytics dashboard to initial state"""
        self.analytics_history.clear()
        if hasattr(self, 'risk_history'):
            self.risk_history.clear()
        if hasattr(self, 'analytics_history_text'):
            self.analytics_history_text.setText("Awaiting data stream...")
        if hasattr(self, 'ana_crowd_val'):
            self.ana_crowd_val.setText("--")
        if hasattr(self, 'ana_motion_val'):
            self.ana_motion_val.setText("--")
        if hasattr(self, 'ana_threat_val'):
            self.ana_threat_val.setText("--")
        if hasattr(self, 'ana_risk_val'):
            self.ana_risk_val.setText("--")
        if hasattr(self, 'ana_accel_val'):
            self.ana_accel_val.setText("--")
        if hasattr(self, 'ana_dir_val'):
            self.ana_dir_val.setText("--")
        self._update_analytics_tab({}, {}, 0, 0.0) # Reset bars

    def _update_analytics_tab(self, threat, motion_data, crowd_count, fps):
        if not hasattr(self, 'ana_crowd_val'): return
        if self.ana_crowd_val:
            self.ana_crowd_val.setText(str(crowd_count))
        if hasattr(self, 'ana_motion_val') and self.ana_motion_val:
            self.ana_motion_val.setText(f"{motion_data.get('motion_score', 0)*100:.1f}%")
        level = threat.get('threat_level', 'NORMAL')
        if hasattr(self, 'ana_threat_val') and self.ana_threat_val:
            self.ana_threat_val.setText(level) # This is fine
            self.ana_threat_val.setStyleSheet(f"color: {_LEVEL_COLORS.get(level, '#FFF')}; font-size: 18px; font-weight: bold;")
        if hasattr(self, 'ana_accel_val') and self.ana_accel_val:
            self.ana_accel_val.setText(f"{threat.get('accel_score', 0)*100:.1f}%")
        if hasattr(self, 'ana_dir_val') and self.ana_dir_val:
            self.ana_dir_val.setText(f"{threat.get('dir_variance', 0):.3f}")

        # Update Bars (Task 3: bar_conf is updated via timer and history buffer)
        motion_val = int(motion_data.get('motion_score', 0) * 100)
        accel_val = int(threat.get('accel_score', 0) * 100)
        
        if hasattr(self, 'bar_motion') and self.bar_motion:
            self.bar_motion.setValue(motion_val) # Relocated: Was outside method
        if hasattr(self, 'bar_accel') and self.bar_accel:
            self.bar_accel.setValue(accel_val) # Relocated: Was outside method
        
        # Cache density score for DB logging # Relocated: Was outside method
        self._last_density_score = threat.get('density_score', 0.0)

        # Task 6: Visual Feedback Styling
        if hasattr(self, 'bar_motion'):
            self._apply_bar_style(self.bar_motion, motion_val)
        if hasattr(self, 'bar_accel'):
            self._apply_bar_style(self.bar_accel, accel_val)

        now = time.time()
        if motion_data.get('motion_score', 0) > 0.4 and (now - self.last_analytics_log_time > 10.0):
            self._add_analytics_event("Motion Spike", f"Intensity: {motion_data.get('motion_score', 0)*100:.1f}%")
            self.last_analytics_log_time = now
        if level != self.prev_threat_level and level != "NORMAL":
            self._add_analytics_event("Threat Update", f"Escalated to {level}")