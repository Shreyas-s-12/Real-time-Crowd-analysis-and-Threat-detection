"""
Alerts Panel Module for Real-Time Crowd Analysis and Threat Detection
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor
from typing import Dict, Optional
from real_time_crowd_analysis.ui.theme import get_stylesheet
from real_time_crowd_analysis.utils.logger import setup_logger
from real_time_crowd_analysis.utils.alerts import trigger_alert
from real_time_crowd_analysis.utils.config import config
import hashlib
import time

logger = setup_logger("alerts_panel")


class AlertsPanel(QWidget):
    """Panel for displaying threat alerts and notifications"""

    def __init__(self):
        super().__init__()
        self.alerts = []          # list of alert dicts
        self.max_alerts = 100

        # Duplicate suppression: hash -> last display timestamp
        self._seen_hashes: Dict[str, float] = {}
        self._dedup_window: float = config.ALERT_COOLDOWN_SECONDS

        # Setup UI
        self.setup_ui()

        # Auto-clear old alerts every 30 s
        self.clear_timer = QTimer()
        self.clear_timer.timeout.connect(self.clear_old_alerts)
        self.clear_timer.start(30_000)

        logger.info("Alerts panel initialized")

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title_label = QLabel("Threat Alerts & Notifications")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #FF6B6B;
            margin: 5px;
        """)
        layout.addWidget(title_label)
        
        # Alert counter
        self.alert_counter_label = QLabel("Active Alerts: 0")
        self.alert_counter_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.alert_counter_label.setStyleSheet("""
            font-size: 12px;
            color: #FFA500;
            margin: 2px;
        """)
        layout.addWidget(self.alert_counter_label)
        
        # Scroll area for alerts
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(QFrame.Shape.NoFrame)
        
        # Alerts container
        self.alerts_container = QWidget()
        self.alerts_layout = QVBoxLayout(self.alerts_container)
        self.alerts_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.alerts_layout.setSpacing(3)
        
        scroll_area.setWidget(self.alerts_container)
        layout.addWidget(scroll_area)
        
        # Clear button
        from PyQt6.QtWidgets import QPushButton
        clear_button = QPushButton("Clear All Alerts")
        clear_button.setStyleSheet("""
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
        clear_button.clicked.connect(self.clear_alerts)
        layout.addWidget(clear_button)

    def add_alert(self, message: str, threat_level: str = "INFO",
                  metrics: Optional[Dict] = None):
        """
        Add a structured alert card to the panel.

        Args:
            message     : Primary alert headline (e.g. threat type string)
            threat_level: NORMAL | SUSPICIOUS | HIGH RISK | CRITICAL | INFO
            metrics     : Optional dict with keys:
                            confidence_pct, motion_score, accel_score,
                            dir_variance, crowd_count
        """
        try:
            # ---- Duplicate suppression (UI level) ----
            sig = f"{message}|{threat_level}"
            sig_hash = hashlib.md5(sig.encode()).hexdigest()[:8]
            now = time.time()
            if sig_hash in self._seen_hashes:
                if now - self._seen_hashes[sig_hash] < self._dedup_window:
                    return  # suppress duplicate
            self._seen_hashes[sig_hash] = now

            # ---- Colour by threat tier ----
            color_map = {
                'NORMAL':    '#00FF00',
                'SUSPICIOUS':'#FFFF00',
                'HIGH RISK': '#FF6600',
                'CRITICAL':  '#FF00FF',
                # Legacy compat
                'NONE':   '#00FF00',
                'LOW':    '#FFFF00',
                'MEDIUM': '#FFA500',
                'HIGH':   '#FF0000',
                'INFO':   '#00FFFF',
            }
            bg_color   = color_map.get(threat_level.upper(), '#FFFFFF')
            dark_levels = {'NORMAL', 'SUSPICIOUS', 'NONE', 'LOW'}
            text_color = '#000000' if threat_level.upper() in dark_levels else '#FFFFFF'

            # ---- Build card widget ----
            alert_frame = QFrame()
            alert_frame.setFrameStyle(QFrame.Shape.StyledPanel)
            alert_frame.setLineWidth(1)
            alert_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: rgba{self._hex_to_rgba(bg_color, 0.15)};
                    border-left: 4px solid {bg_color};
                    border-radius: 4px;
                    margin: 2px;
                    padding: 6px;
                }}
            """)

            alert_layout = QVBoxLayout(alert_frame)
            alert_layout.setContentsMargins(6, 4, 6, 4)
            alert_layout.setSpacing(2)

            # Header row: timestamp + level badge
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S")
            header_layout = QHBoxLayout()

            time_lbl = QLabel(f"[{timestamp}]")
            time_lbl.setStyleSheet("font-size: 10px; color: #888888;")
            header_layout.addWidget(time_lbl)
            header_layout.addStretch()

            level_lbl = QLabel(threat_level.upper())
            level_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            level_lbl.setStyleSheet(f"""
                background-color: {bg_color};
                color: {'#000000' if threat_level.upper() in dark_levels else '#FFFFFF'};
                font-size: 10px; font-weight: bold;
                border-radius: 3px; padding: 2px 6px;
            """)
            header_layout.addWidget(level_lbl)
            alert_layout.addLayout(header_layout)

            # Headline message
            msg_lbl = QLabel(f"▶ {message}")
            msg_lbl.setWordWrap(True)
            msg_lbl.setStyleSheet(f"""
                font-size: 12px;
                color: {bg_color};
                font-weight: bold;
                margin-top: 2px;
            """)
            alert_layout.addWidget(msg_lbl)

            # ---- Metrics detail row ----
            if metrics:
                # conf_pct  = metrics.get('confidence_pct',  'N/A') # Old confidence, now using risk_score_pct
                ms_val    = metrics.get('motion_score',    0.0) # This is fine
                ac_val    = metrics.get('accel_score',     0.0)
                dir_val   = metrics.get('dir_variance',    0.0)
                count_val = metrics.get('crowd_count',     0)

                def score_label(v: float) -> str:
                    if v >= 0.7: return 'HIGH'
                    if v >= 0.4: return 'MED'
                    return 'LOW'

                detail_text = ( # Changed from Confidence to Risk Score
                    f"Risk Score: {metrics.get('risk_score_pct', 'N/A')}   "
                    f"Motion: {score_label(ms_val)}   "
                    f"Accel: {score_label(ac_val)}   "
                    f"Dir Chaos: {dir_val:.2f}   "
                    f"Persons: {count_val}"
                )
                detail_lbl = QLabel(detail_text)
                detail_lbl.setWordWrap(True)
                detail_lbl.setStyleSheet("font-size: 10px; color: #CCCCCC; margin-top: 1px;")
                alert_layout.addWidget(detail_lbl)

            # Insert at top (newest first)
            self.alerts_layout.insertWidget(0, alert_frame)
            self.alerts.insert(0, {
                'message': message,
                'threat_level': threat_level,
                'timestamp': now,
                'widget': alert_frame
            })

            # Cap alert count
            if len(self.alerts) > self.max_alerts:
                oldest = self.alerts.pop()
                oldest['widget'].deleteLater()

            self.update_alert_counter()

            # System-level alert for severe threats
            if threat_level.upper() in ('HIGH RISK', 'CRITICAL', 'HIGH'):
                conf = float((metrics or {}).get('risk_score_pct', '0%').rstrip('%')) / 100.0
                trigger_alert(threat_type=message,
                              confidence=conf if conf > 0 else 0.7)

            logger.info(f"Alert added: [{threat_level}] {message}")

        except Exception as e:
            logger.error(f"Error adding alert: {e}")

    def clear_alerts(self):
        """Clear all alerts"""
        try:
            # Remove all alert widgets
            for alert in self.alerts:
                alert['widget'].deleteLater()
            
            # Clear alerts list
            self.alerts.clear()
            
            # Update counter
            self.update_alert_counter()
            
            logger.info("All alerts cleared")
            
        except Exception as e:
            logger.error(f"Error clearing alerts: {e}")

    def clear_old_alerts(self):
        """Clear alerts older than 1 hour"""
        try:
            current_time = time.time()
            one_hour_ago = current_time - 3600  # 1 hour in seconds
            
            alerts_to_remove = []
            for alert in self.alerts:
                if alert['timestamp'] < one_hour_ago:
                    alerts_to_remove.append(alert)
            
            for alert in alerts_to_remove:
                self.alerts.remove(alert)
                alert['widget'].deleteLater()
            
            if alerts_to_remove:
                logger.info(f"Cleared {len(alerts_to_remove)} old alerts")
                self.update_alert_counter()
                
        except Exception as e:
            logger.error(f"Error clearing old alerts: {e}")

    def update_alert_counter(self):
        """Update the alert counter display"""
        count = len(self.alerts)
        self.alert_counter_label.setText(f"Active Alerts: {count}")
        
        # Change color based on count
        if count == 0:
            color = "#00FF00"
        elif count < 5:
            color = "#FFA500"
        else:
            color = "#FF0000"
            
        self.alert_counter_label.setStyleSheet(f"""
            font-size: 12px;
            color: {color};
            margin: 2px;
            font-weight: bold;
        """)

    def _hex_to_rgba(self, hex_color: str, alpha: float) -> tuple:
        """Convert hex color to rgba tuple"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        elif len(hex_color) != 6:
            return (255, 255, 255, alpha)  # Default to white
        
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b, alpha)