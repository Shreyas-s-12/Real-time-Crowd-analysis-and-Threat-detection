import csv
import json
from datetime import datetime
from PyQt6.QtWidgets import ( # Keep QtWidgets imports
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QComboBox, QLineEdit, QHeaderView, QFileDialog, QMessageBox, QFrame # Keep QtWidgets imports
)
from PyQt6.QtCore import Qt, QTimer
from real_time_crowd_analysis.utils.database import db_manager

class HistoryPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.load_data)
        self.refresh_timer.start(5000) # Refresh every 5 seconds
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header_label = QLabel("Surveillance Event History")
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FFFF;")
        layout.addWidget(header_label)

        # Filter Container
        filter_frame = QFrame()
        filter_frame.setStyleSheet("background-color: #1a1a1a; border-radius: 4px; border: 1px solid #333;")
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(10, 10, 10, 10)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by event type...")
        self.search_input.textChanged.connect(self.load_data)
        self.search_input.setStyleSheet("padding: 8px; border-radius: 2px; background: #000; color: #00FFFF; border: 1px solid #00FFFF;")
        filter_layout.addWidget(self.search_input)

        self.threat_filter = QComboBox()
        self.threat_filter.addItems(["All Threats", "NORMAL", "SUSPICIOUS", "HIGH RISK", "CRITICAL", "WEAPON"])
        self.threat_filter.currentTextChanged.connect(self.load_data)
        self.threat_filter.setStyleSheet("padding: 8px; border-radius: 2px; background: #222; color: white;")
        filter_layout.addWidget(self.threat_filter)

        self.camera_filter = QComboBox()
        self.camera_filter.addItems(["All Cameras", "Webcam", "Wired Camera", "Wireless Camera", "RTSP Camera", "Mobile Camera"])
        self.camera_filter.currentTextChanged.connect(self.load_data)
        self.camera_filter.setStyleSheet("padding: 8px; border-radius: 2px; background: #222; color: white;")
        filter_layout.addWidget(self.camera_filter)

        export_csv_btn = QPushButton("Export CSV")
        export_csv_btn.setStyleSheet("background-color: #333; color: white; padding: 8px 16px; border: 1px solid #555;")
        export_csv_btn.clicked.connect(self.export_csv)
        filter_layout.addWidget(export_csv_btn)

        export_json_btn = QPushButton("Export JSON")
        export_json_btn.setStyleSheet("background-color: #333; color: white; padding: 8px 16px; border: 1px solid #555;")
        export_json_btn.clicked.connect(self.export_json)
        filter_layout.addWidget(export_json_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setStyleSheet("background-color: #0066CC; color: white; padding: 8px 16px; border-radius: 2px;")
        refresh_btn.clicked.connect(self.load_data)
        filter_layout.addWidget(refresh_btn)

        layout.addWidget(filter_frame)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Timestamp", "Event", "Threat Level", "Camera", "Location", "Confidence", "Motion", "Details"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #0a0a0a;
                color: #B0B0B0;
                border: 1px solid #333;
                gridline-color: #222;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #00FFFF;
                padding: 8px;
                border: 1px solid #333;
                font-weight: bold;
                text-transform: uppercase;
            }
        """)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        
        self.load_data()

    def load_data(self):
        filters = {}
        if self.threat_filter.currentText() != "All Threats":
            filters['threat_level'] = self.threat_filter.currentText()
        if self.camera_filter.currentText() != "All Cameras":
            cam_val = self.camera_filter.currentText()
            if cam_val == "RTSP Camera":
                filters['camera_source'] = "RTSP Stream"
            else:
                filters['camera_source'] = cam_val
        if self.search_input.text():
            filters['event_type'] = self.search_input.text()

        records = db_manager.get_event_history(limit=200, filters=filters)
        
        self.table.setRowCount(0)
        for row_idx, record in enumerate(records):
            self.table.insertRow(row_idx)
            
            # Timestamp
            ts_item = QTableWidgetItem(str(record.get('timestamp', '')))
            self.table.setItem(row_idx, 0, ts_item)
            
            # Event
            event_item = QTableWidgetItem(str(record.get('event_type', '')))
            self.table.setItem(row_idx, 1, event_item)
            
            # Threat Level
            level = str(record.get('threat_level', 'NORMAL'))
            level_item = QTableWidgetItem(level)
            if level == "NORMAL": level_item.setForeground(Qt.GlobalColor.green)
            elif level == "SUSPICIOUS": level_item.setForeground(Qt.GlobalColor.yellow)
            elif level == "HIGH RISK": level_item.setForeground(Qt.GlobalColor.red)
            elif level in ["CRITICAL", "WEAPON"]: 
                level_item.setForeground(Qt.GlobalColor.red)
                level_item.setBackground(Qt.GlobalColor.black)
            self.table.setItem(row_idx, 2, level_item)
            
            # Camera
            cam = f"{record.get('camera_source', '')} [#{record.get('camera_id', '')}]"
            self.table.setItem(row_idx, 3, QTableWidgetItem(cam))
            
            # Location
            loc = f"{record.get('location', 'N/A')} | {record.get('area', 'N/A')}"
            self.table.setItem(row_idx, 4, QTableWidgetItem(loc))
            
            # Confidence
            conf = float(record.get('confidence') or 0.0)
            self.table.setItem(row_idx, 5, QTableWidgetItem(f"{conf*100:.1f}%"))
            
            # Motion
            motion = float(record.get('motion_score') or 0.0)
            self.table.setItem(row_idx, 6, QTableWidgetItem(f"{motion*100:.1f}%"))
            
            # Notes
            self.table.setItem(row_idx, 7, QTableWidgetItem(str(record.get('notes', ''))))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export to CSV", "", "CSV Files (*.csv)")
        if not path: return
        try:
            records = db_manager.get_event_history(limit=1000)
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if records:
                    writer.writerow(records[0].keys())
                    for r in records: writer.writerow(r.values())
            QMessageBox.information(self, "Export Success", "History exported to CSV.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error: {e}")

    def export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export to JSON", "", "JSON Files (*.json)")
        if not path: return
        try:
            records = db_manager.get_event_history(limit=1000)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=4)
            QMessageBox.information(self, "Export Success", "History exported to JSON.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error: {e}")
