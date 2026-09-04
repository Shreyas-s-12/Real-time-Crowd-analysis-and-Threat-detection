from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QComboBox, QLineEdit, QHeaderView, QFormLayout, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt # Keep Qt import
from real_time_crowd_analysis.utils.database import db_manager # Absolute import
from real_time_crowd_analysis.utils.logger import setup_logger # Absolute import

logger = setup_logger("camera_manager")

class CameraManagerPanel(QWidget):
    def __init__(self, parent_dashboard=None):
        super().__init__()
        self.dashboard = parent_dashboard
        self.setup_ui()
        self.load_cameras()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("Camera Management")
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #00FFFF;")
        layout.addWidget(header)

        # Splitter-like layout: Top (Registration), Bottom (List)
        reg_frame = QFrame()
        reg_frame.setStyleSheet("background-color: #1a1a1a; border-radius: 4px; border: 1px solid #333;")
        reg_layout = QVBoxLayout(reg_frame)
        
        reg_title = QLabel("Register New Camera")
        reg_title.setStyleSheet("font-weight: bold; color: #00FFFF; border: none;")
        reg_layout.addWidget(reg_title)

        form = QFormLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g., Main Entrance 01")
        
        self.type_input = QComboBox()
        self.type_input.addItems(["Webcam", "Wired CCTV", "Wireless CCTV", "RTSP Stream", "IP Camera", "Mobile Camera"])
        
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("e.g., 192.168.1.100")
        
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("e.g., rtsp://user:pass@ip:554/stream")
        
        self.loc_input = QLineEdit()
        self.loc_input.setPlaceholderText("e.g., Mysore Palace")
        
        self.area_input = QLineEdit()
        self.area_input.setPlaceholderText("e.g., North Gate")

        form.addRow("Camera Name:", self.name_input)
        form.addRow("Camera Type:", self.type_input)
        form.addRow("IP Address:", self.ip_input)
        form.addRow("RTSP URL:", self.rtsp_input)
        form.addRow("Location:", self.loc_input)
        form.addRow("Area/Zone:", self.area_input)
        
        reg_layout.addLayout(form)

        save_btn = QPushButton("Save Camera Registration")
        save_btn.setStyleSheet("background-color: #0066CC; color: white; font-weight: bold; padding: 10px;")
        save_btn.clicked.connect(self.save_camera)
        reg_layout.addWidget(save_btn)

        layout.addWidget(reg_frame)

        # Camera List
        list_label = QLabel("Registered Cameras")
        list_label.setStyleSheet("font-weight: bold; color: #00FFFF;")
        layout.addWidget(list_label)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Name", "Type", "Location", "Area", "Status", "Connect", "Delete"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("""
            QTableWidget { background-color: #121212; color: #E0E0E0; border: 1px solid #333; }
            QHeaderView::section { background-color: #222; color: white; padding: 5px; font-weight: bold; }
        """)
        layout.addWidget(self.table)

    def load_cameras(self):
        cameras = db_manager.get_cameras()
        self.table.setRowCount(0)
        for i, cam in enumerate(cameras):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(cam['name']))
            self.table.setItem(i, 1, QTableWidgetItem(cam['type']))
            self.table.setItem(i, 2, QTableWidgetItem(cam['location']))
            self.table.setItem(i, 3, QTableWidgetItem(cam['area']))
            
            status_item = QTableWidgetItem(cam['status'])
            status_item.setForeground(Qt.GlobalColor.green if cam['status'] == 'ONLINE' else Qt.GlobalColor.red)
            self.table.setItem(i, 4, status_item)

            connect_btn = QPushButton("Connect")
            connect_btn.setStyleSheet("background-color: #00AA00; color: white; font-size: 11px;")
            connect_btn.clicked.connect(lambda _, c=cam: self.connect_camera(c))
            self.table.setCellWidget(i, 5, connect_btn)

            delete_btn = QPushButton("Delete")
            delete_btn.setStyleSheet("background-color: #8B0000; color: white; font-size: 11px;")
            delete_btn.clicked.connect(lambda _, c_id=cam['id']: self.delete_camera(c_id))
            self.table.setCellWidget(i, 6, delete_btn)

    def save_camera(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Camera name is required.")
            return

        data = {
            'name': name,
            'type': self.type_input.currentText(),
            'ip_address': self.ip_input.text().strip(),
            'rtsp_url': self.rtsp_input.text().strip(),
            'location': self.loc_input.text().strip(),
            'area': self.area_input.text().strip()
        }
        
        if db_manager.insert_camera(data):
            self.name_input.clear()
            self.ip_input.clear()
            self.rtsp_input.clear()
            self.loc_input.clear()
            self.area_input.clear()
            self.load_cameras()
            logger.info(f"Registered camera: {name}")
        else:
            QMessageBox.critical(self, "Database Error", "Failed to register camera. Name might already exist.")

    def connect_camera(self, camera):
        """Signal the dashboard to switch to this camera"""
        if self.dashboard:
            # Update dashboard fields and trigger connection
            self.dashboard.location_input.setText(camera['location'])
            self.dashboard.area_input.setText(camera['area'])
            self.dashboard.camera_id_input.setText(str(camera['id']))
            
            # Select correct source type in combo
            source_map = {
                "Webcam": "Webcam",
                "Wired CCTV": "Wired Camera",
                "Wireless CCTV": "Wireless Camera",
                "RTSP Stream": "RTSP Camera",
                "IP Camera": "RTSP Camera",
                "Mobile Camera": "Mobile Camera"
            }
            target_source = source_map.get(camera['type'], "Webcam")
            index = self.dashboard.camera_source_combo.findText(target_source)
            if index >= 0:
                self.dashboard.camera_source_combo.setCurrentIndex(index)
            
            # Fill URL if RTSP or Mobile Camera
            if target_source == "RTSP Camera":
                self.dashboard.rtsp_url_input.setText(camera['rtsp_url'])
            elif target_source == "Wired Camera":
                self.dashboard.wired_ip_input.setText(camera['ip_address'])
                self.dashboard.wired_rtsp_url_input.setText(camera['rtsp_url'])
            elif target_source == "Mobile Camera":
                ip = camera['ip_address'] or "192.168.1.5"
                port = "8080"
                if camera['rtsp_url'] and ":" in camera['rtsp_url']:
                    try:
                        parts = camera['rtsp_url'].split(":")
                        if len(parts) >= 3:
                            port = parts[2].split("/")[0]
                    except:
                        pass
                self.dashboard.mobile_ip_input.setText(ip)
                self.dashboard.mobile_port_input.setText(port)
            
            # Switch to Dashboard view
            self.dashboard.btn_nav_dashboard.click()
            
            # Trigger connection automatically
            if target_source == "Webcam":
                self.dashboard._on_connect_webcam()
            elif target_source == "Wired Camera":
                self.dashboard._on_connect_wired()
            elif target_source == "Wireless Camera":
                self.dashboard._on_connect_wireless()
            elif target_source == "RTSP Camera":
                self.dashboard._on_connect_rtsp()
            elif target_source == "Mobile Camera":
                self.dashboard._on_connect_mobile()

            QMessageBox.information(self, "Camera Switching", f"Switching to {camera['name']}...")


    def delete_camera(self, camera_id):
        if QMessageBox.question(self, "Confirm Delete", "Are you sure you want to delete this camera?") == QMessageBox.StandardButton.Yes:
            if db_manager.delete_camera(camera_id):
                self.load_cameras()
