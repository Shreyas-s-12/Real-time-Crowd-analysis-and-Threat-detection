"""
CSV Manager for Real-Time Crowd Analysis and Threat Detection
"""

import csv
import os
from datetime import datetime # Keep datetime import
from real_time_crowd_analysis.utils.config import config # Absolute import
from real_time_crowd_analysis.utils.logger import setup_logger # Absolute import

logger = setup_logger("csv_manager")

class CSVManager:
    """Manages CSV file operations for logging"""
    
    def __init__(self):
        self.datasets_dir = config.DATASETS_DIR
        self.ensure_directories()
        
        # CSV file paths
        self.crowd_logs_path = os.path.join(self.datasets_dir, "crowd_logs.csv")
        self.threat_logs_path = os.path.join(self.datasets_dir, "threat_logs.csv")
        self.network_logs_path = os.path.join(self.datasets_dir, "network_logs.csv")
        
        # Initialize CSV files with headers
        self.init_csv_files()
    
    def ensure_directories(self):
        """Ensure datasets directory exists"""
        os.makedirs(self.datasets_dir, exist_ok=True)
    
    def init_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        files_and_headers = [
            (self.crowd_logs_path, [
                'timestamp', 'location_name', 'area_name', 'camera_id', 'camera_type',
                'network_name', 'ip_address', 'crowd_count', 'threat_status', 
                'density_level', 'session_duration', 'connection_status', 'reconnect_attempts'
            ]),
            (self.threat_logs_path, [
                'timestamp', 'location_name', 'area_name', 'camera_id', 'threat_type',
                'confidence', 'frame_saved', 'description'
            ]),
            (self.network_logs_path, [
                'timestamp', 'camera_type', 'network_name', 'ip_address', 
                'connection_status', 'signal_strength', 'latency'
            ])
        ]
        
        for file_path, headers in files_and_headers:
            if not os.path.exists(file_path):
                try:
                    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(headers)
                    logger.info(f"Initialized CSV file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to initialize CSV file {file_path}: {e}")
    
    def log_crowd_data(self, data: dict):
        """Log crowd analytics data to CSV"""
        try:
            row = [
                datetime.now().isoformat(),
                data.get('location_name', ''),
                data.get('area_name', ''),
                data.get('camera_id', ''),
                data.get('camera_type', ''),
                data.get('network_name', ''),
                data.get('ip_address', ''),
                data.get('crowd_count', 0),
                data.get('threat_status', ''),
                data.get('density_level', ''),
                data.get('session_duration', 0.0),
                data.get('connection_status', ''),
                data.get('reconnect_attempts', 0)
            ]
            
            with open(self.crowd_logs_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
                
        except Exception as e:
            logger.error(f"Failed to log crowd data to CSV: {e}")
    
    def log_threat_event(self, data: dict):
        """Log threat event data to CSV"""
        try:
            row = [
                datetime.now().isoformat(),
                data.get('location_name', ''),
                data.get('area_name', ''),
                data.get('camera_id', ''),
                data.get('threat_type', ''),
                data.get('confidence', 0.0),
                data.get('frame_saved', ''),
                data.get('description', '')
            ]
            
            with open(self.threat_logs_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
                
        except Exception as e:
            logger.error(f"Failed to log threat event to CSV: {e}")
    
    def log_network_data(self, data: dict):
        """Log network data to CSV"""
        try:
            row = [
                datetime.now().isoformat(),
                data.get('camera_type', ''),
                data.get('network_name', ''),
                data.get('ip_address', ''),
                data.get('connection_status', ''),
                data.get('signal_strength', 0.0),
                data.get('latency', 0.0)
            ]
            
            with open(self.network_logs_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
                
        except Exception as e:
            logger.error(f"Failed to log network data to CSV: {e}")
    
    def get_crowd_logs(self, limit: int = 1000):
        """Get recent crowd logs from CSV"""
        try:
            if not os.path.exists(self.crowd_logs_path):
                return []
            
            with open(self.crowd_logs_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                return rows[-limit:] if len(rows) > limit else rows
        except Exception as e:
            logger.error(f"Failed to read crowd logs: {e}")
            return []
    
    def get_threat_logs(self, limit: int = 1000):
        """Get recent threat logs from CSV"""
        try:
            if not os.path.exists(self.threat_logs_path):
                return []
            
            with open(self.threat_logs_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                return rows[-limit:] if len(rows) > limit else rows
        except Exception as e:
            logger.error(f"Failed to read threat logs: {e}")
            return []
    
    def get_network_logs(self, limit: int = 1000):
        """Get recent network logs from CSV"""
        try:
            if not os.path.exists(self.network_logs_path):
                return []
            
            with open(self.network_logs_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                return rows[-limit:] if len(rows) > limit else rows
        except Exception as e:
            logger.error(f"Failed to read network logs: {e}")
            return []

# Global CSV manager instance
csv_manager = CSVManager()