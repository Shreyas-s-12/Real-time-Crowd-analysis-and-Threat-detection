"""
Logging Utility for Real-Time Crowd Analysis and Threat Detection
"""

import logging
import os
from datetime import datetime # Keep datetime import
from real_time_crowd_analysis.utils.config import config # Absolute import

def setup_logger(name: str = "crowd_analysis") -> logging.Logger:
    """Setup and configure logger"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def log_event(event_type: str, message: str, level: str = "INFO"):
    """Log an event with timestamp"""
    logger = setup_logger()
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, f"[{event_type}] {message}")

def log_crowd_data(location: str, area: str, camera_id: str, 
                   crowd_count: int, threat_status: str, density_level: str,
                   network_name: str, ip_address: str, connection_status: str):
    """Log crowd analytics data"""
    logger = setup_logger("crowd_data")
    logger.info(
        f"LOCATION:{location}|AREA:{area}|CAMERA_ID:{camera_id}|"
        f"CROWD_COUNT:{crowd_count}|THREAT_STATUS:{threat_status}|"
        f"DENSITY_LEVEL:{density_level}|NETWORK:{network_name}|"
        f"IP:{ip_address}|CONNECTION:{connection_status}"
    )

def log_threat_event(location: str, area: str, camera_id: str,
                     threat_type: str, confidence: float):
    """Log threat detection events"""
    logger = setup_logger("threat_events")
    logger.warning(
        f"THREAT DETECTED - LOCATION:{location}|AREA:{area}|CAMERA_ID:{camera_id}|"
        f"TYPE:{threat_type}|CONFIDENCE:{confidence:.2f}"
    )