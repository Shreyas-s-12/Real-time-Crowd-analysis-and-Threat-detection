"""
Wired Camera Input Source Module for Real-Time Crowd Analysis and Threat Detection
"""

import cv2
import threading
import time
from typing import Optional, Callable
from ..utils.config import config
from ..utils.logger import setup_logger
from ..models.network_validator import validate_camera_connection

logger = setup_logger("wired_camera")

class WiredCameraSource:
    """Handles wired IP camera video capture"""
    
    def __init__(self, ip_address: str, port: int = 80, 
                 username: str = None, password: str = None,
                 width: int = None, height: int = None):
        self.ip_address = ip_address
        self.port = port
        self.username = username
        self.password = password
        self.width = width or config.FRAME_WIDTH
        self.height = height or config.FRAME_HEIGHT
        self.cap = None
        self.is_running = False
        self.frame = None
        self.lock = threading.Lock()
        self.capture_thread = None
        self.fps = 0
        self.last_frame_time = 0
        self.connection_status = "DISCONNECTED"
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.RECONNECT_ATTEMPTS
        
    def _build_url(self) -> str:
        """Build RTSP or HTTP URL for the camera"""
        # Common RTSP formats for IP cameras
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        
        # Try RTSP first
        rtsp_url = f"rtsp://{auth}{self.ip_address}:{self.port}/stream1"
        return rtsp_url
    
    def start(self) -> bool:
        """Start wired camera capture"""
        try:
            # Validate network connection first
            connection_info = validate_camera_connection(self.ip_address, "wired")
            self.connection_status = connection_info['connection_status']
            
            if self.connection_status not in ['CONNECTED', 'DIFFERENT_NETWORK']:
                logger.warning(f"Camera {self.ip_address} network validation failed: {self.connection_status}")
                # Still try to connect, but warn user
            
            # Build camera URL
            url = self._build_url()
            logger.info(f"Connecting to wired camera at {url}")
            
            self.cap = cv2.VideoCapture(url)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open wired camera {self.ip_address}")
                self.connection_status = "FAILED"
                return False
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Wired camera {self.ip_address} started: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.is_running = True
            self.connection_status = "CONNECTED"
            self.last_frame_time = time.time()
            self.reconnect_attempts = 0
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting wired camera {self.ip_address}: {e}")
            self.connection_status = "ERROR"
            return False
    
    def stop(self):
        """Stop wired camera capture"""
        logger.info(f"Stopping wired camera {self.ip_address}")
        self.is_running = False
        self.connection_status = "DISCONNECTED"
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        with self.lock:
            self.frame = None
    
    def _capture_loop(self):
        """Continuous frame capture loop with reconnection logic"""
        frame_count = 0
        start_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.warning(f"Wired camera {self.ip_address} disconnected, attempting reconnect...")
                    self.connection_status = "RECONNECTING"
                    
                    # Attempt reconnection
                    if self._reconnect():
                        logger.info(f"Wired camera {self.ip_address} reconnected successfully")
                        self.connection_status = "CONNECTED"
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(f"Wired camera {self.ip_address} failed to reconnect after {max_consecutive_failures} attempts")
                            self.connection_status = "FAILED"
                            break
                        time.sleep(config.RECONNECT_DELAY)
                        continue
                
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning(f"Failed to read frame from wired camera {self.ip_address}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Wired camera {self.ip_address} too many frame read failures")
                        self.connection_status = "FRAME_ERROR"
                        # Try to reconnect
                        if self._reconnect():
                            consecutive_failures = 0
                            self.connection_status = "CONNECTED"
                        else:
                            time.sleep(config.RECONNECT_DELAY)
                        continue
                    
                    time.sleep(0.01)
                    continue
                
                # Reset failure counter on successful frame
                consecutive_failures = 0
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - start_time >= 1.0:
                    self.fps = frame_count / (current_time - start_time)
                    frame_count = 0
                    start_time = current_time
                
                # Store frame with thread safety
                with self.lock:
                    self.frame = frame.copy() if frame is not None else None
                
                # Control frame rate
                time.sleep(max(0, (1.0/config.FPS) - (time.time() - current_time)))
                
            except Exception as e:
                logger.error(f"Error in wired camera capture loop: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
                
                if consecutive_failures >= max_consecutive_failures:
                    break
        
        logger.info(f"Wired camera {self.ip_address} capture loop ended")
        self.connection_status = "DISCONNECTED"
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to the camera"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts reached for wired camera {self.ip_address}")
            return False
        
        self.reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} for {self.ip_address}")
        
        try:
            if self.cap:
                self.cap.release()
            
            url = self._build_url()
            self.cap = cv2.VideoCapture(url)
            
            if self.cap.isOpened():
                # Set properties again
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Reconnection error for wired camera {self.ip_address}: {e}")
            return False
    
    def get_frame(self) -> Optional[any]:
        """Get the latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps
    
    def get_resolution(self) -> tuple:
        """Get current resolution"""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)
    
    def get_connection_status(self) -> str:
        """Get current connection status"""
        return self.connection_status
    
    def get_reconnect_attempts(self) -> int:
        """Get number of reconnection attempts"""
        return self.reconnect_attempts

# Convenience function
def create_wired_camera_source(ip_address: str, port: int = 80, 
                              username: str = None, password: str = None,
                              width: int = None, height: int = None) -> WiredCameraSource:
    """Create and return a wired camera source"""
    return WiredCameraSource(ip_address, port, username, password, width, height)