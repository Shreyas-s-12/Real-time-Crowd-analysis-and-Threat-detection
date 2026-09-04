"""
RTSP Stream Input Source Module for Real-Time Crowd Analysis and Threat Detection
"""

import cv2
import threading
import time
from typing import Optional, Callable
from ..utils.config import config
from ..utils.logger import setup_logger
from ..models.network_validator import validate_camera_connection

logger = setup_logger("rtsp_stream")

class RTSPStreamSource:
    """Handles RTSP video stream capture"""
    
    def __init__(self, url: str, username: str = None, password: str = None,
                 width: int = None, height: int = None):
        self.url = url
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
        """Build RTSP URL with authentication if needed"""
        if not self.url.startswith(('rtsp://', 'http://')):
            # Assume it's just an IP/hostname, build standard RTSP URL
            if self.username and self.password:
                auth = f"{self.username}:{self.password}@"
            else:
                auth = ""
            return f"rtsp://{auth}{self.url}/stream1"
        
        # If URL already has protocol, add auth if needed
        if self.username and self.password and '@' not in self.url.split('://')[1]:
            # Insert credentials into URL
            protocol, rest = self.url.split('://', 1)
            return f"{protocol}://{self.username}:{self.password}@{rest}"
        
        return self.url
    
    def start(self) -> bool:
        """Start RTSP stream capture"""
        try:
            # Extract IP/hostname from URL for network validation
            host = self._extract_host_from_url()
            if host:
                # Validate network connection first
                connection_info = validate_camera_connection(host, "rtsp")
                self.connection_status = connection_info['connection_status']
                
                if self.connection_status not in ['CONNECTED', 'DIFFERENT_NETWORK']:
                    logger.warning(f"RTSP stream {host} network validation failed: {self.connection_status}")
                    # Still try to connect, but warn user
            
            # Build camera URL
            url = self._build_url()
            logger.info(f"Connecting to RTSP stream at {url}")
            
            self.cap = cv2.VideoCapture(url)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open RTSP stream {self.url}")
                self.connection_status = "FAILED"
                return False
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
            
            # Reduce buffer size for lower latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"RTSP stream started: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.is_running = True
            self.connection_status = "CONNECTED"
            self.last_frame_time = time.time()
            self.reconnect_attempts = 0
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting RTSP stream {self.url}: {e}")
            self.connection_status = "ERROR"
            return False
    
    def _extract_host_from_url(self) -> Optional[str]:
        """Extract hostname/IP from URL for network validation"""
        try:
            # Remove protocol
            if '://' in self.url:
                host_part = self.url.split('://')[1]
            else:
                host_part = self.url
            
            # Remove path, port, credentials
            host_part = host_part.split('/')[0]
            host_part = host_part.split(':')[0]
            
            # Remove credentials if present
            if '@' in host_part:
                host_part = host_part.split('@')[1]
            
            return host_part
        except Exception:
            return None
    
    def stop(self):
        """Stop RTSP stream capture"""
        logger.info(f"Stopping RTSP stream {self.url}")
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
                    logger.warning(f"RTSP stream {self.url} disconnected, attempting reconnect...")
                    self.connection_status = "RECONNECTING"
                    
                    # Attempt reconnection
                    if self._reconnect():
                        logger.info(f"RTSP stream {self.url} reconnected successfully")
                        self.connection_status = "CONNECTED"
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(f"RTSP stream {self.url} failed to reconnect after {max_consecutive_failures} attempts")
                            self.connection_status = "FAILED"
                            break
                        time.sleep(config.RECONNECT_DELAY)
                        continue
                
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    logger.warning(f"Failed to read frame from RTSP stream {self.url}")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"RTSP stream {self.url} too many frame read failures")
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
                logger.error(f"Error in RTSP stream capture loop: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
                
                if consecutive_failures >= max_consecutive_failures:
                    break
        
        logger.info(f"RTSP stream {self.url} capture loop ended")
        self.connection_status = "DISCONNECTED"
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to the stream"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts reached for RTSP stream {self.url}")
            return False
        
        self.reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} for {self.url}")
        
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
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Low latency
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Reconnection error for RTSP stream {self.url}: {e}")
            return False
    
    def get_frame(self) -> Optional[any]:
        """Get the latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def is_opened(self) -> bool:
        """Check if stream is opened"""
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
def create_rtsp_stream_source(url: str, username: str = None, password: str = None,
                             width: int = None, height: int = None) -> RTSPStreamSource:
    """Create and return an RTSP stream source"""
    return RTSPStreamSource(url, username, password, width, height)