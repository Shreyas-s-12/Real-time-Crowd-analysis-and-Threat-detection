"""
Mobile Camera Input Source Module for Real-Time Crowd Analysis and Threat Detection
Supports IP Webcam and DroidCam streams over HTTP/MJPEG
"""

import cv2
import threading
import time
import socket
from typing import Optional
from ..utils.config import config
from ..utils.logger import setup_logger
from ..models.network_validator import validate_camera_connection

logger = setup_logger("mobile_camera")

class MobileCameraSource:
    """Handles mobile phone camera video capture via IP Webcam / DroidCam HTTP streams"""
    
    def __init__(self, ip_address: str, port: int = 8080, 
                 path: str = "/video", width: int = None, height: int = None):
        self.ip_address = ip_address
        self.port = port
        self.path = path
        self.width = width or config.FRAME_WIDTH
        self.height = height or config.FRAME_HEIGHT
        self.cap = None
        self.is_running = False
        self.frame = None
        self.lock = threading.Lock()
        self.capture_thread = None
        self.watchdog_thread = None
        self.fps = 0
        self.last_frame_time = time.time()
        self.connection_status = "DISCONNECTED"
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config.RECONNECT_ATTEMPTS
        
    def _build_url(self) -> str:
        """Build HTTP stream URL for IP Webcam / DroidCam"""
        return f"http://{self.ip_address}:{self.port}{self.path}"
    
    def check_connection_quick(self, timeout: float = 2.0) -> bool:
        """Perform a quick TCP socket check to verify if the port is open and listening"""
        try:
            socket.setdefaulttimeout(timeout)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.ip_address, self.port))
            return True
        except Exception as e:
            logger.warning(f"Quick connection check failed for {self.ip_address}:{self.port} - {e}")
            return False
            
    def start(self) -> bool:
        """Start mobile camera capture"""
        try:
            # 1. Quick Socket Check to avoid long OpenCV blocking timeouts
            if not self.check_connection_quick():
                logger.error(f"Mobile camera port {self.port} at {self.ip_address} is unreachable.")
                self.connection_status = "FAILED"
                return False

            # 2. Validate network connection first using network validator
            connection_info = validate_camera_connection(self.ip_address, "wireless")
            self.connection_status = connection_info.get('connection_status', 'UNKNOWN')
            
            # Build camera URL
            url = self._build_url()
            logger.info(f"Connecting to mobile camera at {url}")
            
            self.cap = cv2.VideoCapture(url)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open mobile camera stream at {url}")
                self.connection_status = "FAILED"
                return False
            
            # Set properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Mobile camera started: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            self.is_running = True
            self.connection_status = "CONNECTED"
            self.last_frame_time = time.time()
            self.reconnect_attempts = 0
            
            # Start capture thread for non-blocking UI stream reading
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            # Start watchdog thread to monitor for indefinite stream freezes
            self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
            self.watchdog_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting mobile camera: {e}")
            self.connection_status = "ERROR"
            return False
    
    def stop(self):
        """Stop mobile camera capture"""
        logger.info(f"Stopping mobile camera {self.ip_address}")
        self.is_running = False
        self.connection_status = "DISCONNECTED"
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            self.watchdog_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        with self.lock:
            self.frame = None

    def _watchdog_loop(self):
        """Monitors for frozen cap.read() calls and triggers restart if stuck"""
        while self.is_running:
            time.sleep(2.0)
            
            # If we haven't received a frame in 5 seconds and we are supposed to be connected
            if self.connection_status == "CONNECTED":
                time_since_last_frame = time.time() - self.last_frame_time
                if time_since_last_frame > 5.0:
                    logger.error(f"Watchdog triggered! No frames from camera for {time_since_last_frame:.1f}s. Stream likely frozen.")
                    self.connection_status = "RECONNECTING"
                    # Force a restart of the capture object
                    self._force_restart_stream()

    def _force_restart_stream(self):
        """Forcefully releases and rebuilds the capture object from the watchdog"""
        logger.info("Watchdog forcing stream restart...")
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing frozen capture: {e}")
            self.cap = None
        
        if self._reconnect():
            logger.info("Watchdog successfully restarted stream")
            self.connection_status = "CONNECTED"
        else:
            logger.error("Watchdog failed to restart stream")
            self.connection_status = "FAILED"

    def _capture_loop(self):
        """Continuous frame capture loop with robust reconnection logic and zero-latency grabbing"""
        frame_count = 0
        start_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.warning(f"Mobile camera disconnected, attempting reconnect...")
                    self.connection_status = "RECONNECTING"
                    
                    if self._reconnect():
                        logger.info(f"Mobile camera reconnected successfully")
                        self.connection_status = "CONNECTED"
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(f"Mobile camera failed to reconnect after {max_consecutive_failures} attempts")
                            self.connection_status = "FAILED"
                            break
                        time.sleep(config.RECONNECT_DELAY)
                        continue
                
                # Use grab() to quickly empty the OpenCV buffer and eliminate latency
                if not self.cap.grab():
                    logger.warning(f"Failed to grab frame from mobile camera")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Mobile camera frame read failures exceeded limit")
                        self.connection_status = "STREAM_ERROR"
                        if self._reconnect():
                            consecutive_failures = 0
                            self.connection_status = "CONNECTED"
                        else:
                            time.sleep(config.RECONNECT_DELAY)
                        continue
                    
                    time.sleep(0.01)
                    continue
                
                # Retrieve only the absolute latest frame
                ret, frame = self.cap.retrieve()
                if not ret or frame is None:
                    continue
                
                # Reset failure counter on successful frame
                consecutive_failures = 0
                self.last_frame_time = time.time()
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                if current_time - start_time >= 1.0:
                    self.fps = frame_count / (current_time - start_time)
                    frame_count = 0
                    start_time = current_time
                
                # Store strictly the latest frame, overwriting any old ones immediately
                with self.lock:
                    self.frame = frame.copy()
                
                # We do NOT sleep here. We run as fast as possible to keep the buffer empty.
                
            except Exception as e:
                logger.error(f"Error in mobile camera capture loop: {e}")
                consecutive_failures += 1
                time.sleep(0.1)
                
                if consecutive_failures >= max_consecutive_failures:
                    break
        
        logger.info(f"Mobile camera capture loop ended")
        if self.is_running:
            self.connection_status = "FAILED"
        else:
            self.connection_status = "DISCONNECTED"
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to the mobile stream"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts reached for mobile camera")
            return False
        
        self.reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} for mobile camera")
        
        try:
            if self.cap:
                self.cap.release()
            
            # Verify socket before opening capture
            if not self.check_connection_quick(timeout=1.0):
                return False

            url = self._build_url()
            self.cap = cv2.VideoCapture(url)
            
            if self.cap.isOpened():
                # Set properties again
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.last_frame_time = time.time() # Reset watchdog timer
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Reconnection error for mobile camera: {e}")
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
