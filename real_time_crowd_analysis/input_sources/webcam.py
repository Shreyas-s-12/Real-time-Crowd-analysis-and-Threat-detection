"""
Webcam Input Source Module for Real-Time Crowd Analysis and Threat Detection
"""

import cv2
import threading
import time
from typing import Optional, Callable
from PyQt6.QtWidgets import QMessageBox
from ..utils.config import config
from ..utils.logger import setup_logger

logger = setup_logger("webcam")

class WebcamSource:
    """Handles webcam video capture"""
    
    def __init__(self, camera_id: int = 0, width: int = None, height: int = None):
        self.requested_camera_id = camera_id
        self.camera_id = camera_id
        self.width = width or config.FRAME_WIDTH
        self.height = height or config.FRAME_HEIGHT
        self.cap = None
        self.backend_name = ""
        self.is_running = False
        self.frame = None
        self.lock = threading.Lock()
        self.capture_thread = None
        self.fps = 0
        self.last_frame_time = 0

    def _try_open_index_with_backends(self, target_id: int) -> bool:
        """Attempt to open a specific camera index using supported backends and verify a frame can be read."""
        backends = [
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY, "Default")
        ]
        
        for backend, name in backends:
            try:
                logger.info(f"Testing camera index {target_id} with backend: {name}")
                if backend is not None:
                    cap = cv2.VideoCapture(target_id, backend)
                else:
                    cap = cv2.VideoCapture(target_id)

                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, config.FPS)
                    
                    # Verify frame acquisition with warmup reads
                    test_read_success = False
                    test_frame = None
                    for attempt in range(5):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            test_read_success = True
                            break
                        time.sleep(0.05)

                    if test_read_success:
                        self.cap = cap
                        self.backend_name = name
                        self.camera_id = target_id
                        with self.lock:
                            self.frame = test_frame.copy()
                        logger.info(f"Successfully opened camera {target_id} with backend {name} (res: {test_frame.shape[1]}x{test_frame.shape[0]})")
                        return True
                    else:
                        logger.warning(f"Camera index {target_id} opened with {name} but failed to read frames.")
                        cap.release()
                else:
                    if cap:
                        cap.release()
            except Exception as e:
                logger.error(f"Error testing camera index {target_id} with backend {name}: {e}")

        return False
        
    def start(self) -> bool:
        """Start webcam capture by trying requested camera_id and fallback indices."""
        self.stop()
        
        # 1. Try requested index
        if self._try_open_index_with_backends(self.requested_camera_id):
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            return True

        # 2. Try fallback indices if primary requested index fails
        fallback_indices = [idx for idx in [0, 1, 2, 3] if idx != self.requested_camera_id]
        logger.warning(f"Camera index {self.requested_camera_id} failed. Attempting fallback indices {fallback_indices}")

        for fallback_id in fallback_indices:
            if self._try_open_index_with_backends(fallback_id):
                logger.info(f"Fallback successful: Connected to camera index {fallback_id} instead of requested index {self.requested_camera_id}")
                self.is_running = True
                self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
                self.capture_thread.start()
                return True

        logger.error(f"Failed to open webcam (tried requested index {self.requested_camera_id} and fallbacks) with any backend.")
        return False
    
    def stop(self):
        """Stop webcam capture and release hardware resources cleanly."""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                logger.error(f"Error releasing VideoCapture: {e}")
            self.cap = None
        
        with self.lock:
            self.frame = None
    
    def _capture_loop(self):
        """Continuous frame capture loop with rate control and error handling"""
        frame_count = 0
        start_time = time.time()
        consecutive_failures = 0
        logger.info(f"Webcam capture loop started for camera index {self.camera_id}")
        
        while self.is_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures % 50 == 1:
                        logger.warning(f"Failed to read frame from webcam {self.camera_id} (consecutive failures: {consecutive_failures})")
                    time.sleep(0.02)
                    if consecutive_failures > 150: # 3 seconds of continuous frame loss
                        logger.error(f"Webcam {self.camera_id} stream lost after consecutive failures.")
                        break
                    continue
                
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
                    self.frame = frame.copy()
                
                # Control frame rate
                time.sleep(max(0.001, (1.0/config.FPS) - (time.time() - current_time)))
                
            except Exception as e:
                logger.error(f"Error in webcam capture loop: {e}")
                break
        
        self.is_running = False
        logger.info(f"Webcam {self.camera_id} capture loop ended")
    
    def get_frame(self) -> Optional[any]:
        """Get the latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def is_opened(self) -> bool:
        """Check if webcam is active and producing frames"""
        return self.cap is not None and self.cap.isOpened() and self.is_running
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps
    
    def get_resolution(self) -> tuple:
        """Get current resolution"""
        with self.lock:
            if self.frame is not None:
                return (self.frame.shape[1], self.frame.shape[0])
        if self.cap:
            try:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return (width, height)
            except Exception:
                pass
        return (0, 0)

# Convenience function
def create_webcam_source(camera_id: int = 0, width: int = None, height: int = None) -> WebcamSource:
    """Create and return a webcam source"""
    return WebcamSource(camera_id, width, height)