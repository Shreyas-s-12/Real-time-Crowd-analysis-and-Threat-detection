"""
Object Detection Module for Real-Time Crowd Analysis and Threat Detection
"""

import cv2
import numpy as np
import time
import gc
from collections import deque
from typing import List, Tuple, Optional
from real_time_crowd_analysis.utils.config import config
from real_time_crowd_analysis.utils.logger import setup_logger

logger = setup_logger("detector")

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics or torch not available. Using OpenCV HOG detector as fallback.")

class PersonDetector:
    """Person detection using YOLOv8 or OpenCV HOG as fallback"""
    
    def __init__(self, model_path: str = None, person_confidence: float = None, weapon_confidence: float = None):
        self.model_path = model_path or config.YOLO_MODEL
        self.iou_threshold = config.YOLO_IOU_THRESHOLD
        self.person_confidence = person_confidence or config.YOLO_PERSON_CONFIDENCE
        self.weapon_confidence = weapon_confidence or config.YOLO_WEAPON_CONFIDENCE
        self.model = None
        self.hog = None
        self.use_yolo = YOLO_AVAILABLE
        
        # GPU detection automatically
        self.device = "cuda" if YOLO_AVAILABLE and torch.cuda.is_available() else "cpu"
        logger.info(f"YOLO will use device: {self.device}")
        
        # Performance monitoring for adaptive resolution
        self.current_imgsz = getattr(config, 'YOLO_IMG_SIZE', 640)
        self.perf_history = deque(maxlen=10)
        
        self.person_class_id = 0 # Default COCO person class ID
        self.last_log_time = 0.0
        
        self.initialize_detector()
    
    def _inspect_model(self):
        """Inspect loaded model and resolve person class ID and class names dynamically"""
        if not self.model or not hasattr(self.model, 'names'):
            return

        names_dict = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        
        # Dynamic search for 'person' class ID
        found_person = False
        for cid, cname in names_dict.items():
            if str(cname).lower() == "person":
                self.person_class_id = int(cid)
                found_person = True
                break
                
        task = getattr(self.model, 'task', 'detect')
        logger.info("=" * 50)
        logger.info(f"YOLO Model Diagnostic Summary:")
        logger.info(f"  - Model Path: {self.model_path}")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Task: {task}")
        logger.info(f"  - Person Class ID: {self.person_class_id} (Resolved: {found_person})")
        logger.info(f"  - Total Classes: {len(names_dict)}")
        logger.info(f"  - Sample Classes: {list(names_dict.values())[:5]}")
        logger.info("=" * 50)

    def initialize_detector(self):
        """Initialize the detection model with safe fallback"""
        if self.use_yolo:
            try:
                # Attempt to load primary model
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                logger.info(f"Loaded detection model: {self.model_path} on {self.device}")
                self._inspect_model()
            except Exception as e:
                logger.error(f"Failed to load YOLO model {self.model_path}: {e}")
                
                # Safe fallback to yolov8n.pt
                fallback_model = getattr(config, 'YOLO_MODEL_FALLBACK', "yolov8n.pt")
                if self.model_path != fallback_model:
                    try:
                        logger.info(f"Attempting fallback to {fallback_model}...")
                        self.model_path = fallback_model
                        self.model = YOLO(fallback_model)
                        self.model.to(self.device)
                        logger.info(f"Loaded fallback detection model: {fallback_model} on {self.device}")
                        self._inspect_model()
                        return
                    except Exception as e2:
                        logger.error(f"Fallback model {fallback_model} failed: {e2}")

                logger.info("Falling back to OpenCV HOG detector")
                self.use_yolo = False
                self.initialize_hog()
        else:
            self.initialize_hog()
    
    def initialize_hog(self):
        """Initialize OpenCV HOG detector as fallback"""
        try:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            logger.info("OpenCV HOG detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize HOG detector: {e}")
            raise
    
    def detect_all(self, frame: np.ndarray) -> List[dict]:
        """Detect both persons and weapons in a single pass for performance optimization"""
        try:
            if frame is None or frame.size == 0:
                return []

            # Compute scaling factors to map boxes back to original frame
            orig_h, orig_w = frame.shape[:2]
            target_w, target_h = 640, 360
            inference_frame = cv2.resize(frame, (target_w, target_h))
            
            scale_x = orig_w / float(target_w)
            scale_y = orig_h / float(target_h)

            if self.use_yolo and self.model is not None:
                try:
                    start_time = time.time()
                    
                    # Target person class and weapon classes (COCO class IDs: 0=person, 43=knife, 76=scissors, 34=baseball bat)
                    target_classes = list(set([self.person_class_id, 43, 76, 34]))
                    
                    use_half = self.device == "cuda"
                    results = self.model(
                        inference_frame,
                        imgsz=self.current_imgsz,
                        iou=self.iou_threshold,
                        classes=target_classes,
                        verbose=False,
                        device=self.device,
                        half=use_half
                    )
                    inference_time = time.time() - start_time
                    self.perf_history.append(inference_time)
                    
                    detections = []
                    names_dict = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))

                    for result in results:
                        if result.boxes:
                            for box in result.boxes:
                                cls = int(box.cls[0])
                                conf = float(box.conf[0])
                                
                                # Reclassify scissors (76) to knife (43) if conf < 0.60
                                if cls == 76 and conf < 0.60:
                                    cls = 43
                                
                                # Apply class-specific confidence thresholds
                                if cls == self.person_class_id and conf < self.person_confidence:
                                    continue
                                if cls != self.person_class_id and conf < self.weapon_confidence:
                                    continue

                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                # Scale coordinates back to input frame size
                                x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                                x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                                bw, bh = max(1, x2 - x1), max(1, y2 - y1)
                                
                                label_name = names_dict.get(cls, f"class_{cls}").upper()
                                
                                detections.append({
                                    'box': (x1, y1, bw, bh),
                                    'class': cls,
                                    'label': label_name,
                                    'conf': conf
                                })
                    
                    # Periodic controlled diagnostic logging (every 4 seconds)
                    now = time.time()
                    if now - self.last_log_time > 4.0:
                        person_dets = [d for d in detections if d['class'] == self.person_class_id]
                        max_c = max([d['conf'] for d in person_dets], default=0.0)
                        logger.info(f"[DETECTION DIAGNOSTIC] Persons: {len(person_dets)} | Total Detections: {len(detections)} | Max Person Conf: {max_c:.2f}")
                        self.last_log_time = now

                    del results
                    gc.collect()

                    return detections
                except Exception as e:
                    logger.error(f"YOLO inference failed: {e}")
                    return []
            else:
                # Fallback HOG only supports person detection
                hog_boxes = self._detect_with_hog(inference_frame)
                detections = []
                for (x, y, w, h) in hog_boxes:
                    x1, y1 = int(x * scale_x), int(y * scale_y)
                    x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)
                    detections.append({
                        'box': (x1, y1, x2 - x1, y2 - y1), 
                        'class': 0, 
                        'label': 'PERSON', 
                        'conf': 0.8
                    })
                return detections
        except Exception as e:
            logger.error(f"Detection execution error (YOLO={self.use_yolo}): {e}")
            return []

    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Backward compatible person detection filtered from multi-class pass"""
        detections = self.detect_all(frame)
        return [d['box'] for d in detections if d['class'] == 0]
    
    def _detect_with_hog(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect persons using OpenCV HOG"""
        try:
            boxes, weights = self.hog.detectMultiScale(
                frame, 
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
            filtered_boxes = []
            for (x, y, w, h), weight in zip(boxes, weights):
                if weight > 0.5:
                    filtered_boxes.append((x, y, w, h))
            return filtered_boxes
        except Exception as e:
            logger.error(f"HOG detection error: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw bounding boxes on frame"""
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

# Global detector instance
person_detector = PersonDetector(person_confidence=config.YOLO_PERSON_CONFIDENCE, weapon_confidence=config.YOLO_WEAPON_CONFIDENCE)

# Convenience function
def detect_persons(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect persons in frame"""
    return person_detector.detect_persons(frame)