"""
Face Detection Module for Real-Time Crowd Analysis and Threat Detection
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Dict # Keep typing imports
from real_time_crowd_analysis.utils.config import config # Absolute import
from real_time_crowd_analysis.utils.logger import setup_logger # Absolute import

logger = setup_logger("face_detection")

class FaceDetector:
    """Face detection and recognition using OpenCV"""
    
    def __init__(self):
        # Load pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Load pre-trained eye detection model for better face validation
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Face recognizer (will be trained if suspect images are available)
        self.face_recognizer = None
        self.has_suspects = False
        self.suspect_names = {}  # label -> name mapping
        
        # Try to load trained recognizer
        self._load_suspect_recognizer()
    
    def _load_suspect_recognizer(self):
        """Load or create face recognizer for suspect identification"""
        try:
            # Check if we have the face module available
            if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
                logger.warning("Face recognition not available: install opencv-contrib-python")
                return
            
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Try to load existing training data
            suspects_dir = "suspects"
            if self._train_suspect_recognizer(suspects_dir):
                logger.info("Suspect recognizer loaded and trained")
            else:
                logger.info("No suspect data available for training")
                
        except Exception as e:
            logger.error(f"Failed to initialize face recognizer: {e}")
            self.face_recognizer = None
    
    def _train_suspect_recognizer(self, suspects_dir: str) -> bool:
        """Train face recognizer with suspect images"""
        try:
            if not os.path.exists(suspects_dir):
                return False
            
            # Get list of suspect images
            valid_images = [f for f in os.listdir(suspects_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not valid_images:
                return False
            
            faces_data = []
            labels = []
            label_map = {}  # Map label to person name
            current_label = 0
            
            for filename in valid_images:
                # Extract person name from filename (without extension)
                person_name = os.path.splitext(filename)[0]
                filepath = os.path.join(suspects_dir, filename)
                
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Detect faces in the image
                detected_faces = self.face_cascade.detectMultiScale(
                    img, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50)
                )
                
                for (x, y, w, h) in detected_faces:
                    face_roi = img[y:y+h, x:x+w]
                    faces_data.append(face_roi)
                    labels.append(current_label)
                
                label_map[current_label] = person_name
                current_label += 1
            
            if len(faces_data) == 0:
                return False
            
            # Train the recognizer
            self.face_recognizer.train(faces_data, np.array(labels))
            self.suspect_names = label_map
            self.has_suspects = True
            
            logger.info(f"Trained suspect recognizer with {len(faces_data)} faces from {len(label_map)} persons")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train suspect recognizer: {e}")
            return False
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame
        
        Returns:
            List of bounding boxes as (x, y, width, height)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to list of tuples
            face_list = []
            for (x, y, w, h) in faces:
                face_list.append((x, y, w, h))
            
            return face_list
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def validate_face(self, face_roi: np.ndarray) -> bool:
        """
        Validate if a face region contains a valid face (using eye detection)
        
        Args:
            face_roi: Grayscale face region
            
        Returns:
            True if valid face, False otherwise
        """
        try:
            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            
            # Require at least 2 eyes for a valid face
            return len(eyes) >= 2
        except Exception:
            return False
    
    def recognize_face(self, face_roi: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face and return person name and confidence
        
        Args:
            face_roi: Grayscale face region
            
        Returns:
            Tuple of (person_name, confidence) where confidence is distance (lower is better)
        """
        if not self.has_suspects or self.face_recognizer is None:
            return None, float('inf')
        
        try:
            # Resize face to standard size for recognition
            face_resized = cv2.resize(face_roi, (100, 100))
            
            # Predict
            label, confidence = self.face_recognizer.predict(face_resized)
            
            # Get person name
            person_name = self.suspect_names.get(label, f"Unknown_{label}")
            
            return person_name, confidence
            
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
            return None, float('inf')
    
    def detect_and_analyze_faces(self, frame: np.ndarray) -> List[Dict[str, any]]:
        """
        Detect faces in frame and perform analysis
        
        Returns:
            List of dictionaries containing face information
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detect_faces(frame)
            
            face_analysis = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Validate face
                is_valid = self.validate_face(face_roi)
                
                # Recognize face if we have suspect data
                person_name, confidence = None, float('inf')
                if self.has_suspects:
                    person_name, confidence = self.recognize_face(face_roi)
                
                # Determine if this is a threat (known suspect with low confidence)
                is_threat = False
                threat_level = "NONE"
                
                if self.has_suspects and person_name and confidence < 80:  # Lower confidence = better match
                    is_threat = True
                    threat_level = "HIGH" if confidence < 60 else "MEDIUM"
                
                face_info = {
                    'bbox': (x, y, w, h),
                    'confidence': 1.0,  # Face detection confidence (simplified)
                    'is_valid': is_valid,
                    'person_name': person_name,
                    'recognition_confidence': confidence,
                    'is_threat': is_threat,
                    'threat_level': threat_level
                }
                
                face_analysis.append(face_info)
            
            return face_analysis
            
        except Exception as e:
            logger.error(f"Face detection and analysis error: {e}")
            return []
    
    def draw_faces(self, frame: np.ndarray, face_analysis: List[Dict[str, any]]) -> np.ndarray:
        """Draw face detection results on frame"""
        for face_info in face_analysis:
            x, y, w, h = face_info['bbox']
            
            # Choose color based on threat level
            if face_info['is_threat']:
                color = (0, 0, 255)  # Red for threats
                thickness = 3
            elif face_info['is_valid']:
                color = (0, 255, 0)  # Green for valid faces
                thickness = 2
            else:
                color = (0, 255, 255)  # Yellow for uncertain
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            label_parts = []
            if face_info['person_name']:
                label_parts.append(face_info['person_name'])
            
            if face_info['recognition_confidence'] != float('inf'):
                label_parts.append(f"Conf: {face_info['recognition_confidence']:.1f}")
            
            if label_parts:
                label = " | ".join(label_parts)
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw threat indicator
            if face_info['is_threat']:
                cv2.putText(frame, "THREAT", (x, y + h + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame

# Global face detector instance
face_detector = FaceDetector()

# Convenience functions
def detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces in frame"""
    return face_detector.detect_faces(frame)

def analyze_faces(frame: np.ndarray) -> List[Dict[str, any]]:
    """Detect and analyze faces in frame"""
    return face_detector.detect_and_analyze_faces(frame)

def draw_faces(frame: np.ndarray, face_analysis: List[Dict[str, any]]) -> np.ndarray:
    """Draw face detection results on frame"""
    return face_detector.draw_faces(frame, face_analysis)