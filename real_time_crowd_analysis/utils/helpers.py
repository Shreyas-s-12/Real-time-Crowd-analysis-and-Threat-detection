"""
Helper Functions for Real-Time Crowd Analysis and Threat Detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Any # Keep typing imports
from real_time_crowd_analysis.utils.config import config # Absolute import
from real_time_crowd_analysis.utils.logger import setup_logger # Absolute import

logger = setup_logger("helpers")


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: (x1, y1) coordinates
        point2: (x2, y2) coordinates
        
    Returns:
        Euclidean distance between the points
    """
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def calculate_speed(distance: float, time_elapsed: float) -> float:
    """
    Calculate speed given distance and time
    
    Args:
        distance: Distance traveled (pixels)
        time_elapsed: Time taken (seconds)
        
    Returns:
        Speed in pixels per second
    """
    if time_elapsed <= 0:
        return 0.0
    return distance / time_elapsed


def draw_text_with_background(frame: np.ndarray, text: str, position: Tuple[int, int], 
                             font_scale: float = 0.6, thickness: int = 2,
                             text_color: Tuple[int, int, int] = (255, 255, 255),
                             bg_color: Tuple[int, int, int] = (0, 0, 0),
                             padding: int = 5) -> np.ndarray:
    """
    Draw text with a background rectangle for better visibility
    
    Args:
        frame: Input frame
        text: Text to draw
        position: Bottom-left corner of text (x, y)
        font_scale: Font scale factor
        thickness: Text thickness
        text_color: RGB color of text
        bg_color: RGB color of background
        padding: Padding around text
        
    Returns:
        Frame with text drawn
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Calculate background rectangle coordinates
    x, y = position
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding
    
    # Draw background rectangle
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)
    
    return frame


def resize_frame(frame: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame: Input frame
        width: Target width (optional)
        height: Target height (optional)
        
    Returns:
        Resized frame
    """
    if width is None and height is None:
        return frame
    
    h, w = frame.shape[:2]
    
    if width is None:
        # Calculate width based on height
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
    elif height is None:
        # Calculate height based on width
        aspect_ratio = h / w
        height = int(width * aspect_ratio)
    
    return cv2.resize(frame, (width, height))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_timestamp(timestamp: float = None) -> str:
    """
    Format timestamp as HH:MM:SS
    
    Args:
        timestamp: Unix timestamp (optional, defaults to current time)
        
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        import time
        timestamp = time.time()
    
    from datetime import datetime
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")


def validate_coordinates(x: int, y: int, width: int, height: int) -> bool:
    """
    Validate that coordinates are within frame bounds
    
    Args:
        x: X coordinate
        y: Y coordinate
        width: Frame width
        height: Frame height
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    return 0 <= x < width and 0 <= y < height


def draw_crosshair(frame: np.ndarray, center: Tuple[int, int], size: int = 20, 
                  color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw a crosshair at the specified center point
    
    Args:
        frame: Input frame
        center: Center point (x, y)
        size: Size of crosshair arms
        color: RGB color of crosshair
        thickness: Thickness of crosshair lines
        
    Returns:
        Frame with crosshair drawn
    """
    x, y = center
    
    # Draw horizontal line
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    
    # Draw vertical line
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
    
    return frame


def draw_circle(frame: np.ndarray, center: Tuple[int, int], radius: int, 
               color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw a circle at the specified center point
    
    Args:
        frame: Input frame
        center: Center point (x, y)
        radius: Radius of circle
        color: RGB color of circle
        thickness: Thickness of circle line (use -1 for filled)
        
    Returns:
        Frame with circle drawn
    """
    return cv2.circle(frame, center, radius, color, thickness)


def overlay_transparent(background: np.ndarray, overlay: np.ndarray, 
                       position: Tuple[int, int], alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a transparent image onto a background
    
    Args:
        background: Background image
        overlay: Overlay image (with alpha channel if 4 channels)
        position: Top-left corner position (x, y)
        alpha: Transparency value (0.0 to 1.0)
        
    Returns:
        Background with overlay applied
    """
    try:
        x, y = position
        h, w = overlay.shape[:2]
        
        # Check bounds
        if x >= background.shape[1] or y >= background.shape[0]:
            return background
        
        # Adjust width and height if necessary
        if x + w > background.shape[1]:
            w = background.shape[1] - x
            overlay = overlay[:, :w]
        
        if y + h > background.shape[0]:
            h = background.shape[0] - y
            overlay = overlay[:h, :]
        
        if w <= 0 or h <= 0:
            return background
        
        # Extract alpha channel if present, otherwise use specified alpha
        if overlay.shape[2] == 4:
            overlay_img = overlay[:, :, :3]
            mask = overlay[:, :, 3:] / 255.0
        else:
            overlay_img = overlay
            mask = np.full((h, w, 1), alpha, dtype=np.float32)
        
        # Blend images
        background[y:y+h, x:x+w] = (
            background[y:y+h, x:x+w] * (1 - mask) + overlay_img * mask
        ).astype(np.uint8)
        
        return background
    except Exception as e:
        logger.error(f"Error in overlay_transparent: {e}")
        return background