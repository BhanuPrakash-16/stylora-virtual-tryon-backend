"""
Image Processing Utilities
===========================
Helper functions for image format conversion and validation.
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import Tuple, Optional


def bytes_to_numpy(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Convert image bytes to numpy array (OpenCV format).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Numpy array (BGR format) or None if invalid
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def bytes_to_pil(image_bytes: bytes) -> Optional[Image.Image]:
    """
    Convert image bytes to PIL Image.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        PIL Image or None if invalid
    """
    try:
        return Image.open(BytesIO(image_bytes))
    except Exception:
        return None


def numpy_to_base64(img: np.ndarray) -> str:
    """
    Convert numpy array to base64 string.
    
    Args:
        img: Numpy array (OpenCV BGR format)
        
    Returns:
        Base64 encoded string
    """
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_numpy(b64_str: str) -> Optional[np.ndarray]:
    """
    Convert base64 string to numpy array.
    
    Args:
        b64_str: Base64 encoded image string
        
    Returns:
        Numpy array or None if invalid
    """
    try:
        # Remove data URL prefix if present
        if ',' in b64_str:
            b64_str = b64_str.split(',')[1]
        
        img_bytes = base64.b64decode(b64_str)
        return bytes_to_numpy(img_bytes)
    except Exception:
        return None


def resize_image(img: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        img: Input image (numpy array)
        max_size: Maximum dimension (width or height)
        
    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    
    if max(h, w) <= max_size:
        return img
    
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_image_dimensions(img: np.ndarray) -> Tuple[int, int]:
    """
    Get image dimensions.
    
    Args:
        img: Input image (numpy array)
        
    Returns:
        Tuple of (height, width)
    """
    return img.shape[:2]


def validate_image_format(image_bytes: bytes) -> bool:
    """
    Validate that image bytes can be decoded.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        True if valid image, False otherwise
    """
    img = bytes_to_numpy(image_bytes)
    return img is not None and img.size > 0


def detect_skin_tone_hsv(img: np.ndarray) -> np.ndarray:
    """
    Detect skin-tone regions using HSV color space.
    
    NOTE: This is a simple heuristic-based approach.
    It can produce false positives (e.g., face, arms, furniture).
    Used for conservative safety checks only.
    
    Args:
        img: Input image (BGR format)
        
    Returns:
        Binary mask where skin-tone regions are white
    """
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    # These ranges cover various skin tones
    lower_skin_1 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin_1 = np.array([20, 255, 255], dtype=np.uint8)
    
    lower_skin_2 = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin_2 = np.array([25, 150, 255], dtype=np.uint8)
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_skin_1, upper_skin_1)
    mask2 = cv2.inRange(hsv, lower_skin_2, upper_skin_2)
    
    # Combine masks
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    return skin_mask


def calculate_region_area(mask: np.ndarray, region_bbox: Tuple[int, int, int, int] = None) -> float:
    """
    Calculate the area ratio of white pixels in a mask.
    
    Args:
        mask: Binary mask (0 or 255)
        region_bbox: Optional bounding box (x, y, w, h) to focus on specific region
        
    Returns:
        Ratio of white pixels to total pixels (0.0 to 1.0)
    """
    if region_bbox:
        x, y, w, h = region_bbox
        mask = mask[y:y+h, x:x+w]
    
    total_pixels = mask.size
    white_pixels = np.count_nonzero(mask)
    
    return white_pixels / total_pixels if total_pixels > 0 else 0.0
