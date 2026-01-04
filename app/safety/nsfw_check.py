"""
NSFW and Skin Exposure Detection
=================================
Detects inappropriate content using heuristic-based skin exposure analysis.

DISCLAIMER: This is NOT a perfect ML-based NSFW classifier.
It uses simple color-based heuristics to detect skin exposure.
Can produce false positives (e.g., faces, arms) and false negatives.

We use CONSERVATIVE thresholds to prioritize safety over convenience.
"""

import cv2
import numpy as np
from typing import Dict
from app.utils.image_utils import bytes_to_numpy, detect_skin_tone_hsv, calculate_region_area
from app.config import config


def check_nsfw_content(image_bytes: bytes) -> Dict:
    """
    Check for NSFW content using skin exposure heuristics.
    
    Method:
    1. Detect skin-tone regions using HSV color space
    2. Calculate skin exposure ratio (skin pixels / total pixels)
    3. Analyze skin distribution (concentrated vs scattered)
    4. Check for excessive skin in torso region
    
    LIMITATIONS:
    - Simple color-based detection (not ML)
    - Can flag images with large faces/arms as unsafe
    - Cannot perfectly differentiate context (beach vs inappropriate)
    - We err on the side of caution with conservative thresholds
    
    Thresholds:
    - Total skin exposure > 25% → REJECT
    - Concentrated skin in torso > 15% → REJECT
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Dict with keys:
        - safe (bool): True if content appears safe
        - reason (str): Explanation
        - confidence (float): Confidence score
        - details (dict): Additional information
    """
    img = bytes_to_numpy(image_bytes)
    
    if img is None:
        return {
            "safe": False,
            "reason": "invalid_image",
            "confidence": 0.0,
            "details": {"error": "Could not decode image"}
        }
    
    try:
        # Detect skin-tone regions
        skin_mask = detect_skin_tone_hsv(img)
        
        # Calculate total skin exposure ratio
        total_skin_ratio = calculate_region_area(skin_mask)
        
        # Check against threshold
        if total_skin_ratio > config.SKIN_EXPOSURE_THRESHOLD:
            return {
                "safe": False,
                "reason": "excessive_skin_exposure",
                "confidence": min(1.0, (total_skin_ratio - config.SKIN_EXPOSURE_THRESHOLD) / 0.2),
                "details": {
                    "skin_exposure_ratio": total_skin_ratio,
                    "threshold": config.SKIN_EXPOSURE_THRESHOLD,
                    "message": "Excessive skin exposure detected. Please use appropriate clothing."
                }
            }
        
        # Analyze skin distribution in torso region
        # Torso is approximately center 50% width, top 40-80% height
        img_height, img_width = img.shape[:2]
        torso_bbox = (
            int(img_width * 0.25),   # x: 25% from left
            int(img_height * 0.40),  # y: 40% from top
            int(img_width * 0.50),   # w: 50% width
            int(img_height * 0.40)   # h: 40% height
        )
        
        torso_skin_ratio = calculate_region_area(skin_mask, torso_bbox)
        
        # Check for concentrated skin in torso area
        if torso_skin_ratio > config.CONCENTRATED_SKIN_THRESHOLD:
            return {
                "safe": False,
                "reason": "unsafe_torso_exposure",
                "confidence": min(1.0, (torso_skin_ratio - config.CONCENTRATED_SKIN_THRESHOLD) / 0.15),
                "details": {
                    "torso_skin_ratio": torso_skin_ratio,
                    "threshold": config.CONCENTRATED_SKIN_THRESHOLD,
                    "message": "Inappropriate clothing detected. Please wear appropriate attire."
                }
            }
        
        # Additional check: Look for very large contiguous skin regions
        # (Could indicate nudity/semi-nudity)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area_ratio = cv2.contourArea(largest_contour) / (img_height * img_width)
            
            # If single skin region is > 20% of image, might be unsafe
            if largest_area_ratio > 0.20:
                return {
                    "safe": False,
                    "reason": "large_skin_region",
                    "confidence": min(1.0, (largest_area_ratio - 0.20) / 0.15),
                    "details": {
                        "largest_skin_region_ratio": largest_area_ratio,
                        "threshold": 0.20,
                        "message": "Detected large exposed skin region. Please use appropriate clothing."
                    }
                }
        
        # All checks passed
        # Confidence is higher when skin exposure is lower
        confidence = 1.0 - (total_skin_ratio / config.SKIN_EXPOSURE_THRESHOLD)
        
        return {
            "safe": True,
            "reason": "content_safe",
            "confidence": max(0.5, confidence),  # Minimum 0.5 confidence
            "details": {
                "total_skin_ratio": total_skin_ratio,
                "torso_skin_ratio": torso_skin_ratio,
                "note": "Content appears appropriate (heuristic-based check)"
            }
        }
        
    except Exception as e:
        # Conservative: if check fails, reject
        return {
            "safe": False,
            "reason": "nsfw_check_error",
            "confidence": 0.0,
            "details": {"error": str(e)}
        }


def check_transparency(image_bytes: bytes) -> Dict:
    """
    Check if clothing appears transparent or sheer.
    
    This is a simplified check looking for alpha channel or
    very light clothing that might be see-through.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Dict with safe/reason/confidence/details
    """
    try:
        from PIL import Image
        from io import BytesIO
        
        img_pil = Image.open(BytesIO(image_bytes))
        
        # Check if image has alpha channel (transparency)
        if img_pil.mode == 'RGBA':
            # Convert to numpy array
            img_array = np.array(img_pil)
            alpha_channel = img_array[:, :, 3]
            
            # Calculate average opacity
            avg_opacity = np.mean(alpha_channel) / 255.0
            
            # If average opacity is low, clothing might be transparent
            if avg_opacity < 0.8:
                return {
                    "safe": False,
                    "reason": "transparent_clothing",
                    "confidence": 1.0 - avg_opacity,
                    "details": {
                        "avg_opacity": avg_opacity,
                        "message": "Transparent or sheer clothing is not allowed."
                    }
                }
        
        return {
            "safe": True,
            "reason": "no_transparency",
            "confidence": 1.0,
            "details": {}
        }
        
    except Exception:
        # If check fails, assume safe (don't reject on technical error)
        return {
            "safe": True,
            "reason": "transparency_check_skipped",
            "confidence": 0.5,
            "details": {}
        }
